"""
Hardamissa Lotto web application (cleaned + live-friendly).

Key changes:
- Manual score edits are preserved (won't be overwritten by API)
- Manual date/time edits allow score updates from API
- Predictions lock at kickoff time (not when results are available)
- Predictions are revealed at kickoff time (not when results are available)
- POSTPONED status support for fixtures
"""

import os
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from secrets import token_urlsafe

import requests
from sqlalchemy import func, distinct
from flask import (
    Blueprint, Flask, abort, flash, g, redirect, render_template, request, url_for
)
from flask_login import (
    LoginManager, UserMixin, current_user, login_required, login_user, logout_user
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

# Utility: default dictionary for coverage computation
from collections import defaultdict

# -----------------------------------------------------------------------------
# App / DB config
# -----------------------------------------------------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

# Session configuration - keep users logged in for 30 days
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)
app.config["REMEMBER_COOKIE_DURATION"] = timedelta(days=30)
app.config["REMEMBER_COOKIE_SECURE"] = False  # Set to True in production with HTTPS
app.config["REMEMBER_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SECURE"] = False  # Set to True in production with HTTPS
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Database URL (Render Postgres or local SQLite)
raw_db_url = os.environ.get("DATABASE_URL", "sqlite:///serie_a.db")

# Normalize "postgres://" -> "postgresql+psycopg2://"
if raw_db_url.startswith("postgres://"):
    raw_db_url = raw_db_url.replace("postgres://", "postgresql+psycopg2://", 1)

# Ensure sslmode=require for Postgres
if raw_db_url.startswith("postgresql"):
    parsed = urlparse(raw_db_url)
    q = parse_qs(parsed.query)
    if "sslmode" not in q:
        q["sslmode"] = ["require"]
        raw_db_url = urlunparse(parsed._replace(query=urlencode(q, doseq=True)))

app.config["SQLALCHEMY_DATABASE_URI"] = raw_db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,
    "pool_recycle": 300,
    "pool_size": 5,
    "max_overflow": 5,
    "pool_timeout": 30,
}

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# -----------------------------------------------------------------------------
# Valid fixture statuses
# -----------------------------------------------------------------------------
FIXTURE_STATUSES = (
    "SCHEDULED",   # Not yet started, time confirmed
    "TIMED",       # Not yet started, time confirmed (API variant)
    "IN_PLAY",     # Currently playing
    "PAUSED",      # Half-time or other pause
    "FINISHED",    # Match completed
    "POSTPONED",   # Match postponed to a later date
    "CANCELLED",   # Match cancelled entirely
    "SUSPENDED",   # Match suspended mid-game
)

# Statuses that indicate the fixture should be excluded from "current" view
EXCLUDED_FROM_CURRENT = ("POSTPONED", "CANCELLED", "SUSPENDED")

# -----------------------------------------------------------------------------
# Competitions
# -----------------------------------------------------------------------------
# Each competition is a distinct tournament with its own fixtures, theme, and
# leaderboard. Predictions stay scoped to a competition implicitly via the
# fixture they reference.
COMPETITIONS = {
    "SA": {
        "code": "SA",
        "display_name": "Serie A",
        "api_code": "SA",          # football-data.org competition code
        "theme_class": "theme-gazzetta",
        "url_prefix": "",          # Serie A is the default at "/"
    },
    "WC": {
        "code": "WC",
        "display_name": "World Cup 2026",
        "api_code": "WC",
        "theme_class": "theme-worldcup",
        "url_prefix": "/world-cup",
    },
}
DEFAULT_COMPETITION = "SA"

# -----------------------------------------------------------------------------
# Country flags for national teams
# -----------------------------------------------------------------------------
# Maps the team-name strings football-data.org returns for national sides to
# ISO 3166-1 alpha-2 codes. Each entry covers the common spelling variants we
# see in the wild (e.g. "USA" vs "United States", "Czechia" vs "Czech Republic").
# Includes all 2022 WC qualifiers + UEFA / CONMEBOL / AFC / CAF / CONCACAF / OFC
# contenders likely to feature in 2026. Adding a new country is a one-line
# append below — the flag emoji is computed from the ISO code at lookup time.
COUNTRY_TO_ISO: dict[str, str] = {
    # Hosts
    "USA": "US", "United States": "US",
    "Canada": "CA",
    "Mexico": "MX",
    # CONMEBOL
    "Argentina": "AR", "Brazil": "BR", "Uruguay": "UY", "Colombia": "CO",
    "Ecuador": "EC", "Paraguay": "PY", "Peru": "PE", "Chile": "CL",
    "Venezuela": "VE", "Bolivia": "BO",
    # UEFA
    "France": "FR", "Germany": "DE", "Italy": "IT", "Spain": "ES",
    "Portugal": "PT", "Netherlands": "NL", "Belgium": "BE", "Croatia": "HR",
    "Switzerland": "CH", "Denmark": "DK", "Poland": "PL", "Austria": "AT",
    "Serbia": "RS", "Turkey": "TR", "Türkiye": "TR", "Hungary": "HU",
    "Norway": "NO", "Sweden": "SE", "Ukraine": "UA",
    "Czech Republic": "CZ", "Czechia": "CZ",
    "Republic of Ireland": "IE", "Ireland": "IE", "Northern Ireland": "GB-NIR",
    "Slovakia": "SK", "Slovenia": "SI", "Romania": "RO", "Bulgaria": "BG",
    "Greece": "GR", "Iceland": "IS", "Finland": "FI", "Russia": "RU",
    "Bosnia and Herzegovina": "BA", "North Macedonia": "MK", "Albania": "AL",
    "Montenegro": "ME", "Kosovo": "XK", "Belarus": "BY", "Moldova": "MD",
    "Georgia": "GE", "Armenia": "AM", "Azerbaijan": "AZ", "Cyprus": "CY",
    "Estonia": "EE", "Latvia": "LV", "Lithuania": "LT", "Luxembourg": "LU",
    "Malta": "MT", "Israel": "IL", "Kazakhstan": "KZ",
    # Home nations — these use subdivision sequences (handled separately)
    "England": "GB-ENG", "Scotland": "GB-SCT", "Wales": "GB-WLS",
    # CONCACAF
    "Costa Rica": "CR", "Panama": "PA", "Honduras": "HN", "Jamaica": "JM",
    "El Salvador": "SV", "Guatemala": "GT", "Haiti": "HT",
    "Trinidad and Tobago": "TT", "Cuba": "CU", "Curaçao": "CW", "Curacao": "CW",
    "Suriname": "SR",
    # AFC
    "Japan": "JP",
    "South Korea": "KR", "Korea Republic": "KR", "Republic of Korea": "KR",
    "North Korea": "KP", "Korea DPR": "KP",
    "Iran": "IR", "IR Iran": "IR",
    "Saudi Arabia": "SA", "Australia": "AU", "Qatar": "QA", "Iraq": "IQ",
    "United Arab Emirates": "AE", "UAE": "AE",
    "Uzbekistan": "UZ", "Jordan": "JO", "Oman": "OM", "Lebanon": "LB",
    "Syria": "SY", "Palestine": "PS", "China": "CN", "China PR": "CN",
    "Vietnam": "VN", "Thailand": "TH", "Indonesia": "ID", "Malaysia": "MY",
    "Philippines": "PH", "Singapore": "SG", "India": "IN", "Bahrain": "BH",
    "Kuwait": "KW", "Yemen": "YE", "Kyrgyzstan": "KG", "Tajikistan": "TJ",
    "Turkmenistan": "TM",
    # CAF
    "Morocco": "MA", "Senegal": "SN", "Tunisia": "TN", "Algeria": "DZ",
    "Cameroon": "CM", "Egypt": "EG", "Ghana": "GH", "Nigeria": "NG",
    "Ivory Coast": "CI", "Côte d'Ivoire": "CI", "Cote d'Ivoire": "CI",
    "South Africa": "ZA", "Mali": "ML", "Burkina Faso": "BF",
    "Cape Verde": "CV", "Cape Verde Islands": "CV", "Cabo Verde": "CV",
    "DR Congo": "CD", "Congo DR": "CD",
    "Democratic Republic of the Congo": "CD", "Congo": "CG", "Gabon": "GA",
    "Equatorial Guinea": "GQ", "Mauritania": "MR", "Sudan": "SD",
    "South Sudan": "SS", "Ethiopia": "ET", "Uganda": "UG", "Kenya": "KE",
    "Tanzania": "TZ", "Zambia": "ZM", "Zimbabwe": "ZW", "Botswana": "BW",
    "Madagascar": "MG", "Mozambique": "MZ", "Namibia": "NA", "Guinea": "GN",
    "Guinea-Bissau": "GW", "Sierra Leone": "SL", "Liberia": "LR",
    "Benin": "BJ", "Togo": "TG", "Niger": "NE", "Comoros": "KM",
    "Angola": "AO", "Libya": "LY", "Rwanda": "RW", "Burundi": "BI",
    "Central African Republic": "CF", "Chad": "TD", "Eritrea": "ER",
    "Lesotho": "LS", "Malawi": "MW", "Eswatini": "SZ", "Mauritius": "MU",
    # OFC
    "New Zealand": "NZ", "Fiji": "FJ", "Solomon Islands": "SB", "Tahiti": "PF",
    "Papua New Guinea": "PG", "Vanuatu": "VU", "New Caledonia": "NC",
    "Samoa": "WS", "Tonga": "TO",
}

# Subdivision flags (England, Scotland, Wales, N. Ireland) — these are built
# from a black-flag base + Unicode tag characters spelling the ISO subdivision.
_SUBDIVISION_FLAGS = {
    "GB-ENG":   "🏴\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F",
    "GB-SCT":   "🏴\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F",
    "GB-WLS":   "🏴\U000E0067\U000E0062\U000E0077\U000E006C\U000E0073\U000E007F",
    "GB-NIR":   "🇬🇧",  # No standard tag sequence — fall back to UK flag.
}


def country_flag(team_name: str | None) -> str:
    """Return the emoji flag for a national team name, or '' if not a country.

    Used as a Jinja filter (`{{ team | country_flag }}`). Returns empty for
    club names so Serie A pages render unchanged — only WC fixtures get flags.
    """
    if not team_name:
        return ""
    iso = COUNTRY_TO_ISO.get(team_name)
    if not iso:
        return ""
    if iso in _SUBDIVISION_FLAGS:
        return _SUBDIVISION_FLAGS[iso]
    if len(iso) == 2 and iso.isalpha():
        # Regional indicator letters: 'A' (0x41) + 0x1F185 → 🇦, etc.
        return "".join(chr(0x1F1E6 + (ord(c.upper()) - ord("A"))) for c in iso)
    return ""

# -----------------------------------------------------------------------------
# Jinja filters
# -----------------------------------------------------------------------------

@app.template_filter("utc_iso")
def utc_iso(dt):
    """
    Emit an ISO-8601 UTC string for client-side local time conversion.
    """
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    s = dt.astimezone(timezone.utc).isoformat()
    return s[:-6] + "Z" if s.endswith("+00:00") else s


@app.template_filter("country_flag")
def _country_flag_filter(team_name):
    """Jinja shim around country_flag(). Empty string for non-country teams."""
    return country_flag(team_name)

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class User(db.Model, UserMixin):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String, unique=True, nullable=False)
    password_hash = db.Column(db.String, nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    invites_used = db.Column(db.Integer, default=0)
    bonus_points = db.Column(db.Integer, default=0)  # Admin-adjustable bonus points

    predictions = db.relationship("Prediction", back_populates="user", cascade="all, delete-orphan")

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    @property
    def prediction_points(self) -> int:
        """
        Points earned from correct predictions only.
        """
        total = 0
        for pred in self.predictions:
            fix = pred.fixture
            if fix is None or fix.home_score is None or fix.away_score is None:
                continue
            outcome = fix.outcome_code()
            if outcome and pred.selection == outcome:
                total += 1
        return total

    @property
    def points(self) -> int:
        """
        Total points: predictions + bonus points.
        """
        return self.prediction_points + (self.bonus_points or 0)


class Invite(db.Model):
    __tablename__ = "invites"
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String, unique=True, nullable=False)
    used_by_user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)


class Fixture(db.Model):
    __tablename__ = "fixtures"
    id = db.Column(db.Integer, primary_key=True)
    match_id = db.Column(db.String, unique=True, nullable=False)
    match_date = db.Column(db.DateTime, nullable=False)  # Original scheduled date (UTC)
    home_team = db.Column(db.String, nullable=False)
    away_team = db.Column(db.String, nullable=False)
    season = db.Column(db.String, nullable=False)
    matchday = db.Column(db.String, nullable=True)
    # Which competition this fixture belongs to (e.g. 'SA' for Serie A, 'WC'
    # for World Cup 2026). Keyed against the COMPETITIONS dict above.
    competition_code = db.Column(
        db.String, nullable=False, default="SA", server_default="SA", index=True,
    )
    status = db.Column(db.String, default="SCHEDULED")  # SCHEDULED/TIMED/IN_PLAY/PAUSED/FINISHED/POSTPONED/CANCELLED/SUSPENDED
    home_score = db.Column(db.Integer, nullable=True)
    away_score = db.Column(db.Integer, nullable=True)
    # Track manual edits
    scores_manually_edited = db.Column(db.Boolean, default=False)
    status_manually_edited = db.Column(db.Boolean, default=False)
    # For postponed fixtures: the new scheduled date (if known)
    rescheduled_date = db.Column(db.DateTime, nullable=True)
    # Notes for admin (e.g., "Postponed due to weather")
    admin_notes = db.Column(db.String, nullable=True)

    predictions = db.relationship("Prediction", back_populates="fixture", cascade="all, delete-orphan")

    def outcome_code(self) -> str | None:
        if self.home_score is None or self.away_score is None:
            return None
        if self.home_score > self.away_score:
            return "1"
        if self.home_score < self.away_score:
            return "2"
        return "X"

    def is_postponed(self) -> bool:
        """Check if the fixture is postponed, cancelled, or suspended."""
        return self.status in EXCLUDED_FROM_CURRENT

    def is_open_for_prediction(self) -> bool:
        """
        Determine whether predictions can still be made for this fixture.
        
        Predictions are locked once the match kickoff time has passed,
        OR if the fixture is postponed/cancelled/suspended.
        """
        # Can't predict on postponed/cancelled fixtures
        if self.is_postponed():
            return False
            
        now_utc = datetime.now(timezone.utc)
        kickoff = self.match_date
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        return now_utc < kickoff

    def display_status(self) -> str:
        """
        Human-friendly status for display in the UI.
        """
        # Handle postponed/cancelled/suspended first
        if self.status == 'POSTPONED':
            return 'PP'  # Short for postponed
        if self.status == 'CANCELLED':
            return 'CANC'
        if self.status == 'SUSPENDED':
            return 'SUSP'
            
        # If both scores are present, treat as finished
        if (self.home_score is not None) and (self.away_score is not None):
            return 'FT'
            
        base = self.status.split('_')[0] if '_' in (self.status or '') else (self.status or '')
        
        if base in ('IN_PLAY', 'PAUSED'):
            return 'LIVE'
        if base == 'FINISHED':
            return 'FT'
        return 'TIMED'

    def display_status_long(self) -> str:
        """
        Longer human-friendly status for tooltips and admin views.
        """
        status_map = {
            'SCHEDULED': 'Scheduled',
            'TIMED': 'Scheduled',
            'IN_PLAY': 'Live',
            'PAUSED': 'Half-time',
            'FINISHED': 'Full Time',
            'POSTPONED': 'Postponed',
            'CANCELLED': 'Cancelled',
            'SUSPENDED': 'Suspended',
        }
        return status_map.get(self.status, self.status or 'Unknown')


class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    fixture_id = db.Column(db.Integer, db.ForeignKey("fixtures.id"))
    selection = db.Column(db.String, nullable=False)  # '1', 'X', or '2'
    points_awarded = db.Column(db.Integer, nullable=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(ZoneInfo("America/New_York")))

    user = db.relationship("User", back_populates="predictions")
    fixture = db.relationship("Fixture", back_populates="predictions")

# -----------------------------------------------------------------------------
# Login manager
# -----------------------------------------------------------------------------

@login_manager.user_loader
def load_user(user_id: str):
    return db.session.get(User, int(user_id))

# -----------------------------------------------------------------------------
# Data fetch / sync
# -----------------------------------------------------------------------------

# Ordering for World Cup stage labels — used to sort "matchday" strings into
# chronological order. Higher number = later in the tournament.
WC_STAGE_ORDER = {
    "Group MD 1":      10,
    "Group MD 2":      11,
    "Group MD 3":      12,
    "Round of 32":     20,
    "Round of 16":     30,
    "Quarter-Finals":  40,
    "Semi-Finals":     50,
    "Third Place":     60,
    "Final":           70,
}


def _wc_matchday_label(stage: str | None, matchday_num) -> str:
    """Translate a football-data.org WC stage into a friendly matchday label.

    Group stage matches get a per-round label (Group MD 1/2/3) using the
    API's `matchday` field. Knockout stages are mapped to their common
    English names. Unknown stages fall back to a title-cased version of
    the raw API value so nothing is silently dropped.
    """
    s = (stage or "").upper().replace("-", "_")
    if s == "GROUP_STAGE":
        n = matchday_num if matchday_num else 1
        return f"Group MD {n}"
    mapping = {
        "LAST_32":            "Round of 32",
        "ROUND_OF_32":        "Round of 32",
        "LAST_16":            "Round of 16",
        "ROUND_OF_16":        "Round of 16",
        "LAST_8":             "Quarter-Finals",
        "QUARTER_FINALS":     "Quarter-Finals",
        "LAST_4":             "Semi-Finals",
        "SEMI_FINALS":        "Semi-Finals",
        "THIRD_PLACE":        "Third Place",
        "THIRD_PLACE_FINAL":  "Third Place",
        "FINAL":              "Final",
    }
    return mapping.get(s, s.replace("_", " ").title()) if s else ""


def _matchday_sort_key(md: str | None, competition_code: str):
    """Sort key for matchdays.

    Serie A matchdays are integers ('1'..'38'). World Cup matchdays are
    stage labels; we look them up in WC_STAGE_ORDER for a chronological
    sort instead of a misleading lexicographic one.
    """
    if md is None:
        return (9, 9999, "")
    if competition_code == "WC":
        return (0, WC_STAGE_ORDER.get(md, 999), md)
    try:
        return (0, int(md), "")
    except (TypeError, ValueError):
        return (1, 0, md)


def fetch_fixtures_from_api(competition_code: str = "SA") -> list[dict]:
    """Fetch fixtures from football-data.org for one competition.

    Empty list on missing API key, unknown competition, non-200 response,
    or any network error — callers should treat empty as "nothing to sync
    this tick" rather than fatal.
    """
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        return []

    comp = COMPETITIONS.get(competition_code)
    if not comp:
        return []
    api_code = comp["api_code"]

    today = datetime.now().date()

    # Each competition decides its own season string. World Cup 2026 is a
    # single-summer tournament so it gets a fixed "2026" tag; club leagues
    # use the standard "YYYY-YY" form rolling over in July.
    if competition_code == "WC":
        season_start_year = 2026
        season_str = "2026"
    else:
        season_start_year = today.year if today.month >= 7 else today.year - 1
        season_str = f"{season_start_year}-{(season_start_year + 1) % 100:02d}"

    url = f"https://api.football-data.org/v4/competitions/{api_code}/matches"
    headers = {"X-Auth-Token": api_key}
    params = {"season": season_start_year}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
    except Exception:
        return []

    fixtures: list[dict] = []
    for match in data.get("matches", []):
        status = match.get("status")
        # Include POSTPONED status from API
        if status not in ("SCHEDULED", "TIMED", "IN_PLAY", "PAUSED", "FINISHED", "POSTPONED", "CANCELLED", "SUSPENDED"):
            continue

        # World Cup knockout fixtures appear on the schedule before the
        # qualifying teams are known. The API returns them with
        # homeTeam.name / awayTeam.name == null. Skip these — they'll
        # be picked up on a later sync once the bracket fills in.
        home_team_name = (match.get("homeTeam") or {}).get("name")
        away_team_name = (match.get("awayTeam") or {}).get("name")
        if not home_team_name or not away_team_name:
            continue

        utc_date_str = match["utcDate"]
        utc_dt = datetime.fromisoformat(utc_date_str.replace("Z", "+00:00"))

        score = match.get("score", {}) or {}
        ft = score.get("fullTime") or {}
        home_ft = ft.get("home")
        away_ft = ft.get("away")

        if home_ft is None or away_ft is None:
            reg = score.get("regularTime") or {}
            home_ft = reg.get("home") if home_ft is None else home_ft
            away_ft = reg.get("away") if away_ft is None else away_ft

        # Matchday label depends on the competition shape.
        if competition_code == "WC":
            matchday_label = _wc_matchday_label(match.get("stage"), match.get("matchday"))
        else:
            matchday_label = str(match.get("matchday"))

        fixtures.append({
            "match_id": str(match["id"]),
            "match_date": utc_dt,
            "home_team": home_team_name,
            "away_team": away_team_name,
            "season": season_str,
            "matchday": matchday_label,
            "status": status,
            "home_score": home_ft if home_ft is not None else None,
            "away_score": away_ft if away_ft is not None else None,
            "competition_code": competition_code,
        })
    return fixtures


def fetch_fixtures_from_fallback() -> list[dict]:
    fallback_path = Path(__file__).resolve().parent / "data" / "seriea_2024_25.json"
    if not fallback_path.exists():
        return []

    with open(fallback_path, "r", encoding="utf-8") as f:
        season_data = json.load(f)

    fixtures: list[dict] = []
    for match in season_data.get("matches", []):
        score = match.get("score", {})
        ft = score.get("ft")
        
        date_str = match["date"]
        time_str = match.get("time", "18:00")
        dt_naive = datetime.fromisoformat(f"{date_str}T{time_str}")
        dt_rome = dt_naive.replace(tzinfo=ZoneInfo("Europe/Rome"))
        utc_dt = dt_rome.astimezone(timezone.utc)
        
        # Determine status based on score availability
        if ft:
            status = "FINISHED"
            home_score = ft[0]
            away_score = ft[1]
        else:
            status = "SCHEDULED"
            home_score = None
            away_score = None
            
        fixtures.append({
            "match_id": f"{date_str}-{match['team1']}-{match['team2']}",
            "match_date": utc_dt,
            "home_team": match["team1"],
            "away_team": match["team2"],
            "season": season_data.get("name", "2024/25"),
            "matchday": match.get("round"),
            "status": status,
            "home_score": home_score,
            "away_score": away_score,
            "competition_code": "SA",
        })
    return fixtures


def update_fixtures() -> None:
    """
    Sync local fixtures across all competitions (Serie A + World Cup).

    IMPORTANT:
    - Skip updates for fixtures where status_manually_edited is True
    - Only skip score updates when scores_manually_edited is True
    """
    try:
        db.create_all()
    except Exception:
        pass

    # Sync each competition independently. If the API call for one fails we
    # still want the others to proceed.
    fixtures_to_use: list[dict] = []

    # Serie A: API first, fall back to bundled JSON if the call returns empty
    sa_from_api = fetch_fixtures_from_api("SA")
    fixtures_to_use.extend(sa_from_api if sa_from_api else fetch_fixtures_from_fallback())

    # World Cup 2026: API only. Silent no-op if the plan tier excludes it.
    fixtures_to_use.extend(fetch_fixtures_from_api("WC"))

    for fi in fixtures_to_use:
        # Defensive guard: a fixture dict with missing teams or match_id
        # is unusable. fetch_fixtures_from_api already skips these for the
        # WC knockout-placeholder case, but this protects against any
        # future caller producing a malformed row.
        if not fi.get('match_id') or not fi.get('home_team') or not fi.get('away_team'):
            continue

        existing = Fixture.query.filter_by(match_id=fi['match_id']).first()
        if existing:
            updated = False
            
            # Skip all updates if status was manually edited (e.g., marked as POSTPONED)
            if existing.status_manually_edited:
                continue
                
            home_sc = fi['home_score']
            away_sc = fi['away_score']
            scores_locked = existing.scores_manually_edited
            
            if not scores_locked:
                if home_sc is not None and home_sc != existing.home_score:
                    existing.home_score = home_sc
                    updated = True
                if away_sc is not None and away_sc != existing.away_score:
                    existing.away_score = away_sc
                    updated = True
                    
                # Update status from API (including POSTPONED)
                if fi['status'] != existing.status:
                    existing.status = fi['status']
                    updated = True
                    
                if home_sc is not None and away_sc is not None:
                    if existing.status != 'FINISHED':
                        existing.status = 'FINISHED'
                        updated = True

            # Update kickoff time if different
            api_dt = fi['match_date']
            if api_dt and existing.match_date:
                try:
                    delta = abs((existing.match_date - api_dt).total_seconds())
                except Exception:
                    delta = None
                if delta is not None and delta > 60:
                    existing.match_date = api_dt
                    updated = True

            if updated:
                db.session.add(existing)
        else:
            # Try to reconcile legacy row
            from sqlalchemy import and_
            dt = fi['match_date']
            lo = dt - timedelta(hours=12)
            hi = dt + timedelta(hours=12)
            comp_code = fi.get('competition_code', 'SA')
            legacy = (
                Fixture.query
                .filter(
                    Fixture.competition_code == comp_code,
                    Fixture.season == fi['season'],
                    func.lower(Fixture.home_team) == fi['home_team'].lower(),
                    func.lower(Fixture.away_team) == fi['away_team'].lower(),
                    Fixture.match_date >= lo,
                    Fixture.match_date <= hi,
                )
                .order_by(Fixture.match_date.asc())
                .first()
            )

            if not legacy and fi.get('matchday'):
                legacy = (
                    Fixture.query
                    .filter(
                        Fixture.competition_code == comp_code,
                        Fixture.season == fi['season'],
                        Fixture.matchday == fi['matchday'],
                        func.lower(Fixture.home_team) == fi['home_team'].lower(),
                        func.lower(Fixture.away_team) == fi['away_team'].lower(),
                    )
                    .order_by(Fixture.match_date.asc())
                    .first()
                )
                
            if legacy:
                # Don't update if manually edited
                if legacy.status_manually_edited:
                    continue
                    
                legacy.match_id = fi['match_id']
                if not legacy.scores_manually_edited:
                    legacy.status = fi['status']
                    legacy.home_score = fi['home_score']
                    legacy.away_score = fi['away_score']
                    if fi['home_score'] is not None and fi['away_score'] is not None:
                        legacy.status = 'FINISHED'
                try:
                    if abs((legacy.match_date - dt).total_seconds()) > 60:
                        legacy.match_date = dt
                except Exception:
                    legacy.match_date = dt
                db.session.add(legacy)
            else:
                home_sc = fi['home_score']
                away_sc = fi['away_score']
                status = fi['status']
                if home_sc is not None and away_sc is not None:
                    status = 'FINISHED'
                db.session.add(Fixture(
                    match_id=fi['match_id'],
                    match_date=fi['match_date'],
                    home_team=fi['home_team'],
                    away_team=fi['away_team'],
                    season=fi['season'],
                    matchday=fi.get('matchday'),
                    status=status,
                    home_score=home_sc,
                    away_score=away_sc,
                    competition_code=fi.get('competition_code', 'SA'),
                    scores_manually_edited=False,
                    status_manually_edited=False,
                ))
        
    db.session.commit()
    evaluate_predictions()


# --- Adaptive fetch throttle ---

FETCH_STATE = {"last_run": None, "last_interval": None}

def _adaptive_min_interval() -> timedelta:
    now_utc = datetime.now(timezone.utc)

    live = Fixture.query.filter(Fixture.status.in_(("IN_PLAY", "PAUSED"))).count()
    if live > 0:
        return timedelta(seconds=60)

    soon = (
        Fixture.query
        .filter(
            Fixture.match_date >= now_utc,
            Fixture.match_date <= now_utc + timedelta(hours=2),
            ~Fixture.status.in_(EXCLUDED_FROM_CURRENT)
        )
        .count()
    )
    if soon > 0:
        return timedelta(seconds=60)

    today_end_utc = now_utc.replace(hour=23, minute=59, second=59, microsecond=999999)
    today = (
        Fixture.query
        .filter(
            Fixture.match_date >= now_utc,
            Fixture.match_date <= today_end_utc,
            ~Fixture.status.in_(EXCLUDED_FROM_CURRENT)
        )
        .count()
    )
    if today > 0:
        return timedelta(seconds=60)

    return timedelta(hours=24)


def update_fixtures_adaptive(force: bool = False) -> None:
    now_utc = datetime.now(timezone.utc)
    min_interval = _adaptive_min_interval()
    last_run = FETCH_STATE["last_run"]

    if not force and last_run is not None and (now_utc - last_run) < min_interval:
        return

    update_fixtures()
    FETCH_STATE["last_run"] = now_utc
    FETCH_STATE["last_interval"] = min_interval

# -----------------------------------------------------------------------------
# Queries / helpers
# -----------------------------------------------------------------------------

def upcoming_fixtures(exclude_postponed: bool = True) -> list[Fixture]:
    """
    Return fixtures to show on the main page.
    
    By default, excludes postponed/cancelled/suspended fixtures.
    """
    now_utc = datetime.now(timezone.utc)
    window_start = now_utc - timedelta(hours=6)
    window_end = now_utc + timedelta(days=7)

    query = Fixture.query.filter(
        Fixture.match_date >= window_start,
        Fixture.match_date <= window_end
    )
    
    if exclude_postponed:
        query = query.filter(~Fixture.status.in_(EXCLUDED_FROM_CURRENT))
    
    base = query.all()

    # Pull entire first upcoming matchday
    first_upcoming_query = (
        Fixture.query.filter(
            Fixture.status.in_(("SCHEDULED", "TIMED")),
            Fixture.match_date >= now_utc
        )
    )
    if exclude_postponed:
        first_upcoming_query = first_upcoming_query.filter(~Fixture.status.in_(EXCLUDED_FROM_CURRENT))
    
    first_upcoming = first_upcoming_query.order_by(Fixture.match_date.asc()).first()

    week1 = []
    if first_upcoming and first_upcoming.matchday:
        week1_query = Fixture.query.filter(Fixture.matchday == first_upcoming.matchday)
        if exclude_postponed:
            week1_query = week1_query.filter(~Fixture.status.in_(EXCLUDED_FROM_CURRENT))
        week1 = week1_query.all()

    merged = {f.id: f for f in base}
    for f in week1:
        merged[f.id] = f
    return sorted(merged.values(), key=lambda f: f.match_date)


def get_postponed_fixtures(season: str = None, competition_code: str = "SA") -> list[Fixture]:
    """
    Get all postponed/cancelled/suspended fixtures for one competition,
    optionally filtered by season.
    """
    query = Fixture.query.filter(
        Fixture.status.in_(EXCLUDED_FROM_CURRENT),
        Fixture.competition_code == competition_code,
    )
    if season:
        query = query.filter(Fixture.season == season)
    return query.order_by(Fixture.match_date.asc()).all()


def predictions_for_fixtures(fixtures: list[Fixture]) -> dict[int, list[tuple[str, str]]]:
    if not fixtures:
        return {}
    ids = [f.id for f in fixtures]
    rows = (
        db.session.query(Prediction.fixture_id, User.username, Prediction.selection)
        .join(User, User.id == Prediction.user_id)
        .filter(Prediction.fixture_id.in_(ids))
        .order_by(User.username.asc())
        .all()
    )
    out: dict[int, list[tuple[str, str]]] = {}
    for fixture_id, username, selection in rows:
        out.setdefault(fixture_id, []).append((username, selection))
    return out


def prediction_matrix(fixtures):
    """Build the prediction matrix for a list of Fixture objects."""
    if not fixtures:
        return [], {}, {}

    fix_ids = [f.id for f in fixtures]
    rows = (
        db.session.query(User.id, User.username, Prediction.fixture_id, Prediction.selection, Fixture.match_id)
        .join(Prediction, Prediction.user_id == User.id)
        .join(Fixture, Fixture.id == Prediction.fixture_id)
        .filter(Prediction.fixture_id.in_(fix_ids))
        .all()
    )

    user_set = {}
    for uid, uname, _, _, _ in rows:
        user_set[uid] = uname
    users = sorted(user_set.items(), key=lambda t: t[1].lower())

    matrix: dict[tuple[str|int,int], str] = {}
    for uid, uname, fid, sel, match_id in rows:
        matrix[(fid, uid)] = sel
        if match_id:
            matrix[(match_id, uid)] = sel

    show_flags = {}
    now_utc = datetime.now(timezone.utc)
    for f in fixtures:
        kickoff = f.match_date
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        show_flags[f.id] = now_utc >= kickoff

    return users, matrix, show_flags


def evaluate_predictions() -> None:
    """Assign points to predictions for fixtures with final scores."""
    finished_fixtures = (
        Fixture.query
        .filter(Fixture.home_score.isnot(None), Fixture.away_score.isnot(None))
        .all()
    )
    for fixture in finished_fixtures:
        outcome = fixture.outcome_code()
        if outcome is None:
            continue
        for prediction in fixture.predictions:
            if prediction.points_awarded is None:
                prediction.points_awarded = 1 if prediction.selection == outcome else 0
                db.session.add(prediction)
    db.session.commit()


# Season/matchday helpers
#
# All of these accept a `competition_code` (default 'SA' so existing Serie A
# routes keep working unchanged). Phase 3 will add World Cup routes that pass
# 'WC' through to isolate the WC view from Serie A data.

def seasons_available(competition_code: str = "SA") -> list[str]:
    rows = (
        db.session.query(Fixture.season)
        .filter(Fixture.competition_code == competition_code)
        .distinct()
        .all()
    )
    return sorted([r[0] for r in rows])


def matchdays_for(season: str, competition_code: str = "SA") -> list[str]:
    rows = (
        db.session.query(Fixture.matchday)
        .filter(
            Fixture.season == season,
            Fixture.competition_code == competition_code,
        )
        .distinct()
        .all()
    )
    days = [r[0] for r in rows if r[0]]
    return sorted(set(days), key=lambda d: _matchday_sort_key(d, competition_code))


def latest_completed_matchday(season: str, competition_code: str = "SA") -> str | None:
    """Find the latest matchday where all non-postponed fixtures have scores."""
    if not season:
        return None

    days = matchdays_for(season, competition_code)
    if not days:
        return None

    sorted_days = sorted(
        set(days),
        key=lambda d: _matchday_sort_key(d, competition_code),
        reverse=True,
    )

    for md in sorted_days:
        # Get non-postponed fixtures for this matchday
        fixtures = Fixture.query.filter(
            Fixture.competition_code == competition_code,
            Fixture.season == season,
            Fixture.matchday == md,
            ~Fixture.status.in_(EXCLUDED_FROM_CURRENT)
        ).all()

        if not fixtures:
            continue

        all_complete = all(
            (f.home_score is not None and f.away_score is not None)
            for f in fixtures
        )

        if all_complete:
            return md

    return None


def current_home_matchday(season: str, competition_code: str = "SA") -> str | None:
    """
    Determine which matchday to present on the home page.

    Prioritises the earliest matchday that has any non-postponed fixture
    without final results.
    """
    if not season:
        return None

    days = matchdays_for(season, competition_code)
    if not days:
        return None

    sorted_days = sorted(
        set(days),
        key=lambda d: _matchday_sort_key(d, competition_code),
    )

    for md in sorted_days:
        # Check for incomplete non-postponed fixtures
        incomplete = (
            db.session.query(Fixture.id)
            .filter(
                Fixture.competition_code == competition_code,
                Fixture.season == season,
                Fixture.matchday == md,
                ~Fixture.status.in_(EXCLUDED_FROM_CURRENT),
                db.or_(
                    Fixture.home_score.is_(None),
                    Fixture.away_score.is_(None)
                )
            )
            .first()
        )
        if incomplete is not None:
            return md

    return latest_completed_matchday(season, competition_code)


def weekly_user_points(season: str, matchday: str, competition_code: str = "SA"):
    """Return (user_id, username, points) for the given season and matchday."""
    predictions = (
        db.session.query(Prediction, Fixture, User)
        .join(Fixture, Fixture.id == Prediction.fixture_id)
        .join(User, User.id == Prediction.user_id)
        .filter(
            Fixture.competition_code == competition_code,
            Fixture.season == season,
            Fixture.matchday == str(matchday),
        )
        .all()
    )
    user_points: dict[int, int] = {}
    user_names: dict[int, str] = {}
    for pred, fix, user in predictions:
        user_names[user.id] = user.username
        if fix.home_score is None or fix.away_score is None:
            continue
        outcome = fix.outcome_code()
        if outcome and pred.selection == outcome:
            user_points[user.id] = user_points.get(user.id, 0) + 1
        else:
            user_points.setdefault(user.id, user_points.get(user.id, 0))
    rows = [(uid, user_names.get(uid, ""), pts) for uid, pts in user_points.items()]
    return sorted(rows, key=lambda r: (-r[2], r[1].lower()))


def current_season_from_db(competition_code: str = "SA") -> str | None:
    row = (
        db.session.query(Fixture.season)
        .filter(Fixture.competition_code == competition_code)
        .order_by(Fixture.match_date.desc())
        .first()
    )
    return row[0] if row else None


def all_matchdays_for_season(season: str, competition_code: str = "SA") -> list[str]:
    rows = (
        db.session.query(distinct(Fixture.matchday))
        .filter(
            Fixture.competition_code == competition_code,
            Fixture.season == season,
        )
        .all()
    )
    mds = [r[0] for r in rows if r[0] is not None]
    return sorted(set(mds), key=lambda d: _matchday_sort_key(d, competition_code))


def classify_matchdays(season: str, competition_code: str = "SA"):
    now_utc = datetime.now(timezone.utc)
    md_status = {}
    for md in all_matchdays_for_season(season, competition_code):
        qs = Fixture.query.filter_by(
            competition_code=competition_code,
            season=season,
            matchday=md,
        ).all()
        statuses = {f.status for f in qs}
        if any(s in ("IN_PLAY","PAUSED") for s in statuses):
            md_status[md] = "live"
        elif statuses and statuses.issubset({"FINISHED"}):
            md_status[md] = "finished"
        elif any(s in ("SCHEDULED","TIMED") for s in statuses):
            future = Fixture.query.filter(
                Fixture.competition_code == competition_code,
                Fixture.season == season,
                Fixture.matchday == md,
                Fixture.match_date >= now_utc,
            ).count()
            md_status[md] = "upcoming" if future else "finished"
        else:
            md_status[md] = "other"

    def _order(lst):
        return sorted(set(lst), key=lambda d: _matchday_sort_key(d, competition_code))

    finished = _order([m for m,s in md_status.items() if s=="finished"])
    live = _order([m for m,s in md_status.items() if s=="live"])
    upcoming = _order([m for m,s in md_status.items() if s=="upcoming"])
    other = _order([m for m,s in md_status.items() if s=="other"])
    return finished, live, upcoming, other


def season_user_points(season: str, competition_code: str = "SA"):
    """Return list of dicts {username: points} for the specified season."""
    predictions = (
        db.session.query(Prediction, Fixture, User)
        .join(Fixture, Fixture.id == Prediction.fixture_id)
        .join(User, User.id == Prediction.user_id)
        .filter(
            Fixture.competition_code == competition_code,
            Fixture.season == season,
        )
        .all()
    )
    user_points: dict[str, int] = {}
    for pred, fix, user in predictions:
        if fix.home_score is None or fix.away_score is None:
            continue
        outcome = fix.outcome_code()
        if outcome and pred.selection == outcome:
            user_points[user.username] = user_points.get(user.username, 0) + 1
        else:
            user_points.setdefault(user.username, user_points.get(user.username, 0))
    rows = [{"username": uname, "points": pts} for uname, pts in user_points.items()]
    return sorted(rows, key=lambda x: (-x["points"], x["username"].lower()))


def overall_user_points(competition_code: str = "SA"):
    """Return list of dicts {username: points} summed across all seasons
    of the given competition. Used for the leaderboard's "Overall" scope —
    intentionally excludes admin bonus points so Serie A adjustments don't
    leak into the World Cup tally and vice versa.
    """
    predictions = (
        db.session.query(Prediction, Fixture, User)
        .join(Fixture, Fixture.id == Prediction.fixture_id)
        .join(User, User.id == Prediction.user_id)
        .filter(Fixture.competition_code == competition_code)
        .all()
    )
    user_points: dict[str, int] = {}
    for pred, fix, user in predictions:
        user_points.setdefault(user.username, 0)
        if fix.home_score is None or fix.away_score is None:
            continue
        outcome = fix.outcome_code()
        if outcome and pred.selection == outcome:
            user_points[user.username] += 1

    # Include users who haven't predicted anything yet so the board lists
    # everyone (with 0 points).
    for u in User.query.all():
        user_points.setdefault(u.username, 0)

    rows = [{"username": uname, "points": pts} for uname, pts in user_points.items()]
    return sorted(rows, key=lambda x: (-x["points"], x["username"].lower()))


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

def _view_fixtures(competition_code: str):
    """Render the home / fixtures page for the given competition."""
    update_fixtures_adaptive()

    season = current_season_from_db(competition_code)
    if not season:
        flash("No season data available yet.", "warning")
        return render_template(
            'index.html',
            fixtures=[],
            user_predictions={},
            users_cols=[],
            pred_matrix={},
            show_preds_flags={},
            season=None,
            matchday=None,
            postponed_count=0,
        )

    md = current_home_matchday(season, competition_code)
    if not md:
        flash("No matchdays available yet.", "warning")
        return render_template(
            'index.html',
            fixtures=[],
            user_predictions={},
            users_cols=[],
            pred_matrix={},
            show_preds_flags={},
            season=season,
            matchday=None,
            postponed_count=0,
        )

    # Get non-postponed fixtures for the current matchday
    fixtures = (
        Fixture.query
        .filter(
            Fixture.competition_code == competition_code,
            Fixture.season == season,
            Fixture.matchday == str(md),
            ~Fixture.status.in_(EXCLUDED_FROM_CURRENT)
        )
        .order_by(Fixture.match_date.asc())
        .all()
    )

    # Count postponed fixtures for the badge
    postponed_count = Fixture.query.filter(
        Fixture.competition_code == competition_code,
        Fixture.season == season,
        Fixture.status.in_(EXCLUDED_FROM_CURRENT)
    ).count()

    user_predictions = {p.fixture_id: p for p in current_user.predictions}
    users_cols, pred_matrix, show_flags = prediction_matrix(fixtures)

    return render_template(
        'index.html',
        fixtures=fixtures,
        user_predictions=user_predictions,
        users_cols=users_cols,
        pred_matrix=pred_matrix,
        show_preds_flags=show_flags,
        season=season,
        matchday=md,
        postponed_count=postponed_count,
    )


def _view_postponed(competition_code: str):
    """Render the postponed / cancelled / suspended list for the competition."""
    season = request.args.get('season') or current_season_from_db(competition_code)
    seasons = seasons_available(competition_code)

    postponed = get_postponed_fixtures(season, competition_code) if season else []

    return render_template(
        'postponed.html',
        fixtures=postponed,
        season=season,
        seasons=seasons,
    )


@app.route('/')
@login_required
def index():
    """Default landing page: send users straight to the World Cup tab."""
    return redirect(url_for('worldcup.index'))


@app.route('/seriea/')
@login_required
def seriea_index():
    """Stable URL for the Serie A fixtures page (decoupled from '/' since
    the root now redirects to the World Cup tab)."""
    return _view_fixtures('SA')


@app.route('/postponed')
@login_required
def postponed_fixtures_view():
    """View all postponed/cancelled/suspended fixtures."""
    return _view_postponed('SA')


@app.route("/predict/<int:fixture_id>", methods=["POST"])
@login_required
def predict(fixture_id: int):
    fixture = db.session.get(Fixture, fixture_id)
    if not fixture:
        abort(404)

    # Redirect back to the fixture's own competition tab.
    return_endpoint = (
        'worldcup.index' if fixture.competition_code == 'WC' else 'seriea_index'
    )

    if not fixture.is_open_for_prediction():
        flash("Predictions are locked for this fixture.", "warning")
        return redirect(url_for(return_endpoint))

    selection = request.form.get("selection")
    if selection not in ("1", "X", "2"):
        flash("Invalid prediction.", "danger")
        return redirect(url_for(return_endpoint))

    prediction = Prediction.query.filter_by(user_id=current_user.id, fixture_id=fixture_id).first()
    if prediction:
        prediction.selection = selection
        flash("Prediction updated.", "success")
    else:
        prediction = Prediction(user_id=current_user.id, fixture_id=fixture_id, selection=selection)
        db.session.add(prediction)
        flash("Prediction submitted.", "success")

    db.session.commit()
    return redirect(url_for(return_endpoint))


def _save_all_predictions_view(competition_code: str, redirect_endpoint: str):
    """Persist all predictions submitted from the fixtures form and redirect.

    Scopes to fixtures of the given competition so a stray hidden input from
    a different tab can't cross-write predictions for the other tournament.
    """
    all_fixtures = Fixture.query.filter_by(competition_code=competition_code).all()
    for fixture in all_fixtures:
        if not fixture.is_open_for_prediction():
            continue
        choice = request.form.get(f"fixture_{fixture.id}")
        if choice not in ("1", "X", "2", None):
            continue
        if choice:
            pred = Prediction.query.filter_by(user_id=current_user.id, fixture_id=fixture.id).first()
            if not pred:
                pred = Prediction(user_id=current_user.id, fixture_id=fixture.id)
                db.session.add(pred)
            pred.selection = choice
    db.session.commit()
    flash("All predictions saved!", "success")
    return redirect(url_for(redirect_endpoint))


@app.route("/save_all_predictions", methods=["POST"])
@login_required
def save_all_predictions():
    return _save_all_predictions_view('SA', 'seriea_index')


def _view_leaderboard(competition_code: str, redirect_endpoint: str):
    update_fixtures_adaptive()
    evaluate_predictions()

    seasons = seasons_available(competition_code)
    current_season = current_season_from_db(competition_code) or (seasons[-1] if seasons else None)

    raw_scope = request.args.get("scope")
    scope = (raw_scope or "season").lower()
    season = request.args.get("season")
    matchday = request.args.get("matchday")

    if raw_scope is None and current_season:
        return redirect(url_for(redirect_endpoint, scope="season", season=current_season))

    if scope == "week":
        if not season:
            season = current_season
        days = matchdays_for(season, competition_code)
        if not matchday:
            matchday = latest_completed_matchday(season, competition_code) or (days[-1] if days else None)
        rows = weekly_user_points(season, matchday, competition_code) if matchday else []
        users_sorted = [{"username": r[1], "points": int(r[2])} for r in rows]
        return render_template(
            "leaderboard.html",
            users=users_sorted,
            scope="week",
            seasons=seasons,
            season=season,
            matchdays=days,
            matchday=matchday
        )

    if scope == "season":
        if not season:
            season = current_season
        users_sorted = season_user_points(season, competition_code)
        return render_template(
            "leaderboard.html",
            users=users_sorted,
            scope="season",
            seasons=seasons,
            season=season,
            matchdays=matchdays_for(season, competition_code) if season else [],
            matchday=None
        )

    # Overall scope: sum points across all seasons of this competition only.
    # This intentionally diverges from the original implementation which used
    # User.points (a mix of all competitions + admin bonus). Each competition
    # gets its own clean tally now.
    users_sorted = overall_user_points(competition_code)
    return render_template(
        "leaderboard.html",
        users=users_sorted,
        scope="overall",
        seasons=seasons,
        season=season or current_season,
        matchdays=matchdays_for(season or current_season, competition_code) if (season or current_season) else [],
        matchday=matchday
    )


def _view_history(competition_code: str):
    update_fixtures_adaptive()

    seasons = seasons_available(competition_code)
    current_season = current_season_from_db(competition_code) or (seasons[-1] if seasons else None)

    season = request.args.get("season") or current_season
    matchday = request.args.get("matchday") or None

    if season:
        if not matchday:
            matchday = latest_completed_matchday(season, competition_code)
            if not matchday:
                days = matchdays_for(season, competition_code)
                matchday = days[0] if days else None
    else:
        matchday = None

    fixtures = []
    if season and matchday:
        fixtures = (
            Fixture.query
            .filter(
                Fixture.competition_code == competition_code,
                Fixture.season == season,
                Fixture.matchday == str(matchday),
            )
            .order_by(Fixture.match_date.asc())
            .all()
        )

    users_cols, pred_matrix, show_flags = prediction_matrix(fixtures)

    return render_template(
        "history.html",
        fixtures=fixtures,
        users_cols=users_cols,
        pred_matrix=pred_matrix,
        show_preds_flags=show_flags,
        seasons=seasons,
        season=season,
        matchdays=matchdays_for(season, competition_code) if season else [],
        matchday=matchday
    )


@app.route("/leaderboard")
@login_required
def leaderboard():
    return _view_leaderboard('SA', 'leaderboard')


@app.route("/history")
@login_required
def history():
    return _view_history('SA')


# -----------------------------------------------------------------------------
# World Cup 2026 — parallel route tree mounted at /world-cup
# -----------------------------------------------------------------------------
# Each WC route reuses the same view body as its Serie A twin, just passing
# 'WC' for competition_code so the queries scope to World Cup fixtures.

worldcup_bp = Blueprint('worldcup', __name__, url_prefix='/world-cup')


@worldcup_bp.before_request
def _set_wc_competition():
    g.competition_code = 'WC'
    g.competition = COMPETITIONS['WC']


@worldcup_bp.route('/')
@login_required
def index():
    return _view_fixtures('WC')


@worldcup_bp.route('/postponed')
@login_required
def postponed_fixtures_view():
    return _view_postponed('WC')


@worldcup_bp.route('/leaderboard')
@login_required
def leaderboard():
    return _view_leaderboard('WC', 'worldcup.leaderboard')


@worldcup_bp.route('/history')
@login_required
def history():
    return _view_history('WC')


@worldcup_bp.route('/save_all_predictions', methods=['POST'])
@login_required
def save_all_predictions():
    return _save_all_predictions_view('WC', 'worldcup.index')


app.register_blueprint(worldcup_bp)


# -----------------------------------------------------------------------------
# Template / request context helpers for multi-competition routing
# -----------------------------------------------------------------------------

@app.before_request
def _set_default_competition():
    """Default to Serie A context unless we're inside the WC blueprint."""
    if not request.endpoint or not request.endpoint.startswith('worldcup.'):
        g.competition_code = 'SA'
        g.competition = COMPETITIONS['SA']


@app.context_processor
def _inject_competition_helpers():
    """Expose competition-aware helpers to all templates.

    Templates call `url_for_comp(name)` instead of `url_for('index')` etc.
    The helper resolves the right endpoint based on the current request's
    competition context, so a single layout.html can serve both tabs.
    """
    endpoint_map = {
        'SA': {
            'fixtures':    'seriea_index',
            'postponed':   'postponed_fixtures_view',
            'leaderboard': 'leaderboard',
            'history':     'history',
            'save_all':    'save_all_predictions',
        },
        'WC': {
            'fixtures':    'worldcup.index',
            'postponed':   'worldcup.postponed_fixtures_view',
            'leaderboard': 'worldcup.leaderboard',
            'history':     'worldcup.history',
            'save_all':    'worldcup.save_all_predictions',
        },
    }

    def url_for_comp(name, code=None, **kwargs):
        code = code or getattr(g, 'competition_code', 'SA')
        return url_for(endpoint_map[code][name], **kwargs)

    return {
        'url_for_comp': url_for_comp,
        'current_competition': getattr(g, 'competition', COMPETITIONS['SA']),
        'COMPETITIONS': COMPETITIONS,
    }


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=True)  # Keep user logged in for 30 days
            return redirect(url_for("index"))
        flash("Invalid username or password", "danger")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        try:
            username = request.form.get("username")
            password = request.form.get("password")
            invite_code = request.form.get("invite")

            invite = Invite.query.filter_by(code=invite_code, used_by_user_id=None).first()
            if not invite:
                flash("Invalid or used invite code.", "danger")
                return render_template("register.html")

            if User.query.filter_by(username=username).first():
                flash("Username already exists.", "danger")
                return render_template("register.html")

            if not username or not password:
                flash("Username and password are required.", "danger")
                return render_template("register.html")

            user = User(username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.flush()

            invite.used_by_user_id = user.id
            db.session.add(invite)
            db.session.commit()

            flash("Registration successful. Please log in.", "success")
            return redirect(url_for("login"))
        except Exception as e:
            import traceback
            print("[REGISTER_ERROR]", repr(e))
            traceback.print_exc()
            db.session.rollback()
            flash("Registration failed due to a server error. Please try again.", "danger")
            return render_template("register.html")
    return render_template("register.html")


@app.route("/admin", methods=["GET", "POST"])
@login_required
def admin():
    if not current_user.is_admin:
        abort(403)
    if request.method == "POST":
        code = request.form.get("code")
        if code:
            if Invite.query.filter_by(code=code).first():
                flash("Invite code already exists.", "danger")
            else:
                invite = Invite(code=code)
                db.session.add(invite)
                db.session.commit()
                flash("Invite created.", "success")
    invites = Invite.query.all()
    users = User.query.order_by(User.username.asc()).all()
    
    # Get postponed fixtures count for admin dashboard
    postponed_count = Fixture.query.filter(Fixture.status.in_(EXCLUDED_FROM_CURRENT)).count()
    
    return render_template("admin.html", invites=invites, users=users, postponed_count=postponed_count)


@app.route("/admin/reset_password/<int:user_id>", methods=["POST"])
@login_required
def admin_reset_password(user_id: int):
    if not current_user.is_admin:
        abort(403)
    new_password = request.form.get("new_password") or ""
    user = User.query.get(user_id)
    if not user:
        flash("User not found.", "danger")
    elif not new_password:
        flash("New password must not be empty.", "danger")
    else:
        user.set_password(new_password)
        db.session.add(user)
        db.session.commit()
        flash(f"Password reset for {user.username}.", "success")
    return redirect(url_for("admin"))


@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
@login_required
def admin_delete_user(user_id: int):
    if not current_user.is_admin:
        abort(403)
    user = User.query.get(user_id)
    if not user:
        flash("User not found.", "danger")
    elif user.id == current_user.id:
        flash("You cannot delete your own account.", "danger")
    elif user.is_admin:
        flash("Cannot delete another admin user.", "danger")
    else:
        username = user.username
        Invite.query.filter_by(used_by_user_id=user.id).update({Invite.used_by_user_id: None})
        db.session.delete(user)
        db.session.commit()
        flash(f"User {username} deleted.", "success")
    return redirect(url_for("admin"))


@app.route("/admin/adjust_points/<int:user_id>", methods=["POST"])
@login_required
def admin_adjust_points(user_id: int):
    if not current_user.is_admin:
        abort(403)
    user = User.query.get(user_id)
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("admin"))

    try:
        bonus_points = int(request.form.get("bonus_points", 0))
        user.bonus_points = bonus_points
        db.session.commit()
        flash(f"Bonus points for {user.username} set to {bonus_points}.", "success")
    except ValueError:
        flash("Invalid points value.", "danger")

    return redirect(url_for("admin"))


@app.route("/admin/refresh", methods=["POST"])
@login_required
def admin_refresh():
    if not current_user.is_admin:
        abort(403)
    update_fixtures_adaptive(force=True)
    flash("Fixtures refreshed.", "success")
    return redirect(url_for("index"))


@app.route("/admin/results", methods=["GET"])
@login_required
def admin_results():
    if not current_user.is_admin:
        abort(403)

    # Competition selector — defaults to Serie A; falls back to SA if unknown.
    competition_code = request.args.get("competition") or "SA"
    if competition_code not in COMPETITIONS:
        competition_code = "SA"

    seasons = seasons_available(competition_code)
    if not seasons:
        return render_template(
            "admin_results.html",
            fixtures=[],
            seasons=[],
            season=None,
            matchday=None,
            matchdays=[],
            competition_code=competition_code,
            competitions=COMPETITIONS,
        )

    season = request.args.get("season") or seasons[-1]
    matchdays = matchdays_for(season, competition_code) if season else []
    matchday = request.args.get("matchday") or (matchdays[0] if matchdays else None)

    fixtures = []
    if season and matchday:
        fixtures = (
            Fixture.query
            .filter_by(
                competition_code=competition_code,
                season=season,
                matchday=matchday,
            )
            .order_by(Fixture.match_date.asc())
            .all()
        )
    return render_template(
        "admin_results.html",
        fixtures=fixtures,
        seasons=seasons,
        season=season,
        matchday=matchday,
        matchdays=matchdays,
        EXCLUDED_FROM_CURRENT=EXCLUDED_FROM_CURRENT,
        competition_code=competition_code,
        competitions=COMPETITIONS,
    )


@app.route("/admin/update_result/<int:fixture_id>", methods=["POST"])
@login_required
def admin_update_result(fixture_id: int):
    if not current_user.is_admin:
        abort(403)
    fixture = Fixture.query.get(fixture_id)
    if not fixture:
        flash("Fixture not found.", "danger")
        return redirect(url_for("admin_results"))
    
    def parse_score(s):
        try:
            return int(s) if s is not None and s != "" else None
        except Exception:
            return None
    
    home_score = parse_score(request.form.get("home_score"))
    away_score = parse_score(request.form.get("away_score"))
    
    if home_score != fixture.home_score or away_score != fixture.away_score:
        fixture.home_score = home_score
        fixture.away_score = away_score
        fixture.scores_manually_edited = True
        if home_score is not None and away_score is not None:
            fixture.status = 'FINISHED'
            fixture.status_manually_edited = True
        elif fixture.status == 'FINISHED':
            fixture.status = 'TIMED'
    
    # Handle status change
    new_status = request.form.get("status")
    if new_status and new_status in FIXTURE_STATUSES:
        if new_status != fixture.status:
            fixture.status = new_status
            fixture.status_manually_edited = True
    
    # Handle rescheduled date (for postponed fixtures)
    rescheduled_str = request.form.get("rescheduled_date")
    if rescheduled_str:
        try:
            dt = datetime.fromisoformat(rescheduled_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            fixture.rescheduled_date = dt
        except Exception:
            pass
    elif request.form.get("clear_rescheduled"):
        fixture.rescheduled_date = None
    
    # Handle admin notes
    admin_notes = request.form.get("admin_notes")
    if admin_notes is not None:
        fixture.admin_notes = admin_notes.strip() if admin_notes.strip() else None
    
    # Handle match_date update
    match_date_str = request.form.get("match_date")
    if match_date_str:
        try:
            dt = datetime.fromisoformat(match_date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if fixture.match_date != dt:
                fixture.match_date = dt
        except Exception:
            pass

    db.session.add(fixture)
    db.session.commit()
    evaluate_predictions()
    
    flash(f"Updated: {fixture.home_team} vs {fixture.away_team}.", "success")
    return redirect(url_for(
        "admin_results",
        competition=fixture.competition_code,
        season=fixture.season,
        matchday=fixture.matchday,
    ))


@app.route("/admin/postpone/<int:fixture_id>", methods=["POST"])
@login_required
def admin_postpone_fixture(fixture_id: int):
    """Quick action to mark a fixture as postponed."""
    if not current_user.is_admin:
        abort(403)
    fixture = Fixture.query.get(fixture_id)
    if not fixture:
        flash("Fixture not found.", "danger")
        return redirect(url_for("admin_results"))
    
    fixture.status = "POSTPONED"
    fixture.status_manually_edited = True
    
    # Optional: set rescheduled date if provided
    rescheduled_str = request.form.get("rescheduled_date")
    if rescheduled_str:
        try:
            dt = datetime.fromisoformat(rescheduled_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            fixture.rescheduled_date = dt
        except Exception:
            pass
    
    # Optional: add notes
    notes = request.form.get("notes")
    if notes:
        fixture.admin_notes = notes.strip()
    
    db.session.add(fixture)
    db.session.commit()
    
    flash(f"Marked as POSTPONED: {fixture.home_team} vs {fixture.away_team}.", "warning")
    return redirect(url_for(
        "admin_results",
        competition=fixture.competition_code,
        season=fixture.season,
        matchday=fixture.matchday,
    ))


@app.route("/admin/unpostpone/<int:fixture_id>", methods=["POST"])
@login_required
def admin_unpostpone_fixture(fixture_id: int):
    """Restore a postponed fixture to scheduled status."""
    if not current_user.is_admin:
        abort(403)
    fixture = Fixture.query.get(fixture_id)
    if not fixture:
        flash("Fixture not found.", "danger")
        return redirect(url_for("admin_results"))
    
    # If there's a rescheduled date, use it as the new match_date
    if fixture.rescheduled_date:
        fixture.match_date = fixture.rescheduled_date
        fixture.rescheduled_date = None
    
    fixture.status = "SCHEDULED"
    fixture.status_manually_edited = False  # Allow API to update again
    fixture.admin_notes = None
    
    db.session.add(fixture)
    db.session.commit()
    
    flash(f"Restored to SCHEDULED: {fixture.home_team} vs {fixture.away_team}.", "success")
    return redirect(url_for(
        "admin_results",
        competition=fixture.competition_code,
        season=fixture.season,
        matchday=fixture.matchday,
    ))


@app.route("/history/refresh", methods=["POST"])
@login_required
def history_refresh():
    season = request.form.get("season")
    matchday = request.form.get("matchday")
    update_fixtures_adaptive(force=True)
    flash("Fixtures refreshed.", "success")
    return redirect(url_for("history", season=season, matchday=matchday))


def prediction_coverage(season: str, matchday: str, competition_code: str = "SA"):
    """Return coverage statistics for each fixture in a round."""
    fixtures = (Fixture.query
                .filter_by(
                    competition_code=competition_code,
                    season=season,
                    matchday=matchday,
                )
                .order_by(Fixture.match_date.asc())
                .all())
    if not fixtures:
        return []

    players = (User.query
               .filter(User.is_admin == False)
               .order_by(User.username.asc())
               .all())
    player_ids = {u.id for u in players}
    id_to_user = {u.id: u for u in players}

    fixture_ids = [f.id for f in fixtures]

    pairs = (db.session.query(Prediction.user_id, Prediction.fixture_id)
             .filter(Prediction.fixture_id.in_(fixture_ids),
                     Prediction.user_id.in_(player_ids))
             .all())

    predicted_by_fixture = defaultdict(set)
    for uid, fid in pairs:
        predicted_by_fixture[fid].add(uid)

    total_players = len(players)
    rows = []
    for f in fixtures:
        submitted = predicted_by_fixture.get(f.id, set())
        missing_ids = sorted(player_ids - submitted)
        rows.append({
            "fixture": f,
            "submitted_count": len(submitted),
            "total_players": total_players,
            "missing_users": [id_to_user[i] for i in missing_ids],
        })
    return rows


@app.route("/admin/coverage", methods=["GET"])
@login_required
def admin_coverage():
    if not current_user.is_admin:
        abort(403)

    competition_code = request.args.get("competition") or "SA"
    if competition_code not in COMPETITIONS:
        competition_code = "SA"

    seasons = seasons_available(competition_code)
    current_season = current_season_from_db(competition_code) or (seasons[-1] if seasons else None)
    season = request.args.get("season") or current_season
    md_param = request.args.get("matchday")
    if md_param:
        md = md_param
    else:
        if season:
            md = current_home_matchday(season, competition_code) or (matchdays_for(season, competition_code) or [None])[-1]
        else:
            md = None

    matchdays = matchdays_for(season, competition_code) if season else []
    rows = prediction_coverage(season, md, competition_code) if season and md else []

    return render_template(
        "admin_coverage.html",
        seasons=seasons,
        season=season,
        matchdays=matchdays,
        matchday=md,
        rows=rows,
        competition_code=competition_code,
        competitions=COMPETITIONS,
    )


# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------

@app.cli.command("init-db")
def init_db_command() -> None:
    db.create_all()
    if not User.query.filter_by(is_admin=True).first():
        admin_user = User(username="admin", is_admin=True)
        admin_user.set_password("admin")
        db.session.add(admin_user)
        invite = Invite(code="demo-invite")
        db.session.add(invite)
        db.session.commit()
        print('Admin user created with username "admin" and password "admin".')
    else:
        print("Admin user already exists.")


@app.cli.command("add-postponed-columns")
def add_postponed_columns() -> None:
    """Migration: Add new columns for postponed fixture support."""
    from sqlalchemy import inspect, text
    inspector = inspect(db.engine)
    columns = [c['name'] for c in inspector.get_columns('fixtures')]
    
    with db.engine.connect() as conn:
        if 'scores_manually_edited' not in columns:
            conn.execute(text('ALTER TABLE fixtures ADD COLUMN scores_manually_edited BOOLEAN DEFAULT FALSE'))
            print("Added scores_manually_edited column")
        if 'status_manually_edited' not in columns:
            conn.execute(text('ALTER TABLE fixtures ADD COLUMN status_manually_edited BOOLEAN DEFAULT FALSE'))
            print("Added status_manually_edited column")
        if 'rescheduled_date' not in columns:
            conn.execute(text('ALTER TABLE fixtures ADD COLUMN rescheduled_date DATETIME'))
            print("Added rescheduled_date column")
        if 'admin_notes' not in columns:
            conn.execute(text('ALTER TABLE fixtures ADD COLUMN admin_notes VARCHAR'))
            print("Added admin_notes column")
        conn.commit()
    print("Migration complete.")


# Ensure DB tables exist on startup
with app.app_context():
    db.create_all()

# Automatic migration: add any missing columns to existing tables
# This runs on every startup to ensure schema is up-to-date
with app.app_context():
    from sqlalchemy import inspect, text
    try:
        inspector = inspect(db.engine)
        if 'fixtures' in inspector.get_table_names():
            columns = [c['name'] for c in inspector.get_columns('fixtures')]
            
            with db.engine.connect() as conn:
                # Add scores_manually_edited if missing (original feature)
                if 'scores_manually_edited' not in columns:
                    conn.execute(text('ALTER TABLE fixtures ADD COLUMN scores_manually_edited BOOLEAN DEFAULT FALSE'))
                    print("[MIGRATION] Added scores_manually_edited column")
                
                # Add status_manually_edited if missing (postponed feature)
                if 'status_manually_edited' not in columns:
                    conn.execute(text('ALTER TABLE fixtures ADD COLUMN status_manually_edited BOOLEAN DEFAULT FALSE'))
                    print("[MIGRATION] Added status_manually_edited column")
                
                # Add rescheduled_date if missing (postponed feature)
                if 'rescheduled_date' not in columns:
                    conn.execute(text('ALTER TABLE fixtures ADD COLUMN rescheduled_date TIMESTAMP'))
                    print("[MIGRATION] Added rescheduled_date column")
                
                # Add admin_notes if missing (postponed feature)
                if 'admin_notes' not in columns:
                    conn.execute(text('ALTER TABLE fixtures ADD COLUMN admin_notes VARCHAR'))
                    print("[MIGRATION] Added admin_notes column")

                # Add competition_code if missing (multi-competition support)
                # Existing rows are backfilled to 'SA' (Serie A) via the DEFAULT.
                if 'competition_code' not in columns:
                    conn.execute(text(
                        "ALTER TABLE fixtures ADD COLUMN competition_code VARCHAR "
                        "NOT NULL DEFAULT 'SA'"
                    ))
                    print("[MIGRATION] Added competition_code column (defaulted to 'SA')")

                conn.commit()

        # Migration for users table
        if 'users' in inspector.get_table_names():
            user_columns = [c['name'] for c in inspector.get_columns('users')]

            with db.engine.connect() as conn:
                # Add bonus_points if missing
                if 'bonus_points' not in user_columns:
                    conn.execute(text('ALTER TABLE users ADD COLUMN bonus_points INTEGER DEFAULT 0'))
                    print("[MIGRATION] Added bonus_points column to users")

                conn.commit()
    except Exception as e:
        print(f"[MIGRATION] Warning: Could not check/add columns: {e}")

# Optional bootstrap via env variables
with app.app_context():
    flag = (os.getenv("ADMIN_FORCE_RESET", "0") or "").strip()
    uname = (os.getenv("ADMIN_USERNAME", "admin") or "").strip()
    pwd = os.getenv("ADMIN_PASSWORD")
    invite_code = (os.getenv("INITIAL_INVITE_CODE", os.getenv("INVITE_CODE", "demo-invite")) or "").strip()

    print(f"[BOOTSTRAP] Reset flag={flag} username={uname}")
    if flag == "1" and uname and pwd:
        u = User.query.filter_by(username=uname).first()
        if not u:
            u = User(username=uname, is_admin=True)
            db.session.add(u)
        u.is_admin = True
        u.set_password(pwd)
        db.session.commit()
        print(f"[BOOTSTRAP] Admin reset for {uname}")

        if invite_code and not Invite.query.filter_by(code=invite_code).first():
            db.session.add(Invite(code=invite_code))
            db.session.commit()
            print(f"[BOOTSTRAP] Invite code ensured: {invite_code}")


@app.cli.command("fix-times-utc")
def fix_times_utc():
    from sqlalchemy import select
    changed = 0
    for f in db.session.execute(select(Fixture)).scalars():
        md = f.match_date
        if md is None:
            continue
        if md.tzinfo is None:
            md = md.replace(tzinfo=ZoneInfo("Europe/Rome")).astimezone(timezone.utc)
            f.match_date = md
            changed += 1
        else:
            try:
                if getattr(md.tzinfo, "key", None) == "America/New_York":
                    f.match_date = md.astimezone(timezone.utc)
                    changed += 1
            except Exception:
                f.match_date = md.astimezone(timezone.utc)
                changed += 1
    db.session.commit()
    print(f"Normalized {changed} fixture times to UTC.")


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)
