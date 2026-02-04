"""
Hardamessa Lotto web application (cleaned + live-friendly).

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
    Flask, abort, flash, redirect, render_template, request, url_for
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

def fetch_fixtures_from_api() -> list[dict]:
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        return []

    today = datetime.now().date()
    season_start_year = today.year if today.month >= 7 else today.year - 1
    season_str = f"{season_start_year}-{(season_start_year + 1) % 100:02d}"

    url = "https://api.football-data.org/v4/competitions/SA/matches"
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

        fixtures.append({
            "match_id": str(match["id"]),
            "match_date": utc_dt,
            "home_team": match["homeTeam"]["name"],
            "away_team": match["awayTeam"]["name"],
            "season": season_str,
            "matchday": str(match.get("matchday")),
            "status": status,
            "home_score": home_ft if home_ft is not None else None,
            "away_score": away_ft if away_ft is not None else None,
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
        })
    return fixtures


def update_fixtures() -> None:
    """
    Sync local fixtures with API (if key present) or fallback file.
    
    IMPORTANT: 
    - Skip updates for fixtures where status_manually_edited is True
    - Only skip score updates when scores_manually_edited is True
    """
    try:
        db.create_all()
    except Exception:
        pass

    fixtures_from_api = fetch_fixtures_from_api()
    fixtures_to_use = fixtures_from_api if fixtures_from_api else fetch_fixtures_from_fallback()

    for fi in fixtures_to_use:
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
            legacy = (
                Fixture.query
                .filter(
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


def get_postponed_fixtures(season: str = None) -> list[Fixture]:
    """
    Get all postponed/cancelled/suspended fixtures, optionally filtered by season.
    """
    query = Fixture.query.filter(Fixture.status.in_(EXCLUDED_FROM_CURRENT))
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
def seasons_available() -> list[str]:
    rows = db.session.query(Fixture.season).distinct().all()
    return sorted([r[0] for r in rows])


def matchdays_for(season: str) -> list[str]:
    rows = (
        db.session.query(Fixture.matchday)
        .filter(Fixture.season == season)
        .distinct()
        .all()
    )
    days = [r[0] for r in rows if r[0]]
    try:
        return [str(x) for x in sorted({int(d) for d in days})]
    except Exception:
        return sorted(set(days))


def latest_completed_matchday(season: str) -> str | None:
    """Find the latest matchday where all non-postponed fixtures have scores."""
    if not season:
        return None
    
    days = matchdays_for(season)
    if not days:
        return None
    
    try:
        sorted_days = [str(n) for n in sorted({int(d) for d in days}, reverse=True)]
    except Exception:
        sorted_days = sorted(set(days), key=lambda s: (len(s), s), reverse=True)
    
    for md in sorted_days:
        # Get non-postponed fixtures for this matchday
        fixtures = Fixture.query.filter(
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


def current_home_matchday(season: str) -> str | None:
    """
    Determine which matchday to present on the home page.
    
    Prioritises the earliest matchday that has any non-postponed fixture
    without final results.
    """
    if not season:
        return None

    days = matchdays_for(season)
    if not days:
        return None
        
    try:
        sorted_days = [str(n) for n in sorted({int(d) for d in days})]
    except Exception:
        sorted_days = sorted(set(days), key=lambda s: (len(s), s))

    for md in sorted_days:
        # Check for incomplete non-postponed fixtures
        incomplete = (
            db.session.query(Fixture.id)
            .filter(
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

    return latest_completed_matchday(season)


def weekly_user_points(season: str, matchday: str):
    """Return (user_id, username, points) for the given season and matchday."""
    predictions = (
        db.session.query(Prediction, Fixture, User)
        .join(Fixture, Fixture.id == Prediction.fixture_id)
        .join(User, User.id == Prediction.user_id)
        .filter(Fixture.season == season, Fixture.matchday == str(matchday))
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


def current_season_from_db() -> str | None:
    row = db.session.query(Fixture.season).order_by(Fixture.season.desc()).first()
    return row[0] if row else None


def all_matchdays_for_season(season: str) -> list[str]:
    rows = db.session.query(distinct(Fixture.matchday)).filter(Fixture.season == season).all()
    mds = [r[0] for r in rows if r[0] is not None]
    try:
        return [str(n) for n in sorted({int(x) for x in mds})]
    except Exception:
        return sorted(set(mds), key=lambda s: (len(s), s))


def classify_matchdays(season: str):
    now_utc = datetime.now(timezone.utc)
    md_status = {}
    for md in all_matchdays_for_season(season):
        qs = Fixture.query.filter_by(season=season, matchday=md).all()
        statuses = {f.status for f in qs}
        if any(s in ("IN_PLAY","PAUSED") for s in statuses):
            md_status[md] = "live"
        elif statuses and statuses.issubset({"FINISHED"}):
            md_status[md] = "finished"
        elif any(s in ("SCHEDULED","TIMED") for s in statuses):
            future = Fixture.query.filter(
                Fixture.season==season,
                Fixture.matchday==md,
                Fixture.match_date>=now_utc
            ).count()
            md_status[md] = "upcoming" if future else "finished"
        else:
            md_status[md] = "other"

    def _order(lst):
        try: return [str(n) for n in sorted({int(x) for x in lst})]
        except: return sorted(set(lst), key=lambda s: (len(s), s))

    finished = _order([m for m,s in md_status.items() if s=="finished"])
    live = _order([m for m,s in md_status.items() if s=="live"])
    upcoming = _order([m for m,s in md_status.items() if s=="upcoming"])
    other = _order([m for m,s in md_status.items() if s=="other"])
    return finished, live, upcoming, other


def season_user_points(season: str):
    """Return list of dicts {username: points} for the specified season."""
    predictions = (
        db.session.query(Prediction, Fixture, User)
        .join(Fixture, Fixture.id == Prediction.fixture_id)
        .join(User, User.id == Prediction.user_id)
        .filter(Fixture.season == season)
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


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.route('/')
@login_required
def index():
    update_fixtures_adaptive()

    season = current_season_from_db()
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

    md = current_home_matchday(season)
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
            Fixture.season == season,
            Fixture.matchday == str(md),
            ~Fixture.status.in_(EXCLUDED_FROM_CURRENT)
        )
        .order_by(Fixture.match_date.asc())
        .all()
    )

    # Count postponed fixtures for the badge
    postponed_count = Fixture.query.filter(
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


@app.route('/postponed')
@login_required
def postponed_fixtures_view():
    """View all postponed/cancelled/suspended fixtures."""
    season = request.args.get('season') or current_season_from_db()
    seasons = seasons_available()
    
    postponed = get_postponed_fixtures(season)
    
    return render_template(
        'postponed.html',
        fixtures=postponed,
        season=season,
        seasons=seasons,
    )


@app.route("/predict/<int:fixture_id>", methods=["POST"])
@login_required
def predict(fixture_id: int):
    fixture = db.session.get(Fixture, fixture_id)
    if not fixture:
        abort(404)
    if not fixture.is_open_for_prediction():
        flash("Predictions are locked for this fixture.", "warning")
        return redirect(url_for("index"))

    selection = request.form.get("selection")
    if selection not in ("1", "X", "2"):
        flash("Invalid prediction.", "danger")
        return redirect(url_for("index"))

    prediction = Prediction.query.filter_by(user_id=current_user.id, fixture_id=fixture_id).first()
    if prediction:
        prediction.selection = selection
        flash("Prediction updated.", "success")
    else:
        prediction = Prediction(user_id=current_user.id, fixture_id=fixture_id, selection=selection)
        db.session.add(prediction)
        flash("Prediction submitted.", "success")

    db.session.commit()
    return redirect(url_for("index"))


@app.route("/save_all_predictions", methods=["POST"])
@login_required
def save_all_predictions():
    all_fixtures = Fixture.query.all()
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
    return redirect(url_for("index"))


@app.route("/leaderboard")
@login_required
def leaderboard():
    update_fixtures_adaptive()
    evaluate_predictions()

    seasons = seasons_available()
    current_season = current_season_from_db() or (seasons[-1] if seasons else None)

    raw_scope = request.args.get("scope")
    scope = (raw_scope or "season").lower()
    season = request.args.get("season")
    matchday = request.args.get("matchday")

    if raw_scope is None and current_season:
        return redirect(url_for("leaderboard", scope="season", season=current_season))

    if scope == "week":
        if not season:
            season = current_season
        days = matchdays_for(season)
        if not matchday:
            matchday = latest_completed_matchday(season) or (days[-1] if days else None)
        rows = weekly_user_points(season, matchday) if matchday else []
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
        users_sorted = season_user_points(season)
        return render_template(
            "leaderboard.html",
            users=users_sorted,
            scope="season",
            seasons=seasons,
            season=season,
            matchdays=matchdays_for(season) if season else [],
            matchday=None
        )

    users = User.query.all()
    users_sorted = sorted(users, key=lambda u: (-u.points, u.username.lower()))
    return render_template(
        "leaderboard.html",
        users=users_sorted,
        scope="overall",
        seasons=seasons,
        season=season or current_season,
        matchdays=matchdays_for(season or current_season) if (season or current_season) else [],
        matchday=matchday
    )


@app.route("/history")
@login_required
def history():
    update_fixtures_adaptive()

    seasons = seasons_available()
    current_season = current_season_from_db() or (seasons[-1] if seasons else None)

    season = request.args.get("season") or current_season
    matchday = request.args.get("matchday") or None

    if season:
        if not matchday:
            matchday = latest_completed_matchday(season)
            if not matchday:
                days = matchdays_for(season)
                matchday = days[0] if days else None
    else:
        matchday = None

    fixtures = []
    if season and matchday:
        fixtures = (
            Fixture.query
            .filter(Fixture.season == season, Fixture.matchday == str(matchday))
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
        matchdays=matchdays_for(season) if season else [],
        matchday=matchday
    )


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
    seasons = seasons_available()
    if not seasons:
        return render_template("admin_results.html", fixtures=[], seasons=[], season=None, matchday=None, matchdays=[])
    season = request.args.get("season") or seasons[-1]
    matchdays = matchdays_for(season) if season else []
    matchday = request.args.get("matchday") or (matchdays[0] if matchdays else None)
    fixtures = []
    if season and matchday:
        fixtures = (
            Fixture.query
            .filter_by(season=season, matchday=matchday)
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
    return redirect(url_for("admin_results", season=fixture.season, matchday=fixture.matchday))


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
    return redirect(url_for("admin_results", season=fixture.season, matchday=fixture.matchday))


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
    return redirect(url_for("admin_results", season=fixture.season, matchday=fixture.matchday))


@app.route("/history/refresh", methods=["POST"])
@login_required
def history_refresh():
    season = request.form.get("season")
    matchday = request.form.get("matchday")
    update_fixtures_adaptive(force=True)
    flash("Fixtures refreshed.", "success")
    return redirect(url_for("history", season=season, matchday=matchday))


def prediction_coverage(season: str, matchday: str):
    """Return coverage statistics for each fixture in a round."""
    fixtures = (Fixture.query
                .filter_by(season=season, matchday=matchday)
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
    seasons = seasons_available()
    current_season = current_season_from_db() or (seasons[-1] if seasons else None)
    season = request.args.get("season") or current_season
    md_param = request.args.get("matchday")
    if md_param:
        md = md_param
    else:
        if season:
            md = current_home_matchday(season) or (matchdays_for(season) or [None])[-1]
        else:
            md = None

    matchdays = matchdays_for(season) if season else []
    rows = prediction_coverage(season, md) if season and md else []

    return render_template(
        "admin_coverage.html",
        seasons=seasons,
        season=season,
        matchdays=matchdays,
        matchday=md,
        rows=rows,
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
