"""
Serie A Predictor web application (cleaned + live-friendly).

Key changes from your last working baseline:
- Keep live/just-started games visible in the main list by widening the time window.
- Provide a Jinja filter `utc_iso` to emit ISO 8601 UTC for client-side local time conversion.
- DO NOT reference any non-existent attributes (like kickoff_utc); always use Fixture.match_date.
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

# -----------------------------------------------------------------------------
# App / DB config
# -----------------------------------------------------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

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

    predictions = db.relationship("Prediction", back_populates="user", cascade="all, delete-orphan")

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    @property
    def points(self) -> int:
        return sum(p.points_awarded or 0 for p in self.predictions)


class Invite(db.Model):
    __tablename__ = "invites"
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String, unique=True, nullable=False)
    used_by_user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)


class Fixture(db.Model):
    __tablename__ = "fixtures"
    id = db.Column(db.Integer, primary_key=True)
    match_id = db.Column(db.String, unique=True, nullable=False)
    match_date = db.Column(db.DateTime, nullable=False)  # stored in UTC (aware or naive UTC)
    home_team = db.Column(db.String, nullable=False)
    away_team = db.Column(db.String, nullable=False)
    season = db.Column(db.String, nullable=False)
    matchday = db.Column(db.String, nullable=True)
    status = db.Column(db.String, default="SCHEDULED")  # SCHEDULED/TIMED/IN_PLAY/PAUSED/FINISHED
    home_score = db.Column(db.Integer, nullable=True)
    away_score = db.Column(db.Integer, nullable=True)

    predictions = db.relationship("Prediction", back_populates="fixture", cascade="all, delete-orphan")

    def outcome_code(self) -> str | None:
        if self.home_score is None or self.away_score is None:
            return None
        if self.home_score > self.away_score:
            return "1"
        if self.home_score < self.away_score:
            return "2"
        return "X"

    def is_open_for_prediction(self) -> bool:
        now_utc = datetime.now(timezone.utc)
        md = self.match_date
        if md.tzinfo is None:
            md = md.replace(tzinfo=timezone.utc)
        return now_utc < md

    def display_status(self) -> str:
        """
        Human-friendly status with time-based fallback so we don't stay on 'TIMED'.
        """
        # Normalize kickoff to UTC
        md = self.match_date
        if md.tzinfo is None:
            md = md.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)

        # Trust explicit statuses first
        if self.status in ('IN_PLAY', 'PAUSED'):
            return 'LIVE'
        if self.status == 'FINISHED':
            return 'FT'

        # If API is stale:
        if now < md:
            return 'TIMED'                            # before kickoff
        if now <= md + timedelta(hours=2, minutes=30):
            return 'LIVE'                             # around match window
        if (self.home_score is not None and self.away_score is not None) or now > md + timedelta(hours=3):
            return 'FT'                               # long after KO or score present

        return 'LIVE'

class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    fixture_id = db.Column(db.Integer, db.ForeignKey("fixtures.id"))
    selection = db.Column(db.String, nullable=False)  # '1', 'X', or '2'
    points_awarded = db.Column(db.Integer, nullable=True)  # 1 for correct, 0 for incorrect, None until evaluated
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
        if status not in ("SCHEDULED", "TIMED", "IN_PLAY", "PAUSED", "FINISHED"):
            continue
        utc_date_str = match["utcDate"]  # e.g. '2025-08-23T18:45:00Z'
        utc_dt = datetime.fromisoformat(utc_date_str.replace("Z", "+00:00"))
        # Scores: if live or finished, football-data provides live or full-time in score
        score = match.get("score", {}) or {}
        ft = score.get("fullTime") or {}
        home_ft = ft.get("home")
        away_ft = ft.get("away")

        # For IN_PLAY / PAUSED they often populate "fullTime" as None but "halfTime"/"duration"/"winner" etc.
        # We’ll also check "regularTime" if available:
        if home_ft is None or away_ft is None:
            reg = score.get("regularTime") or {}
            home_ft = reg.get("home") if home_ft is None else home_ft
            away_ft = reg.get("away") if away_ft is None else away_ft

        fixtures.append({
            "match_id": str(match["id"]),
            "match_date": utc_dt,  # KEEP IN UTC
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
    # Your repo path (adjust if your fallback lives elsewhere)
    fallback_path = Path(__file__).resolve().parent / "data" / "seriea_2024_25.json"
    if not fallback_path.exists():
        return []

    with open(fallback_path, "r", encoding="utf-8") as f:
        season_data = json.load(f)

    fixtures: list[dict] = []
    for match in season_data.get("matches", []):
        score = match.get("score", {})
        ft = score.get("ft")
        if not ft:
            date_str = match["date"]
            time_str = match.get("time", "18:00")
            # Europe/Rome -> UTC
            dt_naive = datetime.fromisoformat(f"{date_str}T{time_str}")
            dt_rome = dt_naive.replace(tzinfo=ZoneInfo("Europe/Rome"))
            utc_dt = dt_rome.astimezone(timezone.utc)
            fixtures.append({
                "match_id": f"{date_str}-{match['team1']}-{match['team2']}",
                "match_date": utc_dt,
                "home_team": match["team1"],
                "away_team": match["team2"],
                "season": season_data.get("name", "2024/25"),
                "matchday": match.get("round"),
                "status": "SCHEDULED",
                "home_score": None,
                "away_score": None,
            })
    return fixtures


def update_fixtures() -> None:
    """
    Sync local fixtures with API (if key present) or fallback file.
    Insert new fixtures; update status/scores on existing ones.
    Then evaluate predictions for finished matches.
    """
    fixtures_from_api = fetch_fixtures_from_api()
    fixtures_to_use = fixtures_from_api if fixtures_from_api else fetch_fixtures_from_fallback()

    for fi in fixtures_to_use:
                existing = Fixture.query.filter_by(match_id=fi['match_id']).first()
                if existing:
                    updated = False
                    # Update status and scores on the existing fixture.
                    # Prioritise final scores: if both home and away scores
                    # are present (non-None) then this fixture is finished,
                    # regardless of what the API status says.  Otherwise
                    # update the status only if the API provides a new status.
                    home_sc = fi['home_score']
                    away_sc = fi['away_score']
                    if home_sc is not None and home_sc != existing.home_score:
                        existing.home_score = home_sc; updated = True
                    if away_sc is not None and away_sc != existing.away_score:
                        existing.away_score = away_sc; updated = True

                    # If both scores are present, force the status to FINISHED
                    if home_sc is not None and away_sc is not None:
                        if existing.status != 'FINISHED':
                            existing.status = 'FINISHED'; updated = True
                    else:
                        # Otherwise rely on API status updates
                        if fi['status'] != existing.status:
                            existing.status = fi['status']; updated = True

                    if updated:
                        db.session.add(existing)
                else:
                    # --- NEW: try to reconcile a legacy/fallback row by teams + kickoff time ---
                    from sqlalchemy import and_
                    dt = fi['match_date']
                    # 12h window to be tolerant of timezone differences
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
                    if legacy:
                        # Adopt the official API id and update fields in-place
                        legacy.match_id = fi['match_id']
                        legacy.status = fi['status']
                        legacy.home_score = fi['home_score']
                        legacy.away_score = fi['away_score']
                        if abs((legacy.match_date - dt).total_seconds()) > 60:
                            legacy.match_date = dt  # normalize to the API UTC time
                        db.session.add(legacy)
                    else:
                        # No legacy row; insert fresh.  If both scores are
                        # present then mark the fixture as finished; otherwise
                        # use the API status as-is.
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
                        ))
        
    db.session.commit()
    evaluate_predictions()  # finalize finished games


# --- Adaptive fetch throttle (poll more often around live games) ---

FETCH_STATE = {"last_run": None, "last_interval": None}

def _adaptive_min_interval() -> timedelta:
    now_utc = datetime.now(timezone.utc)

    # Poll every 60s when there are live games or within 2 hours of kickoff
    live = Fixture.query.filter(Fixture.status.in_(("IN_PLAY", "PAUSED"))).count()
    if live > 0:
        return timedelta(seconds=60)

    soon = (
        Fixture.query
        .filter(Fixture.match_date >= now_utc, Fixture.match_date <= now_utc + timedelta(hours=2))
        .count()
    )
    if soon > 0:
        return timedelta(seconds=60)

    # On match days, lighter polling
    today_end_utc = now_utc.replace(hour=23, minute=59, second=59, microsecond=999999)
    today = (
        Fixture.query
        .filter(Fixture.match_date >= now_utc, Fixture.match_date <= today_end_utc)
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

def upcoming_fixtures() -> list[Fixture]:
    """
    Return fixtures to show on the main page.

    IMPORTANT: include *live/just-started* games so they don't "disappear".
    We include fixtures from 6 hours *before* now up to 7 days ahead,
    and then also include the rest of the first upcoming matchday for context.
    """
    now_utc = datetime.now(timezone.utc)
    window_start = now_utc - timedelta(hours=6)    # keep games that just kicked off
    window_end = now_utc + timedelta(days=7)

    base = (
        Fixture.query.filter(
            Fixture.match_date >= window_start,
            Fixture.match_date <= window_end
        ).all()
    )

    # Pull entire first upcoming matchday (by kickoff) if one exists,
    # so the page shows the whole round together.
    first_upcoming = (
        Fixture.query.filter(
            Fixture.status.in_(("SCHEDULED", "TIMED")),
            Fixture.match_date >= now_utc
        )
        .order_by(Fixture.match_date.asc())
        .first()
    )

    week1 = []
    if first_upcoming and first_upcoming.matchday:
        week1 = Fixture.query.filter(
            Fixture.matchday == first_upcoming.matchday
        ).all()

    merged = {f.id: f for f in base}
    for f in week1:
        merged[f.id] = f
    return sorted(merged.values(), key=lambda f: f.match_date)

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
    """
    Build the prediction matrix for a list of Fixture objects.

    Returns:
      users: list of (user_id, username) sorted alphabetically by username
      matrix: dict keyed by (fixture_key, user_id) -> prediction ('1','X','2')
        where fixture_key is the stable match_id if available, falling back to
        the internal fixture.id.  This allows predictions attached to older
        fixture rows (with different ids but same match_id) to be matched
        against the current fixture list.
      show_flags: dict[fixture.id] -> bool indicating whether to reveal picks
        for that fixture.  Picks are revealed if the fixture is in play,
        paused, finished, or if kickoff time has passed.
    """
    if not fixtures:
        return [], {}, {}

    fix_ids = [f.id for f in fixtures]
    # Join Users, Predictions and Fixtures to obtain match_id for each prediction.
    rows = (
        db.session.query(User.id, User.username, Prediction.fixture_id, Prediction.selection, Fixture.match_id)
        .join(Prediction, Prediction.user_id == User.id)
        .join(Fixture, Fixture.id == Prediction.fixture_id)
        .filter(Prediction.fixture_id.in_(fix_ids))
        .all()
    )

    # Build a unique sorted list of users who made predictions for these fixtures.
    user_set = {}
    for uid, uname, _, _, _ in rows:
        user_set[uid] = uname
    users = sorted(user_set.items(), key=lambda t: t[1].lower())

    # Build the matrix keyed by both (fixture_id, user_id) and (match_id, user_id).
    # We prefer match_id when present (it stays constant across re-imports),
    # but also store predictions under the internal fixture.id to support
    # legacy rows or templates that still index by id.  Without this dual
    # mapping, re-importing fixtures can cause historical picks to vanish.
    matrix: dict[tuple[str|int,int], str] = {}
    for uid, uname, fid, sel, match_id in rows:
        # Always store under the internal fixture.id
        matrix[(fid, uid)] = sel
        # Additionally, store under the stable match_id when available
        if match_id:
            matrix[(match_id, uid)] = sel

    # Compute reveal flags using the same logic as before: reveal if the
    # fixture is LIVE/PAUSED/FINISHED or the current time has reached kickoff.
    utc_now = datetime.now(timezone.utc)
    show_flags = {}
    for f in fixtures:
        kickoff = f.match_date.replace(tzinfo=timezone.utc) if f.match_date.tzinfo is None else f.match_date
        show_flags[f.id] = (f.status in ("IN_PLAY", "PAUSED", "FINISHED")) or (utc_now >= kickoff)

    return users, matrix, show_flags

def evaluate_predictions() -> None:
    """
    Finalize points for games that ended.
    """
    finished_fixtures = Fixture.query.filter(Fixture.status == "FINISHED").all()
    for fixture in finished_fixtures:
        outcome = fixture.outcome_code()
        for prediction in fixture.predictions:
            if prediction.points_awarded is None:
                prediction.points_awarded = 1 if prediction.selection == outcome else 0
                db.session.add(prediction)
    db.session.commit()

# Season/matchday helpers (unchanged)
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
    rows = (
        db.session.query(Fixture.matchday)
        .filter(Fixture.season == season, Fixture.status == "FINISHED")
        .distinct()
        .all()
    )
    days = [r[0] for r in rows if r[0]]
    if not days:
        return None
    try:
        return str(max(int(d) for d in days))
    except Exception:
        return sorted(set(days))[-1]

def current_home_matchday(season: str) -> str | None:
    """
    Determine which matchday to present on the home page.  The logic
    prioritises the earliest matchday that has any fixture not yet
    finished, so that users remain on the current round until every
    game is completed.  If a later round has begun (i.e. there are
    scheduled or timed fixtures for the next round and all earlier
    rounds are finished), then that round will be shown.  Only when
    every round in the season is completed will the latest
    matchday be returned.

    Returns the matchday as a string, or None if the season has no
    matchdays.
    """
    if not season:
        return None

    # Gather all matchdays for the season, attempting numeric sort when
    # possible.  This ensures matchday "10" follows "9" rather than
    # lexicographically after "1".
    days = matchdays_for(season)
    if not days:
        return None
    try:
        sorted_days = [str(n) for n in sorted({int(d) for d in days})]
    except Exception:
        sorted_days = sorted(set(days), key=lambda s: (len(s), s))

    # Iterate through matchdays in ascending order and return the first
    # one that still has any fixture not finished.  A fixture is
    # considered not finished if its status is anything other than
    # 'FINISHED'.  This keeps users on the current round until it has
    # fully concluded.
    for md in sorted_days:
        remaining = (
            db.session.query(Fixture.id)
            .filter(
                Fixture.season == season,
                Fixture.matchday == md,
                Fixture.status != 'FINISHED'
            )
            .first()
        )
        if remaining is not None:
            return md

    # If all matchdays are complete (every fixture is finished), fall
    # back to the latest completed matchday.  This maintains backward
    # compatibility with existing behaviour at end of season.
    return latest_completed_matchday(season)

def weekly_user_points(season: str, matchday: str):
    rows = (
        db.session.query(
            User.id,
            User.username,
            func.coalesce(func.sum(Prediction.points_awarded), 0).label("pts"),
        )
        .join(Prediction, Prediction.user_id == User.id)
        .join(Fixture, Fixture.id == Prediction.fixture_id)
        .filter(Fixture.season == season, Fixture.matchday == str(matchday))
        .group_by(User.id, User.username)
        .all()
    )
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
    rows = (
        db.session.query(
            User.username,
            func.coalesce(func.sum(Prediction.points_awarded), 0).label("pts"),
        )
        .join(Prediction, Prediction.user_id == User.id)
        .join(Fixture, Fixture.id == Prediction.fixture_id)
        .filter(Fixture.season == season)
        .group_by(User.username)
        .all()
    )
    return sorted(
        [{"username": r[0], "points": int(r[1])} for r in rows],
        key=lambda x: (-x["points"], x["username"].lower())
    )

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.route('/')
@login_required
def index():
    update_fixtures_adaptive()

    # Decide which season & matchday to show
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
        )

    # Pull fixtures for THIS matchday only
    fixtures = (
        Fixture.query
        .filter(Fixture.season == season, Fixture.matchday == str(md))
        .order_by(Fixture.match_date.asc())
        .all()
    )

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
        matchday=md,  # <-- tell the template which matchday we’re on
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
    # Save only for fixtures still open
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

    seasons = seasons_available()
    current_season = current_season_from_db() or (seasons[-1] if seasons else None)

    raw_scope = request.args.get("scope")
    scope = (raw_scope or "season").lower()
    season = request.args.get("season")
    matchday = request.args.get("matchday", type=int)

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
    """
    Display historical fixtures and predictions for a given season and matchday.
    This view allows users to browse finished matchdays.  Predictions are
    always revealed regardless of kickoff time since the matches have
    concluded.  If no matchday is specified, the latest completed
    matchday for the chosen season (or current season) is used.
    """
    update_fixtures_adaptive()

    # Available seasons and the current one.  Fall back to the last season
    # in the list if none is marked current.
    seasons = seasons_available()
    current_season = current_season_from_db() or (seasons[-1] if seasons else None)

    # Season and matchday come from query parameters.  If missing, use
    # sensible defaults.  ``request.args.get`` returns ``None`` when not
    # present; convert matchday to string to avoid type issues.
    season = request.args.get("season") or current_season
    matchday = request.args.get("matchday") or None

    # Determine the matchday to display.  If unspecified, choose the
    # latest completed matchday; if none are completed yet, default to
    # the first available matchday for the season (if any).
    if season:
        if not matchday:
            matchday = latest_completed_matchday(season)
            if not matchday:
                days = matchdays_for(season)
                matchday = days[0] if days else None
    else:
        matchday = None

    # Fetch fixtures for the requested season/matchday.  An empty list
    # yields an empty table rather than an error.
    fixtures = []
    if season and matchday:
        fixtures = (
            Fixture.query
            .filter(Fixture.season == season, Fixture.matchday == str(matchday))
            .order_by(Fixture.match_date.asc())
            .all()
        )

    # Build the prediction matrix.  ``prediction_matrix`` returns a list of
    # users, a prediction mapping and a set of flags indicating when
    # predictions should be revealed.  For the history view we rely on
    # these flags so that picks remain hidden until kickoff.  This
    # prevents revealing predictions for future fixtures if a user
    # navigates to a not-yet-started matchday.
    users_cols, pred_matrix, _show_flags = prediction_matrix(fixtures)
    # Override show flags for the history view.  We want to reveal
    # predictions once a fixture has actually begun or concluded.  In
    # practice some APIs leave the status as TIMED/SCHEDULED even after
    # kickoff or full‑time.  To avoid hiding picks in those cases, also
    # reveal predictions when a final score is present.  This means
    # users will see predictions for any fixture with a recorded result,
    # or with a live/paused/finished status, but not for future games.
    show_flags = {}
    for f in fixtures:
        reveal_by_status = f.status in ("IN_PLAY", "PAUSED", "FINISHED")
        reveal_by_score = (f.home_score is not None and f.away_score is not None)
        show_flags[f.id] = reveal_by_status or reveal_by_score

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
            login_user(user)
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
    return render_template("admin.html", invites=invites)

@app.route("/admin/refresh", methods=["POST"])
@login_required
def admin_refresh():
    if not current_user.is_admin:
        abort(403)
    update_fixtures_adaptive(force=True)
    flash("Fixtures refreshed.", "success")
    return redirect(url_for("index"))

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

# Ensure DB tables exist on startup
with app.app_context():
    db.create_all()

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
