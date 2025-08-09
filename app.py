"""
Serie A Predictor web application.

Public signup, predictions (1/X/2) per fixture, live/finished scoring,
leaderboard, and simple admin (users: edit/reset/delete). Fixtures fetched
from football-data.org when API key is set; otherwise a local fallback JSON.

Times are stored in UTC. Templates can render local time via JS.
"""

import os
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from secrets import token_urlsafe
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

import requests
from sqlalchemy import func

from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

# Prefer Render Postgres if present, else local SQLite
raw_db_url = os.environ.get("DATABASE_URL", "sqlite:///serie_a.db")

# Render sometimes supplies "postgres://"
if raw_db_url.startswith("postgres://"):
    raw_db_url = raw_db_url.replace("postgres://", "postgresql+psycopg2://", 1)

# Ensure sslmode=require for Postgres if not already present
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

# Allow templates to access UTC sentinel if needed
app.jinja_env.globals["utc"] = timezone.utc


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class User(db.Model, UserMixin):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String, unique=True, nullable=False)
    password_hash = db.Column(db.String, nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

    predictions = db.relationship(
        "Prediction", back_populates="user", cascade="all, delete-orphan"
    )

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    @property
    def points(self) -> int:
        """Persisted points from FINISHED matches."""
        return sum(p.points_awarded or 0 for p in self.predictions)


class Fixture(db.Model):
    __tablename__ = "fixtures"
    id = db.Column(db.Integer, primary_key=True)
    match_id = db.Column(db.String, unique=True, nullable=False)
    match_date = db.Column(db.DateTime, nullable=False)  # stored in UTC (aware or coerced)
    home_team = db.Column(db.String, nullable=False)
    away_team = db.Column(db.String, nullable=False)
    season = db.Column(db.String, nullable=False)
    matchday = db.Column(db.String, nullable=True)
    status = db.Column(db.String, default="SCHEDULED")  # SCHEDULED/TIMED/IN_PLAY/PAUSED/FINISHED
    home_score = db.Column(db.Integer, nullable=True)
    away_score = db.Column(db.Integer, nullable=True)

    predictions = db.relationship(
        "Prediction", back_populates="fixture", cascade="all, delete-orphan"
    )

    def outcome_code(self) -> str | None:
        """Return '1' for home win, 'X' for draw, '2' for away win, else None."""
        if self.home_score is None or self.away_score is None:
            return None
        if self.home_score > self.away_score:
            return "1"
        if self.home_score < self.away_score:
            return "2"
        return "X"

    def is_open_for_prediction(self) -> bool:
        """Predictions close at kickoff."""
        now_utc = datetime.now(timezone.utc)
        md = self.match_date
        if md.tzinfo is None:  # coerce to UTC if naive
            md = md.replace(tzinfo=timezone.utc)
        return now_utc < md


class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    fixture_id = db.Column(db.Integer, db.ForeignKey("fixtures.id"))
    selection = db.Column(db.String, nullable=False)  # '1', 'X', '2'
    # Persisted once a match is FINISHED (keeps leaderboard stable)
    points_awarded = db.Column(db.Integer, nullable=True)
    timestamp = db.Column(
        db.DateTime, default=lambda: datetime.now(ZoneInfo("America/New_York"))
    )

    user = db.relationship("User", back_populates="predictions")
    fixture = db.relationship("Fixture", back_populates="predictions")


# -----------------------------------------------------------------------------
# Login loader
# -----------------------------------------------------------------------------

@login_manager.user_loader
def load_user(user_id: str):
    return db.session.get(User, int(user_id))


# -----------------------------------------------------------------------------
# Fixtures: Fetch & Update
# -----------------------------------------------------------------------------

def fetch_fixtures_from_api() -> list[dict]:
    """
    football-data.org returns UTC timestamps (utcDate). We keep them in UTC.
    """
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        return []

    today = datetime.now(timezone.utc).astimezone(ZoneInfo("Europe/Rome")).date()
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

        # Keep as timezone-aware UTC
        utc_dt = datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00"))

        fixtures.append(
            {
                "match_id": str(match["id"]),
                "match_date": utc_dt,  # UTC aware
                "home_team": match["homeTeam"]["name"],
                "away_team": match["awayTeam"]["name"],
                "season": season_str,
                "matchday": str(match.get("matchday")),
                "status": status,
                "home_score": (
                    match["score"]["fullTime"]["home"]
                    if match["score"]["fullTime"]["home"] is not None
                    else None
                ),
                "away_score": (
                    match["score"]["fullTime"]["away"]
                    if match["score"]["fullTime"]["away"] is not None
                    else None
                ),
            }
        )
    return fixtures


def fetch_fixtures_from_fallback() -> list[dict]:
    fallback_path = Path(__file__).resolve().parent / 'data' / 'seriea_2024_25.json'
    if not fallback_path.exists():
        return []
    with open(fallback_path, 'r', encoding='utf-8') as f:
        season_data = json.load(f)

    fixtures: list[dict] = []
    for match in season_data.get('matches', []):
        score = match.get('score', {})
        ft = score.get('ft')
        if not ft:
            date_str = match['date']
            time_str = match.get('time', '18:00')
            dt = datetime.fromisoformat(f"{date_str}T{time_str}")
            # interpret as Italy time, then convert to UTC
            dt_local = dt.replace(tzinfo=ZoneInfo('Europe/Rome'))
            utc_dt = dt_local.astimezone(timezone.utc)
            fixtures.append({
                'match_id': f"{date_str}-{match['team1']}-{match['team2']}",
                'match_date': utc_dt,
                'home_team': match['team1'],
                'away_team': match['team2'],
                'season': season_data.get('name', '2024/25'),
                'matchday': match.get('round'),
                'status': 'SCHEDULED',
                'home_score': None,
                'away_score': None,
            })
    return fixtures

def update_fixtures() -> None:
    """
    Sync/update fixtures and persist final points for FINISHED matches.
    """
    fixtures_from_api = fetch_fixtures_from_api()
    fixtures_to_use = fixtures_from_api if fixtures_from_api else fetch_fixtures_from_fallback()

    for fi in fixtures_to_use:
        existing = Fixture.query.filter_by(match_id=fi["match_id"]).first()
        if existing:
            changed = False
            for k in ("status", "home_score", "away_score"):
                if fi[k] != getattr(existing, k):
                    setattr(existing, k, fi[k])
                    changed = True
            # Keep match_date up to date if API changes it
            if fi["match_date"] != existing.match_date:
                existing.match_date = fi["match_date"]
                changed = True
            if changed:
                db.session.add(existing)
        else:
            db.session.add(
                Fixture(
                    match_id=fi["match_id"],
                    match_date=fi["match_date"],
                    home_team=fi["home_team"],
                    away_team=fi["away_team"],
                    season=fi["season"],
                    matchday=fi.get("matchday"),
                    status=fi["status"],
                    home_score=fi["home_score"],
                    away_score=fi["away_score"],
                )
            )

    db.session.commit()
    evaluate_predictions()  # persist points for FINISHED matches


# Adaptive throttle for fetching
FETCH_STATE = {"last_run": None, "last_interval": None}


def _adaptive_min_interval() -> timedelta:
    """Aggressive polling near/live, relaxed otherwise."""
    now = datetime.now(timezone.utc)

    live = Fixture.query.filter(Fixture.status.in_(("IN_PLAY", "PAUSED"))).count()
    if live > 0:
        return timedelta(seconds=60)

    soon = (
        Fixture.query.filter(
            Fixture.match_date >= now, Fixture.match_date <= now + timedelta(hours=2)
        ).count()
    )
    if soon > 0:
        return timedelta(seconds=60)

    today_end = (
        now.astimezone(ZoneInfo("Europe/Rome"))
        .replace(hour=23, minute=59, second=59, microsecond=999999)
        .astimezone(timezone.utc)
    )
    today = Fixture.query.filter(
        Fixture.match_date >= now, Fixture.match_date <= today_end
    ).count()
    if today > 0:
        return timedelta(hours=6)

    return timedelta(hours=24)


def update_fixtures_adaptive(force: bool = False) -> None:
    now = datetime.now(timezone.utc)
    min_interval = _adaptive_min_interval()
    last_run = FETCH_STATE["last_run"]

    if not force and last_run is not None and (now - last_run) < min_interval:
        return

    update_fixtures()
    FETCH_STATE["last_run"] = now
    FETCH_STATE["last_interval"] = min_interval


# -----------------------------------------------------------------------------
# Queries & Live-style scoring
# -----------------------------------------------------------------------------

def upcoming_fixtures() -> list[Fixture]:
    """
    Show fixtures starting within the next 7 days,
    PLUS all fixtures from the season's first matchweek (Matchday 1)
    even if they're further out. (Kickoff lock still applies.)
    """
    now_utc = datetime.now(timezone.utc)
    next_week_utc = now_utc + timedelta(days=7)

    base = (
        Fixture.query
        .filter(Fixture.match_date >= now_utc, Fixture.match_date <= next_week_utc)
        .order_by(Fixture.match_date.asc())
        .all()
    )

    first_upcoming = (
        Fixture.query
        .filter(Fixture.status.in_(('SCHEDULED', 'TIMED')), Fixture.match_date >= now_utc)
        .order_by(Fixture.match_date.asc())
        .first()
    )

    week1 = []
    if first_upcoming and first_upcoming.matchday:
        week1 = (
            Fixture.query
            .filter(
                Fixture.status.in_(('SCHEDULED', 'TIMED')),
                Fixture.matchday == first_upcoming.matchday
            )
            .all()
        )

    merged = {f.id: f for f in base}
    for f in week1:
        merged[f.id] = f
    return sorted(merged.values(), key=lambda f: f.match_date)

def predictions_for_fixtures_detailed(fixtures):
    """
    Return:
      user_cols: list[(user_id, username)] ordered by username
      picks: dict[(fixture_id, user_id)] = '1' | 'X' | '2'
    """
    if not fixtures:
        return [], {}

    ids = [f.id for f in fixtures]
    rows = (
        db.session.query(
            Prediction.fixture_id, User.id, User.username, Prediction.selection
        )
        .join(User, User.id == Prediction.user_id)
        .filter(Prediction.fixture_id.in_(ids))
        .all()
    )

    users = {}
    picks = {}
    for fixture_id, user_id, username, selection in rows:
        users[user_id] = username
        picks[(fixture_id, user_id)] = selection

    user_cols = sorted([(uid, uname) for uid, uname in users.items()],
                       key=lambda t: t[1].lower())
    return user_cols, picks


def dynamic_points_for_fixtures(fixtures):
    """
    Always compute points on the fly for matches that are IN_PLAY or FINISHED.
    Returns dict[(fixture_id, user_id)] = 0|1.
    (No points for SCHEDULED/TIMED/PAUSED.)
    """
    live = {}
    for f in fixtures:
        if f.status not in ("IN_PLAY", "FINISHED"):
            continue
        oc = f.outcome_code()
        if oc is None:
            continue
        for p in f.predictions:
            live[(f.id, p.user_id)] = 1 if p.selection == oc else 0
    return live


def evaluate_predictions() -> None:
    """
    Persist points for FINISHED matches only (keeps leaderboard durable).
    """
    finished = Fixture.query.filter(Fixture.status == "FINISHED").all()
    for fixture in finished:
        oc = fixture.outcome_code()
        if oc is None:
            continue
        for prediction in fixture.predictions:
            if prediction.points_awarded is None:
                prediction.points_awarded = 1 if prediction.selection == oc else 0
                db.session.add(prediction)
    db.session.commit()


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.route("/")
@login_required
def index():
    update_fixtures_adaptive()

    fixtures = upcoming_fixtures()
    user_predictions = {p.fixture_id: p for p in current_user.predictions}

    # Matrix data + always-live scoring (for IN_PLAY & FINISHED)
    user_cols, picks_grid = predictions_for_fixtures_detailed(fixtures)
    live_points = dynamic_points_for_fixtures(fixtures)

    return render_template(
        "index.html",
        fixtures=fixtures,
        user_predictions=user_predictions,
        user_cols=user_cols,      # [(user_id, username)]
        picks_grid=picks_grid,    # {(fixture_id, user_id): '1'|'X'|'2'}
        live_points=live_points,  # {(fixture_id, user_id): 0|1} for IN_PLAY/FINISHED
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

    pred = Prediction.query.filter_by(user_id=current_user.id, fixture_id=fixture_id).first()
    if not pred:
        pred = Prediction(user_id=current_user.id, fixture_id=fixture_id)
        db.session.add(pred)
    pred.selection = selection

    db.session.commit()
    flash("Prediction saved.", "success")
    return redirect(url_for("index"))


@app.route("/save_all_predictions", methods=["POST"])
@login_required
def save_all_predictions():
    # Save any choices for fixtures that are still open
    for fixture in Fixture.query.all():
        if not fixture.is_open_for_prediction():
            continue
        choice = request.form.get(f"fixture_{fixture.id}")
        if choice not in ("1", "X", "2", None):
            continue
        if not choice:
            continue
        pred = Prediction.query.filter_by(
            user_id=current_user.id, fixture_id=fixture.id
        ).first()
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
    users = User.query.all()
    users_sorted = sorted(users, key=lambda u: (-u.points, u.username.lower()))
    return render_template("leaderboard.html", users=users_sorted)


@app.route('/login', methods=['GET', 'POST'])
def login():
    next_url = request.args.get('next')
    if current_user.is_authenticated:
        return redirect(next_url or url_for('index'))

    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        # case-insensitive lookup is friendlier
        user = User.query.filter(func.lower(User.username) == username.lower()).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(next_url or url_for('index'))

        flash('Invalid username or password', 'danger')
        return render_template('login.html')   # <— IMPORTANT: return on POST failure

    return render_template('login.html')       # <— IMPORTANT: return on GET

@app.route('/admin/users', methods=['GET'])
@login_required
def admin_users():
    if not current_user.is_admin:
        abort(403)
    users = User.query.order_by(func.lower(User.username)).all()
    return render_template('admin_users.html', users=users)

