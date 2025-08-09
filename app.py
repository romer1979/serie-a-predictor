"""
Serie A Predictor web application.

This Flask application allows invited users to register, log in and submit simple
predictions (1/X/2) for upcoming Serie A fixtures. The app automatically
retrieves fixtures and results from a football API when possible. Predictions
are locked at kickoff and points are awarded for correct outcomes. A public
leaderboard ranks users by total points. An optional admin interface allows
invite codes to be managed.

The goal of this project is to demonstrate how the core features described
in the specification can be implemented in a concise, readable manner. The
application uses SQLite for persistence, SQLAlchemy as an ORM and
Flask‑Login for session management. Where available, the application will
attempt to fetch upcoming fixtures and update results via the free
`football-data.org` API. If no API key is configured the app falls back to a
read‑only JSON schedule bundled with this repository (see
`data/seriea_2024_25.json`).

To run the application locally:

    python app.py

The `FLASK_SECRET_KEY` and `FOOTBALL_DATA_API_KEY` environment variables
control secret session data and access to football data. When deploying
publicly these values **must** be set to secure random strings. See
README.md for full deployment instructions.
"""

import os
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from secrets import token_urlsafe
from sqlalchemy import func, distinct


import requests
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
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

# -----------------------------------------------------------------------------
# Configuration
#
# The secret key should be set via environment variable in production. For
# local development a reasonable default is provided.

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key')

# DB URL: prefer env (Render Postgres), fall back to local SQLite
raw_db_url = os.environ.get('DATABASE_URL', 'sqlite:///serie_a.db')

# Render sometimes provides postgres URLs as "postgres://"
if raw_db_url.startswith('postgres://'):
    raw_db_url = raw_db_url.replace('postgres://', 'postgresql+psycopg2://', 1)

# Ensure sslmode=require for Postgres if not already present
if raw_db_url.startswith('postgresql'):
    parsed = urlparse(raw_db_url)
    q = parse_qs(parsed.query)
    if 'sslmode' not in q:
        q['sslmode'] = ['require']
        new_query = urlencode(q, doseq=True)
        raw_db_url = urlunparse(parsed._replace(query=new_query))

app.config['SQLALCHEMY_DATABASE_URI'] = raw_db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Engine options to keep connections healthy on Render
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,   # test connections before using them
    'pool_recycle': 300,     # recycle connections every 5 minutes
    'pool_size': 5,          # small pool for free/Starter dynos
    'max_overflow': 5,       # allow brief bursts
    'pool_timeout': 30,      # seconds to wait for a connection
    # psycopg2 will also see sslmode=require via the URL; connect_args is optional
}

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@app.template_filter('utc_iso')
def utc_iso(dt):
    if dt is None:
        return ''
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    s = dt.astimezone(timezone.utc).isoformat()
    return s[:-6] + 'Z' if s.endswith('+00:00') else s
    
# -----------------------------------------------------------------------------
# Models
#

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String, unique=True, nullable=False)
    password_hash = db.Column(db.String, nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    invites_used = db.Column(db.Integer, default=0)

    predictions = db.relationship('Prediction', back_populates='user', cascade='all, delete-orphan')

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    @property
    def points(self) -> int:
        """Total points for the user across all predictions."""
        return sum(p.points_awarded or 0 for p in self.predictions)


class Invite(db.Model):
    __tablename__ = 'invites'
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String, unique=True, nullable=False)
    used_by_user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)


class Fixture(db.Model):
    __tablename__ = 'fixtures'
    id = db.Column(db.Integer, primary_key=True)
    match_id = db.Column(db.String, unique=True, nullable=False)
    match_date = db.Column(db.DateTime, nullable=False)  # optional: DateTime(timezone=True)
    home_team = db.Column(db.String, nullable=False)
    away_team = db.Column(db.String, nullable=False)
    season = db.Column(db.String, nullable=False)
    matchday = db.Column(db.String, nullable=True)
    status = db.Column(db.String, default='SCHEDULED')
    home_score = db.Column(db.Integer, nullable=True)
    away_score = db.Column(db.Integer, nullable=True)

    predictions = db.relationship('Prediction', back_populates='fixture', cascade='all, delete-orphan')

    def outcome_code(self) -> str | None:
        if self.home_score is None or self.away_score is None:
            return None
        if self.home_score > self.away_score:
            return '1'
        if self.home_score < self.away_score:
            return '2'
        return 'X'

    def is_open_for_prediction(self) -> bool:
        now_utc = datetime.now(timezone.utc)
        md = self.match_date
        if md.tzinfo is None:
            md = md.replace(tzinfo=timezone.utc)
        return now_utc < md



class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    fixture_id = db.Column(db.Integer, db.ForeignKey('fixtures.id'))
    selection = db.Column(db.String, nullable=False)  # '1', 'X' or '2'
    points_awarded = db.Column(db.Integer, nullable=True)  # 1 for correct, 0 for incorrect, None until evaluated
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(ZoneInfo('America/New_York')))

    user = db.relationship('User', back_populates='predictions')
    fixture = db.relationship('Fixture', back_populates='predictions')


# -----------------------------------------------------------------------------
# Authentication and user loading
#

@login_manager.user_loader
def load_user(user_id: str):
    return db.session.get(User, int(user_id))


# -----------------------------------------------------------------------------
# Utility functions
#
def fetch_fixtures_from_api() -> list[dict]:
    api_key = os.environ.get('FOOTBALL_DATA_API_KEY')
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
    for match in data.get('matches', []):
        status = match.get('status')
        if status not in ('SCHEDULED', 'TIMED', 'IN_PLAY', 'PAUSED', 'FINISHED'):
            continue
        utc_date_str = match['utcDate']  # ISO string in UTC
        utc_dt = datetime.fromisoformat(utc_date_str.replace('Z', '+00:00'))  # aware UTC
        fixtures.append({
            'match_id': str(match['id']),
            'match_date': utc_dt,  # <-- keep UTC
            'home_team': match['homeTeam']['name'],
            'away_team': match['awayTeam']['name'],
            'season': season_str,
            'matchday': str(match.get('matchday')),
            'status': status,
            'home_score': match['score']['fullTime']['home'] if match['score']['fullTime']['home'] is not None else None,
            'away_score': match['score']['fullTime']['away'] if match['score']['fullTime']['away'] is not None else None,
        })
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
            # Europe/Rome -> UTC
            dt_naive = datetime.fromisoformat(f"{date_str}T{time_str}")
            dt_rome = dt_naive.replace(tzinfo=ZoneInfo('Europe/Rome'))
            utc_dt = dt_rome.astimezone(timezone.utc)
            fixtures.append({
                'match_id': f"{date_str}-{match['team1']}-{match['team2']}",
                'match_date': utc_dt,  # <-- UTC
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
    Sync local fixtures with API (if key present) or fallback file.
    Insert new fixtures; update status/scores on existing ones.
    Then evaluate predictions for finished matches.
    """
    fixtures_from_api = fetch_fixtures_from_api()  # requires FOOTBALL_DATA_API_KEY or returns []
    fixtures_to_use = fixtures_from_api if fixtures_from_api else fetch_fixtures_from_fallback()

    for fi in fixtures_to_use:
        existing = Fixture.query.filter_by(match_id=fi['match_id']).first()
        if existing:
            updated = False
            if fi['status'] != existing.status:
                existing.status = fi['status']; updated = True
            if fi['home_score'] is not None and fi['home_score'] != existing.home_score:
                existing.home_score = fi['home_score']; updated = True
            if fi['away_score'] is not None and fi['away_score'] != existing.away_score:
                existing.away_score = fi['away_score']; updated = True
            if updated:
                db.session.add(existing)
        else:
            db.session.add(Fixture(
                match_id=fi['match_id'],
                match_date=fi['match_date'],
                home_team=fi['home_team'],
                away_team=fi['away_team'],
                season=fi['season'],
                matchday=fi.get('matchday'),
                status=fi['status'],
                home_score=fi['home_score'],
                away_score=fi['away_score'],
            ))

    db.session.commit()
    evaluate_predictions()
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
    # keep as strings to match your storage; filter out None/'' then sort numerically if possible
    days = [r[0] for r in rows if r[0]]
    try:
        return [str(x) for x in sorted({int(d) for d in days})]
    except Exception:
        return sorted(set(days))

def latest_completed_matchday(season: str) -> str | None:
    # Pick the highest matchday that has at least one FINISHED fixture
    rows = (
        db.session.query(Fixture.matchday)
        .filter(Fixture.season == season, Fixture.status == 'FINISHED')
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

def weekly_user_points(season: str, matchday: str):
    # returns list of tuples: (user_id, username, points) for that week
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

# --- Adaptive fetch throttle ---
FETCH_STATE = {"last_run": None, "last_interval": None}

def _adaptive_min_interval() -> timedelta:
    """Return how often we should hit the API right now."""
    now_utc = datetime.now(timezone.utc)

    # If anything is live, poll every 60s
    live = Fixture.query.filter(Fixture.status.in_(('IN_PLAY', 'PAUSED'))).count()
    if live > 0:
        return timedelta(seconds=60)

    # If matches kick off within the next 2 hours, poll every 60s
    soon = (
        Fixture.query
        .filter(Fixture.match_date >= now_utc, Fixture.match_date <= now_utc + timedelta(hours=2))
        .count()
    )
    if soon > 0:
        return timedelta(seconds=60)

    # If matches are today (UTC day), poll every 6 hours
    today_end_utc = now_utc.replace(hour=23, minute=59, second=59, microsecond=999999)
    today = (
        Fixture.query
        .filter(Fixture.match_date >= now_utc, Fixture.match_date <= today_end_utc)
        .count()
    )
    if today > 0:
        return timedelta(hours=6)

    # Otherwise quiet times
    return timedelta(hours=24)

def update_fixtures_adaptive(force: bool = False) -> None:
    """Call update_fixtures() only if the adaptive interval has elapsed."""
    now_utc = datetime.now(timezone.utc)

    min_interval = _adaptive_min_interval()
    last_run = FETCH_STATE["last_run"]

    if not force and last_run is not None and (now_utc - last_run) < min_interval:
        return

    update_fixtures()

    FETCH_STATE["last_run"] = now_utc
    FETCH_STATE["last_interval"] = min_interval

def upcoming_fixtures() -> list[Fixture]:
    now_utc = datetime.now(timezone.utc)
    next_week_utc = now_utc + timedelta(days=7)

    base = (
        Fixture.query.filter(
            Fixture.match_date >= now_utc,
            Fixture.match_date <= next_week_utc
        ).all()
    )

    first_upcoming = (
        Fixture.query.filter(
            Fixture.status.in_(('SCHEDULED', 'TIMED')),
            Fixture.match_date >= now_utc
        )
        .order_by(Fixture.match_date.asc())
        .first()
    )

    week1 = []
    if first_upcoming and first_upcoming.matchday:
        week1 = (
            Fixture.query.filter(
                Fixture.status.in_(('SCHEDULED', 'TIMED')),
                Fixture.matchday == first_upcoming.matchday
            ).all()
        )

    merged = {f.id: f for f in base}
    for f in week1:
        merged[f.id] = f
    return sorted(merged.values(), key=lambda f: f.match_date)

def evaluate_predictions() -> None:
    """
    Iterate through predictions and award points for matches that have finished
    and haven't been evaluated yet. Each correct prediction yields 1 point.
    """
    finished_fixtures = Fixture.query.filter(Fixture.status == 'FINISHED').all()
    for fixture in finished_fixtures:
        outcome = fixture.outcome_code()
        for prediction in fixture.predictions:
            if prediction.points_awarded is None:
                prediction.points_awarded = 1 if prediction.selection == outcome else 0
                db.session.add(prediction)
    db.session.commit()

def predictions_for_fixtures(fixtures: list[Fixture]) -> dict[int, list[tuple[str, str]]]:
    """Return {fixture_id: [(username, selection), ...]} for the given fixtures."""
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
def current_season_from_db() -> str | None:
    row = db.session.query(Fixture.season).order_by(Fixture.season.desc()).first()
    return row[0] if row else None

def all_matchdays_for_season(season: str) -> list[str]:
    rows = (
        db.session.query(distinct(Fixture.matchday))
        .filter(Fixture.season == season)
        .all()
    )
    mds = [r[0] for r in rows if r[0] is not None]
    # Try numeric sort when possible, fallback to lexical
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
        if any(s in ('IN_PLAY','PAUSED') for s in statuses):
            md_status[md] = 'live'
        elif statuses and statuses.issubset({'FINISHED'}):
            md_status[md] = 'finished'
        elif any(s in ('SCHEDULED','TIMED') for s in statuses):
            future = Fixture.query.filter(
                Fixture.season==season,
                Fixture.matchday==md,
                Fixture.match_date>=now_utc
            ).count()
            md_status[md] = 'upcoming' if future else 'finished'
        else:
            md_status[md] = 'other'

    def _order(lst):
        try: return [str(n) for n in sorted({int(x) for x in lst})]
        except: return sorted(set(lst), key=lambda s: (len(s), s))

    finished = _order([m for m,s in md_status.items() if s=='finished'])
    live = _order([m for m,s in md_status.items() if s=='live'])
    upcoming = _order([m for m,s in md_status.items() if s=='upcoming'])
    other = _order([m for m,s in md_status.items() if s=='other'])
    return finished, live, upcoming, other
def prediction_matrix(fixtures):
    """
    Return (users, matrix, show_flags)
      users: list of (user_id, username) who have at least one prediction among these fixtures (sorted by username)
      matrix: dict[(fixture_id, user_id)] -> '1' | 'X' | '2' (or None if no pick)
      show_flags: dict[fixture_id] -> bool  (True if predictions can be shown now)
    """
    from datetime import timezone
    if not fixtures:
        return [], {}, {}

    fix_ids = [f.id for f in fixtures]

    # Pull predictions joined with users for just these fixtures
    rows = (
        db.session.query(User.id, User.username, Prediction.fixture_id, Prediction.selection)
        .join(Prediction, Prediction.user_id == User.id)
        .filter(Prediction.fixture_id.in_(fix_ids))
        .all()
    )

    # Who appears at least once?
    user_set = {}
    for uid, uname, _, _ in rows:
        user_set[uid] = uname
    # Order users by username
    users = sorted(user_set.items(), key=lambda t: t[1].lower())

    # Build the (fixture_id, user_id) -> selection matrix
    matrix = {}
    for uid, uname, fid, sel in rows:
        matrix[(fid, uid)] = sel

    # Compute reveal flags per fixture: show on/after kickoff or if live/paused/finished
    utc_now = datetime.now(timezone.utc)
    show_flags = {}
    for f in fixtures:
        kickoff = f.match_date
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        show_flags[f.id] = (f.status in ('IN_PLAY', 'PAUSED', 'FINISHED')) or (utc_now >= kickoff)

    return users, matrix, show_flags

def season_user_points(season: str):
    """
    Return list of dicts [{username, points}] for the given season,
    ordered by points desc then username asc.
    """
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
        [{'username': r[0], 'points': int(r[1])} for r in rows],
        key=lambda x: (-x['points'], x['username'].lower())
    )

# -----------------------------------------------------------------------------
# Routes
#

@app.route('/')
@login_required
def index():
    update_fixtures_adaptive()
    fixtures = upcoming_fixtures()
    user_predictions = {p.fixture_id: p for p in current_user.predictions}

    users_cols, pred_matrix, show_flags = prediction_matrix(fixtures)

    return render_template(
        'index.html',
        fixtures=fixtures,
        user_predictions=user_predictions,
        users_cols=users_cols,        # list[(user_id, username)]
        pred_matrix=pred_matrix,      # dict[(fixture_id, user_id)] -> '1'|'X'|'2'
        show_preds_flags=show_flags,  # dict[fixture_id] -> bool
    )


@app.route('/predict/<int:fixture_id>', methods=['POST'])
@login_required
def predict(fixture_id: int):
    fixture = db.session.get(Fixture, fixture_id)
    if not fixture:
        abort(404)
    if not fixture.is_open_for_prediction():
        flash('Predictions are locked for this fixture.', 'warning')
        return redirect(url_for('index'))
    selection = request.form.get('selection')
    if selection not in ('1', 'X', '2'):
        flash('Invalid prediction.', 'danger')
        return redirect(url_for('index'))
    prediction = Prediction.query.filter_by(user_id=current_user.id, fixture_id=fixture_id).first()
    if prediction:
        prediction.selection = selection
        flash('Prediction updated.', 'success')
    else:
        prediction = Prediction(user_id=current_user.id, fixture_id=fixture_id, selection=selection)
        db.session.add(prediction)
        flash('Prediction submitted.', 'success')
    db.session.commit()
    return redirect(url_for('index'))

@app.route('/save_all_predictions', methods=['POST'])
@login_required
def save_all_predictions():
    # Only consider fixtures still open for prediction
    open_fixtures = Fixture.query.all()
    for fixture in open_fixtures:
        if not fixture.is_open_for_prediction():
            continue
        choice = request.form.get(f'fixture_{fixture.id}')
        if choice not in ('1', 'X', '2', None):
            continue
        if choice:
            pred = Prediction.query.filter_by(user_id=current_user.id, fixture_id=fixture.id).first()
            if not pred:
                pred = Prediction(user_id=current_user.id, fixture_id=fixture.id)
                db.session.add(pred)
            pred.selection = choice
    db.session.commit()
    flash("All predictions saved!", "success")
    return redirect(url_for('index'))

@app.route('/leaderboard')
@login_required
def leaderboard():
    update_fixtures_adaptive()

    # Compute available + current season
    seasons = seasons_available()
    current_season = current_season_from_db() or (seasons[-1] if seasons else None)

    # Default scope is SEASON
    raw_scope = request.args.get('scope')
    scope = (raw_scope or 'season').lower()

    # If the user opened /leaderboard with no params, redirect to canonical Season URL
    if raw_scope is None and current_season:
        return redirect(url_for('leaderboard', scope='season', season=current_season))

    season = request.args.get('season') or current_season
    matchday = request.args.get('matchday')

    # ---- Matchday scope (unchanged) ----
    if scope == 'week' and season:
        days = matchdays_for(season)
        if not matchday:
            matchday = latest_completed_matchday(season) or (days[-1] if days else None)
        rows = weekly_user_points(season, matchday) if matchday else []
        users_sorted = [{'username': r[1], 'points': int(r[2])} for r in rows]
        return render_template('leaderboard.html',
                               users=users_sorted,
                               scope='week',
                               seasons=seasons, season=season,
                               matchdays=days, matchday=matchday)

    # ---- Season scope (default) ----
    if scope == 'season' and season:
        users_sorted = season_user_points(season)
        return render_template('leaderboard.html',
                               users=users_sorted,
                               scope='season',
                               seasons=seasons, season=season,
                               matchdays=matchdays_for(season) if season else [],
                               matchday=None)

    # ---- Overall scope (fallback) ----
    users = User.query.all()
    users_sorted = sorted(users, key=lambda u: (-u.points, u.username.lower()))
    return render_template('leaderboard.html',
                           users=users_sorted,
                           scope='overall',
                           seasons=seasons, season=season,
                           matchdays=matchdays_for(season) if season else [],
                           matchday=matchday)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            password = request.form.get('password')
            invite_code = request.form.get('invite')

            invite = Invite.query.filter_by(code=invite_code, used_by_user_id=None).first()
            if not invite:
                flash('Invalid or used invite code.', 'danger')
                return render_template('register.html')

            if User.query.filter_by(username=username).first():
                flash('Username already exists.', 'danger')
                return render_template('register.html')

            if not username or not password:
                flash('Username and password are required.', 'danger')
                return render_template('register.html')

            user = User(username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.flush()  # get user.id

            invite.used_by_user_id = user.id
            db.session.add(invite)
            db.session.commit()

            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            # Log full error to Render logs
            import traceback
            print("[REGISTER_ERROR]", repr(e))
            traceback.print_exc()
            db.session.rollback()
            flash('Registration failed due to a server error. Please try again.', 'danger')
            return render_template('register.html')
    return render_template('register.html')

@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin():
    if not current_user.is_admin:
        abort(403)
    if request.method == 'POST':
        code = request.form.get('code')
        if code:
            if Invite.query.filter_by(code=code).first():
                flash('Invite code already exists.', 'danger')
            else:
                invite = Invite(code=code)
                db.session.add(invite)
                db.session.commit()
                flash('Invite created.', 'success')
    invites = Invite.query.all()
    return render_template('admin.html', invites=invites)

@app.route('/history')
@login_required
def history():
    update_fixtures_adaptive()  # keep data fresh

    seasons = seasons_available()
    if not seasons:
        flash("No seasons available yet.", "warning")
        return render_template('history.html', seasons=[], matchdays=[], season=None, matchday=None,
                               fixtures=[], all_preds={}, weekly_rows=[])

    season = request.args.get('season') or seasons[-1]  # default to latest season lexically
    days = matchdays_for(season)
    if not days:
        flash("No matchdays for that season yet.", "warning")
        return render_template('history.html', seasons=seasons, matchdays=[], season=season, matchday=None,
                               fixtures=[], all_preds={}, weekly_rows=[])

    # default to latest completed matchday if not provided, else latest available
    matchday = request.args.get('matchday')
    if not matchday:
        matchday = latest_completed_matchday(season) or days[-1]

    # fixtures of that week
    fixtures = (
        Fixture.query
        .filter(Fixture.season == season, Fixture.matchday == str(matchday))
        .order_by(Fixture.match_date.asc())
        .all()
    )

    all_preds = predictions_for_fixtures(fixtures)
    weekly_rows = weekly_user_points(season, matchday)

    return render_template(
        'history.html',
        seasons=seasons,
        matchdays=days,
        season=season,
        matchday=matchday,
        fixtures=fixtures,
        all_preds=all_preds,
        weekly_rows=weekly_rows,
        utc_now=datetime.now(timezone.utc),
    )
@app.route('/season/<season>/matchdays')
@login_required
def matchdays(season):
    have_any = Fixture.query.filter_by(season=season).first()
    if not have_any:
        flash(f"No fixtures found for season {season}.", "warning")
        return redirect(url_for('index'))

    finished, live, upcoming, other = classify_matchdays(season)
    next_two = upcoming[:2]
    return render_template(
        'matchdays.html',
        season=season,
        finished=finished,
        live=live,
        next_two=next_two,
        remaining_upcoming=upcoming[2:],
        other=other,
    )
    
@app.route('/matchdays')
@login_required
def matchdays_current():
    season = current_season_from_db()
    if not season:
        flash("No season data available yet.", "warning")
        return redirect(url_for('index'))
    return redirect(url_for('matchdays', season=season))

# --- Admin: list users, delete user, reset password ---

@app.route('/admin/users', methods=['GET'])
@login_required
def admin_users():
    if not current_user.is_admin:
        abort(403)
    users = User.query.order_by(func.lower(User.username)).all()
    return render_template('admin_users.html', users=users)

@app.post("/admin/users/<int:user_id>/delete")
@login_required
def admin_delete_user(user_id: int):
    if not current_user.is_admin:
        abort(403)

    if current_user.id == user_id:
        flash("You can't delete your own account from here.", "warning")
        return redirect(url_for("admin_users"))

    user = db.session.get(User, user_id)
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("admin_users"))

    # Optional: prevent deleting the last remaining admin
    if user.is_admin:
        admin_count = User.query.filter_by(is_admin=True).count()
        if admin_count <= 1:
            flash("Cannot delete the last remaining admin.", "warning")
            return redirect(url_for("admin_users"))

    try:
        # Clear invites referencing this user (FK constraint)
        Invite.query.filter_by(used_by_user_id=user.id).update(
            {"used_by_user_id": None}, synchronize_session=False
        )
        # Delete predictions (relationship cascade should handle this, but explicit is fine)
        Prediction.query.filter_by(user_id=user.id).delete(synchronize_session=False)

        db.session.delete(user)
        db.session.commit()
        flash("User deleted.", "success")
    except Exception as e:
        db.session.rollback()
        print("[ADMIN_DELETE_USER_ERROR]", repr(e))
        flash("Failed to delete user due to a server/database error.", "danger")

    return redirect(url_for("admin_users"))


    # safety rails
    if user.id == current_user.id:
        flash("You cannot delete your own account while logged in as admin.", "warning")
        return redirect(url_for('admin_users'))
    if user.is_admin:
        admin_count = User.query.filter_by(is_admin=True).count()
        if admin_count <= 1:
            flash("Cannot delete the last remaining admin.", "warning")
            return redirect(url_for('admin_users'))

    # cascades will remove predictions thanks to relationship
    db.session.delete(user)
    db.session.commit()
    flash(f"Deleted user '{user.username}'.", "success")
    return redirect(url_for('admin_users'))

@app.route('/admin/users/<int:user_id>/reset_password', methods=['POST'])
@login_required
def admin_reset_password(user_id: int):
    if not current_user.is_admin:
        abort(403)
    user = db.session.get(User, user_id)
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for('admin_users'))

    new_pw = (request.form.get('new_password') or "").strip()
    if not new_pw:
        # generate a secure temporary password if admin left it blank
        new_pw = token_urlsafe(12)

    user.set_password(new_pw)
    db.session.add(user)
    db.session.commit()
    flash(f"Password reset for '{user.username}'. New password: {new_pw}", "success")
    return redirect(url_for('admin_users'))

@app.route("/admin/users/<int:user_id>/edit", methods=["GET", "POST"])
@login_required
def admin_edit_user(user_id: int):
    if not current_user.is_admin:
        abort(403)

    user = db.session.get(User, user_id)
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("admin_users"))

    if request.method == "POST":
        new_username = (request.form.get("username") or "").strip()
        make_admin = request.form.get("is_admin") == "on"
        new_password = (request.form.get("new_password") or "").strip()

        # Basic validation
        if not new_username:
            flash("Username cannot be empty.", "danger")
            return redirect(url_for("admin_edit_user", user_id=user.id))

        # Unique username check (ignore current user)
        existing = User.query.filter(
            func.lower(User.username) == new_username.lower(),
            User.id != user.id
        ).first()
        if existing:
            flash("That username is already taken.", "danger")
            return redirect(url_for("admin_edit_user", user_id=user.id))

        # If unchecking admin, make sure we're not removing the last admin
        if user.is_admin and not make_admin:
            admin_count = User.query.filter_by(is_admin=True).count()
            if admin_count <= 1:
                flash("You cannot remove admin rights from the last remaining admin.", "warning")
                return redirect(url_for("admin_edit_user", user_id=user.id))

        # Apply updates
        user.username = new_username
        user.is_admin = make_admin
        if new_password:
            user.set_password(new_password)

        try:
            db.session.add(user)
            db.session.commit()
            flash("User updated successfully.", "success")
            return redirect(url_for("admin_users"))
        except Exception as e:
            db.session.rollback()
            print("[ADMIN_EDIT_USER_ERROR]", repr(e))
            flash("Failed to update user due to a server/database error.", "danger")
            return redirect(url_for("admin_edit_user", user_id=user.id))

    # GET
    return render_template("admin_user_edit.html", user=user)
    
@app.route('/admin/refresh', methods=['POST'])
@login_required
def admin_refresh():
    if not current_user.is_admin:
        abort(403)
    update_fixtures_adaptive(force=True)
    flash("Fixtures refreshed.", "success")
    return redirect(url_for('index'))

# -----------------------------------------------------------------------------
# CLI commands for setup and administration
#

@app.cli.command('init-db')
def init_db_command() -> None:
    """Initialise the database tables and create an initial admin account."""
    db.create_all()
    if not User.query.filter_by(is_admin=True).first():
        admin_user = User(username='admin', is_admin=True)
        admin_user.set_password('admin')
        db.session.add(admin_user)
        invite = Invite(code='demo-invite')
        db.session.add(invite)
        db.session.commit()
        print('Admin user created with username "admin" and password "admin".')
    else:
        print('Admin user already exists.')

# Ensure DB tables exist on startup (safe: will not overwrite existing data)
with app.app_context():
    db.create_all()

# --- one-time admin force reset (controlled by env) ---
with app.app_context():
    flag = (os.getenv("ADMIN_FORCE_RESET", "0") or "").strip()
    uname = (os.getenv("ADMIN_USERNAME", "admin") or "").strip()
    pwd = os.getenv("ADMIN_PASSWORD")
    invite_code = (os.getenv("INITIAL_INVITE_CODE", os.getenv("INVITE_CODE", "demo-invite")) or "").strip()

    print(f"[BOOTSTRAP] Reset flag={flag} username={uname}")  # visibility in logs (no password printed)

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

@app.cli.command('fix-times-utc')
def fix_times_utc():
    from sqlalchemy import select
    changed = 0
    for f in db.session.execute(select(Fixture)).scalars():
        md = f.match_date
        if md is None:
            continue
        if md.tzinfo is None:
            # Assume legacy were Italy local; convert to UTC
            md = md.replace(tzinfo=ZoneInfo('Europe/Rome')).astimezone(timezone.utc)
            f.match_date = md
            changed += 1
        else:
            # If stored as America/New_York, convert to UTC
            try:
                if getattr(md.tzinfo, "key", None) == "America/New_York":
                    f.match_date = md.astimezone(timezone.utc)
                    changed += 1
            except Exception:
                # If tzinfo doesn’t have .key, still normalize to UTC
                f.match_date = md.astimezone(timezone.utc)
                changed += 1
    db.session.commit()
    print(f"Normalized {changed} fixture times to UTC.")
    
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)
