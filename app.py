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
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

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

# -----------------------------------------------------------------------------
# Configuration
#
# The secret key should be set via environment variable in production. For
# local development a reasonable default is provided.

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key')

# Use SQLite by default. Override with DATABASE_URL for PostgreSQL or other
# engines. SQLAlchemy can parse URLs like `postgresql://user:pass@host/dbname`.
database_url = os.environ.get('DATABASE_URL', 'sqlite:///serie_a.db')
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


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
    match_date = db.Column(db.DateTime, nullable=False)
    home_team = db.Column(db.String, nullable=False)
    away_team = db.Column(db.String, nullable=False)
    season = db.Column(db.String, nullable=False)
    matchday = db.Column(db.String, nullable=True)
    status = db.Column(db.String, default='SCHEDULED')
    home_score = db.Column(db.Integer, nullable=True)
    away_score = db.Column(db.Integer, nullable=True)

    predictions = db.relationship('Prediction', back_populates='fixture', cascade='all, delete-orphan')

    def outcome_code(self) -> str | None:
        """Return '1' for home win, 'X' for draw, '2' for away win, or None if not finished."""
        if self.home_score is None or self.away_score is None:
            return None
        if self.home_score > self.away_score:
            return '1'
        if self.home_score < self.away_score:
            return '2'
        return 'X'

    from zoneinfo import ZoneInfo
from datetime import datetime

class Fixture(db.Model):
    # ...
    match_date = db.Column(db.DateTime, nullable=False)
    # ...

    def is_open_for_prediction(self) -> bool:
        """Return True if predictions are still allowed for this fixture."""
        now = datetime.now(ZoneInfo('America/New_York'))
        md = self.match_date
        if md.tzinfo is None:
            # Treat naive timestamps as America/New_York
            md = md.replace(tzinfo=ZoneInfo('America/New_York'))
        return now < md

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
    """
    Attempt to fetch Serie A fixtures from the football-data.org API. Returns a
    list of fixture dicts with keys matching Fixture fields (match_id, date,
    home_team, away_team, season, matchday, status, home_score, away_score).

    The free football-data.org plan requires an API key to be sent in the
    `X-Auth-Token` header【267555250325710†L134-L145】. If the API key is not set, this function
    returns an empty list.
    """
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
        utc_date_str = match['utcDate']
        utc_dt = datetime.fromisoformat(utc_date_str.replace('Z', '+00:00'))
        local_dt = utc_dt.astimezone(ZoneInfo('America/New_York'))
        fixtures.append({
            'match_id': str(match['id']),
            'match_date': local_dt,
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
    """
    Load fixtures from a bundled JSON file under data/. This fallback includes
    the full 2024/25 Serie A season from the openfootball project. Each match
    dictionary contains fields similar to the API. Only matches with no final
    score yet (future fixtures) are returned. Times are assumed to be local
    18:00 Eastern if unspecified.
    """
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
            dt_local = dt.replace(tzinfo=ZoneInfo('Europe/Rome')).astimezone(ZoneInfo('America/New_York'))
            fixtures.append({
                'match_id': f"{date_str}-{match['team1']}-{match['team2']}",
                'match_date': dt_local,
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

def upcoming_fixtures() -> list[Fixture]:
    """
    Show fixtures starting within the next 7 days,
    PLUS all fixtures from the season's first matchweek (Matchday 1)
    even if they're further out. Predictions still lock at kickoff.
    """
    tz = ZoneInfo('America/New_York')
    now = datetime.now(tz)
    next_week = now + timedelta(days=7)

    # Base: next 7 days
    base_q = (
        Fixture.query.filter(
            Fixture.match_date >= now,
            Fixture.match_date <= next_week
        )
    )

    base = base_q.all()

    # Find the very first upcoming scheduled fixture to detect Week 1
    first_upcoming = (
        Fixture.query.filter(
            Fixture.status.in_(('SCHEDULED', 'TIMED')),
            Fixture.match_date >= now
        )
        .order_by(Fixture.match_date.asc())
        .first()
    )

    week1 = []
    if first_upcoming and first_upcoming.matchday:
        # Include ALL fixtures that share that first match's matchday (Week 1),
        # regardless of whether they’re more than 7 days away.
        week1 = (
            Fixture.query.filter(
                Fixture.status.in_(('SCHEDULED', 'TIMED')),
                Fixture.matchday == first_upcoming.matchday
            )
            .all()
        )

    # Merge + de-dup + sort
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

# -----------------------------------------------------------------------------
# Routes
#

@app.route('/')
@login_required
def index():
    update_fixtures()
    fixtures = upcoming_fixtures()
    user_predictions = {p.fixture_id: p for p in current_user.predictions}
    # NEW: add all users' predictions for visible-after-kickoff column
    all_preds = predictions_for_fixtures(fixtures)
    return render_template(
        'index.html',
        fixtures=fixtures,
        user_predictions=user_predictions,
        all_preds=all_preds,  # <-- pass to template
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


@app.route('/leaderboard')
@login_required
def leaderboard():
    update_fixtures()
    users = User.query.all()
    users_sorted = sorted(users, key=lambda u: (-u.points, u.username.lower()))
    return render_template('leaderboard.html', users=users_sorted)


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

    
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)
