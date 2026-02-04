# Hardamissa Lotto

This repository contains a simple web application that allows invited users to
predict the outcomes of upcoming Serie A matches and compete on a public
leaderboard. Predictions are private until fixtures have been played and are
locked at kickoff. Correct predictions earn one point each week.

## Features

- **User authentication** – Only users with a valid invite code can register.
  Passwords are hashed securely. Logged in users can submit or edit
  predictions for upcoming fixtures.
- **Automatic fixture import** – The app attempts to fetch fixtures and results
  from the [football‑data.org](https://www.football-data.org/) API. You need to
  supply an API key via the `FOOTBALL_DATA_API_KEY` environment variable to
  enable live data. If no key is set the app falls back to a bundled JSON file
  containing the 2024/25 Serie A schedule.
- **Prediction locking and scoring** – Once a match has started predictions
  cannot be changed. After a fixture finishes the app awards one point per
  correct prediction and updates the leaderboard.
- **Leaderboard** – A public table ranks users by total points earned. Only
  usernames and totals are shown; individual predictions remain confidential.
- **Admin panel** – An admin user can log in to create additional invite codes
  and view which codes have been used. The initial admin account is created
  automatically when you initialise the database.

## Setup and running locally

1. **Install dependencies**

   You need Python 3.11+ and `pip`. Install required packages using:

   ```bash
   pip install Flask Flask-SQLAlchemy flask-login requests
   ```

2. **Initialise the database**

   Run the following command to create the SQLite database and the default
   admin user (`admin`/`admin`) with a demo invite code (`demo-invite`):

   ```bash
   FLASK_APP=serie_a_predictor/app.py flask init-db
   ```

3. **(Optional) Configure environment variables**

   - `FLASK_SECRET_KEY` – set this to a random secret in production for secure
     sessions.
   - `FOOTBALL_DATA_API_KEY` – your API token from football‑data.org if you
     want live fixtures and results. Without it the app falls back to the
     bundled schedule.
   - `DATABASE_URL` – override the default SQLite database (e.g. to
     PostgreSQL) if desired.

4. **Run the application**

   Start the development server with:

   ```bash
   python serie_a_predictor/app.py
   ```

   The site will be available at `http://127.0.0.1:5000/`.

## Deployment

The application is self‑contained and can be deployed to any platform that
supports Python and a WSGI server (e.g. Render, Railway or Fly.io). When
deploying, ensure you:

- Set a secure `FLASK_SECRET_KEY` and supply your `FOOTBALL_DATA_API_KEY`.
- Use a production WSGI server such as Gunicorn rather than the built‑in
  development server.
- Run the `init-db` command once to set up the database and create an admin
  account.

## Data sources

The application uses the following data sources:

- **football‑data.org API** – When an API key is provided, fixtures and
  results are downloaded from the `/v4/competitions/SA/matches` endpoint.
  Authentication requires sending an `X‑Auth‑Token` header【267555250325710†L134-L145】.
- **OpenFootball fallback** – If no API key is set the app reads from a JSON
  schedule derived from the [openfootball/football.json](https://github.com/openfootball/football.json)
  project. This file (`data/seriea_2024_25.json`) is bundled in the `data`
  directory and contains the full 2024/25 Italian Serie A fixture list.

## Limitations and future work

This project is a demonstration and does not include every feature you might
expect in a production fantasy application. Potential enhancements include:

- Sending weekly email summaries to users with their results and upcoming
  fixtures.
- Exposing an API to allow integration with other services.
- Improving the user interface with richer statistics and match details.
- Persisting fixture and result data in a background job rather than on
  each page load.