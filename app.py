# app.py
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, jsonify, render_template
from zoneinfo import ZoneInfo

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
ITALY_TZ = ZoneInfo("Europe/Rome")          # Incoming times are Italy local
DEFAULT_TZ = timezone.utc                   # Normalize everything to UTC
API_URL = os.getenv("FIXTURES_API_URL", "").strip()
API_KEY = os.getenv("FIXTURES_API_KEY", "").strip()
LOCAL_JSON = os.getenv("FIXTURES_LOCAL_JSON", "seriea_2024_25.json")

app = Flask(__name__)

# -----------------------------------------------------------------------------
# DATA ACCESS
# -----------------------------------------------------------------------------
def _parse_italy_time_to_utc(italy_str: str) -> datetime:
    """
    Parse a timestamp that represents local Italy time into a timezone-aware UTC datetime.

    Accepts either:
      - naive ISO   : 'YYYY-MM-DDTHH:MM:SS'
      - offset ISO  : 'YYYY-MM-DDTHH:MM:SS±HH:MM'
    """
    try:
        dt = datetime.fromisoformat(italy_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ITALY_TZ)
    except ValueError:
        dt = datetime.strptime(italy_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=ITALY_TZ)
    return dt.astimezone(DEFAULT_TZ)


def fetch_fixtures_from_api() -> List[Dict[str, Any]]:
    """
    Fetch raw fixtures. Expected minimal fields per item (rename as needed):
      - id            (str/int)
      - home, away    (str)
      - kickoff       (str, Italy local time ISO)
      - status        (str) e.g. 'TIMED'|'SCHEDULED'|'LIVE'|'IN_PLAY'|'FINISHED'|'FT'
      - home_score    (int or None)
      - away_score    (int or None)

    Source priority:
      1) HTTP API if FIXTURES_API_URL is set,
      2) Local JSON file (default 'seriea_2024_25.json') if present,
      3) Small demo list fallback.
    """
    # 1) HTTP API
    if API_URL:
        headers = {}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
        resp = requests.get(API_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "fixtures" in data:
            return data["fixtures"]
        return data  # assume it's already a list

    # 2) Local JSON
    if LOCAL_JSON and os.path.exists(LOCAL_JSON):
        with open(LOCAL_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "fixtures" in data:
            return data["fixtures"]
        return data

    # 3) Demo fallback
    return [
        {
            "id": "102",
            "home": "Lazio",
            "away": "Juventus",
            "kickoff": "2025-08-23T18:30:00",  # Italy local
            "status": "TIMED",
            "home_score": 1,
            "away_score": 1,
        },
        {
            "id": "103",
            "home": "Milan",
            "away": "Roma",
            "kickoff": "2025-08-23T20:45:00",
            "status": "TIMED",
            "home_score": 2,
            "away_score": 0,
        },
    ]


def _normalize_status(api_status: str,
                      ko_utc: datetime,
                      home_score: Optional[int],
                      away_score: Optional[int]) -> str:
    """
    Make status UX-friendly and time-aware:
      - Before KO: UPCOMING
      - At/after KO: LIVE (even if API still says TIMED/SCHEDULED)
      - Far after KO with scores or explicit final: FT
    """
    now = datetime.now(DEFAULT_TZ)
    s = (api_status or "").upper()

    # explicit finals from API
    if s in {"FINISHED", "FT", "FULL_TIME"}:
        return "FT"

    if now < ko_utc:
        return "UPCOMING" if s in {"TIMED", "SCHEDULED", ""} else s

    # KO passed:
    if home_score is not None and away_score is not None and now >= ko_utc + timedelta(hours=3):
        return "FT"

    if s in {"TIMED", "SCHEDULED"}:
        return "LIVE"
    if s in {"IN_PLAY", "LIVE", "1ST_HALF", "2ND_HALF", "PAUSED"}:
        return "LIVE"

    # Fallbacks
    return "LIVE" if now >= ko_utc else "UPCOMING"


def load_fixtures() -> List[Dict[str, Any]]:
    """
    End-to-end load + normalize + sort. IMPORTANT: no filtering by time.
    """
    raw = fetch_fixtures_from_api()
    fixtures: List[Dict[str, Any]] = []

    for r in raw:
        # Map/rename incoming fields as needed
        match_id = str(r.get("id"))
        home = r.get("home") or r.get("homeTeam") or r.get("home_name")
        away = r.get("away") or r.get("awayTeam") or r.get("away_name")
        kickoff_src = r.get("kickoff") or r.get("kickoff_it") or r.get("date")

        if not (match_id and home and away and kickoff_src):
            # Skip malformed records but don't crash the page
            continue

        ko_utc = _parse_italy_time_to_utc(kickoff_src)
        status = _normalize_status(
            r.get("status", ""),
            ko_utc,
            r.get("home_score"),
            r.get("away_score"),
        )

        fixtures.append({
            "id": match_id,
            "home": home,
            "away": away,
            "kickoff_utc": ko_utc.isoformat(),
            "status": status,
            "home_score": r.get("home_score"),
            "away_score": r.get("away_score"),
            # If you keep user picks, attach them here, e.g.:
            # "user_pick": get_user_pick(match_id)
        })

    # Show everything; just order by kickoff
    fixtures.sort(key=lambda x: x["kickoff_utc"])
    return fixtures

# -----------------------------------------------------------------------------
# JINJA FILTERS
# -----------------------------------------------------------------------------
@app.template_filter("to_utc_iso")
def to_utc_iso(dt: datetime | str | None) -> str:
    """
    Ensure a UTC ISO string for templating.
    - If dt is already an ISO string, return as-is.
    - If dt is a datetime, convert to UTC ISO.
    - If None, return empty string.
    """
    if dt is None:
        return ""
    if isinstance(dt, str):
        return dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=DEFAULT_TZ)
    return dt.astimezone(DEFAULT_TZ).isoformat()

# -----------------------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    fixtures = load_fixtures()
    # Do NOT filter out LIVE/FT fixtures; keep them visible
    return render_template("index.html", fixtures=fixtures)

@app.route("/api/fixtures")
def api_fixtures():
    """
    JSON for client-side refresh. Your template can poll this
    (e.g., every 30s) to keep LIVE status and scores updated.
    """
    return jsonify({"fixtures": load_fixtures()})

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # In production, run via gunicorn/uvicorn; debug only for local dev
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
