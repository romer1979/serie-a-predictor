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
UTC = timezone.utc                          # Normalize everything to UTC

API_URL = os.getenv("FIXTURES_API_URL", "").strip()
API_KEY = os.getenv("FIXTURES_API_KEY", "").strip()
LOCAL_JSON = os.getenv("FIXTURES_LOCAL_JSON", "seriea_2024_25.json")

app = Flask(__name__)

# -----------------------------------------------------------------------------
# (Optional) provide a dummy current_user so templates never 500 if Flask-Login
# isn't configured but layout.html expects it.
# -----------------------------------------------------------------------------
class _AnonUser:
    is_authenticated = False

@app.context_processor
def inject_current_user():
    return {"current_user": _AnonUser()}

# -----------------------------------------------------------------------------
# DATA ACCESS
# -----------------------------------------------------------------------------
def _parse_italy_time_to_utc(italy_str: str) -> datetime:
    """Parse a timestamp representing Italy local time into UTC-aware datetime."""
    try:
        dt = datetime.fromisoformat(italy_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ITALY_TZ)
    except ValueError:
        dt = datetime.strptime(italy_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=ITALY_TZ)
    return dt.astimezone(UTC)

def fetch_fixtures_from_api() -> List[Dict[str, Any]]:
    """
    Minimal expected fields per item (adapt names if your source differs):
      - id (str/int)
      - home, away (str)
      - kickoff (str, Italy local time in ISO)
      - status (str) e.g. 'TIMED'|'SCHEDULED'|'LIVE'|'IN_PLAY'|'FINISHED'|'FT'
      - home_score, away_score (int or None)
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
        return data

    # 2) Local JSON fallback
    if LOCAL_JSON and os.path.exists(LOCAL_JSON):
        with open(LOCAL_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "fixtures" in data:
            return data["fixtures"]
        return data

    # 3) Tiny demo
    return [
        {
            "id": "102",
            "home": "Lazio",
            "away": "Juventus",
            "kickoff": "2025-08-23T18:30:00",
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
    UX-friendly status:
      - Before KO: UPCOMING
      - From KO until final: LIVE (even if API says TIMED/SCHEDULED)
      - 3h after KO with scores or explicit final: FT
    """
    now = datetime.now(UTC)
    s = (api_status or "").upper()

    if s in {"FINISHED", "FT", "FULL_TIME"}:
        return "FT"

    if now < ko_utc:
        return "UPCOMING"

    # KO passed
    if home_score is not None and away_score is not None and now >= ko_utc + timedelta(hours=3):
        return "FT"

    if s in {"TIMED", "SCHEDULED"}:
        return "LIVE"
    if s in {"IN_PLAY", "LIVE", "1ST_HALF", "2ND_HALF", "PAUSED"}:
        return "LIVE"

    return "LIVE"  # safe default after KO

def load_fixtures() -> List[Dict[str, Any]]:
    """
    Load + normalize + sort. IMPORTANT: do NOT filter by time (keeps LIVE/FT visible).
    """
    raw = fetch_fixtures_from_api()
    fixtures: List[Dict[str, Any]] = []

    for r in raw:
        match_id = str(r.get("id"))
        home = r.get("home") or r.get("homeTeam") or r.get("home_name")
        away = r.get("away") or r.get("awayTeam") or r.get("away_name")
        kickoff_src = r.get("kickoff") or r.get("kickoff_it") or r.get("date")

        if not (match_id and home and away and kickoff_src):
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
        })

    fixtures.sort(key=lambda x: x["kickoff_utc"])
    return fixtures

# -----------------------------------------------------------------------------
# JINJA FILTERS
# Provide BOTH names so templates can use either.
# -----------------------------------------------------------------------------
@app.template_filter("to_utc_iso")
def to_utc_iso(dt: datetime | str | None) -> str:
    if dt is None:
        return ""
    if isinstance(dt, str):
        return dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()

@app.template_filter("utc_iso")
def utc_iso(dt: datetime | str | None) -> str:
    # alias for convenience / backwards-compat
    return to_utc_iso(dt)

# -----------------------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    fixtures = load_fixtures()
    return render_template("index.html", fixtures=fixtures)

@app.route("/api/fixtures")
def api_fixtures():
    return jsonify({"fixtures": load_fixtures()})

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
