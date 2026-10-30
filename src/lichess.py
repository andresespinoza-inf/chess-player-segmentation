"""Fetch a player's recent games from the public Lichess API and shape them like the training data."""
from __future__ import annotations

import json
import os
import re

import pandas as pd
import requests

from .features import GAME_COLUMNS

API_URL = "https://lichess.org/api/games/user/{username}"
USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{2,30}$")
PERF_TYPES = ("bullet", "blitz", "rapid", "classical")
TOKEN_ENV = "LICHESS_TOKEN"
# Lichess API `status` -> the four values used in the training data
STATUS_MAP = {"mate": "mate", "resign": "resign", "outoftime": "outoftime", "draw": "draw", "stalemate": "draw"}


class LichessError(Exception):
    """Raised with a user-readable message."""


def validate_username(username: str) -> str:
    username = (username or "").strip()
    if not USERNAME_RE.fullmatch(username):
        raise LichessError("Invalid username: use 2-30 letters, digits, '_' or '-'.")
    return username


def parse_ndjson(text: str) -> pd.DataFrame:
    """Parse the NDJSON game export. Games without a standard result/clock/moves are skipped, never invented."""
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            g = json.loads(line)
        except json.JSONDecodeError:
            continue
        players = g.get("players", {})
        w, b = players.get("white", {}), players.get("black", {})
        status = STATUS_MAP.get(g.get("status"))
        clock = g.get("clock")
        moves = g.get("moves")
        if not (status and clock and moves and "rating" in w and "rating" in b):
            continue
        rows.append({
            "white_id": str(w.get("user", {}).get("id", "anonymous")).lower(),
            "white_rating": w["rating"],
            "black_id": str(b.get("user", {}).get("id", "anonymous")).lower(),
            "black_rating": b["rating"],
            "winner": g.get("winner", "draw"),
            "victory_status": status,
            "turns": len(moves.split()),            # plies, same definition as the training data
            "increment_code": f"{clock.get('initial', 0) // 60}+{clock.get('increment', 0)}",
            "opening_name": g.get("opening", {}).get("name", "Unknown"),
        })
    return pd.DataFrame(rows, columns=GAME_COLUMNS)


def fetch_games(username: str, perf_type: str = "blitz", max_games: int = 100, session=requests,
                timeout: int = 20, token: str | None = None) -> pd.DataFrame:
    username = validate_username(username)
    if perf_type not in PERF_TYPES:
        raise LichessError(f"perf_type must be one of {PERF_TYPES}.")

    token = (token or os.environ.get(TOKEN_ENV, "")).strip()
    if not token:
        raise LichessError(
            "No Lichess API token configured. Create one at "
            "https://lichess.org/account/oauth/token/create and set LICHESS_TOKEN."
        )

    params = {"max": min(int(max_games), 300), "rated": "true", "perfType": perf_type, "opening": "true", "moves": "true"}
    headers = {"Accept": "application/x-ndjson", "Authorization": f"Bearer {token}"}
    try:
        r = session.get(API_URL.format(username=username), headers=headers, params=params, timeout=timeout)
    except requests.exceptions.Timeout as e:
        raise LichessError("Lichess did not answer in time. Try again later.") from e
    except requests.exceptions.RequestException as e:
        raise LichessError(f"Could not reach Lichess: {e}") from e
    if r.status_code in (401, 403):
        raise LichessError("Lichess rejected the API token (HTTP 401/403). Check LICHESS_TOKEN.")
    if r.status_code == 404:
        raise LichessError(f"User '{username}' does not exist or their games are not publicly exportable.")
    if r.status_code == 429:
        raise LichessError("Lichess rate limit reached (HTTP 429). Wait a minute and retry.")
    if r.status_code != 200:
        raise LichessError(f"Lichess returned HTTP {r.status_code}.")
    games = parse_ndjson(r.text)
    if games.empty:
        raise LichessError(f"No usable rated {perf_type} games found for '{username}'.")
    return games
