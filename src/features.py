"""Turn game-level rows into player-level profiles.

Clustering inputs describe *who plays and how games end* (rating, game length, resign/timeout rates, preferred speed).
Outcomes (win rate, score over expectation) and openings are computed here too but are NOT used to build clusters:
they are used afterwards to describe them, which avoids circular conclusions.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Lichess speed classes by estimated duration = base + 40 * increment (seconds).
SPEED_BOUNDS = [(30, "ultrabullet"), (180, "bullet"), (480, "blitz"), (1500, "rapid")]
GAME_COLUMNS = ["white_id", "white_rating", "black_id", "black_rating", "winner",
                "victory_status", "turns", "increment_code", "opening_name"]


def speed_class(increment_code: str) -> str:
    """'10+0' -> 'rapid', '3+2' -> 'blitz'. Unparseable codes -> 'unknown'."""
    m = re.fullmatch(r"(\d+)\+(\d+)", str(increment_code))
    if not m:
        return "unknown"
    est = int(m.group(1)) * 60 + 40 * int(m.group(2))
    for bound, name in SPEED_BOUNDS:
        if est < bound:
            return name
    return "classical"


def expected_score(own, opp):
    return 1.0 / (1.0 + 10.0 ** ((np.asarray(opp, float) - np.asarray(own, float)) / 400.0))


def stack_players(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, player). Draws score 0.5."""
    missing = set(GAME_COLUMNS) - set(games.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    parts = []
    for side, other in (("white", "black"), ("black", "white")):
        d = pd.DataFrame({
            "player_id": games[f"{side}_id"].astype(str).str.lower(),
            "rating": games[f"{side}_rating"].astype(float),
            "opp_rating": games[f"{other}_rating"].astype(float),
            "score": np.where(games["winner"] == side, 1.0, np.where(games["winner"] == "draw", 0.5, 0.0)),
            "turns": games["turns"].astype(float),
            "status": games["victory_status"],
            "speed": games["increment_code"].map(speed_class),
            "base_opening": games["opening_name"].astype(str).str.split(":").str[0].str.split("|").str[0].str.strip(),
        })
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out["expected"] = expected_score(out["rating"], out["opp_rating"])
    return out


def _mode(s: pd.Series):
    m = s.mode()
    return m.iloc[0] if not m.empty else np.nan


def build_player_profiles(games: pd.DataFrame, min_games: int = 10, players: list | None = None) -> pd.DataFrame:
    """Aggregate to one row per player with at least `min_games` games (optionally only for `players`)."""
    long = stack_players(games)
    if players is not None:
        long = long[long["player_id"].isin([str(p).lower() for p in players])]
    g = long.groupby("player_id")
    prof = pd.DataFrame({
        "n_games": g.size(),
        "avg_rating": g["rating"].mean(),
        "avg_turns": g["turns"].mean(),
        "resign_rate": g["status"].apply(lambda s: (s == "resign").mean()),
        "timeout_rate": g["status"].apply(lambda s: (s == "outoftime").mean()),
        "fav_speed": g["speed"].apply(_mode),
        # outcome / context variables: never used to form clusters
        "win_rate": g["score"].mean(),
        "score_over_expected": (long["score"] - long["expected"]).groupby(long["player_id"]).mean(),
        "fav_opening": g["base_opening"].apply(_mode),
    })
    prof = prof[prof["n_games"] >= min_games].reset_index()
    return prof
