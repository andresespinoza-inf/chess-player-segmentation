import numpy as np
import pandas as pd
import pytest

from src.features import build_player_profiles, expected_score, speed_class, stack_players


def toy_games():
    # alice: white win by resign (10+0), black loss by outoftime (10+0), white draw (3+2)
    return pd.DataFrame({
        "white_id": ["Alice", "bob", "Alice"], "white_rating": [1600, 1500, 1600],
        "black_id": ["bob", "Alice", "bob"], "black_rating": [1500, 1600, 1600],
        "winner": ["white", "white", "draw"], "victory_status": ["resign", "outoftime", "draw"],
        "turns": [40, 60, 80], "increment_code": ["10+0", "10+0", "3+2"],
        "opening_name": ["Sicilian Defense: Alapin Variation", "Sicilian Defense | Kan", "Italian Game"],
    })


def test_speed_class():
    assert speed_class("1+0") == "bullet"
    assert speed_class("3+2") == "blitz"       # 3*60 + 40*2 = 260
    assert speed_class("10+0") == "rapid"      # 600
    assert speed_class("30+0") == "classical"  # 1800
    assert speed_class("garbage") == "unknown"


def test_expected_score():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1900, 1500) == pytest.approx(1 / (1 + 10 ** (-400 / 400)))


def test_profiles_hand_computed():
    p = build_player_profiles(toy_games(), min_games=3).set_index("player_id")
    a = p.loc["alice"]
    assert a["n_games"] == 3
    assert a["avg_rating"] == pytest.approx(1600)
    assert a["avg_turns"] == pytest.approx(60)
    assert a["win_rate"] == pytest.approx((1 + 0 + 0.5) / 3)           # win, loss, draw
    assert a["resign_rate"] == pytest.approx(1 / 3) and a["timeout_rate"] == pytest.approx(1 / 3)
    assert a["fav_speed"] == "rapid"
    assert a["fav_opening"] == "Sicilian Defense"
    assert "bob" not in p.index or p.loc["bob", "n_games"] >= 3


def test_min_games_filter_and_players_option():
    assert build_player_profiles(toy_games(), min_games=4).empty
    only = build_player_profiles(toy_games(), min_games=1, players=["ALICE"])
    assert list(only["player_id"]) == ["alice"]


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        stack_players(toy_games().drop(columns=["turns"]))


def test_draws_score_half_for_both_sides():
    long = stack_players(toy_games())
    draw_rows = long[long["status"] == "draw"]
    assert (draw_rows["score"] == 0.5).all() and len(draw_rows) == 2
