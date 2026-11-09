import numpy as np
import pandas as pd
import pytest

from src.evaluation import opening_association, outcomes_by_cluster, wilson_interval


def test_wilson_known_values():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.404, abs=1e-3) and hi == pytest.approx(0.596, abs=1e-3)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0 or lo == pytest.approx(0, abs=1e-9)
    assert hi == pytest.approx(0.278, abs=1e-3)
    assert np.isnan(wilson_interval(0, 0)[0])


def test_outcomes_are_weighted_by_games_not_players():
    prof = pd.DataFrame({"n_games": [10, 90], "win_rate": [1.0, 0.0], "score_over_expected": [0.5, -0.1]})
    out = outcomes_by_cluster(prof, np.array([0, 0])).iloc[0]
    assert out["games"] == 100 and out["score"] == pytest.approx(0.1)
    assert out["score_over_expected"] == pytest.approx((0.5 * 10 - 0.1 * 90) / 100)


def test_opening_lift():
    prof = pd.DataFrame({"fav_opening": ["A"] * 20 + ["B"] * 20 + ["A"] * 10 + ["B"] * 10})
    labels = np.array([0] * 40 + [1] * 20)
    lift, chi = opening_association(prof, labels, top_n=2, min_count=5)
    assert lift["lift"].round(2).tolist().count(1.0) >= 2      # openings split evenly -> lift 1
    assert 0 <= chi["p_value"] <= 1
