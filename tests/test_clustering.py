import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.clustering import PlayerSegmenter, describe_clusters


def synthetic(n=60, seed=0):
    rng = np.random.default_rng(seed)
    low = pd.DataFrame({"avg_rating": rng.normal(1300, 40, n), "avg_turns": rng.normal(45, 4, n),
                        "resign_rate": rng.normal(0.8, 0.03, n), "timeout_rate": rng.normal(0.05, 0.01, n), "fav_speed": "blitz"})
    high = pd.DataFrame({"avg_rating": rng.normal(2000, 40, n), "avg_turns": rng.normal(80, 4, n),
                         "resign_rate": rng.normal(0.4, 0.03, n), "timeout_rate": rng.normal(0.2, 0.01, n), "fav_speed": "rapid"})
    df = pd.concat([high, low], ignore_index=True)  # high-rated players come first on purpose
    truth = np.array([1] * n + [0] * n)
    return df, truth


def test_recovers_separated_groups_and_orders_by_rating():
    df, truth = synthetic()
    seg = PlayerSegmenter(2)
    labels = seg.fit(df)
    assert adjusted_rand_score(truth, labels) == 1.0
    assert (labels == truth).all()            # cluster 0 is the lower-rated group


def test_predict_matches_fit_and_is_deterministic():
    df, _ = synthetic()
    a, b = PlayerSegmenter(2, seed=1), PlayerSegmenter(2, seed=1)
    la, lb = a.fit(df), b.fit(df)
    assert (la == lb).all()
    assert (a.predict(df) == la).all()


def test_mixed_distance_properties():
    df, _ = synthetic(n=10)
    seg = PlayerSegmenter(2)
    seg.fit(df)
    d = seg.mixed_distances(df)
    assert np.allclose(d, d.T) and np.allclose(np.diag(d), 0)


def test_describe_clusters_labels_are_neutral_and_sized():
    df, truth = synthetic()
    desc = describe_clusters(df, truth)
    assert desc["players"].tolist() == [60, 60]
    assert "higher rated" in desc.loc[desc.cluster == 1, "description"].iloc[0]
    assert "lower rated" in desc.loc[desc.cluster == 0, "description"].iloc[0]
