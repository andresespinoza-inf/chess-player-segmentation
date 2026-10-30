"""Validation and post-hoc description of clusters."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .clustering import PlayerSegmenter
from .config import SEED


def wilson_interval(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (draws counted as half wins are approximated by the score total)."""
    if n == 0:
        return (np.nan, np.nan)
    p = wins / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (centre - half, centre + half)


def silhouette(seg: PlayerSegmenter, df: pd.DataFrame, labels: np.ndarray) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(silhouette_score(seg.mixed_distances(df), labels, metric="precomputed"))


def bootstrap_stability(df: pd.DataFrame, k: int, reference: np.ndarray, n_boot: int = 20, frac: float = 0.8) -> dict:
    """Fit on random 80% subsamples, assign ALL players, compare with the reference labelling (ARI)."""
    rng = np.random.default_rng(SEED)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(len(df), int(frac * len(df)), replace=False)
        seg = PlayerSegmenter(k, seed=int(rng.integers(1_000_000)))
        seg.fit(df.iloc[idx])
        scores.append(adjusted_rand_score(reference, seg.predict(df)))
    s = np.array(scores)
    return {"n_boot": n_boot, "ari_mean": round(float(s.mean()), 3), "ari_std": round(float(s.std()), 3),
            "ari_min": round(float(s.min()), 3), "scores": s.round(4).tolist()}


def outcomes_by_cluster(profiles: pd.DataFrame, labels: np.ndarray, games_col: str = "n_games") -> pd.DataFrame:
    """Win rate (score) and score over Elo expectation per cluster, computed on games, not on player averages."""
    df = profiles.assign(cluster=labels)
    rows = []
    for c, g in df.groupby("cluster"):
        n = int(g[games_col].sum())
        score_sum = float((g["win_rate"] * g[games_col]).sum())
        lo, hi = wilson_interval(score_sum, n)
        soe = float((g["score_over_expected"] * g[games_col]).sum() / n)
        rows.append({"cluster": int(c), "players": len(g), "games": n, "score": round(score_sum / n, 3),
                     "score_ci_low": round(lo, 3), "score_ci_high": round(hi, 3), "score_over_expected": round(soe, 3),
                     "median_games_per_player": float(g[games_col].median())})
    return pd.DataFrame(rows)


def opening_association(profiles: pd.DataFrame, labels: np.ndarray, top_n: int = 8, min_count: int = 10) -> tuple[pd.DataFrame, dict]:
    """Lift of each favourite opening in each cluster (openings were NOT used to form clusters)."""
    df = profiles.assign(cluster=labels)
    top = df["fav_opening"].value_counts().head(top_n).index
    df["opening_grp"] = np.where(df["fav_opening"].isin(top), df["fav_opening"], "Other")
    ct = pd.crosstab(df["cluster"], df["opening_grp"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    rows = []
    n = len(df)
    for c in ct.index:
        for o in ct.columns:
            cnt = int(ct.loc[c, o])
            if o == "Other" or cnt < min_count:
                continue
            p_o_given_c = cnt / ct.loc[c].sum()
            p_o = ct[o].sum() / n
            rows.append({"cluster": int(c), "favourite_opening": o, "players": cnt, "share_in_cluster": round(p_o_given_c, 3),
                         "share_overall": round(p_o, 3), "lift": round(p_o_given_c / p_o, 2)})
    out = pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)
    return out, {"chi2": round(float(chi2), 2), "dof": int(dof), "p_value": float(p), "min_expected_count": round(float(stats.contingency.expected_freq(ct).min()), 2)}
