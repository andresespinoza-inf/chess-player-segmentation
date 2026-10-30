"""K-Prototypes segmentation with deterministic, interpretable cluster ids."""
from __future__ import annotations

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler

from .config import CAT_FEATURES, NUM_FEATURES, SEED


class PlayerSegmenter:
    """Scales numeric features, runs K-Prototypes and renumbers clusters by ascending mean rating.

    Renumbering makes ids reproducible across library versions (raw K-Prototypes ids are arbitrary).
    """

    def __init__(self, n_clusters: int = 4, seed: int = SEED, num_features=None, cat_features=None):
        self.n_clusters = n_clusters
        self.seed = seed
        self.num_features = list(num_features or NUM_FEATURES)
        self.cat_features = list(cat_features or CAT_FEATURES)
        self.scaler = StandardScaler()
        self.model = None
        self.order_: dict[int, int] = {}
        self.cost_: float = float("nan")

    def _matrix(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        num = self.scaler.fit_transform(df[self.num_features]) if fit else self.scaler.transform(df[self.num_features])
        cat = df[self.cat_features].astype(str).to_numpy()
        return np.hstack([num.astype(object), cat])

    @property
    def cat_idx(self) -> list[int]:
        return list(range(len(self.num_features), len(self.num_features) + len(self.cat_features)))

    def fit(self, df: pd.DataFrame) -> np.ndarray:
        X = self._matrix(df, fit=True)
        self.model = KPrototypes(n_clusters=self.n_clusters, init="Huang", n_init=5, random_state=self.seed, n_jobs=1)
        raw = self.model.fit_predict(X, categorical=self.cat_idx)
        self.cost_ = float(self.model.cost_)
        rating_means = pd.Series(df["avg_rating"].to_numpy()).groupby(raw).mean().sort_values()
        self.order_ = {int(old): new for new, old in enumerate(rating_means.index)}
        return np.array([self.order_[int(r)] for r in raw])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        raw = self.model.predict(self._matrix(df, fit=False), categorical=self.cat_idx)
        return np.array([self.order_[int(r)] for r in raw])

    def mixed_distances(self, df: pd.DataFrame) -> np.ndarray:
        """Pairwise distance matching the K-Prototypes objective: sqrt(squared Euclidean + gamma * mismatches)."""
        X = self._matrix(df, fit=False)
        num = X[:, : len(self.num_features)].astype(float)
        cat = X[:, len(self.num_features):]
        sq = ((num[:, None, :] - num[None, :, :]) ** 2).sum(-1)
        mism = (cat[:, None, :] != cat[None, :, :]).sum(-1)
        return np.sqrt(sq + float(self.model.gamma) * mism)


def describe_clusters(profiles: pd.DataFrame, labels: np.ndarray, num_features=None) -> pd.DataFrame:
    """Per-cluster size, feature means and a neutral descriptive label built from z-scores of the features."""
    num_features = list(num_features or NUM_FEATURES)
    df = profiles.assign(cluster=labels)
    means = df.groupby("cluster")[num_features].mean()
    z = (means - profiles[num_features].mean()) / profiles[num_features].std()
    names = {"avg_rating": ("higher rated", "lower rated"), "avg_turns": ("longer games", "shorter games"),
             "resign_rate": ("games often end by resignation", "games rarely end by resignation"),
             "timeout_rate": ("many games lost on time", "few games lost on time")}
    rows = []
    for c in means.index:
        parts = []
        for f in num_features:
            if abs(z.loc[c, f]) >= 0.5:
                parts.append(names[f][0] if z.loc[c, f] > 0 else names[f][1])
        speed = df.loc[df["cluster"] == c, "fav_speed"].mode().iloc[0]
        label = ", ".join(parts) if parts else "close to the sample average"
        rows.append({"cluster": int(c), "players": int((df["cluster"] == c).sum()),
                     "share": round(float((df["cluster"] == c).mean()), 3),
                     **{f: round(float(means.loc[c, f]), 3) for f in num_features},
                     "most_common_speed": speed, "description": f"{label} (mostly {speed})"})
    return pd.DataFrame(rows)
