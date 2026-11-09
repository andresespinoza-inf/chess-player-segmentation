"""Figures for the segmentation study."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from .config import NUM_FEATURES

sns.set_theme(style="whitegrid")
DPI = 200


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_games_per_player(n_games_all: pd.Series, min_games: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(n_games_all.clip(upper=60), bins=60, color="steelblue")
    ax.axvline(min_games, color="red", ls="--", label=f"minimum for profiling ({min_games})")
    ax.set(title="Games per player in the dataset (clipped at 60)", xlabel="Games", ylabel="Players", yscale="log")
    ax.legend()
    _save(fig, out)


def plot_k_selection(sel: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(sel["k"], sel["cost"], "o-"); ax[0].set(title="K-Prototypes cost", xlabel="k")
    ax[1].plot(sel["k"], sel["silhouette"], "o-", color="darkgreen"); ax[1].set(title="Silhouette (same distance as the objective)", xlabel="k")
    ax[2].errorbar(sel["k"], sel["ari_mean"], yerr=sel["ari_std"], fmt="o-", color="purple", capsize=3)
    ax[2].set(title="Bootstrap stability (ARI, 80% subsamples)", xlabel="k", ylim=(0, 1.02))
    _save(fig, out)


def plot_cluster_heatmap(profiles: pd.DataFrame, labels: np.ndarray, out: Path) -> None:
    z = (profiles.assign(cluster=labels).groupby("cluster")[NUM_FEATURES].mean() - profiles[NUM_FEATURES].mean()) / profiles[NUM_FEATURES].std()
    fig, ax = plt.subplots(figsize=(7, 0.8 * len(z) + 1.5))
    sns.heatmap(z, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "z-score vs all players"})
    ax.set(title="Cluster profiles on the clustering features")
    _save(fig, out)


def plot_pca(profiles: pd.DataFrame, labels: np.ndarray, out: Path) -> None:
    X = pd.get_dummies(profiles[NUM_FEATURES + ["fav_speed"]], columns=["fav_speed"]).astype(float)
    X = (X - X.mean()) / X.std().replace(0, 1)
    p = PCA(n_components=2, random_state=0).fit(X)
    xy = p.transform(X)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=labels, palette="tab10", s=18, ax=ax)
    ax.set(title="2-D PCA projection (visualisation only, not a validation)",
           xlabel=f"PC1 ({p.explained_variance_ratio_[0]:.0%} of variance)", ylabel=f"PC2 ({p.explained_variance_ratio_[1]:.0%})")
    ax.legend(title="cluster")
    _save(fig, out)


def plot_outcomes(out_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(out_df["cluster"], out_df["score"], yerr=[out_df["score"] - out_df["score_ci_low"], out_df["score_ci_high"] - out_df["score"]],
                fmt="o", capsize=4, color="black")
    ax.axhline(0.5, color="red", ls="--")
    ax.set(title="Score per cluster (95% Wilson interval), not used to form clusters", xlabel="cluster", ylabel="Average score")
    ax.set_xticks(out_df["cluster"])
    _save(fig, out)
