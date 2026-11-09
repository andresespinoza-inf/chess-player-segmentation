"""Run the full segmentation study: audit -> k selection -> final model -> validation -> tables, figures, model bundle.

Usage:
    python run_pipeline.py                 # all stages (about 6-8 minutes)
    python run_pipeline.py --stage select  # audit + choice of k  -> results/k_selection.csv
    python run_pipeline.py --stage fit     # final model, validation, figures, model bundle (needs the file above)
"""
import argparse
import json
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import adjusted_rand_score

from src import plots as P
from src.clustering import PlayerSegmenter, describe_clusters
from src.config import DATA_PATH, MIN_GAMES, MODEL_PATH, RESULTS_DIR, SEED
from src.evaluation import bootstrap_stability, opening_association, outcomes_by_cluster, silhouette
from src.features import build_player_profiles, stack_players

warnings.filterwarnings("ignore")
K_RANGE = range(2, 8)


def data_audit(games: pd.DataFrame, n_players_all: int, profiles: pd.DataFrame) -> dict:
    ply_per_name = games.groupby("opening_name")["opening_ply"].nunique()
    created = pd.to_datetime(games["created_at"], unit="ms")
    return {
        "games": len(games), "players": n_players_all, "first_game": str(created.min().date()), "last_game": str(created.max().date()),
        "share_casual_games": round(float((~games["rated"].astype(bool)).mean()), 3),
        "share_draws": round(float((games["winner"] == "draw").mean()), 3),
        "players_with_min_games": len(profiles), "min_games": MIN_GAMES,
        "share_opening_names_with_single_opening_ply": round(float((ply_per_name == 1).mean()), 3),
    }


def choose_k(sel: pd.DataFrame) -> int:
    """Rule fixed before looking at results: among k in 3..6 with bootstrap ARI >= 0.8, take the best silhouette."""
    ok = sel[(sel["k"].between(3, 6)) & (sel["ari_mean"] >= 0.8)]
    pool = ok if len(ok) else sel[sel["k"].between(3, 6)]
    return int(pool.sort_values("silhouette", ascending=False).iloc[0]["k"])


def load():
    RESULTS_DIR.mkdir(exist_ok=True)
    games = pd.read_csv(DATA_PATH)
    games_per_player = stack_players(games).groupby("player_id").size()
    profiles = build_player_profiles(games, MIN_GAMES)
    return games, games_per_player, profiles


def stage_select() -> None:
    games, games_per_player, profiles = load()
    audit = data_audit(games, len(games_per_player), profiles)
    print("Audit:", audit)
    (RESULTS_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    P.plot_games_per_player(games_per_player, MIN_GAMES, RESULTS_DIR / "01_games_per_player.png")
    rows = []
    for k in K_RANGE:
        seg = PlayerSegmenter(k)
        lab = seg.fit(profiles)
        stab = bootstrap_stability(profiles, k, lab, n_boot=6)
        rows.append({"k": k, "cost": round(seg.cost_, 2), "silhouette": round(silhouette(seg, profiles, lab), 4),
                     "ari_mean": stab["ari_mean"], "ari_std": stab["ari_std"]})
        print(rows[-1], flush=True)
    sel = pd.DataFrame(rows)
    sel.to_csv(RESULTS_DIR / "k_selection.csv", index=False)
    P.plot_k_selection(sel, RESULTS_DIR / "02_k_selection.png")
    print("Chosen k:", choose_k(sel), flush=True)


def stage_fit() -> None:
    games, _, profiles = load()
    audit = json.loads((RESULTS_DIR / "data_audit.json").read_text())
    sel = pd.read_csv(RESULTS_DIR / "k_selection.csv")
    k = choose_k(sel)

    # --- final model and validation
    seg = PlayerSegmenter(k)
    labels = seg.fit(profiles)
    stability = bootstrap_stability(profiles, k, labels, n_boot=20)
    desc = describe_clusters(profiles, labels)
    outcomes = outcomes_by_cluster(profiles, labels)
    lift, chi = opening_association(profiles, labels)
    desc.to_csv(RESULTS_DIR / "cluster_profiles.csv", index=False)
    outcomes.to_csv(RESULTS_DIR / "cluster_outcomes.csv", index=False)
    lift.to_csv(RESULTS_DIR / "cluster_opening_lift.csv", index=False)
    pd.crosstab(profiles["fav_speed"], labels).to_csv(RESULTS_DIR / "cluster_by_speed.csv")
    P.plot_cluster_heatmap(profiles, labels, RESULTS_DIR / "03_cluster_profiles.png")
    P.plot_pca(profiles, labels, RESULTS_DIR / "04_pca.png")
    P.plot_outcomes(outcomes, RESULTS_DIR / "05_outcomes.png")

    # --- sensitivity to the minimum-games threshold (ARI on players present in both runs)
    sens = []
    for t in (5, 10, 20):
        pr = build_player_profiles(games, t)
        s2 = PlayerSegmenter(k)
        l2 = s2.fit(pr)
        common = profiles["player_id"].isin(pr["player_id"]).to_numpy()
        ref = pd.Series(labels, index=profiles["player_id"])
        new = pd.Series(l2, index=pr["player_id"])
        shared = ref.index[common]
        sens.append({"min_games": t, "players": len(pr), "silhouette": round(silhouette(s2, pr, l2), 4),
                     "ari_vs_reference_on_shared_players": round(float(adjusted_rand_score(ref[shared], new[shared])), 3)})
    pd.DataFrame(sens).to_csv(RESULTS_DIR / "threshold_sensitivity.csv", index=False)

    summary = {"audit": audit, "k": k, "silhouette": float(sel.loc[sel.k == k, "silhouette"].iloc[0]),
               "bootstrap_stability": {kk: v for kk, v in stability.items() if kk != "scores"},
               "opening_association": chi, "threshold_sensitivity": sens}
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"segmenter": seg, "clusters": desc.to_dict("records"), "outcomes": outcomes.to_dict("records"),
                 "meta": {"k": k, "players": len(profiles), "min_games": MIN_GAMES, "python": platform.python_version(),
                          "scikit_learn": sklearn.__version__, "seed": SEED}}, MODEL_PATH, compress=3)
    print("Done.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["all", "select", "fit"], default="all")
    stage = ap.parse_args().stage
    if stage in ("all", "select"):
        stage_select()
    if stage in ("all", "fit"):
        stage_fit()


if __name__ == "__main__":
    main()
