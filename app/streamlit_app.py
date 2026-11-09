"""Streamlit demo: place a Lichess player in the segments learned from a small public dataset."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import streamlit as st

from src.config import MIN_GAMES, MODEL_PATH, NUM_FEATURES
from src.features import build_player_profiles
from src.lichess import PERF_TYPES, LichessError, fetch_games, validate_username

st.set_page_config(page_title="Chess player segments", page_icon="♟️", layout="centered")
st.title("♟️ Which segment does a Lichess player fall into?")
st.warning(
    "Exploratory demo. The segments were learned from a small public sample (2013-2017 games, a few hundred players). "
    "They describe how players in that sample differ in rating, game length and how games end. "
    "They are not a skill assessment and give no training advice."
)


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def _token() -> str | None:
    """Read the Lichess token from Streamlit secrets or the environment."""
    try:
        return st.secrets.get("LICHESS_TOKEN") or os.environ.get("LICHESS_TOKEN")
    except Exception:
        return os.environ.get("LICHESS_TOKEN")


@st.cache_data(ttl=1800, show_spinner=False)
def get_games(username: str, perf: str, n: int, token: str) -> pd.DataFrame:
    return fetch_games(username, perf, n, token=token)


bundle = load_bundle()
if bundle is None:
    st.error("Model not found. Run `python run_pipeline.py` first to create `models/segmenter.joblib`.")
    st.stop()

with st.sidebar:
    username = st.text_input("Lichess username", value="")
    perf = st.selectbox("Time control", PERF_TYPES, index=1)
    n_games = st.slider("Recent rated games to use", 20, 300, 100, step=10)
    go = st.button("Analyse", type="primary")

if go:
    token = _token()
    if not token:
        st.error(
            "No Lichess API token configured. Set `LICHESS_TOKEN` or add it to "
            "`.streamlit/secrets.toml` (see README)."
        )
        st.stop()
    try:
        username = validate_username(username)
        with st.spinner("Fetching games from Lichess..."):
            games = get_games(username, perf, n_games, token)
    except LichessError as e:
        st.error(str(e))
        st.stop()

    profile = build_player_profiles(games, min_games=1, players=[username])
    if profile.empty:
        st.error("None of the downloaded games could be attributed to this username.")
        st.stop()
    row = profile.iloc[0]
    if int(row["n_games"]) < MIN_GAMES:
        st.warning(f"Only {int(row['n_games'])} usable games found; at least {MIN_GAMES} are needed for a meaningful profile.")
        st.stop()

    seg = bundle["segmenter"]
    cluster = int(seg.predict(profile)[0])
    info = next(c for c in bundle["clusters"] if c["cluster"] == cluster)
    outcome = next(o for o in bundle["outcomes"] if o["cluster"] == cluster)

    st.success(f"{username}: {int(row['n_games'])} rated {perf} games analysed")
    st.subheader(f"Closest segment: {cluster}")
    st.write(f"**{info['description'].capitalize()}**")

    cmp = pd.DataFrame({"this player": [row[f] for f in NUM_FEATURES], "segment average": [info[f] for f in NUM_FEATURES]}, index=NUM_FEATURES).round(2)
    st.dataframe(cmp)
    st.caption(
        f"In the training sample this segment holds {info['players']} players "
        f"({info['share']:.0%}); their average score was {outcome['score']:.2f} "
        f"(95% interval {outcome['score_ci_low']:.2f} to {outcome['score_ci_high']:.2f})."
    )
    st.caption("Games from the last years differ from the 2013-2017 sample; time-control mix and rating inflation may shift results.")
