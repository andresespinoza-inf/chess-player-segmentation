from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "games.csv"
RESULTS_DIR = ROOT / "results"
MODEL_PATH = ROOT / "models" / "segmenter.joblib"

SEED = 42
MIN_GAMES = 10          # a player needs at least this many games to be profiled
NUM_FEATURES = ["avg_rating", "avg_turns", "resign_rate", "timeout_rate"]
CAT_FEATURES = ["fav_speed"]
