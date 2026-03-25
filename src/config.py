# config.py
from pathlib import Path

SEED        = 42
N           = 300
K           = 128
MAX_NEW_TOK = 128
TOP_P       = 0.9
TEMPERATURE = 1.0

MT_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
LM_MODEL = "asi/gpt-fr-cased-small"
SRC_LANG = "en_XX"
TGT_LANG = "fr_XX"

BETAS = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2,
         0.1, 0.2, 0.5, 1.0, 2.0, 5.0,
         10.0, 50.0, 100.0, 1e3, 1e4]

DATASETS = {
    "emea":            "EMEA (Medical)",
    "news_commentary": "News Commentary",
    "opus_books":      "Opus Books (Literary)",
}

SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("models")
RUN_DIR    = Path(f"runs/n{N}_k{K}")
DATA_DIR   = RUN_DIR / "data"
RESULT_DIR = RUN_DIR / "results"
