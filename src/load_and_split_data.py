# split_data.py
#
# One-time script to load and split all datasets into fixed train/val/dev/test splits.
# Run this ONCE before anything else. Do not re-run — splits must stay fixed.
#
# Split strategy:
#   - train/val scale with corpus size (proportional)
#   - dev/test are capped at DEV_TEST_CAP for consistent cross-dataset eval
#   - remainder after capping goes to train
#
# Output:
#   data/splits/{name}_train.csv   — for fine-tuning
#   data/splits/{name}_val.csv     — for early stopping
#   data/splits/{name}_dev.csv     — for Pareto front dev work (run many times)
#   data/splits/{name}_test.csv    — SEALED: only use for final dissertation numbers

import random
import urllib.request
import zipfile
import io
import pandas as pd
from pathlib import Path
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────

SEED = 42

SPLIT_RATIOS = {
    "train": 0.80,
    "val":   0.08,
    "dev":   0.06,
    "test":  0.06,
}

DEV_TEST_CAP = 300   # cap dev and test at this many sentences for consistent eval

SPLITS_DIR = Path("data/splits")
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_emea_raw() -> list[tuple[str, str]]:
    url = "https://object.pouta.csc.fi/OPUS-EMEA/v3/moses/en-fr.txt.zip"
    print("  Downloading EMEA...")
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
        en_lines = z.read("EMEA.en-fr.en").decode().splitlines()
        fr_lines = z.read("EMEA.en-fr.fr").decode().splitlines()
    data = list(zip(en_lines, fr_lines))
    print("EMEA LOADED SIZE", len(data))
    return data

def load_news_commentary_raw() -> list[tuple[str, str]]:
    print("  Loading News Commentary...")
    ds    = load_dataset("Helsinki-NLP/news_commentary", "en-fr")
    split = list(ds.keys())[0]
    data  = ds[split]
    data = [(data[i]["translation"]["en"], data[i]["translation"]["fr"])
            for i in range(len(data))]
    print("NEWS COMMENTARY LOADED SIZE", len(data))
    return data

def load_opus_books_raw() -> list[tuple[str, str]]:
    print("  Loading Opus Books...")
    ds    = load_dataset("Helsinki-NLP/opus_books", "en-fr")
    split = list(ds.keys())[0]
    data  = ds[split]
    data = [(data[i]["translation"]["en"], data[i]["translation"]["fr"])
            for i in range(len(data))]
    print("OPUS BOOKS LOADED SIZE", len(data))
    return data

DATASETS = {
    "emea":            load_emea_raw,
    "news_commentary": load_news_commentary_raw,
    "opus_books":      load_opus_books_raw,
}

# ── Split logic ───────────────────────────────────────────────────────────────

def compute_sizes(n: int) -> dict[str, int]:
    sizes = {k: int(v * n) for k, v in SPLIT_RATIOS.items()}

    # Cap dev and test, redistribute surplus to train
    for split_name in ("dev", "test"):
        if sizes[split_name] > DEV_TEST_CAP:
            sizes["train"] += sizes[split_name] - DEV_TEST_CAP
            sizes[split_name] = DEV_TEST_CAP

    # Absorb any rounding remainder into train
    sizes["train"] += n - sum(sizes.values())

    assert sum(sizes.values()) == n, "Split sizes don't sum to n — bug in compute_sizes"
    return sizes

# ── Split & save ──────────────────────────────────────────────────────────────

def split_and_save(name: str, load_fn):
    existing = [SPLITS_DIR / f"{name}_{s}.csv" for s in SPLIT_RATIOS]

    if all(p.exists() for p in existing):
        print(f"[{name}] All splits exist — skipping. Delete to re-split.")
        return
    if any(p.exists() for p in existing):
        raise RuntimeError(
            f"[{name}] Partial splits found. Delete all {name}_*.csv in "
            f"{SPLITS_DIR} and re-run to avoid inconsistent splits."
        )

    pairs = load_fn()
    pairs = [(e.strip(), f.strip()) for e, f in pairs if e.strip() and f.strip()]
    print(f"  {len(pairs)} valid pairs after filtering.")

    min_required = DEV_TEST_CAP * 2 + 100   # at least 2×cap for dev+test plus some train
    if len(pairs) < min_required:
        raise RuntimeError(
            f"[{name}] Only {len(pairs)} pairs — need at least {min_required}."
        )

    random.shuffle(pairs)

    sizes = compute_sizes(len(pairs))
    print(f"  Split sizes: { {k: v for k, v in sizes.items()} }")

    cursor = 0
    for split_name, size in sizes.items():
        chunk  = pairs[cursor:cursor + size]
        cursor += size
        df = pd.DataFrame([{"id": i, "src_en": e, "ref_fr": f}
                            for i, (e, f) in enumerate(chunk)])
        path = SPLITS_DIR / f"{name}_{split_name}.csv"
        df.to_csv(path, index=False)
        print(f"  [{split_name:>5}] {len(df):>6} pairs → {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Splits directory : {SPLITS_DIR}")
    print(f"Split ratios     : {SPLIT_RATIOS}")
    print(f"Dev/test cap     : {DEV_TEST_CAP}")
    print(f"Seed             : {SEED}")
    print()
    print("WARNING: Do not re-run once experiments have started.")
    print("         Re-splitting invalidates all existing results.\n")

    for name, load_fn in DATASETS.items():
        print(f"[{name}]")
        split_and_save(name, load_fn)
        print()

    print("All splits saved.")
    print("→ Commit data/splits/ to git now for reproducibility.")
    print("→ REMINDER: test splits are SEALED — do not use until final evaluation.")

if __name__ == "__main__":
    main()
