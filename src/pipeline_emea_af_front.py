# pipeline_af_front.py
#
# Constructs Accuracy-Fluency Pareto fronts for multiple en-fr datasets
# via oracle reranking over K sampled candidates, replicating Flamich et al.
#
# Stages (each checkpointed per dataset):
#   1. Load & sample eval set      -> data/{name}_eval.csv
#   2. Generate K candidates       -> data/{name}_candidates.jsonl
#   3. Score adequacy (chrF)       -> data/{name}_scored_adq.jsonl
#   4. Score fluency (log-ppl)     -> data/{name}_scored_flu.jsonl
#   5. Oracle rerank across beta   -> results/{name}_af_front.csv
#   6. Per-dataset plots           -> results/{name}_af_front.png
#   7. Combined plot               -> results/combined_af_front.png

import random
import urllib.request
import zipfile
import io
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import torch
from datasets import load_dataset
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import sacrebleu
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────

SEED        = 42
EVAL_SIZE   = 16
K           = 4
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

DATA_DIR   = Path("data");    DATA_DIR.mkdir(exist_ok=True)
RESULT_DIR = Path("results"); RESULT_DIR.mkdir(exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Dataset registry ──────────────────────────────────────────────────────────
# To add a new dataset, add an entry here with a `load_fn` that returns
# a list of {"src_en": ..., "ref_fr": ...} dicts.

def load_emea(n: int) -> list[dict]:
    url = "https://object.pouta.csc.fi/OPUS-EMEA/v3/moses/en-fr.txt.zip"
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
        en_lines = z.read("EMEA.en-fr.en").decode().splitlines()
        fr_lines = z.read("EMEA.en-fr.fr").decode().splitlines()
    paired = list(zip(en_lines, fr_lines))
    random.shuffle(paired)
    return [{"src_en": e, "ref_fr": f} for e, f in paired[:n]]

def load_news_commentary(n: int) -> list[dict]:
    ds = load_dataset("Helsinki-NLP/news_commentary", "en-fr")
    split = list(ds.keys())[0]
    data  = ds[split].shuffle(seed=SEED).select(range(min(n, len(ds[split]))))
    return [{"src_en": data[i]["translation"]["en"],
             "ref_fr": data[i]["translation"]["fr"]} for i in range(len(data))]

def load_opus_books(n: int) -> list[dict]:
    ds    = load_dataset("Helsinki-NLP/opus_books", "en-fr")
    split = list(ds.keys())[0]
    data  = ds[split].shuffle(seed=SEED).select(range(min(n, len(ds[split]))))
    return [{"src_en": data[i]["translation"]["en"],
             "ref_fr": data[i]["translation"]["fr"]} for i in range(len(data))]

DATASETS = {
    "emea":             {"load_fn": load_emea,             "label": "EMEA (Medical)"},
    "news_commentary":  {"load_fn": load_news_commentary,  "label": "News Commentary"},
    "opus_books":       {"load_fn": load_opus_books,       "label": "Opus Books (Literary)"},
}

# ── Stage functions ───────────────────────────────────────────────────────────

def stage1_load(name: str, cfg: dict) -> pd.DataFrame:
    path = DATA_DIR / f"{name}_eval.csv"
    if path.exists():
        print(f"[{name}][Stage 1] Skipping — found {path}")
        return pd.read_csv(path)
    print(f"[{name}][Stage 1] Loading dataset...")
    rows = cfg["load_fn"](EVAL_SIZE)
    if not rows:
        raise RuntimeError(f"[{name}] load_fn returned empty/None — check loader.")
    df   = pd.DataFrame([{"id": i, **r} for i, r in enumerate(rows)])
    df.to_csv(path, index=False)
    print(f"[{name}][Stage 1] Saved {len(df)} sentences.")
    return df


def stage2_generate(name: str, eval_df: pd.DataFrame,
                    mt_model, mt_tokenizer) -> pd.DataFrame:
    path = DATA_DIR / f"{name}_candidates.jsonl"
    if path.exists():
        print(f"[{name}][Stage 2] Skipping — found {path}")
        return pd.read_json(path, orient="records", lines=True)
    forced_bos = mt_tokenizer.lang_code_to_id[TGT_LANG]

    def sample_k(src: str) -> list[str]:
        inputs = mt_tokenizer(src, return_tensors="pt",
                              truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = mt_model.generate(
                **inputs, do_sample=True, top_p=TOP_P, temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOK, forced_bos_token_id=forced_bos,
                num_return_sequences=K,
            )
        return [mt_tokenizer.decode(o, skip_special_tokens=True) for o in out]

    rows = []
    for row in tqdm(eval_df.itertuples(index=False), total=len(eval_df),
                    desc=f"[{name}] Generating"):
        for kid, hyp in enumerate(sample_k(row.src_en)):
            rows.append({"id": int(row.id), "cand_id": kid,
                         "src_en": row.src_en, "ref_fr": row.ref_fr, "hyp_fr": hyp})
    df = pd.DataFrame(rows)
    df.to_json(path, orient="records", lines=True, force_ascii=False)
    print(f"[{name}][Stage 2] Saved {len(df)} candidates.")
    return df


def stage3_score_adequacy(name: str, cand_df: pd.DataFrame) -> pd.DataFrame:
    path = DATA_DIR / f"{name}_scored_adq.jsonl"
    if path.exists():
        print(f"[{name}][Stage 3] Skipping — found {path}")
        return pd.read_json(path, orient="records", lines=True)
    chrf_metric = sacrebleu.CHRF()
    cand_df["chrf"] = [
        chrf_metric.sentence_score(row.hyp_fr, [row.ref_fr]).score
        for row in tqdm(cand_df.itertuples(index=False), total=len(cand_df),
                        desc=f"[{name}] chrF")
    ]
    cand_df.to_json(path, orient="records", lines=True, force_ascii=False)
    print(f"[{name}][Stage 3] Saved adequacy scores.")
    return cand_df


def stage4_score_fluency(name: str, chrf_df: pd.DataFrame,
                         lm_model, lm_tokenizer) -> pd.DataFrame:
    path = DATA_DIR / f"{name}_scored_flu.jsonl"
    if path.exists():
        print(f"[{name}][Stage 4] Skipping — found {path}")
        return pd.read_json(path, orient="records", lines=True)

    def log_ppl(text: str) -> float:
        enc = lm_tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=256).to(device)
        with torch.no_grad():
            return lm_model(**enc, labels=enc["input_ids"]).loss.item()

    chrf_df["log_ppl"] = [
        log_ppl(row.hyp_fr)
        for row in tqdm(chrf_df.itertuples(index=False), total=len(chrf_df),
                        desc=f"[{name}] Fluency")
    ]
    chrf_df["fluency"] = -chrf_df["log_ppl"]
    chrf_df.to_json(path, orient="records", lines=True, force_ascii=False)
    print(f"[{name}][Stage 4] Saved fluency scores.")
    return chrf_df


def stage5_rerank(name: str, scored_df: pd.DataFrame) -> pd.DataFrame:
    path = RESULT_DIR / f"{name}_af_front.csv"
    if path.exists():
        print(f"[{name}][Stage 5] Skipping — found {path}")
        return pd.read_csv(path)
    g = scored_df.groupby("id")
    scored_df["chrf_norm"]    = (scored_df["chrf"]    - g["chrf"].transform("min"))    / (g["chrf"].transform("max")    - g["chrf"].transform("min")    + 1e-9)
    scored_df["fluency_norm"] = (scored_df["fluency"] - g["fluency"].transform("min")) / (g["fluency"].transform("max") - g["fluency"].transform("min") + 1e-9)
    rows = []
    for beta in tqdm(BETAS, desc=f"[{name}] Sweeping beta"):
        scored_df["score"] = beta * scored_df["chrf_norm"] + scored_df["fluency_norm"]
        best = scored_df.loc[scored_df.groupby("id")["score"].idxmax()]
        rows.append({"beta": beta, "chrf": best["chrf"].mean(), "fluency": best["fluency"].mean()})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"[{name}][Stage 5] Saved front.")
    return df


def plot_single(name: str, label: str,
                front_df: pd.DataFrame, scored_df: pd.DataFrame):
    path = RESULT_DIR / f"{name}_af_front.png"
    if path.exists():
        print(f"[{name}][Stage 6] Skipping — found {path}")
        return
    baseline_chrf    = scored_df.groupby("id")["chrf"].mean().mean()
    baseline_fluency = scored_df.groupby("id")["fluency"].mean().mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(front_df["chrf"], front_df["fluency"],
            marker="o", linewidth=2.5, markersize=7, label=f"Oracle front")
    ax.scatter([baseline_chrf], [baseline_fluency],
               marker="x", s=120, linewidths=2.5, zorder=5, label="Mean candidate baseline")
    ax.set_xlabel("Adequacy (chrF)")
    ax.set_ylabel("Fluency (neg-NLL)")
    ax.set_title(f"A-F Oracle Front — {label}\nOracle reranking over K={K} candidates")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"[{name}][Stage 6] Saved plot.")


def plot_combined(results: dict):
    path = RESULT_DIR / "combined_af_front.png"
    if path.exists():
        print("[Combined] Skipping — found combined plot.")
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, (front_df, scored_df, label) in results.items():
        baseline_chrf    = scored_df.groupby("id")["chrf"].mean().mean()
        baseline_fluency = scored_df.groupby("id")["fluency"].mean().mean()
        line, = ax.plot(front_df["chrf"], front_df["fluency"],
                        marker="o", linewidth=2.5, markersize=7, label=label)
        ax.scatter([baseline_chrf], [baseline_fluency], marker="x",
                   s=120, linewidths=2.5, color=line.get_color(), zorder=5)
    ax.set_xlabel("Adequacy (chrF)")
    ax.set_ylabel("Fluency (neg-NLL)")
    ax.set_title(f"A-F Oracle Fronts — All Domains\n× = mean candidate baseline per domain")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()
    print("[Combined] Saved combined plot.")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load MT model once, run stage 2 for all datasets, then free
    mt_needed = any(
        not (DATA_DIR / f"{name}_candidates.jsonl").exists()
        for name in DATASETS
    )
    if mt_needed:
        print(f"Loading MT model: {MT_MODEL}")
        mt_tokenizer = MBart50TokenizerFast.from_pretrained(MT_MODEL)
        mt_tokenizer.src_lang = SRC_LANG
        mt_model = MBartForConditionalGeneration.from_pretrained(MT_MODEL).to(device)
        mt_model.eval()
    else:
        mt_model = mt_tokenizer = None

    eval_dfs = {}
    cand_dfs = {}
    for name, cfg in DATASETS.items():
        eval_dfs[name] = stage1_load(name, cfg)
        cand_dfs[name] = stage2_generate(name, eval_dfs[name], mt_model, mt_tokenizer)

    if mt_model is not None:
        del mt_model, mt_tokenizer; torch.cuda.empty_cache()

    # Stage 3: chrF — no model needed
    chrf_dfs = {name: stage3_score_adequacy(name, cand_dfs[name]) for name in DATASETS}

    # Load LM once, run stage 4 for all datasets, then free
    lm_needed = any(
        not (DATA_DIR / f"{name}_scored_flu.jsonl").exists()
        for name in DATASETS
    )
    if lm_needed:
        print(f"Loading LM: {LM_MODEL}")
        lm_tokenizer = AutoTokenizer.from_pretrained(LM_MODEL)
        lm_model     = AutoModelForCausalLM.from_pretrained(LM_MODEL).to(device)
        lm_model.eval()
    else:
        lm_model = lm_tokenizer = None

    scored_dfs = {name: stage4_score_fluency(name, chrf_dfs[name], lm_model, lm_tokenizer)
                  for name in DATASETS}

    if lm_model is not None:
        del lm_model, lm_tokenizer; torch.cuda.empty_cache()

    # Stages 5 & 6
    results = {}
    for name, cfg in DATASETS.items():
        front_df = stage5_rerank(name, scored_dfs[name])
        plot_single(name, cfg["label"], front_df, scored_dfs[name])
        results[name] = (front_df, scored_dfs[name], cfg["label"])

    plot_combined(results)
    print("Done.")

if __name__ == "__main__":
    main()
