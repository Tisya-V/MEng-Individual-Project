import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import (
    MBartForConditionalGeneration, MBart50TokenizerFast,
    AutoTokenizer, AutoModelForCausalLM,
)
from config import *
from af_front_pipeline_stages import (
    generate_candidates, generate_greedy,
    score_adequacy, score_fluency, score_greedy, rerank,
)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Run directory: {RUN_DIR}")

DATASETS = {
    "emea":            {"label": "EMEA (Medical)"},
    "news_commentary": {"label": "News Commentary"},
    "opus_books":      {"label": "Opus Books (Literary)"},
}


def write_config():
    cfg_path = RUN_DIR / "config.txt"
    if cfg_path.exists():
        return
    lines = [
        f"SEED        = {SEED}",
        f"N           = {N}",
        f"K           = {K}",
        f"MAX_NEW_TOK = {MAX_NEW_TOK}",
        f"TOP_P       = {TOP_P}",
        f"TEMPERATURE = {TEMPERATURE}",
        f"",
        f"MT_MODEL    = {MT_MODEL}",
        f"LM_MODEL    = {LM_MODEL}",
        f"SRC_LANG    = {SRC_LANG}",
        f"TGT_LANG    = {TGT_LANG}",
        f"",
        f"BETAS       = {BETAS}",
    ]
    cfg_path.write_text("\n".join(lines))
    print(f"Config saved to {cfg_path}")


def plot_single(name, label, front_df, baseline):
    path = RESULT_DIR / f"{name}_af_front.png"
    if path.exists():
        print(f"[{name}][Stage 6] Skipping — found {path}")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(front_df["chrf"], front_df["fluency"], linewidth=2.5, markersize=7, label="Oracle front")
    ax.scatter([baseline["chrf"]], [baseline["fluency"]],
               marker="x", s=120, linewidths=2.5, zorder=5,
               label="Beam decode (baseline)")
    ax.set_xlabel("Adequacy (chrF)")
    ax.set_ylabel("Fluency (neg-NLL)")
    ax.set_title(f"A-F Oracle Front — {label}\nN={N}, K={K}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"[{name}][Stage 6] Saved plot.")


def plot_combined(results):
    import matplotlib.cm as cm
    path = RESULT_DIR / "combined_af_front.png"
    if path.exists():
        print("[Combined] Skipping — found combined plot.")
        return
    colours = cm.tab10.colors
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (name, (front_df, baseline, label)) in enumerate(results.items()):
        c = colours[i]
        line, = ax.plot(front_df["chrf"], front_df["fluency"],
                        color=c, linewidth=2.5, markersize=7, label=label)
        ax.scatter([baseline["chrf"]], [baseline["fluency"]],
                   marker="x", s=120, linewidths=2.5, color=c, zorder=5)
    ax.set_xlabel("Adequacy (chrF)")
    ax.set_ylabel("Fluency (neg-NLL)")
    ax.set_title(f"A-F Oracle Fronts — All Domains (N={N}, K={K})\nx = beam decode baseline per domain")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()
    print("[Combined] Saved combined plot.")


def main():
    for d in [RUN_DIR, DATA_DIR, RESULT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    write_config()

    mt_needed = any(
        not (DATA_DIR / f"{name}_candidates.jsonl").exists()
        or not (DATA_DIR / f"{name}_greedy.jsonl").exists()
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
    greedy_dfs = {}
    for name, cfg in DATASETS.items():
        eval_dfs[name]   = pd.read_csv(SPLITS_DIR / f"{name}_dev.csv")
        cand_dfs[name]   = generate_candidates(name, eval_dfs[name], mt_model, mt_tokenizer, DATA_DIR)
        greedy_dfs[name] = generate_greedy(name, eval_dfs[name], mt_model, mt_tokenizer, DATA_DIR)

    if mt_model is not None:
        del mt_model, mt_tokenizer; torch.cuda.empty_cache()

    lm_needed = any(
        not (DATA_DIR / f"{name}_scored_flu.jsonl").exists()
        or not (DATA_DIR / f"{name}_greedy_scored.jsonl").exists()
        for name in DATASETS
    )
    if lm_needed:
        print(f"Loading LM: {LM_MODEL}")
        lm_tokenizer = AutoTokenizer.from_pretrained(LM_MODEL)
        lm_model     = AutoModelForCausalLM.from_pretrained(LM_MODEL).to(device)
        lm_model.eval()
    else:
        lm_model = lm_tokenizer = None

    scored_dfs       = {name: score_fluency(name, score_adequacy(name, cand_dfs[name], DATA_DIR), DATA_DIR, lm_model, lm_tokenizer) for name in DATASETS}
    greedy_baselines = {name: score_greedy(name, greedy_dfs[name], DATA_DIR, lm_model, lm_tokenizer) for name in DATASETS}

    if lm_model is not None:
        del lm_model, lm_tokenizer; torch.cuda.empty_cache()

    results = {}
    for name, cfg in DATASETS.items():
        front_df = rerank(name, scored_dfs[name], RESULT_DIR)
        plot_single(name, cfg["label"], front_df, greedy_baselines[name])
        results[name] = (front_df, greedy_baselines[name], cfg["label"])

    plot_combined(results)
    print("Done.")


if __name__ == "__main__":
    main()
