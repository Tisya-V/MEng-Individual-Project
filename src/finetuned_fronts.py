import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from transformers import (
    MBartForConditionalGeneration, MBart50TokenizerFast,
    AutoTokenizer, AutoModelForCausalLM,
)
from peft import PeftModel
from config import *
from af_front_pipeline_stages import (
    generate_candidates, generate_greedy,
    score_adequacy, score_fluency, score_greedy, rerank,
)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FT_DIR        = RUN_DIR / "finetuned"
FT_DATA_DIR   = FT_DIR  / "data"
FT_RESULT_DIR = FT_DIR  / "results"

DATASETS = {
    "emea":            {"label": "EMEA (Medical)"},
    "news_commentary": {"label": "News Commentary"},
    "opus_books":      {"label": "Opus Books (Literary)"},
}


def plot_single(name, label, base_front, base_greedy, ft_front, ft_greedy):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(base_front["chrf"], base_front["fluency"],
            color="steelblue", linewidth=2.5, linestyle="--", label="Baseline oracle front")
    ax.scatter([base_greedy["chrf"]], [base_greedy["fluency"]],
               marker="x", s=120, linewidths=2, color="steelblue", zorder=5, label="Baseline beam decode")
    ax.plot(ft_front["chrf"], ft_front["fluency"],
            color="tomato", linewidth=2.5, linestyle="-", label="Fine-tuned oracle front")
    ax.scatter([ft_greedy["chrf"]], [ft_greedy["fluency"]],
               marker="*", s=200, color="tomato", zorder=5, label="Fine-tuned beam decode")
    ax.set_xlabel("Adequacy (chrF)"); ax.set_ylabel("Fluency (neg-NLL)")
    ax.set_title(f"Baseline vs Fine-tuned A-F Front — {label}\nN={N}, K={K}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(FT_RESULT_DIR / f"{name}_af_front.png"), dpi=150)
    plt.close()
    print(f"[{name}] Saved per-domain plot.")


def plot_combined(all_results):
    colours = cm.tab10.colors
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, (label, base_front, base_greedy, ft_front, ft_greedy)) in enumerate(all_results.items()):
        c = colours[i]
        ax.plot(base_front["chrf"], base_front["fluency"],
                color=c, linewidth=2, linestyle="--", alpha=0.8, label=f"{label} baseline")
        ax.scatter([base_greedy["chrf"]], [base_greedy["fluency"]],
                   marker="x", s=120, linewidths=2, color=c, zorder=4)
        ax.plot(ft_front["chrf"], ft_front["fluency"],
                color=c, linewidth=2, linestyle="-", label=f"{label} fine-tuned")
        ax.scatter([ft_greedy["chrf"]], [ft_greedy["fluency"]],
                   marker="*", s=200, color=c, zorder=5)
    ax.scatter([], [], marker="x", s=120, linewidths=2, color="grey", label="Beam decode")
    ax.scatter([], [], marker="*", s=200,               color="grey", label="Fine-tuned beam decode")
    ax.set_xlabel("Adequacy (chrF)"); ax.set_ylabel("Fluency (neg-NLL)")
    ax.set_title(f"Baseline vs Fine-tuned A-F Fronts (N={N}, K={K})\ndashed = baseline   solid = fine-tuned")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(FT_RESULT_DIR / "combined_af_front.png"), dpi=150)
    plt.close()
    print("Saved combined plot.")


def main():
    for d in [FT_DATA_DIR, FT_RESULT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading LM: {LM_MODEL}")
    lm_tokenizer = AutoTokenizer.from_pretrained(LM_MODEL)
    lm_model     = AutoModelForCausalLM.from_pretrained(LM_MODEL).to(device)
    lm_model.eval()

    all_results = {}

    for name, cfg in DATASETS.items():
        adapter_dir = MODELS_DIR / name
        if not (adapter_dir / "adapter_config.json").exists():
            print(f"[{name}] No adapter found, skipping.")
            continue

        # Load fine-tuned model
        print(f"\n[{name}] Loading fine-tuned model...")
        mt_tokenizer = MBart50TokenizerFast.from_pretrained(adapter_dir)
        mt_tokenizer.src_lang = SRC_LANG
        base     = MBartForConditionalGeneration.from_pretrained(MT_MODEL)
        mt_model = PeftModel.from_pretrained(base, adapter_dir).to(device)
        mt_model.eval()

        eval_df    = pd.read_csv(SPLITS_DIR / f"{name}_dev.csv").dropna(subset=["src_en", "ref_fr"]).head(N)
        cand_df    = generate_candidates(name, eval_df, mt_model, mt_tokenizer, FT_DATA_DIR, tag="FT ")
        greedy_df  = generate_greedy(name, eval_df, mt_model, mt_tokenizer,     FT_DATA_DIR, tag="FT ")

        del mt_model, base; torch.cuda.empty_cache()

        # Score fine-tuned outputs
        chrf_df   = score_adequacy(name, cand_df,   FT_DATA_DIR, tag="FT ")
        scored_df = score_fluency( name, chrf_df,   FT_DATA_DIR, lm_model, lm_tokenizer, tag="FT ")
        ft_front  = rerank(        name, scored_df, FT_RESULT_DIR, tag="FT ")
        ft_greedy = score_greedy(  name, greedy_df, FT_DATA_DIR, lm_model, lm_tokenizer, tag="FT ")

        # Load baseline front + greedy (pre-computed)
        base_front_path  = RESULT_DIR / f"{name}_af_front.csv"
        base_greedy_path = DATA_DIR   / f"{name}_greedy_scored.jsonl"
        if not base_front_path.exists() or not base_greedy_path.exists():
            print(f"[{name}] Missing baseline files — run pipeline_af_front.py first.")
            continue
        base_front     = pd.read_csv(base_front_path)
        base_greedy_df = pd.read_json(base_greedy_path, orient="records", lines=True)
        base_greedy    = {"chrf": base_greedy_df["chrf"].mean(),
                          "fluency": base_greedy_df["fluency"].mean()}

        all_results[name] = (cfg["label"], base_front, base_greedy, ft_front, ft_greedy)
        plot_single(name, cfg["label"], base_front, base_greedy, ft_front, ft_greedy)

    del lm_model; torch.cuda.empty_cache()

    if all_results:
        plot_combined(all_results)

    print("Done.")


if __name__ == "__main__":
    main()
