# mrt_eval.py — evaluate MRT checkpoints and plot against fine-tuned baseline
# python src/mrt_eval.py --domain emea

import argparse
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
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

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--domain", required=True,
                    choices=["emea", "news_commentary", "opus_books"])
parser.add_argument("--lambdas", nargs="+", type=float,
                    default=[0.1, 0.3, 0.5, 0.7, 0.9])
args = parser.parse_args()

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | domain: {args.domain}")

MRT_DIR  = RUN_DIR / "mrt_baseline" / args.domain
EVAL_DIR = MRT_DIR / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_LABELS = {
    "emea":            "EMEA (Medical)",
    "news_commentary": "News Commentary",
    "opus_books":      "Opus Books (Literary)",
}
label = DOMAIN_LABELS[args.domain]

# ── Load eval set ─────────────────────────────────────────────────────────────

eval_df = pd.read_csv(SPLITS_DIR / f"{args.domain}_dev.csv") \
            .dropna(subset=["src_en", "ref_fr"]).head(N)

# ── Determine which λ checkpoints still need computation ─────────────────────

def needs_compute(lam: float) -> bool:
    lam_tag     = str(lam).replace(".", "p")
    data_subdir = EVAL_DIR / f"lambda_{lam_tag}"
    cached_front  = data_subdir / f"{args.domain}_af_front.csv"
    cached_greedy = data_subdir / f"{args.domain}_greedy_scored.jsonl"
    return not (cached_front.exists() and cached_greedy.exists())

lambdas_needing_compute = [
    lam for lam in args.lambdas
    if (MRT_DIR / f"lambda_{str(lam).replace('.','p')}" / "checkpoint").exists()
    and needs_compute(lam)
]

# ── Load LM only if at least one λ needs computation ─────────────────────────

if lambdas_needing_compute:
    print(f"Loading frozen LM: {LM_MODEL} (needed for {lambdas_needing_compute})")
    lm_tokenizer = AutoTokenizer.from_pretrained(LM_MODEL)
    lm_model     = AutoModelForCausalLM.from_pretrained(LM_MODEL).to(device)
    lm_model.eval()
    for p in lm_model.parameters():
        p.requires_grad = False
else:
    print("All results cached — skipping LM load.")
    lm_model = lm_tokenizer = None

# ── Helper: run full pipeline for one checkpoint ──────────────────────────────

def eval_checkpoint(name: str, mt_model, mt_tokenizer, data_subdir: Path, tag: str):
    data_subdir.mkdir(parents=True, exist_ok=True)
    cand_df   = generate_candidates(name, eval_df, mt_model, mt_tokenizer, data_subdir, tag=tag)
    greedy_df = generate_greedy(name, eval_df, mt_model, mt_tokenizer, data_subdir, tag=tag)
    chrf_df   = score_adequacy(name, cand_df,   data_subdir, tag=tag)
    scored_df = score_fluency( name, chrf_df,   data_subdir, lm_model, lm_tokenizer, tag=tag)
    front_df  = rerank(        name, scored_df, data_subdir, tag=tag)
    greedy    = score_greedy(  name, greedy_df, data_subdir, lm_model, lm_tokenizer, tag=tag)
    return front_df, greedy

# ── Load RAW baseline front + greedy ─────────────────────────────────────────

RAW_DATA_DIR   = RUN_DIR / "baseline" / "data"
RAW_RESULT_DIR = RUN_DIR / "baseline" / "results"

raw_front_path  = RAW_RESULT_DIR / f"{args.domain}_af_front.csv"
raw_greedy_path = RAW_DATA_DIR   / f"{args.domain}_greedy_scored.jsonl"

if not raw_front_path.exists() or not raw_greedy_path.exists():
    print(f"WARNING: Missing raw baseline files for {args.domain} — will not plot raw baseline.")
    raw_front  = None
    raw_greedy = None
else:
    raw_front     = pd.read_csv(raw_front_path)
    raw_greedy_df = pd.read_json(raw_greedy_path, orient="records", lines=True)
    raw_greedy    = {"chrf": raw_greedy_df["chrf"].mean(), "fluency": raw_greedy_df["fluency"].mean()}
    print(f"Loaded raw baseline: chrF={raw_greedy['chrf']:.2f}, fluency={raw_greedy['fluency']:.3f}")

# ── Load FINE-TUNED baseline front + greedy ───────────────────────────────────

FT_DATA_DIR   = RUN_DIR / "finetuned" / "data"
FT_RESULT_DIR = RUN_DIR / "finetuned" / "results"

base_front_path  = FT_RESULT_DIR / f"{args.domain}_af_front.csv"
base_greedy_path = FT_DATA_DIR   / f"{args.domain}_greedy_scored.jsonl"

if not base_front_path.exists() or not base_greedy_path.exists():
    raise FileNotFoundError(
        f"Missing fine-tuned baseline files for {args.domain}. "
        "Run finetuned_fronts.py first."
    )

ft_front     = pd.read_csv(base_front_path)
ft_greedy_df = pd.read_json(base_greedy_path, orient="records", lines=True)
ft_greedy    = {"chrf": ft_greedy_df["chrf"].mean(), "fluency": ft_greedy_df["fluency"].mean()}
print(f"Loaded fine-tuned baseline: chrF={ft_greedy['chrf']:.2f}, fluency={ft_greedy['fluency']:.3f}")

# ── Evaluate each λ checkpoint ────────────────────────────────────────────────

lambda_results = {}   # λ → (front_df, greedy_scores)

for lam in args.lambdas:
    lam_tag     = str(lam).replace(".", "p")
    ckpt_dir    = MRT_DIR / f"lambda_{lam_tag}" / "checkpoint"
    data_subdir = EVAL_DIR / f"lambda_{lam_tag}"

    if not ckpt_dir.exists():
        print(f"[λ={lam}] No checkpoint found at {ckpt_dir} — skipping.")
        continue

    # ── Load directly from cache if everything already exists ────────────
    cached_front  = data_subdir / f"{args.domain}_af_front.csv"
    cached_greedy = data_subdir / f"{args.domain}_greedy_scored.jsonl"

    if cached_front.exists() and cached_greedy.exists():
        print(f"[λ={lam}] All cached — loading results directly, skipping model load.")
        front_df  = pd.read_csv(cached_front)
        greedy_df = pd.read_json(cached_greedy, orient="records", lines=True)
        greedy    = {"chrf": greedy_df["chrf"].mean(), "fluency": greedy_df["fluency"].mean()}
        lambda_results[lam] = (front_df, greedy)
        print(f"[λ={lam}] greedy → chrF={greedy['chrf']:.2f}, fluency={greedy['fluency']:.3f}")
        continue   # ← no model loaded, no GPU memory used

    # ── Only reach here if computation is actually needed ────────────────
    print(f"\n[λ={lam}] Loading checkpoint from {ckpt_dir}")
    mt_tokenizer_lam = MBart50TokenizerFast.from_pretrained(MT_MODEL)
    mt_tokenizer_lam.src_lang = SRC_LANG

    base_model = MBartForConditionalGeneration.from_pretrained(MT_MODEL)
    if (ckpt_dir / "adapter_config.json").exists():
        mt_model_lam = PeftModel.from_pretrained(base_model, str(ckpt_dir)).to(device)
    else:
        mt_model_lam = base_model.to(device)
    mt_model_lam.eval()

    front_df, greedy = eval_checkpoint(
        name         = args.domain,
        mt_model     = mt_model_lam,
        mt_tokenizer = mt_tokenizer_lam,
        data_subdir  = data_subdir,
        tag          = f"λ={lam} ",
    )

    lambda_results[lam] = (front_df, greedy)
    print(f"[λ={lam}] greedy → chrF={greedy['chrf']:.2f}, fluency={greedy['fluency']:.3f}")

    del mt_model_lam, base_model
    torch.cuda.empty_cache()

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

# Raw mBART baseline front + greedy
if raw_front is not None:
    ax.plot(raw_front["chrf"], raw_front["fluency"],
            color="grey", linewidth=2, linestyle=":",
            label="Raw baseline oracle front", zorder=2)
    ax.scatter([raw_greedy["chrf"]], [raw_greedy["fluency"]],
               marker="D", s=150, color="grey", zorder=5,
               label="Raw baseline greedy")

# Fine-tuned baseline front + greedy
ax.plot(ft_front["chrf"], ft_front["fluency"],
        color="black", linewidth=2.5, linestyle="--",
        label="Fine-tuned baseline oracle front", zorder=3)
ax.scatter([ft_greedy["chrf"]], [ft_greedy["fluency"]],
           marker="*", s=250, color="black", zorder=6,
           label="Fine-tuned baseline greedy")

# λ sweep — colour-mapped from blue (low λ, fluency) to red (high λ, adequacy)
cmap     = cm.coolwarm
lam_vals = sorted(lambda_results.keys())
norm     = plt.Normalize(vmin=min(lam_vals), vmax=max(lam_vals))

for lam in lam_vals:
    front_df, greedy = lambda_results[lam]
    colour = cmap(norm(lam))
    ax.plot(front_df["chrf"], front_df["fluency"],
            color=colour, linewidth=1.5, linestyle="-", alpha=0.6)
    ax.scatter([greedy["chrf"]], [greedy["fluency"]],
               marker="o", s=120, color=colour, zorder=5,
               edgecolors="black", linewidths=0.5,
               label=f"λ={lam}")

# Colourbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("λ (0=fluency, 1=adequacy)")

ax.set_xlabel("Adequacy (chrF)")
ax.set_ylabel("Fluency (neg-NLL)")
ax.set_title(f"MRT λ Sweep vs Fine-tuned Baseline — {label}\nN={N}, K={K}")
ax.legend(fontsize=8, loc="lower left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = EVAL_DIR / f"{args.domain}_mrt_sweep.png"
plt.savefig(str(plot_path), dpi=150)
plt.close()
print(f"\nSaved plot to {plot_path}")

# ── Save greedy scores as CSV ─────────────────────────────────────────────────

rows = [
    {"lambda": "baseline_raw", "chrf": raw_greedy["chrf"]  if raw_greedy else None,
                                "fluency": raw_greedy["fluency"] if raw_greedy else None},
    {"lambda": "baseline_ft",  "chrf": ft_greedy["chrf"],   "fluency": ft_greedy["fluency"]},
]
for lam, (_, greedy) in sorted(lambda_results.items()):
    rows.append({"lambda": lam, "chrf": greedy["chrf"], "fluency": greedy["fluency"]})

results_df = pd.DataFrame(rows)
results_df.to_csv(EVAL_DIR / f"{args.domain}_greedy_scores.csv", index=False)
print(results_df.to_string(index=False))