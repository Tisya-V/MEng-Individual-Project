# plot_finetuned_on_front.py
#
# Evaluates each fine-tuned LoRA model (greedy decode) and plots its
# (adequacy, fluency) operating point on the existing oracle A-F fronts.
#
# Reads:   runs/n{N}_k{K}/results/{name}_af_front.csv   (existing fronts)
#          runs/n{N}_k{K}/data/{name}_greedy_scored.jsonl (beam baselines)
#          data/splits/{name}_dev.csv
#          models/{name}/  (LoRA adapters)
# Writes:  runs/n{N}_k{K}/results/finetuned_on_front.png


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from tqdm.auto import tqdm
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sacrebleu
from config import *


# ── Helpers ───────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_adequacy(hyps, refs):
    metric = sacrebleu.CHRF()
    return np.mean([metric.sentence_score(h, [r]).score for h, r in zip(hyps, refs)])


def score_fluency(hyps, lm_model, lm_tokenizer):
    losses = []
    empty = sum(1 for h in hyps if not h or not h.strip())
    for h in tqdm(hyps, desc="  fluency"):
        if not h or not h.strip():
            continue
        enc = lm_tokenizer(h, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            loss = lm_model(**enc, labels=enc["input_ids"]).loss.item()
        if not np.isnan(loss):
            losses.append(loss)
    return -np.mean(losses) if losses else float("nan"), empty


def evaluate_model(name, eval_df, mt_model, mt_tokenizer, lm_model, lm_tokenizer):
    forced_bos = mt_tokenizer.lang_code_to_id[TGT_LANG]
    hyps = []
    for src in tqdm(eval_df["src_en"].tolist(), desc=f"  [{name}] generating"):
        inputs = mt_tokenizer(src, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = mt_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                num_beams=4,
                max_new_tokens=128,
                max_length=None,
            )
        hyps.append(mt_tokenizer.decode(out[0], skip_special_tokens=True))
    refs = eval_df["ref_fr"].tolist()
    adq = score_adequacy(hyps, refs)
    flu, empty = score_fluency(hyps, lm_model, lm_tokenizer)
    return adq, flu, empty


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading LM: {LM_MODEL}")
    lm_tokenizer = AutoTokenizer.from_pretrained(LM_MODEL)
    lm_model     = AutoModelForCausalLM.from_pretrained(LM_MODEL).to(device)
    lm_model.eval()

    points = {}  # name -> (adq, flu)

    for name in DATASETS:
        adapter_dir = MODELS_DIR / name
        if not (adapter_dir / "adapter_config.json").exists():
            print(f"[{name}] No adapter found at {adapter_dir}, skipping.")
            continue

        print(f"\n[{name}] Loading fine-tuned model...")
        mt_tokenizer = MBart50TokenizerFast.from_pretrained(adapter_dir)
        mt_tokenizer.src_lang = SRC_LANG
        base = MBartForConditionalGeneration.from_pretrained(MT_MODEL)
        mt_model = PeftModel.from_pretrained(base, adapter_dir).to(device)
        mt_model.eval()

        eval_df = pd.read_csv(SPLITS_DIR / f"{name}_dev.csv").dropna(subset=["src_en", "ref_fr"]).head(N)
        adq, flu, empty = evaluate_model(name, eval_df, mt_model, mt_tokenizer, lm_model, lm_tokenizer)
        points[name] = (adq, flu)
        print(f"  [{name}] chrF={adq:.2f}  fluency={flu:.4f}  empty hyps={empty}")

        del mt_model, base; torch.cuda.empty_cache()

    del lm_model; torch.cuda.empty_cache()

    # ── Plot ──────────────────────────────────────────────────────────────────
    colours = cm.tab10.colors
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, (name, cfg) in enumerate(DATASETS.items()):
        label = cfg["label"]
        front_path  = RESULT_DIR / f"{name}_af_front.csv"
        greedy_path = DATA_DIR   / f"{name}_greedy_scored.jsonl"

        if not front_path.exists():
            print(f"[{name}] No front CSV found, skipping.")
            continue
        if not greedy_path.exists():
            print(f"[{name}] No greedy scored file found — run pipeline_af_front.py first.")
            continue

        front_df  = pd.read_csv(front_path)
        greedy_df = pd.read_json(greedy_path, orient="records", lines=True)
        baseline_chrf    = greedy_df["chrf"].mean()
        baseline_fluency = greedy_df["fluency"].mean()

        c = colours[i]

        # Oracle front
        ax.plot(front_df["chrf"], front_df["fluency"],
                color=c, linewidth=2, alpha=0.7, label=f"{label} — oracle front")

        # Beam decode baseline (×)
        ax.scatter([baseline_chrf], [baseline_fluency],
                   marker="x", s=120, linewidths=2, color=c, zorder=4,
                   label=f"{label} — beam baseline")

        # Fine-tuned greedy (★)
        if name in points:
            adq, flu = points[name]
            ax.scatter([adq], [flu], marker="*", s=250, color=c, zorder=5,
                       label=f"{label} — fine-tuned")

    ax.set_xlabel("Adequacy (chrF)")
    ax.set_ylabel("Fluency (neg-NLL)")
    ax.set_title(
        f"Fine-tuned Models on A-F Oracle Fronts (N={N}, K={K})\n"
        f"× = beam decode baseline   ★ = fine-tuned model"
    )
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = RESULT_DIR / "finetuned_on_front.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
