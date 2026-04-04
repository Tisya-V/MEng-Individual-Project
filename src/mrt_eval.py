# mrt_eval_run.py — evaluate a single MRT checkpoint for one λ
# python src/mrt_eval_run.py --domain emea --lam 0.1

import argparse
import random
import numpy as np
import torch
import pandas as pd
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

parser = argparse.ArgumentParser()
parser.add_argument("--domain", required=True, choices=["emea", "news_commentary", "opus_books"])
parser.add_argument("--lam",    type=float, required=True)
args = parser.parse_args()

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | domain: {args.domain} | λ={args.lam}")

lam_tag     = str(args.lam).replace(".", "p")
MRT_DIR     = RUN_DIR / "mrt" / args.domain
ckpt_dir    = MRT_DIR / f"lambda_{lam_tag}" / "checkpoint"
EVAL_DIR    = MRT_DIR / "eval"
data_subdir = EVAL_DIR / f"lambda_{lam_tag}"

# ── Guard: checkpoint must exist ─────────────────────────────────────────────

if not ckpt_dir.exists():
    print(f"No checkpoint found at {ckpt_dir} — exiting.")
    exit(1)

# ── Guard: already fully cached ──────────────────────────────────────────────

cached_front  = data_subdir / f"{args.domain}_af_front.csv"
cached_greedy = data_subdir / f"{args.domain}_greedy_scored.jsonl"

if cached_front.exists() and cached_greedy.exists():
    print(f"[λ={args.lam}] Already fully cached — nothing to do.")
    exit(0)

data_subdir.mkdir(parents=True, exist_ok=True)

# ── Load eval set ─────────────────────────────────────────────────────────────

eval_df = pd.read_csv(SPLITS_DIR / f"{args.domain}_dev.csv") \
            .dropna(subset=["src_en", "ref_fr"]).head(N)

# ── Load LM ───────────────────────────────────────────────────────────────────

print(f"Loading frozen LM: {LM_MODEL}")
lm_tokenizer = AutoTokenizer.from_pretrained(LM_MODEL)
lm_model     = AutoModelForCausalLM.from_pretrained(LM_MODEL).to(device)
lm_model.eval()
for p in lm_model.parameters():
    p.requires_grad = False

# ── Load MT checkpoint ────────────────────────────────────────────────────────

print(f"Loading checkpoint from {ckpt_dir}")
mt_tokenizer = MBart50TokenizerFast.from_pretrained(MT_MODEL)
mt_tokenizer.src_lang = SRC_LANG

base_model = MBartForConditionalGeneration.from_pretrained(MT_MODEL)
if (ckpt_dir / "adapter_config.json").exists():
    print(f"Detected PEFT adapter checkpoint.")
    mt_model = PeftModel.from_pretrained(base_model, str(ckpt_dir)).to(device)
else:
    print(f"Detected full model checkpoint.")
    mt_model = MBartForConditionalGeneration.from_pretrained(str(ckpt_dir)).to(device)
mt_model.eval()

# ── Run pipeline ──────────────────────────────────────────────────────────────

tag = f"λ={args.lam} "
cand_df   = generate_candidates(args.domain, eval_df, mt_model, mt_tokenizer, data_subdir, tag=tag)
greedy_df = generate_greedy(args.domain, eval_df, mt_model, mt_tokenizer, data_subdir, tag=tag)
chrf_df   = score_adequacy(args.domain, cand_df, data_subdir, tag=tag)
scored_df = score_fluency(args.domain, chrf_df, data_subdir, lm_model, lm_tokenizer, tag=tag)
rerank(args.domain, scored_df, data_subdir, tag=tag)
score_greedy(args.domain, greedy_df, data_subdir, lm_model, lm_tokenizer, tag=tag)

print(f"[λ={args.lam}] Done. Results saved to {data_subdir}")