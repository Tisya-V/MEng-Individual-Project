# mrt_train.py — MRT fine-tuning with scalarised A-F reward
# Starts from raw mBART baseline (no domain adapter)
# python mrt_train.py --domain emea --lambda_val 0.7

import argparse
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import sacrebleu
from pathlib import Path
from tqdm.auto import tqdm
from transformers import (
    MBartForConditionalGeneration, MBart50TokenizerFast,
    AutoTokenizer, AutoModelForCausalLM,
)
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from config import *
from af_front_pipeline_stages import _is_valid, _log_ppl

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--domain",     required=True,
                    choices=["emea", "news_commentary", "opus_books"])
parser.add_argument("--lambda_val", type=float, required=True,
                    help="λ ∈ [0,1]. High=more adequate, Low=more fluent.")
parser.add_argument("--epochs",     type=int,   default=3)
parser.add_argument("--batch_size", type=int,   default=1)
parser.add_argument("--n_samples",  type=int,   default=4)
parser.add_argument("--train_size", type=int,   default=2000,
                    help="Max training sentences to use.")
parser.add_argument("--alpha",      type=float, default=0.8,
                    help="Mix weight: α·MRT + (1-α)·MLE.")
parser.add_argument("--lr",         type=float, default=2e-5)
args = parser.parse_args()

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | domain: {args.domain} | λ={args.lambda_val}")

# ── Output paths ──────────────────────────────────────────────────────────────

# ── Output paths ──────────────────────────────────────────────────────────────

lam_tag  = str(args.lambda_val).replace(".", "p")
out_dir  = RUN_DIR / "mrt_baseline" / args.domain / f"lambda_{lam_tag}"
out_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir = out_dir / "checkpoint"

# Check for actual saved weights, not just directory existence
ckpt_complete = (ckpt_dir / "pytorch_model.bin").exists() or \
                (ckpt_dir / "model.safetensors").exists()
if ckpt_complete:
    print(f"Checkpoint exists at {ckpt_dir} — skipping.")
    exit(0)

# ── Load data ─────────────────────────────────────────────────────────────────

train_df = pd.read_csv(SPLITS_DIR / f"{args.domain}_train.csv").dropna(subset=["src_en", "ref_fr"])
train_df = train_df[train_df["src_en"].apply(_is_valid) & train_df["ref_fr"].apply(_is_valid)]
if len(train_df) > args.train_size:
    train_df = train_df.sample(n=args.train_size, random_state=SEED)
print(f"Training on {len(train_df)} sentence pairs.")

# ── Load MT model — raw mBART, no adapter ────────────────────────────────────

print(f"Loading raw mBART: {MT_MODEL}")
mt_tokenizer = MBart50TokenizerFast.from_pretrained(MT_MODEL)
mt_tokenizer.src_lang = SRC_LANG
forced_bos = mt_tokenizer.lang_code_to_id[TGT_LANG]

mt_model = MBartForConditionalGeneration.from_pretrained(MT_MODEL).to(device)
mt_model.train()

# All parameters trainable — no frozen base
for param in mt_model.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in mt_model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable:,}")

optimizer = AdamW(mt_model.parameters(), lr=args.lr)

# ── Load frozen LM ────────────────────────────────────────────────────────────

print(f"Loading frozen LM: {LM_MODEL}")
lm_tokenizer = AutoTokenizer.from_pretrained(LM_MODEL)
lm_model     = AutoModelForCausalLM.from_pretrained(LM_MODEL).to(device)
lm_model.eval()
for p in lm_model.parameters():
    p.requires_grad = False

chrf_metric = sacrebleu.CHRF()

# ── Scoring ───────────────────────────────────────────────────────────────────

def score_batch(hypotheses, references):
    adq, flu = [], []
    for hyp, ref in zip(hypotheses, references):
        a = chrf_metric.sentence_score(hyp, [ref]).score / 100.0  # [0,100] → [0,1]
        adq.append(a)
        raw = _log_ppl(hyp, lm_model, lm_tokenizer)
        flu.append(-raw if raw is not None else 0.0)

    adq_t = torch.tensor(adq, dtype=torch.float32)
    flu_t = torch.tensor(flu, dtype=torch.float32)

    # Min-max normalise to [0,1] within the sample
    def minmax(t):
        r = t.max() - t.min()
        return (t - t.min()) / (r + 1e-8) if r > 1e-8 else torch.zeros_like(t)

    reward = args.lambda_val * minmax(adq_t) + (1 - args.lambda_val) * minmax(flu_t)
    return reward

# ── Batched log-prob ──────────────────────────────────────────────────────────

def batched_log_probs(src_ids: torch.Tensor, sample_ids: torch.Tensor) -> torch.Tensor:
    N = sample_ids.size(0)
    hyp_padded = pad_sequence(
        [sample_ids[i] for i in range(N)],
        batch_first=True, padding_value=mt_tokenizer.pad_token_id
    )
    src_expanded = src_ids.expand(N, -1)
    labels = hyp_padded.clone()
    labels[labels == mt_tokenizer.pad_token_id] = -100

    logits = mt_model(input_ids=src_expanded, labels=labels).logits

    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(N, -1)

    mask = (shift_labels != -100).float()
    return -(loss_per_token * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

# ── MRT step ──────────────────────────────────────────────────────────────────

def mrt_step(src_batch: list[str], ref_batch: list[str]) -> float:
    total_loss = torch.tensor(0.0, device=device)

    for src, ref in zip(src_batch, ref_batch):
        src_enc = mt_tokenizer(
            src, return_tensors="pt", truncation=True, max_length=256
        ).to(device)
        src_ids = src_enc["input_ids"]

        with torch.no_grad():
            sample_ids = mt_model.generate(
                **src_enc,
                do_sample=True,
                top_p=TOP_P,
                temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOK,
                max_length=None,
                forced_bos_token_id=forced_bos,
                num_return_sequences=args.n_samples,
                num_beams=1,
            )

        hypotheses = mt_tokenizer.batch_decode(sample_ids, skip_special_tokens=True)
        rewards    = score_batch(hypotheses, [ref] * args.n_samples)
        advantages = (rewards - rewards.mean()).to(device)
        log_probs  = batched_log_probs(src_ids, sample_ids)

        reinforce_loss = -(advantages.detach() * log_probs).mean()

        ref_ids  = mt_tokenizer(
            ref, return_tensors="pt", truncation=True, max_length=256
        ).to(device)["input_ids"]
        mle_loss = mt_model(input_ids=src_ids, labels=ref_ids).loss

        total_loss = total_loss + args.alpha * reinforce_loss + (1 - args.alpha) * mle_loss

    total_loss = total_loss / len(src_batch)
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(mt_model.parameters(), 1.0)
    optimizer.step()
    return total_loss.item()

# ── Training loop ─────────────────────────────────────────────────────────────

log_rows = []

for epoch in range(args.epochs):
    shuffled     = train_df.sample(frac=1, random_state=SEED + epoch)
    epoch_losses = []

    for i in tqdm(range(0, len(shuffled), args.batch_size),
                  desc=f"Epoch {epoch+1}/{args.epochs}"):
        batch = shuffled.iloc[i : i + args.batch_size]
        loss  = mrt_step(batch["src_en"].tolist(), batch["ref_fr"].tolist())
        epoch_losses.append(loss)

    mean_loss = float(np.mean(epoch_losses))
    print(f"Epoch {epoch+1} | mean loss: {mean_loss:.4f}")
    log_rows.append({"epoch": epoch + 1, "mean_loss": mean_loss})

# ── Save ──────────────────────────────────────────────────────────────────────

mt_model.save_pretrained(ckpt_dir)
mt_tokenizer.save_pretrained(ckpt_dir)
print(f"Saved checkpoint to {ckpt_dir}")

pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)

with open(out_dir / "run_meta.json", "w") as f:
    json.dump({
        "domain":     args.domain,
        "lambda_val": args.lambda_val,
        "base_model": MT_MODEL,
        "train_size": args.train_size,
        "epochs":     args.epochs,
        "n_samples":  args.n_samples,
        "alpha":      args.alpha,
        "lr":         args.lr,
        "batch_size": args.batch_size,
    }, f, indent=2)

print("Done.")