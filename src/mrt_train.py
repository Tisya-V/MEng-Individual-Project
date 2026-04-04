# mrt_train.py — MRT fine-tuning with scalarised A-F reward
# Starts from fine-tuned LoRA checkpoint (merged into base)
# python src/mrt_train.py --domain emea --lambda_val 0.5

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
    get_linear_schedule_with_warmup,
)
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from config import *
from af_front_pipeline_stages import _is_valid, _log_ppl


# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--domain",     required=True,
                    choices=["emea", "news_commentary", "opus_books"])
parser.add_argument("--lambda_val", type=float, required=True)
parser.add_argument("--epochs",     type=int,   default=3)
parser.add_argument("--batch_size", type=int,   default=1)
parser.add_argument("--n_samples",  type=int,   default=32)
parser.add_argument("--train_size", type=int,   default=2000)
parser.add_argument("--alpha",      type=float, default=0.7,
                    help="Mix weight: α·REINFORCE + (1-α)·MLE. Wu et al. best=0.7")
parser.add_argument("--lr",         type=float, default=5e-6)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--ema_decay",  type=float, default=0.99)
parser.add_argument("--flu_min",    type=float, default=-12.0,
                    help="Fluency normalisation lower bound (neg-NLL). Calibrate from dev set.")
parser.add_argument("--flu_max",    type=float, default=-0.4,
                    help="Fluency normalisation upper bound (neg-NLL). Calibrate from dev set.")
args = parser.parse_args()

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | domain: {args.domain} | λ={args.lambda_val} | α={args.alpha}")


# ── Output paths ──────────────────────────────────────────────────────────────

lam_tag  = str(args.lambda_val).replace(".", "p")
OUT_DIR  = RUN_DIR / "mrt" / args.domain / f"lambda_{lam_tag}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ckpt_dir = OUT_DIR / "checkpoint"

ckpt_complete = (ckpt_dir / "adapter_config.json").exists()
if ckpt_complete:
    print(f"Checkpoint exists at {ckpt_dir} — skipping.")
    exit(0)


# ── Load data ─────────────────────────────────────────────────────────────────

train_df = pd.read_csv(SPLITS_DIR / f"{args.domain}_train.csv").dropna(subset=["src_en", "ref_fr"])
train_df = train_df[train_df["src_en"].apply(_is_valid) & train_df["ref_fr"].apply(_is_valid)]
if len(train_df) > args.train_size:
    train_df = train_df.sample(n=args.train_size, random_state=SEED)
print(f"Training on {len(train_df)} sentence pairs.")

# Small fixed val set for per-epoch sanity check (no shuffle, no leakage)
val_df = pd.read_csv(SPLITS_DIR / f"{args.domain}_dev.csv") \
           .dropna(subset=["src_en", "ref_fr"]).head(30)


# ── Load MT model — fine-tuned LoRA merged into base ─────────────────────────

FT_CKPT = MODELS_DIR / args.domain 
if not FT_CKPT.exists():
    raise FileNotFoundError(f"Fine-tuned checkpoint not found at {FT_CKPT}. Run finetuning first.")

print(f"Loading fine-tuned checkpoint from {FT_CKPT} and merging LoRA into base...")
base_model = MBartForConditionalGeneration.from_pretrained(MT_MODEL)
ft_peft    = PeftModel.from_pretrained(base_model, str(FT_CKPT))
mt_model   = ft_peft.merge_and_unload()   # bakes LoRA weights into base, removes adapters
print("Fine-tuned LoRA merged.")

# Attach fresh LoRA adapters for MRT steering
lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
mt_model = get_peft_model(mt_model, lora_cfg).to(device)
mt_model.train()
mt_model.print_trainable_parameters()   # sanity check — should be ~4M, not 610M

mt_tokenizer = MBart50TokenizerFast.from_pretrained(MT_MODEL)
mt_tokenizer.src_lang = SRC_LANG
forced_bos = mt_tokenizer.lang_code_to_id[TGT_LANG]


# ── Optimiser + LR scheduler ──────────────────────────────────────────────────

total_steps = args.epochs * (args.train_size // args.batch_size)
optimizer   = AdamW(mt_model.parameters(), lr=args.lr, weight_decay=0.01)
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=total_steps,
)
print(f"Scheduler: {args.warmup_steps} warmup steps over {total_steps} total steps.")


# ── Load frozen LM ────────────────────────────────────────────────────────────

print(f"Loading frozen LM: {LM_MODEL}")
lm_tokenizer = AutoTokenizer.from_pretrained(LM_MODEL)
lm_model     = AutoModelForCausalLM.from_pretrained(LM_MODEL).to(device)
lm_model.eval()
for p in lm_model.parameters():
    p.requires_grad = False

chrf_metric = sacrebleu.CHRF()


# ── Fluency normalisation ─────────────────────────────────────────────────────
# Calibrate flu_min/flu_max from your dev set references once:
#   python -c "
#   from transformers import AutoTokenizer, AutoModelForCausalLM
#   import torch, pandas as pd
#   from config import *
#   from af_front_pipeline_stages import _log_ppl
#   device = torch.device('cuda')
#   tok = AutoTokenizer.from_pretrained(LM_MODEL)
#   lm  = AutoModelForCausalLM.from_pretrained(LM_MODEL).to(device)
#   df  = pd.read_csv(SPLITS_DIR / 'emea_dev.csv').dropna().head(200)
#   scores = [-_log_ppl(r, lm, tok) for r in df.ref_fr if _log_ppl(r, lm, tok)]
#   print(min(scores), max(scores))
#   "
# Then pass --flu_min and --flu_max accordingly.

def normalise_flu(f: float) -> float:
    """Clip neg-NLL to [flu_min, flu_max] then scale to [0, 1]."""
    return max(0.0, min(1.0, (f - args.flu_min) / (args.flu_max - args.flu_min + 1e-8)))


# ── EMA reward baseline ───────────────────────────────────────────────────────
# Initialise from a small warmup sample so EMA starts near the real reward range.

print("Calibrating EMA baseline from 30 warmup samples...")
_warmup_rewards = []
mt_model.eval()
with torch.no_grad():
    for row in train_df.head(30).itertuples():
        enc = mt_tokenizer(row.src_en, return_tensors="pt",
                           truncation=True, max_length=256).to(device)
        ids = mt_model.generate(
            **enc, do_sample=True, top_p=TOP_P, temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOK, max_length=None,
            forced_bos_token_id=forced_bos, num_return_sequences=1, num_beams=1,
        )
        hyp = mt_tokenizer.decode(ids[0], skip_special_tokens=True)
        if hyp and len(hyp.strip()) >= 3:
            a   = chrf_metric.sentence_score(hyp, [row.ref_fr]).score / 100.0
            raw = _log_ppl(hyp, lm_model, lm_tokenizer)
            if raw is not None:
                f = normalise_flu(-raw)
                _warmup_rewards.append(args.lambda_val * a + (1 - args.lambda_val) * f)

reward_baseline = float(np.mean(_warmup_rewards)) if _warmup_rewards else 0.3
print(f"EMA baseline initialised at {reward_baseline:.4f}")
mt_model.train()


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_batch(hypotheses, references):
    """Returns per-hypothesis rewards in [0,1], both terms on same scale."""
    rewards = []
    for hyp, ref in zip(hypotheses, references):
        if not hyp or len(hyp.strip()) < 3:
            rewards.append(0.0)   # maps to worst reward after EMA subtraction
            continue
        a   = chrf_metric.sentence_score(hyp, [ref]).score / 100.0
        raw = _log_ppl(hyp, lm_model, lm_tokenizer)
        f   = normalise_flu(-raw) if raw is not None else 0.0
        rewards.append(args.lambda_val * a + (1 - args.lambda_val) * f)
    return torch.tensor(rewards, dtype=torch.float32)


# ── Batched log-prob ──────────────────────────────────────────────────────────

def batched_log_probs(src_ids: torch.Tensor, sample_ids: torch.Tensor) -> torch.Tensor:
    N = sample_ids.size(0)
    hyp_padded   = pad_sequence([sample_ids[i] for i in range(N)],
                                batch_first=True,
                                padding_value=mt_tokenizer.pad_token_id)
    src_expanded = src_ids.expand(N, -1)
    labels       = hyp_padded.clone()
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

def mrt_step(src_batch: list[str], ref_batch: list[str]) -> tuple[float, float]:
    global reward_baseline
    total_loss = torch.tensor(0.0, device=device)
    batch_reward = 0.0

    for src, ref in zip(src_batch, ref_batch):
        src_enc = mt_tokenizer(src, return_tensors="pt",
                               truncation=True, max_length=256).to(device)
        src_ids = src_enc["input_ids"]

        with torch.no_grad():
            sample_ids = mt_model.generate(
                **src_enc,
                do_sample=True, top_p=TOP_P, temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOK, max_length=None,
                forced_bos_token_id=forced_bos,
                num_return_sequences=args.n_samples, num_beams=1,
            )

        hypotheses = mt_tokenizer.batch_decode(sample_ids, skip_special_tokens=True)
        rewards    = score_batch(hypotheses, [ref] * args.n_samples)   # [0,1] scale

        # EMA baseline: subtract running mean, not within-batch mean
        mean_r = rewards.mean().item()
        reward_baseline = args.ema_decay * reward_baseline + (1 - args.ema_decay) * mean_r
        advantages = (rewards - reward_baseline).to(device)
        batch_reward += mean_r

        log_probs      = batched_log_probs(src_ids, sample_ids)
        reinforce_loss = -(advantages.detach() * log_probs).mean()

        ref_ids  = mt_tokenizer(ref, return_tensors="pt",
                                truncation=True, max_length=256).to(device)["input_ids"]
        mle_loss = mt_model(input_ids=src_ids, labels=ref_ids).loss

        total_loss = total_loss + args.alpha * reinforce_loss + (1 - args.alpha) * mle_loss

    total_loss = total_loss / len(src_batch)
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(mt_model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    return total_loss.item(), batch_reward / len(src_batch)


# ── Per-epoch sanity check ────────────────────────────────────────────────────

def epoch_sanity_check(epoch: int) -> float:
    """Generate greedy decodes on 30 val sentences, return mean chrF."""
    mt_model.eval()
    chrf_scores = []
    with torch.no_grad():
        for row in val_df.itertuples():
            enc = mt_tokenizer(row.src_en, return_tensors="pt",
                               truncation=True, max_length=256).to(device)
            ids = mt_model.generate(
                **enc, do_sample=False, num_beams=4,
                max_new_tokens=MAX_NEW_TOK, max_length=None,
                forced_bos_token_id=forced_bos,
            )
            hyp = mt_tokenizer.decode(ids[0], skip_special_tokens=True)
            if hyp.strip():
                chrf_scores.append(chrf_metric.sentence_score(hyp, [row.ref_fr]).score)
    mt_model.train()
    mean_chrf = float(np.mean(chrf_scores)) if chrf_scores else 0.0
    print(f"Epoch {epoch} sanity | mean chrF: {mean_chrf:.2f} | EMA baseline: {reward_baseline:.4f}")
    if mean_chrf < 45.0:
        print("  ⚠️  WARNING: chrF below 45 — possible collapse. Consider stopping.")
    return mean_chrf


# ── Training loop ─────────────────────────────────────────────────────────────

log_rows = []

for epoch in range(1, args.epochs + 1):
    shuffled     = train_df.sample(frac=1, random_state=SEED + epoch)
    epoch_losses = []
    epoch_rewards = []

    for i in tqdm(range(0, len(shuffled), args.batch_size),
                  desc=f"Epoch {epoch}/{args.epochs}"):
        batch  = shuffled.iloc[i : i + args.batch_size]
        loss, reward = mrt_step(batch["src_en"].tolist(), batch["ref_fr"].tolist())
        epoch_losses.append(loss)
        epoch_rewards.append(reward)

    mean_loss   = float(np.mean(epoch_losses))
    mean_reward = float(np.mean(epoch_rewards))
    print(f"Epoch {epoch} | loss: {mean_loss:.4f} | mean reward: {mean_reward:.4f}")

    val_chrf = epoch_sanity_check(epoch)
    log_rows.append({
        "epoch": epoch, "mean_loss": mean_loss,
        "mean_reward": mean_reward, "val_chrf": val_chrf,
    })


# ── Save ──────────────────────────────────────────────────────────────────────

mt_model.save_pretrained(ckpt_dir)   # saves LoRA adapter only (small)
mt_tokenizer.save_pretrained(ckpt_dir)
print(f"Saved LoRA checkpoint to {ckpt_dir}")

pd.DataFrame(log_rows).to_csv(OUT_DIR / "training_log.csv", index=False)

with open(OUT_DIR / "run_meta.json", "w") as f:
    json.dump({
        "domain":       args.domain,
        "lambda_val":   args.lambda_val,
        "base_model":   MT_MODEL,
        "ft_ckpt":      str(FT_CKPT),
        "train_size":   args.train_size,
        "epochs":       args.epochs,
        "n_samples":    args.n_samples,
        "alpha":        args.alpha,
        "lr":           args.lr,
        "warmup_steps": args.warmup_steps,
        "ema_decay":    args.ema_decay,
        "flu_min":      args.flu_min,
        "flu_max":      args.flu_max,
        "lora_r":       16,
        "lora_alpha":   32,
    }, f, indent=2)

print("Done.")