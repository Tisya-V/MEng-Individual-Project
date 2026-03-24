# finetune_domain.py
#
# Fine-tunes mBART-large-50 with LoRA on each domain using train/val splits.
# Produces one adapter checkpoint per domain, saved to models/{name}/
#
# Usage:
#   python finetune_domain.py                  # fine-tune all domains
#   python finetune_domain.py --domain emea    # fine-tune one domain

import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Config ────────────────────────────────────────────────────────────────────

SEED         = 42
MT_MODEL     = "facebook/mbart-large-50-many-to-many-mmt"
SRC_LANG     = "en_XX"
TGT_LANG     = "fr_XX"
SPLITS_DIR   = Path("data/splits")
MODELS_DIR   = Path("models");  MODELS_DIR.mkdir(exist_ok=True)

MAX_SRC_LEN  = 256
MAX_TGT_LEN  = 256
BATCH_SIZE   = 8
MAX_EPOCHS   = 5
LR           = 5e-4
PATIENCE     = 2          # early stopping: stop after this many epochs without val improvement
WARMUP_STEPS = 100

LORA_CONFIG = LoraConfig(
    task_type    = TaskType.SEQ_2_SEQ_LM,
    r            = 16,
    lora_alpha   = 32,
    lora_dropout = 0.1,
    target_modules = ["q_proj", "v_proj"],
)

DOMAINS = ["emea", "news_commentary", "opus_books"]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Dataset ───────────────────────────────────────────────────────────────────

class TranslationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_src: int, max_tgt: int):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_src   = max_src
        self.max_tgt   = max_tgt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        model_inputs = self.tokenizer(
            row["src_en"],
            text_target=row["ref_fr"],      # handles target tokenisation internally
            max_length=self.max_src,
            max_target_length=self.max_tgt,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        label_ids = model_inputs["labels"].squeeze()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids":      model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels":         label_ids,
        }

# ── Training ──────────────────────────────────────────────────────────────────

def train_domain(domain: str):
    out_dir = MODELS_DIR / domain
    if (out_dir / "adapter_config.json").exists():
        print(f"[{domain}] Checkpoint found at {out_dir} — skipping.")
        return

    print(f"\n[{domain}] Loading splits...")
    train_df = pd.read_csv(SPLITS_DIR / f"{domain}_train.csv")
    val_df   = pd.read_csv(SPLITS_DIR / f"{domain}_val.csv")
    print(f"  Train: {len(train_df)}  Val: {len(val_df)}")

    print(f"[{domain}] Loading tokeniser + model...")
    tokenizer = MBart50TokenizerFast.from_pretrained(MT_MODEL)
    tokenizer.src_lang = SRC_LANG
    forced_bos = tokenizer.lang_code_to_id[TGT_LANG]

    model = MBartForConditionalGeneration.from_pretrained(MT_MODEL)
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    model = model.to(device)

    train_ds = TranslationDataset(train_df, tokenizer, MAX_SRC_LEN, MAX_TGT_LEN)
    val_ds   = TranslationDataset(val_df,   tokenizer, MAX_SRC_LEN, MAX_TGT_LEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_dl) * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_dl, desc=f"[{domain}] Epoch {epoch}/{MAX_EPOCHS} train"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, forced_bos_token_id=forced_bos)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"[{domain}] Epoch {epoch}/{MAX_EPOCHS} val"):
                batch = {k: v.to(device) for k, v in batch.items()}
                val_loss += model(**batch, forced_bos_token_id=forced_bos).loss.item()
        val_loss /= len(val_dl)

        print(f"[{domain}] Epoch {epoch} — train loss: {train_loss:.4f}  val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"  ✓ New best — saved to {out_dir}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{PATIENCE})")
            if epochs_no_improve >= PATIENCE:
                print(f"[{domain}] Early stopping at epoch {epoch}.")
                break

    print(f"[{domain}] Done. Best val loss: {best_val_loss:.4f}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default=None,
                        help="Single domain to fine-tune (default: all)")
    args = parser.parse_args()

    domains = [args.domain] if args.domain else DOMAINS
    for domain in domains:
        if domain not in DOMAINS:
            raise ValueError(f"Unknown domain '{domain}'. Choose from {DOMAINS}")
        train_domain(domain)

if __name__ == "__main__":
    main()
