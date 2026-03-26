# stages.py — shared stage functions for pipeline_af_front.py and pipeline_finetuned_front.py

import numpy as np
import pandas as pd
import torch
import sacrebleu
from tqdm.auto import tqdm
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _is_valid(text) -> bool:
    """Returns False for None, NaN, or whitespace-only strings."""
    if text is None:
        return False
    if not isinstance(text, str):
        try:
            if np.isnan(text):
                return False
        except (TypeError, ValueError):
            pass
    return bool(str(text).strip())


def _log_ppl(text: str, lm_model, lm_tokenizer) -> float | None:
    """Returns log-perplexity (cross-entropy loss) or None if text is invalid/empty."""
    if not _is_valid(text):
        return None
    enc = lm_tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    if enc["input_ids"].shape[1] == 0:
        return None
    with torch.no_grad():
        loss = lm_model(**enc, labels=enc["input_ids"]).loss.item()
    return None if np.isnan(loss) else loss


def generate_candidates(name, eval_df, mt_model, mt_tokenizer, data_dir, tag=""):
    path = data_dir / f"{name}_candidates.jsonl"
    if path.exists():
        print(f"[{name}][{tag}Stage 2] Skipping — found {path}")
        return pd.read_json(path, orient="records", lines=True)
    forced_bos = mt_tokenizer.lang_code_to_id[TGT_LANG]

    def sample_k(src):
        if not _is_valid(src):
            return [""] * K
        inputs = mt_tokenizer(src, return_tensors="pt",
                              truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = mt_model.generate(
                **inputs, do_sample=True, top_p=TOP_P, temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOK, max_length=None, forced_bos_token_id=forced_bos,
                num_return_sequences=K, num_beams=1,
            )
        return [mt_tokenizer.decode(o, skip_special_tokens=True) for o in out]

    rows = []
    for row in tqdm(eval_df.itertuples(index=False), total=len(eval_df),
                    desc=f"[{name}] {tag}generating"):
        for kid, hyp in enumerate(sample_k(row.src_en)):
            rows.append({"id": int(row.id), "cand_id": kid,
                         "src_en": row.src_en, "ref_fr": row.ref_fr, "hyp_fr": hyp})
    df = pd.DataFrame(rows)
    df.to_json(path, orient="records", lines=True, force_ascii=False)
    print(f"[{name}][{tag}Stage 2] Saved {len(df)} candidates.")
    return df


def generate_greedy(name, eval_df, mt_model, mt_tokenizer, data_dir, tag=""):
    path = data_dir / f"{name}_greedy.jsonl"
    if path.exists():
        print(f"[{name}][{tag}Stage 2b] Skipping — found {path}")
        return pd.read_json(path, orient="records", lines=True)
    forced_bos = mt_tokenizer.lang_code_to_id[TGT_LANG]
    rows = []
    for row in tqdm(eval_df.itertuples(index=False), total=len(eval_df),
                    desc=f"[{name}] {tag}greedy"):
        if not _is_valid(row.src_en):
            hyp = ""
        else:
            inputs = mt_tokenizer(row.src_en, return_tensors="pt",
                                  truncation=True, max_length=256).to(device)
            with torch.no_grad():
                out = mt_model.generate(
                    **inputs, do_sample=False, num_beams=4,
                    max_new_tokens=MAX_NEW_TOK, max_length=None,
                    forced_bos_token_id=forced_bos,
                )
            hyp = mt_tokenizer.decode(out[0], skip_special_tokens=True)
        rows.append({"id": int(row.id), "src_en": row.src_en,
                     "ref_fr": row.ref_fr, "hyp_fr": hyp})
    df = pd.DataFrame(rows)
    df.to_json(path, orient="records", lines=True, force_ascii=False)
    print(f"[{name}][{tag}Stage 2b] Saved {len(df)} greedy decodes.")
    return df


def score_adequacy(name, cand_df, data_dir, tag=""):
    path = data_dir / f"{name}_scored_adq.jsonl"
    if path.exists():
        print(f"[{name}][{tag}Stage 3] Skipping — found {path}")
        return pd.read_json(path, orient="records", lines=True)
    chrf_metric = sacrebleu.CHRF()

    def safe_chrf(hyp, ref):
        if not _is_valid(hyp) or not _is_valid(ref):
            return 0.0
        return chrf_metric.sentence_score(hyp, [ref]).score

    cand_df["chrf"] = [
        safe_chrf(row.hyp_fr, row.ref_fr)
        for row in tqdm(cand_df.itertuples(index=False), total=len(cand_df),
                        desc=f"[{name}] {tag}chrF")
    ]
    cand_df.to_json(path, orient="records", lines=True, force_ascii=False)
    print(f"[{name}][{tag}Stage 3] Saved adequacy scores.")
    return cand_df


def score_fluency(name, chrf_df, data_dir, lm_model, lm_tokenizer, tag=""):
    path = data_dir / f"{name}_scored_flu.jsonl"
    if path.exists():
        print(f"[{name}][{tag}Stage 4] Skipping — found {path}")
        return pd.read_json(path, orient="records", lines=True)

    losses = []
    skipped = 0
    for row in tqdm(chrf_df.itertuples(index=False), total=len(chrf_df),
                    desc=f"[{name}] {tag}fluency"):
        loss = _log_ppl(row.hyp_fr, lm_model, lm_tokenizer)
        if loss is None:
            skipped += 1
        losses.append(loss)

    if skipped:
        print(f"[{name}][{tag}Stage 4] WARNING: {skipped} rows had invalid/empty hypotheses — fluency set to NaN.")

    chrf_df["log_ppl"] = losses
    chrf_df["fluency"] = [-l if l is not None else float("nan") for l in losses]
    chrf_df.to_json(path, orient="records", lines=True, force_ascii=False)
    print(f"[{name}][{tag}Stage 4] Saved fluency scores.")
    return chrf_df


def score_greedy(name, greedy_df, data_dir, lm_model, lm_tokenizer, tag=""):
    path = data_dir / f"{name}_greedy_scored.jsonl"
    if path.exists():
        print(f"[{name}][{tag}greedy scoring] Skipping — found {path}")
        df = pd.read_json(path, orient="records", lines=True)
        return {"chrf": df["chrf"].mean(), "fluency": df["fluency"].mean()}

    chrf_metric = sacrebleu.CHRF()

    def safe_chrf(hyp, ref):
        if not _is_valid(hyp) or not _is_valid(ref):
            return 0.0
        return chrf_metric.sentence_score(hyp, [ref]).score

    greedy_df["chrf"] = [
        safe_chrf(r.hyp_fr, r.ref_fr)
        for r in tqdm(greedy_df.itertuples(index=False), total=len(greedy_df),
                      desc=f"[{name}] {tag}greedy chrF")
    ]

    losses = []
    skipped = 0
    for r in tqdm(greedy_df.itertuples(index=False), total=len(greedy_df),
                  desc=f"[{name}] {tag}greedy fluency"):
        loss = _log_ppl(r.hyp_fr, lm_model, lm_tokenizer)
        if loss is None:
            skipped += 1
        losses.append(loss)

    if skipped:
        print(f"[{name}][{tag}greedy scoring] WARNING: {skipped} empty/invalid hypotheses skipped.")

    greedy_df["log_ppl"] = losses
    greedy_df["fluency"] = [-l if l is not None else float("nan") for l in losses]
    greedy_df.to_json(path, orient="records", lines=True, force_ascii=False)
    print(f"[{name}][{tag}greedy scoring] Saved.")
    return {"chrf": greedy_df["chrf"].mean(), "fluency": greedy_df["fluency"].mean()}


def rerank(name, scored_df, result_dir, tag=""):
    path = result_dir / f"{name}_af_front.csv"
    if path.exists():
        print(f"[{name}][{tag}Stage 5] Skipping — found {path}")
        return pd.read_csv(path)

    # Drop rows where fluency is NaN before reranking
    n_before = len(scored_df)
    scored_df = scored_df.dropna(subset=["chrf", "fluency"]).copy()
    if len(scored_df) < n_before:
        print(f"[{name}][{tag}Stage 5] WARNING: dropped {n_before - len(scored_df)} NaN rows before reranking.")

    g = scored_df.groupby("id")
    scored_df["chrf_norm"]    = (scored_df["chrf"]    - g["chrf"].transform("min"))    / (g["chrf"].transform("max")    - g["chrf"].transform("min")    + 1e-9)
    scored_df["fluency_norm"] = (scored_df["fluency"] - g["fluency"].transform("min")) / (g["fluency"].transform("max") - g["fluency"].transform("min") + 1e-9)
    rows = []
    for beta in tqdm(BETAS, desc=f"[{name}] {tag}sweeping beta"):
        scored_df["score"] = beta * scored_df["chrf_norm"] + scored_df["fluency_norm"]
        best = scored_df.loc[scored_df.groupby("id")["score"].idxmax()]
        rows.append({"beta": beta, "chrf": best["chrf"].mean(), "fluency": best["fluency"].mean()})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"[{name}][{tag}Stage 5] Saved front.")
    return df
