"""Re-label assertion-keyword probe rows by the correctness of the NEXT
`<answer>` block following each assertion, not the rollout's final
verifier score.

Existing cache (`extension/cache/probe_cache_n500_clean406/C_outcome_l16_
assertion.npz`) labels every assertion-token row with the rollout's
verifier-scored correctness. That's correct for the final assertion in
a rollout but wrong for assertions that precede non-final `<answer>`
blocks: the user pointed out (correctly) that "this works" tokens in
multi-answer rollouts are claiming confidence about the NEXT answer
block, not the one the verifier scores at the very end.

This script:
  1. For each cached assertion row, recovers the assertion's character
     position in the rollout by re-tokenizing.
  2. Finds the immediately-following `<answer>` block.
  3. Looks up that block's correctness from per_answer_correctness.jsonl
     (Phase 2A).
  4. Re-trains the probe with corrected labels via GroupKFold(5) and
     reports the new AUROC.

If the new AUROC is HIGHER than the original (0.703 at C_outcome L16),
the original label noise was deflating the assertion-position probe and
the "position-decoupling" gap is partly an artifact.
"""

from __future__ import annotations

import json
import os
import re
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ASSERT_CACHE = "extension/cache/probe_cache_n500_clean406/C_outcome_l16_assertion.npz"
ASSERT_META = ASSERT_CACHE.replace(".npz", ".meta.json")
EVAL_JSON = "eval_c_outcome_n500.json"
PER_ANSWER_JSONL = "extension/outputs/n500/per_answer_correctness.jsonl"

_ANSWER_OPEN_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def main():
    print("[relabel] loading existing assertion cache + meta", flush=True)
    with np.load(ASSERT_CACHE) as d:
        X = d["X"]; y_orig = d["y"]
    meta = json.load(open(ASSERT_META))
    print(f"[relabel] {len(meta)} assertion-token rows")

    print("[relabel] loading rollouts + per-answer correctness", flush=True)
    rows = [json.loads(l) for l in open(EVAL_JSON) if l.strip()]
    # per_answer_correctness.jsonl: one entry per multi-answer rollout
    pac = [json.loads(l) for l in open(PER_ANSWER_JSONL) if l.strip()]
    pac_by_rollout = {(int(p["prompt_idx"]), int(p["resp_idx"])): p for p in pac}

    print(f"[relabel] {len(pac_by_rollout)} multi-answer rollouts in per_answer_correctness")

    # Load tokenizer (only needed to convert tok_idx -> char position)
    print("[relabel] loading tokenizer", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("asingh15/qwen-sft-countdown-defaultproj", use_fast=True)

    # Build a (prompt_idx, resp_idx) -> char positions of <answer> opens
    # and corresponding per-block correctness
    rollout_info: dict = {}  # (p, r) -> {"answer_open_chars": [(c, block_correct), ...], "scored_correct": bool}
    for p_idx, row in enumerate(rows):
        prompt_text = row["prompt"]
        for r_idx, resp in enumerate(row["response"]):
            scored_correct = bool(row["scores"][r_idx] == 1.0)
            blocks = []  # list of (char_open, char_close, block_correct)
            # per-block correctness from pac if available
            pac_entry = pac_by_rollout.get((p_idx, r_idx))
            per_block_correct = pac_entry["per_block_correct"] if pac_entry else None
            for k, m in enumerate(_ANSWER_OPEN_RE.finditer(resp)):
                bc = per_block_correct[k] if (per_block_correct and k < len(per_block_correct)) else scored_correct
                # absolute char position in (prompt + response)
                char_open = len(prompt_text) + m.start()
                blocks.append((char_open, bool(bc)))
            rollout_info[(p_idx, r_idx)] = {
                "blocks": blocks,
                "scored_correct": scored_correct,
                "prompt_text": prompt_text,
                "response": resp,
            }

    # For each assertion row, find next <answer> block after the assertion's char
    print("[relabel] mapping each assertion-token row to its next <answer> block", flush=True)
    y_new = np.full(len(meta), -1, dtype=np.int32)
    n_relabeled_diff = 0; n_no_next = 0
    cache_offsets: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for i, m in enumerate(meta):
        p_idx = int(m["prompt_idx"])
        r_idx = int(m["resp_idx"])
        tok_idx = int(m["tok_idx"])
        ri = rollout_info.get((p_idx, r_idx))
        if ri is None:
            y_new[i] = int(y_orig[i])
            continue
        # Tokenize prompt + response to recover offset_mapping (cache once per rollout)
        key = (p_idx, r_idx)
        if key not in cache_offsets:
            full = ri["prompt_text"] + ri["response"]
            enc = tokenizer(full, return_offsets_mapping=True, truncation=True, max_length=2048)
            cache_offsets[key] = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
        offsets = cache_offsets[key]
        if tok_idx >= len(offsets):
            y_new[i] = int(y_orig[i])
            continue
        char_assertion = offsets[tok_idx][0]
        # Find next <answer> block whose char_open > char_assertion
        next_block = None
        for char_open, bc in ri["blocks"]:
            if char_open > char_assertion:
                next_block = bc
                break
        if next_block is None:
            # No next block; fall back to scored-correct (verifier's last)
            y_new[i] = int(ri["scored_correct"])
            n_no_next += 1
        else:
            y_new[i] = int(next_block)
        if y_new[i] != int(y_orig[i]):
            n_relabeled_diff += 1

    print(f"[relabel] {n_relabeled_diff}/{len(meta)} rows have a different label after re-labeling")
    print(f"[relabel] {n_no_next} rows had no <answer> block after the assertion (used verifier's score)")
    print(f"[relabel] original pos%={float(y_orig.mean()):.3f}; new pos%={float(y_new.mean()):.3f}")

    # Re-train probe with GroupKFold(5) for both label schemes
    groups = np.array([int(m["prompt_idx"]) for m in meta])
    print("[relabel] training probes (original labels vs corrected labels)", flush=True)

    def heldout_auroc(y_arr):
        # balanced subsample
        rng = np.random.RandomState(0)
        pos = np.where(y_arr == 1)[0]
        neg = np.where(y_arr == 0)[0]
        nb = min(len(pos), len(neg))
        if nb < 5: return float("nan"), 0
        idx = np.concatenate([rng.choice(pos, nb, replace=False), rng.choice(neg, nb, replace=False)])
        Xs, ys, gs = X[idx], y_arr[idx], groups[idx]
        preds = np.full(len(ys), np.nan)
        for tr, te in GroupKFold(5).split(Xs, ys, gs):
            pipe = Pipeline([("sc", StandardScaler()),
                             ("lr", LogisticRegression(C=0.1, max_iter=2000))])
            pipe.fit(Xs[tr], ys[tr])
            preds[te] = pipe.predict_proba(Xs[te])[:, 1]
        return float(roc_auc_score(ys, preds)), nb

    auc_orig, nb_orig = heldout_auroc(y_orig)
    auc_new, nb_new = heldout_auroc(y_new)

    print("\n=== Headline ===")
    print(f"  Original labels (verifier-scored final-answer correctness):")
    print(f"    assertion-position AUROC @ L16 = {auc_orig:.4f}   (n_balanced/class = {nb_orig})")
    print(f"  Corrected labels (next-<answer>-block correctness):")
    print(f"    assertion-position AUROC @ L16 = {auc_new:.4f}   (n_balanced/class = {nb_new})")
    print(f"  Change: {auc_new - auc_orig:+.4f}")

    out = "extension/outputs/n500/text/28_assertion_relabel.txt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(f"Assertion-position relabel experiment on C_outcome 0.5B L16 clean-406\n\n")
        f.write(f"  Original labels (verifier-scored final correctness):\n")
        f.write(f"    n_balanced/class = {nb_orig}\n")
        f.write(f"    AUROC = {auc_orig:.4f}\n\n")
        f.write(f"  Corrected labels (next-<answer>-block correctness):\n")
        f.write(f"    n_balanced/class = {nb_new}\n")
        f.write(f"    AUROC = {auc_new:.4f}\n\n")
        f.write(f"  Δ AUROC: {auc_new - auc_orig:+.4f}\n")
        f.write(f"  {n_relabeled_diff}/{len(meta)} rows had different labels after correction\n")
        f.write(f"  Original pos%: {float(y_orig.mean()):.3f}; New pos%: {float(y_new.mean()):.3f}\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
