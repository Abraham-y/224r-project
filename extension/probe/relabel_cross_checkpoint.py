"""Cross-checkpoint probe transfer matrix with CORRECTED labels.

Re-runs §2.7 of the writeup: train probe on C_X's cache + C_X's
corrected first-block labels, evaluate on C_Y's cache + C_Y's
corrected labels. Diagonals are held-out via GroupKFold(5);
off-diagonals are trained on full C_X data, tested on C_Y data.

For both pre_answer (at </think>) and assertion-position probes, L16.
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

import argparse as _argparse

# Defaults must exist BEFORE the parser references them. (Previously these were
# assigned after the add_argument calls, which raised NameError at import time
# and made this script unrunnable.)
_DEFAULT_CACHE_DIR = "extension/cache/probe_cache_n500_clean406"
_DEFAULT_EVAL_SFT = "eval_c_sft_n500.json"
_DEFAULT_EVAL_OUT = "eval_c_outcome_n500.json"
_DEFAULT_TOKENIZER = "asingh15/qwen-sft-countdown-defaultproj"

_ap = _argparse.ArgumentParser()
_ap.add_argument("--cache_dir", default=_DEFAULT_CACHE_DIR)
_ap.add_argument("--eval_sft", default=_DEFAULT_EVAL_SFT)
_ap.add_argument("--eval_outcome", default=_DEFAULT_EVAL_OUT)
_ap.add_argument("--tokenizer", default=_DEFAULT_TOKENIZER)
_ap.add_argument("--out", default="extension/outputs/n500/text/40_relabel_cross_checkpoint.txt")
_args, _unknown = _ap.parse_known_args()
CACHE_DIR = _args.cache_dir
EVAL_SFT_PATH = _args.eval_sft
EVAL_OUT_PATH = _args.eval_outcome
TOKENIZER_NAME = _args.tokenizer
OUT = _args.out
_ANSWER_OPEN_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def check_block(eq, target, nums):
    try:
        nums_in_eq = [int(n) for n in re.findall(r"\d+", eq.strip())]
        if sorted(nums_in_eq) != sorted([int(x) for x in nums]):
            return False
        if not re.match(r"^[\d+\-*/().\s]+$", eq.strip()):
            return False
        result = eval(eq.strip(), {"__builtins__": None}, {})
        return abs(result - int(target)) < 1e-5
    except Exception:
        return False


def first_block_labels(eval_path):
    rows = [json.loads(l) for l in open(eval_path) if l.strip()]
    labs = {}
    for p, row in enumerate(rows):
        target = int(row["target"]); nums = list(row["nums"])
        for r_i, resp in enumerate(row["response"]):
            m = _ANSWER_OPEN_RE.search(resp)
            labs[(p, r_i)] = int(bool(m) and check_block(m.group(1), target, nums))
    return labs


def next_block_labels_assertion(meta, eval_rows, tokenizer):
    """For assertion-position rows, label by NEXT <answer> block correctness
    after that row's tok_idx.

    This maps tok_idx -> character offset via the tokenizer's offset_mapping on
    (prompt + response), exactly as relabel_full_grid.py and
    relabel_redo_downstream.py do, then takes the first <answer> block opening
    strictly after that character position.

    The earlier implementation ignored tok_idx entirely and labelled every
    assertion row by the FIRST <answer> block, which is a different label
    definition from the sibling relabel scripts and made the assertion row of
    this transfer table incomparable with the rest of the paper.
    """
    labs = []
    offsets_cache: dict[tuple[int, int], list[tuple[int, int]]] = {}
    n_no_next = 0
    for entry in meta:
        p = int(entry["prompt_idx"]); r = int(entry["resp_idx"])
        tok = int(entry["tok_idx"])
        row = eval_rows[p]
        prompt_text = row["prompt"]
        resp = row["response"][r]
        target = int(row["target"]); nums = list(row["nums"])
        scored_correct = int(float(row["scores"][r]) == 1.0)

        # Block openings, in char offsets of (prompt + response).
        blocks = [(len(prompt_text) + m.start(), check_block(m.group(1), target, nums))
                  for m in _ANSWER_OPEN_RE.finditer(resp)]
        if not blocks:
            labs.append(scored_correct); n_no_next += 1; continue

        if (p, r) not in offsets_cache:
            enc = tokenizer(prompt_text + resp, return_offsets_mapping=True,
                            truncation=True, max_length=2048)
            offsets_cache[(p, r)] = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
        offs = offsets_cache[(p, r)]
        if tok >= len(offs):
            labs.append(scored_correct); n_no_next += 1; continue
        char_pos = offs[tok][0]

        nxt = next((bc for char_open, bc in blocks if char_open > char_pos), None)
        if nxt is None:
            labs.append(scored_correct); n_no_next += 1
        else:
            labs.append(int(nxt))
    if n_no_next:
        print(f"    [assertion labels] {n_no_next}/{len(meta)} rows had no following "
              f"<answer> block; fell back to the verifier's rollout-final score.")
    return np.array(labs, dtype=np.int32)


def load_cache(name):
    p = f"{CACHE_DIR}/{name}.npz"
    with np.load(p) as d: X = d["X"]
    m = json.load(open(f"{CACHE_DIR}/{name}.meta.json"))
    return X, m


def balanced_auroc(y, sc, seed=0):
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    nb = min(len(pos), len(neg))
    if nb == 0: return float("nan")
    rng = np.random.RandomState(seed)
    idx = np.concatenate([rng.choice(pos, nb, replace=False), rng.choice(neg, nb, replace=False)])
    return float(roc_auc_score(y[idx], sc[idx]))


def main():
    labs_sft = first_block_labels(EVAL_SFT_PATH)
    labs_out = first_block_labels(EVAL_OUT_PATH)
    eval_sft = [json.loads(l) for l in open(EVAL_SFT_PATH) if l.strip()]
    eval_out = [json.loads(l) for l in open(EVAL_OUT_PATH) if l.strip()]
    tokenizer = None  # loaded lazily; only the assertion position needs it
    out_lines = ["Cross-checkpoint probe transfer matrix with CORRECTED labels",
                 "Position: L16; off-diagonals trained on full C_X, tested on C_Y",
                 f"Labels: pre_answer = first-<answer>-block correctness; "
                 f"assertion = correctness of the NEXT <answer> block after the cached token",
                 ""]

    for kind in ["pre_answer", "assertion"]:
        out_lines.append(f"=== {kind} (L16) ===")
        X_s, m_s = load_cache(f"C_SFT_l16_{kind}")
        X_o, m_o = load_cache(f"C_outcome_l16_{kind}")
        if kind == "pre_answer":
            y_s = np.array([labs_sft[(int(x['prompt_idx']),int(x['resp_idx']))] for x in m_s], dtype=np.int32)
            y_o = np.array([labs_out[(int(x['prompt_idx']),int(x['resp_idx']))] for x in m_o], dtype=np.int32)
        else:
            if tokenizer is None:
                from transformers import AutoTokenizer
                print(f"[cross-ckpt] loading tokenizer {TOKENIZER_NAME} for assertion labels", flush=True)
                tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
            y_s = next_block_labels_assertion(m_s, eval_sft, tokenizer)
            y_o = next_block_labels_assertion(m_o, eval_out, tokenizer)
        g_s = np.array([int(x['prompt_idx']) for x in m_s])
        g_o = np.array([int(x['prompt_idx']) for x in m_o])

        # Diagonal: held-out via GroupKFold
        diag_s = np.full(len(y_s), np.nan)
        for tr, te in GroupKFold(5).split(X_s, y_s, g_s):
            pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
            pipe.fit(X_s[tr], y_s[tr]); diag_s[te] = pipe.predict_proba(X_s[te])[:, 1]
        diag_o = np.full(len(y_o), np.nan)
        for tr, te in GroupKFold(5).split(X_o, y_o, g_o):
            pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
            pipe.fit(X_o[tr], y_o[tr]); diag_o[te] = pipe.predict_proba(X_o[te])[:, 1]

        # Off-diagonals: train on full C_X, test on C_Y
        pipe_s = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
        pipe_s.fit(X_s, y_s)
        off_so = pipe_s.predict_proba(X_o)[:, 1]
        pipe_o = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
        pipe_o.fit(X_o, y_o)
        off_os = pipe_o.predict_proba(X_s)[:, 1]

        diag_sft_auc = balanced_auroc(y_s, diag_s)
        diag_out_auc = balanced_auroc(y_o, diag_o)
        off_sft_to_out_auc = balanced_auroc(y_o, off_so)
        off_out_to_sft_auc = balanced_auroc(y_s, off_os)

        out_lines.append(f"  Diagonal C_SFT (held-out):  {diag_sft_auc:.3f}")
        out_lines.append(f"  Diagonal C_outcome (held-out): {diag_out_auc:.3f}")
        out_lines.append(f"  Off-diag C_SFT->C_outcome:  {off_sft_to_out_auc:.3f}")
        out_lines.append(f"  Off-diag C_outcome->C_SFT:  {off_out_to_sft_auc:.3f}")
        out_lines.append("")
        print(f"=== {kind} ===")
        print(f"  C_SFT diag:        {diag_sft_auc:.3f}")
        print(f"  C_outcome diag:    {diag_out_auc:.3f}")
        print(f"  SFT->outcome:      {off_sft_to_out_auc:.3f}")
        print(f"  outcome->SFT:      {off_out_to_sft_auc:.3f}")

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
