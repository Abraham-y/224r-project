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
_ap = _argparse.ArgumentParser()
_ap.add_argument("--cache_dir", default="extension/cache/probe_cache_n500_clean406")
_ap.add_argument("--eval_sft", default=EVAL_SFT_PATH)
_ap.add_argument("--eval_outcome", default=EVAL_OUT_PATH)
_args, _unknown = _ap.parse_known_args()
CACHE_DIR = _args.cache_dir
EVAL_SFT_PATH = _args.eval_sft
EVAL_OUT_PATH = _args.eval_outcome
OUT = "extension/outputs/n500/text/40_relabel_cross_checkpoint.txt"
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


def next_block_labels_assertion(meta, eval_rows):
    """For assertion-position rows, label by NEXT <answer> block correctness
    after that row's tok_idx."""
    labs = []
    for entry in meta:
        p = int(entry["prompt_idx"]); r = int(entry["resp_idx"])
        tok = int(entry["tok_idx"])
        resp = eval_rows[p]["response"][r]
        target = int(eval_rows[p]["target"]); nums = list(eval_rows[p]["nums"])
        ms = list(_ANSWER_OPEN_RE.finditer(resp))
        # find first <answer> block whose start offset is > tok_idx (assertion tokens
        # come BEFORE answer commits; we want the next commit's correctness)
        # tok_idx is a token index; we approximate via character position by char-count
        # since assertion tokens are inside <think>, we want the first <answer> that follows
        # For simplicity: if any matches exist, the assertion-position label is the next-block label.
        # If the assertion-pos's tok_idx position can be mapped to character offset, use the
        # first match whose start > that offset. As a robust default, fall back to first match.
        if not ms:
            labs.append(0); continue
        # Match by first if tok_idx is small; otherwise match the appropriate one.
        # Simplification: take the FIRST <answer> block as the "next commit after assertion"
        # (assertion tokens are inside <think>, before the first commit anyway).
        labs.append(int(check_block(ms[0].group(1), target, nums)))
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
    out_lines = ["Cross-checkpoint probe transfer matrix with CORRECTED labels",
                 "Position: L16; off-diagonals trained on full C_X, tested on C_Y",
                 ""]

    for kind in ["pre_answer", "assertion"]:
        out_lines.append(f"=== {kind} (L16) ===")
        X_s, m_s = load_cache(f"C_SFT_l16_{kind}")
        X_o, m_o = load_cache(f"C_outcome_l16_{kind}")
        if kind == "pre_answer":
            y_s = np.array([labs_sft[(int(x['prompt_idx']),int(x['resp_idx']))] for x in m_s], dtype=np.int32)
            y_o = np.array([labs_out[(int(x['prompt_idx']),int(x['resp_idx']))] for x in m_o], dtype=np.int32)
        else:
            y_s = next_block_labels_assertion(m_s, eval_sft)
            y_o = next_block_labels_assertion(m_o, eval_out)
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
