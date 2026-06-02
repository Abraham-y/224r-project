"""Train the C_outcome pre_answer L16 probe with CORRECTED labels (next-block
correctness) and save it as a pickled sklearn Pipeline for use as a reward
function during RL training.

Outputs:
  extension/cache/steering/probe_pipeline_C_outcome_l16_pre_answer.pkl
    - sklearn Pipeline(StandardScaler, LogisticRegression)
  extension/cache/steering/probe_pipeline_meta.json
    - probe_auroc, train_n_pos, train_n_neg, label rule
"""

from __future__ import annotations

import json
import os
import pickle
import re
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

CACHE_NPZ = "extension/cache/probe_cache_n500_clean406/C_outcome_l16_pre_answer.npz"
META = CACHE_NPZ.replace(".npz", ".meta.json")
EVAL = "eval_c_outcome_n500.json"
OUT_DIR = "extension/cache/steering"
PKL = f"{OUT_DIR}/probe_pipeline_C_outcome_l16_pre_answer.pkl"
META_OUT = f"{OUT_DIR}/probe_pipeline_meta.json"

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


def main():
    with np.load(CACHE_NPZ) as d: X = d["X"]
    meta = json.load(open(META))
    groups = np.array([int(m["prompt_idx"]) for m in meta])
    rows = [json.loads(l) for l in open(EVAL) if l.strip()]

    # Corrected labels: first-<answer>-block correctness of the rollout
    labs = {}
    for p, row in enumerate(rows):
        target = int(row["target"]); nums = list(row["nums"])
        for r_i, resp in enumerate(row["response"]):
            m = _ANSWER_OPEN_RE.search(resp)
            labs[(p, r_i)] = int(bool(m) and check_block(m.group(1), target, nums))
    y = np.array([labs.get((int(x['prompt_idx']), int(x['resp_idx'])), 0) for x in meta], dtype=np.int32)

    print(f"n_rows={len(y)}, n_pos={y.sum()}, n_neg={(1-y).sum()}, pos_frac={y.mean():.3f}")

    # Diagnostic: held-out balanced AUROC
    scores = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
        pipe.fit(X[tr], y[tr])
        scores[te] = pipe.predict_proba(X[te])[:, 1]
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    nb = min(len(pos), len(neg))
    rng = np.random.RandomState(0)
    idx = np.concatenate([rng.choice(pos, nb, replace=False), rng.choice(neg, nb, replace=False)])
    auroc = float(roc_auc_score(y[idx], scores[idx]))
    print(f"held-out balanced AUROC (corrected labels): {auroc:.3f}")

    # Final fit on the BALANCED subsample (matches the analysis pipeline)
    Xs, ys = X[idx], y[idx]
    pipe_final = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
    pipe_final.fit(Xs, ys)
    print(f"final probe fit on balanced subsample (n_pos={ys.sum()}, n_neg={(1-ys).sum()})")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PKL, "wb") as f:
        pickle.dump(pipe_final, f)
    with open(META_OUT, "w") as f:
        json.dump({
            "checkpoint": "C_outcome",
            "layer": 16,
            "position": "pre_answer (</think> token)",
            "label_rule": "first-<answer>-block correctness (corrected)",
            "auroc_heldout_balanced": auroc,
            "n_train": int(len(Xs)),
            "n_pos_train": int(ys.sum()),
            "n_neg_train": int((1-ys).sum()),
        }, f, indent=2)
    print(f"wrote {PKL}")
    print(f"wrote {META_OUT}")


if __name__ == "__main__":
    main()
