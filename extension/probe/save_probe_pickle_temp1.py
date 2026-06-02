"""Train + pickle the C_outcome pre_answer L16 probe on TEMPERATURE-MATCHED
rollouts (sampled at temp=1.0, top_p=1.0 — matching RL's vLLM sampling regime)
with rollout-final labels.

This addresses the saturation issue observed in run10/run11: the original
probe was trained on temp=0.6/top_p=0.95 rollouts, whose hidden state
distribution at </think> differs enough from temp=1.0 rollouts that the
probe saturates to ~0.98 when applied online during RL training.

Outputs:
  extension/cache/steering/probe_pipeline_C_outcome_l16_pre_answer_temp1.pkl
  extension/cache/steering/probe_pipeline_temp1_meta.json
"""

from __future__ import annotations
import json, os, pickle, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

CACHE_NPZ = "extension/cache/probe_cache_temp1/C_outcome_temp1_l16_pre_answer.npz"
META = CACHE_NPZ.replace(".npz", ".meta.json")
EVAL = "eval_c_outcome_temp1_asingh_300.json"  # downloaded already? if not, fetch from Modal
OUT_DIR = "extension/cache/steering"
PKL = f"{OUT_DIR}/probe_pipeline_C_outcome_l16_pre_answer_temp1.pkl"
META_OUT = f"{OUT_DIR}/probe_pipeline_temp1_meta.json"


def main():
    if not os.path.exists(EVAL):
        # Fetch from Modal volume
        os.system(f"modal volume get default-proj-training evaluation/eval_results/eval_c_outcome_temp1_asingh_300.json {EVAL}")
    with np.load(CACHE_NPZ) as d: X = d["X"]
    meta = json.load(open(META))
    groups = np.array([int(m["prompt_idx"]) for m in meta])
    rows = [json.loads(l) for l in open(EVAL) if l.strip()]

    # Label by rollout-final verifier score (matches vanilla RLOO reward)
    labs = {}
    for p, row in enumerate(rows):
        for r_i, sc in enumerate(row["scores"]):
            labs[(p, r_i)] = int(sc == 1.0)
    y = np.array([labs.get((int(x['prompt_idx']), int(x['resp_idx'])), 0) for x in meta], dtype=np.int32)
    print(f"n_rows={len(y)}, n_pos={y.sum()}, n_neg={(1-y).sum()}, pos_frac={y.mean():.3f}")

    # Held-out balanced AUROC
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
    print(f"held-out balanced AUROC (temp=1.0 rollouts, rollout-final labels): {auroc:.3f}")

    # Final fit on balanced subsample
    Xs, ys = X[idx], y[idx]
    pipe_final = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
    pipe_final.fit(Xs, ys)
    print(f"final probe fit on balanced subsample (n_pos={ys.sum()}, n_neg={(1-ys).sum()})")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PKL, "wb") as f: pickle.dump(pipe_final, f)
    with open(META_OUT, "w") as f:
        json.dump({
            "checkpoint": "C_outcome",
            "layer": 16,
            "position": "pre_answer (</think> token)",
            "label_rule": "rollout-final verifier correctness",
            "sampling_regime": "temperature=1.0, top_p=1.0 (matches RL's vLLM)",
            "n_prompts": 300,
            "n_rollouts_per_prompt": 8,
            "auroc_heldout_balanced": auroc,
            "n_train": int(len(Xs)),
            "n_pos_train": int(ys.sum()),
            "n_neg_train": int((1-ys).sum()),
        }, f, indent=2)
    print(f"wrote {PKL}\nwrote {META_OUT}")


if __name__ == "__main__":
    main()
