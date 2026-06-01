"""Train the C_outcome pre_answer L16 probe and save its weight vector
in INPUT space (i.e., w / scaler.scale_), normalized to unit length.

Used by the causal-steering experiment as the steering direction.

Outputs:
  extension/cache/steering/C_outcome_l16_pre_answer_direction.npz
    - v_unit: (896,) float32 -- unit-norm correctness direction
    - v_input_raw: (896,) float32 -- raw input-space direction
    - h_mean_norm: float -- typical L2 norm of hidden states at this position
    - probe_auroc: float -- diagonal held-out AUROC for sanity
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

CACHE = "extension/cache/probe_cache_n500_clean406"
CKPT = "C_outcome"
LAYER = 16
KIND = "pre_answer"
OUT_DIR = "extension/cache/steering"


def main():
    npz = os.path.join(CACHE, f"{CKPT}_l{LAYER}_{KIND}.npz")
    meta_path = npz.replace(".npz", ".meta.json")
    with np.load(npz) as d:
        X = d["X"]; y = d["y"]
    meta = json.load(open(meta_path))
    groups = np.array([m["prompt_idx"] for m in meta])

    rng = np.random.RandomState(0)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    n = min(len(pos), len(neg))
    idx = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
    Xs, ys, gs = X[idx], y[idx], groups[idx]

    # Held-out diagonal AUROC for sanity
    preds = np.full(len(ys), np.nan)
    for tr, te in GroupKFold(5).split(Xs, ys, gs):
        sc = StandardScaler().fit(Xs[tr])
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(Xs[tr]), ys[tr])
        preds[te] = clf.predict_proba(sc.transform(Xs[te]))[:, 1]
    mask = ~np.isnan(preds)
    auroc = float(roc_auc_score(ys[mask], preds[mask]))
    print(f"diagonal AUROC: {auroc:.3f}")

    # Train final probe on all balanced data
    scaler = StandardScaler().fit(Xs)
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(scaler.transform(Xs), ys)
    # Translate weight to input space: w_input = w_lr / scaler.scale_
    w_input = clf.coef_[0] / scaler.scale_
    w_unit = w_input / np.linalg.norm(w_input)

    # Hidden-state norm scale (use raw X norms)
    h_norm_mean = float(np.mean([np.linalg.norm(x) for x in X]))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{CKPT}_l{LAYER}_{KIND}_direction.npz")
    np.savez(out_path,
             v_unit=w_unit.astype(np.float32),
             v_input_raw=w_input.astype(np.float32),
             h_mean_norm=np.float32(h_norm_mean),
             probe_auroc=np.float32(auroc))
    print(f"wrote {out_path}")
    print(f"  v_unit shape: {w_unit.shape}, ||v_input||: {np.linalg.norm(w_input):.2f}")
    print(f"  mean hidden state L2 norm: {h_norm_mean:.2f}")


if __name__ == "__main__":
    main()
