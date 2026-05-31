"""Bootstrap 95% confidence intervals on the headline AUROC numbers.

For each headline cell (checkpoint, layer, position), do B=200 bootstrap
replicates over PROMPT INDICES (not individual rows). Re-train the probe
on the bootstrap sample with GroupKFold(5) and report the percentile CI.

This is the proper procedure given that rows within a prompt are correlated
(same problem, different rollouts).
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")


def load_cell(cache_dir: str, ckpt: str, layer: int, kind: str):
    npz = os.path.join(cache_dir, f"{ckpt}_l{layer}_{kind}.npz")
    meta_path = npz.replace(".npz", ".meta.json")
    if not (os.path.exists(npz) and os.path.exists(meta_path)):
        return None
    with np.load(npz) as d:
        X = d["X"]; y = d["y"]
    meta = json.load(open(meta_path))
    groups = np.array([m["prompt_idx"] for m in meta])
    return X, y, groups


def balanced_auroc_cv(X, y, groups, seed=0) -> float:
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n = min(len(pos), len(neg))
    if n < 5:
        return float("nan")
    idx = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
    X, y, groups = X[idx], y[idx], groups[idx]
    preds = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        preds[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    mask = ~np.isnan(preds)
    if len(np.unique(y[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(y[mask], preds[mask]))


def bootstrap_ci(X, y, groups, B: int = 200, seed: int = 0, frac: float = 0.8):
    """Subsample bootstrap WITHOUT replacement over unique prompt IDs.

    Each replicate draws a `frac` fraction of prompts (without replacement),
    keeps all their rows, then runs balanced GroupKFold CV on that subset.
    This avoids the leakage that cluster-with-replacement causes when the
    same prompt lands in both train and test folds of the same replicate.
    """
    rng = np.random.RandomState(seed)
    unique_prompts = np.unique(groups)
    k = max(5, int(round(frac * len(unique_prompts))))
    aurocs = []
    for b in range(B):
        sampled = rng.choice(unique_prompts, size=k, replace=False)
        mask = np.isin(groups, sampled)
        Xb = X[mask]; yb = y[mask]; gb = groups[mask]
        auc = balanced_auroc_cv(Xb, yb, gb, seed=b)
        aurocs.append(auc)
    aurocs = np.array(aurocs)
    aurocs = aurocs[~np.isnan(aurocs)]
    if len(aurocs) == 0:
        return float("nan"), float("nan"), float("nan")
    median = float(np.median(aurocs))
    lo = float(np.percentile(aurocs, 2.5))
    hi = float(np.percentile(aurocs, 97.5))
    return median, lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="extension/cache/probe_cache_n500_clean406")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--B", type=int, default=200)
    parser.add_argument("--out", default="extension/outputs/n500/text/15_bootstrap_cis.txt")
    args = parser.parse_args()

    cells = [
        ("C_SFT",     "pre_answer"),
        ("C_SFT",     "assertion"),
        ("C_SFT",     "neutral"),
        ("C_outcome", "pre_answer"),
        ("C_outcome", "assertion"),
        ("C_outcome", "neutral"),
    ]

    out_lines = [
        f"Bootstrap 95% CI on probe AUROC (L{args.layer}, B={args.B}, cluster on prompt_idx)",
        f"Cache: {args.cache_dir}",
        "",
        f"{'checkpoint':<12} {'kind':<14} {'point':>8}  {'median':>8}  {'95% CI':>20}",
        "-" * 72,
    ]
    for ckpt, kind in cells:
        c = load_cell(args.cache_dir, ckpt, args.layer, kind)
        if c is None:
            out_lines.append(f"{ckpt:<12} {kind:<14}  (missing)")
            continue
        X, y, groups = c
        point = balanced_auroc_cv(X, y, groups, seed=0)
        median, lo, hi = bootstrap_ci(X, y, groups, B=args.B, seed=42)
        out_lines.append(
            f"{ckpt:<12} {kind:<14} {point:>8.3f}  {median:>8.3f}  [{lo:>6.3f}, {hi:>6.3f}]"
        )

    txt = "\n".join(out_lines)
    print(txt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
