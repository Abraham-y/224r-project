"""Robustness checks for analyze_probes.py.

Two checks per (checkpoint, layer, position) cell:

  1. Bootstrap CI: resample groups (prompts) with replacement B times, refit
     the probe and compute AUROC on a held-out problem fold each time.
     Report mean, 2.5%, 97.5% percentiles.

  2. Class-balance subsample: downsample to a 50/50 correct/wrong split
     (within problems, preserving the GroupKFold structure), refit, report
     AUROC. This separates "the probe found a real signal" from "the
     classes are so imbalanced that any direction beats chance."

Pure CPU. Runs in ~1-2 minutes on a laptop for 18 cells.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# Reuse the filename parser from analyze_probes.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from extension.probe.analyze_probes import parse_filename, load_groups


def fit_one_fold(X_train, y_train, X_test, y_test, C=0.1, seed=0):
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return float("nan")
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed, solver="lbfgs")
    clf.fit(scaler.transform(X_train), y_train)
    proba = clf.predict_proba(scaler.transform(X_test))[:, 1]
    return float(roc_auc_score(y_test, proba))


def cv_auroc(X, y, groups, n_splits=5, C=0.1, seed=0):
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        return float("nan")
    gkf = GroupKFold(n_splits=n_splits)
    aucs = []
    for tr, te in gkf.split(X, y, groups=groups):
        a = fit_one_fold(X[tr], y[tr], X[te], y[te], C=C, seed=seed)
        if not np.isnan(a):
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else float("nan")


def bootstrap_groups_auroc(X, y, groups, B=200, n_splits=5, C=0.1, seed=0):
    """Resample unique groups with replacement; refit + compute CV AUROC each draw."""
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return float("nan"), float("nan"), float("nan")
    boot_aucs = []
    for b in range(B):
        chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        # build a resampled index by concatenating all rows from each chosen group
        # (groups can repeat; rows from each repeat are included as separate copies)
        masks_by_group = {g: np.where(groups == g)[0] for g in unique_groups}
        idx_parts = [masks_by_group[g] for g in chosen]
        idx = np.concatenate(idx_parts)
        # Use group ids as repeated tags so GroupKFold sees the resampled structure
        new_groups = np.repeat(np.arange(len(chosen)), [len(p) for p in idx_parts])
        a = cv_auroc(X[idx], y[idx], new_groups, n_splits=n_splits, C=C, seed=seed + b)
        if not np.isnan(a):
            boot_aucs.append(a)
    if not boot_aucs:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(boot_aucs)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def balanced_subsample_auroc(X, y, groups, seed=0, n_splits=5):
    """Downsample so positive and negative class counts are equal, then CV."""
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n = min(len(pos), len(neg))
    sel = np.concatenate([rng.choice(pos, n, replace=False),
                          rng.choice(neg, n, replace=False)])
    return cv_auroc(X[sel], y[sel], groups[sel], n_splits=n_splits, seed=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--bootstrap_B", type=int, default=200)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache_dir, "*.npz")))
    rows = []
    for npz_path in files:
        name = os.path.basename(npz_path)
        parsed = parse_filename(name)
        if not parsed:
            continue
        ckpt, layer, kind = parsed
        meta_path = npz_path.replace(".npz", ".meta.json")
        if not os.path.exists(meta_path):
            continue
        with np.load(npz_path) as data:
            X = data["X"]; y = data["y"]
        groups = load_groups(meta_path)
        n_pos = int(y.sum()); n_neg = int(len(y) - n_pos)

        boot_mean, boot_lo, boot_hi = bootstrap_groups_auroc(X, y, groups, B=args.bootstrap_B)
        bal_auc = balanced_subsample_auroc(X, y, groups)
        rows.append({
            "ckpt": ckpt, "layer": layer, "kind": kind,
            "n_pos": n_pos, "n_neg": n_neg,
            "boot_mean": boot_mean, "boot_lo": boot_lo, "boot_hi": boot_hi,
            "balanced_auc": bal_auc,
        })
        print(f"  done {ckpt} L{layer} {kind}", flush=True)

    print()
    print("=" * 110)
    print("Bootstrap CI (B = {} group-resamples) + balanced-subsample AUROC".format(args.bootstrap_B))
    print("=" * 110)
    header = (
        f"{'ckpt':<12}{'layer':>6}{'kind':>14}"
        f"{'pos/neg':>14}"
        f"{'boot mean':>12}{'95% CI':>20}{'balanced':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        ci = f"[{r['boot_lo']:.3f}, {r['boot_hi']:.3f}]"
        print(
            f"{r['ckpt']:<12}{r['layer']:>6}{r['kind']:>14}"
            f"{r['n_pos']:>6}/{r['n_neg']:<6}"
            f"{r['boot_mean']:>12.3f}{ci:>20}{r['balanced_auc']:>12.3f}"
        )

    # Robustness verdict on the headline H3 cell.
    print()
    print("=" * 110)
    print("Headline H3 robustness verdict")
    print("=" * 110)
    out_l16_ass = next((r for r in rows if r["ckpt"] == "C_outcome" and r["layer"] == 16 and r["kind"] == "assertion"), None)
    out_l16_neu = next((r for r in rows if r["ckpt"] == "C_outcome" and r["layer"] == 16 and r["kind"] == "neutral"), None)
    sft_l16_ass = next((r for r in rows if r["ckpt"] == "C_SFT" and r["layer"] == 16 and r["kind"] == "assertion"), None)
    if out_l16_ass and out_l16_neu:
        print(f"\nC_outcome L16 assertion balanced AUROC: {out_l16_ass['balanced_auc']:.3f}")
        print(f"C_outcome L16 neutral   balanced AUROC: {out_l16_neu['balanced_auc']:.3f}")
        delta = out_l16_ass['balanced_auc'] - out_l16_neu['balanced_auc']
        print(f"  delta (assertion - neutral) on balanced samples: {delta:+.3f}")
        if abs(out_l16_ass['balanced_auc'] - out_l16_ass['boot_mean']) > 0.05:
            print(f"  ⚠️  balanced AUROC differs from raw bootstrap mean by >0.05 — class imbalance was a real factor.")
        else:
            print(f"  ✓ balanced AUROC within 0.05 of raw — class imbalance is NOT explaining the signal.")
    if sft_l16_ass:
        print(f"\nC_SFT L16 assertion balanced AUROC:    {sft_l16_ass['balanced_auc']:.3f}")
        print(f"C_outcome L16 assertion balanced AUROC: {out_l16_ass['balanced_auc']:.3f}")
        delta = sft_l16_ass['balanced_auc'] - out_l16_ass['balanced_auc']
        print(f"  delta (SFT - outcome) at assertion positions, balanced: {delta:+.3f}")
        print(f"  ↑ positive = RL specifically suppressed the assertion-position signal (H3 confirmed)")


if __name__ == "__main__":
    main()
