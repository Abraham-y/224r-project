"""Three follow-up analyses on the cached probe data:

  1. Per-keyword AUROC at assertion positions, per checkpoint.
     Tests whether some keywords ("Perfect", "got it") drive the concealment
     signal more than others ("the answer is", "verified").

  2. Within-problem Cohen's d of probe scores between correct and wrong
     rollouts of the same prompt. Yuan et al.'s benchmark; isolates the
     "the model knows" effect from across-problem easiness.

  3. Per-layer trajectory table from the dynamics CSV.
"""

from __future__ import annotations

import argparse
import csv
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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from extension.probe.analyze_probes import parse_filename, load_groups
from extension.probe.robustness_probes import cv_auroc, balanced_subsample_auroc


# ---------------------------------------------------------------------------
# 1) Per-keyword AUROC at assertion positions.
# ---------------------------------------------------------------------------


def per_keyword_auroc(cache_dir: str, layer: int) -> None:
    print()
    print("=" * 90)
    print(f"[1] Per-keyword probe AUROC at assertion positions (layer L{layer})")
    print("=" * 90)
    header = (
        f"{'checkpoint':<14}{'keyword':<22}{'n':>6}"
        f"{'pos%':>7}{'probe':>10}{'balanced':>12}"
    )
    print(header)
    print("-" * len(header))
    for ckpt in ("C_SFT", "C_outcome"):
        npz_path = os.path.join(cache_dir, f"{ckpt}_l{layer}_assertion.npz")
        meta_path = npz_path.replace(".npz", ".meta.json")
        if not (os.path.exists(npz_path) and os.path.exists(meta_path)):
            print(f"  missing files for {ckpt}, skipping")
            continue
        with np.load(npz_path) as data:
            X = data["X"]; y = data["y"]
        with open(meta_path) as f:
            meta = json.load(f)
        groups = np.array([row["prompt_idx"] for row in meta], dtype=np.int64)
        keywords = np.array([row.get("keyword", "?") for row in meta])

        for kw in sorted(np.unique(keywords)):
            mask = keywords == kw
            if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
                continue
            Xk, yk, gk = X[mask], y[mask], groups[mask]
            probe = cv_auroc(Xk, yk, gk, n_splits=min(5, len(np.unique(gk))))
            bal = balanced_subsample_auroc(Xk, yk, gk)
            n_pos = int(yk.sum())
            pos_pct = 100.0 * n_pos / len(yk)
            print(
                f"{ckpt:<14}{kw:<22}{len(yk):>6}{pos_pct:>6.0f}%"
                f"{probe:>10.3f}{bal:>12.3f}"
            )


# ---------------------------------------------------------------------------
# 2) Within-problem Cohen's d at pre_answer.
# ---------------------------------------------------------------------------


def within_problem_cohens_d(cache_dir: str, layer: int) -> None:
    print()
    print("=" * 90)
    print(f"[2] Within-problem Cohen's d of probe scores at </think> (layer L{layer})")
    print("=" * 90)
    print(
        "  For each held-out problem with >=1 correct AND >=1 wrong rollout,\n"
        "  Cohen's d = (mean probe[correct] - mean probe[wrong]) / pooled_sd.\n"
        "  Positive d = probe ranks correct rollouts higher than wrong ones.\n"
    )

    for ckpt in ("C_SFT", "C_outcome"):
        npz_path = os.path.join(cache_dir, f"{ckpt}_l{layer}_pre_answer.npz")
        meta_path = npz_path.replace(".npz", ".meta.json")
        if not (os.path.exists(npz_path) and os.path.exists(meta_path)):
            print(f"  missing files for {ckpt}, skipping")
            continue
        with np.load(npz_path) as data:
            X = data["X"]; y = data["y"]
        groups = load_groups(meta_path)
        unique_groups = np.unique(groups)

        # Cross-validated within-problem d: for each held-out problem, score
        # its rollouts with a probe trained on the other problems.
        per_problem_d: list[float] = []
        gkf = GroupKFold(n_splits=min(5, len(unique_groups)))
        for tr_idx, te_idx in gkf.split(X, y, groups=groups):
            if len(np.unique(y[tr_idx])) < 2:
                continue
            scaler = StandardScaler().fit(X[tr_idx])
            clf = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs")
            clf.fit(scaler.transform(X[tr_idx]), y[tr_idx])
            scores = clf.predict_proba(scaler.transform(X[te_idx]))[:, 1]
            for g in np.unique(groups[te_idx]):
                gmask = groups[te_idx] == g
                yg = y[te_idx][gmask]
                sg = scores[gmask]
                if len(np.unique(yg)) < 2:
                    continue
                s_correct = sg[yg == 1]
                s_wrong = sg[yg == 0]
                if len(s_correct) < 1 or len(s_wrong) < 1:
                    continue
                # Pooled SD (Welch-style with at-least-one-sample protection).
                var_c = s_correct.var(ddof=1) if len(s_correct) > 1 else 0.0
                var_w = s_wrong.var(ddof=1) if len(s_wrong) > 1 else 0.0
                pooled = float(
                    np.sqrt(((len(s_correct) - 1) * var_c + (len(s_wrong) - 1) * var_w)
                            / max(len(s_correct) + len(s_wrong) - 2, 1))
                )
                if pooled < 1e-9:
                    continue
                d = float(s_correct.mean() - s_wrong.mean()) / pooled
                per_problem_d.append(d)

        if not per_problem_d:
            print(f"  {ckpt}: no problems had both correct and wrong rollouts; can't compute d")
            continue
        arr = np.array(per_problem_d)
        print(
            f"  {ckpt:<14} "
            f"n_problems={len(arr):>3}   "
            f"mean d = {arr.mean():+.3f}   "
            f"median = {float(np.median(arr)):+.3f}   "
            f"IQR = [{float(np.percentile(arr, 25)):+.3f}, {float(np.percentile(arr, 75)):+.3f}]"
        )


# ---------------------------------------------------------------------------
# 3) Per-layer dynamics trajectory.
# ---------------------------------------------------------------------------


def per_layer_dynamics(dynamics_csv: str) -> None:
    if not os.path.exists(dynamics_csv):
        print(f"\n  dynamics CSV not found at {dynamics_csv}, skipping")
        return
    print()
    print("=" * 90)
    print("[3] Per-layer dynamics trajectory (balanced AUROC)")
    print("=" * 90)
    rows = []
    with open(dynamics_csv) as f:
        for row in csv.DictReader(f):
            rows.append({
                "step": int(row["step"]),
                "layer": int(row["layer"]),
                "kind": row["kind"],
                "balanced": float(row["balanced_auc"]),
            })
    for layer in sorted({r["layer"] for r in rows}):
        print(f"\n  --- L{layer} ---")
        print(f"  {'step':>6}  {'pre_answer':>12}  {'assertion':>12}  {'neutral':>12}")
        by_step: dict[int, dict[str, float]] = {}
        for r in rows:
            if r["layer"] != layer:
                continue
            by_step.setdefault(r["step"], {})[r["kind"]] = r["balanced"]
        for step in sorted(by_step):
            d = by_step[step]
            print(
                f"  {step:>6}  "
                f"{d.get('pre_answer', float('nan')):>12.3f}  "
                f"{d.get('assertion',  float('nan')):>12.3f}  "
                f"{d.get('neutral',    float('nan')):>12.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True,
                        help="Cross-checkpoint cache dir (C_SFT/C_outcome .npz)")
    parser.add_argument("--dynamics_csv", default="extension/outputs/dynamics_auroc.csv")
    parser.add_argument("--layer", type=int, default=16)
    args = parser.parse_args()

    per_keyword_auroc(args.cache_dir, args.layer)
    within_problem_cohens_d(args.cache_dir, args.layer)
    per_layer_dynamics(args.dynamics_csv)


if __name__ == "__main__":
    main()
