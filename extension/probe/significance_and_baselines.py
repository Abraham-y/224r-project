"""Significance tests on the matched-pair / Cohen's d results, plus
MLP / random-forest probe baselines for methodology validation.

  [A] Wilcoxon signed-rank test per checkpoint on matched-pair deltas:
      H0: median(correct_mean - wrong_mean) = 0 across prompts.

  [B] Mann-Whitney U between C_SFT and C_outcome on per-problem Cohen's d:
      H0: the two checkpoints' d-distributions are equal.

  [C] Paired t-test as a parametric alternative to (A).

  [D] Linear / RF / MLP probe baselines per (checkpoint, position) at L16,
      with held-out-problem CV. If RF or MLP substantially beats LR on the
      same data, the 'signal is gone' framing changes to 'signal is gone
      *linearly*'.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from extension.probe.analyze_probes import load_groups
from extension.probe.qualitative_examples import cv_per_row_probe_scores


# ---------------------------------------------------------------------------
# Matched-pair deltas helper (reused).
# ---------------------------------------------------------------------------


def matched_pair_deltas(cache_dir: str, ckpt: str, layer: int) -> list[float]:
    npz = os.path.join(cache_dir, f"{ckpt}_l{layer}_assertion.npz")
    meta = npz.replace(".npz", ".meta.json")
    with np.load(npz) as data:
        X = data["X"]; y = data["y"]
    with open(meta) as f:
        meta_rows = json.load(f)
    groups = np.array([row["prompt_idx"] for row in meta_rows], dtype=np.int64)
    scores = cv_per_row_probe_scores(X, y, groups)

    by_rollout: dict[tuple[int, int], dict] = defaultdict(lambda: {"scores": [], "label": None})
    for i, m in enumerate(meta_rows):
        if np.isnan(scores[i]):
            continue
        key = (int(m["prompt_idx"]), int(m["resp_idx"]))
        by_rollout[key]["scores"].append(float(scores[i]))
        by_rollout[key]["label"] = int(y[i])

    by_prompt = defaultdict(lambda: {"correct": [], "wrong": []})
    for (p_idx, _r_idx), v in by_rollout.items():
        bucket = "correct" if v["label"] == 1 else "wrong"
        by_prompt[p_idx][bucket].append(float(np.mean(v["scores"])))
    deltas = []
    for buckets in by_prompt.values():
        if buckets["correct"] and buckets["wrong"]:
            deltas.append(float(np.mean(buckets["correct"])) - float(np.mean(buckets["wrong"])))
    return deltas


def cohens_d_per_problem(cache_dir: str, ckpt: str, layer: int) -> list[float]:
    npz = os.path.join(cache_dir, f"{ckpt}_l{layer}_pre_answer.npz")
    meta = npz.replace(".npz", ".meta.json")
    with np.load(npz) as data:
        X = data["X"]; y = data["y"]
    groups = load_groups(meta)
    out = []
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for tr, te in gkf.split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs")
        clf.fit(scaler.transform(X[tr]), y[tr])
        s = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        for g in np.unique(groups[te]):
            m = groups[te] == g
            yg = y[te][m]; sg = s[m]
            if len(np.unique(yg)) < 2:
                continue
            sc, sw = sg[yg == 1], sg[yg == 0]
            var_c = sc.var(ddof=1) if len(sc) > 1 else 0.0
            var_w = sw.var(ddof=1) if len(sw) > 1 else 0.0
            pooled = float(np.sqrt(((len(sc) - 1) * var_c + (len(sw) - 1) * var_w)
                                    / max(len(sc) + len(sw) - 2, 1)))
            if pooled < 1e-9:
                continue
            out.append((sc.mean() - sw.mean()) / pooled)
    return out


# ---------------------------------------------------------------------------
# Significance tests.
# ---------------------------------------------------------------------------


def section_significance(cache_dir: str, layer: int) -> None:
    print("=" * 86)
    print(f"[A] Wilcoxon signed-rank test on matched-pair deltas (L{layer})")
    print(f"    H0: median delta = 0 (probe ranks correct and wrong equally)")
    print("=" * 86)
    deltas_by_ckpt = {}
    for ckpt in ("C_SFT", "C_outcome"):
        d = matched_pair_deltas(cache_dir, ckpt, layer)
        deltas_by_ckpt[ckpt] = d
        if len(d) < 2:
            print(f"  {ckpt}: n={len(d)}, too few pairs for Wilcoxon")
            continue
        stat, p = stats.wilcoxon(d, alternative="greater")
        median = float(np.median(d))
        print(f"  {ckpt:<14} n={len(d):>3}  median delta = {median:+.3f}   "
              f"W = {stat:.1f}   p (one-sided, > 0) = {p:.4g}")
    print()

    print("=" * 86)
    print("[B] Mann-Whitney U: are C_SFT and C_outcome matched-pair deltas drawn from the same distribution?")
    print("=" * 86)
    if (len(deltas_by_ckpt.get("C_SFT", [])) >= 2
            and len(deltas_by_ckpt.get("C_outcome", [])) >= 2):
        stat, p = stats.mannwhitneyu(
            deltas_by_ckpt["C_SFT"], deltas_by_ckpt["C_outcome"], alternative="greater"
        )
        print(f"  U = {stat:.1f}   p (one-sided, C_SFT deltas > C_outcome deltas) = {p:.4g}")
    print()

    print("=" * 86)
    print("[C] Paired t-test on per-problem matched-pair deltas (parametric alternative to [A])")
    print("=" * 86)
    for ckpt, d in deltas_by_ckpt.items():
        if len(d) < 2:
            continue
        t, p = stats.ttest_1samp(d, popmean=0.0)
        # one-sided > 0:
        p_one_sided = p / 2 if t > 0 else 1 - p / 2
        print(f"  {ckpt:<14} n={len(d):>3}  mean delta = {float(np.mean(d)):+.3f}  "
              f"t = {t:.3f}   p (one-sided, > 0) = {p_one_sided:.4g}")
    print()

    print("=" * 86)
    print(f"[D] Mann-Whitney U on within-problem Cohen's d distributions (L{layer})")
    print("=" * 86)
    sft_d = cohens_d_per_problem(cache_dir, "C_SFT", layer)
    out_d = cohens_d_per_problem(cache_dir, "C_outcome", layer)
    if len(sft_d) >= 2 and len(out_d) >= 2:
        stat, p = stats.mannwhitneyu(sft_d, out_d, alternative="greater")
        print(f"  C_SFT n={len(sft_d):>3}, mean d={float(np.mean(sft_d)):+.3f}")
        print(f"  C_outcome n={len(out_d):>3}, mean d={float(np.mean(out_d)):+.3f}")
        print(f"  U = {stat:.1f}   p (one-sided, C_SFT d > C_outcome d) = {p:.4g}")
    print()


# ---------------------------------------------------------------------------
# Probe-family baselines (LR / RF / MLP).
# ---------------------------------------------------------------------------


def balanced_subsample_with_cv(X, y, groups, clf_factory, n_splits=5, seed=0) -> float:
    """Train `clf_factory()` per CV fold on a balanced subsample (preserving
    the original GroupKFold structure on the subsample), return mean AUROC."""
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n = min(len(pos), len(neg))
    if n == 0:
        return float("nan")
    sel = np.concatenate([
        rng.choice(pos, n, replace=False),
        rng.choice(neg, n, replace=False),
    ])
    Xs, ys, gs = X[sel], y[sel], groups[sel]
    if len(np.unique(gs)) < 2:
        return float("nan")
    n_splits = min(n_splits, len(np.unique(gs)))
    aucs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(Xs, ys, groups=gs):
        if len(np.unique(ys[tr])) < 2 or len(np.unique(ys[te])) < 2:
            continue
        scaler = StandardScaler().fit(Xs[tr])
        clf = clf_factory()
        clf.fit(scaler.transform(Xs[tr]), ys[tr])
        proba = clf.predict_proba(scaler.transform(Xs[te]))[:, 1]
        aucs.append(roc_auc_score(ys[te], proba))
    return float(np.mean(aucs)) if aucs else float("nan")


def section_probe_baselines(cache_dir: str, layer: int) -> None:
    print("=" * 86)
    print(f"[E] Probe-family baselines at L{layer} (balanced 5-fold CV)")
    print(f"    LR  = LogisticRegression(C=0.1)        - what we report in the paper")
    print(f"    RF  = RandomForestClassifier(200 est)  - non-linear baseline")
    print(f"    MLP = MLPClassifier((128,))            - small non-linear baseline")
    print("    If RF/MLP > LR by >> 0.05, our 'signal is gone' claim is")
    print("    'signal is gone *linearly*' instead.")
    print("=" * 86)
    header = f"{'checkpoint':<14}{'kind':>14}{'LR':>10}{'RF':>10}{'MLP':>10}"
    print(header)
    print("-" * len(header))
    factories = {
        "LR":  lambda: LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"),
        "RF":  lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0),
        "MLP": lambda: MLPClassifier(hidden_layer_sizes=(128,), max_iter=200,
                                      early_stopping=True, random_state=0),
    }
    for ckpt in ("C_SFT", "C_outcome"):
        for kind in ("pre_answer", "assertion", "neutral"):
            npz = os.path.join(cache_dir, f"{ckpt}_l{layer}_{kind}.npz")
            meta = npz.replace(".npz", ".meta.json")
            if not (os.path.exists(npz) and os.path.exists(meta)):
                continue
            with np.load(npz) as data:
                X = data["X"]; y = data["y"]
            groups = load_groups(meta)
            row = f"{ckpt:<14}{kind:>14}"
            for name in ("LR", "RF", "MLP"):
                auc = balanced_subsample_with_cv(X, y, groups, factories[name])
                row += f"{auc:>10.3f}"
            print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="extension/cache/probe_cache")
    parser.add_argument("--layer", type=int, default=16)
    args = parser.parse_args()
    section_significance(args.cache_dir, args.layer)
    section_probe_baselines(args.cache_dir, args.layer)


if __name__ == "__main__":
    main()
