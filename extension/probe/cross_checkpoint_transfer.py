"""Cross-checkpoint probe transfer (2x2 matrix per position kind).

Train a probe on C_X's activations -> evaluate on C_Y's activations,
for X, Y in {C_SFT, C_outcome}. Compare the matrix to the diagonal:

  - Preserved off-diagonal AUROC -> "signal suppression without drift":
    both representations live in the same subspace, RL just shifted
    where they fall along it.
  - Degraded off-diagonal AUROC -> "representation drift":
    the post-RL hidden state lives in a different subspace, the SFT-trained
    probe can't find it (or vice versa).

We report this at both pre_answer and assertion positions.
Outputs:
  - text table on stdout
  - extension/outputs/figures/fig6_transfer_heatmap.png (2 panels: pre_answer, assertion)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from extension.probe.analyze_probes import load_groups


def load_cell(cache_dir: str, ckpt: str, layer: int, kind: str):
    npz = os.path.join(cache_dir, f"{ckpt}_l{layer}_{kind}.npz")
    meta = npz.replace(".npz", ".meta.json")
    if not (os.path.exists(npz) and os.path.exists(meta)):
        return None
    with np.load(npz) as data:
        X = data["X"]; y = data["y"]
    groups = load_groups(meta)
    return X, y, groups


def balanced_subset(X, y, seed=0):
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n = min(len(pos), len(neg))
    if n == 0:
        return None
    idx = np.concatenate([
        rng.choice(pos, n, replace=False),
        rng.choice(neg, n, replace=False),
    ])
    return X[idx], y[idx]


def train_probe_full(X, y, C=0.1, seed=0):
    """Train a probe on all of (X, y)."""
    sub = balanced_subset(X, y, seed=seed)
    if sub is None:
        return None, None
    Xs, ys = sub
    scaler = StandardScaler().fit(Xs)
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=seed)
    clf.fit(scaler.transform(Xs), ys)
    return scaler, clf


def auroc_on(scaler, clf, X, y, seed=0):
    """Evaluate (scaler, clf) on a balanced subsample of (X, y)."""
    sub = balanced_subset(X, y, seed=seed)
    if sub is None:
        return float("nan")
    Xs, ys = sub
    if len(np.unique(ys)) < 2:
        return float("nan")
    proba = clf.predict_proba(scaler.transform(Xs))[:, 1]
    return float(roc_auc_score(ys, proba))


def transfer_matrix(cache_dir: str, layer: int, kind: str):
    cells = {ckpt: load_cell(cache_dir, ckpt, layer, kind)
             for ckpt in ("C_SFT", "C_outcome")}
    if any(v is None for v in cells.values()):
        return None
    names = ["C_SFT", "C_outcome"]

    # Train one probe per checkpoint on its FULL balanced data; use this
    # probe only for the OFF-diagonal transfer evaluations (no leakage,
    # since eval data is from the other checkpoint entirely).
    probes_full: dict[str, tuple] = {}
    for ckpt, (X, y, _g) in cells.items():
        scaler, clf = train_probe_full(X, y)
        if scaler is None:
            return None
        probes_full[ckpt] = (scaler, clf)

    # For DIAGONAL entries, report held-out CV AUROC instead of training-set
    # AUROC so the diagonal isn't an overfit ~1.0.
    from extension.probe.robustness_probes import balanced_subsample_auroc as _bal_cv

    mat = np.full((2, 2), float("nan"))
    for i, train_ckpt in enumerate(names):
        for j, eval_ckpt in enumerate(names):
            if i == j:  # diagonal -> held-out CV (leakage-free)
                X, y, g = cells[eval_ckpt]
                mat[i, j] = _bal_cv(X, y, g)
            else:       # off-diagonal -> trained-elsewhere probe -> eval here
                scaler, clf = probes_full[train_ckpt]
                X, y, _g = cells[eval_ckpt]
                mat[i, j] = auroc_on(scaler, clf, X, y)
    return mat, names


def print_matrix(mat, names, title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"  {'train -> eval':<22}{'C_SFT':>14}{'C_outcome':>14}")
    print("  " + "-" * 50)
    for i, train_ckpt in enumerate(names):
        row = f"  {train_ckpt:<22}"
        for j in range(len(names)):
            row += f"{mat[i, j]:>14.3f}"
        print(row)


def make_transfer_heatmap(matrices: dict, names, outpath: str, layer: int) -> None:
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n + 0.6, 4.6), dpi=160)
    if n == 1:
        axes = [axes]
    for ax, (kind, mat) in zip(axes, matrices.items()):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=0.3, vmax=0.9, aspect="equal")
        for i in range(len(names)):
            for j in range(len(names)):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=("white" if abs(v - 0.6) > 0.18 else "black"),
                        fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        ax.set_xlabel("evaluated on")
        ax.set_ylabel("trained on")
        ax.set_title(
            {
                "pre_answer": f"</think> position",
                "assertion": "confidence assertion",
                "neutral": "neutral position",
            }.get(kind, kind),
            fontsize=11,
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Cross-checkpoint probe transfer (balanced AUROC, layer L{layer})\n"
        f"diagonal = in-distribution; off-diagonal = transfer",
        fontsize=11.5, y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"\n[transfer] wrote {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="extension/cache/probe_cache")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--out", default="extension/outputs/figures/fig6_transfer_heatmap.png")
    args = parser.parse_args()

    matrices = {}
    for kind in ("pre_answer", "assertion"):
        result = transfer_matrix(args.cache_dir, args.layer, kind)
        if result is None:
            print(f"[transfer] missing data for {kind}, skipping")
            continue
        mat, names = result
        matrices[kind] = mat
        print_matrix(mat, names, f"Cross-checkpoint transfer @ {kind} (L{args.layer})")

    if matrices:
        make_transfer_heatmap(matrices, ["C_SFT", "C_outcome"], args.out, args.layer)


if __name__ == "__main__":
    main()
