"""Phase 1 diagnostics on the n=500 clean-406 cache.

  1A. Multiple-seed cross-position transfer with EXPLICIT balancing on both
      source training data and target eval data, reporting per-seed AUROCs.
      Mean +/- std across 10 seeds. Also reports symmetric mean of the two
      transfer directions (pre->ass mean(pre->ass, ass->pre)) to test
      whether the asymmetry persists after averaging out seed variance.

  1B. Per-layer (L12, L16, L20) version of the 3x3 cross-position transfer
      matrix, both checkpoints. Output:
        - 3 layers x 2 ckpts x 3x3 = 54 cells in a flat table
        - heatmap PNG with one panel per (layer, ckpt) -> 6 small heatmaps

  1C. Random-pairs control: include `neutral` in the 3x3 grid (already there
      in the current 3x3, but now we explicitly compare pre_answer->assertion
      to pre_answer->neutral, etc.) to test whether the orthogonality is
      assertion-specific or a general pre-vs-other phenomenon.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from itertools import product

import matplotlib.pyplot as plt
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


def balanced_idx(y: np.ndarray, seed: int) -> np.ndarray | None:
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n = min(len(pos), len(neg))
    if n < 5:
        return None
    return np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])


def fit_predict(X_tr, y_tr, X_te) -> np.ndarray:
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X_tr), y_tr)
    return clf.predict_proba(sc.transform(X_te))[:, 1]


def diagonal_groupkfold_auroc(X, y, groups, seed: int) -> float:
    """Balanced GroupKFold(5) AUROC, balanced WITH the seed."""
    idx = balanced_idx(y, seed)
    if idx is None:
        return float("nan")
    Xs, ys, gs = X[idx], y[idx], groups[idx]
    preds = np.full(len(ys), np.nan)
    for tr, te in GroupKFold(5).split(Xs, ys, gs):
        proba = fit_predict(Xs[tr], ys[tr], Xs[te])
        preds[te] = proba
    mask = ~np.isnan(preds)
    if len(np.unique(ys[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(ys[mask], preds[mask]))


def cross_position_auroc(X_tr, y_tr, X_te, y_te, seed: int) -> float:
    """Train on a balanced subset of the source position, eval on a
    balanced subset of the target position. Both subsamples use the same
    seed for reproducibility but DIFFERENT random indices because the
    cells contain different rows."""
    idx_tr = balanced_idx(y_tr, seed)
    idx_te = balanced_idx(y_te, seed + 1000)  # different seed offset for target
    if idx_tr is None or idx_te is None:
        return float("nan")
    Xt, yt = X_tr[idx_tr], y_tr[idx_tr]
    Xe, ye = X_te[idx_te], y_te[idx_te]
    if len(np.unique(ye)) < 2:
        return float("nan")
    proba = fit_predict(Xt, yt, Xe)
    return float(roc_auc_score(ye, proba))


def run_phase1(cache_dir: str, layers: list[int], n_seeds: int, out_path: str, fig_path: str):
    kinds = ("pre_answer", "assertion", "neutral")
    ckpts = ("C_SFT", "C_outcome")

    # Holds AUROCs[layer][ckpt][train_kind][eval_kind] = list of seed AUROCs
    aurocs: dict = {layer: {c: {k1: {k2: [] for k2 in kinds} for k1 in kinds} for c in ckpts}
                    for layer in layers}

    # Pre-load all cells.
    cells: dict = {}
    for layer in layers:
        for ckpt in ckpts:
            for kind in kinds:
                cells[(layer, ckpt, kind)] = load_cell(cache_dir, ckpt, layer, kind)

    # Run per (layer, ckpt, train_kind, eval_kind, seed)
    for layer, ckpt in product(layers, ckpts):
        for train_kind, eval_kind in product(kinds, kinds):
            tr = cells[(layer, ckpt, train_kind)]
            te = cells[(layer, ckpt, eval_kind)]
            if tr is None or te is None:
                continue
            X_tr, y_tr, g_tr = tr
            X_te, y_te, g_te = te
            for seed in range(n_seeds):
                if train_kind == eval_kind:
                    a = diagonal_groupkfold_auroc(X_tr, y_tr, g_tr, seed=seed)
                else:
                    a = cross_position_auroc(X_tr, y_tr, X_te, y_te, seed=seed)
                aurocs[layer][ckpt][train_kind][eval_kind].append(a)

    # Build text output.
    out_lines: list[str] = []
    out_lines.append("PHASE 1 DIAGNOSTICS")
    out_lines.append(f"  cache: {cache_dir}")
    out_lines.append(f"  layers: {layers}")
    out_lines.append(f"  seeds per cell: {n_seeds}  (balanced subsample of both source train and target eval)")
    out_lines.append("=" * 100)

    for layer in layers:
        out_lines.append(f"\n--- Layer L{layer} ---")
        for ckpt in ckpts:
            out_lines.append(f"\n{ckpt}:")
            out_lines.append("  train\\eval ".ljust(20) + "  ".join(f"{k:>20}" for k in kinds))
            for train_kind in kinds:
                row = f"  {train_kind:<18}"
                for eval_kind in kinds:
                    vals = aurocs[layer][ckpt][train_kind][eval_kind]
                    if not vals or all(np.isnan(vals)):
                        row += f"{'n/a':>22}"
                    else:
                        m = float(np.nanmean(vals))
                        s = float(np.nanstd(vals))
                        row += f"  {m:>7.3f} +/- {s:.3f} "
                out_lines.append(row)

    out_lines.append("")
    out_lines.append("=" * 100)
    out_lines.append("ASYMMETRY & SYMMETRIC-MEAN DIAGNOSTIC")
    out_lines.append("=" * 100)
    out_lines.append(f"{'layer':>6}  {'ckpt':<10}  {'pre->ass':>12} {'ass->pre':>12} {'symmetric mean':>16}  {'asymmetry':>12}")
    out_lines.append("-" * 90)
    for layer in layers:
        for ckpt in ckpts:
            a = np.nanmean(aurocs[layer][ckpt]["pre_answer"]["assertion"])
            b = np.nanmean(aurocs[layer][ckpt]["assertion"]["pre_answer"])
            sym = (a + b) / 2
            asym = a - b
            out_lines.append(f"  L{layer:<4}  {ckpt:<10}  {a:>12.3f} {b:>12.3f} {sym:>16.3f}  {asym:>+12.3f}")

    out_lines.append("")
    out_lines.append("=" * 100)
    out_lines.append("NEUTRAL-CONTROL: is pre_answer specifically orthogonal to ASSERTION, or to all non-pre_answer?")
    out_lines.append("=" * 100)
    out_lines.append(f"{'layer':>6}  {'ckpt':<10}  {'pre->ass':>12} {'pre->neu':>12} {'ass->pre':>12} {'ass->neu':>12} {'neu->pre':>12} {'neu->ass':>12}")
    out_lines.append("-" * 100)
    for layer in layers:
        for ckpt in ckpts:
            vals = {(a, b): np.nanmean(aurocs[layer][ckpt][a][b])
                    for a in kinds for b in kinds}
            row = f"  L{layer:<4}  {ckpt:<10}"
            for a, b in (("pre_answer","assertion"), ("pre_answer","neutral"),
                         ("assertion","pre_answer"), ("assertion","neutral"),
                         ("neutral","pre_answer"), ("neutral","assertion")):
                row += f"  {vals[(a,b)]:>10.3f}"
            out_lines.append(row)

    txt = "\n".join(out_lines)
    print(txt)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(txt + "\n")

    # Heatmap: 6 panels (3 layers x 2 ckpts), each a 3x3 grid.
    fig, axes = plt.subplots(len(ckpts), len(layers), figsize=(4 * len(layers), 4 * len(ckpts)), dpi=140)
    if len(ckpts) == 1 or len(layers) == 1:
        axes = np.array(axes).reshape(len(ckpts), len(layers))
    for j, ckpt in enumerate(ckpts):
        for i, layer in enumerate(layers):
            mat = np.zeros((3, 3))
            for a, train_kind in enumerate(kinds):
                for b, eval_kind in enumerate(kinds):
                    mat[a, b] = np.nanmean(aurocs[layer][ckpt][train_kind][eval_kind])
            ax = axes[j, i]
            im = ax.imshow(mat, vmin=0.4, vmax=1.0, cmap="RdBu_r", aspect="equal")
            for a in range(3):
                for b in range(3):
                    color = "white" if mat[a, b] > 0.78 or mat[a, b] < 0.48 else "black"
                    ax.text(b, a, f"{mat[a, b]:.3f}", ha="center", va="center",
                            color=color, fontsize=10, fontweight="bold")
            ax.set_xticks(range(3)); ax.set_xticklabels(kinds, rotation=30, ha="right", fontsize=9)
            ax.set_yticks(range(3)); ax.set_yticklabels(kinds, fontsize=9)
            ax.set_title(f"{ckpt}  L{layer}", fontsize=11)
            if i == 0:
                ax.set_ylabel("trained on", fontsize=10)
            if j == len(ckpts) - 1:
                ax.set_xlabel("evaluated on", fontsize=10)
    fig.suptitle("Phase 1B/1C: per-layer cross-position probe transfer (clean-406, mean of 10 seeds)", fontsize=12, y=1.02)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="AUROC", shrink=0.6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path) or ".", exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote heatmap: {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="extension/cache/probe_cache_n500_clean406")
    parser.add_argument("--layers", type=int, nargs="+", default=[12, 16, 20])
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--out", default="extension/outputs/n500/text/16_phase1_diagnostics.txt")
    parser.add_argument("--fig", default="extension/outputs/n500/figures/fig_phase1_transfer_heatmap.png")
    args = parser.parse_args()
    run_phase1(args.cache_dir, args.layers, args.n_seeds, args.out, args.fig)


if __name__ == "__main__":
    main()
