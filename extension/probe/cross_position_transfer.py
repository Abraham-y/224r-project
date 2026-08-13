"""Cross-position probe transfer matrix.

For each checkpoint, train the probe on hidden states at one position kind
(pre_answer / assertion / neutral) and evaluate it on a different position
kind. If the off-diagonals collapse, that is direct mechanistic evidence
that the trace-final correctness representation and the assertion-position
representation are NOT the same linear subspace — i.e., position-dependent
decoupling at the representational level, not just the readout level.

Output: a 3×3 grid per checkpoint, balanced-class GroupKFold(5) AUROC.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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


def balanced_subsample(X, y, groups, seed=0):
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n = min(len(pos), len(neg))
    if n < 5:
        return None
    idx = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
    return X[idx], y[idx], groups[idx]


def train_and_eval(X_tr, y_tr, X_te, y_te) -> float:
    """Fit on all of train, evaluate on all of eval (already balanced)."""
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X_tr), y_tr)
    if len(np.unique(y_te)) < 2:
        return float("nan")
    proba = clf.predict_proba(sc.transform(X_te))[:, 1]
    return float(roc_auc_score(y_te, proba))


def grouped_transfer_auroc(X_tr, y_tr, g_tr, X_te, y_te, g_te, n_splits: int = 5) -> float:
    """Cross-position AUROC with the SAME prompt-level holdout as the diagonal.

    Positions are NOT independent within a prompt: the pre_answer row and the
    assertion rows of a given rollout come from the same forward pass and carry
    the same rollout-level label. Training on all of kind A and testing on all
    of kind B therefore lets the model see the training-side rows of the very
    rollouts it is scored on -- and puts a leaky estimator in the same table as
    a clean GroupKFold diagonal, so the two are not comparable.

    Here: fold the union of prompts, train on kind-A rows of the train prompts,
    score kind-B rows of the held-out prompts, pool scores, one AUROC.
    """
    preds = np.full(len(y_te), np.nan)
    all_prompts = np.unique(np.concatenate([g_tr, g_te]))
    dummy = np.zeros(len(all_prompts))
    for tr_p, te_p in GroupKFold(n_splits).split(dummy, dummy, all_prompts):
        train_prompts = set(all_prompts[tr_p].tolist())
        test_prompts = set(all_prompts[te_p].tolist())
        tr_mask = np.array([g in train_prompts for g in g_tr])
        te_mask = np.array([g in test_prompts for g in g_te])
        if tr_mask.sum() < 10 or te_mask.sum() == 0:
            continue
        if len(np.unique(y_tr[tr_mask])) < 2:
            continue
        sc = StandardScaler().fit(X_tr[tr_mask])
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(
            sc.transform(X_tr[tr_mask]), y_tr[tr_mask])
        preds[te_mask] = clf.predict_proba(sc.transform(X_te[te_mask]))[:, 1]
    m = ~np.isnan(preds)
    if m.sum() == 0 or len(np.unique(y_te[m])) < 2:
        return float("nan")
    return float(roc_auc_score(y_te[m], preds[m]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="extension/cache/probe_cache_n500_clean406")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--out", default="extension/outputs/n500/text/13_cross_position_transfer.txt")
    args = parser.parse_args()

    kinds = ("pre_answer", "assertion", "neutral")
    out_lines: list[str] = []
    out_lines.append(f"Cross-position probe transfer matrix (L{args.layer})\n")
    out_lines.append(f"Cache: {args.cache_dir}\n")
    out_lines.append("=" * 78 + "\n")

    for ckpt in ("C_SFT", "C_outcome"):
        cells = {}
        for kind in kinds:
            c = load_cell(args.cache_dir, ckpt, args.layer, kind)
            if c is None:
                continue
            cells[kind] = balanced_subsample(*c, seed=0)

        out_lines.append(f"\n{ckpt}:")
        header = "  train  \\  eval ".ljust(20)
        for k in kinds:
            header += f"{k:>12}"
        out_lines.append(header)
        out_lines.append("  " + "-" * (18 + 12 * len(kinds)))
        for train_kind in kinds:
            row = f"  {train_kind:>15}  "
            for eval_kind in kinds:
                if cells.get(train_kind) is None or cells.get(eval_kind) is None:
                    row += f"{'n/a':>12}"
                    continue
                X_tr, y_tr, g_tr = cells[train_kind]
                X_te, y_te, g_te = cells[eval_kind]
                if train_kind == eval_kind:
                    # held-out CV
                    preds = np.full(len(y_tr), np.nan)
                    for tr, te in GroupKFold(5).split(X_tr, y_tr, g_tr):
                        sc = StandardScaler().fit(X_tr[tr])
                        clf = LogisticRegression(C=0.1, max_iter=2000).fit(
                            sc.transform(X_tr[tr]), y_tr[tr])
                        preds[te] = clf.predict_proba(sc.transform(X_tr[te]))[:, 1]
                    mask = ~np.isnan(preds)
                    auc = float(roc_auc_score(y_tr[mask], preds[mask]))
                else:
                    # cross-position, SAME prompt-level holdout as the diagonal
                    # so the two are comparable (see grouped_transfer_auroc).
                    auc = grouped_transfer_auroc(X_tr, y_tr, g_tr, X_te, y_te, g_te)
                row += f"{auc:>12.3f}"
            out_lines.append(row)

    # Diagonal vs off-diagonal summary
    out_lines.append("\n" + "=" * 78)
    out_lines.append(
        "Every cell (diagonal and off-diagonal) uses the same prompt-level\n"
        "GroupKFold(5) holdout, so they are directly comparable.\n"
        "\n"
        "If diagonals > off-diagonals: the probe direction for one position\n"
        "does NOT linearly transfer to another -- evidence that pre_answer\n"
        "and assertion encode distinct correctness subspaces."
    )

    txt = "\n".join(out_lines)
    print(txt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
