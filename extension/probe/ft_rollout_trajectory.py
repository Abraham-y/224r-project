"""F->T rollout per-block trajectory: probe rising on the "save" move.

Mirror of the T->F analysis from phase2a_position_appropriate_probe.py but
for the F->T transition class (rollouts where the first <answer> is wrong
and the last <answer> is correct -- the model "saves itself"). If Pattern A
holds in this direction too, we expect the probe to rise across blocks as
the per-block correctness rate rises.

Uses the position-appropriate probe trained on <answer>-opening hidden
states (block_correct labels, held-out GroupKFold(5) by prompt). Same
hyperparameters as Phase 2A.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")


def load_answers(cache_dir: str, ckpt: str, layer: int):
    npz = os.path.join(cache_dir, f"{ckpt}_l{layer}_answers.npz")
    meta_path = npz.replace(".npz", ".meta.json")
    with np.load(npz) as d:
        X = d["X"]
    meta = json.load(open(meta_path))
    return X, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers_dir", default="extension/cache/probe_cache_n500_answers")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--out", default="extension/outputs/n500/text/20_ft_trajectory.txt")
    parser.add_argument("--fig", default="extension/outputs/n500/figures/fig10_ft_rollout_trajectory.png")
    args = parser.parse_args()

    X_all, meta_all = load_answers(args.answers_dir, "C_outcome", args.layer)
    mask = np.array([m["block_correct"] is not None for m in meta_all])
    X = X_all[mask]
    meta = [meta_all[i] for i in range(len(meta_all)) if mask[i]]
    y = np.array([1 if m["block_correct"] else 0 for m in meta], dtype=np.int32)
    groups = np.array([int(m["prompt_idx"]) for m in meta])

    # Build held-out probes
    prompt_to_fold: dict[int, int] = {}
    gkf = GroupKFold(n_splits=5)
    for fi, (_tr, te) in enumerate(gkf.split(X, y, groups)):
        for p in np.unique(groups[te]):
            prompt_to_fold[int(p)] = fi
    fold_probes: dict[int, tuple] = {}
    for f in range(5):
        train_mask = np.array([prompt_to_fold[int(g)] != f for g in groups])
        Xt, yt = X[train_mask], y[train_mask]
        pos = np.where(yt == 1)[0]; neg = np.where(yt == 0)[0]
        n = min(len(pos), len(neg))
        if n < 5:
            continue
        rng = np.random.RandomState(42 + f)
        idx = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
        sc = StandardScaler().fit(Xt[idx])
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(Xt[idx]), yt[idx])
        fold_probes[f] = (sc, clf)

    scores = np.full(len(y), np.nan)
    for i in range(len(y)):
        f = prompt_to_fold[int(groups[i])]
        if f not in fold_probes:
            continue
        sc, clf = fold_probes[f]
        scores[i] = clf.predict_proba(sc.transform(X[i:i+1]))[:, 1][0]

    # Group by rollout
    by_r: dict[tuple[int, int], list] = defaultdict(list)
    for i, m in enumerate(meta):
        if not np.isnan(scores[i]):
            by_r[(int(m["prompt_idx"]), int(m["resp_idx"]))].append(
                (int(m["answer_block_idx"]), float(scores[i]), bool(m["block_correct"]), int(m["n_blocks"]))
            )
    for k in by_r:
        by_r[k].sort()

    # Filter F->T rollouts (first wrong, last correct, n_blocks >= 2)
    ft_rollouts = []
    tf_rollouts = []
    for k, lst in by_r.items():
        if len(lst) < 2:
            continue
        first_c, last_c = lst[0][2], lst[-1][2]
        if not first_c and last_c:
            ft_rollouts.append((k, lst))
        elif first_c and not last_c:
            tf_rollouts.append((k, lst))
    print(f"F->T rollouts: {len(ft_rollouts)} (rescue moves)")
    print(f"T->F rollouts: {len(tf_rollouts)} (drift to wrong)  [reference]")

    # Per-block-index trajectory for each
    def trajectory(rollouts):
        per_block: dict[int, list[float]] = defaultdict(list)
        per_block_corr: dict[int, list[bool]] = defaultdict(list)
        for _k, lst in rollouts:
            for bi, s, c, _n in lst:
                per_block[bi].append(s)
                per_block_corr[bi].append(c)
        return per_block, per_block_corr

    ft_pb, ft_pc = trajectory(ft_rollouts)
    tf_pb, tf_pc = trajectory(tf_rollouts)

    summary_lines = [
        "F->T (rescue) rollouts: per-block-index probe trajectory",
        f"  ckpt=C_outcome, L{args.layer}, position-appropriate probe",
        f"  n_rollouts: F->T = {len(ft_rollouts)}, T->F = {len(tf_rollouts)}",
        "",
        "F->T trajectory (first block wrong, last block correct):",
        f"{'block':>6}  {'n':>5}  {'%corr':>6}  {'mean_probe':>11}",
    ]
    for bi in sorted(ft_pb)[:13]:
        n = len(ft_pb[bi]); pc = float(np.mean(ft_pc[bi])); pp = float(np.mean(ft_pb[bi]))
        summary_lines.append(f"{bi:>6}  {n:>5}  {pc:>6.2f}  {pp:>11.3f}")

    summary_lines.append("")
    summary_lines.append("T->F trajectory (for direct comparison; from §8 of writeup):")
    summary_lines.append(f"{'block':>6}  {'n':>5}  {'%corr':>6}  {'mean_probe':>11}")
    for bi in sorted(tf_pb)[:13]:
        n = len(tf_pb[bi]); pc = float(np.mean(tf_pc[bi])); pp = float(np.mean(tf_pb[bi]))
        summary_lines.append(f"{bi:>6}  {n:>5}  {pc:>6.2f}  {pp:>11.3f}")

    summary_lines.append("")
    summary_lines.append("Pattern A predicts probe tracks per-block %corr in BOTH directions.")
    if ft_pb:
        first_p = float(np.mean(ft_pb[min(ft_pb.keys())]))
        last_p = float(np.mean(ft_pb[max(ft_pb.keys())]))
        last_n = len(ft_pb[max(ft_pb.keys())])
        summary_lines.append(f"  F->T first-block probe mean: {first_p:.3f}")
        summary_lines.append(f"  F->T last-block probe mean (n={last_n}): {last_p:.3f}")
        summary_lines.append(f"  Difference: {last_p - first_p:+.3f}")

    txt = "\n".join(summary_lines)
    print(txt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(txt + "\n")

    # Figure: both trajectories on one plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    # T->F
    bis = sorted(tf_pb)
    tf_means = [np.mean(tf_pb[bi]) for bi in bis]
    tf_ns = [len(tf_pb[bi]) for bi in bis]
    keep = [i for i, n in enumerate(tf_ns) if n >= 20]
    bis_k = [bis[i] for i in keep]
    tf_means_k = [tf_means[i] for i in keep]
    ax.plot(bis_k, tf_means_k, marker="o", color="#c45252", label=f"T→F drift (n={len(tf_rollouts)} rollouts)", linewidth=2)

    # F->T
    bis = sorted(ft_pb)
    ft_means = [np.mean(ft_pb[bi]) for bi in bis]
    ft_ns = [len(ft_pb[bi]) for bi in bis]
    keep = [i for i, n in enumerate(ft_ns) if n >= 5]
    bis_k = [bis[i] for i in keep]
    ft_means_k = [ft_means[i] for i in keep]
    ax.plot(bis_k, ft_means_k, marker="s", color="#dba24c", label=f"F→T rescue (n={len(ft_rollouts)} rollouts)", linewidth=2)

    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.set_xlabel("answer block index within rollout", fontsize=11)
    ax.set_ylabel("mean position-appropriate probe(correct)", fontsize=11)
    ax.set_title(f"Per-block probe trajectory in C_outcome multi-answer rollouts (L{args.layer})\n"
                 f"position-appropriate probe; both drift directions track per-block correctness", fontsize=10)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
    fig.savefig(args.fig)
    plt.close(fig)
    print(f"\nwrote {args.fig}")


if __name__ == "__main__":
    main()
