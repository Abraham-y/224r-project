"""Task A defensive control: position-appropriate probe for Phase 2A.

Trains a new probe on `<answer>`-opening hidden states (NOT on `</think>`),
labeled by the correctness of the equation that follows in that `<answer>`
block. Same hyperparameters as the trace-final probe (LR C=0.1, balanced
classes, GroupKFold(5) by prompt_idx). Then repeats the within-rollout
trajectory analysis using this position-appropriate probe.

Reports:
  - diagonal AUROC on held-out `<answer>` positions (signal sanity check)
  - 4-transition table (TT/TF/FT/FF) with probe(last) means
  - per-block-index trajectory (mean probe at each block_idx)
  - Pattern A/B/C verdict
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

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
    parser.add_argument("--ckpt", default="C_outcome")
    parser.add_argument("--out", default="extension/outputs/n500/text/19_phase2a_position_appropriate.txt")
    parser.add_argument("--fig", default="extension/outputs/n500/figures/fig9b_within_rollout_position_appropriate.png")
    args = parser.parse_args()

    X_all, meta_all = load_answers(args.answers_dir, args.ckpt, args.layer)
    print(f"[task-a] answers cache: X={X_all.shape}")

    # Filter to rows where block_correct is not None.
    mask = np.array([m["block_correct"] is not None for m in meta_all])
    X = X_all[mask]
    meta = [meta_all[i] for i in range(len(meta_all)) if mask[i]]
    y = np.array([1 if m["block_correct"] else 0 for m in meta], dtype=np.int32)
    groups = np.array([int(m["prompt_idx"]) for m in meta])
    print(f"[task-a] usable rows (block_correct known): {len(y)}  pos%={y.mean():.1%}")

    # Train held-out probe per prompt: GroupKFold(5), balanced subsample of training side.
    prompt_to_fold: dict[int, int] = {}
    gkf = GroupKFold(n_splits=5)
    for fold_idx, (_tr, te) in enumerate(gkf.split(X, y, groups)):
        for p in np.unique(groups[te]):
            prompt_to_fold[int(p)] = fold_idx

    fold_probes: dict[int, tuple] = {}
    for f in range(5):
        train_mask = np.array([prompt_to_fold[int(g)] != f for g in groups])
        Xt, yt = X[train_mask], y[train_mask]
        # balanced subsample
        pos = np.where(yt == 1)[0]; neg = np.where(yt == 0)[0]
        n = min(len(pos), len(neg))
        if n < 5:
            continue
        rng = np.random.RandomState(42 + f)
        idx = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
        sc = StandardScaler().fit(Xt[idx])
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(Xt[idx]), yt[idx])
        fold_probes[f] = (sc, clf)
    print(f"[task-a] trained {len(fold_probes)} folds")

    # Per-row held-out probe scores
    scores = np.full(len(y), np.nan)
    for i in range(len(y)):
        f = prompt_to_fold[int(groups[i])]
        if f not in fold_probes:
            continue
        sc, clf = fold_probes[f]
        scores[i] = clf.predict_proba(sc.transform(X[i:i+1]))[:, 1][0]

    valid = ~np.isnan(scores)
    diag_auc = float(roc_auc_score(y[valid], scores[valid]))
    print(f"[task-a] held-out diagonal AUROC on `<answer>` positions: {diag_auc:.3f}")

    # Aggregate per rollout (using ALL rows, since they all have known block_correct).
    from collections import defaultdict
    by_r: dict[tuple[int, int], list] = defaultdict(list)
    for i, m in enumerate(meta):
        if not np.isnan(scores[i]):
            by_r[(int(m["prompt_idx"]), int(m["resp_idx"]))].append(
                (int(m["answer_block_idx"]), float(scores[i]),
                 bool(m["block_correct"]), int(m["n_blocks"]))
            )
    for k in by_r:
        by_r[k].sort()

    # Transition table
    transition_data: dict[str, list[dict]] = {"TT": [], "TF": [], "FT": [], "FF": []}
    for k, lst in by_r.items():
        if len(lst) < 2:
            continue
        first = lst[0]; last = lst[-1]
        p_first, p_last = first[1], last[1]
        first_corr, last_corr = first[2], last[2]
        tag = ("T" if first_corr else "F") + ("T" if last_corr else "F")
        transition_data[tag].append({
            "prompt_idx": k[0], "resp_idx": k[1],
            "p_first": p_first, "p_last": p_last,
            "first_correct": first_corr, "last_correct": last_corr,
            "n_blocks": first[3],
        })

    summary = [
        "Task A: POSITION-APPROPRIATE PROBE for Phase 2A trajectory analysis",
        f"  ckpt={args.ckpt}, layer=L{args.layer}",
        f"  answers cache: {args.answers_dir}",
        "",
        f"Sanity check: held-out diagonal AUROC on `<answer>` positions = {diag_auc:.3f}",
        f"             (probe trained on `<answer>` positions, with block_correct labels)",
        "",
        "Transition table -- probe(last) means with POSITION-APPROPRIATE probe:",
        "",
        f"{'transition':<6} {'n':>5} {'probe(first)':>15} {'probe(last)':>14} {'last_acc':>10} {'first_acc':>10}",
        "-" * 78,
    ]
    for tag, lst in transition_data.items():
        if not lst:
            continue
        pf = float(np.mean([d["p_first"] for d in lst]))
        pl = float(np.mean([d["p_last"] for d in lst]))
        first_acc = float(np.mean([d["first_correct"] for d in lst]))
        last_acc = float(np.mean([d["last_correct"] for d in lst]))
        summary.append(
            f"{tag:<6} {len(lst):>5}  {pf:>14.3f}  {pl:>13.3f}  {last_acc:>10.2f}  {first_acc:>10.2f}"
        )

    # Per-block-index trajectory for T->F rollouts
    summary.append("")
    summary.append("=" * 78)
    summary.append("Per-block-index trajectory on T->F rollouts (with position-appropriate probe)")
    summary.append("=" * 78)
    summary.append("")
    tf_keys = {(d["prompt_idx"], d["resp_idx"]) for d in transition_data["TF"]}
    tf_per_block: dict[int, list[float]] = defaultdict(list)
    tf_per_block_correct: dict[int, list[bool]] = defaultdict(list)
    for k in tf_keys:
        for bi, s, c, _n in by_r[k]:
            tf_per_block[bi].append(s)
            tf_per_block_correct[bi].append(c)
    summary.append(f"{'block':>6}  {'n':>5}  {'%corr':>6}  {'mean_probe':>11}")
    for bi in sorted(tf_per_block)[:13]:
        nb = len(tf_per_block[bi])
        pc = float(np.mean(tf_per_block_correct[bi]))
        pp = float(np.mean(tf_per_block[bi]))
        summary.append(f"{bi:>6}  {nb:>5}  {pc:>6.2f}  {pp:>11.3f}")

    # Per-block-index, separated by block_correct on FULL data
    summary.append("")
    summary.append("=" * 78)
    summary.append("Per-block-index, ALL rollouts: probe at block_idx separated by block correctness")
    summary.append("=" * 78)
    all_idx_correct: dict[int, list[float]] = defaultdict(list)
    all_idx_wrong: dict[int, list[float]] = defaultdict(list)
    for lst in by_r.values():
        for bi, s, c, _n in lst:
            (all_idx_correct if c else all_idx_wrong)[bi].append(s)
    summary.append(f"{'block_idx':>10}  {'n_corr':>7}  {'n_wrng':>7}  {'p_corr':>8}  {'p_wrng':>8}  {'diff':>8}")
    for bi in sorted(set(list(all_idx_correct.keys()) + list(all_idx_wrong.keys())))[:13]:
        nc = len(all_idx_correct[bi]); nw = len(all_idx_wrong[bi])
        pc = float(np.mean(all_idx_correct[bi])) if nc else float("nan")
        pw = float(np.mean(all_idx_wrong[bi])) if nw else float("nan")
        diff = pc - pw if (nc and nw) else float("nan")
        summary.append(f"{bi:>10}  {nc:>7}  {nw:>7}  {pc:>8.3f}  {pw:>8.3f}  {diff:>+8.3f}")

    summary.append("")
    summary.append("=" * 78)
    summary.append("Pattern A vs B verdict (with position-appropriate probe)")
    summary.append("=" * 78)
    if transition_data["TF"] and transition_data["TT"] and transition_data["FF"]:
        pl_tf = float(np.mean([d["p_last"] for d in transition_data["TF"]]))
        pl_tt = float(np.mean([d["p_last"] for d in transition_data["TT"]]))
        pl_ff = float(np.mean([d["p_last"] for d in transition_data["FF"]]))
        summary.append(f"probe(last) on T->F: {pl_tf:.3f}")
        summary.append(f"probe(last) on T->T: {pl_tt:.3f}  (reference: TRUE-correct ceiling)")
        summary.append(f"probe(last) on F->F: {pl_ff:.3f}  (reference: TRUE-wrong floor)")
        # if pl_tf ~ pl_ff -> Pattern A (probe tracks last)
        # if pl_tf ~ pl_tt -> Pattern B (probe anchored to first)
        if pl_tf < (pl_tt + pl_ff) / 2:
            summary.append("")
            summary.append("=> probe(last) on T->F is closer to F->F than to T->T.")
            summary.append("   PATTERN A confirmed with position-appropriate probe.")
        else:
            summary.append("")
            summary.append("=> probe(last) on T->F is closer to T->T than to F->F.")
            summary.append("   PATTERN B detected -- representation anchored to first answer.")

    txt = "\n".join(summary)
    print(txt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(txt + "\n")

    # Scatter figure
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    colors = {"TT": "#3a8b2f", "TF": "#c45252", "FT": "#dba24c", "FF": "#888888"}
    labels = {"TT": "both correct", "TF": "drift correct->wrong",
              "FT": "drift wrong->correct", "FF": "both wrong"}
    for tag, lst in transition_data.items():
        if not lst:
            continue
        xs = [d["p_first"] for d in lst]
        ys = [d["p_last"] for d in lst]
        ax.scatter(xs, ys, s=15, alpha=0.45, c=colors[tag], label=f"{labels[tag]} (n={len(lst)})")
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=0.8, alpha=0.6, label="y=x")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("position-appropriate probe at FIRST <answer>", fontsize=11)
    ax.set_ylabel("position-appropriate probe at LAST <answer>", fontsize=11)
    ax.set_title(f"Within-rollout probe trajectory, POSITION-APPROPRIATE probe\n"
                  f"{args.ckpt} L{args.layer}, diagonal AUROC={diag_auc:.3f}, n={sum(len(v) for v in transition_data.values())}",
                  fontsize=10)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
    fig.savefig(args.fig)
    plt.close(fig)
    print(f"\nwrote {args.fig}")


if __name__ == "__main__":
    main()
