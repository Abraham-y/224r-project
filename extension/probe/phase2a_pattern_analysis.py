"""Phase 2A pattern analysis: probe trajectories within multi-answer rollouts.

Trains the trace-final (pre_answer / </think>) probe on C_outcome's
clean-406 cache using GroupKFold-by-prompt, then applies it to every
`<answer>`-opening hidden state in the same rollout. For each rollout we
record probe(first_answer), probe(last_answer), and per-block trajectory.

Outputs:
  - extension/outputs/n500/text/18_phase2a_patterns.txt: aggregate stats
  - extension/outputs/n500/figures/fig9_within_rollout_trajectory.png:
    scatter of (probe_first, probe_last) colored by (first_correct, last_correct).

The patterns we discriminate:
  A: probe tracks each successive answer's correctness
     -> probe(first) ~ first_correct AND probe(last) ~ last_correct
  B: probe stays anchored to first answer
     -> probe(last) tracks first_correct, not last_correct
  C: drifts unpredictably
     -> neither pattern A nor B holds
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


def load_pre_answer(cache_dir: str, ckpt: str, layer: int):
    npz = os.path.join(cache_dir, f"{ckpt}_l{layer}_pre_answer.npz")
    meta_path = npz.replace(".npz", ".meta.json")
    with np.load(npz) as d:
        X = d["X"]; y = d["y"]
    meta = json.load(open(meta_path))
    groups = np.array([m["prompt_idx"] for m in meta])
    return X, y, groups, meta


def load_answers(cache_dir: str, ckpt: str, layer: int):
    npz = os.path.join(cache_dir, f"{ckpt}_l{layer}_answers.npz")
    meta_path = npz.replace(".npz", ".meta.json")
    if not os.path.exists(npz):
        return None, None
    with np.load(npz) as d:
        X = d["X"]
    meta = json.load(open(meta_path))
    return X, meta


def train_probe_with_holdout(X, y, groups, n_splits: int = 5):
    """Fit one probe per held-out fold; return dict prompt_idx -> (scaler, clf)
    where the probe was NOT trained on that prompt."""
    # Map prompt -> fold index
    prompt_to_fold: dict[int, int] = {}
    folds_X, folds_y = [], []
    gkf = GroupKFold(n_splits=n_splits)
    fold_idx = 0
    for tr_idx, te_idx in gkf.split(X, y, groups):
        for p in np.unique(groups[te_idx]):
            prompt_to_fold[int(p)] = fold_idx
        fold_idx += 1

    # Train one probe per fold (trained on rows NOT in that fold's eval set)
    fold_probes: dict[int, tuple] = {}
    for f in range(n_splits):
        # train mask = rows whose prompt is NOT in fold f
        train_mask = np.array([prompt_to_fold[int(g)] != f for g in groups])
        Xs, ys = X[train_mask], y[train_mask]
        # balanced subsample
        pos = np.where(ys == 1)[0]; neg = np.where(ys == 0)[0]
        n = min(len(pos), len(neg))
        if n < 5:
            continue
        rng = np.random.RandomState(42 + f)
        idx = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
        scaler = StandardScaler().fit(Xs[idx])
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(scaler.transform(Xs[idx]), ys[idx])
        fold_probes[f] = (scaler, clf)
    return fold_probes, prompt_to_fold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="extension/cache/probe_cache_n500_clean406")
    parser.add_argument("--answers_dir", default="extension/cache/probe_cache_n500_answers")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--ckpt", default="C_outcome")
    parser.add_argument("--out", default="extension/outputs/n500/text/18_phase2a_patterns.txt")
    parser.add_argument("--fig", default="extension/outputs/n500/figures/fig9_within_rollout_trajectory.png")
    args = parser.parse_args()

    X, y, groups, _ = load_pre_answer(args.cache_dir, args.ckpt, args.layer)
    print(f"[phase2a] pre_answer cache: X={X.shape}, pos%={y.mean():.1%}")

    fold_probes, prompt_to_fold = train_probe_with_holdout(X, y, groups, n_splits=5)
    print(f"[phase2a] trained {len(fold_probes)} probes (one per held-out fold)")

    # Sanity: AUROC on diagonal (just to check the probe is solid)
    preds = np.full(len(y), np.nan)
    for i, p in enumerate(groups):
        f = prompt_to_fold[int(p)]
        if f not in fold_probes:
            continue
        sc, clf = fold_probes[f]
        preds[i] = clf.predict_proba(sc.transform(X[i:i+1]))[:, 1][0]
    mask = ~np.isnan(preds)
    diag_auc = roc_auc_score(y[mask], preds[mask])
    print(f"[phase2a] held-out diagonal AUROC at pre_answer: {diag_auc:.3f}")

    # Apply probe to answer-position hidden states
    Xa, meta_a = load_answers(args.answers_dir, args.ckpt, args.layer)
    if Xa is None:
        raise SystemExit(f"missing answers cache at {args.answers_dir}")
    print(f"[phase2a] answers cache: X={Xa.shape}, n_rows={len(meta_a)}")

    probe_scores = np.full(len(meta_a), np.nan)
    for i, m in enumerate(meta_a):
        f = prompt_to_fold.get(int(m["prompt_idx"]))
        if f is None or f not in fold_probes:
            continue
        sc, clf = fold_probes[f]
        probe_scores[i] = clf.predict_proba(sc.transform(Xa[i:i+1]))[:, 1][0]

    # Group by (prompt_idx, resp_idx)
    by_rollout: dict[tuple[int, int], list] = {}
    for i, m in enumerate(meta_a):
        key = (int(m["prompt_idx"]), int(m["resp_idx"]))
        by_rollout.setdefault(key, []).append((int(m["answer_block_idx"]), m, probe_scores[i]))

    # Sort each rollout by answer_block_idx
    for k in by_rollout:
        by_rollout[k].sort(key=lambda x: x[0])

    # For each multi-answer rollout, extract (probe_first, probe_last, first_correct, last_correct)
    points: list[dict] = []
    for key, lst in by_rollout.items():
        if len(lst) < 2:
            continue
        first = lst[0]
        last = lst[-1]
        p_first, p_last = first[2], last[2]
        if np.isnan(p_first) or np.isnan(p_last):
            continue
        first_corr = first[1]["block_correct"]
        last_corr = last[1]["block_correct"]
        if first_corr is None or last_corr is None:
            continue
        points.append({
            "prompt_idx": key[0], "resp_idx": key[1],
            "p_first": float(p_first), "p_last": float(p_last),
            "first_correct": bool(first_corr), "last_correct": bool(last_corr),
            "n_blocks": len(lst),
            "trajectory": [(b, m["block_correct"], float(s)) for b, m, s in lst],
        })

    print(f"[phase2a] {len(points)} multi-answer rollouts with valid probe scores at first and last")

    # Group by transition class
    classes = {
        "TT": [p for p in points if p["first_correct"] and p["last_correct"]],
        "TF": [p for p in points if p["first_correct"] and not p["last_correct"]],
        "FT": [p for p in points if not p["first_correct"] and p["last_correct"]],
        "FF": [p for p in points if not p["first_correct"] and not p["last_correct"]],
    }

    def mean_pf(ps, key): return float(np.mean([p[key] for p in ps])) if ps else float("nan")
    def median_pf(ps, key): return float(np.median([p[key] for p in ps])) if ps else float("nan")

    summary = [
        "PHASE 2A pattern analysis (within-rollout probe trajectory)",
        f"  ckpt={args.ckpt}, layer=L{args.layer}",
        f"  cache: {args.cache_dir}; answers: {args.answers_dir}",
        "",
        f"Probe diagonal AUROC (held-out by prompt) at pre_answer: {diag_auc:.3f}",
        f"Multi-answer rollouts analyzed: {len(points)}",
        "",
        f"{'transition':<6} {'n':>5} {'probe(first)':>15} {'probe(last)':>14} {'last-first':>11} {'first_acc':>10} {'last_acc':>10}",
        "-" * 76,
    ]
    for tag, ps in classes.items():
        if not ps:
            continue
        pf = mean_pf(ps, "p_first")
        pl = mean_pf(ps, "p_last")
        diff = pl - pf
        first_acc = mean_pf(ps, "first_correct")
        last_acc = mean_pf(ps, "last_correct")
        summary.append(
            f"{tag:<6} {len(ps):>5}  {pf:>14.3f}  {pl:>13.3f}  {diff:>+10.3f}  {first_acc:>10.2f}  {last_acc:>10.2f}"
        )

    # Pattern diagnostics: for the headline T->F (correct -> wrong) class:
    tf = classes["TF"]
    ft = classes["FT"]
    summary.append("")
    summary.append("=" * 78)
    summary.append("Pattern A vs B vs C discrimination")
    summary.append("=" * 78)
    summary.append("")
    summary.append("Pattern A predicts: probe(last) tracks last_correct -> probe(last) on T->F is LOW (~0.3)")
    summary.append("Pattern B predicts: probe(last) tracks first_correct -> probe(last) on T->F is HIGH (~0.7)")
    summary.append("")
    if tf:
        ploast_tf = mean_pf(tf, "p_last")
        pfirst_tf = mean_pf(tf, "p_first")
        summary.append(f"On T->F rollouts (n={len(tf)}):")
        summary.append(f"  mean probe(first) = {pfirst_tf:.3f}  (first answer is correct, probe expects ~high)")
        summary.append(f"  mean probe(last)  = {ploast_tf:.3f}  (last answer is wrong, probe expects ~low under A, ~high under B)")
        # Compare to TT (where both are correct, baseline for "high")
        if classes["TT"]:
            pl_tt = mean_pf(classes["TT"], "p_last")
            summary.append(f"  reference: probe(last) on T->T rollouts = {pl_tt:.3f}")
        if classes["FF"]:
            pl_ff = mean_pf(classes["FF"], "p_last")
            summary.append(f"  reference: probe(last) on F->F rollouts = {pl_ff:.3f}")

    if ft:
        pflast_ft = mean_pf(ft, "p_last")
        pffirst_ft = mean_pf(ft, "p_first")
        summary.append("")
        summary.append(f"On F->T rollouts (n={len(ft)}):")
        summary.append(f"  mean probe(first) = {pffirst_ft:.3f}")
        summary.append(f"  mean probe(last)  = {pflast_ft:.3f}")

    txt = "\n".join(summary)
    print(txt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(txt + "\n")

    # Scatter figure
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    colors = {"TT": "#3a8b2f", "TF": "#c45252", "FT": "#dba24c", "FF": "#888888"}
    labels = {"TT": "both correct", "TF": "drift correct→wrong",
              "FT": "drift wrong→correct", "FF": "both wrong"}
    for tag, ps in classes.items():
        if not ps:
            continue
        xs = [p["p_first"] for p in ps]
        ys = [p["p_last"] for p in ps]
        ax.scatter(xs, ys, s=15, alpha=0.45, c=colors[tag], label=f"{labels[tag]} (n={len(ps)})")
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=0.8, alpha=0.6, label="y=x")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("probe(correct) at FIRST <answer> in rollout", fontsize=11)
    ax.set_ylabel("probe(correct) at LAST <answer> in rollout", fontsize=11)
    ax.set_title(f"Within-rollout probe trajectory ({args.ckpt}, L{args.layer}, n={len(points)} rollouts)\n"
                  f"clean-406, multi-answer rollouts", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
    fig.savefig(args.fig)
    plt.close(fig)
    print(f"\nwrote {args.fig}")


if __name__ == "__main__":
    main()
