"""Per-problem probe-AUROC vs accuracy-delta correlation at TRACE-FINAL.

Parallel of `probe_behavioral_correlation.py` (PR cherry-pick, §8.5), but at
the `</think>` (pre_answer) position instead of the `<answer>`-opening
position. Uses the **existing** clean-406 hidden-state cache so no Modal
forward passes are needed.

Procedure (purely local, ~5 min CPU):
  1. Load `extension/cache/probe_cache_n500_clean406/C_{SFT,outcome}_l16_
     pre_answer.npz` (+ meta with prompt_idx, resp_idx).
  2. For each checkpoint, train held-out probes via GroupKFold(5) by
     prompt_idx (matches §2.4's methodology): for each fold, fit
     Pipeline(StandardScaler, LogisticRegression(C=0.1)) on rows whose
     prompt is NOT in the fold, predict on the fold's rows. Yields one
     held-out probe score per rollout.
  3. Per-problem AUROC over its K rollouts' (probe_score, label) pairs
     (skip if all-same-label -- AUROC undefined).
  4. Per-problem accuracy from the rollout JSONs.
  5. Spearman correlation between probe_drop and accuracy_delta;
     quadrant counts; scatter PNG.

Why parallel to §8.5. The §8.5 PR experiment ran at the `<answer>`-opening
position (where aggregate AUROC drops 0.785 -> 0.703 under RL). This script
runs at trace-final, where aggregate AUROC RISES under RL (0.804 -> 0.896).
If the per-problem correlation is also ~zero, the decoupling is general;
if it's non-zero (positive coupling), the trace-final probe is still
behaviorally tied even though its aggregate AUROC improves.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")


def load_cell(npz_path: str):
    if not os.path.exists(npz_path):
        return None
    with np.load(npz_path) as d:
        X = d["X"]; y = d["y"]
    meta = json.load(open(npz_path.replace(".npz", ".meta.json")))
    groups = np.array([int(m["prompt_idx"]) for m in meta])
    resp = np.array([int(m["resp_idx"]) for m in meta])
    return X, y, groups, resp


def heldout_scores(X, y, groups, n_splits=5, C=0.1):
    """One held-out probe score per row via GroupKFold(n_splits) by prompt."""
    scores = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        pipe = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(C=C, max_iter=2000))])
        pipe.fit(X[tr], y[tr])
        scores[te] = pipe.predict_proba(X[te])[:, 1]
    return scores


def per_problem_auroc(scores, labels, groups):
    """Per-prompt AUROC over its rollouts. NaN if all-same-label."""
    by_p = defaultdict(list)
    for i in range(len(scores)):
        if not np.isnan(scores[i]):
            by_p[int(groups[i])].append((scores[i], int(labels[i])))
    n = max(by_p.keys()) + 1 if by_p else 0
    aurocs = np.full(n, np.nan)
    for p, lst in by_p.items():
        s = np.array([t[0] for t in lst])
        l = np.array([t[1] for t in lst])
        if len(set(l.tolist())) >= 2:
            aurocs[p] = roc_auc_score(l, s)
    return aurocs


def per_problem_acc_from_eval_json(eval_path, n_problems):
    rows = [json.loads(l) for l in open(eval_path) if l.strip()]
    acc = np.full(n_problems, np.nan)
    for p_idx, row in enumerate(rows):
        if p_idx >= n_problems:
            break
        scores = row.get("scores", [])
        if not scores:
            continue
        # Use first 16 (consistent with K=16)
        scores = scores[:16]
        acc[p_idx] = float(np.mean([float(s) == 1.0 for s in scores]))
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="extension/cache/probe_cache_n500_clean406")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--sft_eval", default="eval_c_sft_n500.json")
    parser.add_argument("--outcome_eval", default="eval_c_outcome_n500.json")
    parser.add_argument("--contam_json", default="extension/data/contaminated_prompt_idx.json")
    parser.add_argument("--out_dir", default="extension/outputs/n500/probe_behavioral")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sft = load_cell(os.path.join(args.cache_dir, f"C_SFT_l{args.layer}_pre_answer.npz"))
    out = load_cell(os.path.join(args.cache_dir, f"C_outcome_l{args.layer}_pre_answer.npz"))
    if sft is None or out is None:
        raise SystemExit(f"missing cache in {args.cache_dir}")
    X_sft, y_sft, g_sft, _ = sft
    X_out, y_out, g_out, _ = out
    print(f"[pre_answer] C_SFT     rows={len(y_sft)}  prompts={len(set(g_sft.tolist()))}  pos%={y_sft.mean():.1%}")
    print(f"[pre_answer] C_outcome rows={len(y_out)}  prompts={len(set(g_out.tolist()))}  pos%={y_out.mean():.1%}")

    # Held-out probe scores via GroupKFold(5) by prompt
    print("[pre_answer] training held-out probes (GroupKFold(5)) ...")
    scores_sft = heldout_scores(X_sft, y_sft, g_sft)
    scores_out = heldout_scores(X_out, y_out, g_out)

    # n_problems = max prompt_idx + 1 (clean-406 indices preserve their original 0..499 ids)
    n_problems = int(max(g_sft.max(), g_out.max())) + 1

    auroc_sft = per_problem_auroc(scores_sft, y_sft, g_sft)
    auroc_rloo = per_problem_auroc(scores_out, y_out, g_out)

    # Accuracy from eval JSONs (using original prompt_idx 0..499)
    acc_sft = per_problem_acc_from_eval_json(args.sft_eval, n_problems)
    acc_rloo = per_problem_acc_from_eval_json(args.outcome_eval, n_problems)

    # Filter to clean-406 if contam available
    if os.path.exists(args.contam_json):
        clean = set(int(i) for i in json.load(open(args.contam_json))["clean"])
        clean_mask = np.array([i in clean for i in range(n_problems)])
        for arr in (auroc_sft, auroc_rloo):
            arr[~clean_mask] = np.nan
        print(f"[pre_answer] clean-only filter: {clean_mask.sum()} of {n_problems} kept")
    accuracy_delta = acc_rloo - acc_sft
    probe_drop = auroc_sft - auroc_rloo

    # Spearman over problems with defined probe_drop AND accuracy_delta
    from scipy.stats import spearmanr
    valid = np.isfinite(probe_drop) & np.isfinite(accuracy_delta)
    n_valid = int(valid.sum())
    spearman_r, spearman_p = spearmanr(probe_drop[valid], accuracy_delta[valid])
    print(f"[pre_answer] Spearman r={spearman_r:.4f} p={spearman_p:.3e}  n={n_valid}")

    # Quadrant counts
    pd_v = probe_drop[valid]; ad_v = accuracy_delta[valid]
    tr = int(((pd_v > 0) & (ad_v > 0)).sum())   # decoupling
    br = int(((pd_v > 0) & (ad_v < 0)).sum())   # damage
    tl = int(((pd_v < 0) & (ad_v > 0)).sum())
    bl = int(((pd_v < 0) & (ad_v < 0)).sum())
    on_axis = int(((pd_v == 0) | (ad_v == 0)).sum())
    print(f"  decoupling (probe down, acc up):  {tr} ({100*tr/n_valid:.1f}%)")
    print(f"  damage     (probe down, acc down): {br} ({100*br/n_valid:.1f}%)")
    print(f"  both up    (probe up, acc up):     {tl} ({100*tl/n_valid:.1f}%)")
    print(f"  noise      (probe up, acc down):   {bl} ({100*bl/n_valid:.1f}%)")
    print(f"  on axis:                            {on_axis} ({100*on_axis/n_valid:.1f}%)")

    # AUROC distributions
    print(f"\nC_SFT     per-problem AUROC: mean={np.nanmean(auroc_sft):.3f} med={np.nanmedian(auroc_sft):.3f} n={(~np.isnan(auroc_sft)).sum()}")
    print(f"C_outcome per-problem AUROC: mean={np.nanmean(auroc_rloo):.3f} med={np.nanmedian(auroc_rloo):.3f} n={(~np.isnan(auroc_rloo)).sum()}")
    print(f"probe_drop: mean={np.nanmean(probe_drop):.3f} med={np.nanmedian(probe_drop):.3f}")
    print(f"accuracy_delta: mean={np.nanmean(accuracy_delta):.3f}  (RLOO - SFT)")

    # Scatter
    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.axvline(0, color="k", lw=0.8, alpha=0.5)
    colors = np.where(ad_v > 0.1, "tab:green",
                      np.where(ad_v < -0.1, "tab:red", "tab:gray"))
    ax.scatter(pd_v, ad_v, c=colors, s=36, alpha=0.75, edgecolors="white", linewidths=0.4)
    # Regression line
    if n_valid >= 2:
        b, a = np.polyfit(pd_v, ad_v, 1)
        xs = np.linspace(pd_v.min(), pd_v.max(), 100)
        ax.plot(xs, b*xs + a, "k--", lw=1.6, label=f"fit: y = {b:.2f}x + {a:.2f}")
        ax.legend(loc="lower left", fontsize=9)
    ax.set_xlabel("probe_drop  =  AUROC(SFT) - AUROC(RLOO)   (>0 : probe got worse)  [pre_answer / </think>]")
    ax.set_ylabel("accuracy_delta  =  acc(RLOO) - acc(SFT)")
    ax.set_title("Per-problem probe AUROC drop vs behavioral accuracy change  (TRACE-FINAL position)")
    p_sci = (spearman_p is not None and not math.isnan(spearman_p) and spearman_p < 1e-3)
    p_txt = (f"{spearman_p:.2e}" if p_sci else f"{spearman_p:.3f}")
    ax.text(0.5, 0.99, f"Spearman r = {spearman_r:.3f}, p = {p_txt} (n={n_valid})",
            transform=ax.transAxes, ha="center", va="top", fontsize=11,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    quad = dict(fontsize=8.5, alpha=0.65, style="italic")
    ax.text(0.02, 0.92, "probe stable\naccuracy improved", transform=ax.transAxes,
            ha="left", va="top", color="tab:green", **quad)
    ax.text(0.98, 0.92, "probe dropped\naccuracy improved\n(decoupling)",
            transform=ax.transAxes, ha="right", va="top", color="tab:olive", **quad)
    ax.text(0.02, 0.06, "probe stable\naccuracy degraded", transform=ax.transAxes,
            ha="left", va="bottom", color="tab:gray", **quad)
    ax.text(0.98, 0.06, "probe dropped\naccuracy degraded\n(damage)",
            transform=ax.transAxes, ha="right", va="bottom", color="tab:red", **quad)
    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "probe_behavioral_correlation_pre_answer.png")
    fig.savefig(out_png)
    plt.close(fig)
    print(f"\n[pre_answer] wrote {out_png}")

    # JSON
    payload = {
        "position": "pre_answer (</think>)",
        "layer": args.layer,
        "methodology": "GroupKFold(5) by prompt for held-out probe scores",
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "n_problems_used": n_valid,
        "quadrants": {"decoupling": tr, "damage": br, "both_up": tl, "noise": bl, "on_axis": on_axis},
        "probe_drop_mean": float(np.nanmean(probe_drop)),
        "accuracy_delta_mean": float(np.nanmean(accuracy_delta)),
        "auroc_sft_mean": float(np.nanmean(auroc_sft)),
        "auroc_rloo_mean": float(np.nanmean(auroc_rloo)),
        "n_problems_with_defined_sft_auroc": int((~np.isnan(auroc_sft)).sum()),
        "n_problems_with_defined_rloo_auroc": int((~np.isnan(auroc_rloo)).sum()),
        "auroc_sft": [None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x) for x in auroc_sft],
        "auroc_rloo": [None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x) for x in auroc_rloo],
        "accuracy_sft": [None if math.isnan(x) else float(x) for x in acc_sft],
        "accuracy_rloo": [None if math.isnan(x) else float(x) for x in acc_rloo],
        "probe_drop": [None if math.isnan(x) else float(x) for x in probe_drop],
        "accuracy_delta": [None if math.isnan(x) else float(x) for x in accuracy_delta],
    }
    out_json = os.path.join(args.out_dir, "probe_behavioral_correlation_pre_answer.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[pre_answer] wrote {out_json}")


if __name__ == "__main__":
    main()
