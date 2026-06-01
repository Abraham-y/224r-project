"""Headline figure: pre_answer vs assertion AUROC over training (Option B),
with bootstrap 95% CIs per snapshot.

Inputs:
  - extension/cache/probe_cache_n500_clean406/   (C_SFT, C_outcome final)
  - extension/cache/probe_cache_dynamics_optB/   (snapshots 30, 60, 90)

Outputs:
  - extension/outputs/n500/figures/fig13_headline_dynamics.png
  - extension/outputs/n500/text/24_headline_dynamics_with_cis.txt
"""

from __future__ import annotations

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

FINAL_CACHE = "extension/cache/probe_cache_n500_clean406"
DYN_CACHE = "extension/cache/probe_cache_dynamics_optB"
LAYER = 16


def load_cell(npz_path):
    if not os.path.exists(npz_path):
        return None
    with np.load(npz_path) as d:
        X = d["X"]; y = d["y"]
    meta = json.load(open(npz_path.replace(".npz", ".meta.json")))
    groups = np.array([m["prompt_idx"] for m in meta])
    return X, y, groups


def balanced_auroc(X, y, groups, seed=0):
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    n = min(len(pos), len(neg))
    if n < 5: return float("nan")
    idx = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
    Xs, ys, gs = X[idx], y[idx], groups[idx]
    preds = np.full(len(ys), np.nan)
    for tr, te in GroupKFold(5).split(Xs, ys, gs):
        sc = StandardScaler().fit(Xs[tr])
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(Xs[tr]), ys[tr])
        preds[te] = clf.predict_proba(sc.transform(Xs[te]))[:, 1]
    mask = ~np.isnan(preds)
    if len(np.unique(ys[mask])) < 2: return float("nan")
    return float(roc_auc_score(ys[mask], preds[mask]))


def bootstrap_ci(X, y, groups, B=80, frac=0.8, seed=42):
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    k = max(5, int(round(frac * len(uniq))))
    aurocs = []
    for b in range(B):
        sampled = rng.choice(uniq, size=k, replace=False)
        mask = np.isin(groups, sampled)
        a = balanced_auroc(X[mask], y[mask], groups[mask], seed=b)
        if not np.isnan(a):
            aurocs.append(a)
    aurocs = np.array(aurocs)
    return float(np.median(aurocs)), float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))


def main():
    snapshots = [
        ("C_SFT (pre-RL)", 0, f"{FINAL_CACHE}/C_SFT_l{LAYER}_pre_answer.npz", f"{FINAL_CACHE}/C_SFT_l{LAYER}_assertion.npz"),
        ("step 30",        30, f"{DYN_CACHE}/C_outcome_step_30_l{LAYER}_pre_answer.npz", f"{DYN_CACHE}/C_outcome_step_30_l{LAYER}_assertion.npz"),
        ("step 60",        60, f"{DYN_CACHE}/C_outcome_step_60_l{LAYER}_pre_answer.npz", f"{DYN_CACHE}/C_outcome_step_60_l{LAYER}_assertion.npz"),
        ("step 90",        90, f"{DYN_CACHE}/C_outcome_step_90_l{LAYER}_pre_answer.npz", f"{DYN_CACHE}/C_outcome_step_90_l{LAYER}_assertion.npz"),
        ("C_outcome (final)", 100, f"{FINAL_CACHE}/C_outcome_l{LAYER}_pre_answer.npz", f"{FINAL_CACHE}/C_outcome_l{LAYER}_assertion.npz"),
    ]

    rows = []
    for label, step, pre_path, ass_path in snapshots:
        pre = load_cell(pre_path); ass = load_cell(ass_path)
        if pre is None or ass is None:
            print(f"missing {label}"); continue
        print(f"[{label}] computing point + bootstrap CIs ...", flush=True)
        pre_med, pre_lo, pre_hi = bootstrap_ci(*pre)
        ass_med, ass_lo, ass_hi = bootstrap_ci(*ass)
        # Point estimate too (single seed, balanced GroupKFold)
        pre_pt = balanced_auroc(*pre)
        ass_pt = balanced_auroc(*ass)
        gap = pre_pt - ass_pt
        rows.append({
            "label": label, "step": step,
            "pre_pt": pre_pt, "pre_med": pre_med, "pre_lo": pre_lo, "pre_hi": pre_hi,
            "ass_pt": ass_pt, "ass_med": ass_med, "ass_lo": ass_lo, "ass_hi": ass_hi,
            "gap": gap,
        })
        print(f"  pre={pre_pt:.3f} CI[{pre_lo:.3f},{pre_hi:.3f}]  ass={ass_pt:.3f} CI[{ass_lo:.3f},{ass_hi:.3f}]  gap={gap:+.3f}", flush=True)

    # Text table
    lines = ["Headline dynamics with bootstrap 95% CIs (L16, n_problems=406 clean-406; snapshots n=200)",
             f"{'snapshot':<22}{'step':>5} {'pre_answer (point [CI])':>30} {'assertion (point [CI])':>30}  {'gap':>6}"]
    for r in rows:
        lines.append(
            f"{r['label']:<22}{r['step']:>5}  "
            f"{r['pre_pt']:.3f} [{r['pre_lo']:.3f},{r['pre_hi']:.3f}]  "
            f"{r['ass_pt']:.3f} [{r['ass_lo']:.3f},{r['ass_hi']:.3f}]   {r['gap']:+.3f}"
        )
    txt = "\n".join(lines)
    print()
    print(txt)
    os.makedirs("extension/outputs/n500/text", exist_ok=True)
    with open("extension/outputs/n500/text/24_headline_dynamics_with_cis.txt", "w") as f:
        f.write(txt + "\n")

    # Figure
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=170)
    xs = [r["step"] for r in rows]
    pre_y = [r["pre_pt"] for r in rows]
    pre_lo = [r["pre_lo"] for r in rows]; pre_hi = [r["pre_hi"] for r in rows]
    ass_y = [r["ass_pt"] for r in rows]
    ass_lo = [r["ass_lo"] for r in rows]; ass_hi = [r["ass_hi"] for r in rows]

    # 95% CI bands
    ax.fill_between(xs, pre_lo, pre_hi, color="#3a8b2f", alpha=0.18)
    ax.fill_between(xs, ass_lo, ass_hi, color="#c45252", alpha=0.18)

    ax.plot(xs, pre_y, marker="o", color="#3a8b2f", linewidth=2.2, markersize=9, label="probe AUROC at `</think>` (trace-final)")
    ax.plot(xs, ass_y, marker="s", color="#c45252", linewidth=2.2, markersize=9, label="probe AUROC at confidence-asserting tokens")

    # Gap shading between the two lines (visual emphasis on the headline quantity)
    ax.fill_between(xs, ass_y, pre_y, where=[p > a for p, a in zip(pre_y, ass_y)],
                    color="#888888", alpha=0.13, interpolate=True, label="gap (pre − assertion)")

    # Gap annotations at each point
    for r in rows:
        ax.annotate(f"gap={r['gap']:+.3f}",
                    xy=(r["step"], (r["pre_pt"] + r["ass_pt"]) / 2),
                    xytext=(6, 0), textcoords="offset points", fontsize=8.5, color="#444",
                    va="center", ha="left")

    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.7, alpha=0.55, label="chance")
    ax.set_xlabel("RLOO training step (Option B: fresh per-snapshot rollouts; n=200)", fontsize=11)
    ax.set_ylabel("balanced GroupKFold(5) probe AUROC (L16)", fontsize=11)
    ax.set_title("Position-decoupling emerges over outcome RL\n"
                 "shaded bands = bootstrap 95% CIs; gap grows from 0.019 (C_SFT) to 0.193 (C_outcome)",
                 fontsize=11.5)
    # x-axis ticks at the actual step values, labeled with snapshot names
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['label']}\n(step {r['step']})" for r in rows], fontsize=9)
    ax.set_ylim(0.45, 0.95)
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.92)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    out_fig = "extension/outputs/n500/figures/fig13_headline_dynamics.png"
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_fig}")


if __name__ == "__main__":
    main()
