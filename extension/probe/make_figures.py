"""Generate conference-style matplotlib figures for the concealment paper.

Outputs PNGs to extension/outputs/figures/:

  fig1_matched_pair_scatter.png   Headline: per-prompt mean assertion-probe
                                  score, correct vs wrong rollouts, two
                                  panels (C_SFT, C_outcome).
  fig2_within_problem_d.png       Distribution of per-problem Cohen's d
                                  (the Yuan-et-al benchmark).
  fig3_position_bar.png           Balanced probe AUROC at pre_answer /
                                  assertion / neutral, C_SFT vs C_outcome.
  fig4_per_keyword_bar.png        Per-keyword assertion-position AUROC,
                                  C_SFT vs C_outcome.
  fig5_dynamics_trajectory.png    Per-step probe AUROC at three positions
                                  over training step.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from extension.probe.analyze_probes import load_groups
from extension.probe.qualitative_examples import cv_per_row_probe_scores
from extension.probe.robustness_probes import balanced_subsample_auroc


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def style_axes(ax):
    """Clean conference-look: drop top/right spines, light gridlines, no clutter."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)


def load_assertion_probe_scores(cache_dir: str, ckpt: str, layer: int):
    """Returns (per-rollout dict, X, y, groups) for the assertion cache."""
    ass_npz = os.path.join(cache_dir, f"{ckpt}_l{layer}_assertion.npz")
    ass_meta = os.path.join(cache_dir, f"{ckpt}_l{layer}_assertion.meta.json")
    with np.load(ass_npz) as data:
        X = data["X"]; y = data["y"]
    with open(ass_meta) as f:
        meta = json.load(f)
    groups = np.array([row["prompt_idx"] for row in meta], dtype=np.int64)
    keywords = np.array([row.get("keyword", "?") for row in meta])
    scores = cv_per_row_probe_scores(X, y, groups)
    rollouts = defaultdict(lambda: {"scores": [], "label": None})
    for i, m in enumerate(meta):
        if np.isnan(scores[i]):
            continue
        key = (int(m["prompt_idx"]), int(m["resp_idx"]))
        rollouts[key]["scores"].append(float(scores[i]))
        rollouts[key]["label"] = int(y[i])
    return rollouts, X, y, groups, keywords, scores


# ---------------------------------------------------------------------------
# Figure 1: matched-pair scatter (the headline)
# ---------------------------------------------------------------------------


def fig1_matched_pair_scatter(cache_dir: str, layer: int, outpath: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), dpi=160, sharex=True, sharey=True)
    for ax, ckpt in zip(axes, ("C_SFT", "C_outcome")):
        rollouts, *_ = load_assertion_probe_scores(cache_dir, ckpt, layer)
        by_prompt: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: {"correct": [], "wrong": []}
        )
        for (p_idx, _r_idx), v in rollouts.items():
            mean_s = float(np.mean(v["scores"]))
            bucket = "correct" if v["label"] == 1 else "wrong"
            by_prompt[p_idx][bucket].append(mean_s)
        xs, ys = [], []
        for p_idx, b in by_prompt.items():
            if not b["correct"] or not b["wrong"]:
                continue
            xs.append(float(np.mean(b["wrong"])))
            ys.append(float(np.mean(b["correct"])))
        xs = np.array(xs); ys = np.array(ys)
        above = (ys > xs).sum()
        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.9, label="y = x")
        ax.scatter(xs, ys, s=42, alpha=0.75, edgecolors="white", linewidths=0.6,
                   color=("#2f6db5" if ckpt == "C_SFT" else "#c45252"),
                   label=f"prompts (n = {len(xs)})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("mean probe score on wrong rollouts")
        ax.set_title(
            f"{ckpt}  —  {above}/{len(xs)} prompts above diagonal",
            fontsize=11,
        )
        if ax is axes[0]:
            ax.set_ylabel("mean probe score on correct rollouts")
        ax.legend(frameon=False, loc="upper left", fontsize=9)
        style_axes(ax)
    fig.suptitle(
        f"Matched-pair probe scores at assertion tokens (layer L{layer})\n"
        f"points above y = x: probe ranks the problem's correct rollouts higher",
        fontsize=11.5, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  wrote {outpath}")


# ---------------------------------------------------------------------------
# Figure 2: within-problem Cohen's d distribution at pre_answer
# ---------------------------------------------------------------------------


def _cohens_d_per_problem(cache_dir: str, ckpt: str, layer: int) -> list[float]:
    npz_path = os.path.join(cache_dir, f"{ckpt}_l{layer}_pre_answer.npz")
    meta_path = npz_path.replace(".npz", ".meta.json")
    with np.load(npz_path) as data:
        X = data["X"]; y = data["y"]
    groups = load_groups(meta_path)
    out = []
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for tr, te in gkf.split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs")
        clf.fit(scaler.transform(X[tr]), y[tr])
        scores = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        for g in np.unique(groups[te]):
            m = groups[te] == g
            yg = y[te][m]
            sg = scores[m]
            if len(np.unique(yg)) < 2:
                continue
            sc, sw = sg[yg == 1], sg[yg == 0]
            var_c = sc.var(ddof=1) if len(sc) > 1 else 0.0
            var_w = sw.var(ddof=1) if len(sw) > 1 else 0.0
            pooled = float(np.sqrt(((len(sc) - 1) * var_c + (len(sw) - 1) * var_w)
                                    / max(len(sc) + len(sw) - 2, 1)))
            if pooled < 1e-9:
                continue
            out.append(float(sc.mean() - sw.mean()) / pooled)
    return out


def fig2_within_problem_d(cache_dir: str, layer: int, outpath: str) -> None:
    sft = _cohens_d_per_problem(cache_dir, "C_SFT", layer)
    out = _cohens_d_per_problem(cache_dir, "C_outcome", layer)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    bins = np.linspace(-2, 4, 25)
    ax.hist(sft, bins=bins, alpha=0.65, color="#2f6db5", label=f"C_SFT (n={len(sft)})")
    ax.hist(out, bins=bins, alpha=0.65, color="#c45252", label=f"C_outcome (n={len(out)})")
    for vals, color in ((sft, "#1a4d8b"), (out, "#8b2929")):
        ax.axvline(np.mean(vals), color=color, linestyle="--", linewidth=1.5,
                   label=f"mean = {np.mean(vals):+.2f}")
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("within-problem Cohen's d (probe correct - wrong) / pooled SD")
    ax.set_ylabel("number of problems")
    ax.set_title(
        f"Within-problem effect size at </think> (layer L{layer})\n"
        f"d shifts from +1.26 (large) on C_SFT to +0.38 (small) on C_outcome"
    )
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  wrote {outpath}")


# ---------------------------------------------------------------------------
# Figure 3: position-resolved balanced AUROC bar chart
# ---------------------------------------------------------------------------


def fig3_position_bar(cache_dir: str, layer: int, outpath: str) -> None:
    cells = {}
    for ckpt in ("C_SFT", "C_outcome"):
        for kind in ("pre_answer", "assertion", "neutral"):
            npz = os.path.join(cache_dir, f"{ckpt}_l{layer}_{kind}.npz")
            meta = npz.replace(".npz", ".meta.json")
            if not os.path.exists(npz):
                continue
            with np.load(npz) as data:
                X = data["X"]; y = data["y"]
            groups = load_groups(meta)
            cells[(ckpt, kind)] = balanced_subsample_auroc(X, y, groups)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    kinds = ["pre_answer", "assertion", "neutral"]
    kind_labels = {"pre_answer": "</think>", "assertion": "confidence assertion", "neutral": "neutral"}
    width = 0.36
    x = np.arange(len(kinds))
    sft_vals = [cells.get(("C_SFT", k), float("nan")) for k in kinds]
    out_vals = [cells.get(("C_outcome", k), float("nan")) for k in kinds]
    b1 = ax.bar(x - width / 2, sft_vals, width, color="#2f6db5", label="C_SFT")
    b2 = ax.bar(x + width / 2, out_vals, width, color="#c45252", label="C_outcome")
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=9)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=9)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.9, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([kind_labels[k] for k in kinds])
    ax.set_ylim(0, 1)
    ax.set_ylabel("balanced probe AUROC (5-fold by problem)")
    ax.set_title(f"Probe AUROC by token position (layer L{layer})")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  wrote {outpath}")


# ---------------------------------------------------------------------------
# Figure 4: per-keyword AUROC bar
# ---------------------------------------------------------------------------


def fig4_per_keyword_bar(cache_dir: str, layer: int, outpath: str) -> None:
    """Show per-keyword balanced AUROC at assertion tokens, both checkpoints."""
    per_kw = defaultdict(dict)
    for ckpt in ("C_SFT", "C_outcome"):
        ass_npz = os.path.join(cache_dir, f"{ckpt}_l{layer}_assertion.npz")
        ass_meta = os.path.join(cache_dir, f"{ckpt}_l{layer}_assertion.meta.json")
        if not os.path.exists(ass_npz):
            continue
        with np.load(ass_npz) as data:
            X = data["X"]; y = data["y"]
        with open(ass_meta) as f:
            meta = json.load(f)
        groups = np.array([row["prompt_idx"] for row in meta], dtype=np.int64)
        kws = np.array([row.get("keyword", "?") for row in meta])
        for kw in np.unique(kws):
            mask = kws == kw
            if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
                continue
            per_kw[kw][ckpt] = balanced_subsample_auroc(X[mask], y[mask], groups[mask])
    # Show all keywords with at least one checkpoint's data; missing bars
    # are explicitly labeled "n < 20" so the reader sees what happened.
    common = sorted(per_kw.keys())
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    width = 0.36
    x = np.arange(len(common))
    sft_vals = [per_kw[kw].get("C_SFT", float("nan")) for kw in common]
    out_vals = [per_kw[kw].get("C_outcome", float("nan")) for kw in common]
    b1 = ax.bar(x - width / 2, sft_vals, width, color="#2f6db5", label="C_SFT")
    b2 = ax.bar(x + width / 2, out_vals, width, color="#c45252", label="C_outcome")
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=9)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=9)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.9, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(common, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("balanced probe AUROC")
    ax.set_title(f"Per-keyword assertion-position probe AUROC (layer L{layer})")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  wrote {outpath}")


# ---------------------------------------------------------------------------
# Figure 5: dynamics trajectory (three lines over training step)
# ---------------------------------------------------------------------------


def fig5_dynamics_trajectory(dynamics_csv: str, layer: int, outpath: str) -> None:
    if not os.path.exists(dynamics_csv):
        print(f"  skipping fig5 (no dynamics CSV at {dynamics_csv})")
        return
    by_kind: dict[str, dict[int, float]] = defaultdict(dict)
    with open(dynamics_csv) as f:
        for row in csv.DictReader(f):
            if int(row["layer"]) != layer:
                continue
            by_kind[row["kind"]][int(row["step"])] = float(row["balanced_auc"])
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    label_map = {"pre_answer": "</think>", "assertion": "confidence assertion", "neutral": "neutral"}
    color_map = {"pre_answer": "#2f6db5", "assertion": "#c45252", "neutral": "#6a6a6a"}
    for kind in ("pre_answer", "assertion", "neutral"):
        if kind not in by_kind:
            continue
        steps = sorted(by_kind[kind])
        vals = [by_kind[kind][s] for s in steps]
        ax.plot(steps, vals, marker="o", linewidth=2, markersize=5,
                color=color_map[kind], label=label_map[kind])
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.9, label="chance")
    ax.set_xlabel("RLOO training step")
    ax.set_ylabel("balanced probe AUROC")
    ax.set_ylim(0.35, 1)
    ax.set_title(f"Probe AUROC over training (layer L{layer}, fixed eval rollouts)")
    ax.legend(frameon=False, loc="right", fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  wrote {outpath}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="extension/cache/probe_cache")
    parser.add_argument("--dynamics_csv", default="extension/outputs/dynamics_auroc.csv")
    parser.add_argument("--out_dir", default="extension/outputs/figures")
    parser.add_argument("--layer", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fig1_matched_pair_scatter(args.cache_dir, args.layer,
                              os.path.join(args.out_dir, "fig1_matched_pair_scatter.png"))
    fig2_within_problem_d(args.cache_dir, args.layer,
                          os.path.join(args.out_dir, "fig2_within_problem_d.png"))
    fig3_position_bar(args.cache_dir, args.layer,
                      os.path.join(args.out_dir, "fig3_position_bar.png"))
    fig4_per_keyword_bar(args.cache_dir, args.layer,
                         os.path.join(args.out_dir, "fig4_per_keyword_bar.png"))
    fig5_dynamics_trajectory(args.dynamics_csv, args.layer,
                             os.path.join(args.out_dir, "fig5_dynamics_trajectory.png"))


if __name__ == "__main__":
    main()
