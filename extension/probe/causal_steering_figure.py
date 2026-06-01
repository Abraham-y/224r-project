"""Causal steering bar chart (fig12).

Reads extension/outputs/n500/causal_steering_full.jsonl, plots accuracy per
(direction, alpha) with binomial 95% CIs. Highlights the null result: probe
direction is indistinguishable from random direction at matched magnitude.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


def binom_ci(k, n, conf=0.95):
    """Wilson 95% CI on accuracy = k/n."""
    if n == 0: return (0.0, 0.0)
    z = 1.96
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def main():
    rows = [json.loads(l) for l in open("extension/outputs/n500/causal_steering_full.jsonl") if l.strip()]
    print(f"loaded {len(rows)} rows")

    # acc per (alpha, direction)
    cells: dict[tuple[float, str], list[bool]] = defaultdict(list)
    for r in rows:
        cells[(r["alpha"], r["direction"])].append(r["new_score"] == 1.0)

    # Build the bar chart structure
    alphas = sorted({k[0] for k in cells if k[0] != 0.0})  # [0.5, 1.0, 2.0]
    baseline_n = len(cells[(0.0, "zero")])
    baseline_k = sum(cells[(0.0, "zero")])
    baseline_acc = baseline_k / baseline_n
    baseline_lo, baseline_hi = binom_ci(baseline_k, baseline_n)

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=170)

    # x positions: 3 groups (one per alpha), 2 bars per group (probe + random)
    group_centers = np.arange(len(alphas))
    bar_width = 0.36

    probe_means = []; probe_los = []; probe_his = []
    rand_means = []; rand_los = []; rand_his = []
    for alpha in alphas:
        for direction_name, ms, los, his in (("probe", probe_means, probe_los, probe_his),
                                             ("rand", rand_means, rand_los, rand_his)):
            cell = cells.get((alpha, direction_name), [])
            n = len(cell); k = sum(cell)
            if n == 0:
                ms.append(0); los.append(0); his.append(0)
            else:
                acc = k / n
                lo, hi = binom_ci(k, n)
                ms.append(acc); los.append(lo); his.append(hi)

    probe_err = [[ms - lo for ms, lo in zip(probe_means, probe_los)],
                 [hi - ms for hi, ms in zip(probe_his, probe_means)]]
    rand_err = [[ms - lo for ms, lo in zip(rand_means, rand_los)],
                [hi - ms for hi, ms in zip(rand_his, rand_means)]]

    ax.bar(group_centers - bar_width / 2, probe_means, bar_width, yerr=probe_err,
           capsize=4, color="#3a6dba", label="probe direction", alpha=0.9,
           edgecolor="black", linewidth=0.6)
    ax.bar(group_centers + bar_width / 2, rand_means, bar_width, yerr=rand_err,
           capsize=4, color="#aaaaaa", label="random direction (matched ||·||)", alpha=0.85,
           edgecolor="black", linewidth=0.6)

    # Baseline line + shaded CI
    ax.axhline(baseline_acc, color="#c45252", linestyle="-", linewidth=1.8,
               label=f"baseline α=0 ({baseline_acc:.3f})", zorder=1)
    ax.axhspan(baseline_lo, baseline_hi, color="#c45252", alpha=0.13, zorder=0)

    # Annotations with delta (probe − random) per alpha
    for i, alpha in enumerate(alphas):
        d = probe_means[i] - rand_means[i]
        ymax = max(probe_his[i], rand_his[i]) + 0.02
        ax.text(group_centers[i], ymax, f"Δ = {d:+.3f}", ha="center", fontsize=9,
                color="black", fontweight="bold")

    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"α = {a} · h_mean_norm" for a in alphas], fontsize=10)
    ax.set_xlabel("steering magnitude (h_mean_norm ≈ 21.84)", fontsize=11)
    ax.set_ylabel("accuracy (n=97 prefixes; Wilson 95% CIs)", fontsize=11)
    ax.set_title("Causal steering at `</think>` (L16 residual, C_outcome)\n"
                 "probe direction is INDISTINGUISHABLE from random-direction perturbation "
                 "(probe−random Δ ∈ [−0.07, +0.02])",
                 fontsize=11)
    ax.set_ylim(0, 0.85)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.35)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.92)
    fig.tight_layout()
    out = "extension/outputs/n500/figures/fig12_causal_steering.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
