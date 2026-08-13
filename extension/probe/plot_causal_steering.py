"""Regenerate the post-Goodhart causal-steering figure WITH 95% bootstrap CI error bars.

Three conditions at alpha=1.0, computed from the saved steering JSONLs via the
same cluster-bootstrap (on prompt_idx) used in causal_steering_stats.py:
  - optimized probe direction on the vanilla C_outcome checkpoint  (reference: ~null)
  - optimized probe direction on the runA post-Goodhart checkpoint  (the effect)
  - assertion direction on the same post-Goodhart checkpoint        (specificity control)

The shaded band is the vanilla probe-vs-random null band [-0.07, +0.02].
Writes figures/poster_post_goodhart_delta.pdf (the file report.tex includes).
Pure CPU analysis on existing data.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causal_steering_stats import load, index_by_prompt, cluster_bootstrap_delta  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ALPHA = 1.0
B = 10000
SEED = 0
NULL_LO, NULL_HI = -0.07, 0.02

CONDS = [
    ("Optimized dir,\nvanilla $C_\\mathrm{outcome}$", "extension/outputs/n500/causal_steering_full.jsonl", "probe", "#9e9e9e"),
    ("Optimized dir,\npost-Goodhart", "causal_steering_runA_postRL.jsonl", "probe", "#d62728"),
    ("Assertion dir,\npost-Goodhart", "causal_steering_runA_assertion_control.jsonl", "probe", "#1f77b4"),
]


def main():
    labels, deltas, los, his = [], [], [], []
    for label, path, direction, _ in CONDS:
        rows = load(path)
        s_by = index_by_prompt(rows, ALPHA, direction)
        r_by = index_by_prompt(rows, ALPHA, "rand")
        d, lo, hi, _, _ = cluster_bootstrap_delta(s_by, r_by, np.random.default_rng(SEED), B)
        labels.append(label); deltas.append(d); los.append(lo); his.append(hi)
        print(f"{label.replace(chr(10), ' '):42s} Delta={d:+.3f}  CI=[{lo:+.3f},{hi:+.3f}]")

    x = np.arange(len(labels))
    yerr = np.array([[d - lo for d, lo in zip(deltas, los)],
                     [hi - d for d, hi in zip(deltas, his)]])
    colors = [c for _, _, _, c in CONDS]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.axhspan(NULL_LO, NULL_HI, color="0.85", zorder=0,
               label=f"vanilla null band [{NULL_LO:+.2f}, {NULL_HI:+.2f}]")
    ax.axhline(0, color="0.4", lw=0.8, zorder=1)
    ax.bar(x, deltas, color=colors, width=0.6, zorder=2, alpha=0.9)
    ax.errorbar(x, deltas, yerr=yerr, fmt="none", ecolor="black",
                elinewidth=1.3, capsize=5, capthick=1.3, zorder=3)
    for xi, d, hi in zip(x, deltas, his):
        ax.annotate(f"{d:+.3f}", (xi, hi), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"$\Delta$ accuracy (steered $-$ random) at $\alpha=1.0$")
    ax.set_title("Causal-axis installation is target-locked\n(95% bootstrap CI, cluster on prompt)",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.margins(y=0.18)
    fig.tight_layout()
    out = "figures/poster_post_goodhart_delta.pdf"
    os.makedirs("figures", exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
