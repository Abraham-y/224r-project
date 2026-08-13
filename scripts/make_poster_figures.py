"""Generate polished PDF figures for the poster.

Produces:
  figures/poster_probe_auroc.pdf       (column 1: probe AUROC horizontal bars)
  figures/poster_causal_steering.pdf   (column 2: probe vs random line plot)
  figures/poster_goodhart_runA.pdf     (column 2: runA training trajectory)

NOT produced here: figures/poster_post_goodhart_delta.pdf. That figure is owned
by `extension/probe/plot_causal_steering.py`, which derives it (with bootstrap
CIs) from the steering JSONLs. This script used to write the same path from
hardcoded literals, so whichever ran last silently won.

All numbers are read from `extension/outputs/poster_numbers.json`. Regenerate it
first:

    python scripts/collect_poster_numbers.py     # data -> poster_numbers.json
    python scripts/make_poster_figures.py        # json -> figures/*.pdf
    python extension/probe/plot_causal_steering.py   # the 4th figure

Upload the resulting PDFs to Overleaf in a figures/ directory, then
in poster.tex replace each \\begin{tikzpicture}...\\end{tikzpicture}
block with \\includegraphics[width=0.95\\linewidth]{figures/poster_*.pdf}
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.linewidth": 1.2,
    "axes.edgecolor": "#222",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.color": "#222",
    "ytick.color": "#222",
    "axes.labelcolor": "#222",
    "axes.titlecolor": "#222",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "pdf.fonttype": 42,  # editable text in PDF
})

STANFORD_RED = "#8C1515"
STANFORD_RED_LIGHT = "#C0A0A0"
GRID = {"linestyle": ":", "color": "#888", "alpha": 0.5, "linewidth": 0.8}

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

NUMBERS_PATH = Path("extension/outputs/poster_numbers.json")

# Which of the two held-out AUROC estimators in the codebase to plot. See
# scripts/collect_poster_numbers.py for why there are two. Keep this consistent
# with whichever one the paper text quotes.
AUROC_ESTIMATOR = "auroc_balance_then_cv"


def load_numbers():
    if not NUMBERS_PATH.exists():
        sys.exit(f"{NUMBERS_PATH} not found. Run:\n"
                 f"    python scripts/collect_poster_numbers.py")
    return json.loads(NUMBERS_PATH.read_text())


def fig_probe_auroc(nums):
    # Thin/flat aspect: wider, shorter
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    positions = ["neutral", "assertion", "pre_answer"]
    tbl = nums["probe_auroc_l16"]
    sft = [tbl["C_SFT"][p][AUROC_ESTIMATOR] for p in positions]
    out = [tbl["C_outcome"][p][AUROC_ESTIMATOR] for p in positions]
    y = np.arange(len(positions))
    h = 0.32
    bars_sft = ax.barh(y - h/2, sft, height=h, color=STANFORD_RED_LIGHT,
                       edgecolor="#5a0f0f", linewidth=0.8, label="$C_{SFT}$ (pre-RL)")
    bars_out = ax.barh(y + h/2, out, height=h, color=STANFORD_RED,
                       edgecolor="#5a0f0f", linewidth=0.8, label="$C_{outcome}$ (post-RL)")
    for bars, vals in [(bars_sft, sft), (bars_out, out)]:
        for bar, v in zip(bars, vals):
            ax.text(v + 0.008, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels(positions)
    ax.set_xticks([0.5, 0.7, 0.9, 1.0])
    ax.set_xlim(0.5, 1.07)
    ax.set_xlabel("Held-out AUROC (chance = 0.5, oracle = 1.0)")
    ax.xaxis.grid(True, **GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=11, ncol=2,
              bbox_to_anchor=(1.0, -0.05))
    plt.tight_layout()
    plt.savefig(OUT / "poster_probe_auroc.pdf")
    plt.close(fig)


def fig_causal_steering(nums):
    st = nums["causal_steering_vanilla"]
    if not st:
        print("  [skip] poster_causal_steering.pdf — no steering data in poster_numbers.json")
        return
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    alphas, probe, rand, base = st["alphas"], st["probe"], st["rand"], st["baseline_acc"]
    ax.plot(alphas, probe, "o-", color=STANFORD_RED, linewidth=2.2,
            markersize=10, label="Probe direction")
    ax.plot(alphas, rand, "s-", color="#444", linewidth=2.2,
            markersize=9, label="Random direction")
    ax.axhline(y=base, linestyle="--", color="#888", linewidth=1, alpha=0.7)
    ax.text(max(alphas) + 0.05, base, "baseline", fontsize=10, color="#666", va="center")
    ax.set_xlabel(r"Steering magnitude $\alpha$")
    ax.set_ylabel("Generation accuracy")
    ax.set_xticks(alphas)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7])
    ax.set_ylim(0.40, 0.72)
    ax.set_xlim(-0.05, 2.25)
    ax.yaxis.grid(True, **GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "poster_causal_steering.pdf")
    plt.close(fig)


def fig_goodhart_runA(nums):
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    g = nums["goodhart_runA"]
    steps, probe, verif = g["steps"], g["probe"], g["verifier"]
    ax.plot(steps, probe, "o-", color=STANFORD_RED, linewidth=2.4,
            markersize=8, label="Probe (the reward)")
    ax.plot(steps, verif, "s-", color="#222", linewidth=2.4,
            markersize=7.5, label="Verifier accuracy")
    ax.axvspan(28, 42, color="#FFE", alpha=0.35, zorder=0)
    ax.text(35, 0.95, "Goodhart\nonset", ha="center", fontsize=10, color="#666")
    ax.set_xlabel("RLOO step")
    ax.set_ylabel("value")
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 100)
    ax.set_title(r"runA: init from $C_{outcome}$ (delayed Goodhart)", fontsize=12)
    ax.yaxis.grid(True, **GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "poster_goodhart_runA.pdf")
    plt.close(fig)


# NOTE: fig_post_goodhart_delta() used to live here and wrote
# figures/poster_post_goodhart_delta.pdf from the literals [0.021, 0.083].
# extension/probe/plot_causal_steering.py writes the SAME path, but derived from
# the steering JSONLs with bootstrap CIs. Two writers, one path, last-one-wins.
# Removed here; plot_causal_steering.py is now the single owner of that figure.


if __name__ == "__main__":
    nums = load_numbers()
    fig_probe_auroc(nums)
    fig_causal_steering(nums)
    fig_goodhart_runA(nums)
    print(f"Wrote 3 figures to {OUT.resolve()}/")
    print("For figures/poster_post_goodhart_delta.pdf run:")
    print("    python extension/probe/plot_causal_steering.py")
