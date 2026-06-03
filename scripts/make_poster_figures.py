"""Generate polished PDF figures for the poster.

Produces:
  figures/poster_probe_auroc.pdf       (column 1: probe AUROC horizontal bars)
  figures/poster_causal_steering.pdf   (column 2: probe vs random line plot)
  figures/poster_goodhart_runA.pdf     (column 2: runA training trajectory)
  figures/poster_post_goodhart_delta.pdf (column 3: Delta probe-vs-random bars)

Run:  python scripts/make_poster_figures.py
Upload the resulting PDFs to Overleaf in a figures/ directory, then
in poster.tex replace each \\begin{tikzpicture}...\\end{tikzpicture}
block with \\includegraphics[width=0.95\\linewidth]{figures/poster_*.pdf}
"""
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


def fig_probe_auroc():
    # Thin/flat aspect: wider, shorter
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    positions = ["neutral", "assertion", "pre_answer"]
    sft = [0.562, 0.887, 0.904]
    out = [0.562, 0.852, 0.980]
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


def fig_causal_steering():
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    alphas = [0, 0.5, 1, 2]
    probe = [0.577, 0.567, 0.598, 0.515]
    rand = [0.577, 0.639, 0.577, 0.546]
    ax.plot(alphas, probe, "o-", color=STANFORD_RED, linewidth=2.2,
            markersize=10, label="Probe direction")
    ax.plot(alphas, rand, "s-", color="#444", linewidth=2.2,
            markersize=9, label="Random direction")
    ax.axhline(y=0.577, linestyle="--", color="#888", linewidth=1, alpha=0.7)
    ax.text(2.05, 0.577, "baseline", fontsize=10, color="#666", va="center")
    ax.set_xlabel(r"Steering magnitude $\alpha$")
    ax.set_ylabel("Generation accuracy")
    ax.set_xticks([0, 0.5, 1, 2])
    ax.set_yticks([0.4, 0.5, 0.6, 0.7])
    ax.set_ylim(0.40, 0.72)
    ax.set_xlim(-0.05, 2.25)
    ax.yaxis.grid(True, **GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "poster_causal_steering.pdf")
    plt.close(fig)


def fig_goodhart_runA():
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    steps = [0, 10, 20, 30, 40, 50, 60, 70, 90, 99]
    probe = [0.452, 0.447, 0.561, 0.553, 0.687, 0.809, 0.947, 0.978, 0.988, 0.991]
    verif = [0.572, 0.490, 0.582, 0.525, 0.528, 0.479, 0.385, 0.337, 0.310, 0.321]
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


def fig_post_goodhart_delta():
    # Thin/flat aspect: wider, shorter
    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    labels = ["vanilla\n($C_{outcome}$)", "post-Goodhart\n(runA)"]
    deltas = [0.021, 0.083]
    # null band shading
    ax.axvspan(-0.07, 0.02, color="#bbb", alpha=0.35, zorder=0, label="null band")
    bars = ax.barh(labels, deltas, color=STANFORD_RED, edgecolor="#5a0f0f",
                   linewidth=0.8, height=0.5, zorder=2)
    for bar, v in zip(bars, deltas):
        ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                f"$\\Delta = {v:+.3f}$", va="center", fontsize=11)
    # null-band annotation positioned above the top bar
    ax.text(-0.025, 1.62, "null band $[-0.07, +0.02]$",
            ha="center", fontsize=10, color="#555")
    ax.axvline(x=0, color="#444", linewidth=1, alpha=0.6)
    ax.set_xticks([-0.10, -0.05, 0, 0.05, 0.10])
    ax.set_xlim(-0.12, 0.14)
    ax.set_xlabel(r"$\Delta$ probe-vs-random accuracy at $\alpha = 1.0$")
    ax.xaxis.grid(True, **GRID)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(OUT / "poster_post_goodhart_delta.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_probe_auroc()
    fig_causal_steering()
    fig_goodhart_runA()
    fig_post_goodhart_delta()
    print(f"Wrote 4 figures to {OUT.resolve()}/")
