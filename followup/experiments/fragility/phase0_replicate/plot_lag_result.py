"""The lag figure: the warning period you do not get.

The lag is a claim about time -- two things separating, and a third staying flat
while they do -- and it has been carried by a table. A figure states it in one
glance, which is worth a page of prose in a 6-page paper.

Two stacked panels sharing the RLOO-step axis:

  top     the attack. Probe score climbs, true accuracy falls. These diverge
          from step ~10.
  bottom  the monitor's own report. Frozen AUROC on each checkpoint's OWN fresh
          rollouts, with the across-seed band. Flat until step 50. The vanilla
          verifier-RL control is drawn alongside and RISES, so the decline is
          not something training does to a frozen probe.

The shaded span is the interval where the policy is already gaming the monitor
and the monitor still reports healthy discrimination -- the figure's entire
point, and the reason it is annotated rather than left for the reader to infer.

Regenerated from the parquet metric store, never from a derived CSV: quoting a
stale single-seed CSV is what put wrong numbers in the paper in the first place.
Run verify_lag_result.py first; this reads the JSON it writes, so the figure and
the table cannot disagree.

    python followup/experiments/fragility/phase0_replicate/verify_lag_result.py
    python followup/experiments/fragility/phase0_replicate/plot_lag_result.py
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_RESULT = os.path.join(_ROOT, "followup", "results", "fragility", "lag_result.json")

FLAT_THROUGH, DROP_AT = 40, 50

INK, MUTED = "#1B1F24", "#6B7279"
ATTACK, MONITOR, CONTROL = "#A2452A", "#1B6A5E", "#7A8794"


def series(run: dict, key: str) -> tuple[list[int], list[float]]:
    steps = sorted(int(s) for s in run["by_step"])
    return steps, [run["by_step"][str(s)][key] for s in steps]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="followup/results/fragility/figures/lag.pdf")
    args = ap.parse_args()

    if not os.path.exists(_RESULT):
        raise SystemExit(f"{_RESULT} missing -- run verify_lag_result.py first")
    d = json.load(open(_RESULT))
    runA = d["runs"]["runA (probe-as-reward)"]
    ctrl = d["runs"].get("vanilla RLOO (control)")

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(6.2, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.16})

    # --- top: the attack -----------------------------------------------------
    steps, probe = series(runA, "probe_score_mean")
    _, acc = series(runA, "verifier_acc")
    ax0.plot(steps, probe, "-o", ms=3.4, lw=1.7, color=ATTACK,
             label="monitor's mean score")
    ax0.plot(steps, acc, "-s", ms=3.4, lw=1.7, color=INK, label="true accuracy")
    ax0.set_ylabel("score / accuracy", fontsize=8.5)
    ax0.set_ylim(0, 1.05)
    ax0.legend(fontsize=7.6, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.02))
    ax0.set_title("Reward hacking starts ~40 steps before the monitor notices",
                  fontsize=10, color=INK, pad=7, loc="left")

    # --- bottom: what the monitor reports about itself ------------------------
    # Draw the BALANCED variant: it varies with the class-balancing subsample, so
    # its across-seed band is a real interval. The unbalanced variant is exactly
    # reproducible, so a band drawn from it is zero-width and would imply a
    # precision the measurement does not have.
    bal = runA["balanced_by_step"]
    bsteps = sorted(int(k) for k in bal)
    aur = [bal[str(s)]["mean"] for s in bsteps]
    lo = [bal[str(s)]["min"] for s in bsteps]
    hi = [bal[str(s)]["max"] for s in bsteps]
    steps = bsteps
    ax1.fill_between(steps, lo, hi, color=MONITOR, alpha=0.20, lw=0,
                     label="across-seed range")
    ax1.plot(steps, aur, "-o", ms=3.4, lw=1.9, color=MONITOR,
             label="monitor AUROC, attacked run")
    if ctrl:
        cs, ca = series(ctrl, "auroc_mean")
        ax1.plot(cs, ca, "--^", ms=3.4, lw=1.5, color=CONTROL,
                 label="same monitor, verifier-RL control")
    ax1.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax1.text(99, 0.512, "chance", fontsize=7, color=MUTED, ha="right")
    ax1.set_ylabel("frozen AUROC\n(on each step's own rollouts)", fontsize=8.5)
    ax1.set_xlabel("RLOO step", fontsize=9)
    ax1.set_ylim(0.47, 0.99)
    ax1.legend(fontsize=7.6, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0))

    # The point of the whole figure, marked rather than left to be inferred.
    for ax in (ax0, ax1):
        ax.axvspan(0, FLAT_THROUGH, color=ATTACK, alpha=0.06, lw=0)
        ax.axvline(DROP_AT, color=MUTED, lw=0.9, ls="--")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8, colors=MUTED)
    ax0.annotate("the policy is already\ngaming the monitor here",
                 xy=(20, 0.86), fontsize=7.8, color=ATTACK, ha="center")
    ax1.annotate("...and the monitor's own metric\nis flat throughout",
                 xy=(24, 0.80), xytext=(2, 0.925), fontsize=7.8, color=MONITOR,
                 arrowprops=dict(arrowstyle="->", color=MONITOR, lw=0.9))
    ax1.annotate("AUROC finally moves", xy=(DROP_AT + 1, 0.685),
                 xytext=(60, 0.76), fontsize=7.8, color=INK,
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))

    out = os.path.join(_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {args.out} (+ .png)")


if __name__ == "__main__":
    main()
