"""Calibration metrics: ECE, MCE, Brier, and reliability diagrams.

Previously this module was a stub -- `ece()` and `plot_reliability()` both raised
NotImplementedError while being listed in the README's repo layout. They are
implemented here because the probe's calibration is load-bearing for two claims
in the paper:

  * the abstention / selective-prediction curves threshold the probe's OUTPUT
    PROBABILITY, so its calibration (not just its ranking) matters;
  * the "probe-mean estimates dataset accuracy" result is, mechanically, a
    statement that the probe is calibrated in the mean -- see
    `probe_as_eval_proxy.py`, which now runs a noise-feature null control
    showing a zero-signal probe already gets within ~1.4 pp.

The verbalized-confidence elicitation described below is a separate experiment
whose script was pruned; the functions here operate on any (probs, correct) pair,
including held-out probe scores.

Usage:

    from extension.metrics.calibration import ece, reliability_table, plot_reliability
    ece(probs, correct, n_bins=10)                      # scalar
    reliability_table(probs, correct, n_bins=10)        # per-bin rows
    plot_reliability(probs, correct, "reliability.png")
"""

from __future__ import annotations

import numpy as np

CONFIDENCE_PROMPT = (
    "Below is a Countdown problem and a candidate answer. "
    "Rate your confidence from 0 to 100 that the candidate answer is correct. "
    "Reply with only an integer between 0 and 100.\n\n"
    "Problem: {problem}\n"
    "Candidate answer: {answer}\n"
    "Confidence (0-100):"
)


def _clean(probs, correct):
    p = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(correct, dtype=float).ravel()
    if p.shape != y.shape:
        raise ValueError(f"probs {p.shape} and correct {y.shape} must match")
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    if p.size == 0:
        raise ValueError("no finite (prob, correct) pairs")
    if p.min() < 0.0 or p.max() > 1.0:
        raise ValueError(f"probs must lie in [0, 1]; got [{p.min()}, {p.max()}]")
    return p, y


def reliability_table(probs, correct, n_bins: int = 10) -> list[dict]:
    """Equal-width binning of `probs` in [0, 1] with per-bin accuracy.

    Bins are half-open [lo, hi) except the last, which is closed, so p == 1.0
    lands in the top bin rather than falling out of the table.
    """
    p, y = _clean(probs, correct)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        rows.append({
            "bin": b,
            "lo": float(edges[b]),
            "hi": float(edges[b + 1]),
            "n": n,
            "mean_confidence": float(p[sel].mean()) if n else float("nan"),
            "accuracy": float(y[sel].mean()) if n else float("nan"),
            "gap": float(p[sel].mean() - y[sel].mean()) if n else float("nan"),
        })
    return rows


def ece(probs, correct, n_bins: int = 10) -> float:
    """Expected Calibration Error: sum_b (n_b/N) * |acc_b - conf_b|."""
    p, y = _clean(probs, correct)
    total = 0.0
    for r in reliability_table(p, y, n_bins):
        if r["n"]:
            total += (r["n"] / p.size) * abs(r["gap"])
    return float(total)


def mce(probs, correct, n_bins: int = 10) -> float:
    """Maximum Calibration Error: max_b |acc_b - conf_b| over non-empty bins."""
    gaps = [abs(r["gap"]) for r in reliability_table(probs, correct, n_bins) if r["n"]]
    return float(max(gaps)) if gaps else float("nan")


def brier(probs, correct) -> float:
    """Brier score (mean squared error of the probability forecast)."""
    p, y = _clean(probs, correct)
    return float(np.mean((p - y) ** 2))


def plot_reliability(probs, correct, out_path: str, n_bins: int = 10,
                     title: str | None = None) -> str:
    """Reliability diagram + confidence histogram. Returns `out_path`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    p, y = _clean(probs, correct)
    rows = reliability_table(p, y, n_bins)
    e, m, b = ece(p, y, n_bins), mce(p, y, n_bins), brier(p, y)

    centers = [(r["lo"] + r["hi"]) / 2 for r in rows]
    accs = [r["accuracy"] for r in rows]
    counts = [r["n"] for r in rows]

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(6, 7), dpi=160, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="perfect calibration")
    ax.bar(centers, accs, width=1.0 / n_bins * 0.9, color="#3a6dba",
           alpha=0.85, edgecolor="#22406e", label="observed accuracy")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(title or f"Reliability  (ECE={e:.3f}, MCE={m:.3f}, Brier={b:.3f}, n={p.size})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, ls="--", lw=0.4, alpha=0.3)

    ax2.bar(centers, counts, width=1.0 / n_bins * 0.9, color="#888", alpha=0.85)
    ax2.set_xlabel("predicted probability")
    ax2.set_ylabel("count")
    ax2.set_xlim(0, 1)
    ax2.grid(True, ls="--", lw=0.4, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
