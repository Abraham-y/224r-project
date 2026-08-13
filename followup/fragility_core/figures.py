"""Paper figures. One function per figure, regenerable from parquet alone.

Every function:
  - reads only `results/fragility/metrics/*.parquet` (never activations, never a model)
  - writes `<name>.pdf` AND `<name>.csv` side by side. The CSV is the table view:
    it is the accessibility relief for the figure and it is what gets quoted in
    the text, so a figure can never drift from its numbers.
  - leaves GAPS for checkpoints with no measurement. Absent rows stay absent;
    nothing is interpolated (followup/CLAUDE.md).

Design constraints, applied deliberately:
  - One y-axis per panel. "Plot A alongside B" (e.g. causal_strength beside
    auroc_frozen) is stacked panels sharing an x-axis, never a twin axis — a
    dual-scale chart lets the reader infer a crossing that isn't there.
  - Series colors are assigned from a fixed, CVD-validated order and keyed to the
    METRIC, never to its position in the plot. Adding or dropping a series never
    repaints the others.
  - Every series also carries a distinct marker and linestyle, so identity
    survives greyscale printing and the two adjacent hues whose colorblind
    separation sits in the 6-8 floor band.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import metrics_io, paths  # noqa: E402

# Fixed categorical order. Validated (light surface #fcfcfb): lightness band,
# chroma floor, CVD adjacent separation, normal-vision floor, and >=3:1 contrast
# all pass. Do not reorder; do not extend by generating a hue.
PALETTE: tuple[str, ...] = ("#0072B2", "#D55E00", "#00806A", "#8C4F9F", "#8C6D1F")
NEUTRAL = "#4a4a48"
GRID = "#d8d8d4"

# metric -> (color, marker, linestyle, label). Identity is per-metric, so a
# metric keeps its appearance across every figure in the paper.
SERIES: dict[str, tuple[str, str, str, str]] = {
    "auroc_frozen": (PALETTE[0], "o", "-", "frozen probe (t=0 weights)"),
    "auroc_retrained_linear": (PALETTE[1], "s", "--", "retrained linear"),
    "auroc_retrained_mlp": (PALETTE[2], "^", ":", "retrained MLP (256)"),
    "evasion_gap": (PALETTE[3], "D", "-", "evasion gap"),
    "nonlinear_gain": (PALETTE[4], "v", "--", "MLP − linear"),
    "auroc_template": (PALETTE[4], "v", "--", "template score alone"),
    "auroc_frozen_cond_template": (PALETTE[3], "D", "-.", "frozen | template held fixed"),
    "corr_probe_template_spearman": (PALETTE[1], "s", "--", "corr(probe, template)"),
    "cka_vs_step0": (PALETTE[2], "^", "-", "CKA vs t=0"),
    "dir_drift_whitened": (PALETTE[0], "o", "-", "cos(v₀, vₜ) whitened"),
    "dir_drift_input": (PALETTE[1], "s", "--", "cos(v₀, vₜ) input space"),
    "sep": (PALETTE[2], "^", "-", "sep(t)"),
    "causal_strength": (PALETTE[3], "D", "-", "causal strength (\u0394acc @ \u03b1=1)"),
    "verifier_acc": (PALETTE[1], "s", "-", "true accuracy (verifier)"),
    "probe_score_mean": (PALETTE[0], "o", "-", "mean probe score"),
    "reward_mean": (PALETTE[4], "v", "--", "RL reward mean"),
    "kl_from_init": (PALETTE[2], "^", ":", "KL from init"),
    "patch_effect_mean": (PALETTE[1], "s", "--", "patch effect (patched \u2212 control)"),
}
_FALLBACK = (NEUTRAL, "x", "-", None)


def _style(metric: str) -> tuple[str, str, str, str]:
    color, marker, ls, label = SERIES.get(metric, _FALLBACK)
    return color, marker, ls, (label or metric)


def _init(figsize: tuple[float, float], nrows: int = 1, **kw):
    fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=True, **kw)
    axes = [axes] if nrows == 1 else list(axes)
    for ax in axes:
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(NEUTRAL)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=NEUTRAL, labelsize=9)
    return fig, axes


def _plot_series(ax, wide: pd.DataFrame, metrics: Sequence[str]) -> int:
    n = 0
    for m in metrics:
        if m not in wide.columns:
            continue
        color, marker, ls, label = _style(m)
        ax.plot(
            wide.index,
            wide[m],
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=2.0,
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
        )
        n += 1
    return n


def _finish(fig, axes, out: Path, wide: pd.DataFrame, *, xlabel: str) -> Path:
    axes[-1].set_xlabel(xlabel, color=NEUTRAL, fontsize=10)
    for ax in axes:
        handles, _labels = ax.get_legend_handles_labels()
        if len(handles) >= 2:
            ax.legend(frameon=False, fontsize=8.5, loc="best")
        elif len(handles) == 1:
            # A single series needs no legend box; the axis label names it.
            pass
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    wide.to_csv(out.with_suffix(".csv"))
    print(f"[figures] wrote {out} (+ .csv table view)")
    return out


def _wide(
    run_id: str,
    phase: str,
    metrics: Iterable[str],
    *,
    arm: str | None,
    layer: int | None = None,
) -> pd.DataFrame:
    """Wide (step x metric) frame for ONE arm.

    `arm` has no default and must be passed explicitly, including as None when
    pooling really is intended. `pivot_curve` aggregates with mean, so arm=None
    silently AVERAGES every arm that carries the metric — and phase-2 arms are
    `steering_off{0,-1}_{scale_ref}`, i.e. different hook positions and different
    alpha-scale conventions measuring different interventions. Averaging them
    produced one plausible curve out of several incompatible ones.
    """
    df = metrics_io.read_metrics(phase=phase, run_id=run_id)
    wide = metrics_io.pivot_curve(df, metrics, layer=layer, arm=arm)
    if not wide.empty:
        # Absent measurements become NaN on the shared step grid -> visible gaps.
        wide = wide.reindex(sorted(set(wide.index)))
    return wide


def _resolve_arm(
    run_id: str, phase: str, metric: str, prefix: str
) -> tuple[str | None, int]:
    """(the single arm carrying `metric`, how many arms carry it).

    Used only to keep `fig_phase2_causal(run_id, layer=...)` callable without the
    caller having to know the steering arm's spelling. Never picks one of several:
    silently choosing is the same class of error as silently averaging. The count
    is returned so the caller can tell "not measured" (0) from "measured under
    several incomparable conventions" (>1) — the first may still be a legitimate
    figure, the second must not be drawn at all.
    """
    df = metrics_io.latest(metrics_io.read_metrics(phase=phase, run_id=run_id))
    if df.empty:
        return None, 0
    arms = sorted({a for a in df[df["metric"] == metric]["arm"].unique()
                   if isinstance(a, str) and a.startswith(prefix)})
    if len(arms) == 1:
        return arms[0], 1
    if not arms:
        print(f"[figures] no {phase} arm starting {prefix!r} carries {metric!r} for {run_id}")
    else:
        print(f"[figures] {metric!r} spans several arms for {run_id}: {arms}. "
              "Pass arm= to choose one; they are not comparable and are not averaged.")
    return None, len(arms)


# ---------------------------------------------------------------------------
# Phase 0
# ---------------------------------------------------------------------------


def fig_phase0_collapse(
    run_id: str,
    *,
    layer: int,
    arm: str = "on_policy",
    out: Path | None = None,
) -> Path | None:
    """Reward / true accuracy / frozen-probe AUROC vs RL step (three panels).

    Three panels rather than three lines on one axis: they are measured on
    different scales, and superimposing them invites reading a crossing point
    that has no meaning.
    """
    wide = _wide(
        run_id,
        "phase0",
        ["probe_score_mean", "verifier_acc", "auroc_frozen", "reward_mean", "kl_from_init"],
        layer=layer,
        arm=arm,
    )
    if wide.empty:
        print(f"[figures] no phase0 metrics for {run_id}; skipping")
        return None
    out = out or paths.FIGURES / f"phase0_collapse__{run_id}__L{layer}.pdf"
    fig, axes = _init((6.4, 6.4), nrows=3)
    _plot_series(axes[0], wide, ["probe_score_mean", "reward_mean"])
    axes[0].set_ylabel("probe / reward", fontsize=10, color=NEUTRAL)
    _plot_series(axes[1], wide, ["verifier_acc"])
    axes[1].set_ylabel("true accuracy", fontsize=10, color=NEUTRAL)
    _plot_series(axes[2], wide, ["auroc_frozen"])
    axes[2].axhline(0.5, color=NEUTRAL, linewidth=0.9, linestyle=(0, (2, 3)))
    axes[2].set_ylabel("frozen-probe AUROC", fontsize=10, color=NEUTRAL)
    axes[0].set_title(f"{run_id} · layer {layer} · {arm}", fontsize=10.5, color=NEUTRAL)
    return _finish(fig, axes, out, wide, xlabel="RL step")


# ---------------------------------------------------------------------------
# Phase 1 — the headline
# ---------------------------------------------------------------------------


def fig_phase1_three_aurocs(
    run_id: str,
    *,
    layer: int,
    arm: str = "on_policy",
    out: Path | None = None,
) -> Path | None:
    """The H1-vs-H2 figure: frozen / retrained-linear / retrained-MLP vs step.

    Both retrained curves high while frozen collapses -> H1 (or H0).
    All three collapsing -> H2.
    """
    wide = _wide(
        run_id,
        "phase1",
        ["auroc_frozen", "auroc_retrained_linear", "auroc_retrained_mlp"],
        layer=layer,
        arm=arm,
    )
    if wide.empty:
        print(f"[figures] no phase1 metrics for {run_id} L{layer} {arm}; skipping")
        return None
    out = out or paths.FIGURES / f"phase1_three_aurocs__{run_id}__L{layer}__{arm}.pdf"
    fig, axes = _init((6.4, 3.8))
    _plot_series(axes[0], wide, ["auroc_frozen", "auroc_retrained_linear", "auroc_retrained_mlp"])
    axes[0].axhline(0.5, color=NEUTRAL, linewidth=0.9, linestyle=(0, (2, 3)))
    axes[0].set_ylabel("AUROC (held-out prompts)", fontsize=10, color=NEUTRAL)
    axes[0].set_title(
        f"Correctness decodability vs RL step · {run_id} · L{layer} · {arm}",
        fontsize=10.5,
        color=NEUTRAL,
    )
    return _finish(fig, axes, out, wide, xlabel="RL step")


def fig_phase1_per_layer(
    run_id: str,
    *,
    layers: Sequence[int],
    arm: str = "on_policy",
    metric: str = "auroc_retrained_linear",
    out: Path | None = None,
) -> Path | None:
    """Relocation check: does one layer's decodability rise as the probe layer falls?

    Small multiples over layers rather than one axis with a colour per layer,
    because layer is an ordered dimension and colour would imply identity.
    """
    frames = {}
    for L in layers:
        w = _wide(run_id, "phase1", [metric, "auroc_frozen"], layer=L, arm=arm)
        if not w.empty:
            frames[L] = w
    if not frames:
        print(f"[figures] no per-layer phase1 metrics for {run_id}; skipping")
        return None
    out = out or paths.FIGURES / f"phase1_per_layer__{run_id}__{metric}__{arm}.pdf"
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 3.2), sharey=True)
    axes = [axes] if n == 1 else list(axes)
    for ax, (L, w) in zip(axes, sorted(frames.items())):
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        _plot_series(ax, w, [metric, "auroc_frozen"])
        ax.axhline(0.5, color=NEUTRAL, linewidth=0.9, linestyle=(0, (2, 3)))
        ax.set_title(f"L{L}", fontsize=10, color=NEUTRAL)
        ax.set_xlabel("RL step", fontsize=9, color=NEUTRAL)
        ax.tick_params(colors=NEUTRAL, labelsize=8)
    axes[0].set_ylabel("AUROC", fontsize=10, color=NEUTRAL)
    axes[-1].legend(frameon=False, fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    pd.concat(frames, names=["layer"]).to_csv(out.with_suffix(".csv"))
    print(f"[figures] wrote {out} (+ .csv table view)")
    return out


def fig_phase1_geometry(
    run_id: str,
    *,
    layer: int,
    arm: str = "on_policy",
    out: Path | None = None,
) -> Path | None:
    """Direction drift, separation, and CKA vs step. Global drift vs targeted movement."""
    wide = _wide(
        run_id,
        "phase1",
        ["dir_drift_whitened", "dir_drift_input", "sep", "cka_vs_step0"],
        layer=layer,
        arm=arm,
    )
    if wide.empty:
        print(f"[figures] no geometry metrics for {run_id} L{layer}; skipping")
        return None
    out = out or paths.FIGURES / f"phase1_geometry__{run_id}__L{layer}__{arm}.pdf"
    fig, axes = _init((6.4, 5.4), nrows=2)
    _plot_series(axes[0], wide, ["dir_drift_whitened", "dir_drift_input", "cka_vs_step0"])
    axes[0].set_ylabel("cosine / CKA", fontsize=10, color=NEUTRAL)
    _plot_series(axes[1], wide, ["sep"])
    axes[1].axhline(0.0, color=NEUTRAL, linewidth=0.9, linestyle=(0, (2, 3)))
    axes[1].set_ylabel("sep(t) along v₀", fontsize=10, color=NEUTRAL)
    axes[0].set_title(f"{run_id} · L{layer} · {arm}", fontsize=10.5, color=NEUTRAL)
    return _finish(fig, axes, out, wide, xlabel="RL step")


def fig_h0_confound(
    run_id: str,
    *,
    layer: int,
    arm: str = "on_policy",
    out: Path | None = None,
) -> Path | None:
    """H0 panel: is the frozen probe's signal separable from the writing template?"""
    wide = _wide(
        run_id,
        "phase1",
        [
            "auroc_frozen",
            "auroc_template",
            "auroc_frozen_cond_template",
            "corr_probe_template_spearman",
        ],
        layer=layer,
        arm=arm,
    )
    if wide.empty:
        print(f"[figures] no H0 metrics for {run_id} L{layer}; skipping")
        return None
    out = out or paths.FIGURES / f"h0_confound__{run_id}__L{layer}__{arm}.pdf"
    fig, axes = _init((6.4, 5.4), nrows=2)
    _plot_series(axes[0], wide, ["auroc_frozen", "auroc_template", "auroc_frozen_cond_template"])
    axes[0].axhline(0.5, color=NEUTRAL, linewidth=0.9, linestyle=(0, (2, 3)))
    axes[0].set_ylabel("AUROC", fontsize=10, color=NEUTRAL)
    _plot_series(axes[1], wide, ["corr_probe_template_spearman"])
    axes[1].axhline(0.0, color=NEUTRAL, linewidth=0.9, linestyle=(0, (2, 3)))
    axes[1].set_ylabel("Spearman ρ", fontsize=10, color=NEUTRAL)
    axes[0].set_title(
        f"H0 structural-confound check · {run_id} · L{layer}", fontsize=10.5, color=NEUTRAL
    )
    return _finish(fig, axes, out, wide, xlabel="RL step")


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------


def fig_phase2_causal(
    run_id: str,
    *,
    layer: int,
    arm: str | None = None,
    patch_arm: str = "patching",
    auroc_arm: str = "on_policy",
    out: Path | None = None,
) -> Path | None:
    """causal_strength(t) beside auroc_frozen(t) — does causality decay first?

    Series names verified against `fragility_core.steering` rather than assumed:
    `causal_strength` is the "" key `causal_strength()` emits through `_emit`,
    and `patch_effect_mean` is the prefix `flip_rate()` emits for the signed
    (w2c - c2w)/2 contrast.

    The two live in DIFFERENT arms and are read separately. `run_steering.py`
    writes `steering_off{offset}_{scale_ref}`, one arm per hook position per
    alpha-scale convention; `run_patching.py` writes `patching`. `arm=None` means
    "resolve the steering arm if the store holds exactly one", not "average them"
    — pass `arm` when a run has several.

    Returns None unless BOTH panels have data, and also when `arm` was left to
    resolve itself and more than one steering arm matched. With the previous
    `causal.empty and auroc.empty` guard a run with no phase-2 rows at all still
    produced a two-panel figure whose top panel was blank, which reads as a
    measured null rather than as an absent measurement.
    """
    if arm is not None:
        steer_arm = arm
    else:
        steer_arm, n_arms = _resolve_arm(run_id, "phase2", "causal_strength", "steering")
        if n_arms > 1:
            # Measured, but under several incomparable conventions. Drawing the
            # patching series alone here would look like a complete figure whose
            # causal curve happens to be missing.
            return None
    causal = (
        _wide(run_id, "phase2", ["causal_strength"], layer=layer, arm=steer_arm)
        if steer_arm is not None
        else pd.DataFrame()
    )
    patch = _wide(run_id, "phase2", ["patch_effect_mean"], layer=layer, arm=patch_arm)
    auroc = _wide(run_id, "phase1", ["auroc_frozen"], layer=layer, arm=auroc_arm)
    if causal.empty and patch.empty:
        print(f"[figures] no phase2 causal/patching metrics for {run_id} L{layer}; skipping")
        return None
    if auroc.empty:
        print(f"[figures] no phase1 auroc_frozen for {run_id} L{layer} {auroc_arm}; "
              "the lower panel would be blank, skipping")
        return None
    parts = [d for d in (causal, patch, auroc) if not d.empty]
    wide = parts[0]
    for d in parts[1:]:
        wide = wide.join(d, how="outer")
    out = out or paths.FIGURES / f"phase2_causal__{run_id}__L{layer}.pdf"
    fig, axes = _init((6.4, 5.4), nrows=2)
    _plot_series(axes[0], wide, ["causal_strength", "patch_effect_mean"])
    axes[0].axhline(0.0, color=NEUTRAL, linewidth=0.9, linestyle=(0, (2, 3)))
    # Both series are signed accuracy differences, but against DIFFERENT
    # baselines: steering is probe minus matched-random at the same alpha,
    # patching is patched minus its own unpatched regeneration. One label cannot
    # name both without misdescribing one of them.
    axes[0].set_ylabel("Δ accuracy (signed intervention effect)", fontsize=10, color=NEUTRAL)
    _plot_series(axes[1], wide, ["auroc_frozen"])
    axes[1].axhline(0.5, color=NEUTRAL, linewidth=0.9, linestyle=(0, (2, 3)))
    axes[1].set_ylabel("frozen-probe AUROC", fontsize=10, color=NEUTRAL)
    axes[0].set_title(
        f"Causal status vs readability · {run_id} · L{layer} · {steer_arm or patch_arm}",
        fontsize=10.5, color=NEUTRAL,
    )
    return _finish(fig, axes, out, wide, xlabel="RL step")


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------


def fig_phase3_scatter(
    table: pd.DataFrame,
    *,
    feature: str,
    severity: str = "delta_true_accuracy",
    out: Path | None = None,
) -> Path | None:
    """Pre-hoc probe feature vs collapse severity, one point per probe in the zoo."""
    if table.empty or feature not in table.columns or severity not in table.columns:
        print(f"[figures] phase3 table missing {feature!r}/{severity!r}; skipping")
        return None
    out = out or paths.FIGURES / f"phase3_scatter__{feature}__vs__{severity}.pdf"
    fig, axes = _init((4.6, 4.0))
    ax = axes[0]
    ax.scatter(
        table[feature],
        table[severity],
        s=46,
        color=PALETTE[0],
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    # Direct labels: the probe zoo is small enough that every point is named.
    for _, row in table.iterrows():
        ax.annotate(
            str(row.get("probe_id", "")),
            (row[feature], row[severity]),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=7,
            color=NEUTRAL,
        )
    r = table[[feature, severity]].corr().iloc[0, 1]
    ax.set_xlabel(feature, fontsize=10, color=NEUTRAL)
    ax.set_ylabel(severity, fontsize=10, color=NEUTRAL)
    ax.set_title(f"Pearson r = {r:+.2f} (n={len(table)})", fontsize=10.5, color=NEUTRAL)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    table.to_csv(out.with_suffix(".csv"), index=False)
    print(f"[figures] wrote {out} (+ .csv table view)")
    return out


def fig_transfer_matrix(
    matrix: pd.DataFrame,
    *,
    out: Path | None = None,
    title: str = "Frozen-probe AUROC on post-RL activations",
) -> Path | None:
    """Transfer matrix heatmap: rows = policy trained against probe A, cols = probe B.

    Sequential single-hue ramp (magnitude), with every cell value printed, so the
    encoding never has to be read off the colour alone.
    """
    if matrix.empty:
        print("[figures] empty transfer matrix; skipping")
        return None
    out = out or paths.FIGURES / "transfer_matrix.pdf"
    fig, ax = plt.subplots(figsize=(1.1 * len(matrix.columns) + 2.2, 1.0 * len(matrix) + 2.0))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="Blues", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=8)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            v = matrix.to_numpy(dtype=float)[i, j]
            if pd.isna(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color=NEUTRAL)
            else:
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if v > 0.78 else NEUTRAL,
                )
    ax.set_title(title, fontsize=10.5, color=NEUTRAL)
    ax.set_xlabel("evaluated probe", fontsize=10, color=NEUTRAL)
    ax.set_ylabel("policy optimised against", fontsize=10, color=NEUTRAL)
    fig.colorbar(im, ax=ax, shrink=0.8, label="AUROC")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    matrix.to_csv(out.with_suffix(".csv"))
    print(f"[figures] wrote {out} (+ .csv table view)")
    return out
