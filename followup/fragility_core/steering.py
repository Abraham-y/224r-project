"""Residual-stream steering and activation patching.

Two interventions, both applied as a forward hook on `model.model.layers[L]`
(the same mechanism as extension/probe/causal_steering.py, so results are
directly comparable to the paper's published steering numbers):

  ADD      h[pos] <- h[pos] + alpha * scale * v      (activation addition)
  PATCH    h[pos] <- donor                           (activation patching)

Because the K/V cache for `pos` is written from the modified post-layer-L
residual, every later generated token attends to the modified representation.
The hook only fires on the prefill pass (seq_len > 1); autoregressive steps are
left alone.

alpha scale conventions
-----------------------
followup/CLAUDE.md specifies alpha in units of "activation std at that layer".
The original paper's published steering results use alpha x h_mean_norm (the
mean L2 norm of the hidden state, 21.86 at L16). Those differ by ~30x: the mean
per-dimension std at L16 is ~0.09 while the per-dimension RMS is ~0.73, because
the residual stream carries a large constant offset. Literally following the
CLAUDE.md numbers would inject ~3% of the paper's alpha=1.0 magnitude and
manufacture a null.

So the scale reference is an explicit config choice, and the INJECTED L2 NORM is
recorded for every condition. Whatever convention a caller picks, the numbers
remain comparable across conventions.

  h_mean_norm  mean ||h||_2 over the eval set. Reproduces the paper's protocol.
  proj_std     std of the projection of h onto the probe direction. "alpha
               standard deviations of this feature" — the most interpretable.
  act_std      mean per-dimension std. The literal CLAUDE.md reading; kept so
               the spec can still be honoured, with the caveat above.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np

from .geometry import projections, unit

SCALE_REFS = ("h_mean_norm", "proj_std", "act_std")


def scale_reference(X: np.ndarray, v: np.ndarray, ref: str) -> float:
    """The multiplier that turns a dimensionless alpha into an injected L2 norm."""
    X = np.asarray(X, dtype=np.float64)
    if ref == "h_mean_norm":
        return float(np.mean(np.linalg.norm(X, axis=1)))
    if ref == "proj_std":
        return float(np.std(projections(X, v)))
    if ref == "act_std":
        return float(np.mean(X.std(axis=0)))
    raise ValueError(f"unknown scale reference {ref!r}; expected one of {SCALE_REFS}")


def all_scale_references(X: np.ndarray, v: np.ndarray) -> dict[str, float]:
    """Every convention at once, so a run's alphas can be re-expressed later."""
    return {ref: scale_reference(X, v, ref) for ref in SCALE_REFS}


def random_matched_direction(d: int, *, seed: int) -> np.ndarray:
    """Unit-norm random control direction (the paper's specificity control)."""
    rng = np.random.RandomState(seed)
    r = rng.randn(d).astype(np.float32)
    return r / np.linalg.norm(r)


# ---------------------------------------------------------------------------
# hook plumbing
# ---------------------------------------------------------------------------


@dataclass
class InterventionState:
    """Mutable state read by the hook on each forward pass.

    mode      "off" | "add" | "patch"
    position  token index within the prefill sequence to modify
    vector    the added vector (mode="add") or the replacement (mode="patch")
    applied   set True by the hook; assert on it to catch silent no-ops
    """

    mode: str = "off"
    position: int | None = None
    vector: Any = None
    applied: bool = False
    n_calls: int = field(default=0)
    # L2 norm of the change actually written into the residual stream. `applied`
    # only proves the hook ran; this proves it changed something, and by how
    # much. Comparing it against the intended alpha*scale catches a zero vector,
    # a dtype underflow, a wrong position, and a truncated prefix at once — the
    # failure modes that all present as a clean null result.
    delta_norm: float = field(default=0.0)

    def reset(self) -> None:
        self.applied = False
        self.n_calls = 0
        self.delta_norm = 0.0


def make_hook(state: InterventionState) -> Callable:
    """Forward hook implementing ADD / PATCH at a single prefill position."""

    def hook(_module, _inputs, outputs):
        if state.mode == "off" or state.vector is None or state.position is None:
            return outputs
        is_tuple = isinstance(outputs, tuple)
        hs = outputs[0] if is_tuple else outputs
        # Only touch the prefill pass. Decode steps have seq_len == 1.
        if hs.dim() != 3 or hs.shape[1] == 1:
            return outputs
        pos = state.position
        if pos < 0:
            pos = hs.shape[1] + pos
        if pos >= hs.shape[1]:
            return outputs
        state.n_calls += 1
        vec = state.vector.to(hs.dtype).to(hs.device)
        new_hs = hs.clone()
        if state.mode == "add":
            new_hs[:, pos] = new_hs[:, pos] + vec
        elif state.mode == "patch":
            new_hs[:, pos] = vec
        else:
            raise ValueError(f"unknown intervention mode {state.mode!r}")
        state.delta_norm = float((new_hs[:, pos] - hs[:, pos]).float().norm())
        state.applied = True
        return (new_hs,) + outputs[1:] if is_tuple else new_hs

    return hook


@contextmanager
def intervention(model: Any, layer: int) -> Iterator[InterventionState]:
    """Attach the hook for the duration of the block; always detaches.

    Usage:
        with intervention(model, 16) as st:
            st.mode, st.position, st.vector = "add", -1, torch_vec
            out = model.generate(...)
            assert st.applied
    """
    state = InterventionState()
    handle = model.model.layers[layer].register_forward_hook(make_hook(state))
    try:
        yield state
    finally:
        handle.remove()


def to_torch(v: np.ndarray, *, device: str = "cuda", dtype: Any = None) -> Any:
    import torch

    dtype = torch.bfloat16 if dtype is None else dtype
    return torch.from_numpy(np.ascontiguousarray(np.asarray(v, dtype=np.float32))).to(
        device=device, dtype=dtype
    )


# ---------------------------------------------------------------------------
# prompt-clustered bootstrap
# ---------------------------------------------------------------------------
#
# phase2_predictions.md 5 pre-registers "Bootstrap CIs over prompts (10k
# resamples)". These are the implementation of that line.
#
# Clustering matters: `n_prompts=100` x `n_rollouts_per_prompt=2` gives ~200
# rows, but the two rollouts of a prompt share its difficulty, so treating rows
# as independent understates the SE. The unit of resampling is the PROMPT.

BOOT_N = 10_000
BOOT_SEED = 0
BOOT_CI = 0.95
# The pre-registered comparison dose. The paper's published post-Goodhart result
# (+0.083 probe-minus-random on accuracy) and its null band are both quoted at
# alpha = 1.0 x h_mean_norm; `phase2_default.yaml:steering.ref_alpha` carries it
# so it is a config choice, not a constant buried in code.
CAUSAL_REF_ALPHA = 1.0


def _suff_stats(clusters: Any, vals: Any, index: list) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster (sum, count) aligned to `index`. Non-finite values dropped."""
    pos = {k: i for i, k in enumerate(index)}
    c = np.asarray([pos[k] for k in clusters], dtype=np.int64)
    v = np.asarray(vals, dtype=np.float64)
    keep = np.isfinite(v)
    c, v = c[keep], v[keep]
    s = np.zeros(len(index), dtype=np.float64)
    n = np.zeros(len(index), dtype=np.float64)
    np.add.at(s, c, v)
    np.add.at(n, c, 1.0)
    return s, n


def _boot_ratio_draws(
    stats: list[tuple[np.ndarray, np.ndarray]],
    *,
    n_boot: int,
    seed: int,
    chunk: int = 2000,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Resample clusters with replacement; return each arm's bootstrap means.

    All arms are gathered from the SAME cluster draw, so a prompt appears on
    every side of a contrast in the same resample. That is what makes the
    contrast paired at the prompt level.

    Returns (per-arm mean arrays, validity mask). A draw is invalid for an arm
    if the drawn clusters contain zero rows of it; such draws are dropped from
    every arm together.
    """
    n_clusters = len(stats[0][0])
    rng = np.random.default_rng(seed)
    means = [np.empty(n_boot, dtype=np.float64) for _ in stats]
    valid = np.ones(n_boot, dtype=bool)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, n_clusters, size=(b, n_clusters))
        sl = slice(done, done + b)
        for j, (s, n) in enumerate(stats):
            num = s[idx].sum(axis=1)
            den = n[idx].sum(axis=1)
            ok = den > 0
            valid[sl] &= ok
            means[j][sl] = np.where(ok, num / np.where(ok, den, 1.0), np.nan)
        done += b
    return means, valid


def _summarise_draws(est: float, draws: np.ndarray, *, n_boot: int, ci_level: float) -> dict:
    """Percentile CI, bootstrap SE, and a two-sided percentile p-value.

    The p-value is `2 * min(P(draw <= 0), P(draw >= 0))`, i.e. the CI-inversion
    p, floored at 1/n_boot. A reported 1e-4 with n_boot=10000 means "no resample
    landed on the other side of zero", not an exact value.
    """
    lo_q = (1.0 - ci_level) / 2.0
    d = draws[np.isfinite(draws)]
    if d.size == 0:
        return {}
    lo, hi = np.quantile(d, [lo_q, 1.0 - lo_q])
    p = 2.0 * min(float(np.mean(d <= 0.0)), float(np.mean(d >= 0.0)))
    p = min(1.0, max(p, 1.0 / max(n_boot, 1)))
    return {
        "": float(est),
        "_ci_lo": float(lo),
        "_ci_hi": float(hi),
        "_se": float(np.std(d, ddof=1)) if d.size > 1 else float("nan"),
        "_p": float(p),
        "_significant": float(lo > 0.0 or hi < 0.0),
        "_n_boot_valid": float(d.size),
    }


def cluster_bootstrap_contrast(
    clusters_a: Any,
    vals_a: Any,
    clusters_b: Any,
    vals_b: Any,
    *,
    n_boot: int = BOOT_N,
    seed: int = BOOT_SEED,
    ci_level: float = BOOT_CI,
) -> dict[str, float]:
    """mean(vals_a) - mean(vals_b) with a prompt-clustered bootstrap CI.

    The point estimate is the ratio of sums (sum_a/n_a - sum_b/n_b), which is
    exactly the unweighted difference of row means — resampling changes the CI,
    never the point estimate.

    Keys: "" (the estimate), _ci_lo, _ci_hi, _se, _p, _significant,
    _n_clusters, _n_a, _n_b, _n_boot_valid. Empty dict if either arm is empty.
    """
    index = sorted(set(map(str, clusters_a)) | set(map(str, clusters_b)))
    sa, na = _suff_stats([str(c) for c in clusters_a], vals_a, index)
    sb, nb = _suff_stats([str(c) for c in clusters_b], vals_b, index)
    if na.sum() == 0 or nb.sum() == 0:
        return {}
    est = sa.sum() / na.sum() - sb.sum() / nb.sum()
    (ma, mb), valid = _boot_ratio_draws([(sa, na), (sb, nb)], n_boot=n_boot, seed=seed)
    out = _summarise_draws(est, (ma - mb)[valid], n_boot=n_boot, ci_level=ci_level)
    if out:
        out["_n_clusters"] = float(len(index))
        out["_n_a"] = float(na.sum())
        out["_n_b"] = float(nb.sum())
    return out


def cluster_bootstrap_mean(
    clusters: Any,
    vals: Any,
    *,
    n_boot: int = BOOT_N,
    seed: int = BOOT_SEED,
    ci_level: float = BOOT_CI,
) -> dict[str, float]:
    """mean(vals) with a prompt-clustered bootstrap CI. Same key convention."""
    index = sorted(set(map(str, clusters)))
    s, n = _suff_stats([str(c) for c in clusters], vals, index)
    if n.sum() == 0:
        return {}
    est = s.sum() / n.sum()
    (m,), valid = _boot_ratio_draws([(s, n)], n_boot=n_boot, seed=seed)
    out = _summarise_draws(est, m[valid], n_boot=n_boot, ci_level=ci_level)
    if out:
        out["_n_clusters"] = float(len(index))
        out["_n_rows"] = float(n.sum())
    return out


def _emit(out: dict[str, float], prefix: str, boot: dict[str, float]) -> None:
    """Write a bootstrap result under `prefix` ("" key becomes the prefix itself)."""
    for suffix, value in boot.items():
        out[prefix + suffix] = value


# ---------------------------------------------------------------------------
# causal_strength aggregation (reads steering result rows; no GPU needed)
# ---------------------------------------------------------------------------


def causal_strength(
    rows: list[dict[str, Any]],
    *,
    accuracy_key: str = "new_score_correct",
    probe_key: str | None = "new_probe_score",
    ref_alpha: float = CAUSAL_REF_ALPHA,
    cluster_key: str = "prompt_idx",
    n_boot: int = BOOT_N,
    boot_seed: int = BOOT_SEED,
    ci_level: float = BOOT_CI,
) -> dict[str, float]:
    """causal_strength(t) from one checkpoint's steering rows.

    Rows are expected to carry: alpha, direction ("probe" | "rand" | "zero"),
    `cluster_key`, and the outcome keys. The reported effect is always PROBE
    MINUS MATCHED RANDOM at the same alpha, which is the paper's protocol — a
    bare probe-direction delta conflates the intervention's effect with the
    generic damage done by perturbing the residual stream at all.

    HEADLINE `causal_strength` = the SIGNED probe-minus-random accuracy delta at
    the single pre-registered dose `ref_alpha` (default +1.0), with a
    prompt-clustered percentile bootstrap CI and p-value under
    `causal_strength_ci_lo/_ci_hi/_se/_p/_significant`.

    Why a single pre-registered dose rather than a summary over the grid — both
    alternatives were measured on the paper's published steering runs
    (`experiments/fragility/phase2_causal_status/validate_estimator.py`
    re-derives every number below from those two files):

      max|delta| over alphas   unsigned, and its noise floor grows with the
                               number of alphas searched. It reports 0.0825 on
                               the real effect and 0.0155 on the inert control,
                               but has no way to call the second a null. Under a
                               simulated true null matched to the published
                               design (100 prompts, ~2 rollouts, base rate 0.20)
                               on the config's 6-alpha `both_signs` grid, its
                               95th percentile is 0.0825 (N=2000) — the effect
                               it exists to detect is 0.0825.
      OLS slope on alpha       returns -0.0722 on the +0.0825 run, because the
                               dose-response is an inverted U (+0.041, +0.083,
                               -0.052 at alpha 0.5/1/2) and least squares gives
                               the alpha=2 damage point the most leverage. On
                               the assertion control it returns +0.0110 with a
                               1-dof residual SE of 0.0038 (t=+2.89), i.e. it
                               calls a published null significant. On a
                               sign-symmetric alpha grid it sees only the odd
                               component of the dose-response and is exactly 0
                               for any even one.

    Both are kept below under `diag_` names, with the size of the alpha grid and
    the fit's dof recorded alongside, and neither is the headline.

    Calibration of the headline, MEASURED by Monte Carlo on that same simulated
    design (empirical per-prompt base rates and per-prompt row counts taken from
    the published run's random arm), 4000 replicates x 2000 bootstrap resamples:

      true null       significance rate 0.051 (95% MC interval +-0.007, nominal
                      0.05); CI coverage of 0 = 0.949
      true +0.083     power 0.641 (+-0.015); CI coverage of 0.083 = 0.950

    Power 0.64 is the honest limit of a 100-prompt run at this effect size: a
    single non-significant checkpoint is not evidence of no effect, and the
    trajectory has to be read from the CIs, not from the significance flags.
    `validate_estimator.py` re-runs a reduced version of this as part of the
    gate (300 replicates, so its own MC error is about +-0.025).

    Raises KeyError if `cluster_key` is absent: an unclustered CI on rows that
    share a prompt is anti-conservative, and silently producing one is exactly
    the failure this function exists to avoid.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    if cluster_key not in df.columns:
        raise KeyError(
            f"steering rows carry no {cluster_key!r}; the pre-registered CI is "
            "clustered on prompts and cannot be computed from rows that do not "
            "say which prompt they came from"
        )
    out: dict[str, float] = {}

    base = df[df["direction"] == "zero"]
    if not base.empty:
        out["steer_baseline_acc"] = float(base[accuracy_key].mean())

    per_alpha: list[tuple[float, float]] = []
    for alpha, grp in df[df["direction"] != "zero"].groupby("alpha"):
        pr = grp[grp["direction"] == "probe"]
        rd = grp[grp["direction"] == "rand"]
        if pr.empty or rd.empty:
            continue
        d_acc = float(pr[accuracy_key].mean() - rd[accuracy_key].mean())
        if not np.isfinite(d_acc):
            continue
        out[f"steer_delta_acc_alpha{alpha:g}"] = d_acc
        out[f"steer_acc_probe_alpha{alpha:g}"] = float(pr[accuracy_key].mean())
        out[f"steer_acc_rand_alpha{alpha:g}"] = float(rd[accuracy_key].mean())
        out[f"steer_n_probe_alpha{alpha:g}"] = float(pr[accuracy_key].notna().sum())
        per_alpha.append((float(alpha), d_acc))
        if probe_key and probe_key in grp.columns:
            dp = float(pr[probe_key].mean() - rd[probe_key].mean())
            if np.isfinite(dp):
                out[f"steer_delta_probe_alpha{alpha:g}"] = dp

    if not per_alpha:
        return out

    alphas = np.array([a for a, _ in per_alpha], dtype=np.float64)
    deltas = np.array([d for _, d in per_alpha], dtype=np.float64)
    out["n_alphas_used"] = float(len(alphas))

    # ---- HEADLINE: signed contrast at the pre-registered dose ----------------
    # `causal_strength_negalpha` is the same contrast at -ref_alpha (the config
    # sweeps `both_signs`), reported separately rather than pooled: pooling the
    # two signs is what makes a summary blind to an even dose-response.
    # `causal_strength_assertion` is the pre-registered specificity control —
    # the correctness-correlated but never-optimised direction minus the same
    # matched random — reported through the identical estimator so it is
    # directly comparable to the headline instead of eyeballed.
    for target, arm_dir, prefix in (
        (float(ref_alpha), "probe", "causal_strength"),
        (-float(ref_alpha), "probe", "causal_strength_negalpha"),
        (float(ref_alpha), "assertion", "causal_strength_assertion"),
    ):
        sub = df[np.isclose(df["alpha"].to_numpy(dtype=float), target)]
        pr = sub[sub["direction"] == arm_dir]
        rd = sub[sub["direction"] == "rand"]
        if pr.empty or rd.empty:
            continue
        boot = cluster_bootstrap_contrast(
            pr[cluster_key].tolist(), pr[accuracy_key].to_numpy(dtype=float),
            rd[cluster_key].tolist(), rd[accuracy_key].to_numpy(dtype=float),
            n_boot=n_boot, seed=boot_seed, ci_level=ci_level,
        )
        _emit(out, prefix, boot)
        out[f"{prefix}_alpha"] = target
    if "causal_strength" not in out:
        # Never silently substitute a different dose: that is per-run tuning of
        # a frozen analysis choice. Mark it absent instead.
        out["causal_strength_ref_alpha_missing"] = float(ref_alpha)

    # ---- DIAGNOSTICS (not the headline; multiplicity recorded) ---------------
    i_max = int(np.argmax(np.abs(deltas)))
    out["diag_maxabs_delta"] = float(abs(deltas[i_max]))
    out["diag_maxabs_delta_signed"] = float(deltas[i_max])
    out["diag_maxabs_at_alpha"] = float(alphas[i_max])
    out["diag_maxabs_n_alphas_searched"] = float(len(alphas))
    if len(alphas) >= 2 and np.std(alphas) > 0:
        slope, intercept = np.polyfit(alphas, deltas, 1)
        resid = deltas - (slope * alphas + intercept)
        dof = max(len(alphas) - 2, 1)
        se = float(
            np.sqrt(
                np.sum(resid**2) / dof / max(np.sum((alphas - alphas.mean()) ** 2), 1e-12)
            )
        )
        out["diag_ols_slope"] = float(slope)
        out["diag_ols_slope_intercept"] = float(intercept)
        # SE from the residuals of the fit, i.e. from `dof` degrees of freedom.
        # On the published 3-alpha grid dof=1, which is why this SE can be
        # arbitrarily small and the slope arbitrarily "significant". Always read
        # it next to `diag_ols_slope_dof`.
        out["diag_ols_slope_se"] = se
        out["diag_ols_slope_dof"] = float(dof)
        out["diag_ols_slope_n_alphas"] = float(len(alphas))
        # On a sign-symmetric grid the OLS slope is a pure odd-component
        # estimator and is identically 0 for any even dose-response, so it
        # cannot detect a symmetric effect at all. Flag it rather than let a
        # reader take the 0 at face value.
        sym = np.allclose(np.sort(alphas), np.sort(-alphas))
        out["diag_ols_slope_grid_sign_symmetric"] = float(sym)
    return out


def flip_rate(
    rows: list[dict[str, Any]],
    *,
    cluster_key: str = "prompt_idx",
    n_boot: int = BOOT_N,
    boot_seed: int = BOOT_SEED,
    ci_level: float = BOOT_CI,
) -> dict[str, float]:
    """Activation-patching effect, measured against an UNPATCHED regeneration.

    The effect is always `patched - control` on matched prefixes, never
    `patched vs the recipient's original label`. Regenerating from `</think>` at
    temperature 0.6 re-derives the answer from scratch, so the outcome differs
    from the original rollout maybe 30-50% of the time with no intervention at
    all. Differencing against the original label therefore measures resampling
    noise plus patch effect, with no way to separate them — and since that noise
    floor tracks the checkpoint's own accuracy and entropy, both of which
    collapse here, it would trace a beautiful declining curve that means nothing.

    Reported the same way as `causal_strength`: a signed point estimate plus a
    prompt-clustered percentile bootstrap CI, SE, p and significance flag
    (`patch_effect_{c2w,w2c,mean}` + `_ci_lo/_ci_hi/_se/_p/_significant`).
    `patch_effect_mean` is bootstrapped on the SAME cluster draw as its two
    components, not by combining their separate intervals.

    Requires `control_correct` and `cluster_key` on every row (run_patching
    generates both).
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    if "control_correct" not in df.columns:
        raise KeyError(
            "patching rows carry no `control_correct`; without an unpatched "
            "regeneration from the same prefix the flip rate is uninterpretable"
        )
    if cluster_key not in df.columns:
        raise KeyError(
            f"patching rows carry no {cluster_key!r}; the pre-registered CI is "
            "clustered on prompts and cannot be computed without it"
        )
    out: dict[str, float] = {}
    arm_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    index = sorted(set(str(c) for c in df[cluster_key]))
    for name, sub in (
        ("c2w", df[df["donor_label"] == 0]),
        ("w2c", df[df["donor_label"] == 1]),
    ):
        if sub.empty:
            continue
        patched = sub["new_correct"].to_numpy(dtype=float)
        control = sub["control_correct"].to_numpy(dtype=float)
        # PAIRED at the row level (same prefix, same sampling seed) and
        # CLUSTERED at the prompt level, so the resampling unit is the prompt.
        paired = patched - control
        clusters = [str(c) for c in sub[cluster_key]]
        _emit(
            out,
            f"patch_effect_{name}",
            cluster_bootstrap_mean(
                clusters, paired, n_boot=n_boot, seed=boot_seed, ci_level=ci_level
            ),
        )
        arm_stats[name] = _suff_stats(clusters, paired, index)
        out[f"patch_acc_patched_{name}"] = float(np.mean(patched))
        out[f"patch_acc_control_{name}"] = float(np.mean(control))
        # How often the two regenerations differ with NO patch: the noise floor.
        out[f"patch_baseline_churn_{name}"] = float(
            np.mean(control != sub["orig_correct"].to_numpy(dtype=float))
        )
        out[f"patch_n_{name}"] = float(len(sub))
    if "c2w" in arm_stats and "w2c" in arm_stats:
        # Signed so the two directions do not cancel: a real transplant should
        # push w2c up and c2w down.
        est = (out["patch_effect_w2c"] - out["patch_effect_c2w"]) / 2.0
        (m_c2w, m_w2c), valid = _boot_ratio_draws(
            [arm_stats["c2w"], arm_stats["w2c"]], n_boot=n_boot, seed=boot_seed
        )
        _emit(
            out,
            "patch_effect_mean",
            _summarise_draws(
                est, ((m_w2c - m_c2w) / 2.0)[valid], n_boot=n_boot, ci_level=ci_level
            ),
        )
    return out


def alpha_norms(alphas: list[float], scale: float) -> dict[float, float]:
    """Injected L2 norm per alpha, for the provenance record."""
    return {float(a): float(a) * float(scale) for a in alphas}


def probe_unit_direction(w: np.ndarray) -> np.ndarray:
    return unit(w).astype(np.float32)
