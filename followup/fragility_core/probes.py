"""Probe evaluation: frozen, retrained-linear, retrained-MLP.

The three quantities that decide H1 vs H2 (followup/CLAUDE.md):

  auroc_frozen(t)            original probe, weights fixed at t=0, applied to
                             checkpoint-t activations.
  auroc_retrained_linear(t)  fresh logistic probe fit on checkpoint-t train rows.
  auroc_retrained_mlp(t)     fresh 1-hidden-layer (256, ReLU) MLP, same rows.
  evasion_gap(t)             retrained_linear - frozen.

Fixed globally, never tuned per checkpoint:
  - the train/val split (by prompt_idx; see labels.prompt_split)
  - the linear probe's hyperparameters (StandardScaler + LogisticRegression(C),
    matching the original paper's probe recipe so retrained and frozen probes
    are the same model class)
  - the MLP's architecture and early-stopping rule (early stopping uses an
    internal slice of the TRAIN rows, so the reported val AUROC stays clean)

Balanced AUROC is reported alongside plain AUROC because the positive rate moves
a lot across checkpoints (post-Goodhart accuracy collapses), and the paper's
existing numbers use the balanced convention.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LINEAR_C = 0.1
LINEAR_MAX_ITER = 2000
MLP_HIDDEN = (256,)
MLP_MAX_ITER = 400


# ---------------------------------------------------------------------------
# loading the frozen probe
# ---------------------------------------------------------------------------


class DirectionProbe:
    """A bare direction used as a probe: score = sigmoid(x . w + b).

    Uncalibrated, but AUROC only depends on the ranking, so this is a valid
    frozen probe for every metric in this project.

    `b` defaults to None, meaning NO INTERCEPT WAS EVER FIT: the score is x . w
    and the z = 0 contour falls wherever the residual stream happens to sit, not
    on a decision boundary. `intercept_fitted` records that so
    `standardized_margin` can refuse to measure a distance from a boundary that
    does not exist (see its docstring for the artefact this caused). Pass `b`
    explicitly to declare a fitted intercept.
    """

    def __init__(self, w: np.ndarray, b: float | None = None, name: str = "direction"):
        self.w = np.asarray(w, dtype=np.float32).ravel()
        self.intercept_fitted = b is not None
        self.b = 0.0 if b is None else float(b)
        self.name = name

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32) @ self.w + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        return np.stack([1.0 - p, p], axis=1)


def load_frozen_probe(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load the original paper's probe. Returns (probe, provenance).

    Accepts:
      .pkl  a pickled sklearn Pipeline (StandardScaler + LogisticRegression)
      .npz  a saved direction (`v_input_raw` preferred, else `v_unit`)

    sklearn emits InconsistentVersionWarning when unpickling across versions.
    We suppress the warning but RECORD the fact in the provenance dict, so a
    version skew is auditable rather than invisible.
    """
    p = Path(path)
    prov: dict[str, Any] = {"probe_path": str(p), "kind": p.suffix}
    if p.suffix == ".npz":
        with np.load(p) as d:
            w = d["v_input_raw"] if "v_input_raw" in d.files else d["v_unit"]
            prov["npz_fields"] = list(d.files)
            if "probe_auroc" in d.files:
                prov["probe_auroc_at_train"] = float(d["probe_auroc"])
        return DirectionProbe(w, name=p.stem), prov

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with open(p, "rb") as f:
            probe = pickle.load(f)
        prov["unpickle_warnings"] = sorted({str(w.message).split("\n")[0] for w in caught})
    try:
        import sklearn

        prov["sklearn_version_now"] = sklearn.__version__
    except ImportError:
        pass
    return probe, prov


def probe_scores(probe: Any, X: np.ndarray) -> np.ndarray:
    """P(correct) (or any monotone score) for each row of X."""
    if hasattr(probe, "predict_proba"):
        return np.asarray(probe.predict_proba(X))[:, 1]
    if hasattr(probe, "decision_function"):
        return np.asarray(probe.decision_function(X)).ravel()
    raise TypeError(f"{type(probe).__name__} exposes neither predict_proba nor decision_function")


def decision_scores(probe: Any, X: np.ndarray) -> np.ndarray:
    """Signed distance-like score: the decision function, or the score's logit."""
    if hasattr(probe, "decision_function"):
        return np.asarray(probe.decision_function(X), dtype=np.float64).ravel()
    p = np.clip(probe_scores(probe, X), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def has_fitted_boundary(probe: Any) -> bool:
    """True iff the probe's z = 0 contour is a FITTED decision boundary.

    sklearn writes `intercept_` (linear models) or `intercepts_` (MLP) during
    `fit`, and only during fit, so their z = 0 is where the fitted model switches
    class. A `DirectionProbe` built from a bare direction has no fitted intercept
    (see its docstring); `intercept_fitted` is read with `getattr` so probes
    unpickled from before that attribute existed are treated as unfitted, which
    is the safe reading.
    """
    if isinstance(probe, DirectionProbe):
        return bool(getattr(probe, "intercept_fitted", False))
    est = probe.steps[-1][1] if isinstance(probe, Pipeline) else probe
    return hasattr(est, "intercept_") or hasattr(est, "intercepts_")


def standardized_margin(probe: Any, X: np.ndarray, *, center: float | None = None) -> np.ndarray:
    """|distance from the DECISION BOUNDARY|, in units of the score's spread.

    For a FITTED probe the boundary is z = 0 — that is what fitting an intercept
    decides — so the statistic is |z| / std(z). An earlier version subtracted the
    median, which made the statistic blind to the probe's intercept — a probe
    with b=0 and one with b=1e6 (which classifies everything positive) returned
    identical margins — and pinned it near the value implied by any unimodal
    distribution, so it encoded the score distribution's kurtosis rather than
    margin. Across a zoo spanning AUROC 0.33-0.92 it varied only over 0.68-0.89,
    i.e. it was close to a constant regressor. Dividing by the std alone already
    delivers the scale-freeness needed to compare probes.

    A probe with NO fitted intercept gets NaN, because z = 0 is then not a
    boundary. `DirectionProbe(w)` — the zoo's `mean_diff` entry, and any probe
    loaded from a bare-direction .npz — scores x . w with b = 0 by construction,
    and the residual stream's large mean offset puts that contour in an arbitrary
    place. Measured on `results/fragility/probe_zoo_runA` (recomputed from the
    pickled models and the runA t=0 cache, val rows): that artefact gave
    `L16_meandiff` margin_median 1.105, the LARGEST in a zoo whose 14 fitted
    probes span 0.522-0.718, while its AUROC 0.631 was the second lowest of the
    15 (only the deliberately crippled `L16_resid_template`, 0.605, is lower).
    The feature was ranking one of the zoo's weakest probes as its most confident
    one, on the strength of an unfitted constant.

    Pass `center` to give such a probe a boundary explicitly (e.g. the median
    decision score on its TRAINING rows); the statistic is then
    |z - center| / std(z). Nothing is ever inferred from `X`: taking the centre
    from the rows being measured would re-create the median-subtraction defect
    described above.

    Returns NaN (not zeros) for a degenerate probe: undefined, not "maximally
    fragile".
    """
    z = decision_scores(probe, X)
    if center is None:
        if not has_fitted_boundary(probe):
            warnings.warn(
                f"{getattr(probe, 'name', type(probe).__name__)} has no fitted intercept, "
                "so z = 0 is not a decision boundary and its standardized margin is "
                "undefined; returning NaN. Pass center= (e.g. the median training "
                "score) to define a boundary explicitly.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.full_like(z, np.nan)
        c = 0.0
    else:
        c = float(center)
    sd = np.std(z)
    if sd == 0:
        return np.full_like(z, np.nan)
    return np.abs((z - c) / sd)


def probe_direction(probe: Any) -> np.ndarray | None:
    """The probe's direction in INPUT (raw activation) space, unnormalised.

    For Pipeline(StandardScaler, LogisticRegression) the score is
    w . (x - mu) / sigma + b, so the input-space direction is w / sigma. This is
    the same convention as extension/probe/save_probe_direction_temp1.py, so
    cosines here are comparable to the paper's published cosines.

    Returns None for probes with no single linear direction (e.g. the MLP).
    """
    if isinstance(probe, DirectionProbe):
        return probe.w.copy()
    if isinstance(probe, Pipeline):
        final = probe.steps[-1][1]
        if not hasattr(final, "coef_"):
            return None  # e.g. an MLP: no single direction exists
        w = np.asarray(final.coef_, dtype=np.float64).ravel()
        # Walk the preprocessing steps backwards, pulling the direction back into
        # input space. Each step's inverse acts on the direction, not the point,
        # so translations (centering) drop out and only the linear map matters.
        if np.asarray(final.coef_).shape[0] != 1:
            return None  # multi-class: no single direction; ravel() would fabricate one
        for _name, step in reversed(probe.steps[:-1]):
            comps = getattr(step, "components_", None)
            if comps is not None:  # PCA / truncated SVD: y = (x - mu) @ V^T
                if getattr(step, "whiten", False):
                    # whitened PCA divides by sqrt(explained_variance_); omitting
                    # this yields a direction that still correlates at ~0.9997
                    # with the truth, so the error would never be noticed.
                    w = w / np.sqrt(np.asarray(step.explained_variance_, dtype=np.float64))
                w = w @ np.asarray(comps, dtype=np.float64)
                continue
            if hasattr(step, "scale_"):  # StandardScaler
                scale = step.scale_
                if scale is None:
                    continue  # with_std=False -> identity, not "unrecognised"
                w = w / np.asarray(scale, dtype=np.float64)
                continue
            # An unrecognised transform would silently invalidate the direction.
            return None
        return w
    if hasattr(probe, "coef_"):
        return np.asarray(probe.coef_, dtype=np.float64).ravel()
    return None


def _named(pipe: Pipeline, candidates: tuple[str, ...]) -> Any:
    for key in candidates:
        if key in pipe.named_steps:
            return pipe.named_steps[key]
    return None


# ---------------------------------------------------------------------------
# AUROC helpers
# ---------------------------------------------------------------------------


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """AUROC, or NaN when the labels are single-class (measured-but-undefined)."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score, dtype=np.float64)))


def auroc_balanced(
    y: np.ndarray, score: np.ndarray, *, seed: int = 0, n_draws: int = 25
) -> float:
    """AUROC averaged over `n_draws` class-balanced subsamples.

    Kept because it is the original paper's reporting convention, but note what
    it does and does not do. AUROC is the Mann-Whitney statistic P(s+ > s-) and
    is already prevalence-invariant in expectation, so balancing corrects NO
    bias — it only discards data and adds draw-to-draw variance. That variance
    is worst exactly where this project's story lives: at a 4.6% positive rate a
    single draw carries +-0.022 of pure subsampling noise, enough to manufacture
    visible wiggle in an AUROC-vs-step curve.

    Averaging over draws removes most of that. Plain `auroc` remains the
    preferred headline.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n = min(len(pos), len(neg))
    if n == 0:
        return float("nan")
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(max(1, n_draws)):
        idx = np.concatenate(
            [rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)]
        )
        vals.append(auroc(y[idx], score[idx]))
    return float(np.nanmean(vals))


# ---------------------------------------------------------------------------
# frozen / retrained evaluation
# ---------------------------------------------------------------------------


def eval_frozen(
    probe: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, float]:
    """Frozen-probe metrics on the given rows (normally the val split)."""
    s = probe_scores(probe, X)
    y = np.asarray(y).astype(int)
    out = {
        "auroc_frozen": auroc(y, s),
        "auroc_frozen_balanced": auroc_balanced(y, s, seed=seed),
        "probe_score_mean": float(np.mean(s)),
        "probe_score_std": float(np.std(s)),
        "probe_score_mean_correct": float(np.mean(s[y == 1])) if (y == 1).any() else float("nan"),
        "probe_score_mean_wrong": float(np.mean(s[y == 0])) if (y == 0).any() else float("nan"),
        "probe_score_frac_above_095": float(np.mean(s > 0.95)),
        "pos_rate": float(np.mean(y)),
    }
    return out


def make_linear_probe(*, C: float = LINEAR_C, seed: int = 0) -> Pipeline:
    """The project's canonical linear probe. Identical recipe to the paper's."""
    return Pipeline(
        [
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=C, max_iter=LINEAR_MAX_ITER, random_state=seed)),
        ]
    )


def make_mlp_probe(*, seed: int = 0) -> Pipeline:
    """1 hidden layer of 256 ReLU units, early stopping on an internal train slice."""
    return Pipeline(
        [
            ("sc", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=MLP_HIDDEN,
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=MLP_MAX_ITER,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=15,
                    random_state=seed,
                ),
            ),
        ]
    )


def _fit_and_score(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    *,
    seed: int,
    prefix: str,
) -> tuple[dict[str, float], Pipeline | None]:
    y = np.asarray(y).astype(int)
    Xtr, ytr = X[train_mask], y[train_mask]
    Xva, yva = X[val_mask], y[val_mask]
    nan = {
        f"auroc_{prefix}": float("nan"),
        f"auroc_{prefix}_balanced": float("nan"),
    }
    if len(np.unique(ytr)) < 2 or len(yva) == 0:
        return nan, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(Xtr, ytr)
    s = np.asarray(model.predict_proba(Xva))[:, 1]
    return (
        {
            f"auroc_{prefix}": auroc(yva, s),
            f"auroc_{prefix}_balanced": auroc_balanced(yva, s, seed=seed),
        },
        model,
    )


def retrain_linear(
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    *,
    seed: int = 0,
    C: float = LINEAR_C,
) -> tuple[dict[str, float], Pipeline | None]:
    return _fit_and_score(
        make_linear_probe(C=C, seed=seed),
        X, y, train_mask, val_mask, seed=seed, prefix="retrained_linear",
    )


def retrain_mlp(
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    *,
    seed: int = 0,
) -> tuple[dict[str, float], Pipeline | None]:
    return _fit_and_score(
        make_mlp_probe(seed=seed),
        X, y, train_mask, val_mask, seed=seed, prefix="retrained_mlp",
    )


def half_fit_directions(
    X: np.ndarray,
    labels: pd.DataFrame,
    *,
    label_rule: str,
    train_mask: np.ndarray,
    seed: int = 0,
    C: float = LINEAR_C,
) -> tuple[np.ndarray, np.ndarray] | None:
    """The two probe directions fit on disjoint PROMPT halves of the train split.

    Factored out so `probe_suite` can pay for these two logistic fits ONCE and
    read both the input-space and the whitened self-consistency cosine off them.
    It used to call `direction_self_consistency` twice with the same seed — same
    halves, same fits, same directions — purely to change which basis the cosine
    was taken in, i.e. four fits per (checkpoint, layer, seed) where two suffice.

    Returns None when the split cannot be formed (too few prompts, a single-class
    half, or a probe with no linear direction).
    """
    y = labels[label_rule].to_numpy().astype(int)
    prompts = labels["prompt_idx"].to_numpy()
    tr = np.asarray(train_mask)
    uniq = np.unique(prompts[tr])
    if len(uniq) < 4:
        return None
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(uniq)
    half_a = set(shuffled[: len(shuffled) // 2].tolist())

    mask_a = tr & np.isin(prompts, list(half_a))
    mask_b = tr & ~np.isin(prompts, list(half_a))
    dirs = []
    for m in (mask_a, mask_b):
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = make_linear_probe(C=C, seed=seed).fit(X[m], y[m])
        d = probe_direction(model)
        if d is None:
            return None
        dirs.append(d)
    return dirs[0], dirs[1]


def self_consistency_cosine(
    dirs: tuple[np.ndarray, np.ndarray], *, act_std: np.ndarray | None = None
) -> float:
    """cos(v_halfA, v_halfB), in input space or (with `act_std`) whitened."""
    from .geometry import cosine

    a, b = dirs
    if act_std is None:
        return cosine(a, b)
    s = np.asarray(act_std, dtype=np.float64).ravel()
    s = np.where(s > 0, s, 0.0)
    return cosine(a * s, b * s)


def direction_self_consistency(
    X: np.ndarray,
    labels: pd.DataFrame,
    *,
    label_rule: str,
    train_mask: np.ndarray,
    seed: int = 0,
    C: float = LINEAR_C,
    act_std: np.ndarray | None = None,
    return_half_direction: bool = False,
) -> float | tuple[float, np.ndarray | None]:
    """Cosine between two linear probes fit on disjoint halves of the SAME data.

    This is the reference scale for dir_drift, and without it dir_drift is
    uninterpretable. Two logistic directions fit on 896 dimensions with C=0.1
    are only loosely determined: on the control ladder, probes fit on genuinely
    different checkpoints have cos ~0.13-0.55, which looks like catastrophic
    drift until you notice that two probes fit on *the same* checkpoint do not
    reach 1.0 either.

    Reported as `dir_drift_self_baseline`.

    IT IS NOT A VALID DENOMINATOR for cos(v0, v_half), and the ratio
    `dir_drift_normalized` built from it downstream is biased high. Both fits
    here are noisy, so this cosine is c^2 where c is a single fit's alignment
    with the truth, while cos(v0, v_half) is (c_v0 * c_half) — one factor of c,
    not two. See `probe_suite`, which computes the correctly-scaled floor
    (`dir_drift_floor_v2`) and the ratio `dir_drift_normalized_v2` from it.

    Halves are split by PROMPT so the two fits share no problem.

    Two corrections over the naive version, both of which moved the number:

    - **Sample size.** Each half-fit sees n/2 prompts, but `dir_drift` compares
      probes fit on the FULL train split. Cosine self-consistency is strongly
      size-dependent (measured: 0.134 at 35 prompts, 0.251 at 70, 0.305 at 140),
      so a half-size floor understates the true floor and makes drift look
      milder than it is. `return_half_direction=True` also hands back one of the
      half-fit directions, so the caller can form a size-MATCHED drift
      (cos(v0, v_half)) whose denominator is this baseline. Both sides then use
      the same per-fit sample size.
    - **Basis.** `dir_drift_whitened` is a cosine in the whitened basis; this
      baseline was returning an input-space cosine. Pass `act_std` to get the
      floor in the same basis as the quantity it normalises.
    """
    fail: Any = (float("nan"), None) if return_half_direction else float("nan")
    dirs = half_fit_directions(
        X, labels, label_rule=label_rule, train_mask=train_mask, seed=seed, C=C
    )
    if dirs is None:
        return fail
    base = self_consistency_cosine(dirs, act_std=act_std)
    return (base, dirs[0]) if return_half_direction else base


def _dir_drift_v2(
    frozen_probe: Any,
    lin_model: Pipeline | None,
    half_dir: np.ndarray | None,
    base_w: float,
    act_std: np.ndarray,
    *,
    n_cols: int,
) -> dict[str, float]:
    """The dir_drift family with a noise floor that is on the right SCALE.

    Writes (probe layer only; all cosines in the whitened basis, i.e. each
    coordinate scaled by this checkpoint's per-dim activation std):

      dir_drift_matched_v2      cos_w(v0, v_half)   the numerator
      dir_drift_floor_halves_v2 cos_w(v_halfA, v_halfB) = `base_w`, restated so
                                the ratio's ingredients sit together
      dir_drift_floor_nested_v2 cos_w(v_full, v_half)   DIAGNOSTIC ONLY, see below
      dir_drift_floor_v2        the zero-drift reference actually used
      dir_drift_normalized_v2   matched / floor_v2   -> ~1.0 at zero drift
      dir_drift_disattenuated_v2 matched / sqrt(base_w) -> estimates cos(v0, u_t)

    THE NAME IS NEW ON PURPOSE. `dir_drift_normalized` in
    results/fragility/metrics already holds several formulas under one name, and
    `metrics_io.latest()` keeps only the newest row per key, so a further
    redefinition under that name would silently overwrite the record rather than
    extend it. Read out of the store on 2026-08-03, the single key
    (phase0_harvest_runA, on_policy, seed 0, step 0, L16) carries three values
    written at three times, each reproducible from the sibling metrics in its own
    shard:

      3.9907  = dir_drift_whitened 0.116559 / dir_drift_self_baseline 0.029208
                (whitened numerator over an INPUT-space floor)
      1.6399  = dir_drift_matched 0.117980 / ..._self_baseline_whitened 0.071945
      1.1296  = dir_drift_matched 0.063404 / ..._self_baseline_whitened 0.056130
                (the formula currently in the tree)

    68 such cells exist across the five ladders; 39 of them exceed 1.0, up to
    7.71.

    Why a new floor. Two logistic fits on DISJOINT rows of the same checkpoint
    have cos_w = c_m * c_n, where c_n is a fit's own alignment with the true
    direction at n rows. So cos_w(v_halfA, v_halfB) = c_half^2 while the
    numerator cos_w(v0, v_half) carries only ONE such factor from this
    checkpoint. Dividing one by the other leaves a stray factor of 1/c_half.

    That multiplicative structure is not assumed, it was checked out of sample on
    real activations: measure c_q from disjoint quarters and c_h from disjoint
    halves, then PREDICT cos_w(half, disjoint quarter) = c_h * c_q. Measured /
    predicted = 0.83, 0.96, 0.98, 1.06 on vanilla_rloo_ladder steps 0 and 100 and
    phase0_harvest_runA steps 0 and 99 (L16, seed 0).

    `dir_drift_floor_v2` therefore rescales the halves floor to the numerator's
    fit sizes: with B = cos_w(halfA, halfB) = 1/(1 + r) and r the noise-to-signal
    ratio at half size, a full-size fit has c_full^2 = 1/(1 + r/2), so the
    zero-drift value of the numerator is c_full * c_half = B * sqrt(2 / (1 + B)).

    What NOT to use, and why this is not the fix that was requested.
    cos_w(v_full, v_half) looks like a directly measured zero-drift reference,
    but v_full is fit on rows that CONTAIN v_half's, so the two share estimation
    noise and the cosine is dominated by that overlap rather than by signal. In
    simulations with a class shift so small that the disjoint-halves cosine ran
    -0.001 to 0.011 — no recoverable signal at all — it still sat at 0.649-0.662,
    close to the sqrt(1/2) a pure shared-noise pairing predicts. On real data it
    is 4.0x-8.8x the corrected floor (0.6662 vs 0.1491 at vanilla_rloo_ladder
    step 0 L16). It is written here as a diagnostic, never used as a denominator.

    Calibration, MEASURED on synthetic data where the true drift angle is known
    by construction (d=256, 2000 rows per fit source, positive rate 0.27, this
    module's `make_linear_probe`, isotropic N(0, I) plus a class-mean shift tuned
    so cos_w(halfA, halfB) = 0.110 against the real ladder's 0.111; v0 is a
    full-size fit on an INDEPENDENT sample from the undrifted direction;
    60 replicates, ratio of means):

      true cos(theta)        1.000  0.940  0.866  0.500  0.000
      matched / halves floor 1.299  1.277  1.257  0.646  0.060
      matched / nested floor 0.204  0.197  0.186  0.107  0.010
      matched / floor_v2     0.892  0.905  0.924  0.460  0.045

    Mean absolute error against the truth: 0.057 for floor_v2, 0.247 for the
    halves floor, 0.524 for the nested floor. floor_v2 is the best of the three
    and is NOT exact — it read 0.89 where the truth was 1.0 — so a v2 value
    within roughly +-0.1 of 1.0 should be read as "no detectable drift", not as a
    point estimate.

    Recomputed over all 32 cached probe-layer snapshots (5 ladders, L16, seed 0):
    median ratio 1.205 with the halves floor, 0.890 with floor_v2, 0.118 with the
    nested floor. The nested floor puts phase0_harvest_runA STEP 0 — where drift
    is ~0 by construction — at 0.094, i.e. it would report the direction as
    almost entirely rotated away before any optimisation has happened.

    A v2 ratio above 1.0 is not automatically an error. The floor references a
    full-train fit on THIS checkpoint's rows, so a v0 fit on a larger or cleaner
    sample can legitimately beat it: on vanilla_rloo_ladder the frozen probe was
    fit on that ladder's own endpoint from the bigger n500 cache, and its ratio
    rises 0.78 (step 0) -> 2.36 (step 100), which is the expected direction.
    13 of the 32 snapshots exceed 1.0 under floor_v2, against 22 of 32 under the
    halves floor.

    `dir_drift_disattenuated_v2` divides by c_half = sqrt(B) instead, which needs
    only the multiplicative law above and no extrapolation in fit size. It
    estimates cos_w(v0, u_t): the alignment of the FROZEN probe direction with
    the checkpoint's true correctness direction. It does not reach 1.0 at zero
    drift (v0 is itself a finite-sample fit), so it answers "how well does the
    deployed monitor point at correctness now", not "how far has it moved".
    """
    from .geometry import cosine

    out: dict[str, float] = {}
    if lin_model is None or half_dir is None:
        return out
    v0 = probe_direction(frozen_probe)
    v_full = probe_direction(lin_model)
    if v0 is None or v_full is None:
        return out
    if v0.size != n_cols or v_full.size != n_cols or np.asarray(half_dir).size != n_cols:
        return out

    s = np.asarray(act_std, dtype=np.float64).ravel()
    if s.size != n_cols:
        return out
    s = np.where(s > 0, s, 0.0)
    half = np.asarray(half_dir, dtype=np.float64).ravel()

    matched = cosine(v0 * s, half * s)
    out["dir_drift_matched_v2"] = matched
    out["dir_drift_floor_halves_v2"] = float(base_w)
    out["dir_drift_floor_nested_v2"] = cosine(v_full * s, half * s)

    B = float(base_w)
    if not (B > 0.0):
        # A non-positive halves cosine means the refit noise swamps the signal
        # entirely; there is no scale to normalise by. Absent, not zero.
        out["dir_drift_floor_v2"] = float("nan")
        out["dir_drift_normalized_v2"] = float("nan")
        out["dir_drift_disattenuated_v2"] = float("nan")
        return out
    floor = B * np.sqrt(2.0 / (1.0 + B))
    out["dir_drift_floor_v2"] = float(floor)
    out["dir_drift_normalized_v2"] = float(matched / floor)
    out["dir_drift_disattenuated_v2"] = float(matched / np.sqrt(B))
    return out


def probe_suite(
    X: np.ndarray,
    labels: pd.DataFrame,
    *,
    frozen_probe: Any,
    label_rule: str,
    seed: int = 0,
    C: float = LINEAR_C,
    fit_mlp: bool = True,
    include_frozen: bool = True,
) -> tuple[dict[str, float], dict[str, Any]]:
    """All Phase-1 probe metrics for one (checkpoint, layer) snapshot.

    `labels` must carry the chosen label column and a "split" column. Returns
    (metrics, fitted) where `fitted` holds the retrained models so geometry code
    can pull their directions without refitting.

    `include_frozen=False` for BRACKETING layers. The frozen probe's weights
    live in the probe layer's basis; applying them to another layer's residual
    stream is not a weakened monitor, it is a category error, and the resulting
    "collapse" would be an artefact of the layer swap. Bracketing layers exist to
    answer a different question — does RETRAINED decodability rise elsewhere as
    the probe layer falls (relocation)? — so only the retrained metrics are
    defined there. evasion_gap, which differences the two, is likewise
    probe-layer-only.
    """
    if label_rule not in labels.columns:
        raise KeyError(f"label column {label_rule!r} not in labels ({list(labels.columns)})")
    if "split" not in labels.columns:
        raise KeyError("labels needs a 'split' column; call labels.attach_split first")

    y = labels[label_rule].to_numpy().astype(int)
    train_mask = (labels["split"] == "train").to_numpy()
    val_mask = (labels["split"] == "val").to_numpy()

    metrics: dict[str, float] = {
        "n_rows": float(len(y)),
        "n_train": float(train_mask.sum()),
        "n_val": float(val_mask.sum()),
    }
    fitted_extra: dict[str, Any] = {}
    if include_frozen:
        metrics.update(eval_frozen(frozen_probe, X[val_mask], y[val_mask], seed=seed))

    lin_metrics, lin_model = retrain_linear(X, y, train_mask, val_mask, seed=seed, C=C)
    metrics.update(lin_metrics)

    if fit_mlp:
        mlp_metrics, mlp_model = retrain_mlp(X, y, train_mask, val_mask, seed=seed)
        metrics.update(mlp_metrics)
    else:
        mlp_model = None

    if include_frozen:
        metrics["evasion_gap"] = metrics["auroc_retrained_linear"] - metrics["auroc_frozen"]
        metrics["evasion_gap_balanced"] = (
            metrics["auroc_retrained_linear_balanced"] - metrics["auroc_frozen_balanced"]
        )
    metrics["nonlinear_gain"] = metrics.get("auroc_retrained_mlp", float("nan")) - metrics[
        "auroc_retrained_linear"
    ]
    act_std = X.std(axis=0)
    # ONE pair of half-fits, read in both bases. The two cosines below are taken
    # from the same two logistic fits, which is what they always described — the
    # previous code fit them twice with the same seed to get the same halves.
    half_dirs = half_fit_directions(
        X, labels, label_rule=label_rule, train_mask=train_mask, seed=seed, C=C
    )
    if half_dirs is None:
        metrics["dir_drift_self_baseline"] = float("nan")
        base_w, half_dir = float("nan"), None
    else:
        metrics["dir_drift_self_baseline"] = self_consistency_cosine(half_dirs)
        base_w = self_consistency_cosine(half_dirs, act_std=act_std)
        half_dir = half_dirs[0]
    metrics["dir_drift_self_baseline_whitened"] = base_w
    # The half-fit direction is handed to the caller so dir_drift can be formed
    # at the SAME per-fit sample size as its own noise floor.
    fitted_extra["half_direction"] = half_dir
    if include_frozen:
        metrics.update(
            _dir_drift_v2(
                frozen_probe, lin_model, half_dir, base_w, act_std, n_cols=X.shape[1]
            )
        )

    # Function-space agreement between the frozen and retrained probes.
    #
    # This exists because weight-space cosine (dir_drift) turns out to be a weak
    # instrument here: two linear probes fit on disjoint halves of the SAME
    # checkpoint agree at cos ~0.1 while both score AUROC ~0.85. The residual
    # stream's dimensions are correlated enough that many directions implement
    # nearly the same classifier, so a small cosine does not imply the probes
    # disagree. Rank correlation between their OUTPUTS is well determined and is
    # what "the monitor and a fresh monitor now disagree" actually means.
    if include_frozen and lin_model is not None and val_mask.sum() > 2:
        s_frozen = probe_scores(frozen_probe, X[val_mask])
        s_retrained = np.asarray(lin_model.predict_proba(X[val_mask]))[:, 1]
        if np.std(s_frozen) > 0 and np.std(s_retrained) > 0:
            metrics["score_corr_frozen_retrained"] = float(
                stats.spearmanr(s_frozen, s_retrained).statistic
            )
        else:
            metrics["score_corr_frozen_retrained"] = float("nan")

    return metrics, {"linear": lin_model, "mlp": mlp_model, **fitted_extra}
