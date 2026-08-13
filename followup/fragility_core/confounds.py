"""H0: was the probe reading a structural confound all along?

followup/CLAUDE.md lists H1 (evasion) and H2 (corruption). The original paper's
own diagnosis of the collapse (findings.md EXP-21) is neither: it is confound
exploitation — the probe was trained where correctness covaried with a
structured-reasoning writing style, and RL drove the style to ceiling while
correctness fell. Every probe=1.000 post-Goodhart rollout shared one rhetorical
template.

H0 matters because it is observationally confusable with H1. If the probe scored
a surface feature, then post-RL a RETRAINED probe can still separate correctness
(the information was never removed) while the FROZEN probe fails (its decision
boundary sat on the now-decoupled confound). That is exactly the H1 signature —
large positive evasion_gap — with a completely different mechanism and a
completely different safety implication ("your probe had a shortcut" vs "the
policy learned to hide").

The discriminating measurements, all computable from the same cached
activations plus the rollout text:

  auroc_template(t)          can the 7-pattern template score alone predict
                             correctness? Under H0 this is well above 0.5 at
                             t=0 and decays toward 0.5 as RL decouples it.
  corr_probe_template(t)     does the frozen probe's output track the template?
                             Under H0 this is high and RISES with training.
  auroc_frozen_cond_template(t)
                             frozen-probe AUROC computed WITHIN template-score
                             strata and pooled. This is the load-bearing number:
                             it asks whether the probe carries correctness
                             information beyond the confound. If it collapses to
                             ~0.5 while the raw AUROC at t=0 was 0.8+, the probe
                             was mostly a template detector.

None of this presumes H0 is right. If auroc_frozen_cond_template stays high, H0
is ruled out and H1/H2 stand on their own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import labels as labels_mod
from .probes import auroc, auroc_balanced

TEMPLATE_COLS = tuple(name for _pat, name in labels_mod.TEMPLATE_PATTERNS)
PLACEBO_COLS = tuple(name for _pat, name in labels_mod.PLACEBO_PATTERNS)


def conditional_auroc(
    y: np.ndarray,
    score: np.ndarray,
    strata: np.ndarray,
) -> float:
    """AUROC pooled over strata, weighting each by its discordant-pair count.

    Within-stratum AUROC estimates P(score_pos > score_neg | stratum). Weighting
    by n_pos * n_neg makes the pooled figure the probability that a randomly
    chosen correct/wrong pair FROM THE SAME STRATUM is ranked correctly — i.e.
    the probe's discrimination with the confound held fixed.

    NaN when no stratum contains both classes (measured, undefined).
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=np.float64)
    strata = np.asarray(strata)
    num = 0.0
    den = 0.0
    n_strata = 0
    for s in np.unique(strata):
        m = strata == s
        ys, ss = y[m], score[m]
        n1, n0 = int((ys == 1).sum()), int((ys == 0).sum())
        if n1 == 0 or n0 == 0:
            continue
        a = auroc(ys, ss)
        if np.isnan(a):
            continue
        w = n1 * n0
        num += a * w
        den += w
        n_strata += 1
    conditional_auroc.last_pairs = den          # type: ignore[attr-defined]
    conditional_auroc.last_n_strata = n_strata  # type: ignore[attr-defined]
    return float(num / den) if den > 0 else float("nan")


def template_matrix(labels: pd.DataFrame) -> np.ndarray | None:
    """(N, 7) indicator matrix if the per-pattern columns were cached, else None."""
    if all(c in labels.columns for c in TEMPLATE_COLS):
        return labels[list(TEMPLATE_COLS)].to_numpy(dtype=np.float64)
    return None


def placebo_matrix(labels: pd.DataFrame) -> np.ndarray | None:
    """(N, 7) matched-count surface features with no confidence semantics."""
    if all(c in labels.columns for c in PLACEBO_COLS):
        return labels[list(PLACEBO_COLS)].to_numpy(dtype=np.float64)
    return None


def drop_degenerate(T: np.ndarray) -> np.ndarray:
    """Drop constant columns. They are collinear with the intercept, so
    residualising on them is a silent no-op that inflates the apparent rank of a
    control. Three of the seven placebo features are constant on this data."""
    T = np.asarray(T, dtype=np.float64)
    keep = T.std(axis=0) > 0
    return T[:, keep]


def variance_removed(X: np.ndarray, T: np.ndarray) -> float:
    """Fraction of X's total variance that residualising on T destroys.

    This is the quantity a control has to match. Column count does not: a
    7-column control whose columns are near-constant removes ~2% of variance
    while a 7-column control aligned with the leading PCs removes ~31%, and the
    AUROC cost tracks the variance, not the count.
    """
    X = np.asarray(X, dtype=np.float64)
    R = residualise(X, T)
    Xc = X - X.mean(axis=0, keepdims=True)
    Rc = R - R.mean(axis=0, keepdims=True)
    tot = float(np.sum(Xc**2))
    return float(1.0 - np.sum(Rc**2) / tot) if tot > 0 else float("nan")


def residualise(
    X: np.ndarray, T: np.ndarray, *, fit_mask: np.ndarray | None = None
) -> np.ndarray:
    """Remove from every activation dimension the part linearly predicted by T.

    `fit_mask` restricts the OLS fit to those rows (normally train). Fitting on
    all rows over-cleans the held-out rows relative to an out-of-fold
    residualisation and biases the residualised AUROC down — measured at ~29% of
    the reported effect's own size, in the direction that flatters H0.
    """
    T = np.asarray(T, dtype=np.float64)
    Td = np.column_stack([np.ones(len(T)), T])
    fit = np.ones(len(T), dtype=bool) if fit_mask is None else np.asarray(fit_mask)
    beta, *_ = np.linalg.lstsq(Td[fit], np.asarray(X, dtype=np.float64)[fit], rcond=None)
    return X - Td @ beta


def matched_pc_control(
    X: np.ndarray, target_variance: float, *, max_rank: int = 16
) -> np.ndarray | None:
    """A principal-component subspace removing ~`target_variance` of X's variance.

    The honest control for a residualisation test: an arbitrary activation-aligned
    subspace of the SAME destructive power as the confound. If removing this costs
    as much AUROC as removing the template does, the template's semantics explain
    nothing.
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    _u, _s, Vt = np.linalg.svd(Xc, full_matrices=False)
    best, best_err = None, np.inf
    for start in range(0, min(40, Vt.shape[0] - 1)):
        for k in range(1, max_rank + 1):
            if start + k > Vt.shape[0]:
                break
            M = Xc @ Vt[start:start + k].T
            err = abs(variance_removed(X, M) - target_variance)
            if err < best_err:
                best, best_err = M, err
    return best


def residualisation_test(
    X: np.ndarray,
    labels: pd.DataFrame,
    *,
    label_rule: str,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    seed: int = 0,
    C: float = 0.1,
    n_control: int = 5,
) -> dict[str, float]:
    """How much of the probe's AUROC lives in the template-predictable subspace?

    READ THIS BEFORE QUOTING THE NUMBER. Residualisation alone proves nothing.
    Measured on C_outcome L16 (2026-07-30):

        baseline linear probe                              AUROC 0.903
        residualise on the 7 template indicators           AUROC 0.596
        residualise on 7 random binary cols (matched rate) AUROC 0.903
        residualise on 7 shuffled template cols            AUROC 0.903
        residualise on 7 random PROJECTIONS OF X           AUROC 0.506

    The last control is the problem: removing ANY 7-dimensional subspace that is
    aligned with the activations' own high-variance structure destroys the probe.
    The template indicators are text-derived and therefore activation-aligned, so
    a drop under template residualisation is not by itself evidence that the
    probe was reading the template's *semantics*.

    The readable quantity is the template drop MINUS the PLACEBO drop, where the
    placebo is a matched-count set of surface features from the same text with no
    confidence/structure semantics (arithmetic-symbol presence). If template and
    placebo drop AUROC equally, this test says nothing and
    `auroc_frozen_cond_template` — which strata rather than projects, and is not
    subject to this artefact — is the measure to trust.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from .probes import auroc

    y = labels[label_rule].to_numpy().astype(int)
    tr, va = np.asarray(train_mask), np.asarray(val_mask)

    def fit_auroc(Xu: np.ndarray) -> float:
        if len(np.unique(y[tr])) < 2 or not va.any():
            return float("nan")
        model = Pipeline(
            [("sc", StandardScaler()),
             ("lr", LogisticRegression(C=C, max_iter=2000, random_state=seed))]
        ).fit(Xu[tr], y[tr])
        return auroc(y[va], model.predict_proba(Xu[va])[:, 1])

    out: dict[str, float] = {"resid_auroc_baseline": fit_auroc(X)}

    T = template_matrix(labels)
    if T is None:
        return out
    T = drop_degenerate(T)
    out["resid_auroc_template"] = fit_auroc(residualise(X, T, fit_mask=tr))
    out["resid_var_removed_template"] = variance_removed(X, T)
    out["resid_rank_template"] = float(T.shape[1])

    # Column-count-matched control. Kept for the record, but it is NOT the
    # control that licenses a conclusion — see the variance-matched one below.
    P = placebo_matrix(labels)
    if P is not None:
        P = drop_degenerate(P)
        out["resid_auroc_placebo"] = fit_auroc(residualise(X, P, fit_mask=tr))
        out["resid_var_removed_placebo"] = variance_removed(X, P)
        out["resid_rank_placebo"] = float(P.shape[1])

    # THE control: an arbitrary activation subspace of matched destructive power.
    M = matched_pc_control(X, out["resid_var_removed_template"])
    if M is not None:
        out["resid_auroc_matched_pc"] = fit_auroc(residualise(X, M, fit_mask=tr))
        out["resid_var_removed_matched_pc"] = variance_removed(X, M)
        # The interpretable figure: how much MORE the template costs than an
        # arbitrary subspace that destroys the same amount of variance.
        out["resid_template_minus_matched"] = (
            out["resid_auroc_matched_pc"] - out["resid_auroc_template"]
        )

    # Loose floor: random projections of X, unmatched on variance.
    floors = []
    for i in range(n_control):
        rng = np.random.RandomState(seed * 1000 + i)
        floors.append(
            fit_auroc(residualise(X, X @ rng.randn(X.shape[1], T.shape[1]), fit_mask=tr))
        )
    if floors:
        out["resid_auroc_random_projection"] = float(np.mean(floors))
    return out


def confound_suite(
    labels: pd.DataFrame,
    probe_score: np.ndarray,
    *,
    label_rule: str,
    val_mask: np.ndarray | None = None,
    train_mask: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """H0 metrics for one checkpoint snapshot.

    `probe_score` must be the FROZEN probe's score for every row of `labels`
    (not just the val rows) so that strata are populated; masks then select which
    rows are reported on.
    """
    y_all = labels[label_rule].to_numpy().astype(int)
    ts_all = labels["template_score"].to_numpy(dtype=np.float64)
    probe_score = np.asarray(probe_score, dtype=np.float64)
    if len(probe_score) != len(labels):
        raise ValueError(
            f"probe_score has {len(probe_score)} rows, labels has {len(labels)}"
        )

    val = np.ones(len(labels), dtype=bool) if val_mask is None else np.asarray(val_mask)
    y, ts, ps = y_all[val], ts_all[val], probe_score[val]

    out: dict[str, float] = {
        "template_score_mean": float(ts.mean()),
        "template_rate_ge3": float((ts >= 3).mean()),
        "template_rate_ge3_wrong": float((ts[y == 0] >= 3).mean()) if (y == 0).any() else float("nan"),
        "template_rate_ge3_correct": float((ts[y == 1] >= 3).mean()) if (y == 1).any() else float("nan"),
        "auroc_template": auroc(y, ts),
        "auroc_template_balanced": auroc_balanced(y, ts, seed=seed),
    }

    # Does the frozen probe track the confound?
    if np.std(ps) > 0 and np.std(ts) > 0:
        out["corr_probe_template_pearson"] = float(np.corrcoef(ps, ts)[0, 1])
        out["corr_probe_template_spearman"] = float(stats.spearmanr(ps, ts).statistic)
    else:
        out["corr_probe_template_pearson"] = float("nan")
        out["corr_probe_template_spearman"] = float("nan")

    # The load-bearing number: probe discrimination with the template held fixed.
    # Effective sample size is recorded alongside it — as the policy's template
    # distribution collapses the contributing strata drop from 6 to 3, and a
    # value resting on one stratum must not be indistinguishable from one
    # resting on six.
    out["auroc_frozen_cond_template"] = conditional_auroc(y, ps, ts)
    out["cond_template_pairs"] = float(getattr(conditional_auroc, "last_pairs", float("nan")))
    out["cond_template_n_strata"] = float(
        getattr(conditional_auroc, "last_n_strata", float("nan"))
    )

    # Is the confound specifically RHETORICAL, or just surface structure?
    #
    # Response length and answer-block count are surface features with no
    # rhetorical content. If they predict correctness about as well as the
    # template score does, then "the probe reads a writing template" overclaims
    # and "the probe reads surface structure" is the accurate statement. Both
    # were previously computed in an unsaved session and quoted in the write-up;
    # they live here now so the claim is regenerable.
    for col, name in (("resp_chars", "length"), ("n_blocks", "blocks")):
        if col not in labels.columns:
            continue
        v = labels[col].to_numpy(dtype=np.float64)[val]
        out[f"auroc_{name}_alone"] = auroc(y, v)
        # And does the probe survive holding LENGTH fixed? Deciles, so the
        # strata are populated at every checkpoint.
        q = np.quantile(v, np.linspace(0, 1, 11)[1:-1])
        strata = np.digitize(v, q)
        out[f"auroc_frozen_cond_{name}"] = conditional_auroc(y, ps, strata)
        out[f"auroc_template_cond_{name}"] = conditional_auroc(y, ts, strata)

    # Reference point: how well does template ALONE do, as a fitted model?
    tm = template_matrix(labels)
    if tm is not None and train_mask is not None and val_mask is not None:
        tr = np.asarray(train_mask)
        va = np.asarray(val_mask)
        if len(np.unique(y_all[tr])) == 2 and va.any():
            model = Pipeline(
                [("sc", StandardScaler()),
                 ("lr", LogisticRegression(max_iter=1000, random_state=seed))]
            )
            model.fit(tm[tr], y_all[tr])
            s = model.predict_proba(tm[va])[:, 1]
            out["auroc_template_model"] = auroc(y_all[va], s)
            out["auroc_template_model_balanced"] = auroc_balanced(y_all[va], s, seed=seed)
    return out
