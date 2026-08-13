"""A probe that cannot read the cheap surface features, for use as an RL reward.

Motivation. The workshop paper's line 179 asks, as an open question, whether a
probe trained to ignore rhetorical scaffolding would still be near-oracle and
would resist confound exploitation under RL. This module builds that probe.

The measured facts it is responding to (all on clean-406, C_outcome):

  - the probe reads more than trace length: AUROC 0.9807 stratified within
    </think>-length deciles, so it is not a length detector;
  - but a large share of its signal is surface-explainable in the broader sense:
    residualising the activations against the full 39-feature battery drops
    held-out AUROC 0.9775 -> 0.8363, and drops the correlation with a pure
    surface model's score from +0.858 to +0.255;

    (39, not 23. The 23-feature figure elsewhere in this project is the subset
    used for the dose-response correlation, where near-constant features were
    dropped because a standardised shift on them explodes. Residualisation uses
    all 39 — dropping a rare feature there would leave its component in the
    activation, which is the opposite of the intent.)
  - and the attacked policy moves exactly those features, in proportion to how
    much correctness signal each one carried.

So the question is whether the 0.836 probe, which is weaker read-only, is
*harder to Goodhart*.

Scoring is `LR(h - s @ B)`, where `h` is the </think> hidden state, `s` is the
z-scored surface feature vector of the same rollout (plus an intercept), and `B`
is the least-squares map from surface features to activations, fit ONCE on the
baseline distribution and frozen.

Two things about that definition that must not be quietly forgotten:

  - `B` is fit on the baseline distribution. Off-distribution — which is exactly
    where the policy goes — `h - s @ B` is no longer a calibrated "surface-free"
    activation. This is a property of the experiment, not a bug, but it bounds
    what a null result would mean.
  - the residualised probe is WEAKER read-only (0.836 vs 0.978). A smaller
    collapse is therefore ambiguous between "less surface coupling helps" and
    "weaker probes collapse less". The surface-model-as-reward positive control
    is what disambiguates; do not run this arm alone and read it as an answer.

The class carries `needs_text = True`. The forked reward builder in
`residual_reward.py` checks that attribute and passes the rollout text; probes
without it are called exactly as before, so the existing reward path is
unchanged.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

# The 24 reasoning-marker words. With the 15 structural features in
# `surface_features` they make the 39-FEATURE battery -- not 23. "23" is the
# count that survives dropping near-constant columns, and it applies only to the
# dose-response correlation; residualisation uses all 39 (see the module
# docstring). Mislabelling this list "the 23-feature battery" is what put
# "the full 23-feature surface battery" next to the 0.9775 -> 0.8363 drop in
# REVISION_PACK B2, which is a 39-feature result.
#
# This list is the analysis artefact: it is the same set used to measure that
# drop and the dose-response, and changing it silently would decouple the probe
# from the numbers that motivate it.
MARKER_WORDS = [
    "wait", "hmm", "actually", "let me", "alternatively", "try", "instead",
    "no,", "but", "check", "recheck", "close", "not quite", "hold on",
    "therefore", "thus", "so the answer", "final", "perfect", "great", "yes",
    "correct", "verify", "confirm",
]

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_CLOSE = "</think>"


def surface_features(text: str) -> dict[str, float]:
    """Cheap textual features of one rollout. No model, no tokeniser.

    Everything is measured on the reasoning trace (text before `</think>`)
    except the answer-derived and tail features, matching the analysis.
    """
    cut = text.find(THINK_CLOSE)
    think = text[:cut] if cut >= 0 else text
    low = think.lower()
    m = _ANSWER_RE.search(text)
    ans = m.group(1) if m else ""
    words = think.split()
    feats: dict[str, float] = {
        "len_think": float(len(think)),
        "n_equals": float(think.count("=")),
        "n_lines": float(think.count("\n")),
        "n_ops": float(len(re.findall(r"[+\-*/]", think))),
        "n_digits": float(len(re.findall(r"\d", think))),
        "n_paren": float(think.count("(")),
        "n_sent": float(think.count(".")),
        "n_comma": float(think.count(",")),
        "ans_len": float(len(ans)),
        "ans_ops": float(len(re.findall(r"[+\-*/]", ans))),
        "ans_paren": float(ans.count("(")),
        "mean_line": len(think) / max(think.count("\n"), 1),
        "n_words": float(len(words)),
        "uniq_ratio": len(set(words)) / max(len(words), 1),
        "tail_len": float(len(text) - cut if cut >= 0 else 0),
    }
    for w in MARKER_WORDS:
        feats["w_" + w.replace(" ", "_").replace(",", "")] = float(low.count(w))
    return feats


def feature_matrix(texts: list[str], keys: list[str]) -> np.ndarray:
    return np.array([[surface_features(t)[k] for k in keys] for t in texts], dtype=np.float64)


class LinearScorer:
    """StandardScaler + LogisticRegression, as plain arrays and numpy.

    The sklearn objects are NOT carried into the artefact. Pickling a fitted
    `Pipeline(StandardScaler, LogisticRegression)` embeds sklearn's internal
    attribute layout, and a container running a different sklearn than the
    machine that fit it dies inside `predict_proba` — observed as
    `AttributeError: 'LogisticRegression' object has no attribute 'multi_class'`
    on a Modal worker, AFTER the model had loaded and training had begun.

    A scaler plus a logistic regression is `sigmoid(((x - mu)/sd) @ w + b)`.
    Writing those four arrays down makes the artefact independent of every
    sklearn version, which is worth more than the convenience of pickling.
    """

    def __init__(self, mu, sd, w, b):
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sd = np.asarray(sd, dtype=np.float64)
        self.w = np.asarray(w, dtype=np.float64).reshape(-1)
        self.b = float(b)

    @classmethod
    def from_pipeline(cls, pipe) -> "LinearScorer":
        sc, lr = pipe.steps[0][1], pipe.steps[-1][1]
        if lr.coef_.shape[0] != 1:
            raise ValueError("binary logistic regression expected")
        return cls(sc.mean_, sc.scale_, lr.coef_[0], float(lr.intercept_[0]))

    def predict_proba(self, X):
        z = ((np.asarray(X, dtype=np.float64) - self.mu) / self.sd) @ self.w + self.b
        p = 1.0 / (1.0 + np.exp(-z))
        return np.stack([1.0 - p, p], axis=1)


class SurfaceResidualProbe:
    """`LR(h - s @ B)`. Frozen; nothing here is refit at RL time.

    `needs_text` tells the forked reward builder to pass the rollout text. A
    probe object without that attribute is called with the activation alone, so
    the unmodified reward path is bit-identical for every existing probe.
    """

    needs_text = True

    def __init__(self, keys, mu, sigma, B, pipe, *, provenance=None, identity=False):
        self.keys = list(keys)
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)      # (n_feat + 1, d_model)
        self.pipe = pipe
        self.provenance = provenance or {}
        # identity=True zeroes the residualisation. Used ONLY by the gate, where
        # it must reproduce a plain probe exactly.
        self.identity = bool(identity)

    def _residual(self, X: np.ndarray, texts: list[str]) -> np.ndarray:
        if self.identity:
            # Return the caller's array untouched, dtype included. Upcasting
            # float32 activations to float64 here shifted scores by ~1.5e-07,
            # which is negligible as a reward but made the identity gate
            # inexact — and an identity path that is only approximately the
            # identity is not a gate, it is a tolerance.
            return np.asarray(X)
        X = np.asarray(X, dtype=np.float64)
        S = feature_matrix(texts, self.keys)
        Sz = (S - self.mu) / self.sigma
        Sz = np.concatenate([Sz, np.ones((len(Sz), 1))], axis=1)
        return X - Sz @ self.B

    def predict_proba(self, X, text=None):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[None, :]
        if text is None:
            raise ValueError(
                "SurfaceResidualProbe needs the rollout text. The caller must "
                "check `needs_text` and pass text=..., or the residualisation "
                "silently does not happen."
            )
        texts = [text] if isinstance(text, str) else list(text)
        if len(texts) != len(X):
            raise ValueError(f"{len(X)} activations but {len(texts)} texts")
        return self.pipe.predict_proba(self._residual(X, texts))


def fit(X: np.ndarray, texts: list[str], y: np.ndarray, groups: np.ndarray,
        *, C: float = 0.05, keys: list[str] | None = None,
        provenance: dict[str, Any] | None = None) -> tuple[SurfaceResidualProbe, dict]:
    """Fit the residualiser and the probe on a prompt-disjoint training split.

    `B`, the scaler stats, and the LR are all fit on the SAME rows. Held-out
    AUROC is reported on the complementary prompts and is the number to compare
    against the raw probe's 0.9775.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    keys = keys or sorted(surface_features(texts[0]))
    S = feature_matrix(texts, keys)

    import hashlib
    te = np.array([int(hashlib.sha256(str(int(g)).encode()).hexdigest(), 16) % 2 == 0
                   for g in groups])
    tr = ~te

    mu, sd = S[tr].mean(0), S[tr].std(0) + 1e-9
    Sz = np.concatenate([(S - mu) / sd, np.ones((len(S), 1))], axis=1)
    B = np.linalg.lstsq(Sz[tr], X[tr], rcond=None)[0]
    Xr = X - Sz @ B

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=C, class_weight="balanced"),
    ).fit(Xr[tr], y[tr])

    raw = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=C, class_weight="balanced"),
    ).fit(X[tr], y[tr])

    report = {
        "n_rows": int(len(y)),
        "n_train_rows": int(tr.sum()),
        "n_features": len(keys),
        "auroc_residual_heldout": float(roc_auc_score(y[te], pipe.predict_proba(Xr[te])[:, 1])),
        "auroc_raw_heldout": float(roc_auc_score(y[te], raw.predict_proba(X[te])[:, 1])),
        "keys": keys,
    }
    scorer = LinearScorer.from_pipeline(pipe)
    # the artefact must score identically to the sklearn pipeline it came from
    _a = pipe.predict_proba(Xr[te])[:, 1]
    _b = scorer.predict_proba(Xr[te])[:, 1]
    if not np.allclose(_a, _b, atol=1e-9, rtol=0):
        raise RuntimeError(
            f"LinearScorer disagrees with the sklearn pipeline "
            f"(max {np.abs(_a - _b).max():.3e}); refusing to save.")
    report["scorer_max_abs_dev_from_sklearn"] = float(np.abs(_a - _b).max())
    probe = SurfaceResidualProbe(keys, mu, sd, B, scorer,
                                 provenance={**(provenance or {}), "report": report})
    return probe, report


def save(probe: SurfaceResidualProbe, path: str) -> None:
    """Persist as a plain dict, NOT as a pickled `SurfaceResidualProbe`.

    Pickling the instance embeds this module's import path in the artefact, so
    `pickle.load` in a Modal container would need `surface_residual_probe`
    importable under exactly that name — and the loader in
    `extension/training/probe_reward_rloo.py` does no sys.path setup. That fails
    at container start, or later, when the reward is first called.

    A dict of arrays plus the sklearn pipeline (whose classes are installed
    everywhere) has no such dependency. `load` rebuilds the object.
    """
    import pickle
    state = {
        "__format__": "surface_residual_probe/v1",
        "keys": probe.keys,
        "mu": probe.mu,
        "sigma": probe.sigma,
        "B": probe.B,
        "scorer": {"mu": probe.pipe.mu, "sd": probe.pipe.sd,
                   "w": probe.pipe.w, "b": probe.pipe.b},
        "provenance": probe.provenance,
    }
    with open(path, "wb") as f:
        pickle.dump(state, f)
    with open(str(path) + ".meta.json", "w") as f:
        json.dump(probe.provenance, f, indent=2, default=str)


def load(path: str) -> SurfaceResidualProbe:
    """Rebuild from `save`'s dict. Raises on anything else rather than guessing."""
    import pickle
    with open(path, "rb") as f:
        state = pickle.load(f)
    if isinstance(state, SurfaceResidualProbe):
        return state  # tolerate an instance saved by an older build
    if not (isinstance(state, dict) and state.get("__format__") == "surface_residual_probe/v1"):
        raise ValueError(
            f"{path} is not a surface_residual_probe/v1 artefact "
            f"(got {type(state).__name__}). Refusing to guess: a probe that "
            "silently loads wrong would train against the wrong reward."
        )
    sc = state["scorer"]
    return SurfaceResidualProbe(
        state["keys"], state["mu"], state["sigma"], state["B"],
        LinearScorer(sc["mu"], sc["sd"], sc["w"], sc["b"]),
        provenance=state.get("provenance"),
    )


class SurfaceOnlyProbe:
    """Scores from the 39 surface features ALONE. Never reads the activation.

    Arm B of the pre-registered design: the falsifier for the surface-occupation
    account in REVISION_PACK.md section G2. If a reward with no access to the
    model's internals still collapses true accuracy, occupying cheap textual
    structure is sufficient to break a monitor. If it does NOT collapse, G2 is
    wrong and should be retracted.

    This is not a trivial test. The surface model is a 0.925-AUROC correctness
    predictor -- genuinely good -- so optimising it could plausibly IMPROVE
    accuracy rather than destroy it.

    `predict_proba` takes `X` for interface compatibility and ignores it. The
    gate asserts that by scoring the same texts against zero and random
    activations and requiring bit-identical output.
    """

    needs_text = True

    def __init__(self, keys, scorer, *, provenance=None):
        self.keys = list(keys)
        self.scorer = scorer
        self.provenance = provenance or {}

    def predict_proba(self, X, text=None):
        if text is None:
            raise ValueError("SurfaceOnlyProbe needs the rollout text.")
        texts = [text] if isinstance(text, str) else list(text)
        return self.scorer.predict_proba(feature_matrix(texts, self.keys))


def fit_surface_only(texts, y, groups, *, C=0.1, keys=None, provenance=None):
    """Fit Arm B's probe on a prompt-disjoint split. Mirrors `fit`'s protocol."""
    import hashlib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    keys = keys or sorted(surface_features(texts[0]))
    S = feature_matrix(texts, keys)
    te = np.array([int(hashlib.sha256(str(int(g)).encode()).hexdigest(), 16) % 2 == 0
                   for g in groups])
    tr = ~te
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, C=C,
                                            class_weight="balanced")).fit(S[tr], y[tr])
    scorer = LinearScorer.from_pipeline(pipe)
    a, b = pipe.predict_proba(S[te])[:, 1], scorer.predict_proba(S[te])[:, 1]
    if not np.allclose(a, b, atol=1e-9, rtol=0):
        raise RuntimeError("LinearScorer disagrees with sklearn; refusing to save.")
    report = {"auroc_heldout": float(roc_auc_score(y[te], b)),
              "n_features": len(keys), "keys": keys}
    return SurfaceOnlyProbe(keys, scorer,
                            provenance={**(provenance or {}), "report": report}), report


def save_surface_only(probe, path):
    import pickle
    st = {"__format__": "surface_only_probe/v1", "keys": probe.keys,
          "scorer": {"mu": probe.scorer.mu, "sd": probe.scorer.sd,
                     "w": probe.scorer.w, "b": probe.scorer.b},
          "provenance": probe.provenance}
    with open(path, "wb") as f:
        pickle.dump(st, f)
    with open(str(path) + ".meta.json", "w") as f:
        json.dump(probe.provenance, f, indent=2, default=str)


def load_any(path):
    """Loader for the RL launcher: handles both v1 artefact formats."""
    import pickle
    with open(path, "rb") as f:
        st = pickle.load(f)
    if isinstance(st, dict) and st.get("__format__") == "surface_only_probe/v1":
        sc = st["scorer"]
        return SurfaceOnlyProbe(st["keys"],
                                LinearScorer(sc["mu"], sc["sd"], sc["w"], sc["b"]),
                                provenance=st.get("provenance"))
    return load(path)
