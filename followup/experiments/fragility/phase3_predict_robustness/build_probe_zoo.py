"""Phase 3 — build the probe zoo at t=0 and compute its pre-hoc features.

No GPU, no RL. Fits 10-15 probes on cached t=0 activations, records the pre-hoc
geometry of each, and pickles them so the RL attack can be pointed at any one.

Every feature here is computed from t=0 ONLY. Nothing may look at a post-RL
checkpoint — the whole H3 claim is that robustness is predictable *in advance*,
and a feature that peeked would make the correlation circular.

Probe kinds:
  linear                 StandardScaler + LogisticRegression(C)
  pca_linear             + PCA(rank) in between: a rank-restricted probe
  mean_diff              difference-of-class-means direction (no fitting)
  mlp                    1x256 ReLU
  ensemble               features concatenated across layers
  adversarial_template   linear, but sample-weighted so that correct and wrong
                         are balanced WITHIN each template-score stratum. The
                         template shortcut earns nothing, so the probe is pushed
                         onto whatever correctness signal is left.
  template_residualised  linear on activations with the template indicators
                         regressed out dimension-by-dimension.

    python experiments/fragility/phase3_predict_robustness/build_probe_zoo.py \
        --config phase3_probe_zoo.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fragility_core import (  # noqa: E402
    activations,
    confounds,
    geometry,
    labels as labels_mod,
    metrics_io,
    paths,
    probes,
    registry,
)


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else paths.PAPER_ROOT / path


# ---------------------------------------------------------------------------
# loading t=0 activations from either cache format
# ---------------------------------------------------------------------------


def load_source(cfg: dict, layers: list[int]) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    """Return {layer: X} and a row-aligned label table for the t=0 snapshot."""
    src = cfg["source_activations"]
    target = src["run_id_or_cache"]

    steps = activations.list_steps(str(target))
    if steps:  # new-style acts cache
        step = min(steps)
        lab = activations.load_labels(str(target), step)
        return {L: activations.load_acts(str(target), step, L) for L in layers}, lab

    # old-style npz cache: {prefix}/{tag}_l{L}_pre_answer.npz + .meta.json
    prefix = _resolve(target) / src["checkpoint_tag"]
    meta_path = Path(f"{prefix}_l{layers[0]}_pre_answer.meta.json")
    if not meta_path.exists():
        raise SystemExit(f"no activations found at {target} (tried acts cache and {meta_path})")
    with open(meta_path) as f:
        meta = json.load(f)
    rows = pd.DataFrame(
        [{"prompt_idx": int(m["prompt_idx"]), "resp_idx": int(m["resp_idx"])} for m in meta]
    )
    lab_tbl = labels_mod.build_label_table(_resolve(src["eval_json"]))
    # validate="one_to_one": a duplicate (prompt_idx, resp_idx) in either frame
    # would silently lengthen `merged`, after which `keep` no longer indexes the
    # cached activation rows it is applied to.
    merged = rows.merge(lab_tbl, on=["prompt_idx", "resp_idx"], how="left",
                        indicator=True, validate="one_to_one")
    keep = (merged["_merge"] == "both").to_numpy()
    clean = labels_mod.clean_prompt_idx()
    if clean is not None:
        keep &= merged["prompt_idx"].isin(clean).to_numpy()

    X_by_layer = {}
    for L in layers:
        with np.load(f"{prefix}_l{L}_pre_answer.npz") as d:
            X_by_layer[L] = d["X"][keep]
    lab = merged.drop(columns=["_merge"])[keep].reset_index(drop=True)
    missing = [c for c in confounds.TEMPLATE_COLS if c not in lab.columns]
    if missing:
        # Without the per-pattern indicators the residualised and adversarial
        # probes silently degenerate into the plain linear probe. Fail loudly.
        raise SystemExit(
            f"label table is missing template indicator columns {missing}; "
            "the H0 probe variants cannot be built from it"
        )
    return X_by_layer, lab


# ---------------------------------------------------------------------------
# probe constructors
# ---------------------------------------------------------------------------


def _template_strata_weights(lab: pd.DataFrame, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Weights that equalise the correct/wrong mass inside each template stratum."""
    ts = lab["template_score"].to_numpy()
    w = np.ones(mask.sum(), dtype=np.float64)
    ts_m, y_m = ts[mask], y[mask]
    for s in np.unique(ts_m):
        in_s = ts_m == s
        for cls in (0, 1):
            sel = in_s & (y_m == cls)
            n = int(sel.sum())
            if n:
                w[sel] = 1.0 / n
    return w * (len(w) / w.sum())


def _residualise_template(
    X: np.ndarray, lab: pd.DataFrame, *, fit_mask: np.ndarray
) -> tuple[np.ndarray, dict | None]:
    """Remove the component of each activation dimension linearly predictable
    from the template indicators.

    The OLS map is fit on `fit_mask` rows ONLY — normally the train split. It
    used to be fit on every row, so the held-out rows were over-cleaned relative
    to an out-of-fold residualisation and the probe's val AUROC was biased down
    (`confounds.residualise` documents the same defect and takes a `fit_mask`
    for exactly this reason; this copy had silently dropped it).

    Returns `(Xr, residualiser)`. `residualiser` carries the fitted design —
    the template column names and `beta` — so the SAME transform can be replayed
    on another checkpoint's activations. Without it a pickled probe fit on
    residualised activations gets scored on RAW ones downstream, which is a
    basis mismatch reported as a result (`transfer_matrix.py` read
    L16_resid_template at 0.749 where the like-for-like figure is 0.606).
    """
    T = confounds.template_matrix(lab)
    if T is None or np.allclose(T.std(axis=0), 0):
        return X, None
    cols = list(confounds.TEMPLATE_COLS)
    Td = np.column_stack([np.ones(len(T)), T])
    fit = np.asarray(fit_mask)
    beta, *_ = np.linalg.lstsq(Td[fit], np.asarray(X, dtype=np.float64)[fit], rcond=None)
    return X - Td @ beta, {"kind": "template_residual", "columns": cols, "beta": beta}


def apply_residualiser(X: np.ndarray, lab: pd.DataFrame, resid: dict) -> np.ndarray:
    """Replay a `_residualise_template` transform on new rows. Raises if the
    label table lacks the columns the map was fit against — a silent fallback to
    raw activations is the bug this function exists to prevent."""
    missing = [c for c in resid["columns"] if c not in lab.columns]
    if missing:
        raise KeyError(
            f"cannot replay the template residualiser: label table is missing "
            f"{missing}. Scoring on raw activations instead would be a basis "
            "mismatch reported as a collapse."
        )
    T = lab[list(resid["columns"])].to_numpy(dtype=np.float64)
    Td = np.column_stack([np.ones(len(T)), T])
    return np.asarray(X, dtype=np.float64) - Td @ np.asarray(resid["beta"], dtype=np.float64)


def build_probe(spec: dict, X_by_layer: dict[int, np.ndarray], lab: pd.DataFrame,
                *, label_rule: str, train_mask: np.ndarray, seed: int):
    """Fit one zoo entry. Returns (model, X_used, note, residualiser|None)."""
    kind = spec["kind"]
    y = lab[label_rule].to_numpy().astype(int)
    C = float(spec.get("C", probes.LINEAR_C))

    if kind == "ensemble":
        X = np.concatenate([X_by_layer[int(L)] for L in spec["layers"]], axis=1)
    else:
        X = X_by_layer[int(spec["layer"])]

    if kind in ("linear", "ensemble"):
        return (probes.make_linear_probe(C=C, seed=seed)
                .fit(X[train_mask], y[train_mask]), X, "", None)

    if kind == "pca_linear":
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        model = Pipeline([
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=int(spec["rank"]), random_state=seed)),
            ("lr", LogisticRegression(C=C, max_iter=probes.LINEAR_MAX_ITER, random_state=seed)),
        ])
        return model.fit(X[train_mask], y[train_mask]), X, f"rank={spec['rank']}", None

    if kind == "mean_diff":
        Xtr, ytr = X[train_mask], y[train_mask]
        mu, sd = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1.0)
        Z = (Xtr - mu) / sd
        w = Z[ytr == 1].mean(0) - Z[ytr == 0].mean(0)
        return probes.DirectionProbe(w / sd, name="mean_diff"), X, "no fitting", None

    if kind == "mlp":
        return probes.make_mlp_probe(seed=seed).fit(X[train_mask], y[train_mask]), X, "", None

    if kind == "adversarial_template":
        model = probes.make_linear_probe(C=C, seed=seed)
        w = _template_strata_weights(lab, y, train_mask)
        model.fit(X[train_mask], y[train_mask], lr__sample_weight=w)
        return model, X, "template-stratum balanced weights", None

    if kind == "template_residualised":
        Xr, resid = _residualise_template(X, lab, fit_mask=train_mask)
        return (probes.make_linear_probe(C=C, seed=seed).fit(Xr[train_mask], y[train_mask]),
                Xr, "template indicators regressed out (map fit on train rows)", resid)

    raise ValueError(f"unknown probe kind {kind!r}")


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    cfg = registry.load_config(args.config)
    seed = int(cfg.get("seed", 0))
    registry.set_all_seeds(seed)
    paths.mkdirs()

    out_dir = Path(args.out_dir) if args.out_dir else paths.RESULTS / "probe_zoo"
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = sorted({int(L) for s in cfg["zoo"] for L in ([s["layer"]] if "layer" in s else s["layers"])})
    X_by_layer, lab = load_source(cfg, layers)
    lab = labels_mod.attach_split(
        lab, val_frac=float(cfg["split"]["val_frac"]), seed=int(cfg["split"]["seed"])
    )
    label_rule = cfg["label_rule"]
    y = lab[label_rule].to_numpy().astype(int)
    train_mask = (lab["split"] == "train").to_numpy()
    val_mask = (lab["split"] == "val").to_numpy()
    print(f"[zoo] t=0 snapshot: {len(lab)} rows, layers {layers}, pos_rate {y.mean():.3f}")

    writer = metrics_io.MetricWriter(cfg["name"], "phase3", seed)

    # How much of the probe's signal lives in the template-predictable subspace,
    # with the placebo and random-projection controls that make the number
    # readable. Run once for the snapshot, not per probe.
    rtest = confounds.residualisation_test(
        X_by_layer[int(cfg["zoo"][0].get("layer", 16))], lab,
        label_rule=label_rule, train_mask=train_mask, val_mask=val_mask, seed=seed,
    )
    print("[zoo] residualisation test (read confounds.residualisation_test docstring):")
    for k, v in rtest.items():
        print(f"[zoo]   {k:36s} {v:.3f}")
        writer.add(k, v, checkpoint_step=0, layer=int(cfg["zoo"][0].get("layer", 16)), arm="t0")

    records = []

    for spec in cfg["zoo"]:
        pid = spec["probe_id"]
        try:
            model, X_used, note, resid = build_probe(
                spec, X_by_layer, lab, label_rule=label_rule, train_mask=train_mask, seed=seed
            )
        except Exception as exc:
            print(f"[zoo] {pid}: FAILED to build ({exc}); no rows written")
            continue

        s_val = probes.probe_scores(model, X_used[val_mask])
        s_all = probes.probe_scores(model, X_used)
        rec = {
            "probe_id": pid,
            "kind": spec["kind"],
            "layer": spec.get("layer", -1),
            # An ensemble spec carries `layers`, not `layer`, so it records -1.
            # Downstream readers must be able to tell "no single layer" from a
            # real layer index rather than inferring it from a sentinel: -1 used
            # to fall through a truthiness check and silently select layer -1.
            "layers": ",".join(str(int(L)) for L in spec["layers"]) if "layers" in spec else "",
            "single_layer": "layer" in spec,
            # A probe fit on TRANSFORMED activations cannot be scored on raw ones.
            # Recorded here so every consumer knows to load the sidecar.
            "requires_transform": "template_residual" if resid else "",
            "note": note,
            "auroc_val_t0": probes.auroc(y[val_mask], s_val),
            "auroc_val_t0_balanced": probes.auroc_balanced(y[val_mask], s_val, seed=seed),
        }

        # Margin: distance from the decision boundary, standardised by the
        # probe's own score spread so it is comparable across the zoo. A probe
        # whose val rows crowd the boundary should be cheaper to push across it.
        margin = probes.standardized_margin(model, X_used[val_mask])
        rec["margin_median"] = float(np.median(margin))
        rec["margin_p10"] = float(np.percentile(margin, 10))

        w = probes.probe_direction(model)
        if w is not None and w.size == X_used.shape[1]:
            rec["participation_ratio"] = geometry.participation_ratio(w)
            rec["projection_variance_fraction"] = geometry.projection_variance_fraction(X_used, w)
            rec.update({k: v for k, v in geometry.sep(X_used[val_mask], y[val_mask], w).items()
                        if k == "sep"})
            rec["sep_t0"] = rec.pop("sep", float("nan"))
        else:
            # MLP and PCA probes have no single input-space direction. Absent,
            # not zero — H3 features that do not exist for a probe stay missing.
            rec["participation_ratio"] = float("nan")
            rec["projection_variance_fraction"] = float("nan")
            rec["sep_t0"] = float("nan")

        conf = confounds.confound_suite(
            lab, s_all, label_rule=label_rule,
            val_mask=val_mask, train_mask=train_mask, seed=seed,
        )
        rec["auroc_cond_template_t0"] = conf.get("auroc_frozen_cond_template", float("nan"))
        rec["corr_template_t0"] = conf.get("corr_probe_template_spearman", float("nan"))
        # Filled in by the Phase-2 steering job at t=0; absent until then.
        rec["causal_strength_t0"] = float("nan")

        with open(out_dir / f"{pid}.pkl", "wb") as f:
            pickle.dump(model, f)
        if resid is not None:
            # Plain arrays in an .npz, NOT a pickled custom object: the loader
            # (transfer_matrix.py) must not need this module importable.
            np.savez(
                out_dir / f"{pid}.residualiser.npz",
                kind=np.array(resid["kind"]),
                columns=np.array(resid["columns"]),
                beta=np.asarray(resid["beta"], dtype=np.float64),
            )
        records.append(rec)
        for k, v in rec.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                writer.add(k, float(v), checkpoint_step=0,
                           layer=int(rec["layer"]), arm="t0", probe_id=pid)

        print(
            f"[zoo] {pid:<22} auroc={rec['auroc_val_t0']:.3f}  "
            f"cond_template={rec['auroc_cond_template_t0']:.3f}  "
            f"sep={rec['sep_t0']:.2f}  PR={rec['participation_ratio']:.1f}  "
            f"margin_p10={rec['margin_p10']:.3f}"
        )

    table = pd.DataFrame(records)
    metrics_io.write_table(table, out_dir / "prehoc_features.csv", index=False)
    writer.flush()
    registry.register(cfg, phase="phase3", n_probes=len(records), out_dir=str(out_dir))

    print(f"\n[zoo] {len(records)}/{len(cfg['zoo'])} probes built -> {out_dir}")
    print("[zoo] NEXT: run transfer_matrix.py before paying for any RL run.")
    print("[zoo] If the transfer matrix shows no spread across probes, H3 is dead for free.")


if __name__ == "__main__":
    main()
