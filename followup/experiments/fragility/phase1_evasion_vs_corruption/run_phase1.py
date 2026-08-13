"""Phase 1 — evasion (H1) vs corruption (H2) vs confound (H0).

Reads only the activation cache. Never loads a model.

For every (arm, layer, checkpoint step, seed) it writes:

  auroc_frozen                 original probe, t=0 weights
  auroc_retrained_linear       fresh linear probe on this checkpoint
  auroc_retrained_mlp          fresh 256-unit MLP on this checkpoint
  evasion_gap                  retrained_linear - frozen   <- the headline
  nonlinear_gain               mlp - retrained_linear
  dir_drift_{input,whitened}   cos(v0, v_t)
  sep                          separation along the FIXED t=0 direction
  cka_vs_step0                 fixed-text arm only
  auroc_template               H0: template score alone
  auroc_frozen_cond_template   H0: frozen probe with the template held fixed
  corr_probe_template_*        H0: does the probe track the template?
  verifier_acc, mean_blocks, ... descriptive context

A checkpoint with no cached snapshot produces NO ROWS. Missing is missing.

    python experiments/fragility/phase1_evasion_vs_corruption/run_phase1.py \
        --config phase1_control_vanilla.yaml --run_id vanilla_rloo_ladder
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

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


def run(cfg: dict, run_id: str, *, arm: str, fit_mlp: bool, seeds: list[int]) -> None:
    probe_cfg = cfg["probe"]
    probe_id = probe_cfg.get("id", "frozen")
    frozen, prov = probes.load_frozen_probe(_resolve(probe_cfg["pipeline"]))
    v0 = probes.probe_direction(frozen)
    if v0 is None:
        raise ValueError("frozen probe exposes no linear direction; geometry metrics need one")
    print(f"[phase1] frozen probe: {probe_id}  d={v0.size}  provenance={prov}")

    label_rule = cfg["label_rule"]
    split_cfg = cfg.get("split", {})
    val_frac = float(split_cfg.get("val_frac", 0.3))
    split_seed = int(split_cfg.get("seed", 0))
    min_rows = int(cfg.get("min_rows_per_snapshot", 0))
    C = float(cfg.get("retrain", {}).get("linear_C", probes.LINEAR_C))
    do_confounds = bool(cfg.get("confounds", {}).get("enabled", True))
    cka_cfg = cfg.get("cka", {}) or {}
    do_cka = bool(cka_cfg.get("enabled", False)) and arm == cka_cfg.get("arm", "fixed_text")
    cka_ref_step = int(cka_cfg.get("reference_step", 0))

    probe_layer = int(probe_cfg.get("layer", -1))
    steps = activations.list_steps(run_id)
    if not steps:
        raise SystemExit(f"no cached snapshots under {activations.run_dir(run_id)}")
    cached_layers = activations.list_layers(run_id, steps[0])
    layers = [L for L in cfg["layers"] if L in cached_layers]
    skipped_layers = [L for L in cfg["layers"] if L not in cached_layers]
    if skipped_layers:
        print(f"[phase1] layers not in cache, skipped (no rows written): {skipped_layers}")
    print(f"[phase1] run_id={run_id} arm={arm} steps={steps} layers={layers} seeds={seeds}")

    for seed in seeds:
        registry.set_all_seeds(seed)
        w1 = metrics_io.MetricWriter(run_id, "phase1", seed)
        w0 = metrics_io.MetricWriter(run_id, "phase0", seed)

        for layer in layers:
            ref_X = None
            if do_cka:
                try:
                    ref_X = activations.load_acts(run_id, cka_ref_step, layer)
                except FileNotFoundError:
                    print(f"[phase1] L{layer}: no step-{cka_ref_step} snapshot; CKA skipped")

            for step in steps:
                try:
                    X, lab = activations.load_snapshot(run_id, step, layer)
                except FileNotFoundError:
                    print(f"[phase1] step {step} L{layer}: absent, no rows written")
                    continue
                if len(lab) < min_rows:
                    # Under-powered, but still measured: the numbers are written
                    # as computed and flagged, not NaN'd. (An earlier version of
                    # this message claimed they were NaN'd; they never were.)
                    print(
                        f"[phase1] step {step} L{layer}: {len(lab)} rows < min {min_rows}; "
                        "metrics are computed but UNDER-POWERED"
                    )
                    w1.add("under_powered", 1.0, checkpoint_step=step, layer=layer, arm=arm)
                lab = labels_mod.attach_split(lab, val_frac=val_frac, seed=split_seed)
                val_mask = (lab["split"] == "val").to_numpy()
                train_mask = (lab["split"] == "train").to_numpy()
                y = lab[label_rule].to_numpy().astype(int)

                # The frozen probe is only interpretable in its own layer's
                # basis; bracketing layers get retrained metrics only.
                is_probe_layer = layer == probe_layer

                # --- probes ---------------------------------------------------
                m, fitted = probes.probe_suite(
                    X, lab, frozen_probe=frozen, label_rule=label_rule,
                    seed=seed, C=C, fit_mlp=fit_mlp, include_frozen=is_probe_layer,
                )
                w1.add_many(m, checkpoint_step=step, layer=layer, arm=arm, probe_id=probe_id)

                # --- geometry -------------------------------------------------
                g: dict[str, float] = {}
                g.update(geometry.activation_stats(X))
                vt = probes.probe_direction(fitted["linear"]) if fitted["linear"] else None
                if vt is not None:
                    g["participation_ratio_t"] = geometry.participation_ratio(vt)
                if is_probe_layer:
                    g.update(geometry.sep(X[val_mask], y[val_mask], v0))
                    g["projection_variance_fraction"] = geometry.projection_variance_fraction(X, v0)
                    g["participation_ratio_0"] = geometry.participation_ratio(v0)
                    if vt is not None:
                        g.update(geometry.dir_drift(v0, vt, act_std=geometry.per_dim_std(X)))
                        # A raw cosine between two 896-d logistic directions is
                        # small even for two fits on the SAME data, so it is
                        # normalised by that refit-noise floor before use.
                        # Normalise a size-MATCHED drift by a size-matched floor:
                        # both operands fit on half the train prompts. Dividing a
                        # full-fit cosine by a half-fit floor inflated the ratio
                        # ~1.5x and made drift look milder than it is.
                        base_w = m.get("dir_drift_self_baseline_whitened", float("nan"))
                        half = fitted.get("half_direction")
                        if half is not None and base_w == base_w and base_w > 0:
                            matched = geometry.dir_drift(
                                v0, half, act_std=geometry.per_dim_std(X)
                            )
                            g["dir_drift_matched"] = matched["dir_drift_whitened"]
                            g["dir_drift_normalized"] = matched["dir_drift_whitened"] / base_w
                if ref_X is not None:
                    if ref_X.shape[0] == X.shape[0]:
                        g["cka_vs_step0"] = geometry.linear_cka(X, ref_X)
                    else:
                        # Row correspondence is what makes CKA meaningful; without
                        # it the number would be fiction. Left absent on purpose.
                        print(
                            f"[phase1] step {step} L{layer}: CKA skipped "
                            f"({X.shape[0]} vs {ref_X.shape[0]} rows, no correspondence)"
                        )
                    angles = geometry.subspace_principal_angles(X, ref_X, k=10)
                    g["subspace_cos_mean_top10"] = float(np.mean(angles))
                w1.add_many(g, checkpoint_step=step, layer=layer, arm=arm, probe_id=probe_id)

                # --- H0 confound ----------------------------------------------
                if do_confounds and is_probe_layer:
                    s_all = probes.probe_scores(frozen, X)
                    c = confounds.confound_suite(
                        lab, s_all, label_rule=label_rule,
                        val_mask=val_mask, train_mask=train_mask, seed=seed,
                    )
                    w1.add_many(c, checkpoint_step=step, layer=layer, arm=arm, probe_id=probe_id)

                # --- descriptive ----------------------------------------------
                desc = {
                    "verifier_acc": float(lab[label_rule].mean()),
                    "verifier_acc_first_block": float(lab["first_block"].mean()),
                    "verifier_acc_last_block": float(lab["last_block"].mean()),
                    "mean_blocks": float(lab["n_blocks"].mean()),
                    "mean_resp_chars": float(lab["resp_chars"].mean()),
                    "n_rows": float(len(lab)),
                }
                w1.add_many(desc, checkpoint_step=step, layer=layer, arm=arm, probe_id=probe_id)
                if is_probe_layer:
                    w0.add_many(
                        {**desc, "auroc_frozen": m["auroc_frozen"],
                         "probe_score_mean": m["probe_score_mean"]},
                        checkpoint_step=step, layer=layer, arm=arm, probe_id=probe_id,
                    )

                nan = float("nan")
                tag = "probe-layer" if is_probe_layer else "bracket    "
                print(
                    f"[phase1] seed{seed} L{layer:<2} step {step:>4} {tag}  "
                    f"frozen={m.get('auroc_frozen', nan):.3f}  "
                    f"lin={m['auroc_retrained_linear']:.3f}  "
                    f"mlp={m.get('auroc_retrained_mlp', nan):.3f}  "
                    f"gap={m.get('evasion_gap', nan):+.3f}  "
                    f"drift={g.get('dir_drift_whitened', nan):+.3f}"
                    f"/{m.get('dir_drift_self_baseline', nan):+.3f}  "
                    f"sep={g.get('sep', nan):+.2f}  acc={desc['verifier_acc']:.3f}"
                )

        p1 = w1.flush()
        p0 = w0.flush()
        print(f"[phase1] seed {seed}: wrote {p1}")
        if p0:
            print(f"[phase1] seed {seed}: wrote {p0}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_id", required=True, help="activation-cache run_id to analyse")
    ap.add_argument("--arm", default=None, help="override the arm tag (on_policy | fixed_text)")
    ap.add_argument("--seeds", default=None, help="comma-separated override of cfg.seeds")
    ap.add_argument("--no_mlp", action="store_true", help="skip the MLP probe (fast iteration)")
    args = ap.parse_args()

    cfg = registry.load_config(args.config)
    paths.mkdirs()
    seeds = (
        [int(s) for s in args.seeds.split(",")]
        if args.seeds
        else [int(s) for s in cfg.get("seeds", [cfg.get("seed", 0)])]
    )
    arms = [args.arm] if args.arm else list(cfg.get("arms", ["on_policy"]))
    registry.register(cfg, phase="phase1", analysed_run_id=args.run_id, seeds=seeds, arms=arms)

    cached = activations.list_steps(args.run_id)
    for arm in arms:
        # The arm is a property of how the cache was built; a cache holds one arm.
        manifest = activations.load_manifest(args.run_id, cached[0] if cached else None)
        cache_arm = manifest.get("arm")
        if cache_arm and cache_arm != arm:
            print(f"[phase1] cache {args.run_id} is arm={cache_arm}; skipping arm={arm}")
            continue
        run(cfg, args.run_id, arm=arm, fit_mlp=not args.no_mlp and
            bool(cfg.get("retrain", {}).get("fit_mlp", True)), seeds=seeds)


if __name__ == "__main__":
    main()
