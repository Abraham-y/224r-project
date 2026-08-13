"""Known-answer gate for `fragility_core.steering.causal_strength`.

Runs the headline statistic against the two steering runs the ORIGINAL paper
published, whose answers are known independently of this code:

  causal_steering_runA_postRL.jsonl
      post-Goodhart runA. Published: Delta accuracy (probe minus matched random)
      = +0.083 at alpha=1.0, on an INVERTED-U dose-response (+0.041 / +0.083 /
      -0.052 at alpha 0.5 / 1.0 / 2.0). A real, positive, significant effect.

  causal_steering_runA_assertion_control.jsonl
      the inert specificity control (L16 assertion direction, correctness-
      correlated but never optimised). Published: -0.015 at alpha=1.0. A null.

The gate asserts that the estimator (a) recovers the published per-alpha deltas,
(b) reports a positive, significant effect on the first file, (c) reports a
non-significant effect on the second, and (d) is not pinned to zero on a
sign-symmetric alpha grid. It also re-measures the two rejected candidate
statistics on the same data, so the reason the headline is what it is stays
checkable rather than a claim in a docstring.

    cd followup && python experiments/fragility/phase2_causal_status/validate_estimator.py

Exit 0 = gate passes. Exit 1 = a failure, or the ground-truth files are missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fragility_core import paths, steering  # noqa: E402

POSTRL = paths.PAPER_ROOT / "causal_steering_runA_postRL.jsonl"
CONTROL = paths.PAPER_ROOT / "causal_steering_runA_assertion_control.jsonl"

# Published values, from the original paper's steering table. Tolerances are
# generous enough to survive a float representation change and tight enough that
# any real change of aggregation trips them.
PUBLISHED = {
    "postRL": {0.5: +0.041, 1.0: +0.083, 2.0: -0.052},
    "assertion_control": {1.0: -0.015},
}
TOL = 0.002

failures: list[str] = []
checks = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(f"{name}: {detail}")


def load(path: Path) -> list[dict]:
    """Steering rows in the schema `causal_strength` expects.

    The published files store the raw verifier score in `new_score` (0 / 0.1 /
    1.0); correctness is `new_score == 1.0`, which is the same rule
    `run_steering.py` applies when it writes `new_score_correct`.
    """
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            r["new_score_correct"] = float(r["new_score"] == 1.0)
            rows.append(r)
    return rows


def mirror_alphas(rows: list[dict]) -> list[dict]:
    """A sign-symmetric alpha grid built from the published positive-alpha data.

    The published runs swept alpha in {0.5, 1, 2}; the frozen Phase 2 config
    sweeps `both_signs`, i.e. {-2, -1, -0.5, 0.5, 1, 2}. No negative-alpha
    ground truth exists, so this constructs the EVEN case — delta(-a) set equal
    to delta(a) — which is the case where the OLS slope's cancellation is
    exact. It is a property test of the estimator's algebra, NOT a measurement
    of anything about the model.
    """
    out = list(rows)
    for r in rows:
        if r["alpha"] > 0:
            m = dict(r)
            m["alpha"] = -r["alpha"]
            out.append(m)
    return out


def main() -> int:
    for p in (POSTRL, CONTROL):
        if not p.exists():
            print(f"MISSING ground-truth file: {p}")
            return 1

    results = {}

    print("1. per-alpha deltas reproduce the published steering table")
    for name, path in (("postRL", POSTRL), ("assertion_control", CONTROL)):
        rows = load(path)
        agg = steering.causal_strength(rows, probe_key=None)
        results[name] = (rows, agg)
        for alpha, expected in PUBLISHED[name].items():
            key = f"steer_delta_acc_alpha{alpha:g}"
            got = agg.get(key, float("nan"))
            check(
                f"{name} delta at alpha={alpha:g}",
                abs(got - expected) < TOL,
                f"got {got:+.4f}, published {expected:+.3f}",
            )

    print("\n2. HEADLINE causal_strength: real effect on postRL")
    agg = results["postRL"][1]
    cs = agg.get("causal_strength", float("nan"))
    check("causal_strength == the published +0.083", abs(cs - 0.083) < TOL,
          f"{cs:+.4f}")
    check("causal_strength is POSITIVE (the OLS slope was -0.0722 here)", cs > 0,
          f"{cs:+.4f}")
    check("pre-registered dose is alpha=1.0", agg.get("causal_strength_alpha") == 1.0,
          f"{agg.get('causal_strength_alpha')}")
    check("CI excludes zero", agg.get("causal_strength_significant") == 1.0,
          f"95% CI [{agg.get('causal_strength_ci_lo'):+.4f}, "
          f"{agg.get('causal_strength_ci_hi'):+.4f}], p={agg.get('causal_strength_p'):.4f}")
    check("CI is bracketing (lo < est < hi)",
          agg["causal_strength_ci_lo"] < cs < agg["causal_strength_ci_hi"])
    check("clustered on prompts, not rows",
          agg.get("causal_strength_n_clusters") == 100.0,
          f"{agg.get('causal_strength_n_clusters')} clusters over "
          f"{agg.get('causal_strength_n_a')} probe rows")

    print("\n3. HEADLINE causal_strength: null on the inert assertion control")
    aggc = results["assertion_control"][1]
    csc = aggc.get("causal_strength", float("nan"))
    check("causal_strength == the published -0.015", abs(csc - (-0.015)) < TOL,
          f"{csc:+.4f}")
    check("NOT significant (the OLS slope called this null t=+2.89)",
          aggc.get("causal_strength_significant") == 0.0,
          f"95% CI [{aggc.get('causal_strength_ci_lo'):+.4f}, "
          f"{aggc.get('causal_strength_ci_hi'):+.4f}], p={aggc.get('causal_strength_p'):.4f}")
    check("control CI covers zero", aggc["causal_strength_ci_lo"] < 0 < aggc["causal_strength_ci_hi"])

    print("\n4. the two runs are separated by the statistic")
    check("postRL effect exceeds the control's upper CI bound",
          cs > aggc["causal_strength_ci_hi"],
          f"{cs:+.4f} vs control CI hi {aggc['causal_strength_ci_hi']:+.4f}")

    print("\n5. not pinned to zero on a sign-symmetric alpha grid")
    for name in ("postRL", "assertion_control"):
        rows_sym = mirror_alphas(results[name][0])
        agg_sym = steering.causal_strength(rows_sym, probe_key=None)
        base = results[name][1]["causal_strength"]
        check(f"{name} causal_strength unchanged by mirroring",
              abs(agg_sym.get("causal_strength", float("nan")) - base) < 1e-12,
              f"{agg_sym.get('causal_strength'):+.4f} vs {base:+.4f}")
        check(f"{name} grid flagged sign-symmetric",
              agg_sym.get("diag_ols_slope_grid_sign_symmetric") == 1.0)
        check(f"{name} the DIAGNOSTIC slope does collapse to 0 there",
              abs(agg_sym.get("diag_ols_slope", float("nan"))) < 1e-12,
              f"diag_ols_slope = {agg_sym.get('diag_ols_slope'):+.6g}")

    print("\n6. determinism (same seed -> same CI)")
    rerun = steering.causal_strength(results["postRL"][0], probe_key=None)
    check("bootstrap is reproducible",
          rerun["causal_strength_ci_lo"] == agg["causal_strength_ci_lo"]
          and rerun["causal_strength_ci_hi"] == agg["causal_strength_ci_hi"])

    print("\n7. the rejected candidates, re-measured on the same data")
    # Recorded so the docstring's justification is a measurement, not a claim.
    for name in ("postRL", "assertion_control"):
        a = results[name][1]
        print(f"  {name}:")
        print(f"    HEADLINE causal_strength      {a['causal_strength']:+.4f} "
              f"[{a['causal_strength_ci_lo']:+.4f}, {a['causal_strength_ci_hi']:+.4f}] "
              f"p={a['causal_strength_p']:.4f}")
        print(f"    diag max|delta| ({int(a['diag_maxabs_n_alphas_searched'])} alphas) "
              f"{a['diag_maxabs_delta']:.4f} at alpha={a['diag_maxabs_at_alpha']:g} "
              "(unsigned; cannot call a null)")
        t = a["diag_ols_slope"] / a["diag_ols_slope_se"]
        print(f"    diag OLS slope                {a['diag_ols_slope']:+.4f} "
              f"se={a['diag_ols_slope_se']:.4f} "
              f"(dof={int(a['diag_ols_slope_dof'])}) t={t:+.2f}")
    check("diag OLS slope is wrong-signed on postRL (why it was demoted)",
          results["postRL"][1]["diag_ols_slope"] < 0,
          f"{results['postRL'][1]['diag_ols_slope']:+.4f} against a "
          f"{results['postRL'][1]['causal_strength']:+.4f} effect")
    t_ctrl = (results["assertion_control"][1]["diag_ols_slope"]
              / results["assertion_control"][1]["diag_ols_slope_se"])
    check("diag OLS slope is spuriously 'significant' on the control", abs(t_ctrl) > 2.0,
          f"t={t_ctrl:+.2f} on a published null")

    print("\n7b. arm routing on the config's 3-direction rows")
    # `phase2_default.yaml` sweeps directions [probe, assertion, rand]. The two
    # published files each contain only probe/rand, so this is a construction
    # with a known answer rather than a measurement: the assertion arm is set
    # EQUAL to the random arm (so its contrast must be exactly 0) and the probe
    # arm is offset by exactly +0.1 (so the headline must be exactly +0.1). It
    # tests that each name is read off the arm it claims.
    rng = np.random.default_rng(21)
    routed = []
    for p in range(80):
        base = float(rng.random()) * 0.5
        for _ in range(2):
            shared = base
            for a in (-1.0, 1.0):
                routed.append({"prompt_idx": p, "alpha": a, "direction": "rand",
                               "new_score_correct": shared})
                routed.append({"prompt_idx": p, "alpha": a, "direction": "assertion",
                               "new_score_correct": shared})
                routed.append({"prompt_idx": p, "alpha": a, "direction": "probe",
                               "new_score_correct": shared + 0.1})
    ra = steering.causal_strength(routed, probe_key=None)
    check("probe arm -> causal_strength == +0.100",
          abs(ra["causal_strength"] - 0.1) < 1e-12, f"{ra['causal_strength']:+.6f}")
    check("assertion arm -> causal_strength_assertion == 0",
          abs(ra["causal_strength_assertion"]) < 1e-12,
          f"{ra['causal_strength_assertion']:+.6f}")
    check("assertion control is reported non-significant here",
          ra["causal_strength_assertion_significant"] == 0.0)
    check("negative dose reported separately, not pooled",
          ra["causal_strength_negalpha_alpha"] == -1.0
          and abs(ra["causal_strength_negalpha"] - 0.1) < 1e-12,
          f"{ra['causal_strength_negalpha']:+.6f} at alpha="
          f"{ra['causal_strength_negalpha_alpha']:+.1f}")

    print("\n8. flip_rate reports its effect the same way")
    # Synthetic patching rows with a KNOWN effect: within each of 40 prompts,
    # w2c is patched-correct 3/4 vs control 1/4 (effect +0.5) and c2w is
    # patched-correct 0/4 vs control 2/4 (effect -0.5). patch_effect_mean must
    # therefore be exactly +0.5.
    prows = []
    for p in range(40):
        for k in range(4):
            prows.append({"prompt_idx": p, "donor_label": 1, "orig_correct": 0.0,
                          "new_correct": float(k < 3), "control_correct": float(k < 1)})
            prows.append({"prompt_idx": p, "donor_label": 0, "orig_correct": 1.0,
                          "new_correct": 0.0, "control_correct": float(k < 2)})
    fr = steering.flip_rate(prows)
    check("patch_effect_w2c == +0.50", abs(fr["patch_effect_w2c"] - 0.5) < 1e-12,
          f"{fr['patch_effect_w2c']:+.4f}")
    check("patch_effect_c2w == -0.50", abs(fr["patch_effect_c2w"] + 0.5) < 1e-12,
          f"{fr['patch_effect_c2w']:+.4f}")
    check("patch_effect_mean == +0.50", abs(fr["patch_effect_mean"] - 0.5) < 1e-12,
          f"{fr['patch_effect_mean']:+.4f}")
    check("patch_effect_mean carries a clustered CI",
          fr.get("patch_effect_mean_significant") == 1.0,
          f"[{fr.get('patch_effect_mean_ci_lo'):+.4f}, "
          f"{fr.get('patch_effect_mean_ci_hi'):+.4f}]")
    # A genuinely null patching set must NOT come out significant.
    rng = np.random.default_rng(7)
    nrows = []
    for p in range(40):
        for k in range(4):
            for lab in (0, 1):
                nrows.append({"prompt_idx": p, "donor_label": lab,
                              "orig_correct": float(1 - lab),
                              "new_correct": float(rng.random() < 0.3),
                              "control_correct": float(rng.random() < 0.3)})
    frn = steering.flip_rate(nrows)
    check("null patching set is not significant",
          frn.get("patch_effect_mean_significant") == 0.0,
          f"{frn['patch_effect_mean']:+.4f} "
          f"[{frn['patch_effect_mean_ci_lo']:+.4f}, {frn['patch_effect_mean_ci_hi']:+.4f}]")

    print("\n9. the bootstrap really is clustered, and really is paired")
    # Two separate properties, each with a known answer. Testing them on the
    # real data alone does not work: clustering WIDENS an interval (rows in a
    # prompt are correlated, so the effective n is the prompt count) while
    # pairing NARROWS it (prompt difficulty is shared by both arms and cancels
    # in the difference). On the published run the two effects run in opposite
    # directions, so the net width is not a test of either.

    # 9a. CLUSTERING. 50 prompts x 10 rows, each prompt entirely 1 or entirely
    # 0, arms drawn independently. Effective n is 50 per arm, not 500, so the
    # clustered SE must be several times the row-independent SE.
    rng = np.random.default_rng(11)
    cl, vl, cl2, vl2 = [], [], [], []
    for p in range(50):
        a_, b_ = float(rng.random() < 0.5), float(rng.random() < 0.5)
        cl += [f"p{p}"] * 10
        vl += [a_] * 10
        cl2 += [f"p{p}"] * 10
        vl2 += [b_] * 10
    # Disjoint namespaces => clustered but NOT paired, isolating the clustering.
    clustered = steering.cluster_bootstrap_contrast(
        [f"A{c}" for c in cl], vl, [f"B{c}" for c in cl2], vl2
    )
    rowwise = steering.cluster_bootstrap_contrast(
        list(range(500)), vl, list(range(500, 1000)), vl2
    )
    check("clustering widens the CI under intra-cluster correlation",
          clustered["_se"] > 2.0 * rowwise["_se"],
          f"clustered se {clustered['_se']:.4f} vs row-independent {rowwise['_se']:.4f}")

    # 9b. PAIRING. Per-prompt base rate b_p, probe arm = b_p + 0.1 and random
    # arm = b_p on every row. The contrast is exactly +0.1 for ANY set of
    # prompts, so a genuinely paired resample must return a zero-width CI; an
    # unpaired one inherits the (large) spread of b_p.
    rng = np.random.default_rng(12)
    cp_, vp_, cr_, vr_ = [], [], [], []
    for p in range(60):
        b = float(rng.random())
        for _ in range(3):
            cp_.append(f"p{p}")
            vp_.append(b + 0.1)
            cr_.append(f"p{p}")
            vr_.append(b)
    paired = steering.cluster_bootstrap_contrast(cp_, vp_, cr_, vr_)
    unpaired = steering.cluster_bootstrap_contrast(
        [f"A{c}" for c in cp_], vp_, [f"B{c}" for c in cr_], vr_
    )
    check("paired contrast recovers the exact +0.100 offset",
          abs(paired[""] - 0.1) < 1e-12, f"{paired['']:+.6f}")
    check("pairing cancels the shared prompt effect (CI width ~ 0)",
          paired["_ci_hi"] - paired["_ci_lo"] < 1e-9,
          f"paired width {paired['_ci_hi'] - paired['_ci_lo']:.2e} vs unpaired "
          f"{unpaired['_ci_hi'] - unpaired['_ci_lo']:.4f}")

    # 9c. The same three-way decomposition on the real postRL data, reported so
    # the direction of the net effect is on the record rather than assumed.
    rows = results["postRL"][0]
    pr = [r for r in rows if r["direction"] == "probe" and r["alpha"] == 1.0]
    rd = [r for r in rows if r["direction"] == "rand" and r["alpha"] == 1.0]
    vpr = [r["new_score_correct"] for r in pr]
    vrd = [r["new_score_correct"] for r in rd]
    real_unpaired = steering.cluster_bootstrap_contrast(
        [f"A{r['prompt_idx']}" for r in pr], vpr,
        [f"B{r['prompt_idx']}" for r in rd], vrd,
    )
    real_rowwise = steering.cluster_bootstrap_contrast(
        list(range(len(pr))), vpr, list(range(len(pr), len(pr) + len(rd))), vrd
    )
    w_paired = agg["causal_strength_ci_hi"] - agg["causal_strength_ci_lo"]
    print(f"    postRL alpha=1 CI widths: paired+clustered {w_paired:.4f} | "
          f"clustered, arms independent "
          f"{real_unpaired['_ci_hi'] - real_unpaired['_ci_lo']:.4f} | "
          f"row-independent {real_rowwise['_ci_hi'] - real_rowwise['_ci_lo']:.4f}")
    check("pairing narrows relative to the same bootstrap unpaired",
          w_paired < real_unpaired["_ci_hi"] - real_unpaired["_ci_lo"],
          "prompt difficulty is shared across arms, so it cancels in the contrast")

    print("\n10. calibration under a simulated true null matched to the design")
    # The assertion control is ONE draw from the null; it can come out
    # non-significant by luck. This measures the rate. The design is taken from
    # the published run itself: 100 prompts, the empirical per-prompt base rates
    # of its random arm, 2 rollouts each, both arms drawn from the same rate.
    #
    # Reduced from the 4000 x 2000 run whose numbers the `causal_strength`
    # docstring quotes (type-I 0.051 +-0.007, coverage 0.949; power 0.641 at
    # +0.083) so the gate stays a few seconds. At n_sim=400 the Monte Carlo 95%
    # interval on a 0.05 rate is about +-0.021, so the bounds below are wide
    # enough not to flake and tight enough to catch an estimator that is not
    # calibrated at all. If you want the precise number, raise n_sim.
    rand_rows = [r for r in results["postRL"][0] if r["direction"] == "rand"]
    by_prompt: dict[int, list[float]] = {}
    for r in rand_rows:
        by_prompt.setdefault(r["prompt_idx"], []).append(r["new_score_correct"])
    prompts = sorted(by_prompt)
    rates = [float(np.mean(by_prompt[k])) for k in prompts]
    # Real per-prompt row counts at alpha=1 (94 prompts x 2 rollouts, 6 x 1).
    sizes = [
        max(1, sum(1 for r in results["postRL"][0]
                   if r["direction"] == "probe" and r["alpha"] == 1.0
                   and r["prompt_idx"] == k))
        for k in prompts
    ]
    n_sim, n_boot_sim = 400, 1000

    def simulate(effect: float, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        ca, va, cb, vb = [], [], [], []
        for i, p in enumerate(rates):
            for _ in range(sizes[i]):
                ca.append(i)
                va.append(float(rng.random() < min(max(p + effect, 0.0), 1.0)))
                cb.append(i)
                vb.append(float(rng.random() < p))
        return steering.cluster_bootstrap_contrast(
            ca, va, cb, vb, n_boot=n_boot_sim, seed=seed
        )

    sig = cov = 0
    for s in range(n_sim):
        d = simulate(0.0, s)
        sig += d["_significant"]
        cov += float(d["_ci_lo"] <= 0.0 <= d["_ci_hi"])
    t1, cover = sig / n_sim, cov / n_sim
    check("type-I error near the nominal 0.05", 0.02 <= t1 <= 0.10,
          f"{t1:.3f} over {n_sim} simulated null runs (MC 95% band ~+-0.021)")
    check("95% CI coverage near nominal", 0.90 <= cover <= 0.99, f"{cover:.3f}")
    sig_p = sum(simulate(0.083, 10_000 + s)["_significant"] for s in range(n_sim))
    check("has power at the published +0.083 effect size", sig_p / n_sim > 0.40,
          f"{sig_p / n_sim:.3f} — power is only ~0.64 at n=100 prompts, so a "
          "non-significant single run is not evidence of no effect")

    print(f"\n{checks - len(failures)}/{checks} estimator checks pass")
    if failures:
        print("\nFAILURES — the headline statistic is not doing what it claims:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("Estimator validated against both published steering runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
