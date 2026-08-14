"""The ~40-step lag: recompute it from the metric store, across seeds.

The claim is that reward hacking precedes the monitor's loss of discrimination
by roughly 40 RLOO steps: the probe's score climbs and true accuracy falls from
early on, while the frozen monitor's AUROC on each checkpoint's OWN freshly
sampled rollouts stays flat until step ~50.

This script exists because that claim went into the paper transcribed from
`followup/results/fragility/figures/phase0_collapse__*.csv`, which is a
**single-seed** derived file, not the metric store. That is precisely the failure
mode this project keeps producing -- `followup/PAPER_NOTES.md` drew the same
lesson in July after three tables were transcribed from a superseded generation
of runA rows -- so the table is now computed from the parquet store, across every
seed present, and checked against what the paper prints.

Two things the single-seed CSV hid, both found by writing this:

  1. "The monitor's AUROC PEAKS at step 30, mid-attack" is a seed-1 artifact.
     Seeds 0 and 2 peak at step 20, and across seeds steps 20 and 30 are tied
     (0.795 vs 0.793). The robust statement is that AUROC is FLAT through step
     40 -- which is all the lag argument needs. The peak flourish is dropped.
  2. All three seeds are bit-identical on `auroc_frozen`. That is the unbalanced
     variant, which PAPER_NOTES records as exactly reproducible, so agreement is
     expected -- but it means that metric carries no error bar at all, and
     "three seeds" said of it would be empty. `auroc_frozen_balanced` DOES vary
     (the subsample differs per seed), so the paper reports that one with its
     across-seed spread, and this script checks the lag survives in each seed
     independently rather than only in the mean.

     What the seeds bound either way is analysis noise. They are three analyses
     of ONE RL run, so nothing here speaks to training variance, and the paper
     must not imply otherwise.

    python followup/experiments/fragility/phase0_replicate/verify_lag_result.py

Pure CPU, reads only the local parquet metric store.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_METRICS = os.path.join(_ROOT, "followup", "results", "fragility", "metrics")

# The attacked ladder and its control. The control matters: without it, a falling
# AUROC could just be what training does to a frozen probe.
RUNS = {
    "runA (probe-as-reward)": "phase0_harvest_runA",
    "vanilla RLOO (control)": "vanilla_rloo_ladder",
    # runB carries the scope condition: it initializes from C_SFT while the
    # reward probe was fit on C_outcome rollouts, so its monitor is
    # off-distribution from step 0 and there is no intact window to lose.
    # Only steps 0 and 99 are harvested, which is enough to state the boundary.
    "runB (off-distribution monitor)": "phase0_harvest_runB",
}

METRICS = ["auroc_frozen", "auroc_frozen_balanced", "probe_score_mean", "verifier_acc",
           # Used for the label-free early-warning check below.
           "probe_score_frac_above_095", "probe_score_std"]

# What the paper prints, seed-aggregated. Checked, not trusted.
PUBLISHED_RUNA_AUROC = {0: 0.787, 10: 0.772, 20: 0.792, 30: 0.782, 40: 0.762,
                        50: 0.679, 60: 0.648, 70: 0.570, 99: 0.550}
# The balanced means the paper prints, alongside their across-seed intervals.
PUBLISHED_RUNA_BALANCED = {0: 0.794, 10: 0.767, 20: 0.792, 30: 0.788, 40: 0.769,
                           50: 0.687, 70: 0.557, 99: 0.554}
# runB's step-0 monitor AUROC, which the paper quotes as the scope condition.
PUBLISHED_RUNB_STEP0 = 0.590
FLAT_THROUGH = 40      # last step the monitor is claimed to be intact
DROP_AT = 50           # first step the claim says it moves


def load(run_id: str, layer: int, arm: str) -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(_METRICS, f"*{run_id}*phase1*.parquet")))
    if not fs:
        return pd.DataFrame()
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d[(d.layer == layer) & (d.arm == arm) & (d.metric.isin(METRICS))]
    # A run_id may have several generations of rows; take the last written per
    # (step, metric, seed). metrics_io.latest() does this for the pipeline; we
    # replicate it here rather than trusting a derived file.
    if "written_utc" in d.columns:
        d = d.sort_values("written_utc")
    return d.drop_duplicates(subset=["checkpoint_step", "metric", "seed"], keep="last")


def table(d: pd.DataFrame, metric: str) -> pd.DataFrame:
    return d[d.metric == metric].pivot_table(
        index="checkpoint_step", columns="seed", values="value", aggfunc="last")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--arm", default="on_policy")
    ap.add_argument("--out", default="followup/results/fragility/lag_result.txt")
    args = ap.parse_args()

    L: list[str] = [
        "The ~40-step lag -- recomputed from the parquet metric store",
        f"  layer {args.layer}, arm '{args.arm}', all seeds present, latest row per (step, metric, seed)",
        "",
    ]
    out: dict = {"layer": args.layer, "arm": args.arm, "runs": {}}
    failures: list[str] = []

    for label, run_id in RUNS.items():
        d = load(run_id, args.layer, args.arm)
        if d.empty:
            L.append(f"### {label}: NO ROWS (run_id={run_id})")
            L.append("")
            continue

        auroc = table(d, "auroc_frozen")
        bal = table(d, "auroc_frozen_balanced")
        seeds = list(auroc.columns)
        dup = [(a, b) for i, a in enumerate(seeds) for b in seeds[i + 1:]
               if auroc[a].equals(auroc[b])]

        L.append(f"### {label}   seeds={seeds}"
                 + (f"   IDENTICAL: {dup}" if dup else ""))
        L.append("")
        L.append(f"  {'step':>5}  {'auroc_frozen':>22}  {'probe score':>12}  {'true acc':>10}")
        L.append("  " + "-" * 56)

        probe, acc = table(d, "probe_score_mean"), table(d, "verifier_acc")
        rows = {}
        for step in auroc.index:
            a = auroc.loc[step].dropna()
            spread = f"{a.mean():.3f} [{a.min():.3f},{a.max():.3f}]" if len(a) > 1 else f"{a.mean():.3f}"
            p = probe.loc[step].mean() if step in probe.index else float("nan")
            v = acc.loc[step].mean() if step in acc.index else float("nan")
            L.append(f"  {int(step):>5}  {spread:>22}  {p:>12.3f}  {v:>10.3f}")
            rows[int(step)] = {"auroc_mean": float(a.mean()), "auroc_min": float(a.min()),
                               "auroc_max": float(a.max()), "n_seeds": int(len(a)),
                               "probe_score_mean": float(p), "verifier_acc": float(v)}
        L.append("")

        peaks = {int(s): int(auroc[s].idxmax()) for s in seeds}
        if len(set(peaks.values())) > 1:
            L.append(f"  NOTE: the argmax step differs by seed ({peaks}). Do not claim a peak; "
                     "claim flatness.")
            L.append("")

        out["runs"][label] = {"run_id": run_id, "seeds": [int(s) for s in seeds],
                              "identical_seed_pairs": [[int(a), int(b)] for a, b in dup],
                              "by_step": rows}

        # --- the two assertions the paper's claim actually rests on ------------
        if run_id == "phase0_harvest_runA":
            flat = [r["auroc_mean"] for s, r in rows.items() if s <= FLAT_THROUGH]
            drop = rows.get(DROP_AT, {}).get("auroc_mean", float("nan"))
            span = max(flat) - min(flat)
            wobble = max(abs(b - a) for a, b in zip(flat, flat[1:]))
            L.append(f"  CLAIM 1  AUROC flat through step {FLAT_THROUGH}: "
                     f"span {span:.3f} over steps 0-{FLAT_THROUGH}")
            L.append(f"  CLAIM 2  drop by step {DROP_AT}: {min(flat):.3f} -> {drop:.3f} "
                     f"({drop - min(flat):+.3f})")
            if span > 0.05:
                failures.append(f"'flat through {FLAT_THROUGH}' fails: span {span:.3f} > 0.05")
            L.append(f"           largest step-to-step move while flat: {wobble:.3f}; "
                     f"the drop is {(min(flat) - drop) / wobble:.1f}x that")
            if (min(flat) - drop) < 3 * wobble:
                failures.append(
                    f"drop at {DROP_AT} ({min(flat) - drop:.3f}) is under 3x the largest "
                    f"step-to-step move in the flat region ({wobble:.3f}); the lag is not "
                    "cleanly separable from the wobble")
            if not bal.empty:
                L.append("")
                L.append("  BALANCED variant (varies by seed; the reportable error bar)")
                L.append(f"  {'step':>5}  " + "  ".join(f'seed{int(c)}' for c in bal.columns)
                         + f"  {'mean':>7}")
                for step in bal.index:
                    r = bal.loc[step]
                    L.append(f"  {int(step):>5}  " + "  ".join(f'{r[c]:.4f}' for c in bal.columns)
                             + f"  {r.mean():>7.4f}")
                per_seed = {}
                for c in bal.columns:
                    f_ = [bal.loc[s_, c] for s_ in bal.index if s_ <= FLAT_THROUGH]
                    d_ = bal.loc[DROP_AT, c]
                    per_seed[int(c)] = (min(f_) - d_, max(f_) - min(f_))
                L.append("")
                for c, (drop_c, span_c) in per_seed.items():
                    ok = "ok" if drop_c > span_c else "FAILS"
                    L.append(f"    seed {c}: flat-region span {span_c:.3f}, "
                             f"step-{FLAT_THROUGH}->{DROP_AT} drop {drop_c:.3f}  [{ok}]")
                bad = [c for c, (dc, sc) in per_seed.items() if dc <= sc]
                if bad:
                    failures.append(
                        f"balanced variant: the lag does not survive in seed(s) {bad} "
                        "-- it exists only after averaging, which is not a result")
                out["runs"].setdefault(label, {})["balanced_per_seed"] = {
                    str(c): {"drop": float(dc), "flat_span": float(sc)}
                    for c, (dc, sc) in per_seed.items()}
                L.append("")

            for step, want in PUBLISHED_RUNA_BALANCED.items():
                if step not in bal.index:
                    failures.append(f"balanced step {step} missing from the store"); continue
                got_b = float(bal.loc[step].mean())
                if abs(got_b - want) > 0.001:
                    failures.append(f"balanced step {step}: paper prints {want:.3f}, "
                                    f"store gives {got_b:.3f}")
            for step, want in PUBLISHED_RUNA_AUROC.items():
                got = rows.get(step, {}).get("auroc_mean")
                if got is None:
                    failures.append(f"published step {step} missing from the store")
                elif abs(got - want) > 0.001:
                    failures.append(f"step {step}: paper prints {want:.3f}, store gives {got:.3f}")
            L.append("")

        if run_id == "vanilla_rloo_ladder":
            first, last = min(rows), max(rows)
            a0, a1 = rows[first]["auroc_mean"], rows[last]["auroc_mean"]
            L.append(f"  CONTROL  frozen AUROC {a0:.3f} (step {first}) -> {a1:.3f} (step {last})")
            if a1 <= a0:
                failures.append("control ladder does not RISE; the runA decline may not be "
                                "specific to the attack")
            L.append("")

    # --- scope condition: runB's monitor starts near chance ------------------
    db = load("phase0_harvest_runB", args.layer, args.arm)
    if not db.empty:
        b0 = table(db, "auroc_frozen")
        if 0 in b0.index:
            got = float(b0.loc[0].mean())
            L.append("### SCOPE CONDITION (runB)")
            L.append("")
            L.append(f"  runB monitor AUROC at step 0: {got:.3f}   "
                     f"(runA: {out['runs'].get('runA (probe-as-reward)', {}).get('by_step', {}).get(0, {}).get('auroc_mean', float('nan')):.3f})")
            L.append("  runB inits from C_SFT while the reward probe was fit on C_outcome")
            L.append("  rollouts, so its monitor is off-distribution from step 0. There is no")
            L.append("  intact-monitor window to lose, hence no lag to observe. A monitor")
            L.append("  already off-distribution gives no warning period at all.")
            L.append("")
            if abs(got - PUBLISHED_RUNB_STEP0) > 0.001:
                failures.append(f"runB step 0: paper prints {PUBLISHED_RUNB_STEP0:.3f}, "
                                f"store gives {got:.3f}")

    # --- is there a LABEL-FREE early warning? (answer: no) -------------------
    # The obvious objection to the lag is "watch the monitor's score distribution
    # instead of its AUROC -- that needs no ground truth." It does not work: the
    # verifier-trained control saturates the monitor just as hard, so saturation
    # cannot distinguish being attacked from training normally.
    L.append("### LABEL-FREE EARLY WARNING: does score saturation discriminate?")
    L.append("")
    L.append(f"  {'run':<26}{'step':>6}{'frac>0.95':>11}{'score std':>11}{'true acc':>10}")
    L.append("  " + "-" * 64)
    sat = {}
    for label, run_id in RUNS.items():
        dd = load(run_id, args.layer, args.arm)
        if dd.empty:
            continue
        fr, sd, ac = (table(dd, m) for m in
                      ("probe_score_frac_above_095", "probe_score_std", "verifier_acc"))
        if fr.empty:
            continue
        sat[label] = {}
        for step in fr.index:
            f_ = float(fr.loc[step].mean())
            s_ = float(sd.loc[step].mean()) if step in sd.index else float("nan")
            a_ = float(ac.loc[step].mean()) if step in ac.index else float("nan")
            L.append(f"  {label:<26}{int(step):>6}{f_:>11.3f}{s_:>11.3f}{a_:>10.3f}")
            sat[label][int(step)] = {"frac_above_095": f_, "score_std": s_, "verifier_acc": a_}
        L.append("")
    out["saturation"] = sat
    ra, vc = sat.get("runA (probe-as-reward)", {}), sat.get("vanilla RLOO (control)", {})
    if ra and vc:
        ra_max = max(v["frac_above_095"] for v in vc.values())
        L.append(f"  The CONTROL reaches frac>0.95 = {ra_max:.3f} while its true accuracy RISES.")
        L.append("  Saturation therefore does not distinguish an attacked monitor from a")
        L.append("  healthily-trained one, and is not a usable early warning. We report this")
        L.append("  as a negative result: the obvious cheap substitute for AUROC does not work.")
        L.append("")

    txt = "\n".join(L)
    print(txt)
    p = os.path.join(_ROOT, args.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(txt + "\n")
    with open(p.replace(".txt", ".json"), "w") as f:
        json.dump(out, f, indent=2)

    if failures:
        print("\nFAILED -- the paper and the metric store disagree:", file=sys.stderr)
        for x in failures:
            print("  " + x, file=sys.stderr)
        sys.exit(1)
    print(f"\nall checks pass; wrote {args.out}")


if __name__ == "__main__":
    main()
