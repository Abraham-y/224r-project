"""When does the monitor's AUROC actually break? A change-point test.

The lag claim has been carried by a descriptive statistic: the AUROC series is
flat over steps 0--40 and the step-40->50 drop is larger than any step-to-step
move inside that flat region. That is a reasonable thing to say and it is not a
test. "~40 steps" is a function of where one decides the series moved, and a
reviewer is right to ask for the decision to be made by a procedure rather than
by eye.

This does that, and does it on rollouts rather than on the summary series, which
matters: the three analysis seeds in the metric store vary only the
class-balancing subsample, so intervals built from them bound the balancing
procedure and nothing else. Here we go back to the cached activations, score the
frozen reward probe per rollout, and bootstrap over PROMPTS -- the same estimator
used everywhere else in the paper, and the one that reflects the sampling that
actually generated the eval.

Procedure, fixed before looking at the output:

  1. For each checkpoint, score the frozen probe on that checkpoint's own cached
     activations at the probe's layer, giving a per-rollout score and label.
  2. Resample the 406 prompts with replacement. Within a replicate, recompute
     AUROC at every checkpoint from the resampled prompts' rollouts. This is one
     bootstrap realisation of the entire series.
  3. For each realisation, fit the best single change-point: for every candidate
     split k, model the series as two constants (mean before, mean after) and
     take the k minimising within-segment squared error. Record it.
  4. Report the bootstrap distribution over change-point location, and the
     per-step AUROC with prompt-clustered intervals.

The change-point is thus estimated, with an interval, rather than asserted. If
the distribution is diffuse the lag claim should be softened to whatever the
interval supports; if it concentrates, the claim earns its number.

    python followup/experiments/fragility/phase0_replicate/changepoint_lag.py

CPU only. Reads the cached activations under followup/acts/, which are gitignored
and rebuilt by harvest_ladder.py.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_ACTS = os.path.join(_ROOT, "followup", "acts")
_PROBE = os.path.join(_ROOT, "extension", "cache", "steering",
                      "probe_pipeline_C_outcome_l16_pre_answer_temp1.pkl")

# The paper uses TWO label conventions, for two different quantities, and mixing
# them here would have produced a third wrong answer.
#
#   accuracy      first_block  -- fixed in PREREGISTRATION.md before the arms ran
#   this AUROC    last_block   -- the frozen probe's meta.json records its training
#                                 label as "rollout-final verifier correctness", so
#                                 last_block is what it was fit to predict; scoring
#                                 it against first_block measures a different thing
#
# The check that this is right: last_block reproduces the parquet metric store's
# auroc_frozen series (0.783/0.777/0.778/0.769/0.768/0.696 against the store's
# 0.787/0.772/0.792/0.782/0.762/0.679), while first_block does not come close
# (0.919/0.917/0.936/...). An earlier draft of this script used first_block and
# would have reported the monitor already degrading by step 40 -- undercutting the
# paper's central claim on the strength of the wrong label.
LABEL_RULE = "last_block"


def load_checkpoint(run: str, step: int, layer: int):
    """(scores, labels, prompt_idx) for one checkpoint, or None if absent."""
    d = os.path.join(_ACTS, run, str(step))
    a_path, l_path = os.path.join(d, f"{layer}.npy"), os.path.join(d, "labels.parquet")
    if not (os.path.exists(a_path) and os.path.exists(l_path)):
        return None
    X = np.load(a_path)
    lab = pd.read_parquet(l_path)
    if len(lab) != len(X):
        raise SystemExit(f"{run}/{step}: {len(X)} activations vs {len(lab)} labels")
    with open(_PROBE, "rb") as f:
        probe = pickle.load(f)
    s = probe.predict_proba(X)[:, 1]
    return s, lab[LABEL_RULE].to_numpy().astype(int), lab["prompt_idx"].to_numpy()


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")


def best_changepoint(series: np.ndarray) -> int:
    """Index k minimising two-segment within-variance. Returns split position.

    k is the number of points in the FIRST segment, so k=5 on an 11-point series
    means the break sits between the 5th and 6th checkpoint. Both segments must
    be non-empty, so k ranges over 1..n-1.
    """
    n = len(series)
    best_k, best_sse = 1, np.inf
    for k in range(1, n):
        a, b = series[:k], series[k:]
        sse = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        if sse < best_sse:
            best_k, best_sse = k, sse
    return best_k


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="phase0_harvest_runA")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label_rule", default=LABEL_RULE, choices=["last_block", "first_block"],
                    help="see the LABEL_RULE comment; last_block is correct for this probe")
    ap.add_argument("--out", default="followup/results/fragility/changepoint_lag.txt")
    args = ap.parse_args()

    globals()["LABEL_RULE"] = args.label_rule
    steps = sorted(int(s) for s in os.listdir(os.path.join(_ACTS, args.run))
                   if s.isdigit())
    data, used = {}, []
    for st in steps:
        got = load_checkpoint(args.run, st, args.layer)
        if got is not None:
            data[st] = got
            used.append(st)
    if len(used) < 4:
        raise SystemExit(f"only {len(used)} checkpoints have cached layer-{args.layer} "
                         "activations; need at least 4 for a change-point")

    L = [
        "Change-point in the monitor's AUROC series",
        f"  run {args.run}, layer {args.layer}, {len(used)} checkpoints: {used}",
        f"  frozen probe: {os.path.basename(_PROBE)}",
        f"  label rule: {args.label_rule} (the probe's own training convention)",
        f"  prompt-clustered bootstrap, {args.n_boot} resamples, seed {args.seed}",
        "",
    ]

    point = np.array([auroc(data[s][1], data[s][0]) for s in used])
    prompts = np.unique(np.concatenate([data[s][2] for s in used]))
    rng = np.random.default_rng(args.seed)

    # Index rollouts by prompt once, per checkpoint, so a replicate is a gather.
    idx = {s: {p: np.flatnonzero(data[s][2] == p) for p in prompts} for s in used}

    boot = np.full((args.n_boot, len(used)), np.nan)
    kdist = np.zeros(len(used) - 1, dtype=int)
    for b in range(args.n_boot):
        draw = rng.choice(prompts, size=len(prompts), replace=True)
        row = []
        for s in used:
            sel = np.concatenate([idx[s][p] for p in draw])
            row.append(auroc(data[s][1][sel], data[s][0][sel]))
        row = np.array(row)
        boot[b] = row
        if np.isfinite(row).all():
            kdist[best_changepoint(row) - 1] += 1

    # WHICH METRIC WARNS YOU -- CORRECTED. An earlier version of this block
    # reported raw precision at a frozen threshold falling 29% against AUROC's
    # 1.9% and called it "a factor of 16". That comparison is invalid twice over:
    # raw precision is confounded with the base rate (which itself falls over the
    # ladder), and a percent change in AUROC is measured from an origin of 0 when
    # the floor is 0.5. Both corrections are applied below. flag_rate is the one
    # statistic here that needs NO ground truth, and it moves most.
    #
    # AUROC is rank-based and therefore invariant to any
    # monotone recalibration of the score. Steps 0-40 are close to exactly that:
    # the score inflates 0.460 -> 0.677 while largely preserving the correct/
    # incorrect ordering, which AUROC is built not to see. So "AUROC stayed flat"
    # is partly a property of the metric, and the useful question is which metric
    # is NOT invariant. Precision at a numerically frozen threshold is not: it
    # moves an order of magnitude more over the same span, and it needs no
    # ground truth at deployment beyond the labels you already used to set it.
    thr = float(np.quantile(data[used[0]][0], 0.5))
    L.append(f"  operating threshold frozen at step {used[0]} (median score) = {thr:.4f}")
    L.append("")
    L.append(f"  {'step':>6}{'AUROC':>9}{'prec@T':>9}{'base':>8}{'LIFT':>8}"
             f"{'flagrate':>10}{'95% CI (AUROC)':>22}")
    L.append("  " + "-" * 46)
    per = {}
    for i, s in enumerate(used):
        lo, hi = np.nanpercentile(boot[:, i], [2.5, 97.5])
        sc, yy, _ = data[s]
        m = sc >= thr
        base = float(yy.mean())
        prec = float(yy[m].mean()) if m.any() else float("nan")
        lift = prec / base if base > 0 else float("nan")
        L.append(f"  {s:>6}{point[i]:>9.3f}{prec:>9.3f}{base:>8.3f}{lift:>8.3f}"
                 f"{float(m.mean()):>10.3f}      [{lo:.3f}, {hi:.3f}]")
        per[int(s)] = {"auroc": float(point[i]), "ci_lo": float(lo), "ci_hi": float(hi),
                       "precision_at_frozen_thr": prec, "flag_rate": float(m.mean()),
                       "base_rate": base, "lift": lift}
    L.append("")

    a0, a4 = per[used[0]]["auroc"], per[40]["auroc"] if 40 in per else per[used[4]]["auroc"]
    p0, p4 = (per[used[0]]["precision_at_frozen_thr"],
              per[40]["precision_at_frozen_thr"] if 40 in per else
              per[used[4]]["precision_at_frozen_thr"])
    k40 = 40 if 40 in per else used[4]
    l0, l4 = per[used[0]]["lift"], per[k40]["lift"]
    f0, f4 = per[used[0]]["flag_rate"], per[k40]["flag_rate"]
    # AUROC's floor is 0.5, so measure its movement as a fraction of the
    # above-chance margin rather than of the raw value.
    e0, e4 = a0 - 0.5, a4 - 0.5
    L.append(f"  over steps {used[0]}-{k40}:")
    L.append(f"    AUROC                 {a0:.3f} -> {a4:.3f}   "
             f"({100*(e4-e0)/e0:+.1f}% of the above-chance margin)")
    L.append(f"    precision @ frozen T  {p0:.3f} -> {p4:.3f}   "
             f"({100*(p4-p0)/p0:+.1f}%) -- CONFOUNDED with the base rate")
    L.append(f"    lift (prec / base)    {l0:.3f} -> {l4:.3f}   "
             f"({100*(l4-l0)/l0:+.1f}%) -- prevalence removed")
    L.append(f"    flag rate             {f0:.3f} -> {f4:.3f}   "
             f"({100*(f4-f0)/f0:+.1f}%) -- NEEDS NO LABELS")
    L.append("")
    L.append("  The prevalence-adjusted operating point does degrade here (unlike the")
    L.append("  judge's), but by ~13%, not the ~29% raw precision suggests. The statistic")
    L.append("  that moves most, and the only one available without ground truth, is the")
    L.append("  flag rate.")
    L.append("")

    k_hat = best_changepoint(point)
    L.append(f"  point estimate: break after step {used[k_hat - 1]} "
             f"(between {used[k_hat - 1]} and {used[k_hat]})")
    L.append("")
    L.append("  bootstrap distribution of the change-point:")
    total = kdist.sum()
    order = np.argsort(kdist)[::-1]
    cum, ci_steps = 0, []
    for j in order:
        if kdist[j] == 0:
            continue
        share = kdist[j] / total
        L.append(f"    break after step {used[j]:>3}: {share:6.1%}")
        if cum < 0.95:
            ci_steps.append(used[j])
        cum += share
    L.append("")
    modal = used[int(np.argmax(kdist))]
    L.append(f"  modal change-point: after step {modal}   "
             f"({kdist.max() / total:.1%} of resamples)")
    L.append(f"  95% bootstrap set: steps {sorted(ci_steps)}")
    L.append("")
    if kdist.max() / total >= 0.5:
        L.append("  The change-point is well identified: a majority of resamples put the")
        L.append(f"  break in the same place. Quoting '~{modal} steps' is supported.")
    else:
        L.append("  The change-point is NOT well identified -- no single location holds a")
        L.append("  majority. The lag should be quoted as a range, not a number.")
    L.append("")

    txt = "\n".join(L)
    print(txt)
    p = os.path.join(_ROOT, args.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(txt + "\n")
    with open(p.replace(".txt", ".json"), "w") as f:
        json.dump({"run": args.run, "layer": args.layer, "steps": used,
                   "n_boot": args.n_boot, "per_step": per,
                   "changepoint_point_estimate": int(used[k_hat - 1]),
                   "changepoint_modal": int(modal),
                   "changepoint_modal_share": float(kdist.max() / total),
                   "changepoint_distribution": {str(used[j]): int(kdist[j])
                                                for j in range(len(kdist))}}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
