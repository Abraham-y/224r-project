"""Does an LLM judge's health metric lag too? Same analysis, different monitor.

Reads the per-checkpoint scores written by score_ladder_with_judge.py and runs
the identical measurement the paper runs on the linear probe:

  - the judge's AUROC against the exact verifier, per checkpoint, on that
    checkpoint's OWN freshly sampled rollouts
  - prompt-clustered bootstrap intervals over the 406 eval prompts
  - precision at a numerically frozen operating threshold, which is the metric
    the paper argues you should actually watch
  - the change-point, located by procedure

The comparison to make in the paper is between two curves on one plot: the
probe's AUROC and the judge's, over the same checkpoints of the same run. Three
outcomes and all of them are publishable, which is the point of running it:

  the judge lags too          the result is about monitors, not about probes,
                              and it transfers to the venue's own object of study
  the judge degrades earlier  even better -- an LLM judge is a LEADING indicator
                              where its own AUROC is a lagging one, and that is
                              a concrete recommendation
  the judge never degrades    the lag is specific to monitors that read
                              activations, which materially narrows the paper's
                              claim and needs to be said

Do not report this as the judge being attacked. It was not: the policy was
optimised against the probe, and the judge is a bystander watching the same
drift. That is the honest framing and it is still the practitioner's question --
"would my judge's accuracy have warned me?"

    python followup/experiments/fragility/judge_lag/analyze_judge_lag.py

CPU only.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_DEFAULT_IN = os.path.join(_ROOT, "followup", "results", "fragility", "judge_lag")


def load(step_dir: str) -> dict[int, dict]:
    """step -> {score, label, prompt} arrays."""
    out = {}
    for fn in sorted(os.listdir(step_dir)):
        if not (fn.startswith("step_") and fn.endswith(".jsonl")):
            continue
        step = int(fn[len("step_"):-len(".jsonl")])
        s, y, p = [], [], []
        with open(os.path.join(step_dir, fn)) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("judge_score") is None or not np.isfinite(r["judge_score"]):
                    continue
                s.append(r["judge_score"]); y.append(r["correct"]); p.append(r["prompt_idx"])
        if s:
            out[step] = {"s": np.array(s), "y": np.array(y), "p": np.array(p)}
    return out


def auroc(y, s) -> float:
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")


def operating_point(y: np.ndarray, sc: np.ndarray, thr: float) -> dict:
    """Operating-point stats at a frozen threshold, WITH prevalence removed.

    Raw precision at a fixed threshold is confounded with the base rate: it falls
    mechanically when prevalence falls, even with discrimination completely
    unchanged. Over this ladder true accuracy roughly halves, so a precision
    "collapse" is largely arithmetic. An earlier version of this analysis reported
    the raw fall (-42.5%) as evidence the monitor degraded; it is not.

    Two prevalence-invariant summaries instead:
      lift = precision / base rate     -- enrichment of the flagged set
      LR+  = TPR / FPR                 -- the likelihood ratio the threshold buys
    Both are 1.0 for a useless monitor and are unchanged by a shift in prevalence
    alone. flag_rate is kept because it needs NO labels at all.
    """
    m = sc >= thr
    base = float(y.mean())
    flag = float(m.mean())
    prec = float(y[m].mean()) if m.any() else float("nan")
    tpr = flag * prec / base if base > 0 else float("nan")
    fpr = flag * (1 - prec) / (1 - base) if base < 1 else float("nan")
    return {"precision": prec, "base_rate": base, "flag_rate": flag,
            "lift": prec / base if base > 0 else float("nan"),
            "tpr": tpr, "fpr": fpr,
            "lr_pos": tpr / fpr if fpr and fpr > 0 else float("nan")}


def paired_auroc_diff(data, idx, prompts, a, b, n_boot, seed):
    """AUROC(a) - AUROC(b) with a prompt-clustered PAIRED bootstrap.

    Every checkpoint is scored on the SAME 406 prompts, so the checkpoints are
    paired and the paired bootstrap is both the correct estimator and strictly
    more powerful than comparing two marginal intervals. An earlier version of
    this analysis concluded "the AUROC never degrades" from overlapping marginal
    CIs; that is not a test of a difference, and the paired test contradicts it.
    """
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        dr = rng.choice(prompts, size=len(prompts), replace=True)
        ia = np.concatenate([idx[a][q] for q in dr])
        ib = np.concatenate([idx[b][q] for q in dr])
        d[i] = (auroc(data[a]["y"][ia], data[a]["s"][ia])
                - auroc(data[b]["y"][ib], data[b]["s"][ib]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d < 0).mean(), (d > 0).mean())
    return float(d.mean()), float(lo), float(hi), float(p)


def best_changepoint(series: np.ndarray) -> int:
    n, best_k, best = len(series), 1, np.inf
    for k in range(1, n):
        a, b = series[:k], series[k:]
        sse = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        if sse < best:
            best_k, best = k, sse
    return best_k


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_dir", default=_DEFAULT_IN)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="followup/results/fragility/judge_lag.txt")
    args = ap.parse_args()

    if not os.path.isdir(args.in_dir):
        raise SystemExit(
            f"{args.in_dir} not found. Run score_ladder_with_judge.py on Modal first, "
            "then: modal volume get default-proj-training fragility/judge_lag "
            "./followup/results/fragility/judge_lag")
    data = load(args.in_dir)
    if len(data) < 4:
        raise SystemExit(f"only {len(data)} checkpoints scored; need >=4")
    steps = sorted(data)

    L = [
        "Does an LLM judge's AUROC lag too?",
        f"  {len(steps)} checkpoints: {steps}",
        "  judge scored against the exact Countdown verifier on each checkpoint's",
        "  own freshly sampled rollouts; prompt-clustered bootstrap "
        f"({args.n_boot} resamples, seed {args.seed})",
        "",
        "  NOTE: the policy was optimised against the PROBE, not against this judge.",
        "  The judge is a bystander watching the same drift. Report it that way.",
        "",
    ]

    point = np.array([auroc(data[s]["y"], data[s]["s"]) for s in steps])
    thr = float(np.quantile(data[steps[0]]["s"], 0.5))  # frozen at the first checkpoint
    prompts = np.unique(np.concatenate([data[s]["p"] for s in steps]))
    idx = {s: {q: np.flatnonzero(data[s]["p"] == q) for q in prompts} for s in steps}

    rng = np.random.default_rng(args.seed)
    boot = np.full((args.n_boot, len(steps)), np.nan)
    kd = np.zeros(len(steps) - 1, dtype=int)
    for b in range(args.n_boot):
        draw = rng.choice(prompts, size=len(prompts), replace=True)
        row = np.array([auroc(data[s]["y"][np.concatenate([idx[s][q] for q in draw])],
                              data[s]["s"][np.concatenate([idx[s][q] for q in draw])])
                        for s in steps])
        boot[b] = row
        if np.isfinite(row).all():
            kd[best_changepoint(row) - 1] += 1

    L.append(f"  operating threshold frozen at step {steps[0]} (median judge score) = {thr:.4f}")
    L.append("")
    L.append(f"  {'step':>6}{'judge AUROC':>13}{'prec@T':>9}{'base':>8}{'LIFT':>8}"
             f"{'LR+':>7}{'flagrate':>10}{'95% CI (AUROC)':>20}")
    L.append("  " + "-" * 85)
    per = {}
    for i, s in enumerate(steps):
        lo, hi = np.nanpercentile(boot[:, i], [2.5, 97.5])
        op = operating_point(data[s]["y"], data[s]["s"], thr)
        L.append(f"  {s:>6}{point[i]:>13.3f}{op['precision']:>9.3f}{op['base_rate']:>8.3f}"
                 f"{op['lift']:>8.3f}{op['lr_pos']:>7.2f}{op['flag_rate']:>10.3f}"
                 f"      [{lo:.3f}, {hi:.3f}]")
        per[int(s)] = {"auroc": float(point[i]), "ci_lo": float(lo), "ci_hi": float(hi),
                       "precision_at_frozen_thr": op["precision"],
                       "flag_rate": op["flag_rate"], "true_acc": op["base_rate"],
                       "lift": op["lift"], "lr_pos": op["lr_pos"],
                       "tpr": op["tpr"], "fpr": op["fpr"]}
    L.append("")

    # --- PAIRED tests, the estimator this comparison actually needs ----------
    L.append("  PAIRED AUROC differences (prompt-clustered; the correct test for")
    L.append("  checkpoints scored on the same prompts, unlike overlapping marginal CIs):")
    L.append("")
    L.append(f"    {'contrast':<26}{'delta':>9}{'95% CI':>22}{'p':>9}")
    L.append("    " + "-" * 66)
    pairs = [(steps[0], t) for t in steps[1:]] + [(30, 80)] if 30 in data and 80 in data \
        else [(steps[0], t) for t in steps[1:]]
    paired_out = {}
    for a, b in pairs:
        m_, lo_, hi_, p_ = paired_auroc_diff(data, idx, prompts, a, b,
                                             args.n_boot, args.seed)
        star = "  *" if p_ < 0.05 else ""
        L.append(f"    step {a} - step {b:<15}{m_:>+9.4f}   [{lo_:+.4f},{hi_:+.4f}]{p_:>9.3f}{star}")
        paired_out[f"{a}-{b}"] = {"delta": m_, "ci_lo": lo_, "ci_hi": hi_, "p": p_}
    out_paired = paired_out
    L.append("")
    sig_down = [k for k, v in paired_out.items() if v["p"] < 0.05 and v["delta"] > 0]
    sig_up = [k for k, v in paired_out.items() if v["p"] < 0.05 and v["delta"] < 0]
    if sig_down:
        L.append(f"  AUROC DOES degrade significantly: {', '.join(sig_down)}.")
    if sig_up:
        L.append(f"  ...and significantly RECOVERS: {', '.join(sig_up)}.")
    if sig_down and sig_up:
        L.append("  The series is non-monotone. Do not describe it as 'never degrades' --")
        L.append("  that reads the least sensitive contrast (first vs last) off a U shape.")
    L.append("")

    k = best_changepoint(point)
    share = kd.max() / max(kd.sum(), 1)
    # best_changepoint ALWAYS returns a k -- it fits the best two-segment split
    # even to a flat, noisy series where no break exists. So the location is only
    # meaningful if the bootstrap concentrates. Below a majority it does not, and
    # reporting "the break is at k" would be reading structure out of noise.
    identified = share >= 0.5
    L.append(f"  change-point: best split after step {steps[k-1]}   "
             f"({share:.1%} of resamples agree)")
    if not identified:
        L.append("  NOT IDENTIFIED -- no location holds a majority. On this series the")
        L.append("  judge's AUROC has no detectable break: treat it as flat, not as")
        L.append("  breaking at the argmin of a fit that is forced to return one.")
    L.append("")

    # The comparison the paper needs: does this curve break where the probe's did?
    probe_json = os.path.join(_ROOT, "followup", "results", "fragility",
                              "changepoint_lag.json")
    if os.path.exists(probe_json):
        pj = json.load(open(probe_json))
        pb = pj.get("per_step", {})
        L.append(f"  {'step':>6}{'probe AUROC':>13}{'judge AUROC':>13}")
        L.append("  " + "-" * 32)
        for s in steps:
            if str(s) in pb:
                L.append(f"  {s:>6}{pb[str(s)]['auroc']:>13.3f}{per[s]['auroc']:>13.3f}")
        L.append("")
        a0, a9 = per[steps[0]]["auroc"], per[steps[-1]]["auroc"]
        p0, p9 = (per[steps[0]]["precision_at_frozen_thr"],
                  per[steps[-1]]["precision_at_frozen_thr"])
        lo0, hi0 = per[steps[0]]["ci_lo"], per[steps[0]]["ci_hi"]
        lo9, hi9 = per[steps[-1]]["ci_lo"], per[steps[-1]]["ci_hi"]
        overlap = not (hi9 < lo0 or hi0 < lo9)
        L.append(f"  probe break: after step {pj.get('changepoint_modal')}   "
                 f"judge: {'no identified break' if not identified else 'after step ' + str(steps[k-1])}")
        L.append("")
        l0, l9 = per[steps[0]]["lift"], per[steps[-1]]["lift"]
        r0, r9 = per[steps[0]]["lr_pos"], per[steps[-1]]["lr_pos"]
        f0, f9 = per[steps[0]]["flag_rate"], per[steps[-1]]["flag_rate"]
        if not identified and overlap:
            L.append("  The judge's AUROC does not move end-to-end: "
                     f"{a0:.3f} [{lo0:.3f},{hi0:.3f}] -> {a9:.3f} [{lo9:.3f},{hi9:.3f}].")
            L.append("")
            L.append(f"  Raw precision at the frozen threshold falls {p0:.3f} -> {p9:.3f} "
                     f"({100*(p9-p0)/p0:+.0f}%), but so does the base rate")
            L.append(f"  ({per[steps[0]]['true_acc']:.3f} -> {per[steps[-1]]['true_acc']:.3f}). "
                     "PREVALENCE-ADJUSTED the flagged set gets")
            L.append(f"  MORE enriched, not less: lift {l0:.2f} -> {l9:.2f} "
                     f"({100*(l9-l0)/l0:+.0f}%), LR+ {r0:.2f} -> {r9:.2f}.")
            L.append("  So the precision fall is arithmetic, not evaluator failure. Do NOT")
            L.append("  report it as the judge degrading.")
            L.append("")
            L.append(f"  What DOES move, and needs no labels at all: flag rate "
                     f"{f0:.3f} -> {f9:.3f} ({100*(f9-f0)/f0:+.0f}%),")
            L.append(f"  against an AUROC change of {100*(a9-a0)/a0:+.1f}%.")
        elif steps[k-1] < pj.get("changepoint_modal", 10**9):
            L.append("  The judge breaks EARLIER than the probe -- the better early warning.")
        else:
            L.append("  The judge breaks LATER than the probe: even less warning, not more.")
        L.append("")

    txt = "\n".join(L)
    print(txt)
    p = os.path.join(_ROOT, args.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(txt + "\n")
    with open(p.replace(".txt", ".json"), "w") as f:
        json.dump({"steps": steps, "frozen_threshold": thr, "per_step": per,
                   "paired_auroc_diffs": out_paired,
                   "changepoint": int(steps[k-1])}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
