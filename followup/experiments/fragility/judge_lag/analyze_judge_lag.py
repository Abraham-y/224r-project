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
    L.append(f"  {'step':>6}{'judge AUROC':>13}{'prec@thr':>10}{'flag rate':>11}"
             f"{'true acc':>10}{'95% CI':>20}")
    L.append("  " + "-" * 72)
    per = {}
    for i, s in enumerate(steps):
        lo, hi = np.nanpercentile(boot[:, i], [2.5, 97.5])
        m = data[s]["s"] >= thr
        prec = float(data[s]["y"][m].mean()) if m.any() else float("nan")
        L.append(f"  {s:>6}{point[i]:>13.3f}{prec:>10.3f}{float(m.mean()):>11.3f}"
                 f"{float(data[s]['y'].mean()):>10.3f}      [{lo:.3f}, {hi:.3f}]")
        per[int(s)] = {"auroc": float(point[i]), "ci_lo": float(lo), "ci_hi": float(hi),
                       "precision_at_frozen_thr": prec, "flag_rate": float(m.mean()),
                       "true_acc": float(data[s]["y"].mean())}
    L.append("")

    k = best_changepoint(point)
    L.append(f"  change-point: break after step {steps[k-1]}   "
             f"({kd.max()/max(kd.sum(),1):.1%} of resamples agree)")
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
        L.append(f"  probe break: after step {pj.get('changepoint_modal')}   "
                 f"judge break: after step {steps[k-1]}")
        if steps[k-1] == pj.get("changepoint_modal"):
            L.append("  Both monitors' discrimination breaks at the same point: the lag is a")
            L.append("  property of monitoring this policy, not of reading activations.")
        elif steps[k-1] < pj.get("changepoint_modal", 10**9):
            L.append("  The judge breaks EARLIER than the probe -- it is the better early")
            L.append("  warning of the two, which is a usable recommendation.")
        else:
            L.append("  The judge breaks LATER than the probe. State this plainly: it means")
            L.append("  an LLM judge gives even less warning, not more.")
        L.append("")

    txt = "\n".join(L)
    print(txt)
    p = os.path.join(_ROOT, args.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(txt + "\n")
    with open(p.replace(".txt", ".json"), "w") as f:
        json.dump({"steps": steps, "frozen_threshold": thr, "per_step": per,
                   "changepoint": int(steps[k-1])}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
