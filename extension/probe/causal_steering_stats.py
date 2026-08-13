"""Statistical hardening for the causal-steering results.

For each steering JSONL (vanilla / post-Goodhart / assertion-control), reports for
every alpha:
  - probe and random accuracy with cluster-bootstrap 95% CIs,
  - Delta = acc(steered) - acc(random) with a paired cluster-bootstrap 95% CI,
  - a one-sided bootstrap p-value for Delta > 0,
  - a test of Delta against the vanilla null-band upper edge (+0.02),
  - McNemar's exact test on the paired (steered vs random) per-prefix outcomes.

Bootstrap resamples PROMPT_IDX (not individual rollouts), because the multiple
responses sampled per prompt are correlated; clustering on the prompt is the
conservative, correct unit of resampling. "probe" is the steered direction in
every file (the optimized pre_answer direction in the main runs, the assertion
direction in the assertion-control file); "rand" is the matched random-direction
control.

Pure CPU analysis on existing data -- no GPU, no Modal.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

import numpy as np

# Upper edge of the vanilla probe-vs-random "null band" [-0.07, +0.02].
#
# CAVEAT -- read before quoting p(D>band) as a p-value. This threshold is the
# MAX of the vanilla checkpoint's own three per-alpha deltas. Testing the
# post-RL delta against it is therefore circular: the band is not an independent
# null distribution, it is the range of three measurements that appear in the
# same figure, with no correction for having taken a max over them. p(D>band) is
# a descriptive "how far outside the vanilla range is this" number, NOT a
# calibrated test. The honest statistics for the headline claim are the paired
# cluster-bootstrap CI on Delta and the exact McNemar p, both already reported
# per alpha -- and those need a multiplicity correction across the alphas swept
# (see MULTIPLICITY_NOTE below).
NULL_BAND_UPPER = 0.02

MULTIPLICITY_NOTE = (
    "Multiplicity: the Delta > 0 claim is evaluated at every alpha in the sweep. "
    "With 3 alphas, a nominal McNemar p of 0.017 becomes ~0.051 under Bonferroni. "
    "Either pre-register a single alpha or report the corrected p."
)


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def correct(r: dict) -> int:
    return int(r["new_score"] == 1.0)


def index_by_prompt(rows: list[dict], alpha: float, direction: str) -> dict[int, list[int]]:
    """prompt_idx -> list of correctness ints for rows matching (alpha, direction)."""
    out: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        if r["alpha"] == alpha and r["direction"] == direction:
            out[r["prompt_idx"]].append(correct(r))
    return out


def cluster_bootstrap_acc(by_prompt: dict[int, list[int]], rng, B: int) -> tuple[float, float, float]:
    prompts = list(by_prompt.keys())
    point = np.mean([c for p in prompts for c in by_prompt[p]])
    boots = np.empty(B)
    for b in range(B):
        sel = rng.choice(prompts, size=len(prompts), replace=True)
        vals = [c for p in sel for c in by_prompt[p]]
        boots[b] = np.mean(vals)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def cluster_bootstrap_delta(steer: dict[int, list[int]], rand: dict[int, list[int]],
                            rng, B: int):
    """Paired delta = acc(steer) - acc(rand), resampling shared prompts."""
    prompts = sorted(set(steer) & set(rand))
    point = np.mean([c for p in prompts for c in steer[p]]) - \
            np.mean([c for p in prompts for c in rand[p]])
    boots = np.empty(B)
    for b in range(B):
        sel = rng.choice(prompts, size=len(prompts), replace=True)
        s = np.mean([c for p in sel for c in steer[p]])
        r = np.mean([c for p in sel for c in rand[p]])
        boots[b] = s - r
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_gt0 = float(np.mean(boots <= 0.0))          # one-sided: Delta > 0
    p_gt_band = float(np.mean(boots <= NULL_BAND_UPPER))  # Delta > null-band upper edge
    return float(point), float(lo), float(hi), p_gt0, p_gt_band


def _exact_mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided binomial p on the discordant pairs, H0: p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _outcomes_by_key(rows: list[dict], alpha: float, direction: str):
    """{(prompt_idx, resp_idx): [correctness, ...]} — a LIST, not a scalar.

    The random arm can hold several draws per prefix. `run_steering.py` writes
    `n_random_seeds` (default 3) random directions and tags every one of them
    `direction="rand"`, keeping the seed only in `direction_detail`. Collapsing
    them into a dict keyed on (prompt_idx, resp_idx) — which is what this
    function used to do — silently kept whichever row came LAST in the file and
    discarded the rest, so McNemar ran against one arbitrary random draw while
    the bootstrap in the same table averaged all of them. The published
    causal_steering_*.jsonl files happen to carry exactly one rand row per key,
    so their numbers are unaffected; a followup steering file is not.
    """
    out: dict[tuple, list[int]] = defaultdict(list)
    for r in rows:
        if r["alpha"] == alpha and r["direction"] == direction:
            out[(r["prompt_idx"], r["resp_idx"])].append(correct(r))
    return out


def mcnemar_exact(rows: list[dict], alpha: float, steer_dir: str, rand_dir: str):
    """Exact two-sided McNemar on paired (steer vs rand) outcomes per prefix.

    With R random replicates per prefix, McNemar is run once per replicate index
    and the REPORTED p is the least favourable (max) across replicates, so a
    result cannot be manufactured by whichever draw happens to look best. The
    returned (b, c) are those of the replicate that produced it. With R = 1 this
    is exactly the original single-pairing test.
    """
    steer = _outcomes_by_key(rows, alpha, steer_dir)
    rand = _outcomes_by_key(rows, alpha, rand_dir)
    keys = sorted(set(steer) & set(rand))
    if not keys:
        return 0, 0, 1.0
    n_rep = max((len(rand[k]) for k in keys), default=1)
    results = []
    for rep in range(n_rep):
        b = c = 0
        for k in keys:
            s = steer[k][min(rep, len(steer[k]) - 1)]
            rd = rand[k][min(rep, len(rand[k]) - 1)]
            b += int(s == 1 and rd == 0)   # steer wins
            c += int(s == 0 and rd == 1)   # rand wins
        results.append((b, c, _exact_mcnemar_p(b, c)))
    return max(results, key=lambda t: t[2])


def analyze(path: str, label: str, B: int, seed: int) -> str:
    rows = load(path)
    rng = np.random.default_rng(seed)
    alphas = sorted({r["alpha"] for r in rows if r["alpha"] != 0.0})
    n_prompts = len({r["prompt_idx"] for r in rows})
    n_prefix = len({(r["prompt_idx"], r["resp_idx"]) for r in rows})
    base = index_by_prompt(rows, 0.0, "zero")
    base_acc, base_lo, base_hi = cluster_bootstrap_acc(base, np.random.default_rng(seed), B)

    L = []
    L.append(f"### {label}")
    L.append(f"  file: {path}")
    L.append(f"  {n_prompts} prompts, {n_prefix} prefixes, baseline acc {base_acc:.3f} "
             f"[{base_lo:.3f}, {base_hi:.3f}]  (cluster bootstrap on prompt_idx, B={B})")
    L.append("")
    L.append(f"  {'alpha':>5} | {'steer acc [95% CI]':>26} | {'rand acc [95% CI]':>26} | "
             f"{'Delta [95% CI]':>26} | {'p(D>0)':>7} | {'p(D>band)':>9} | {'McNemar':>8}")
    L.append("  " + "-" * 130)
    for a in alphas:
        s_by = index_by_prompt(rows, a, "probe")
        r_by = index_by_prompt(rows, a, "rand")
        if not s_by or not r_by:
            continue
        s_acc, s_lo, s_hi = cluster_bootstrap_acc(s_by, np.random.default_rng(seed + 1), B)
        r_acc, r_lo, r_hi = cluster_bootstrap_acc(r_by, np.random.default_rng(seed + 2), B)
        d, d_lo, d_hi, p0, pband = cluster_bootstrap_delta(
            s_by, r_by, np.random.default_rng(seed + 3), B)
        mb, mc, mp = mcnemar_exact(rows, a, "probe", "rand")
        L.append(f"  {a:>5.1f} | {s_acc:>7.3f} [{s_lo:+.3f},{s_hi:+.3f}] | "
                 f"{r_acc:>7.3f} [{r_lo:+.3f},{r_hi:+.3f}] | "
                 f"{d:>+7.3f} [{d_lo:+.3f},{d_hi:+.3f}] | {p0:>7.3f} | {pband:>9.3f} | "
                 f"{mp:>8.3f}")
    L.append("")
    L.append("  p(D>0): one-sided bootstrap prob that the true Delta <= 0 (small = robustly positive).")
    L.append("  p(D>band): bootstrap prob that Delta <= +0.02. NOT a calibrated p-value -- +0.02")
    L.append("             is the MAX of the vanilla checkpoint's own deltas, so this compares a")
    L.append("             measurement against the range of three other measurements shown in the")
    L.append("             same figure. Descriptive only.")
    L.append("  McNemar: exact two-sided p on paired (steered-correct/rand-wrong) vs (rand-correct/steered-wrong).")
    L.append(f"  {MULTIPLICITY_NOTE}")
    n_alpha = len(alphas)
    if n_alpha > 1:
        L.append(f"  Bonferroni across the {n_alpha} alphas swept here: multiply the McNemar column by {n_alpha}.")
    L.append("  Sign consistency across alpha is reported above; a non-monotone Delta(alpha) argues")
    L.append("  against a dose-response reading even when one alpha is individually significant.")
    L.append("")
    return "\n".join(L)


def dose_response(entries: list[str], alpha: float, B: int, seed: int, out_csv: str) -> str:
    """entries: list of 'STEP:path' for the OPTIMIZED direction across training.

    Reports Delta(steer - rand) at the given alpha with cluster-bootstrap CI per
    training step, so you can plot the probe direction becoming progressively
    causal. Writes a CSV (step,delta,ci_lo,ci_hi,p_gt0) for plotting.
    """
    L = [f"### Dose-response: Delta(probe - random) at alpha={alpha} vs training step",
         f"  {'step':>6} | {'Delta [95% CI]':>26} | {'p(D>0)':>7} | n_prompts"]
    L.append("  " + "-" * 64)
    csv = ["step,delta,ci_lo,ci_hi,p_gt0,n_prompts"]
    for e in sorted(entries, key=lambda s: float(s.split(":", 1)[0])):
        step_s, path = e.split(":", 1)
        rows = load(path)
        s_by = index_by_prompt(rows, alpha, "probe")
        r_by = index_by_prompt(rows, alpha, "rand")
        d, lo, hi, p0, _ = cluster_bootstrap_delta(
            s_by, r_by, np.random.default_rng(seed), B)
        npr = len({rr["prompt_idx"] for rr in rows})
        L.append(f"  {step_s:>6} | {d:>+7.3f} [{lo:+.3f},{hi:+.3f}] | {p0:>7.3f} | {npr}")
        csv.append(f"{step_s},{d:.4f},{lo:.4f},{hi:.4f},{p0:.4f},{npr}")
    import os
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    open(out_csv, "w").write("\n".join(csv) + "\n")
    L.append("")
    L.append(f"  CSV written to {out_csv} (plot delta vs step with ci_lo/ci_hi error bars).")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose", nargs="+", default=None,
                    help="dose-response mode: STEP:path entries for the optimized direction, "
                         "e.g. 0:vanilla.jsonl 40:step40.jsonl 60:step60.jsonl 99:postRL.jsonl")
    ap.add_argument("--dose_alpha", type=float, default=1.0)
    ap.add_argument("--dose_csv", default="extension/outputs/causal_steering_dose_response.csv")
    ap.add_argument("--vanilla", default="extension/outputs/n500/causal_steering_full.jsonl")
    ap.add_argument("--postgoodhart", default="causal_steering_runA_postRL.jsonl")
    ap.add_argument("--assertion", default="causal_steering_runA_assertion_control.jsonl")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="extension/outputs/causal_steering_stats.txt")
    args = ap.parse_args()

    if args.dose:
        txt = dose_response(args.dose, args.dose_alpha, args.bootstrap, args.seed, args.dose_csv)
        print(txt)
        return

    blocks = []
    for path, label in [
        (args.vanilla, "Vanilla C_outcome (probe = optimized pre_answer direction)"),
        (args.postgoodhart, "Post-Goodhart runA (probe = optimized pre_answer direction)"),
        (args.assertion, "Post-Goodhart runA, SPECIFICITY CONTROL (probe = assertion direction)"),
    ]:
        try:
            blocks.append(analyze(path, label, args.bootstrap, args.seed))
        except FileNotFoundError:
            blocks.append(f"### {label}\n  file not found: {path}\n")

    txt = "\n".join(blocks)
    print(txt)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write(txt + "\n")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
