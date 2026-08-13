"""Arm A / Arm B: recompute the pre-registered RL arms from their eval rollouts.

REVISION_PACK sections G4 and G5 report two pre-registered probe-as-reward arms:

  A  reward = surface-RESIDUALISED probe, LR(h - s@B) over 39 text features
  B  reward = the 39 surface features ONLY; never reads the activation

and the claim rests on four first-block accuracies. Until 2026-08-12 nothing in
this repository could recompute them: the rollout JSONs lived only on the Modal
volume, so the two newest and most load-bearing results were the two nobody
could check. This script closes that.

Fetch the rollouts first (they are ~10 MB each and gitignored):

    modal volume get default-proj-training \\
        evaluation/eval_results/armA_residual_step100.json ./eval_armA_residual_step100.json
    modal volume get default-proj-training \\
        evaluation/eval_results/armB_surface_step100.json ./eval_armB_surface_step100.json

Then:

    python extension/probe/verify_residual_arms.py

Everything below follows the analysis fixed in PREREGISTRATION.md before the runs
and restated in HANDOFF.md, so none of it is a post-hoc choice:

  - primary outcome  first-block accuracy on the contamination-filtered clean-406
  - verifier         evaluation.countdown -- all numbers used exactly once,
                     arithmetic only, evaluates to target
  - intervals        prompt-clustered PAIRED bootstrap, 10,000 resamples

One population note the pack does not currently state, and should. The two arms
were evaluated at `--num_responses 8` while the two reference checkpoints were
sampled at 16, so an arm contributes 3,248 rollouts against a reference's 6,496.
The bootstrap pairs at the PROMPT level -- per-prompt mean accuracy, then the
difference of those means -- so the contrast is well defined regardless, and all
406 prompts are present on both sides. But the arms' per-prompt means are the
noisier ones, and a reader comparing raw rollout counts deserves to know why they
differ. This script prints the counts rather than hiding them.

Pure CPU, no GPU, no network beyond the two `modal volume get` calls above.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.countdown import evaluate_equation, validate_equation  # noqa: E402

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)

# label -> (path, what it is). The reference rows are the ones the pack quotes.
DEFAULT_ARMS = [
    ("C_outcome (verifier RL)", "eval_c_outcome_FIXEDSTOP_n500.json"),
    ("published probe-as-reward", "eval_runA_postRL_n500.json"),
    ("Arm A, surface-residualised", "eval_armA_residual_step100.json"),
    ("Arm B, surface-only", "eval_armB_surface_step100.json"),
]

# The contrasts REVISION_PACK G4/G5 report, as (treatment, reference).
CONTRASTS = [
    ("Arm A, surface-residualised", "published probe-as-reward"),
    ("Arm A, surface-residualised", "C_outcome (verifier RL)"),
    ("Arm B, surface-only", "published probe-as-reward"),
    ("Arm B, surface-only", "C_outcome (verifier RL)"),
]

# What the pack currently prints, so a drift shows up as a diff rather than as a
# number nobody rechecks. pp for deltas, absolute for accuracies.
PUBLISHED = {
    "C_outcome (verifier RL)": 0.5306,
    "published probe-as-reward": 0.2361,
    "Arm A, surface-residualised": 0.1678,
    "Arm B, surface-only": 0.0000,
}
PUBLISHED_CONTRASTS = {
    ("Arm A, surface-residualised", "published probe-as-reward"): (-6.83, -8.85, -4.79),
    ("Arm A, surface-residualised", "C_outcome (verifier RL)"): (-36.28, -39.36, -33.19),
    ("Arm B, surface-only", "published probe-as-reward"): (-23.61, -26.43, -20.89),
    ("Arm B, surface-only", "C_outcome (verifier RL)"): (-53.06, -57.04, -48.97),
}


def clean_prompts() -> set[int] | None:
    """The 406 prompts with no overlap against the RLOO training pool."""
    p = os.path.join(_REPO_ROOT, "extension", "data", "contaminated_prompt_idx.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return {int(i) for i in json.load(f)["clean"]}


def first_block_correct(resp: str, target: int, nums: list) -> int:
    """Verifier correctness of the FIRST <answer> block -- the paper's convention.

    The pre-registration fixes this rather than last-block, and says why: on the
    published probe-RL run last-block would report -45.7 pp where the paper
    reports -31. Choosing the convention after seeing the arms would have been a
    researcher degree of freedom worth several points of effect size.
    """
    blocks = [m.group(1) for m in _ANSWER_RE.finditer(resp)]
    if not blocks:
        return 0
    eq = blocks[0].strip()
    if not validate_equation(eq, list(nums)):
        return 0
    val = evaluate_equation(eq)
    return int(val is not None and abs(val - int(target)) < 1e-5)


def load_arm(path: str, keep: set[int] | None) -> dict[int, np.ndarray]:
    """prompt_idx -> per-rollout 0/1 correctness. Missing file returns {}."""
    full = os.path.join(_REPO_ROOT, path)
    if not os.path.exists(full):
        return {}
    per: dict[int, list[int]] = {}
    with open(full) as f:
        for p, line in enumerate(l for l in f if l.strip()):
            if keep is not None and p not in keep:
                continue
            row = json.loads(line)
            per[p] = [
                first_block_correct(r, int(row["target"]), row["nums"])
                for r in row["response"]
            ]
    return {k: np.asarray(v, float) for k, v in per.items()}


def paired_bootstrap(a: dict[int, np.ndarray], b: dict[int, np.ndarray],
                     n_boot: int, seed: int) -> tuple[float, float, float]:
    """Delta in pp, with a prompt-clustered paired bootstrap CI.

    Resample PROMPTS with replacement -- not rollouts -- because rollouts from
    one prompt are not independent, and pair the two arms on the resampled
    prompts so the difference is within-prompt. Per-prompt means make the two
    sides comparable even though the arms carry 8 rollouts per prompt against
    the references' 16.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        return float("nan"), float("nan"), float("nan")
    ma = np.array([a[p].mean() for p in shared])
    mb = np.array([b[p].mean() for p in shared])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(shared), size=(n_boot, len(shared)))
    deltas = (ma[idx].mean(axis=1) - mb[idx].mean(axis=1)) * 100.0
    return (
        float((ma.mean() - mb.mean()) * 100.0),
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="extension/outputs/n500/text/62_residual_arms.txt")
    args = ap.parse_args()

    keep = clean_prompts()
    L = [
        "Pre-registered RL arms -- recomputed from the rollout JSONs",
        f"  population: clean-406 ({len(keep) if keep else 'ALL'} prompts), "
        "first-block accuracy, verifier = evaluation.countdown",
        f"  intervals:  prompt-clustered PAIRED bootstrap, {args.n_boot} resamples, seed {args.seed}",
        "",
    ]

    arms, missing = {}, []
    for label, path in DEFAULT_ARMS:
        per = load_arm(path, keep)
        if not per:
            missing.append((label, path))
            continue
        arms[label] = per

    if missing:
        L.append("  MISSING -- fetch from the Modal volume (see this file's docstring):")
        for label, path in missing:
            L.append(f"    {label:32s} {path}")
        L.append("")

    L.append(f"  {'arm':34s}{'prompts':>9}{'rollouts':>10}{'acc':>9}{'published':>11}{'':>3}")
    L.append("  " + "-" * 76)
    results = {}
    for label, path in DEFAULT_ARMS:
        if label not in arms:
            continue
        per = arms[label]
        flat = np.concatenate([per[p] for p in sorted(per)])
        acc = float(flat.mean())
        pub = PUBLISHED.get(label)
        ok = "" if pub is None else ("  ok" if abs(acc - pub) < 5e-4 else "  DIFF")
        L.append(f"  {label:34s}{len(per):>9}{len(flat):>10}{acc:>9.4f}"
                 f"{'' if pub is None else format(pub, '>11.4f')}{ok}")
        results[label] = {"path": path, "n_prompts": len(per), "n_rollouts": len(flat),
                          "first_block_acc": acc, "published": pub}
    L.append("")
    L.append("  Note the rollout counts: the arms were evaluated at 8 responses per")
    L.append("  prompt and the references at 16. The bootstrap below pairs on PROMPTS")
    L.append("  and uses per-prompt means, so the contrast is well defined; the arms'")
    L.append("  per-prompt means are simply the noisier side.")
    L.append("")

    L.append(f"  {'contrast':56s}{'delta pp':>10}{'95% CI (pp)':>22}")
    L.append("  " + "-" * 88)
    contrasts = {}
    for treat, ref in CONTRASTS:
        if treat not in arms or ref not in arms:
            continue
        d, lo, hi = paired_bootstrap(arms[treat], arms[ref], args.n_boot, args.seed)
        pub = PUBLISHED_CONTRASTS.get((treat, ref))
        flag = ""
        if pub is not None:
            flag = "  ok" if abs(d - pub[0]) < 0.05 else "  DIFF"
        L.append(f"  {treat + ' - ' + ref:56s}{d:>+10.2f}   [{lo:+7.2f}, {hi:+7.2f}]{flag}")
        if pub is not None:
            L.append(f"  {'  published:':56s}{pub[0]:>+10.2f}   [{pub[1]:+7.2f}, {pub[2]:+7.2f}]")
        contrasts[f"{treat} - {ref}"] = {
            "delta_pp": d, "ci_lo_pp": lo, "ci_hi_pp": hi,
            "published": None if pub is None else
            {"delta_pp": pub[0], "ci_lo_pp": pub[1], "ci_hi_pp": pub[2]},
        }
    L.append("")
    L.append("  Arm A's pre-registered prediction was that it would collapse LESS than")
    L.append("  the published run. It collapsed more, outside the interval. That is a")
    L.append("  falsified prediction and is reported as one.")

    txt = "\n".join(L)
    print(txt)
    out = os.path.join(_REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(txt + "\n")
    with open(out.replace(".txt", ".json"), "w") as f:
        json.dump({"population": "clean-406", "label_rule": "first_block",
                   "n_boot": args.n_boot, "seed": args.seed,
                   "arms": results, "contrasts": contrasts}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
