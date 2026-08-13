"""Quantify the structural-confound mechanism claimed in writeup §6 (mechanism).

We define a transparent template-match score on the rollout's FULL text — the
whole response, not just the <think> body; an earlier version of this docstring
said "<think> body", which the code has never done. The distinction matters
because the post-Goodhart policy emits most of its rhetorical scaffolding around
the answer blocks. `followup/fragility_core/labels.template_score` scores the
same way, so the two remain directly comparable.

We then compare its association with verifier correctness in two checkpoints:

  - vanilla C_outcome (eval_c_outcome_FIXEDSTOP_n500.json)
  - probe-as-RL-reward runB post-Goodhart (eval_runB_postRL_n500.json)

If the policy decoupled the structural template from actual correctness under
probe-RL, the correlation between template-match score and verifier accuracy
should drop sharply from vanilla to post-Goodhart, while the absolute rate of
template emission rises.
"""
import json
import re
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


TEMPLATE_PATTERNS = [
    (r"\blet me\b", "let_me"),
    (r"\bstep by step\b", "step_by_step"),
    (r"\btherefore\b", "therefore"),
    (r"\bso the answer is\b|\bthe answer is\b", "the_answer_is"),
    (r"\bverified\b|\bvalid\b|\bthis works\b|\bgot it\b|\bperfect\b", "verification"),
    (r"^\s*\d+\.\s", "numbered_step"),
    (r"\bfirst\b.*\bthen\b|\bnext\b|\bafter that\b", "sequential_markers"),
]


def template_score(text: str) -> int:
    """Sum of distinct template-pattern hits (each pattern counted at most once)."""
    lower = text.lower()
    score = 0
    for pat, _name in TEMPLATE_PATTERNS:
        if re.search(pat, lower, flags=re.MULTILINE):
            score += 1
    return score


def load_jsonl(path: Path):
    rollouts = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            for resp, sc in zip(rec["response"], rec["scores"]):
                rollouts.append({"text": resp, "score": float(sc)})
    return rollouts


def analyse(label: str, path: Path):
    rollouts = load_jsonl(path)
    ts = np.array([template_score(r["text"]) for r in rollouts])
    correct = np.array([1.0 if r["score"] >= 0.999 else 0.0 for r in rollouts])

    n = len(rollouts)
    mean_ts = ts.mean()
    pct_with_template = float((ts >= 3).mean())
    acc = correct.mean()

    # Pearson r between template-score and binary correctness
    if ts.std() > 0 and correct.std() > 0:
        r = float(np.corrcoef(ts, correct)[0, 1])
    else:
        r = float("nan")

    # Mean template score in correct vs wrong sub-populations
    if correct.sum() > 0:
        ts_in_correct = float(ts[correct == 1].mean())
    else:
        ts_in_correct = float("nan")
    if (1 - correct).sum() > 0:
        ts_in_wrong = float(ts[correct == 0].mean())
    else:
        ts_in_wrong = float("nan")

    # Rate of template emission AMONG verifier-wrong rollouts
    if (1 - correct).sum() > 0:
        template_rate_wrong = float((ts[correct == 0] >= 3).mean())
    else:
        template_rate_wrong = float("nan")

    print(f"\n=== {label}: {path.name} ===")
    print(f"  n_rollouts                   = {n}")
    print(f"  verifier accuracy (=1.0)     = {acc:.3f}")
    print(f"  mean template-score (0-7)    = {mean_ts:.2f}")
    print(f"  fraction with template>=3    = {pct_with_template:.3f}")
    print(f"  mean template-score | correct = {ts_in_correct:.2f}")
    print(f"  mean template-score | wrong  = {ts_in_wrong:.2f}")
    print(f"  Pearson(template, correct)   = {r:+.3f}")
    print(f"  P(template>=3 | wrong)       = {template_rate_wrong:.3f}")

    return {
        "label": label,
        "n": n,
        "acc": acc,
        "mean_ts": mean_ts,
        "ts_correct": ts_in_correct,
        "ts_wrong": ts_in_wrong,
        "pearson": r,
        "template_rate_wrong": template_rate_wrong,
    }


def main():
    vanilla = analyse("vanilla C_outcome", ROOT / "eval_c_outcome_FIXEDSTOP_n500.json")
    posthack = analyse("probe-RL runB (post-Goodhart)", ROOT / "eval_runB_postRL_n500.json")

    print("\n=== Summary contrast ===")
    print(
        f"Pearson(template, correct):  vanilla {vanilla['pearson']:+.3f}  ->  post-Goodhart {posthack['pearson']:+.3f}"
    )
    print(
        f"Mean template-score:          vanilla {vanilla['mean_ts']:.2f}      ->  post-Goodhart {posthack['mean_ts']:.2f}"
    )
    print(
        f"P(template | wrong rollout):  vanilla {vanilla['template_rate_wrong']:.3f}     ->  post-Goodhart {posthack['template_rate_wrong']:.3f}"
    )


if __name__ == "__main__":
    main()
