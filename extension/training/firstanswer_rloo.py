"""RLOO with FIRST-answer reward instead of last-answer reward.

The standard Countdown verifier (`evaluation/countdown.compute_score`) scores
the LAST `<answer>` block in the rollout. The 0.5B C_outcome model exploits
this by emitting multiple `<answer>` blocks per rollout (mean 7.6, 87%
multi-answer) — a reward hack: try many candidates, only the last is graded.

This trainer rewards the FIRST `<answer>` block instead. Same RLOO algorithm;
same KL penalty; same hyperparameters. Only the verifier's score-extraction
rule changes. Hypothesis: rewarding first-block correctness will suppress
the rambling, because subsequent blocks no longer have reward potential.

This is the text-based equivalent of "probe-as-reward shaping" — the
probe at `</think>` is a near-oracle predictor of first-`<answer>`-block
correctness (held-out AUROC 0.98 on existing C_outcome data with corrected
labels). Rewarding first-block correctness is therefore equivalent to
rewarding what the probe would say, but implemented at the verifier level
without needing to deploy the probe inside RLOO's inner loop.

Architecture matches extension/training/process_rloo.py:
  1. Monkey-patch evaluation.countdown.compute_score to score the FIRST
     <answer> block.
  2. exec rloo_trainer/rloo.py — it imports compute_score AFTER the patch.

Adds CLI flags:
  --firstanswer_disable    revert to vanilla RLOO (control / A-B sanity)
"""

from __future__ import annotations

import os
import re
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RLOO_DIR = os.path.join(_REPO_ROOT, "rloo_trainer")
for path in (_RLOO_DIR, _REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _pop_switch(name: str) -> bool:
    present = False
    out: list[str] = []
    for token in sys.argv:
        if token == f"--{name}":
            present = True
            continue
        out.append(token)
    sys.argv[:] = out
    return present


def _install_firstanswer_reward() -> None:
    """Replace evaluation.countdown.compute_score to score the FIRST <answer> block."""
    import evaluation.countdown as countdown
    import random

    # Import the helpers we need
    validate_equation = countdown.validate_equation
    evaluate_equation = countdown.evaluate_equation

    def firstanswer_compute_score(
        solution_str, ground_truth, method="strict",
        format_score=0.1, score=1.0,
    ):
        """Score the FIRST <answer> block (not the last)."""
        target = ground_truth["target"]
        numbers = ground_truth["numbers"]

        matches = list(_ANSWER_RE.finditer(solution_str))
        if not matches:
            return 0
        first_answer = matches[0].group(1).strip()

        if not validate_equation(first_answer, numbers):
            return format_score

        try:
            result = evaluate_equation(first_answer)
            if result is None:
                return format_score
            if abs(result - target) < 1e-5:
                return score
            return format_score
        except Exception:
            return format_score

    firstanswer_compute_score.__name__ = "compute_score"
    firstanswer_compute_score.__wrapped__ = countdown.compute_score
    countdown.compute_score = firstanswer_compute_score
    print(
        "[firstanswer_rloo] FIRST-<answer> reward active: "
        "verifier scores the FIRST <answer> block (not the last). "
        "Hypothesis: this suppresses the rambling reward-hack.",
        flush=True,
    )


def main() -> None:
    disable = _pop_switch("firstanswer_disable")
    if not disable:
        _install_firstanswer_reward()
    else:
        print("[firstanswer_rloo] --firstanswer_disable set; using vanilla last-answer reward.", flush=True)

    rloo_path = os.path.join(_RLOO_DIR, "rloo.py")
    sys.argv[0] = rloo_path
    with open(rloo_path) as f:
        code = compile(f.read(), rloo_path, "exec")
    exec(code, {"__name__": "__main__", "__file__": rloo_path})


if __name__ == "__main__":
    main()
