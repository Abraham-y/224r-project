"""Compose R = R_outcome + lambda * R_subgoal.

Outcome reward is the existing Countdown verifier (0.0 / 0.1 / 1.0).
Subgoal reward, per extension.md §4.2:

    R_subgoal = clip01( (n_valid_and_achieved - alpha * n_invalid) / max(n_declared, 1) )

where:
    * n_declared           -- total <subgoal> ... </subgoal> blocks the model emitted
    * n_valid              -- subgoals whose (target, available) is reachable via +,-,*,/
                              AND whose declared `available` is a subset of the problem's numbers
    * n_achieved           -- valid subgoals whose subsequent trace segment actually
                              computes the declared target using only the available subset
    * n_invalid            -- n_declared - n_valid

Trace segmentation: for the i-th subgoal we take the slice between its closing
``</subgoal>`` tag and either the next subgoal's opening tag or ``</think>``,
and run :func:`is_achieved` over that slice.

Person A. Day 2.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from .parser import Subgoal, parse_subgoals
from .verifier import is_achieved, is_reachable


THINK_CLOSE = "</think>"


@dataclass
class SubgoalReport:
    n_declared: int
    n_valid: int           # reachable AND uses a subset of the problem numbers
    n_achieved: int        # valid AND computed in the subsequent trace segment
    n_invalid: int         # n_declared - n_valid
    subgoal_score: float   # the clipped R_subgoal in [0, 1]


def _is_multiset_subset(needle: Sequence[int], haystack: Sequence[int]) -> bool:
    """True iff every element of needle appears at least as many times in haystack."""
    n_count = Counter(needle)
    h_count = Counter(haystack)
    return all(h_count[k] >= v for k, v in n_count.items())


def _segments(trace: str, subgoals: Sequence[Subgoal]) -> list[str]:
    """For each subgoal, return the trace segment between its close-tag and the next boundary.

    The next boundary is either the next subgoal's open-tag or ``</think>`` (whichever comes
    first), or the end of the trace if neither is present.
    """
    out: list[str] = []
    for i, sg in enumerate(subgoals):
        start = sg.end
        end = len(trace)
        if i + 1 < len(subgoals):
            end = min(end, subgoals[i + 1].start)
        # Cap at </think> if it appears in this segment.
        think_idx = trace.find(THINK_CLOSE, start, end)
        if think_idx != -1:
            end = think_idx
        out.append(trace[start:end])
    return out


def evaluate_trace_subgoals(
    trace: str, problem_numbers: Sequence[int], alpha: float = 1.0
) -> SubgoalReport:
    """Score the subgoal channel of a single rollout."""
    subgoals = parse_subgoals(trace)
    segments = _segments(trace, subgoals)

    n_declared = len(subgoals)
    n_valid = 0
    n_achieved = 0
    # Subgoals can reference original problem numbers *or* the targets of
    # previously-achieved subgoals (e.g. "reach 68 from [60, 8]" after a prior
    # "reach 60 from [3, 4, 5]"). Track the growing pool of allowed sources.
    allowed: list[int] = list(problem_numbers)
    for sg, segment in zip(subgoals, segments):
        uses_allowed = _is_multiset_subset(sg.available, allowed)
        reachable = is_reachable(sg.target, sg.available)
        if uses_allowed and reachable:
            n_valid += 1
            if is_achieved(sg.target, sg.available, segment):
                n_achieved += 1
                allowed.append(sg.target)
    n_invalid = n_declared - n_valid

    raw = (n_achieved - alpha * n_invalid) / max(n_declared, 1)
    subgoal_score = max(0.0, min(1.0, raw))
    return SubgoalReport(
        n_declared=n_declared,
        n_valid=n_valid,
        n_achieved=n_achieved,
        n_invalid=n_invalid,
        subgoal_score=subgoal_score,
    )


def compute_reward(
    outcome_score: float,
    trace: str,
    problem_numbers: Sequence[int],
    lam: float = 0.3,
    alpha: float = 1.0,
) -> float:
    """Final composite reward: R = R_outcome + lam * R_subgoal.

    Hooks into rloo.py's reward computation for ``C_process``. Pure function;
    unit-testable without a GPU.
    """
    report = evaluate_trace_subgoals(trace, problem_numbers, alpha=alpha)
    return outcome_score + lam * report.subgoal_score


# ---------------------------------------------------------------------------
# Smoke test. Run with: python extension/subgoal/reward.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    nums = [3, 4, 5, 8]
    cases: list[tuple[str, str, float | None, int, int, int]] = [
        # (label, trace, expected_subgoal_score_or_None, n_declared, n_valid, n_achieved)
        (
            "zero subgoals -> score 0",
            "no subgoals here. answer 24.",
            0.0, 0, 0, 0,
        ),
        (
            "one subgoal, valid and achieved",
            "<subgoal> reach 60 from [3, 4, 5] </subgoal> 4*5=20, 20*3=60. </think>",
            1.0, 1, 1, 1,
        ),
        (
            "one subgoal, valid but not achieved",
            "<subgoal> reach 60 from [3, 4, 5] </subgoal> hmm 3+4=7 dead end. </think>",
            0.0, 1, 1, 0,
        ),
        (
            "one subgoal, unreachable target (alpha=1) -> -1 clipped to 0",
            "<subgoal> reach 61 from [3, 4, 5] </subgoal> can't reach. </think>",
            0.0, 1, 0, 0,
        ),
        (
            "two subgoals, both valid and achieved",
            ("<subgoal> reach 60 from [3, 4, 5] </subgoal> 4*5=20, 20*3=60. "
             "<subgoal> reach 68 from [60, 8] </subgoal> 60+8=68. </think>"),
            1.0, 2, 2, 2,
        ),
        (
            "subgoal uses a number not in the problem",
            "<subgoal> reach 12 from [3, 99] </subgoal> 99/3=33. </think>",
            0.0, 1, 0, 0,
        ),
    ]
    failed = 0
    for label, trace, expected_score, exp_decl, exp_valid, exp_achieved in cases:
        rep = evaluate_trace_subgoals(trace, nums)
        ok = (
            rep.n_declared == exp_decl
            and rep.n_valid == exp_valid
            and rep.n_achieved == exp_achieved
            and (expected_score is None or abs(rep.subgoal_score - expected_score) < 1e-6)
        )
        marker = "ok " if ok else "FAIL"
        if not ok:
            failed += 1
        print(
            f"{marker}  {label}: declared={rep.n_declared} valid={rep.n_valid} "
            f"achieved={rep.n_achieved} score={rep.subgoal_score:.3f}"
        )

    # Test composite reward end-to-end.
    composite = compute_reward(
        outcome_score=1.0,
        trace="<subgoal> reach 60 from [3, 4, 5] </subgoal> 4*5=20, 20*3=60. </think>",
        problem_numbers=nums,
        lam=0.3,
    )
    expected_composite = 1.0 + 0.3 * 1.0
    ok = abs(composite - expected_composite) < 1e-6
    print(f"{'ok ' if ok else 'FAIL'}  compute_reward(...) = {composite:.3f} (expected {expected_composite:.3f})")
    if not ok:
        failed += 1

    if failed:
        raise SystemExit(f"{failed} case(s) failed")
    print("\nAll reward smoke tests passed.")
