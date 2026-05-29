"""Compose R = R_outcome + lambda * R_subgoal.

Outcome reward is the existing Countdown verifier (0.0 / 0.1 / 1.0).
Subgoal reward, per extension.md §4.2:

    R_subgoal = clip01(  (n_valid_and_achieved - alpha * n_invalid) / max(n_declared, 1)  )

Person A. Day 2.

TODO:
    * Compute n_declared / n_valid_and_achieved / n_invalid from parse_subgoals
      and verifier.is_reachable / is_achieved.
    * Decide how to segment the trace between consecutive subgoals (the body
      between sg_i.end and sg_{i+1}.start, last one ends at </think>).
    * Smoke-test compute_reward against hand-crafted rollouts including:
        - zero subgoals (R_subgoal must be 0 or NaN, document choice)
        - all subgoals valid+achieved
        - subgoals valid but not achieved
        - subgoals not even reachable
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .parser import Subgoal, parse_subgoals
from .verifier import is_achieved, is_reachable


THINK_CLOSE = "</think>"


@dataclass
class SubgoalReport:
    n_declared: int
    n_reachable: int       # valid in the is_reachable sense
    n_achieved: int        # reachable AND actually computed in the trace
    n_invalid: int         # not reachable
    subgoal_score: float   # the clipped R_subgoal in [0, 1]


def evaluate_trace_subgoals(trace: str, problem_numbers: Sequence[int],
                            alpha: float = 1.0) -> SubgoalReport:
    """Score the subgoal channel of a single rollout.

    Args:
        trace: full model response (includes <think> body).
        problem_numbers: the numbers available to the original problem; subgoals
            must use a subset of these (and we may want to verify this here).
        alpha: penalty weight on invalid subgoals.

    TODO: implement.
    """
    raise NotImplementedError


def compute_reward(outcome_score: float, trace: str, problem_numbers: Sequence[int],
                   lam: float = 0.3, alpha: float = 1.0) -> float:
    """Final composite reward, per §4.2 of extension.md.

    TODO: implement. Will be plugged into rloo.py's reward computation for
    C_process. Until then, this is a pure function students can unit-test.
    """
    raise NotImplementedError
