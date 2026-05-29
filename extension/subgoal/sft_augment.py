"""Inject <subgoal> declarations into the warm-start SFT traces.

Source dataset: ``Asap7772/cog_behav_all_strategies``.
Goal: produce an augmented dataset that we SFT C_SFT on to get C_SFT_aug, the
initialization for C_process.

Person A. Day 3.

TODO:
    * Pull traces from HF.
    * For each trace:
        1. Parse arithmetic in the trace into a small expression DAG.
        2. Identify intermediate values that feed the final answer.
        3. For each intermediate, insert a <subgoal> reach <v> from [<inputs>]
           </subgoal> declaration immediately before the step that computes it.
        4. Validate the resulting trace is still well-formed (the final answer
           still parses, subgoals_are_reachable verifier passes on each).
    * Hand-spot-check 50 augmented examples before pushing the dataset.
    * Push to your own HF namespace (not asingh15's).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AugmentStats:
    n_traces: int
    n_augmented: int
    mean_subgoals_per_trace: float
    n_invalid_after_augment: int


def augment(input_dataset: str, output_dataset: str) -> AugmentStats:
    """TODO: implement. Returns summary stats for the spot-check report."""
    raise NotImplementedError
