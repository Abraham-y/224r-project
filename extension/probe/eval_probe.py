"""Concealment gap, probe earliness, within-problem Cohen's d.

Person B. CPU. Runs on the laptop once probes are trained and
verbalized-confidence elicitation from metrics/calibration.py is in hand.

Quantities (per §3.2 Layer B of extension.md):

    concealment_gap = probe_auroc - verbalized_auroc      # headline number
    probe_earliness  = earliest token position at which sliding-window probe
                       AUROC exceeds 0.7
    cohens_d         = within-problem effect size between correct and wrong
                       trajectories for probe score

TODO:
    * Implement compute_concealment_gap(probe_results, calibration_results).
    * Implement probe_earliness(activations_per_position, labels).
    * Implement within_problem_cohens_d(probe_scores_per_problem, labels).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LayerBSummary:
    probe_auroc: float
    verbalized_auroc: float
    concealment_gap: float
    earliness_pos: int | None
    within_problem_d: float


def summarize(probe_result, calibration_result) -> LayerBSummary:
    """TODO: implement."""
    raise NotImplementedError
