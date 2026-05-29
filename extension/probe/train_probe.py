"""Per-checkpoint, per-(layer, position) logistic regression probe.

Person B. CPU sklearn; runs on the laptop.

Pipeline (per §4.1 of extension.md):
    1. Load cached activations (.npz from cache_hidden_states.py).
    2. Split by *problem id*, not trajectory, for 5-fold CV.
    3. Fit sklearn LogisticRegression(penalty='l2', C=0.1, max_iter=2000) on
       standardized inputs (StandardScaler).
    4. Report:
        - AUROC (held-out trajectory)
        - AUROC (held-out problem)
        - Shuffled-label AUROC (sanity check; should be ~0.5)
        - Random-direction probe AUROC (sanity check; should be ~0.5)
        - Optional: MLP and RF baselines (linear should match or exceed)

TODO:
    * Implement train_probe(cache_path) -> ProbeResult.
    * Persist trained probe coefficients so they can be reused in transfer.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProbeResult:
    auroc_traj_holdout: float
    auroc_problem_holdout: float
    auroc_shuffled: float
    auroc_random_direction: float


def train_probe(cache_path: str) -> ProbeResult:
    """TODO: implement."""
    raise NotImplementedError
