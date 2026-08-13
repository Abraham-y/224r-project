"""Shared utilities for the probe-fragility follow-up.

Nothing here trains or samples a model. These modules operate on:
  - cached activations   (see `activations.py`, layout `acts/{run_id}/{step}/{layer}.npy`)
  - eval-rollout JSONs   (see `labels.py`)
  - pickled sklearn probes from the original paper (`extension/cache/steering/*.pkl`)

Model-touching code lives in `experiments/fragility/*/` and is launched via
`modal_fragility.py`.
"""

from __future__ import annotations

__all__ = [
    "activations",
    "checkpoint_logging",
    "confounds",
    "figures",
    "geometry",
    "labels",
    "metrics_io",
    "paths",
    "probes",
    "registry",
    "steering",
]
