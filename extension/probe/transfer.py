"""Cross-checkpoint probe transfer: 3x3 AUROC matrix.

Train probe on checkpoint_i activations; evaluate it (no retraining) on
checkpoint_j activations. Repeat for every (i, j). Matrix interpretation:

    * Degraded off-diagonal -> representation drift (Taufeeque et al. lens).
    * Preserved off-diagonal -> signal suppression without drift.

Person B. CPU, depends on train_probe.py + cache_hidden_states.py outputs.

TODO:
    * Implement load_probe_and_apply(probe, target_cache).
    * Produce a 3x3 matrix CSV and a heatmap PNG.
    * Optionally extend the matrix with the C_outcome step-50 snapshot to spot
      the inflection point.
"""

from __future__ import annotations

CHECKPOINT_NAMES = ("C_SFT", "C_outcome", "C_process")
