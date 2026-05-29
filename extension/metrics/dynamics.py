"""Layer C: replay Layer A + Layer B metrics on the saved C_outcome snapshots.

The C_outcome run saved a checkpoint every 10 steps at
``/vol/checkpoints/rloo_checkpoints/rloo_training/rloo_fixed_v2/epoch_0_step_{0,10,...,90}/model``
plus ``latest_checkpoint``. For each one we:

    1. Run ``evaluation/countdown_eval.py`` with --output_name <step>.
    2. Apply metrics/behavioral.py to the resulting eval JSON.
    3. Apply probe/eval_probe.py once Person B's pipeline is ready.

The product is a per-step trajectory of accuracy, verbalized AUROC, probe AUROC,
and the concealment gap. This is the headline plot.

Person A drives the eval-launching part (Modal). Person B drives the
probe-replay part.

TODO:
    * Write a small launcher that loops the eval over each snapshot.
    * Cache per-step eval JSONs in extension/outputs/dynamics/.
    * Produce dynamics_curve.png (matplotlib).
"""

from __future__ import annotations

SNAPSHOT_STEPS = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)
CHECKPOINT_PATH_TEMPLATE = (
    "/vol/checkpoints/rloo_checkpoints/rloo_training/rloo_fixed_v2/"
    "epoch_0_step_{step}/model"
)


def per_step_eval_paths(out_dir: str) -> dict[int, str]:
    """Convenience: per-step eval JSON paths after the Modal eval has run."""
    return {step: f"{out_dir}/step_{step:03d}.json" for step in SNAPSHOT_STEPS}
