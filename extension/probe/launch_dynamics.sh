#!/bin/bash

# Layer C dynamics: cache hidden states on each of the 10 saved C_outcome
# snapshots, using the same rollouts as the final-checkpoint eval (Option A).
#
# Spawns 10 Modal jobs in parallel (each ~30 min on one H100) so wall-time
# is ~30 min, not 5 hours. Total cost ~$10-15.
#
# All outputs land at /vol/probe_cache_dynamics/<checkpoint>_l<N>_<kind>.npz
# (a separate dir from the main /vol/probe_cache so the two analyses don't
# tangle).
#
# Requires .env to be loaded for HF_TOKEN/WANDB.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

EVAL_JSON="${EVAL_JSON:-/vol/evaluation/eval_results/rloo_fixed_v2_passk.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/vol/probe_cache_dynamics}"

for step in 0 10 20 30 40 50 60 70 80 90; do
    model_path="/vol/checkpoints/rloo_checkpoints/rloo_training/rloo_fixed_v2/epoch_0_step_${step}/model"
    # Zero-pad the step in the checkpoint name so files sort lexically:
    # C_outcome_step_00, ..., C_outcome_step_90
    checkpoint_name="C_outcome_step_$(printf %02d "${step}")"

    printf 'Launching snapshot step=%s -> %s\n' "${step}" "${checkpoint_name}"
    modal run --detach "${PROJECT_ROOT}/modal_train.py" probe_cache -- \
        --model_path "${model_path}" \
        --eval_json "${EVAL_JSON}" \
        --checkpoint_name "${checkpoint_name}" \
        --output_dir "${OUTPUT_DIR}"
done

printf '\nAll 10 spawn jobs submitted. Monitor on the Modal dashboard.\n'
printf 'When all done, %s should contain 90 files (10 steps x 3 layers x 3 kinds).\n' "${OUTPUT_DIR}"
