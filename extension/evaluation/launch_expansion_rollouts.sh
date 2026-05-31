#!/bin/bash
# Phase 1 launcher: spawn 5 Modal jobs that sample fresh rollouts for the
# expansion + Task 2 (Option B dynamics) work.
#
#   Job 1: C_SFT on n=500 prompts -> eval_c_sft_n500.json
#   Job 2: C_outcome on n=500 prompts -> eval_c_outcome_n500.json
#   Job 3: C_outcome_step_30 on n=200 prompts -> eval_c_outcome_step_30_n200.json
#   Job 4: C_outcome_step_60 on n=200 prompts -> eval_c_outcome_step_60_n200.json
#   Job 5: C_outcome_step_90 on n=200 prompts -> eval_c_outcome_step_90_n200.json
#
# Outputs land in /vol/evaluation/eval_results/ on the Modal volume.
# Estimated wall time: ~2-3 hours (all 5 jobs run in parallel).
# Estimated cost: ~$20-30.
#
# Requires .env loaded (HF_TOKEN). Each job is spawned with --detach so it
# survives if this terminal closes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Input prompt files (paths inside the Modal image — repo is copied to /root/default_proj).
JSONL_N500="/root/default_proj/extension/data/countdown_eval_500.jsonl"
# Task 2 dynamics uses the same 500-problem pool, but only the first 200.

EVAL_OUT_DIR="/vol/evaluation/eval_results"

# Model paths.
C_SFT_MODEL="asingh15/qwen-sft-countdown-defaultproj"
C_OUTCOME_MODEL="/vol/checkpoints/rloo_checkpoints/rloo_training/rloo_fixed_v2/latest_checkpoint/model"
SNAPSHOT_TEMPLATE="/vol/checkpoints/rloo_checkpoints/rloo_training/rloo_fixed_v2/epoch_0_step_%s/model"

spawn_rollout() {
    local model_path="$1"
    local input_jsonl="$2"
    local max_prompts="$3"
    local output_name="$4"

    printf 'Spawning rollouts: %-22s -> %s\n' "${output_name}" "${EVAL_OUT_DIR}/${output_name}.json"
    modal run --detach "${PROJECT_ROOT}/modal_train.py" sample_local -- \
        --model_path "${model_path}" \
        --input_jsonl "${input_jsonl}" \
        --output_json "${EVAL_OUT_DIR}/${output_name}.json" \
        --num_responses 16 \
        --max_prompts "${max_prompts}"
}

echo "Launching 5 parallel rollout jobs..."

# Main expansion: C_SFT and C_outcome on all 500 prompts.
spawn_rollout "${C_SFT_MODEL}" "${JSONL_N500}" 500 "eval_c_sft_n500"
spawn_rollout "${C_OUTCOME_MODEL}" "${JSONL_N500}" 500 "eval_c_outcome_n500"

# Task 2 (Option B dynamics): snapshots 30, 60, 90 on a 200-prompt subset.
for step in 30 60 90; do
    model_path="$(printf "${SNAPSHOT_TEMPLATE}" "${step}")"
    spawn_rollout "${model_path}" "${JSONL_N500}" 200 "eval_c_outcome_step_${step}_n200"
done

printf '\nAll 5 spawn jobs submitted. Monitor on the Modal dashboard.\n'
printf 'When all done, %s should contain:\n' "${EVAL_OUT_DIR}"
printf '  eval_c_sft_n500.json\n'
printf '  eval_c_outcome_n500.json\n'
printf '  eval_c_outcome_step_30_n200.json\n'
printf '  eval_c_outcome_step_60_n200.json\n'
printf '  eval_c_outcome_step_90_n200.json\n'
printf '\nNEXT: once these are all done, run extension/probe/launch_expansion_cache.sh\n'
