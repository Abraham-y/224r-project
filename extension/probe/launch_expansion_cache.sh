#!/bin/bash
# Phase 2 launcher: spawn 5 Modal jobs that cache hidden states on the
# Phase 1 rollouts.
#
#   Job 1: C_SFT cache on n=500 rollouts -> /vol/probe_cache/C_SFT_n500_*
#   Job 2: C_outcome cache on n=500 rollouts -> /vol/probe_cache/C_outcome_n500_*
#   Jobs 3-5: snapshot caches (step 30/60/90) on n=200 rollouts ->
#             /vol/probe_cache_dynamics_optB/C_outcome_step_{30,60,90}_*
#
# The new cache dirs are kept separate so we do NOT overwrite the existing
# n=50 caches in /vol/probe_cache or the Option-A dynamics caches in
# /vol/probe_cache_dynamics.
#
# Estimated wall time: ~4-5 hours (parallel).
# Estimated cost: ~$55-65.
#
# Pre-req: launch_expansion_rollouts.sh has completed, eval JSONs exist at
# /vol/evaluation/eval_results/eval_c_{sft,outcome}_n500.json and
# eval_c_outcome_step_{30,60,90}_n200.json.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

EVAL_DIR="/vol/evaluation/eval_results"
PROBE_OUT="/vol/probe_cache_n500"
DYNAMICS_OUT="/vol/probe_cache_dynamics_optB"

# Model paths (must match the rollout launcher).
C_SFT_MODEL="asingh15/qwen-sft-countdown-defaultproj"
C_OUTCOME_MODEL="/vol/checkpoints/rloo_checkpoints/rloo_training/rloo_fixed_v2/latest_checkpoint/model"
SNAPSHOT_TEMPLATE="/vol/checkpoints/rloo_checkpoints/rloo_training/rloo_fixed_v2/epoch_0_step_%s/model"

spawn_cache() {
    local model_path="$1"
    local eval_json="$2"
    local checkpoint_name="$3"
    local output_dir="$4"

    printf 'Spawning cache: %-30s -> %s\n' "${checkpoint_name}" "${output_dir}"
    modal run --detach "${PROJECT_ROOT}/modal_train.py" probe_cache -- \
        --model_path "${model_path}" \
        --eval_json "${eval_json}" \
        --checkpoint_name "${checkpoint_name}" \
        --output_dir "${output_dir}"
}

echo "Launching 5 parallel cache jobs..."

spawn_cache "${C_SFT_MODEL}"     "${EVAL_DIR}/eval_c_sft_n500.json"     "C_SFT"     "${PROBE_OUT}"
spawn_cache "${C_OUTCOME_MODEL}" "${EVAL_DIR}/eval_c_outcome_n500.json" "C_outcome" "${PROBE_OUT}"

for step in 30 60 90; do
    model_path="$(printf "${SNAPSHOT_TEMPLATE}" "${step}")"
    spawn_cache "${model_path}" \
        "${EVAL_DIR}/eval_c_outcome_step_${step}_n200.json" \
        "C_outcome_step_${step}" \
        "${DYNAMICS_OUT}"
done

printf '\nAll 5 cache spawn jobs submitted. Monitor on the Modal dashboard.\n'
printf 'When done:\n'
printf '  %s/  -> {C_SFT,C_outcome}_l{12,16,20}_{pre_answer,assertion,neutral}.npz\n' "${PROBE_OUT}"
printf '  %s/  -> C_outcome_step_{30,60,90}_l*_*.npz\n' "${DYNAMICS_OUT}"
