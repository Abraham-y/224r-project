#!/usr/bin/env bash
# =============================================================================
# Item 4 (highest-value $30 spend): causal-steering DOSE-RESPONSE over training.
#
# Goal: show the optimized probe direction becoming PROGRESSIVELY causal as the
# probe-as-reward run (runA) proceeds. We already have the two endpoints:
#   step 0  (vanilla C_outcome)        -> extension/outputs/n500/causal_steering_full.jsonl   (Delta@a=1.0 = +0.021, n.s.)
#   step 99 (final post-Goodhart runA) -> causal_steering_runA_postRL.jsonl                   (Delta@a=1.0 = +0.082, p=0.017)
# This script fills in the MISSING MIDDLE: runA checkpoints at step 40 and 60.
# Result is a clean curve Delta(step): ~0 -> ... -> +0.08, which converts the
# single significant point into a dose-response (kills "one noisy point").
#
# Item 5 (bigger n) is baked in via N_PROMPTS below: the new points are run at
# ~500 prefixes so their CIs are tight from the start. (Re-running the existing
# step-0/step-99 files at larger n is optional now that item 1 showed step 99 is
# already significant at n=194.)
#
# !!! PREREQUISITE: the RL checkpoints no longer exist. !!!
# As of this writing the Modal volume `default-proj-training` /checkpoints holds
# ONLY the SFT model -- C_outcome, runA, runB, top-K were all cleaned up, and
# nothing was pushed to HuggingFace. So step 40/60 (and 0/99) weights are GONE.
# You must REGENERATE a fresh, self-contained runA first (steps 1-2 below), then
# steer it (step 3). Because RL is stochastic, do NOT mix the new step-40/60 with
# the old causal_steering_runA_postRL.jsonl -- steer the NEW run's step-99 too.
#
# Cost (full): C_outcome retrain ~$90 + runA retrain ~$90 + steering ~$20 ~= $200.
# This is partner-compute territory, NOT the $30. Item 1 already made the headline
# result significant at existing n, so this dose-response is optional strengthening.
# Save checkpoints this time (--save_every_n_steps 10) and push the keepers to HF.
# Run from repo ROOT with a .env containing HF_TOKEN (+ WANDB_API_KEY if needed).
# =============================================================================
set -euo pipefail
if [[ -f .env ]]; then set -a; source .env; set +a; fi

# ---- STEP 1: re-train C_outcome (100 RLOO from SFT, saving every 10 steps) ----
# modal run --detach modal_train.py rloo -- \
#   --model_name asingh15/qwen-sft-countdown-defaultproj \
#   --ref_model_name asingh15/qwen-sft-countdown-defaultproj \
#   --tokenizer_name asingh15/qwen-sft-countdown-defaultproj \
#   --num_training_steps 100 --save_every_n_steps 10 \
#   --save_dir /vol/checkpoints/rloo_checkpoints \
#   --wandb_project rloo_outcome --wandb_name C_outcome_repro
#
# ---- STEP 2: re-train runA (100 probe-reward RLOO from the new C_outcome) -----
# Init the policy from the new C_outcome step-99 model (check probe_reward_rloo.py
# for its init-model flag). Save every 10 steps so 40/60/99 are snapshotted.
# modal run --detach modal_train.py probe_reward_rloo -- \
#   --num_training_steps 100 --save_every_n_steps 10 --reward_mode probe \
#   --save_dir /vol/checkpoints/rloo_reward_ablation \
#   --wandb_project rloo_reward_ablation --wandb_name probe_reward_v1   # + C_outcome init path
#
# ---- STEP 3: steer the fresh runA snapshots (below). Set RUNA_SAVE_DIR to the
#      step-2 save path: <save_dir>/<wandb_project>/<wandb_name> ----------------
RUNA_SAVE_DIR="/vol/checkpoints/rloo_reward_ablation/rloo_reward_ablation/probe_reward_v1"  # <-- VERIFY after step 2
# Rollouts file that supplies the steering prefixes (prompt + up to </think>).
# Use the SAME eval rollouts you used for the step-0/step-99 steering runs so the
# prefix set is comparable across checkpoints.
EVAL_JSON="/vol/data/countdown_eval_500.jsonl"                                   # <-- VERIFY
# -----------------------------------------------------------------------------

STEER_VEC="extension/cache/steering/C_outcome_l16_pre_answer_direction.npz"  # the OPTIMIZED direction
N_PROMPTS=250            # x2 rollouts/prompt -> ~500 prefixes (item 5: bigger n on the new points)
N_ROLLOUTS=2
ALPHAS="0.0 0.5 1.0 2.0"
LAYER=16
SEED=42

# Steer runA snapshots at 40/60/99 (all from the SAME fresh run). Step 0 of the
# curve is the new C_outcome itself (steer it separately, using its step-1 path).
for STEP in 40 60 99; do
  echo ">>> steering runA checkpoint at step ${STEP}"
  modal run --detach modal_train.py causal_steering -- \
    --model_path   "${RUNA_SAVE_DIR}/epoch_0_step_${STEP}/model" \
    --eval_json    "${EVAL_JSON}" \
    --steer_vec    "${STEER_VEC}" \
    --n_prompts    "${N_PROMPTS}" \
    --n_rollouts_per_prompt "${N_ROLLOUTS}" \
    --alphas       ${ALPHAS} \
    --layer        "${LAYER}" \
    --seed         "${SEED}" \
    --output_jsonl "/vol/outputs/causal_steering_runA_step${STEP}.jsonl"
done

echo
echo "When both jobs finish, download the two JSONLs from the Modal volume to repo root, then run:"
echo "  python extension/probe/causal_steering_stats.py --dose \\"
echo "      0:extension/outputs/n500/causal_steering_full.jsonl \\"
echo "      40:causal_steering_runA_step40.jsonl \\"
echo "      60:causal_steering_runA_step60.jsonl \\"
echo "      99:causal_steering_runA_postRL.jsonl"
echo
echo "OPTIONAL item 5 (tighter CIs on existing points) -- re-run vanilla/final/assertion at N_PROMPTS=250:"
echo "  # vanilla:   --model_path <C_outcome final>/model      --steer_vec C_outcome_l16_pre_answer_direction.npz"
echo "  # assertion: --model_path <runA step99>/model          --steer_vec C_outcome_l16_assertion_direction.npz"
