#!/usr/bin/env bash
# Launch the full probe-best-of-K RLOO run from C_SFT.
#
# The code is already in main (commit 737dc74):
#   rloo_trainer/rloo_update_worker.py -- probe_topk_M masking branch
#   rloo_trainer/rloo.py               -- --probe_topk_M CLI flag
#
# This is the capstone "bridge experiment" -- it tests whether the same
# probe that gives +22.9 pp at deployment-time (best-of-16 selection) also
# helps when used as a selection signal during training (top-M of group_size
# rollouts contribute to the gradient; reward stays the verifier).
#
# Earlier we launched and KILLED this at step 8. This is the full re-run.
#
# Cost: ~$30 H100, ~5h wallclock.
# Run from repo root with .env containing WANDB_API_KEY + HF_TOKEN.

set -euo pipefail

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

modal run --detach modal_train.py rloo -- \
  --model_name asingh15/qwen-sft-countdown-defaultproj \
  --ref_model_name asingh15/qwen-sft-countdown-defaultproj \
  --tokenizer_name asingh15/qwen-sft-countdown-defaultproj \
  --wandb_project rloo_probe_topk_0.5b \
  --wandb_name probe_topk4_csft_FULL \
  --save_dir /vol/checkpoints/rloo_probe_topk_checkpoints \
  --batch_size 128 \
  --group_size 8 \
  --gradient_accumulation_steps 128 \
  --num_training_steps 100 \
  --save_every_n_steps 10 \
  --warmup_ratio 0 \
  --lr_schedule constant \
  --probe_baseline \
  --probe_topk_M 4
