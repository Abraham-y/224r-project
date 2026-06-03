#!/usr/bin/env bash
# Eval all saved checkpoints from the multiplicative RLOO run.
#
# The trainer saves every 10 steps (step_10, step_20, ..., step_90) plus
# latest_checkpoint (the final state after step 99). We eval each on the
# clean-406 held-out set using the same sampler as our other evals
# (temperature=1.0 with both <|endoftext|> and <|im_end|> as stops) for
# direct comparison to the existing C_SFT_FIXEDSTOP / C_outcome_FIXEDSTOP
# eval JSONs.
#
# After all evals complete, scripts/compute_verifier_acc.py reads each JSON
# and prints a clean per-step verifier accuracy curve for the report.
#
# Cost: ~$2 per eval x 11 checkpoints = ~$22. Time: each eval ~12-15 min
# on Modal, parallelizable but sequential is fine (~2h total).

set -euo pipefail

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

CKPT_BASE="/vol/checkpoints/rloo_probe_mult_checkpoints/rloo_probe_mult_0.5b/probe_mult_csft_run1"
EVAL_OUT_BASE="/vol/evaluation/eval_results"

# Saved checkpoints: step_10, step_20, ..., step_90 + latest_checkpoint
STEPS=(10 20 30 40 50 60 70 80 90)

for STEP in "${STEPS[@]}"; do
  echo ">>> launching eval for step ${STEP}..."
  modal run --detach modal_train.py sample_local -- \
    --model_path "${CKPT_BASE}/epoch_0_step_${STEP}/model" \
    --input_jsonl /root/default_proj/extension/data/countdown_eval_500.jsonl \
    --output_json "${EVAL_OUT_BASE}/eval_probe_mult_step${STEP}_n500.json" \
    --num_responses 16 --max_prompts 500 \
    --temperature 1.0 --top_p 1.0 --top_k -1 \
    --extra_stop_token_ids 151645
done

echo ">>> launching final-checkpoint eval (latest_checkpoint)..."
modal run --detach modal_train.py sample_local -- \
  --model_path "${CKPT_BASE}/latest_checkpoint/model" \
  --input_jsonl /root/default_proj/extension/data/countdown_eval_500.jsonl \
  --output_json "${EVAL_OUT_BASE}/eval_probe_mult_final_n500.json" \
  --num_responses 16 --max_prompts 500 \
  --temperature 1.0 --top_p 1.0 --top_k -1 \
  --extra_stop_token_ids 151645

echo ">>> all 10 evals spawned. monitor on Modal dashboard."
echo ">>> when they finish, download with:"
echo "    modal volume get default-proj-training evaluation/eval_results/eval_probe_mult_step10_n500.json ."
echo "    (etc.)"
echo ">>> then run: python scripts/compute_verifier_acc.py"
