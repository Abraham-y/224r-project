# Runbook: Clean Probe Pipeline Replication + Probe-Augmented RLOO

Author: Abraham (code), Anagha (compute / execution).
Last updated: 2026-06-03.

## What this runbook does

1. Re-evaluates `C_SFT` and `C_outcome` with the **sampler bug fixed**
   (stops at `</answer>`, matching the training-time sampler) at the
   project's original eval sampling params (temp=0.6, top_p=0.95, top_k=20).
2. Re-runs the probe pipeline on the fixed-sampler caches and confirms
   probe AUROC numbers are robust to the sampling regime.
3. Trains a probe-augmented RLOO model with a lambda-mix LOO baseline,
   sweeping lambda in {0.3, 0.5, 0.7}.
4. Evaluates the resulting checkpoints under the fixed sampler.

Total expected cost: ~$100 Modal H100 + ~6h elapsed (mostly parallel).

---

## Step 0: prerequisites

- HF_TOKEN, WANDB_API_KEY in your shell env (or in `./.env`).
- Modal account with credits (each RLOO run ~$30, each eval ~$2).
- `pip install modal` (and `modal token new` if you haven't auth'd).
- From repo root.

```bash
set -a; source .env; set +a   # if using .env
```

---

## Step 1: sampler fix (already in main)

The fix is already merged. `extension/evaluation/sample_local_jsonl.py`
now accepts `--stop_strings "</answer>"`. No code action needed.

---

## Step 2: regenerate clean rollouts (parallel)

Launch both evals in parallel. ~$2 each, ~15-20 min wallclock per job.

```bash
# C_SFT, fixed sampler, original eval params
modal run --detach modal_train.py sample_local -- \
  --model_path asingh15/qwen-sft-countdown-defaultproj \
  --input_jsonl /root/default_proj/extension/data/countdown_eval_500.jsonl \
  --output_json /vol/evaluation/eval_results/eval_c_sft_n500_fixed.json \
  --num_responses 16 --max_prompts 500 \
  --temperature 0.6 --top_p 0.95 --top_k 20 \
  --stop_strings '</answer>'

# C_outcome, fixed sampler, original eval params
modal run --detach modal_train.py sample_local -- \
  --model_path /vol/checkpoints/rloo_checkpoints/rloo_training/rloo_fixed_v2/latest_checkpoint/model \
  --input_jsonl /root/default_proj/extension/data/countdown_eval_500.jsonl \
  --output_json /vol/evaluation/eval_results/eval_c_outcome_n500_fixed.json \
  --num_responses 16 --max_prompts 500 \
  --temperature 0.6 --top_p 0.95 --top_k 20 \
  --stop_strings '</answer>'
```

When both finish, download the JSONs to repo root for the next step:

```bash
modal volume get default-proj-training \
  evaluation/eval_results/eval_c_sft_n500_fixed.json eval_c_sft_n500_fixed.json
modal volume get default-proj-training \
  evaluation/eval_results/eval_c_outcome_n500_fixed.json eval_c_outcome_n500_fixed.json
```

Sanity check: `python -c "import json,re; rows=[json.loads(l) for l in open('eval_c_sft_n500_fixed.json')]; print('mean blocks:', sum(len(re.findall(r'<answer>', r)) for row in rows for r in row['response'])/sum(len(row['response']) for row in rows))"`
should print ~1.0 (single-block rollouts confirms the stop fix worked).

---

## Step 3: re-cache + re-run probe pipeline

### 3a: cache hidden states from fixed rollouts (Modal, parallel)

```bash
# C_SFT cache
modal run --detach modal_train.py probe_cache -- \
  --eval_json /root/default_proj/eval_c_sft_n500_fixed.json \
  --model_name asingh15/qwen-sft-countdown-defaultproj \
  --ckpt_tag C_SFT --layers 12,16,20 \
  --output_dir /vol/probe_cache_n500_fixed

# C_outcome cache
modal run --detach modal_train.py probe_cache -- \
  --eval_json /root/default_proj/eval_c_outcome_n500_fixed.json \
  --model_name /vol/checkpoints/rloo_checkpoints/rloo_training/rloo_fixed_v2/latest_checkpoint/model \
  --ckpt_tag C_outcome --layers 12,16,20 \
  --output_dir /vol/probe_cache_n500_fixed
```

(Each ~$1.50, ~15-20 min.)

When done, pull the cache to local for the relabel scripts:

```bash
modal volume get default-proj-training probe_cache_n500_fixed extension/cache/probe_cache_n500_fixed
```

### 3b: filter to clean-406 (local, instant)

```bash
python extension/probe/filter_to_clean.py \
  --src_dir extension/cache/probe_cache_n500_fixed \
  --dst_dir extension/cache/probe_cache_n500_fixed_clean406
```

### 3c: relabel + downstream AUROC table (local, ~1 min)

```bash
python extension/probe/relabel_full_grid.py \
  --cache_dir extension/cache/probe_cache_n500_fixed_clean406 \
  --eval_sft eval_c_sft_n500_fixed.json \
  --eval_outcome eval_c_outcome_n500_fixed.json \
  --out_txt extension/outputs/n500/text/29b_relabel_full_grid_FIXED.txt

python extension/probe/relabel_redo_downstream.py \
  --cache_dir extension/cache/probe_cache_n500_fixed_clean406
```

### 3d: cross-checkpoint transfer (local, ~1 min)

```bash
python extension/probe/relabel_cross_checkpoint.py \
  --cache_dir extension/cache/probe_cache_n500_fixed_clean406 \
  --eval_sft eval_c_sft_n500_fixed.json \
  --eval_outcome eval_c_outcome_n500_fixed.json
```

### 3e: numbers to extract

Print the table of:

| | pre_answer L16 | assertion L16 | gap (pre - ass) | matched-pair MW p |
|---|---|---|---|---|
| C_SFT (fixed) | ? | ? | ? | — |
| C_outcome (fixed) | ? | ? | ? | (Mann-Whitney between) |

Compare to buggy numbers (from writeup `2.1` / `2.5`):

- buggy C_SFT: pre 0.904, ass 0.887, gap +0.017
- buggy C_outcome: pre 0.980, ass 0.852, gap +0.127
- buggy MW p: 0.68 (NS)

If fixed numbers are within ~0.02 AUROC of buggy: the probe AUROCs are
sampling-regime-independent (expected -- the </think> hidden state is at a
position before any post-`</answer>` rambling). If they differ materially,
that's a finding in itself.

### 3f: save the fixed probe direction + pkl (local)

After re-caching, regenerate the probe direction + Pipeline pkl on the fixed
cache (this is what step 4 needs):

```bash
# Edit extension/probe/save_probe_vector.py: change CACHE to point at
# extension/cache/probe_cache_n500_fixed_clean406, then run:
python extension/probe/save_probe_vector.py
```

Outputs:
- `extension/cache/steering/C_outcome_l16_pre_answer_direction.npz`
- `extension/cache/steering/C_outcome_l16_pre_answer_pipeline.pkl`  <-- step 4 uses this

Then push the pkl into the Modal volume so probe_augmented_rloo can load it:

```bash
modal volume put default-proj-training \
  extension/cache/steering/C_outcome_l16_pre_answer_pipeline.pkl \
  steering/C_outcome_l16_pre_answer_pipeline.pkl
```

---

## Step 4: probe-augmented RLOO (lambda-mix sweep)

Three runs, init from C_SFT, 100 RLOO steps each. ~$30, ~5h per run. Run in
parallel to wallclock ~5h.

```bash
# lambda = 0.3 (probe-heavy mix)
modal run --detach modal_train.py probe_augmented_rloo -- \
  --lambda_mix 0.3 \
  --probe /vol/steering/C_outcome_l16_pre_answer_pipeline.pkl \
  --model_name asingh15/qwen-sft-countdown-defaultproj \
  --ref_model_name asingh15/qwen-sft-countdown-defaultproj \
  --tokenizer_name asingh15/qwen-sft-countdown-defaultproj \
  --wandb_project rloo_probe_aug_0.5b --wandb_name probe_aug_lam03 \
  --save_dir /vol/checkpoints/rloo_probe_aug_checkpoints \
  --batch_size 128 --group_size 8 --gradient_accumulation_steps 128 \
  --num_training_steps 100 --save_every_n_steps 10 \
  --warmup_ratio 0 --lr_schedule constant

# lambda = 0.5 (balanced)
modal run --detach modal_train.py probe_augmented_rloo -- \
  --lambda_mix 0.5 \
  --probe /vol/steering/C_outcome_l16_pre_answer_pipeline.pkl \
  ...same flags as above...
  --wandb_name probe_aug_lam05

# lambda = 0.7 (reward-heavy mix)
modal run --detach modal_train.py probe_augmented_rloo -- \
  --lambda_mix 0.7 \
  ...same flags...
  --wandb_name probe_aug_lam07
```

The training trainer prints:
```
[probe_augmented_rloo] ACTIVE: A_i = R_i - LOO_{j!=i}[lam*R_j + (1-lam)*probe_j]. Reward unchanged (verifier).
```
which confirms the lambda-mix path is engaged.

---

## Step 5: downstream eval (after all 3 finish)

For each finished checkpoint, run an eval with the SAME fixed sampler:

```bash
for LAM in 03 05 07; do
  modal run --detach modal_train.py sample_local -- \
    --model_path /vol/checkpoints/rloo_probe_aug_checkpoints/rloo_probe_aug_0.5b/probe_aug_lam${LAM}/latest_checkpoint/model \
    --input_jsonl /root/default_proj/extension/data/countdown_eval_500.jsonl \
    --output_json /vol/evaluation/eval_results/eval_probe_aug_lam${LAM}.json \
    --num_responses 16 --max_prompts 500 \
    --temperature 0.6 --top_p 0.95 --top_k 20 \
    --stop_strings '</answer>'
done
```

For each, compute pass@1 (first-block correct) and mean blocks-per-rollout.
Compare to vanilla C_outcome's pass@1 = 0.55 on the same set.

---

## Step 6: what to look for in results

| lambda | What it tests | "Better" looks like |
|---|---|---|
| 0.3 | Probe-dominant baseline (Anagha's lam=0 extreme has been validated as principled but untested empirically; lam=0.3 is the slight-step-back from that) | pass@1 ~= vanilla; ideally smoother loss curve / faster convergence |
| 0.5 | Even mix | similar pass@1 to vanilla; possibly lower advantage variance per step |
| 0.7 | Reward-dominant (closest to vanilla) | should be very close to vanilla (lam=1.0); use as sanity check |

The HEADLINE expectation: probe-as-baseline preserves the optimization target
(verifier reward) so the final policy should match vanilla RLOO's pass@1
(within sampling noise). The probe enters only via the baseline, which is a
variance-reduction control variate -- so the prediction is "same final
accuracy, possibly faster convergence and lower step-to-step variance."

If lambda=0.3 catastrophically Goodharts (pass@1 << vanilla): that would
indicate the lam-mix is not actually unbiased in practice, suggesting the
LOO over probe-values has some bias issue we missed.

---

## Step 7: optional -- causal steering on each checkpoint

For the report's "reader/writer asymmetry" story, run causal steering on
each finished lam checkpoint and compare the post-Goodhart Delta to:
- vanilla C_outcome: Delta in [-0.07, +0.02] (null band)
- probe-RL runA (probe as REWARD): Delta = +0.083

If lam=0.3 gives Delta ~ 0 (still null), this CONFIRMS that probe-as-baseline
doesn't drag the probe direction toward causality (because the optimization
target is unchanged). Strong evidence for the reader/writer story.

---

## Troubleshooting

- **`probe_value_pkl` not found inside Modal**: confirm the file made it
  into the volume via `modal volume ls default-proj-training steering/`.
- **`PROBE_AUG_LAMBDA` not respected**: check Modal logs for the
  `[probe_augmented_rloo] ACTIVE: A_i = R_i - LOO ...` line. Without it,
  the worker falls back to pure probe-baseline (lam=0).
- **Probe scores all near zero**: indicates probe vs activation distribution
  mismatch. Re-train the probe on the freshly-cached fixed rollouts (see 3f)
  rather than reusing the buggy-cache probe.

---

## Files touched / added for this runbook

- `extension/evaluation/sample_local_jsonl.py` (+`--stop_strings` flag)
- `extension/probe/filter_to_clean.py` (+`--src_dir`/`--dst_dir` flags)
- `extension/probe/relabel_full_grid.py` (+`--cache_dir`/`--eval_*` flags)
- `extension/probe/relabel_cross_checkpoint.py` (+`--cache_dir`/`--eval_*` flags)
- `extension/probe/save_probe_vector.py` (already emits Pipeline pkl)
- `extension/training/probe_augmented_rloo.py` (NEW; lam-mix wrapper)
- `rloo_trainer/rloo_update_worker.py` (lam-mix baseline branch in update())
- `modal_train.py` (probe_augmented_rloo entry)
