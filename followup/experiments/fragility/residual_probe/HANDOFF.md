# Handoff — Arm B

> **STATE, 2026-08-07: Arm B has ALREADY BEEN BUILT, GATED AND LAUNCHED.**
> It needs **evaluating, not launching**. Do not re-run the training.
>
> - artefact: `probe_surface_only.pkl`, held-out AUROC **0.9252**, gated 5/5
>   (including a gate proving it never reads the activation: identical scores for
>   zero vs random `X`)
> - run: `armB_surface_100step`, W&B project `probe_residual`
> - checkpoint: `/vol/fragility/ckpt_armB/probe_residual/armB_surface_100step/latest_checkpoint/model`
> - `reward_mean` pegged at 0.996 by step 30 (from 0.145 at step 0)
> - **next action: run the eval command below, then the analysis.**

## The one task

Build, gate, run, and evaluate **Arm B**: a reward probe that scores from the 39
cheap surface features ONLY, ignoring the activation entirely.

It is the falsifier for the surface-occupation account in `REVISION_PACK.md` §G2.
Prediction, already committed in `PREREGISTRATION.md`: **Arm B collapses at least
as hard as the published probe-as-reward run (0.2361).** If a reward with no access
to the model's internals does NOT collapse, §G2 is wrong and should be retracted,
not defended.

## What already exists and works

| file | state |
|---|---|
| `surface_residual_probe.py` | `surface_features`, `feature_matrix`, `LinearScorer`, `SurfaceResidualProbe`, `fit`, `save`, `load` |
| `residual_reward.py` | forked reward builder; passes rollout text to probes with `needs_text = True` |
| `launch_residual_rl.py` | working launcher; sets `PYTHONPATH` for Ray children |
| `countdown_eval_local.py` | forked eval that accepts a local `.jsonl` |
| `test_gate.py` | 5 gates, all green, re-runnable |
| `PREREGISTRATION.md` | predictions committed before Arm A ran |

Modal entrypoints `residual_rl` and `eval_local` are registered in
`followup/modal_fragility.py`.

## What Arm B needs

A `SurfaceOnlyProbe` with `needs_text = True` whose `predict_proba(X, text=...)`
**ignores `X`** and scores `LinearScorer` on the surface features alone. Fit on the
same clean-406 rows as Arm A.

Gate it before spending anything:
- read-only held-out AUROC reproduces **0.9252** (measured; prompt-disjoint split)
- save/load round-trip identical
- the artefact references NO custom module and NO sklearn (see `save`'s docstring)
- it must NOT crash when handed a zero activation — it should never read `X` at all

## Commands that work

```bash
# gates
python followup/experiments/fragility/residual_probe/test_gate.py

# train (5.5 h, ~$25)
set -a && . ./.env && set +a
modal run --detach followup/modal_fragility.py residual_rl -- \
  --probe /vol/fragility/probes/<artefact>.pkl --reward_mode probe --probe_layer 16 \
  --model_name asingh15/qwen-sft-countdown-defaultproj \
  --ref_model_name asingh15/qwen-sft-countdown-defaultproj \
  --tokenizer_name Qwen/Qwen2.5-0.5B \
  --dataset_name asingh15/countdown_tasks_3to4 \
  --num_training_steps 100 --batch_size 128 --gradient_accumulation_steps 128 \
  --group_size 8 --learning_rate 1e-5 --kl_divergence_coefficient 0.001 \
  --entropy_coefficient 0.001 --lr_schedule constant --warmup_ratio 0.0 \
  --weight_decay 1e-4 --gradient_clipping 0.0 --temperature 1.0 --top_p 1.0 \
  --top_k -1 --min_p 0.0 --save_every_n_steps 10 \
  --save_dir /vol/fragility/ckpt_armB --wandb_project probe_residual --wandb_name armB_surface_100step

# eval (~$4).  NOTE the container path: the repo mounts at /root/default_proj
modal run --detach followup/modal_fragility.py eval_local -- \
  --model_path /vol/fragility/ckpt_armB/probe_residual/armB_surface_100step/latest_checkpoint/model \
  --eval_dataset /root/default_proj/extension/data/countdown_eval_500.jsonl \
  --output_dir /vol/evaluation/eval_results --output_name armB_surface_step100 \
  --num_responses 8 --temperature 1.0 --top_p 1.0 --top_k -1 --max_tokens 1024
```

Upload artefacts with
`modal volume put --force default-proj-training <local.pkl> fragility/probes/<name>.pkl`.

`.env` holds `HF_TOKEN` / `WANDB_API_KEY` and is **not** auto-sourced —
`_secrets()` reads the launching shell. Forgetting this was failure #1 of five.

## Analysis, already fixed — do not re-choose it

- Primary outcome: **first-block** accuracy on **clean-406** (`scratchpad/clean406.json`,
  or derive from `probe_cache_n500_clean406/..._pre_answer.meta.json`).
- Verifier: all numbers used exactly once, arithmetic-only expression, evaluates to
  target. Gated at 99.94% against the harvester's own recorded scores.
- Prompt-clustered paired bootstrap, 10k resamples.
- Reference points, same pipeline: C_outcome **0.5306**, published probe-RL
  **0.2361** (matches the paper's stated 0.236 exactly), Arm A **0.1678**.

## Five things that already went wrong — do not repeat them

All five lived OUTSIDE the code the gates covered. The gates stayed green and were
correct throughout; in-process gating and deployment correctness are close to
disjoint.

1. `.env` not sourced → no credentials in the container.
2. Ray actors are fresh processes; runtime `sys.path` edits do not reach them. Set
   `PYTHONPATH`.
3. Pickling a fitted sklearn `Pipeline` breaks across sklearn versions. Store plain
   arrays (`LinearScorer`).
4. `load_dataset(name, split='test')` cannot address a local file.
5. The repo mounts at `/root/default_proj`, not `/root/followup`.

**Always smoke-test with `--num_training_steps 5` before a 100-step run.** It is a
few dollars and it caught all five.

## Two traps

- **Do not let §G4 become a mechanism claim.** The measured fact is that
  residualising coincided with a *worse* collapse. Why is unestablished.
  "Less-constrained boundary, more room to exploit" is a hypothesis. Stating it as a
  finding is exactly how §3 of the original paper went wrong.
- **n = 1 seed per arm.** Every arm is a demonstration, not an effect size. Label it.

## Bigger picture

The paper itself is submittable without Arm B. Every corrected number is in
`REVISION_PACK.md`; what remains there is prose, which the user is writing.
Target venue JUDGe (backup Verify-Agents), deadline 2026-08-29.
