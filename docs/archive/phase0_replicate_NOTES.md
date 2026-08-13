# Phase 0, step 1 — reading the existing RL-against-probe trainer

Required by `followup/CLAUDE.md` before any code is changed. Nothing in
`extension/`, `rloo_trainer/`, or `evaluation/` was modified to produce this
document.

Sources read: `extension/training/probe_rloo.py`, `extension/training/probe_reward_rloo.py`,
`rloo_trainer/rloo.py`, `rloo_trainer/rloo_update_worker.py` (reward path only),
`evaluation/countdown.py`, `modal_train.py`, `findings.md` EXP-21/EXP-23,
`RUNBOOK_CLEAN_REPLICATION.md`.

---

## 1. Which script actually produced the collapse

Two scripts implement probe-as-reward. They are not interchangeable.

| script | used for | reward-model weights |
|---|---|---|
| `extension/training/probe_rloo.py` | **runA / runB — the published Goodhart result (EXP-21)** | reference model **reloaded from the latest checkpoint each round**, i.e. the CURRENT policy |
| `extension/training/probe_reward_rloo.py` | reward-design ablation arm C, multiplicative shaping (EXP-25) | a **frozen** C_SFT model; the reward is a stationary text→scalar function |

This distinction is load-bearing for the follow-up. `probe_rloo.py` reads hidden
states out of the *current* policy, so the probe is a monitor attached to a
moving representation — which is exactly the threat model this project is about.
`probe_reward_rloo.py` freezes the reader, which makes it a fixed reward model
and a different experiment. **Phase 0 harvests the `probe_rloo.py` ladders.**

Modal entry points (`modal_train.py`): `probe_rloo` → `run_probe_rloo`,
`probe_reward_rloo` → `run_probe_reward_rloo`. Both are thin wrappers around
`_run_training(script, args)`, which shells out to `python <script> <args>` with
`HF_HOME` pointed into the volume and commits the volume in a `finally`.

## 2. Architecture: patch, then exec

`probe_rloo.py` never imports `rloo.py` as a module. It:

1. pops its own flags out of `sys.argv` (`--probe_pkl`, `--probe_hybrid`,
   `--probe_layer`) so `rloo.py`'s argparse never sees them;
2. monkey-patches `evaluation.countdown.compute_score`;
3. monkey-patches `wandb.sdk.wandb_run.Run.log` for dual logging;
4. `exec`s `rloo_trainer/rloo.py` with `__name__ == "__main__"`.

Every follow-up wrapper should use this same idiom. It is the repo's established
way of extending the trainer without editing it.

## 3. Where the reward is computed

`rloo_trainer/rloo.py:284-291`, inside `train()`:

```python
for curr_responses, curr_ground_truth in zip(all_responses, all_ground_truth):
    for x in curr_responses:
        curr_rewards.append(compute_score(x, curr_ground_truth))
```

`compute_score` is resolved at module import of `rloo.py`, so the patch must be
installed **before** the exec. That is the whole reason for the patch-then-exec
ordering.

The probe reward (`probe_rloo.py:213-262`) is:

1. reconstruct the chat-templated prompt from `ground_truth` via
   `_PROMPT_TEMPLATE_BARE` (numpy `array2string` bracket repr, verbatim);
2. tokenize `prompt + full response` with the **fast** tokenizer,
   `return_offsets_mapping=True`, `truncation=True`, `max_length=2048`;
3. locate the token whose span contains the **first character** of the
   response's first `</think>` (`s <= char < e`);
4. forward pass with `output_hidden_states=True`, take
   `hidden_states[16][0, tok_idx]`, cast to float32;
5. `probe.predict_proba(h)[0, 1]`.

**Five of these five steps are load-bearing and were each a bug at some point**
(EXP-21 lists 10 bugs found before runA/runB were valid). The two that silently
produce a plausible-looking but wrong number:

- **token position.** Using the token covering the LAST character of `</think>`
  is 2 tokens later than the cache convention and saturates the probe to ~0.99
  for every input. `cache_hidden_states.py:char_to_token_index` is the reference
  implementation; anything that extracts this activation must match it.
- **prompt reconstruction.** `np.array2string` emits `'[ 7  2 43 63]'` with a
  leading space when widths differ; stripping it mismatches ~15% of prompts and
  moves those activations off-distribution.

> **Follow-up rule.** The harvester must not reimplement activation extraction.
> It imports the helpers from `extension/probe/cache_hidden_states.py`, and the
> harvest job asserts bit-equality against a cached reference vector before
> writing anything.

## 4. Config surface

`rloo.py` argparse, the flags that matter here:

| flag | runA/runB value | note |
|---|---|---|
| `--model_name` | C_outcome (runA) / C_SFT (runB) | also the reference-model init |
| `--num_training_steps` | 100 | |
| `--batch_size` | 128 | prompts per step |
| `--group_size` | 8 | rollouts per prompt |
| `--gradient_accumulation_steps` | 128 | keeps microbatch at 8; OOMs at 1 |
| `--learning_rate` | 1e-5 | |
| `--lr_schedule` | constant | incompatible with `warmup_ratio > 0` |
| `--warmup_ratio` | 0 | must be 0 with a constant schedule |
| `--kl_divergence_coefficient` | 1e-3 | |
| `--save_every_n_steps` | 10 | **the checkpoint-ladder density knob** |
| `--save_dir` | `/vol/checkpoints/rloo_probe_checkpoints` | |
| `--temperature`, `--top_p`, `--top_k` | 1.0, 1.0, -1 | vLLM sampling at train time |
| `--gpu_memory_utilization` | 0.9 (0.75 in `probe_reward_rloo`) | reference model needs headroom |

Wrapper-only flags: `--probe_pkl`, `--probe_layer` (16), `--probe_hybrid`.

## 5. Checkpointing — what already exists

`rloo.py:329-350`. On every step:

- if `save_every_n_steps > 0 and global_step % save_every_n_steps == 0`, the
  destination is `{save_dir}/{wandb_project}/{wandb_name}/epoch_{e}_step_{s}/`;
- otherwise it is `.../latest_checkpoint/`, which is **`shutil.rmtree`d and
  rewritten every step**;
- each directory holds `model/`, `optimizer.pt`, `scheduler.pt`;
- the next step resumes the optimizer from the previous directory, so the two
  newest directories must keep their optimizer state.

### The finding that reshapes Phase 0

Both published Goodhart runs used `--save_every_n_steps 10`, and **their
archival checkpoints are still on the Modal volume** (verified 2026-07-30):

```
checkpoints/rloo_probe_checkpoints/rloo_probe_0.5b/
  probe_rloo_runA_coutcome_FINAL/epoch_0_step_{0,10,20,...,90}  + latest_checkpoint
  probe_rloo_runB_csft_FINAL/    epoch_0_step_{0,10,20,...,90}  + latest_checkpoint
```

11 checkpoints per run, 22 total, each with a complete `model/` directory,
spanning the entire collapse (runA: probe−verifier gap −0.120 → +0.671, verifier
0.572 → 0.321; runB: +0.173 → +0.824, verifier 0.298 → 0.166).

So **Phase 0 does not need to re-run the collapse.** What is missing is not
checkpoints — it is cached activations off those checkpoints. That converts
Phase 0 from a ~$160 / ~10 h retraining job into a forward-pass harvest.

The 11-checkpoint density is below the CLAUDE.md target of 30–50. Deliberate
call: harvest the existing 11 first, look at the curve, and only pay for a
K=3 rerun (34 checkpoints) if the K=10 curve turns out to be under-resolved
where it matters — i.e. if the interesting transition happens inside a single
10-step gap. For runA the published trajectory puts the cliff between steps 30
and 60, which the existing ladder brackets with three points. `--save_every_n_steps 3`
is a one-flag change when that rerun is wanted; `fragility_core.checkpoint_logging`
already carries the disk-hygiene helper it needs.

## 6. What is NOT logged, and what we add

`rloo.py` logs to W&B only: `sampling/reward_mean`, `train/kl_loss`,
`train/*` update-worker metrics. `probe_rloo.py` adds `train/probe_mean`,
`train/verifier_mean`, `train/probe_minus_verifier`.

Gaps for our purposes, and how Phase 0 fills them:

| missing | fix |
|---|---|
| step-level scalars survive only in W&B | `checkpoint_logging.TrainLogWriter` + `patch_wandb_log_to_jsonl` writes a JSONL next to the ladder |
| no activations per checkpoint | `harvest_ladder.py` (this directory) |
| no per-checkpoint downstream eval | the harvester scores every rollout with the real verifier while it samples |
| optimizer state bloats archival checkpoints | `checkpoint_logging.prune_stale_optimizer_state` |

## 7. Sampling-regime hazards inherited from the paper

Two documented traps, both of which change probe numbers without changing
anything real. Both are pinned in the Phase-0 configs.

1. **Temperature.** The probe pickle used as the RL reward was fit on temp=1.0 /
   top_p=1.0 rollouts (`probe_pipeline_temp1_meta.json`, held-out balanced AUROC
   0.810). The paper's headline AUROC 0.982 comes from temp=0.6 / top_p=0.95
   eval rollouts and a different pickle. Mixing them re-derives an old bug: a
   probe applied off its sampling regime saturates. **Harvest at temp=1.0 with
   the temp1 pickle** (`phase0_harvest_run{A,B}.yaml`).
2. **Stop strings.** `</answer>` as a stop string at eval time but not at train
   time confounded two withdrawn experiments (EXP-19, EXP-22; the real
   explanation is EXP-23's `<|im_end|>` discovery). Training-time sampling used
   no stop string, so the harvest uses none — the on-policy arm must reproduce
   the text distribution the reward actually saw. `n_blocks` per rollout is
   recorded regardless, because it is the structural variable that separates
   runA's exploit (15.5 blocks) from runB's (1.27).

## 8. Contamination filter

`extension/data/contaminated_prompt_idx.json` splits `countdown_eval_500.jsonl`
into 94 contaminated and 406 clean prompt indices. All published probe AUROCs are
on the clean-406 subset. `filter_to_clean.py` was pruned in the public-release
refactor (commit cd8cd5e), so `fragility_core.labels.clean_prompt_idx` reads the
manifest directly.

---

## Status of the CLAUDE.md Phase-0 checklist

| # | item | status |
|---|---|---|
| 1 | read the trainer, write NOTES.md | **done** (this file) |
| 2 | `checkpoint_logging.py` | **done** — plus ladder discovery, which turned out to be the operative half |
| 3 | activation capture on a fixed eval set | **done for runA** — all 11 checkpoints harvested via `harvest_ladder.py`, 5 layers, 406 clean prompts × 8 rollouts at temp 1.0. runB has 2 of 11 (steps 0 and 99) |
| 4 | re-run the collapse with logging | **superseded** — the ladders exist; harvest instead. Re-run only if the K=10 spacing proves too coarse |
| 5 | `phase0_report.md` | **done** — written against runA's full ladder; will be extended when runB's remaining 9 checkpoints are harvested |

Acceptance criteria carried over from CLAUDE.md, restated for the harvest:
frozen-probe reward hacking must be visible in the harvested data (probe score up,
verifier accuracy down by a large margin), with ≥ 10 checkpoints of cached
activations per run and no fabricated rows for checkpoints that fail to harvest.

Status: **met for runA** (11 checkpoints; probe score 0.460 → 0.989, verifier
accuracy 0.372 → 0.133), **not yet met for runB** (2 checkpoints; the reward
hacking is visible — probe score 0.498 → 0.996, accuracy 0.159 → 0.068 — but the
checkpoint count is short). No rows were written for unharvested checkpoints.
