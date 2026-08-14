# Reading vs. Writing a Near-Oracle Internal Verifier

**Abraham Yeung & Anagha Ramaswamy.**

This repository contains the code and experimental artifacts for our investigation of when a near-oracle linear correctness probe on a small language model's hidden states is safe to wire into RL training, and when it catastrophically fails. The headline finding is a **reader/writer asymmetry**: the same probe direction is excellent as a deployment selector, causally inert under intervention on the vanilla checkpoint, and a catastrophic Goodhart trap when used as the RL reward — and the difference between those outcomes is entirely determined by how much policy-gradient access the probe receives.

---

## Headline results

Working on Qwen2.5-0.5B (replicated at 1.5B) trained on Countdown arithmetic with RLOO:

| Claim | Evidence |
|---|---|
| A trace-final correctness probe is **near-oracle**, and outcome RL strengthens it. | Held-out AUROC `0.912 → 0.982` (0.5B); `0.974` at 1.5B; within-prompt matched-pair Wilcoxon `p < 1e-7`. |
| The probe is **not a length detector**, though length explains a lot. | Stratified within `</think>`-position deciles it holds `0.939`, against `0.564` for position alone. |
| Within multi-answer rollouts, the model's commit-time representation tracks the commit — **no "knows-but-doesn't-say"** at this scale. | On T→F drift rollouts, probe at the wrong-final commit = `0.154` against an F→F floor of `0.088` — a real but small residue (`+0.065`, CI `[+0.023, +0.105]`), only `8.7%` of the way from floor to the T→T level. |
| The probe is a **reader, not a usable write-axis** on the vanilla checkpoint. | Activation-addition steering vs. matched random direction gives Δ ∈ `[-0.07, +0.02]`; no magnitude gives a positive effect, and the one interval excluding zero (α=0.5, Δ = `−0.072`) points the wrong way. |
| As a **deployment selector** the probe is useful at zero extra inference compute — but a structural baseline captures most of it. | Best-of-16 `+11.7 pp` over a random pick of 16 on C_outcome — `88.8%` of all available headroom — of which `+8.5 pp` is available from "pick the rollout with the shortest `<think>` body" with no probe at all; the probe adds `+3.2 pp` (CI `[+0.99, +5.42]`, p = 0.008). Budgeted restart matches best-of-16 at `−60%` compute; abstention `98%` at `50%` coverage. |
| As the **RL reward**, the probe catastrophically Goodharts in both initialisation regimes. | `runA` (init from C_outcome): delayed Goodhart, `−31.4 pp` first-block accuracy (0.550 → 0.236); `runB` (init from C_SFT): immediate Goodhart, `−22 pp`. |
| **A monitor with no internal access reproduces the failure**, and used as the reward it destroys the task. | 7-feature text-only logistic regression, frozen on baseline: score `0.506 → 0.832` while true accuracy falls `0.517 → 0.130`. As the RL reward (pre-registered Arm B): `0.0000` accuracy, with `99.9%` of rollouts still emitting a well-formed but arithmetic-free `<answer>` block. |
| **Safe constructions** that bound or eliminate probe gradient access give either small lifts or no Goodhart — **and none beats read-only selection.** | Probe-as-baseline (LOO control variate): target-invariant, untested. Multiplicative shaping `r = verifier × probe`: `+2.84 pp` first-block, CI `[+1.85, +3.85]`, n=1 run. Probe-best-of-K in-training selection: `+1.5 pp`, halves mean blocks per rollout, n=1 run. Read-only best-of-16: `+11.7 pp`. |

**Withdrawn.** An earlier version of this README claimed a mech-interp signature of Goodhart — the optimised probe direction becoming causal post-RL (Δ = `+0.083`) while a near-orthogonal control stayed null. That result does not survive checking and is retracted: the steered vector has cosine `0.163` with the probe RL actually optimised (not `1.000`) and AUROC `0.896` (not `0.982`), the steering hook sat one transformer block and 2–3 tokens downstream of the probe's read site, the key contrasts reach only p = 0.063 and p = 0.080, and the three steering runs share zero prompts. `causal_steering.py` now fixes both the layer index and the token position, so it will not reproduce the published JSONLs; use `--layer_convention legacy_block --steer_position last_token` if you need to. See `REVISION_PACK.md` §A.

---

## Repository layout

```
extension/                       # All research-extension code lives here
├── probe/
│   ├── cache_hidden_states.py            # Cache pre_answer, assertion, neutral activations
│   ├── cache_answer_positions.py         # Cache <answer>-opening hidden states
│   ├── causal_steering.py                # Activation-addition steering protocol
│   ├── save_probe_direction_temp1.py     # Persist the L16 pre_answer probe direction
│   ├── cross_position_transfer.py        # Within-checkpoint cross-position cosines
│   ├── phase2a_position_appropriate_probe.py    # Within-rollout T→F drift analysis
│   ├── probe_abstention_and_hybrid.py    # Selective abstention + probe-best hybrid
│   ├── probe_adaptive_budget.py          # Adaptive K_avg budget
│   ├── probe_answer_commit.py            # Probe-commit (within-rollout block selector)
│   ├── probe_applied_scale_comparison.py # 0.5B vs 1.5B applied strategies
│   ├── probe_as_eval_proxy.py            # Probe-mean as dataset-accuracy proxy
│   ├── probe_bestofk_offline.py          # Best-of-K probe selector
│   ├── probe_guided_restart.py           # Budgeted probe-guided restart
│   ├── probe_rl_downstream_analysis.py   # Post-RL probe diagnostics
│   └── ...
├── training/
│   ├── probe_rloo.py                     # Probe-as-RL-reward base (runA, runB)
│   ├── probe_reward_rloo.py              # +multiplicative shaping mode
│   └── probe_augmented_rloo.py           # Probe-best-of-K in-training selector
├── evaluation/
│   └── sample_local_jsonl.py             # vLLM sampling pipeline with stop-token config
└── metrics/
    ├── diagnose_outcome.py               # First-/last-block, length, n-blocks diagnostics
    ├── calibration.py                    # Probe calibration vs verifier
    ├── behavioral.py                     # Behavioural metrics on rollouts
    └── dynamics.py                       # Training-time dynamics

rloo_trainer/
├── rloo_update_worker.py                 # Standard RLOO update + LOO probe-baseline path
└── ...                                   # Scaffolds

sft_trainer/, ipo_trainer/                # Scaffolds (used for the SFT step only)

scripts/
├── make_poster_figures.py                # Matplotlib PDFs for the four headline figures
├── compute_verifier_acc.py               # Pooled verifier accuracy computation
└── eval_multiplicative_all_checkpoints.{sh,ps1}
```

The `extension/cache/` directory holds cached hidden-state activations (`.npz`, `.pkl`) and trained probe direction vectors. These are gitignored; rebuild with the `cache_*.py` scripts.

---

## Reproducing the headline experiments

All training and large evals were run on Modal H100 (single GPU per job). Total project compute spend: **~$265**.

### 1. SFT and the baseline RL checkpoint

We did not retrain SFT. `C_SFT` is `asingh15/qwen-sft-countdown-defaultproj`. `C_outcome` is 100 RLOO steps from `C_SFT` with the standard verifier reward (`batch_size=128`, `kl=1e-3`, `lr=1e-5`).

```bash
bash rloo_trainer/train_rloo_modal.sh   # 100 steps from C_SFT, verifier reward
```

### 2. Cache hidden states and train probes

```bash
python extension/probe/cache_hidden_states.py        # pre_answer / assertion / neutral
python extension/probe/cache_answer_positions.py     # <answer>-opening positions
python extension/probe/save_probe_direction_temp1.py # persist L16 pre_answer direction
```

### 3. Deployment-time applied probes

```bash
python extension/probe/probe_bestofk_offline.py        # best-of-16
python extension/probe/probe_guided_restart.py         # budgeted restart
python extension/probe/probe_abstention_and_hybrid.py  # abstention + majority hybrid
python extension/probe/probe_adaptive_budget.py        # adaptive K_avg
python extension/probe/probe_as_eval_proxy.py          # probe-mean as accuracy estimator
python extension/probe/probe_answer_commit.py          # within-rollout block selector
python extension/probe/probe_applied_scale_comparison.py  # 0.5B vs 1.5B
```

### 4. Causal steering

Two flags on this script changed on 2026-08-12 and both change the numbers, so read
this before comparing against any shipped JSONL.

- `--steer_position` defaults to `probe_read`: the token containing the *first*
  character of `</think>`, which is where `cache_hidden_states.py` fit the probe.
  The original runs injected at the last token of the prefix, 2-3 positions later.
- `--layer_convention` defaults to `hidden_state`: `--layer 16` means
  `hidden_states[16]`, so the hook goes on `model.layers[15]`. The original runs
  hooked `model.layers[16]`, which writes to `hidden_states[17]` — one full
  transformer block downstream of the read site.

```bash
# The vanilla null (Section 4 of the paper), at the corrected site
python extension/probe/causal_steering.py \
    --model_path <C_outcome>  --steer_vec <C_outcome_l16_pre_answer.npz> \
    --alphas 0.0 0.5 1.0 2.0  --n_prompts 100 --n_rollouts_per_prompt 2

# To reproduce the shipped (withdrawn) post-Goodhart JSONLs bit-for-bit
python extension/probe/causal_steering.py \
    --model_path <runA_final>  --steer_vec <C_outcome_l16_pre_answer.npz> \
    --steer_position last_token --layer_convention legacy_block
```

Analyse with `extension/probe/causal_steering_stats.py`, which does the
prompt-clustered bootstrap and exact McNemar. Note that the vanilla and
post-Goodhart JSONLs share **zero prompts** (50/97 prefixes vs 100/194), so any
before/after contrast across those files is unpaired.

### 5. Probe-as-RL-reward (Goodhart demonstration)

The initialisation is `--model_name` (passed straight through to `rloo.py`); the
reward shaping is `--reward_mode`.

```bash
# runA: init from C_outcome (delayed Goodhart)
python extension/training/probe_reward_rloo.py \
    --model_name <C_outcome_path> --reward_mode probe \
    --probe extension/cache/steering/probe_pipeline_C_outcome_l16_pre_answer_temp1.pkl \
    --num_training_steps 100 --save_every_n_steps 10

# runB: init from C_SFT (immediate Goodhart)
python extension/training/probe_reward_rloo.py \
    --model_name asingh15/qwen-sft-countdown-defaultproj --reward_mode probe \
    --probe extension/cache/steering/probe_pipeline_C_outcome_l16_pre_answer_temp1.pkl

# Multiplicative shaping (verifier x probe)
python extension/training/probe_reward_rloo.py \
    --model_name asingh15/qwen-sft-countdown-defaultproj --reward_mode mult
```

`--reward_mode` is one of `probe | probe_gated | blend | mult`. `--reward_disable`
reverts to the vanilla verifier reward as an A/B control.

### 6. Probe-best-of-K in-training selection (the hybrid that beats vanilla RLOO)

Top-M gating lives on `rloo.py` and requires `--probe_baseline`:

```bash
python rloo_trainer/rloo.py \
    --model_name asingh15/qwen-sft-countdown-defaultproj \
    --group_size 8 --probe_baseline --probe_topk_M 4 \
    --probe_value_pkl extension/cache/steering/probe_pipeline_C_outcome_l16_pre_answer_temp1.pkl
```

Add `--probe_topk_renormalize` to rescale the surviving advantages by
`group_size / M`. Without it the top-M arm trains at roughly `M/group_size` of
vanilla RLOO's effective learning rate, so a top-M-vs-vanilla comparison at
matched `--learning_rate` is partly a comparison of step sizes. The flag
defaults off so the original runs reproduce.

The lambda-mix control-variate wrapper is separate:

```bash
python extension/training/probe_augmented_rloo.py --lambda_mix 0.5 \
    --model_name asingh15/qwen-sft-countdown-defaultproj
```

### 6b. Structural controls (run these before quoting any probe number)

```bash
python extension/probe/structural_baselines.py
```

Reports, on the same folds and population as the probe: the AUROC of the
`</think>` token position alone, the probe's AUROC stratified within
position deciles, best-of-K for "pick the shortest `<think>` body" next to
probe-best-of-K, and the size/accuracy of the no-`</think>` rollouts that are
absent from every cached AUROC. See `CODE_AUDIT.md` §C1.

### 7. Probe-as-baseline (target-invariant LOO control variate)

The `--probe_baseline` flag in `rloo_trainer/rloo_update_worker.py` replaces the standard reward-mean baseline with the LOO mean of per-rollout probe values. The optimisation target remains the verifier; the probe enters only through the variance-reduction baseline. This construction is code-complete and theoretically Goodhart-free; the controlled run-time comparison against vanilla RLOO is the obvious next experiment and not in the report.

---

## What this repository does *not* contain

- Trained checkpoints, cached activations, and large eval JSONs (>5 MB). Reproduce with the scripts above.
- The probe-as-baseline empirical results; see the report's "Failed Attempts and Null Results" section — the construction is code-complete but a controlled run-time comparison is out of scope.

---

## References for the methodology

The activation-addition steering protocol and matched random-direction control follow Turner et al. 2023 (ActAdd), Zou et al. 2023 (Representation Engineering), Rimsky et al. 2024 (CAA), and Arditi et al. 2024 (refusal direction). The probe-vs-causation methodological framing this project operationalises is Belinkov 2022 ("Probing Classifiers: Promises, Shortcomings, and Advances"). For the RL setup and baseline construction we follow Ahmadian et al. 2024 (RLOO) and the classical control-variate analysis of Williams 1992.
