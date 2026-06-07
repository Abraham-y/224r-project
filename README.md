# Reading vs. Writing a Near-Oracle Internal Verifier

**Stanford CS 224R class project — Abraham Yeung & Anagha Ramaswamy.**

This repository contains the code and experimental artifacts for our investigation of when a near-oracle linear correctness probe on a small language model's hidden states is safe to wire into RL training, and when it catastrophically fails. The headline finding is a **reader/writer asymmetry**: the same probe direction is excellent as a deployment selector, causally inert under intervention on the vanilla checkpoint, and a catastrophic Goodhart trap when used as the RL reward — and the difference between those outcomes is entirely determined by how much policy-gradient access the probe receives.

The repository is organised around the **`extension/`** directory, which contains the research code that produced the experimental results. The class-provided trainer scaffolds (`sft_trainer/`, `ipo_trainer/`, `rloo_trainer/`, `evaluation/`) are kept for reproducibility, with our additions (notably the LOO probe-baseline integration in `rloo_trainer/rloo_update_worker.py`).

---

## Headline results

Working on Qwen2.5-0.5B (replicated at 1.5B) trained on Countdown arithmetic with RLOO:

| Claim | Evidence |
|---|---|
| A trace-final correctness probe is **near-oracle**, and outcome RL strengthens it. | Held-out AUROC `0.912 → 0.982` (0.5B); `0.974` at 1.5B; within-prompt matched-pair Wilcoxon `p < 1e-7`. |
| Within multi-answer rollouts, the model's commit-time representation tracks the commit — **no "knows-but-doesn't-say"** at this scale. | On T→F drift rollouts, probe at the wrong-final commit = `0.154`, matches F→F floor `0.088`. |
| The probe is a **reader, not a controller** on the vanilla checkpoint. | Activation-addition steering vs. matched random direction gives Δ ∈ `[-0.07, +0.02]` across α. |
| As a **deployment selector** the probe is highly useful at zero extra inference compute. | Best-of-16 `+12.1 pp`; budgeted restart matches best-of-16 at `−60%` compute; abstention `98%` at `50%` coverage; probe-mean estimates dataset accuracy within `±0.4 pp`. |
| As the **RL reward**, the probe catastrophically Goodharts in both initialisation regimes. | `runA` (init from C_outcome): delayed Goodhart, `−25 pp` first-block accuracy; `runB` (init from C_SFT): immediate Goodhart, `−22 pp`. |
| **Mech-interp signature of Goodhart.** The same probe direction — causally inert before RL — becomes measurably causal after, while a near-orthogonal correctness-correlated direction stays null on the same checkpoint. | Probe direction Δ = `+0.083` at α=1.0; assertion direction (cosine `0.038`, AUROC `0.70`) Δ = `−0.015`. |
| **Safe constructions** that bound or eliminate probe gradient access give either small lifts or no Goodhart. | Probe-as-baseline (LOO control variate): target-invariant. Multiplicative shaping `r = verifier × probe`: `+2.8 pp` first-block at `n=8,000`. Probe-best-of-K in-training selection: `+1.5 pp`, halves mean blocks per rollout. |

Full writeup: see the CS 224R report (private; not in the repo).

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
└── ...                                   # Class-provided scaffolds

sft_trainer/, ipo_trainer/                # Class-provided scaffolds (used for the SFT step only)

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

```bash
# Vanilla null (Section 4 of the report)
python extension/probe/causal_steering.py \
    --model_path <C_outcome>  --steer_vec <C_outcome_l16_pre_answer.npz> \
    --alphas 0.0 0.5 1.0 2.0  --n_prompts 100 --n_rollouts_per_prompt 2

# Post-Goodhart + specificity control (Section 6.4)
python extension/probe/causal_steering.py \
    --model_path <runA_final>  --steer_vec <C_outcome_l16_pre_answer.npz>     # +0.083
python extension/probe/causal_steering.py \
    --model_path <runA_final>  --steer_vec <C_outcome_l16_assertion_direction.npz>  # -0.015 (control)
```

### 5. Probe-as-RL-reward (Goodhart demonstration)

```bash
# runA: init from C_outcome (delayed Goodhart)
python extension/training/probe_reward_rloo.py --init C_outcome --mode reward
# runB: init from C_SFT (immediate Goodhart)
python extension/training/probe_reward_rloo.py --init C_SFT --mode reward
# Multiplicative shaping (verifier × probe)
python extension/training/probe_reward_rloo.py --init C_SFT --mode mult
```

### 6. Probe-best-of-K in-training selection (the hybrid that beats vanilla RLOO)

```bash
python extension/training/probe_augmented_rloo.py --K 8 --top_M 4 --init C_SFT
```

### 7. Probe-as-baseline (target-invariant LOO control variate)

The `--probe_baseline` flag in `rloo_trainer/rloo_update_worker.py` replaces the standard reward-mean baseline with the LOO mean of per-rollout probe values. The optimisation target remains the verifier; the probe enters only through the variance-reduction baseline. This construction is code-complete and theoretically Goodhart-free; the controlled run-time comparison against vanilla RLOO is the obvious next experiment and not in the report.

---

## What this repository does *not* contain

- The CS 224R report PDF/source and personal research notes. Those are kept privately and are not pushed to this repository.
- Trained checkpoints, cached activations, and large eval JSONs (>5 MB). Reproduce with the scripts above.
- The probe-as-baseline empirical results; see the report's "Failed Attempts and Null Results" section — the construction is code-complete but a controlled run-time comparison is out of scope.

---

## References for the methodology

The activation-addition steering protocol and matched random-direction control follow Turner et al. 2023 (ActAdd), Zou et al. 2023 (Representation Engineering), Rimsky et al. 2024 (CAA), and Arditi et al. 2024 (refusal direction). The probe-vs-causation methodological framing this project operationalises is Belinkov 2022 ("Probing Classifiers: Promises, Shortcomings, and Advances"). For the RL setup and baseline construction we follow Ahmadian et al. 2024 (RLOO) and the classical control-variate analysis of Williams 1992.
