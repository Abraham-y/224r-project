# CLAUDE.md — Probe Fragility Under Optimization Pressure

## What this project is

Follow-up to the correctness-probe paper (probe achieves AUROC ~0.98 as a read-only
deployment selector, but Goodharts catastrophically when used as an RL reward signal,
causing a ~25pp accuracy collapse). That result is behavioral. This project explains
the mechanism and tests whether monitor robustness is predictable in advance.

This repo EXTENDS the existing probe codebase. Do not rewrite working components
(probe training, RL loop, eval harness). Add new modules under `experiments/fragility/`
and shared utilities under `fragility_core/`. Treat existing code as read-mostly:
patch it only to add hooks and logging.

## Research question

When a policy is optimized against a correctness probe, what happens mechanistically
to the probed information, and can we predict which probes are robust?

## Hypotheses (the paper's spine)

- **H1 Evasion**: correctness info moves off the probe direction but is preserved
  elsewhere. Signature: retrained probes on post-RL activations recover AUROC even
  though the frozen probe collapsed.
- **H2 Corruption**: the correctness information itself degrades. Signature: even
  retrained probes (linear AND nonlinear) fail to recover AUROC.
- **H3 Predictable fragility**: collapse severity is predictable pre-hoc from probe
  geometry (margin, causal strength of the direction, distributedness, basis).

Any outcome is a result. Never write analysis code that assumes one hypothesis.

## Key definitions (use these exact metrics everywhere)

- `auroc_frozen(t)`: AUROC of the ORIGINAL probe (weights fixed at t=0) on
  activations from checkpoint t.
- `auroc_retrained_linear(t)`: AUROC of a fresh linear probe trained on checkpoint-t
  activations (train/val split fixed across all t, same seed).
- `auroc_retrained_mlp(t)`: same, 2-layer MLP probe (hidden=256, ReLU).
- `evasion_gap(t) = auroc_retrained_linear(t) - auroc_frozen(t)`. This is the
  headline quantity. Large positive gap → H1. Both low → H2.
- `dir_drift(t)`: cosine similarity between the t=0 probe direction and the
  checkpoint-t retrained linear probe direction (top direction after whitening).
- `causal_strength(t)`: |Δ behavior| when steering along the t=0 probe direction at
  fixed norm α (sweep α ∈ {2, 4, 8} × activation std at that layer); behavior metric
  = change in task accuracy and change in probe score.
- `sep(t)`: (mean activation projection onto probe direction | correct) minus
  (| incorrect), divided by pooled std.

## Repo layout to create

```
experiments/fragility/
  phase0_replicate/
  phase1_evasion_vs_corruption/
  phase2_causal_status/
  phase3_predict_robustness/
fragility_core/
  checkpoint_logging.py     # dense checkpoint + activation capture hooks
  probes.py                 # frozen-probe eval, retrain-linear, retrain-mlp
  geometry.py               # dir_drift, sep, CKA, projections
  steering.py               # add-vector steering + patching utilities
  registry.py               # experiment/run registry, seeds, config hashing
configs/fragility/          # one YAML per run; NEVER hardcode hyperparams
results/fragility/          # parquet metrics + figures; append-only
```

## Engineering conventions

- Every run gets a config YAML and a content-hash run ID. No untracked experiments.
- All metrics written as parquet rows: (run_id, phase, checkpoint_step, metric, value, seed).
- Seeds: 3 seeds minimum for any claim that goes in the paper; 1 seed for pilots.
- Small-model track first: port the RL setup to a ~1.5–3B model (e.g. Qwen-class)
  for iteration. Only promote to the original paper-scale model after a pilot works.
- Cache activations to disk once per checkpoint; all probe/geometry analysis reads
  the cache, never re-runs the model. Cache layout:
  `acts/{run_id}/{checkpoint_step}/{layer}.npy` + `labels.parquet`.
  (This is where the activation-cache-compression tooling plugs in later.)
- Figures: matplotlib, one function per paper figure in `figures.py`, regenerable
  from parquet alone.

## Phase 0 — Instrumented replication (do this first, completely)

1. Read the existing RL-against-probe training script. Write a summary of its
   entry points, config surface, and where the reward is computed, into
   `experiments/fragility/phase0_replicate/NOTES.md` before changing anything.
2. Add `checkpoint_logging.py`: save full policy checkpoints (or LoRA adapters)
   every K steps (target ~30–50 checkpoints across the collapse; pick K from the
   original run's step count). Also save optimizer step, reward mean, KL-from-init.
3. Add activation capture: for a FIXED eval set (~2k problems, stratified
   correct/incorrect at t=0, held out from RL data), capture residual-stream
   activations at the probe's layer plus 3 bracketing layers, at the answer-token
   position used by the original probe. Store per checkpoint.
4. Re-run the collapse experiment on the small-model track with this logging.
   Acceptance: frozen-probe reward hacking reproduces (probe score up, true
   accuracy down by a large margin) and ≥30 checkpoints with cached activations.
5. Produce `phase0_report.md`: reward curve, true-accuracy curve, frozen-probe
   AUROC curve, all vs step. If collapse does not reproduce at small scale,
   STOP and flag; do not silently escalate scale.

## Phase 1 — Evasion vs corruption

1. Implement `probes.py` retraining: for each checkpoint t, train linear and MLP
   probes on cached activations (fixed split, fixed seed, early stopping on val).
2. Compute per checkpoint: auroc_frozen, auroc_retrained_linear,
   auroc_retrained_mlp, evasion_gap, dir_drift, sep.
3. Headline figure: the three AUROC curves vs RL step on one plot, per layer.
4. CKA between checkpoint-t and checkpoint-0 activations (same eval set) vs step,
   to distinguish global representational drift from targeted movement.
5. Decision memo `phase1_verdict.md`: which of H1/H2 the data supports, with the
   specific numbers, plus per-layer breakdown (info may relocate across layers:
   check whether retrained-probe AUROC at OTHER layers rises as the probe layer falls).

## Phase 2 — Causal status over training

1. Implement `steering.py`: add-vector steering along the t=0 probe direction at
   the probe layer; measure Δ(task accuracy) and Δ(probe score) per checkpoint,
   3 α values, both signs.
2. Plot causal_strength(t) alongside auroc_frozen(t). Key question: does causal
   strength decay before, with, or after the AUROC collapse?
3. Activation patching: patch the probe-layer, answer-position activation from a
   correct-run into an incorrect-run (and reverse) at each checkpoint; measure
   flip rate vs step. Also patch at bracketing layers to detect relocation.
4. Pre-registration note: BEFORE running, write `phase2_predictions.md` stating
   what H1 vs H2 each predict for these curves. Then run.

## Phase 3 — Predicting and buying robustness

1. Probe zoo: train ~10–15 probes at t=0 varying layer, L2 strength, rank
   (PCA-restricted), basis (raw residual vs SAE features if an SAE is available),
   and an adversarially-trained variant.
2. For each probe, run the SAME RL-against-probe recipe (small model, 1 seed
   pilot, then 3 seeds for the interesting cases). Record collapse severity:
   Δ true-accuracy and steps-to-collapse.
3. Pre-hoc features per probe: margin distribution, causal_strength at t=0,
   participation ratio of the probe direction's loadings, sep(0).
4. Correlate pre-hoc features with collapse severity. Deliverable: a ranked
   table + scatter plots. This is the H3 result.
5. Transfer experiment: take a policy trained to evade probe A; evaluate frozen
   probes B, C, ... on its activations. Report the transfer matrix. This runs
   off already-cached activations and is cheap; do it even if H3 is null.

## Phase 4 (only if Phase 3 finds a predictor)

1. Construct a monitor using the predictor (e.g. ensemble of high-causal-strength
   probes, or the most robust basis) and rerun the RL attack against it.
2. Success criterion: substantially smaller accuracy collapse at matched KL budget.

## Writing artifacts as you go

- Keep `PAPER_NOTES.md` at repo root: every figure-worthy result gets one paragraph
  (claim, evidence, caveats) the day it's produced.
- Target: NeurIPS 2026 interp/safety workshop (fallback), ICLR 2027 main (stretch).

## Things Claude Code must NOT do

- Do not modify the original paper's experiment scripts in place; wrap or fork them.
- Do not delete or overwrite anything in `results/`; append-only.
- Do not fabricate or interpolate metrics for missing checkpoints; mark them absent.
- Do not tune analysis choices (splits, layers, α) per-checkpoint; fix them globally
  in the config before running a phase.
