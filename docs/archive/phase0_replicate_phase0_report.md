# Phase 0 report — runA ladder complete, runB still at two checkpoints

Required by `followup/CLAUDE.md` Phase 0.5.

Last updated 2026-07-31.

**Metric conventions used in this file.** AUROC is reported **balanced**
(positives and negatives subsampled to equal size) unless a column says
otherwise, because the positive rate falls from 0.36 to 0.12 across runA and the
unbalanced metric is not comparable across steps. All AUROC, probe-score and
`frac > 0.95` figures are computed on the **val split** (n = 975–984 rollouts,
prompt-level split, fixed across every checkpoint). True accuracy and block counts
are computed on **all harvested rollouts** (n ≈ 3216–3248). Layer is L16, the
probe's own layer, and seed 0, unless stated.

## Acceptance criteria, scored

| CLAUDE.md criterion | status |
|---|---|
| frozen-probe reward hacking reproduces (probe up, accuracy down) | **met** |
| ≥ 30 checkpoints with cached activations | **not met** — 11 harvested for runA (the full archival ladder), 2 for runB |
| collapse reproduces at small scale | **met** — this *is* the small-model track (Qwen2.5-0.5B) |

The checkpoint-count criterion is met against the *available* ladder, not against
CLAUDE.md's target: both published runs were trained with
`--save_every_n_steps 10`, so 11 is all that exists without paying for a re-run.
The K=10 spacing turns out to be adequate — the transitions below are resolved
across three or more checkpoints each, not hidden inside a single 10-step gap —
so the K=3 re-run (`dense_probe_rloo.py`) has not been bought.

## Reward hacking reproduces, on-policy, on held-out prompts

406 contamination-filtered prompts × 8 rollouts, temp 1.0 (the RL sampling
regime):

| ladder | step | frozen probe (bal., val) | mean probe score (val) | frac > 0.95 (val) | true accuracy (all rows) |
|---|---|---|---|---|---|
| runA (init C_outcome) | 0 | 0.785 | 0.460 | 0.050 | 0.372 |
| runA | 99 | **0.533** | **0.989** | **0.971** | **0.133** |
| runB (init C_SFT) | 0 | 0.579 | 0.498 | 0.076 | 0.159 |
| runB | 99 | 0.516 | 0.996 | 0.991 | 0.068 |

The probe reports > 0.95 on 97–99% of rollouts while true accuracy has fallen by
24 pp (runA) and 9 pp (runB). This is on **held-out prompts the policy never
trained on**, so it is a monitor failure, not merely gaming of the training
distribution.

## The three-AUROC comparison

Every column below is balanced AUROC, L16, val split, seed 0 — one aggregation per
column, no mixing:

| ladder | step | frozen | retrained linear | retrained MLP | evasion gap |
|---|---|---|---|---|---|
| runA | 0 | 0.785 | 0.815 | 0.784 | +0.030 |
| runA | 99 | 0.533 | 0.695 | 0.751 | **+0.163** |
| runB | 0 | 0.579 | 0.661 | 0.639 | +0.082 |
| runB | 99 | 0.516 | 0.648 | 0.608 | +0.132 |

The control ladder is quoted separately because it is the one place where the
exactly reproducible unbalanced metric matters: balanced AUROC depends on the
subsample draw and moves by up to ~0.015 between re-analyses of the same
activations, while unbalanced values are bit-stable. Over the vanilla-RLOO
ladder, L16, mean of 3 seeds, **unbalanced**: frozen 0.768 → 0.911, retrained
linear 0.809 → 0.903, retrained MLP 0.801 → 0.894, and the evasion gap stays
within **[−0.021, +0.041]** across all 5 checkpoints × 3 seeds. The balanced band
has been measured at [−0.024, +0.041] and [−0.021, +0.041] on two analysis passes:
its lower edge jitters, its **upper edge is +0.041 in both**, and the upper edge is
what every claim below is tested against.

That null band is what makes the probe-RL gaps readable: runA's +0.163 is four
times its upper edge. Under verifier RL the frozen probe *strengthens*; under
probe RL it falls to chance while a fresh probe keeps working.

## The full runA ladder: a two-phase collapse

With all 11 checkpoints harvested, the collapse resolves into two mechanistically
distinct phases. Frozen, retrained and gap are balanced; `AUROC(templ)` is
unbalanced (its conditional counterpart has no balanced variant); `sep` and
accuracy have no balanced variant and are computed on val rows and all rows
respectively:

| step | frozen | retrained | gap | sep | AUROC(templ) | corr(probe,templ) | acc |
|---|---|---|---|---|---|---|---|
| 0 | 0.785 | 0.815 | +0.030 | 1.096 | 0.743 | 0.531 | 0.372 |
| 10 | 0.768 | 0.782 | +0.014 | 1.021 | 0.721 | 0.548 | 0.308 |
| 20 | 0.779 | 0.809 | +0.030 | 1.200 | 0.742 | 0.589 | 0.312 |
| 30 | 0.766 | 0.772 | +0.006 | 1.050 | 0.723 | 0.547 | 0.291 |
| 40 | 0.767 | 0.799 | +0.032 | 0.941 | 0.694 | 0.475 | 0.302 |
| 50 | 0.708 | 0.737 | +0.029 | 0.585 | 0.572 | 0.226 | 0.260 |
| 60 | 0.646 | 0.690 | +0.043 | 0.488 | 0.542 | 0.072 | 0.186 |
| 70 | 0.541 | 0.681 | **+0.140** | 0.242 | 0.563 | 0.024 | 0.148 |
| 80 | 0.572 | 0.724 | +0.152 | 0.267 | 0.562 | 0.023 | 0.148 |
| 90 | 0.621 | 0.662 | +0.041 | 0.369 | 0.539 | 0.028 | 0.141 |
| 99 | 0.533 | 0.695 | **+0.163** | 0.171 | 0.584 | 0.037 | 0.133 |

- **Steps 0–40 — reward hacking with an intact monitor.** The policy is already
  gaming hard (mean probe score 0.460 → 0.677, blocks per rollout 5.0 → 13.6,
  accuracy 0.372 → 0.302) while the monitor is fine: frozen AUROC 0.785 → 0.767
  and the gap stays inside the balanced null band at every step.
- **Steps 50–60 — general degradation.** Frozen and retrained fall *together*
  (0.767 → 0.646 and 0.799 → 0.690). `sep` along the fixed t=0 direction halves,
  and the structural confound decouples in the same window.
- **Steps 70–99 — divergence.** Retrained stabilises at 0.66–0.72 while frozen
  keeps falling to 0.533. Step 60's gap (0.043) is the first to exceed the band
  top of 0.041, by 0.002; step 70 is the first to clear it with margin.

**The headline ordering result: reward hacking starts ~40 steps before the
monitor stops working.** A deployment watching the probe's *score* would have seen
it climbing from step 10; a deployment watching the probe's *discrimination* would
have seen nothing wrong until step 50. Score inflation and discrimination loss are
separate events.

Confirmed at 3 seeds (mean ± sd, L16, balanced): the gap is 0.041 ± 0.011 at step
40, 0.062 ± 0.029 at 50, 0.036 ± 0.007 at 60, and 0.117 ± 0.020 at 70 — the first
step that clears the band with sd clearance. Full 3-seed table in `PAPER_NOTES.md`.

## Verdict on H1/H2/H0

**Mixture, H2 excluded.** Retrained probes retain 0.695 linear / 0.751 MLP at step
99 where the frozen probe is at chance, so the correctness information is not
destroyed. But retrained AUROC also falls (0.815 → 0.695), so this is not clean
H1 evasion either. Full reasoning and the H0 confound table are in
`PAPER_NOTES.md` (runA entries, 2026-07-30 and 2026-07-31).

**No relocation across depth.** Balanced retrained AUROC declines at all five
harvested layers between step 0 and step 99 (L12 0.811→0.703, L14 0.806→0.732,
L16 0.815→0.695, L18 0.804→0.670, L20 0.813→0.687). No layer picks up what L16
loses.

**Retracted: the linear→nonlinear crossover.** The two-checkpoint version of this
report gave "the MLP overtakes the linear probe at step 99" as its headline
mechanism. The full ladder does not support it: MLP and linear track each other
with no consistent ordering across the 11 checkpoints in either AUROC convention,
and the 3-seed `nonlinear_gain` flips sign from step to step. Two checkpoints
could not separate a trend from seed noise. Do not use that claim.

## Which ladder is the right one

runA, and it is not close. It initialises from C_outcome — the probe's own
training checkpoint — so its step 0 is a genuine in-distribution baseline (frozen
0.785, matching the pickle's 0.810 held-out figure). runB initialises from C_SFT,
where both the probe (0.579) and the structural confound (AUROC(template) 0.563,
unbalanced) are weak before the attack begins, leaving little room to measure a
decline. runB is a robustness check, not the primary ladder.

This corrects the ordering recommendation in the first version of
`ASSESSMENT.md`, which put runB first on the grounds that its collapse is faster.
That is true of the *dynamics* and irrelevant to the *mechanism*.

## What is still open

1. **runB's 9 remaining checkpoints** (~$20). runB's two endpoints cannot settle
   relocation: unbalanced retrained AUROC falls at every layer, but balanced
   retrained AUROC rises at L12 (0.636 → 0.662) and L14 (0.636 → 0.643). One seed
   and two checkpoints is not enough to call it either way.
2. **Phase 2 causal status.** Whether the probe direction's causal strength decays
   before, with, or after the AUROC collapse. Pre-registered in
   `phase2_predictions.md`; do not read the predictions after seeing the data.
3. **The transfer matrix on probe-RL checkpoints.** The recorded transfer results
   cover only the vanilla control ladder, so the "does the attack transfer across
   probes" question is still unanswered. It runs off already-harvested
   activations, so it is a $0 job.

## Reproduce

```bash
# full runA ladder (11 checkpoints)
modal run --detach modal_fragility.py harvest -- \
    --config phase0_harvest_runA.yaml --steps 0,10,20,30,40,50,60,70,80,90,99 \
    --arms on_policy
modal volume get default-proj-training fragility/acts/phase0_harvest_runA acts/
python experiments/fragility/phase1_evasion_vs_corruption/run_phase1.py \
    --config phase1_default.yaml --run_id phase0_harvest_runA --arm on_policy \
    --seeds 0,1,2
```

Cost: ~$5 per ladder for a 2-checkpoint pair; ~$20 per ladder for all 11.

## Engineering notes from the harvest

- Modal's launcher commits the volume once, in a `finally`. The harvester now
  commits after every checkpoint, or a crash at checkpoint 9 of 11 would discard
  all nine and the documented resume behaviour would be fiction.
- `PYTHONUNBUFFERED=1` is set for all jobs. Without it a multi-hour job's progress
  appears only at exit, making a hung job indistinguishable from a quiet one.
- `latest_checkpoint` is mapped to step 99 via `final_step` in the ladder config;
  its step number is not recoverable from the path.
- Step 90 is anomalous on every measure (gap 0.041 against 0.152 at step 80 and
  0.163 at step 99). The published trajectory records a single-batch KL spike to
  1040 at exactly this step, so the checkpoint is odd, not the metric noisy. It is
  reported as measured, not smoothed.
