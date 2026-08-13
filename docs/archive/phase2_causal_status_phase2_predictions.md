# Phase 2 pre-registration — written BEFORE any Phase-2 job is launched

Required by `followup/CLAUDE.md` §Phase 2.4. Committed before the steering and
patching runs so the predictions cannot be fitted to the curves afterwards.

**Status: PRE-RUN.** No Phase-2 measurement exists at the time of writing.
Nothing below may be edited once `run_steering.py` has been launched; the verdict
goes in `phase2_verdict.md` instead.

Frozen alongside this document: `configs/fragility/phase2_default.yaml`
(α grid, scale reference, layer set, checkpoint subset, generation params).

---

## 0. What is already known, and therefore not being predicted

From the original paper (`findings.md` EXP-09, EXP-21):

| measurement | checkpoint | value |
|---|---|---|
| steering Δacc (probe − matched random) | C_outcome, pre-RL | −0.07 … +0.02 (**null band**) |
| steering Δacc at α=1.0 | runA final, post-Goodhart | **+0.083** |
| steering Δacc at α=1.0, assertion direction (control) | runA final | −0.015 |

So the endpoints are known: the probe direction is causally inert before
probe-RL and mildly causal after. **Phase 2 measures the trajectory between
them**, which has never been observed, plus two quantities the original protocol
never recorded (below).

## 1. Two protocol corrections, declared up front

**(a) Layer indexing.** The probe reads `hidden_states[16]`, which in
HuggingFace is the output of `model.model.layers[15]`. The original
`causal_steering.py` hooks `model.model.layers[16]`, whose output is
`hidden_states[17]`. The published steering therefore injected the probe
direction one block *downstream* of where the probe reads.

This does not invalidate the published null — adding a vector one block later is
still a legitimate intervention — but "steering at the probe's layer" is not
literally what was done. Phase 2 runs **both**: `hook_offset=0` (reproduces the
paper, for comparability with the null band and the +0.083) and
`hook_offset=-1` (hooks `layers[15]`, so the modified residual IS the probe's
input). Both are recorded; the paper-matched variant is the headline and the
corrected variant is reported beside it.

**(b) α scale.** `CLAUDE.md` specifies α ∈ {2,4,8} × "activation std". The
published results use α × `h_mean_norm` = α × 21.86. Mean per-dimension std at
L16 is ≈ 0.09, so α=8 × act_std would be ≈ 3% of the paper's α=1.0 and would
return a null by construction. Primary sweep is α ∈ {0.5, 1.0, 2.0} ×
`h_mean_norm`; a secondary sweep uses α ∈ {2,4,8} × `proj_std`. Every condition
records its injected L2 norm so the conventions cross-walk.

## 2. Measurements Phase 2 adds

The original protocol recorded only Δ(task accuracy). Three outcomes are
recorded per steered generation, because they discriminate between hypotheses
that Δaccuracy alone cannot separate:

| outcome | why it discriminates |
|---|---|
| Δ **accuracy** (verifier) | the original quantity |
| Δ **probe score** on the regenerated completion | does pushing the axis make the monitor happier? |
| Δ **template score** and Δ **n_answer_blocks** | does pushing the axis produce the *writing style* the probe was confounded with? **This is the H0-discriminating measurement and it has never been taken.** |

## 3. Predictions

Notation: `t*` = the step at which `auroc_frozen` collapses (runA: between 40 and
60; runB: within the first 10 steps).

### H1 — Evasion (info moves off the probe direction, preserved elsewhere)

1. `causal_strength` on **accuracy** decays toward 0 **at or before `t*`**. The
   direction stops being connected to the answer, which is *why* the frozen
   probe fails.
2. `causal_strength` on **probe score** stays high or rises: the axis is still
   pushable, it just no longer moves the answer. Decoupling is the signature.
3. Patching flip rate at the probe layer **falls**, while flip rate at ≥1
   bracketing layer **rises** (the information went somewhere).
4. Δ template score under steering: no systematic effect.

**Falsified if** accuracy-causal-strength is flat or rising through `t*`, or if
no bracketing layer picks up patching efficacy.

### H2 — Corruption (the correctness information itself degrades)

1. `causal_strength` on accuracy decays **simultaneously with** `t*`, not before.
2. Patching flip rate falls at the probe layer **and at every bracketing layer**.
   Nothing recovers anywhere.
3. Retrained probes (Phase 1) fail too — H2 requires the Phase-1 result to have
   come out "both low". If Phase 1 shows a large positive evasion gap, H2 is
   already dead and Phase 2 is confirmatory only.
4. Δ template score under steering: no systematic effect.

**Falsified if** any layer retains patching efficacy, or if retrained-linear or
retrained-MLP AUROC stays high at any layer.

### H0 — Confound exploitation (the probe scored a writing style)

1. `causal_strength` on **accuracy** stays inside the published null band
   [−0.07, +0.02] for most of training, rising only late and only slightly
   (consistent with the observed +0.083 at runA-final, which is small next to the
   25 pp accuracy drop it is supposed to explain).
2. `causal_strength` on **probe score** rises steeply and early — pushing the
   axis is exactly the thing the policy learned to do.
3. **Δ template score under steering is positive and grows with `t`.** Steering
   along the probe direction should make the model write *more like the template*
   without making it more correct. This is the prediction unique to H0.
4. Patching flip rate on **correctness** stays low at every layer, while patching
   flips the **probe's verdict** readily.
5. Phase 1 will show a large positive evasion gap (H0 mimics H1 there), but
   `auroc_frozen_cond_template` will be near 0.5 while `auroc_retrained_linear`
   stays high — the frozen probe's surviving signal is the confound.

**Falsified if** Δ template score under steering is ≈ 0 at late checkpoints, or
if `auroc_frozen_cond_template` stays high (≥ 0.7) at t=0.

### Summary table

| | Δacc causal strength | Δprobe causal strength | Δtemplate under steering | patch flip (correctness) | bracketing layers |
|---|---|---|---|---|---|
| **H1** | decays at/before t* | high | ≈ 0 | falls at probe layer | ≥1 layer rises |
| **H2** | decays with t* | falls | ≈ 0 | falls everywhere | all fall |
| **H0** | stays ≈ null, small late rise | rises early + steeply | **positive, growing** | stays low everywhere | flat |

## 4. Prior

Stated so it is on the record and can be scored against the outcome.

**H0 is the leading hypothesis, ~0.6.** EXP-21 already documented the mechanism
qualitatively: every probe=1.000 post-Goodhart rollout carried the same
rhetorical scaffold, and the answers inside were not merely wrong but often
invalid (using numbers not in the prompt). The zero-cost control run on the
vanilla-RLOO ladder (`PAPER_NOTES.md`, 2026-07-30) has since put a number on the
confound's strength: a 7-pattern regex predicts correctness at AUROC 0.824 on
C_outcome, and the frozen probe's Spearman correlation with that regex score
rises 0.19 → 0.63 over vanilla RL. runA initialises *from* C_outcome, i.e. from
the checkpoint where the probe is most entangled with style.

H1 ~0.25, H2 ~0.1, other/mixed ~0.05. The likeliest real outcome is a
**mixture** — H0 dominant with an H1 component — and the reporting will
quantify the split rather than declare a winner. Two-thirds of these hypotheses
share the "large positive evasion gap" signature, which is exactly why Phase 2's
Δtemplate measurement, not Phase 1's headline figure, is what actually decides
this.

## 5. Analysis choices, frozen now

- α grid, scale reference, layers, and checkpoint subset: `phase2_default.yaml`.
- 3 random control directions per condition, averaged; effects are always
  reported as **probe minus matched random**, never as a bare probe delta.
- Checkpoints: {0, 20, 40, 60, 90} per ladder. Chosen from the *published*
  trajectory (the cliff sits between 30 and 60 for runA), not from any Phase-2
  data.
- `causal_strength` = max over α of |Δ(probe) − Δ(random)| on accuracy, as
  defined in `fragility_core/steering.py`. **Superseded — see Amendment A1.**
- Bootstrap CIs over prompts (10k resamples); no per-checkpoint tuning of
  anything.
- One-sided claims are not made: a null is reported as a null.

---

## Amendment A1 (2026-08-03) — the estimator behind `causal_strength`

Made while the document is still **PRE-RUN**: no Phase-2 steering or patching
job has been launched, so no curve was seen before this was written. The name
`causal_strength` and everything in §2–§4 are unchanged; only the estimator
under the name changes.

**New definition.** `causal_strength(t)` = the **signed** Δ(probe) − Δ(random)
accuracy contrast at the single pre-registered dose α = `steering.ref_alpha` =
**+1.0**, with a **prompt-clustered percentile bootstrap** (10k resamples, the
CI already frozen in §5 above). Reported with `_ci_lo`, `_ci_hi`, `_se`, `_p`,
`_significant`. The full α grid is unchanged and still reported per α as
`steer_delta_acc_alpha*`; α=1.0 is the summary because it is the point at which
both published anchors in §0 are quoted (+0.083 post-Goodhart, −0.015 for the
assertion control).

**Why.** Both previously-considered summaries were measured against the paper's
own two published steering runs by
`experiments/fragility/phase2_causal_status/validate_estimator.py`, which is the
permanent gate for this decision:

| statistic | postRL (true effect +0.083) | assertion control (true null) |
|---|---|---|
| **signed Δ at α=1 (new)** | **+0.0825, CI [+0.021, +0.148], p=0.011** | **−0.0155, CI [−0.095, +0.067], p=0.75** |
| max\|Δ\| over α (old §5) | 0.0825 — unsigned | 0.0155 — cannot be called a null |
| OLS slope on α | **−0.0722 (wrong sign)** | +0.0110, se 0.0038 (1 dof), **t=+2.89** |

Calibration of the new statistic, by Monte Carlo on a simulated design matched
to that published run (its empirical per-prompt base rates and row counts),
4000 replicates × 2000 resamples: **type-I 0.051 ± 0.007** and CI coverage
0.949 under a true null; **power 0.641** and coverage 0.950 at a true +0.083.
Power 0.64 at n=100 prompts is a real limit of the frozen design — a
non-significant checkpoint is not evidence of no effect, so §2–§4's decay
questions must be read off the CI trajectory, not off the significance flags.
On the same simulated null (N=2000), `max|Δ|` over the 6-alpha `both_signs`
grid has a **95th percentile of 0.0825** — numerically the whole +0.0825 effect
it would be asked to detect. That is why the old §5 summary is retired.

The OLS slope inverts because the dose-response is an inverted U (+0.041,
+0.083, −0.052 at α = 0.5/1/2) and least squares gives the α=2 damage point the
most leverage; on the control its 1-dof residual SE manufactures significance;
and on this config's own sign-symmetric `both_signs` grid it estimates only the
odd component of the dose-response and is exactly 0 for any even one. `max|Δ|`
is unsigned and its null expectation grows with the number of α values searched.

Both are retained as clearly-labelled diagnostics — `diag_ols_slope`,
`diag_maxabs_delta` — each carrying the size of the α grid it was computed over
(`diag_*_n_alphas*`), and `diag_ols_slope_grid_sign_symmetric` flags the grid on
which the slope cannot see a symmetric effect at all.

**Patching.** `flip_rate` reports `patch_effect_{c2w,w2c,mean}` the same way:
signed point estimate plus prompt-clustered bootstrap CI, with
`patch_effect_mean` bootstrapped on the same cluster draw as its components.

**Read the intervals as paired.** The bootstrap resamples prompts jointly across
the probe and random arms, so shared prompt difficulty cancels in the contrast
(prompt-level corr(probe, random) = 0.56 on the published postRL run). That
makes the paired-clustered interval narrower than the same bootstrap with the
arms resampled independently (0.127 vs 0.185 at α=1) and narrower than a
row-independent one (0.163) — which is the intended behaviour of a paired
design, not an undercount of the clustering.

---

## Amendment A2 (2026-08-03) — Δ(probe score) under steering cannot be measured

Made while the document is still **PRE-RUN**: no Phase-2 steering or patching job
has been launched, so no curve was seen before this was written. §3's H1/H2/H0
predictions are unchanged; what changes is that one of the three outcomes in §2
is **withdrawn as unmeasurable under this protocol**, rather than reported as a
null.

**What is withdrawn.** §2's row "Δ **probe score** on the regenerated completion",
and with it the H1 prediction 2 ("`causal_strength` on probe score stays high or
rises") and the H0 prediction 2 ("rises steeply and early"). Those two rows of
§3's summary table are now **untestable by Phase 2 steering**, not tested-and-null.

**Why.** The probe reads the **first `</think>`**, which is the last thing in the
steered *prefix*. Every token steering produces comes *after* it. The only
non-circular way to score the probe on a steered rollout is with the hook off
(with the hook armed the score is the mechanical identity `probe(h + α·s·v)`),
and with the hook off, under causal attention, the hidden state at that position
is a function of the tokens at or before it — i.e. of the prefix alone. The
prefix is byte-identical across the `zero`, `probe`, `assertion` and `rand`
conditions, and BPE is left-to-right, so appending the generated text cannot
change the tokens at or before that position (checked on 160 real rollouts:
160/160 identical). The measurement is therefore **identically zero by
construction**, for every checkpoint and every α.

`run_steering.py` previously computed it anyway, at the cost of one extra full
forward pass per generation, and wrote the zeros into the metrics store where
they would have read as "no effect on the monitor's verdict". The measurement and
its derived metrics are deleted rather than kept.

**What would be needed instead.** A probe read *after* the steered tokens — a
different position rule than the one the probe was calibrated on — which is a
different experiment, not a column of this one. It is not being substituted in
here, because swapping the read position after pre-registration is exactly the
kind of move this document exists to prevent.

**Recorded, not changed: what the §1(b) secondary sweep actually injects.**
Measured on the Phase-0 runA step-0 cache (L16, n = 3216 rows), which is pre-run
data: `h_mean_norm` = 21.836, `proj_std` = 0.0624, `act_std` = 0.1158. So the
frozen secondary sweep α ∈ {2, 4, 8} × `proj_std` injects L2 norms of
0.125 / 0.250 / 0.499 — its **largest** dose is **2.3 %** of the primary sweep's
α = 1.0 (21.836), the same order of magnitude §1(b) rejected `act_std` for. That
is not an error in the sweep: "8 standard deviations of this feature" is a
legitimately different and more interpretable question. But a null in that arm is
a null **at that magnitude**, and is not comparable to the primary arm's null
band. The alphas are **not** rescaled — they are pre-registered — and
`run_steering.py` now prints the injected L2 norm of every dose in every sweep
and warns when a sweep's largest dose falls below 20 % of the primary's headline
dose, so the comparison cannot be made by accident.

**Unaffected.** Δaccuracy and Δtemplate/Δn_blocks are untouched, and Δtemplate —
§4's "what actually decides this" — is now reported **signed**: per α, at −α, and
at the pre-registered α = 1.0 with the same prompt-clustered bootstrap CI as the
accuracy headline. The assertion direction, already in `phase2_default.yaml`'s
`directions`, is now aggregated on the same footing as the probe direction
(`assertion − matched random`, per α and at the headline dose) for **every**
outcome; previously those generations were produced and never aggregated. §1(b)'s
secondary `proj_std` α-sweep now runs as its own arm, tagged
`steering_off{offset}_{scale_ref}`; it does not contain α = 1.0, so it records
`*_ref_alpha_missing` rather than a headline read off a different dose.
