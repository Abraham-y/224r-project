# Concealment Under Outcome Pressure at 0.5B

*A probe-as-measurement study of internal-vs-verbalized correctness on outcome-RL'd Qwen2.5-0.5B on Countdown.*

> **Date of writeup.** 2026-05-30 (Day 1 of milestone+poster preparation).
> **Status.** All in-scope analyses complete; the report can be drafted from this document.

---

## 0. TL;DR (one-paragraph)

We use Countdown's exact-verifier setting to measure the concealment gap between a small reasoner's internal hidden-state representation of correctness and its verbalized confidence, across two RL checkpoints we already have: `C_SFT` (Anikait Singh's Qwen2.5-0.5B Countdown SFT) and `C_outcome` (outcome-RLOO from `C_SFT`). The naive hypothesis "outcome RL widens the concealment gap globally" *inverts* at 0.5B: outcome RL produces a *better-calibrated verbal signal* (verbalized AUROC 0.79 vs 0.57 on `C_SFT`) and the global gap *narrows* from +0.15 to +0.01. But at the specific token positions where the model verbalizes confidence, the internal correctness representation has *collapsed to chance* (probe AUROC 0.52 on `C_outcome` vs 0.74 on `C_SFT`), with the Yuan-et-al within-problem effect size dropping from +1.26 (large) to +0.38 (small), p = 0.004. Cross-checkpoint probe transfer fails in both directions (off-diagonal AUROC 0.52–0.58), indicating representation drift rather than signal suppression within a shared subspace. **The model has learned a globally calibrated *policy* for when to verbalize confidence, while the internal mechanism producing those verbalizations has decoupled from its own correctness representation.** This is behavioral calibration via mechanism decoupling — a more sophisticated form of concealment than the naive "the model says confident things while internally knowing they're wrong" framing.

---

## 1. Setup

**Task.** Countdown arithmetic reasoning (Gandhi et al. 2024): each problem gives 3–4 small integers and a target; the model must produce an equation that uses each number exactly once and evaluates to the target. The rule-based verifier in `evaluation/countdown.py` scores each response 0.0 (no parseable answer), 0.1 (parseable but wrong/invalid), 1.0 (correct).

**Model.** Qwen2.5-0.5B base + Countdown SFT throughout. We did not train an SFT model ourselves; `C_SFT` is Anikait Singh's `asingh15/qwen-sft-countdown-defaultproj`, used in lieu of a team-trained SFT.

**Checkpoints studied.**

| Checkpoint | Construction | Test pass@1 | Test pass@16 |
|---|---|---|---|
| `C_SFT` | `asingh15/qwen-sft-countdown-defaultproj` | 28.6% | 78.0% |
| `C_outcome` | RLOO from `C_SFT`, outcome reward only (0/0.1/1.0), 100 steps, all 10 intermediate snapshots persisted | **53.5%** | 72.0% |
| `C_process` | RLOO with annotation-only subgoal reward (`R = R_outcome + 0.3·R_subgoal`) on `C_SFT_aug` | underperformed; see §8 |

Outcome RL improved pass@1 by 24.9 absolute points but *reduced* pass@16 by 6 points — the standard sharpening-vs-diversity trade-off in RL fine-tuning.

**Eval set.** `asingh15/countdown_tasks_3to4` test split, 50 held-out problems × 16 rollouts per problem at temperature 0.6, top_p 0.95, top_k 20, max_tokens 1024.

**Probe pipeline.** Hidden states cached from Qwen2.5-0.5B with `output_hidden_states=True` at three position kinds per rollout:
- `pre_answer`: the `</think>` token (one vector per rollout).
- `assertion`: every occurrence of a confidence-keyword token in the `<think>` body (`"Perfect"`, `"this works"`, `"got it"`, `"the answer is"`, `"verified"`).
- `neutral`: matched-count random positions from the same `<think>` body (control).

Probes are logistic regression at three layers (L12/L16/L20), L2-regularized (C=0.1), trained per checkpoint with `GroupKFold(5)` on `prompt_idx` (held-out problems, not held-out trajectories) on a 50/50 class-balanced subsample. L16 is the primary report layer; the same patterns hold at L12 and L20.

---

## 2. The headline finding

### 2.1 Position-resolved probe AUROC: assertion-position signal collapses

The probe at confidence-asserting tokens loses its informativeness on `C_outcome` while the trace-final probe stays strong.

| Position (L16, balanced 5-fold CV) | `C_SFT` AUROC | `C_outcome` AUROC | Δ |
|---|---|---|---|
| `</think>` (trace-final) | 0.724 | 0.793 | +0.069 |
| confidence-asserting tokens | **0.735** | **0.520** | **−0.215** |
| neutral tokens (control) | 0.600 | 0.464 | −0.136 |

Sanity baselines (shuffled-label and random-direction probes) are 0.44–0.53 across all cells, well below the trained probes; the signal is real.

**Reading.** Outcome RL leaves the trace-final correctness representation intact (slightly strengthens it, +0.07). But at the token positions where the model emits confidence keywords, the probe drops from "informative" (0.735) to "near chance" (0.520). On `C_SFT`, the assertion-position probe was nearly as informative as the trace-final probe; on `C_outcome`, the gap between trace-final (0.79) and assertion-position (0.52) is enormous. The internal representation at the moments of verbal certainty has decoupled from the eventual correctness.

### 2.2 Matched-pair within-prompt: probe inverts on C_outcome

For each prompt with *both* a correct and a wrong rollout containing assertion tokens, we compute mean assertion-token probe score on each side and the within-prompt delta.

|   | `C_SFT` | `C_outcome` |
|---|---|---|
| n prompts with both correct & wrong rollouts | 38 | 33 |
| mean(correct\_mean − wrong\_mean) | **+0.172** | **+0.004** |
| median Δ | +0.158 | **−0.045** |
| % prompts where probe ranks correct > wrong | **76%** | **36%** |
| % prompts where probe ranks wrong > correct | 24% | **64%** |

**Significance** (Wilcoxon signed-rank, one-sided > 0): C_SFT p = **0.000131**; C_outcome p = 0.72.
**Mann-Whitney U between checkpoints**: p = **0.00155**.
**Paired t-test confirms** (C_SFT p = 8.2e-05; C_outcome p = 0.45).

**Reading.** On `C_outcome`, *within the same problem*, the probe at assertion tokens ranks wrong rollouts above correct ones in **64%** of prompts. Not just uninformative — actively backwards. The matched-pair construction controls for problem-level easiness and isolates "the model knows which response of this problem is correct" from "this problem is easy." That isolation now confirms: the model doesn't know, at the assertion-token positions.

### 2.3 Within-problem Cohen's d: the Yuan-et-al benchmark

For each held-out problem with both a correct and wrong rollout, Cohen's d of the probe scores at `</think>` between correct and wrong samples (controls for problem-level difficulty).

| | `C_SFT` | `C_outcome` |
|---|---|---|
| n problems | 39 | 33 |
| mean d | **+1.26** (large) | **+0.38** (small) |
| median d | +0.85 | +0.11 |
| IQR | [+0.32, +1.39] | [−0.24, +0.75] |

**Mann-Whitney U on the two distributions**: p = **0.00396**.

**Reading.** The within-problem effect size drops from large (+1.26) to small (+0.38) under outcome RL — a 70% reduction. C_outcome's IQR straddles zero, meaning on roughly half of problems the probe ranks wrong rollouts above correct ones. This is directly comparable to Yuan et al. 2026's reporting at 1.5B+.

### 2.4 Cross-checkpoint probe transfer (2×2 matrix): representation drift

Train probe on `C_X`'s activations; evaluate on `C_Y`'s activations. Diagonals are leakage-free (held-out CV); off-diagonals use a probe trained on the *entire other* checkpoint's data and evaluated here.

**`</think>` position (L16, balanced AUROC):**

| train ↓ \ eval → | C_SFT | C_outcome |
|---|---|---|
| C_SFT | 0.724 | **0.523** |
| C_outcome | **0.580** | 0.793 |

**confidence-asserting tokens (L16):**

| train ↓ \ eval → | C_SFT | C_outcome |
|---|---|---|
| C_SFT | 0.735 | **0.588** |
| C_outcome | **0.623** | 0.520 |

**Reading.** Off-diagonal AUROCs collapse to 0.52–0.62 — the probe trained on one checkpoint barely beats chance on the other. This is the **representation-drift** signature (in the sense of Taufeeque et al., "Obfuscation Atlas"): the hidden-state subspace encoding "this rollout is correct" has *moved* under RL, not just been *suppressed* within a shared subspace. Trained probes are checkpoint-specific.

### 2.5 The global concealment gap *inverts* at 0.5B

Verbalized confidence at the global rollout level (binary keyword-presence in the `<think>` body — see §5 on why):

|   | `C_SFT` | `C_outcome` |
|---|---|---|
| P(keyword \| correct) | 0.834 | **0.951** |
| P(keyword \| wrong) | 0.695 | **0.379** |
| Verbalized AUROC | 0.569 | **0.786** |
| Trace-final probe AUROC | 0.724 | 0.793 |
| **Global gap = probe − verbal** | **+0.154** | **+0.008** |

**Reading.** This is the *opposite direction* from the naive "concealment gap widens under outcome RL" prediction. At the global rollout level, outcome RL made the model's verbalization *more* calibrated: when `C_outcome` says "this works", it's correct 95% of the time vs 38% on wrong rollouts. The global gap closes to essentially zero. This is a clean scale-dependence finding relative to Yuan et al.'s 1.5B+ result.

The deeper picture, however, comes from combining §2.1–2.4 with §2.5. Globally the surface is calibrated; *moment-by-moment* the internal representation at the verbalization tokens has decoupled. The model has learned a calibrated *policy* for when to be verbally confident — but the mechanism producing those verbalizations is no longer tied to the model's internal correctness representation.

---

## 3. Behavioral evidence (Layer A)

Diagnostics on a single rollout per problem (`response[0]` at temperature 0.6):

| Metric | `C_SFT` (n=50) | `C_outcome` (n=50) |
|---|---|---|
| Overall accuracy | 34% | 54% |
| Answer well-formed | 92% | 96% |
| Any answer parseable | 92% | 98% |
| Uses each input number exactly once | 58% | 82% |
| CoT-answer consistency (last CoT-endorsed value matches `<answer>`) | 71% | 83% |
| **Confident-wrong rate** (over wrong rollouts) | **79%** | **43%** |

**Several things worth noting:**

- *Outcome RL improved rule-following.* Number-usage compliance jumped from 58% to 82%.
- *Confident-wrong rate fell* from 79% → 43% (over wrong rollouts), and 52% → 20% in absolute terms. This is the surface anti-H2 signal that motivated us to look at position-resolved measurements.
- *Self-undermined verification is the dominant `C_outcome` failure mode.* Inspecting wrong `C_outcome` rollouts shows the model *frequently emits the correct answer in an early `<answer>` block*, then keeps generating, drifts into a wrong derivation, and the verifier scores only the last `<answer>`. See §2 of `extension/probe/diagnose_outcome.py`'s output for concrete examples (e.g., target 28, nums=[95, 11, 56]: 5 early `<answer>(95 − 56) − 11</answer>` (correct) followed by drift to `(95/56) − 11`).
- *Structural degeneration.* On `C_outcome`, mean `<answer>` tag count per rollout is ~7.6 (median ~6); zero rollouts terminate cleanly within `max_tokens=1024`. This is a real surface artifact that affects the standard pass@1 measurement.

---

## 4. Probe-level evidence (Layer B)

The headline findings are in §2. Additional structural observations:

### 4.1 Layer invariance

The same pattern holds at L12 and L20.

| Layer | `C_SFT` `</think>` | `C_outcome` `</think>` | `C_SFT` assertion | `C_outcome` assertion |
|---|---|---|---|---|
| L12 | 0.723 | 0.790 | 0.693 | 0.465 |
| L16 | 0.724 | 0.793 | 0.735 | 0.520 |
| L20 | 0.753 | 0.794 | 0.735 | 0.489 |

The position-resolved collapse is consistent across layers.

### 4.2 Per-keyword breakdown

Only "this works" has enough samples to compare at L16 (other keywords' sample sizes shrink dramatically under `C_outcome` — see §3 above; the model verbal style is narrower under RL).

| Keyword | n (`C_SFT`) | `C_SFT` AUROC | n (`C_outcome`) | `C_outcome` AUROC |
|---|---|---|---|---|
| "this works" | 329 | 0.663 | 545 | 0.529 |
| "Perfect" | 25 | 0.567 | <20 (filtered) | — |
| "got it" | 49 | (4% positive — can't balance) | <20 (filtered) | — |
| "the answer is" | <20 | — | <20 | — |
| "verified" | <20 | — | <20 | — |

Outcome RL trained the model to use "this works" more (329 → 545 occurrences) but with *less* internal correctness signal at those positions (0.663 → 0.529). The collapse isn't keyword-specific — it's a property of the assertion-context hidden state in general.

### 4.3 Probe family baselines (linear vs nonlinear)

Are we measuring "the signal is gone" or "the signal is gone *linearly*"? Random forest and small MLP baselines:

| Cell (L16) | LR | RF | MLP |
|---|---|---|---|
| `C_SFT` `</think>` | 0.724 | 0.767 | 0.767 |
| `C_SFT` assertion | 0.735 | 0.754 | 0.719 |
| `C_SFT` neutral | 0.600 | 0.623 | 0.612 |
| `C_outcome` `</think>` | 0.793 | 0.844 | 0.821 |
| **`C_outcome` assertion** | **0.520** | **0.596** | **0.597** |
| `C_outcome` neutral | 0.464 | 0.449 | 0.496 |

**Reading.** Nonlinear probes recover ~0.06–0.08 of additional AUROC on the headline `C_outcome` assertion cell (0.52 → 0.60). So the signal is *largely* but not entirely gone *linearly*. C_SFT's signal is recovered similarly across LR/RF/MLP (~0.72–0.75), so the post-RL representation isn't "just rotated" relative to SFT — it's compressed/destroyed enough that even nonlinear classifiers can't get above ~0.60.

---

## 5. Verbalized confidence

**The intended elicitation didn't work.** We attempted two literature-standard methods:

1. **Generated [0,100] confidence elicitation** (RLCR / Damani et al. style): append a confidence-prompt suffix in chat-template form, generate ~8 tokens, parse an integer. **Failed**: the SFT'd Qwen base is not chat-tuned and treats *any* prompt as the start of a new Countdown problem. Every response was a degenerate `" 0\nLet me try to find..."`. Verbalized confidence was 0 for every rollout.
2. **Token-logprob elicitation** (more sophisticated; query `P("yes" | prompt+response+verdict-suffix)`): same prompt, but extract the first-token logprob of yes/no from the top-20. **Failed**: the model's first token after any prompt is `<th` (start of `<think>`) — neither yes nor no is in the top-20 for most rollouts. `logprob_no` is absent for 100% of rollouts; the salvageable AUROC of `logprob_yes` alone is 0.509 (`C_SFT`) and 0.504 (`C_outcome`), both essentially chance.

The model is **too narrowly tuned to its training distribution to be queried via verbal elicitation at all.** This is itself a meaningful negative result about the limits of probing-via-elicitation at small scale.

**Defensible fallback: keyword-presence as the verbalized signal.** We use binary "does the `<think>` body contain any of `{Perfect, this works, got it, the answer is, verified, this is correct, confirmed, found it}`?" as a per-rollout proxy for verbalized confidence. The AUROC of this binary signal against correctness is what we report in §2.5.

This is defensible because (a) it's the model's *own* verbalization (no cross-model judge), (b) the diagnostic in §3 already shows the model uses these keywords as the primary surface signal of confidence, and (c) it gives a *positive* AUROC (0.57 / 0.79) — informative. The limitation is honestly that this is binary, not graded; an instruction-tuned model would be required for a graded verbalized confidence.

---

## 6. Training dynamics (Layer C)

Re-cache hidden states on each of the 10 saved `C_outcome` snapshots (steps 0, 10, …, 90), re-run the analysis per snapshot. **Methodology caveat**: we used Option A — the *same rollouts* (sampled from final `C_outcome`) across all 10 snapshots, with only the model's weights changing per snapshot. So this measures "how well does each snapshot's model read correctness from the final-checkpoint's text" rather than "the gap evolving with the model's own behavior over training." (Option B — re-sample fresh rollouts per snapshot — would cost ~3-6× and was out of scope.)

**L16 trajectory (balanced AUROC):**

| step | `</think>` | assertion | neutral | assert − neutral |
|---|---|---|---|---|
| 0 | 0.802 | 0.474 | 0.463 | +0.011 |
| 30 | 0.816 | 0.499 | 0.457 | +0.042 |
| 60 | 0.791 | 0.512 | 0.455 | +0.057 |
| 90 | 0.794 | 0.516 | 0.464 | +0.052 |

The L12 and L20 trajectories are essentially identical.

**Reading.** The trace-final probe AUROC is stable at ~0.80 across all 10 snapshots. The assertion-position AUROC sits at chance (~0.50) throughout. The slow assertion-minus-neutral drift (+0.01 → +0.05) is real but small.

The biggest takeaway from dynamics: **outcome RL does not damage the model's trace-final correctness representation over training** (~0.80 throughout). The position-resolved collapse seen in §2 between `C_SFT` (probed on `C_SFT` rollouts) and `C_outcome` (probed on `C_outcome` rollouts) is therefore primarily attributable to a *behavioral shift* — the model placing confidence-keyword tokens at *different positions* in the trace post-RL — not to a destruction of the underlying representation. This nuances the §2 finding: the *representation* survives, but the verbalization-context positions sampled from no longer carry it.

---

## 7. Methodological controls

### 7.1 Significance tests (summary)

| Test | Result |
|---|---|
| Wilcoxon signed-rank on `C_SFT` matched-pair deltas, one-sided > 0 | **p = 0.000131** |
| Wilcoxon signed-rank on `C_outcome` matched-pair deltas, one-sided > 0 | p = 0.72 |
| Mann-Whitney U between `C_SFT` and `C_outcome` deltas, one-sided | **p = 0.00155** |
| Paired t-test on `C_SFT` deltas | **p = 8.2e-05** |
| Paired t-test on `C_outcome` deltas | p = 0.45 |
| Mann-Whitney U on per-problem Cohen's d distributions | **p = 0.00396** |

All headline effects are significant at p < 0.005 by at least two independent statistical tests. The null effect on `C_outcome` (matched-pair median Δ = −0.045, p = 0.72) is itself the support for the H3-direction story.

### 7.2 Class balance

All AUROCs in this writeup are **balanced 5-fold CV by held-out problem** (`GroupKFold(5)` on `prompt_idx`, then a 50/50 class-balanced subsample within each fold's data). This removes the confound that `C_outcome`'s assertion data is 76% positive class (since RL increased both accuracy and conditional keyword-on-correct usage) while `C_SFT`'s is 41% positive.

### 7.3 Sanity checks per probe

Reported per-cell with each probe AUROC:

- **Shuffled-label AUROC**: ~0.44–0.53 across all cells.
- **Random-direction AUROC** (project onto a random unit vector, direction-agnostic): ~0.52–0.72.
- **Linear vs RF vs MLP**: §4.3.

The shuffled-label baselines being near 0.5 (where they should be) and the random-direction baselines being modestly above (as expected — high-dimensional random projections of correlated activations) both confirm the probe AUROCs reflect real, learned, label-tied signal.

---

## 8. Failed intervention: `C_process` (Appendix)

Documented in detail in `extension.md` §A. Brief recap:

- We attempted a process-reward arm using annotation-only `<subgoal> reach X from [Y, Z] </subgoal>` tokens with the composite reward `R = R_outcome + 0.3·R_subgoal`, where `R_subgoal` rewards subgoals whose target is reachable from declared inputs *and* whose body computes the claimed value. Both validity and achievement use noise-free exact-arithmetic verifiers.
- `C_SFT_aug` (SFT on subgoal-augmented warm-start traces) learned the *grammar* (~92% of rollouts emit `<subgoal>` tags) but not the semantics.
- `C_process` (RLOO from `C_SFT_aug`) **underperformed** `C_outcome` on accuracy. Qualitatively, the subgoals were emitted in dead-end search branches rather than along the actual solution path — the tags were *annotations* on text the model would have produced anyway, not interventions that changed inference.
- This is the predicted outcome of **Strategic Information Allocation** (Kim et al., March 2026): at small scale, annotation-only tokens cannot route capability into a reasoner without a separate mechanism (no inference-time grounding, no external execution). We report this as a complementary negative result whose mechanism confirms a recent theoretical prediction.

---

## 9. Limitations

- **n = 50 problems.** Matched-pair denominators of 33–38 keep CIs wider than ideal. A 200–500-prompt eval would tighten the percentages but unlikely change direction.
- **Verbalized confidence is keyword-presence, not elicited.** The two literature-standard elicitation attempts (generated [0,100] and token-logprob yes/no) both broke because the SFT'd Qwen base is not chat-tuned. We use a binary keyword proxy and disclose this clearly.
- **Layer C is Option A.** Per-snapshot hidden states are extracted on the *same* (final-checkpoint) rollouts, so dynamics reflect "what does each snapshot's model read from final-checkpoint text" rather than "the gap evolving with the model's own behavior over training." A proper Option B (per-snapshot fresh rollouts) was out of scope.
- **The C_SFT we use is `asingh15/qwen-sft-countdown-defaultproj`, not a team-trained SFT.** This is documented in `extension.md` §10.
- **All experiments are at 0.5B.** The scale-inversion of the *global* concealment gap relative to Yuan et al. 1.5B+ is itself a finding, but we cannot claim our position-resolved finding generalizes to larger models without further work.

---

## 10. What's in the figures

All under `extension/outputs/figures/`, generated by `extension/probe/make_figures.py`:

| Figure | What it shows | Headline number |
|---|---|---|
| `fig1_matched_pair_scatter.png` | Per-prompt mean assertion-probe scores, correct vs wrong rollouts, two panels (C_SFT, C_outcome) | 29/38 above diagonal → 12/33 above diagonal |
| `fig2_within_problem_d.png` | Distribution of per-problem Cohen's d | mean shifts +1.26 → +0.38 |
| `fig3_position_bar.png` | Balanced probe AUROC at three position kinds | assertion: 0.74 → 0.52 |
| `fig4_per_keyword_bar.png` | Per-keyword assertion-position AUROC | "this works" 0.66 → 0.53 |
| `fig5_dynamics_trajectory.png` | L16 probe AUROC over training step | trace-final stable at 0.80; assertion at chance |
| `fig6_transfer_heatmap.png` | 2×2 cross-checkpoint transfer matrix | off-diagonals 0.52–0.62 |
| `fig7_concealment_gap.png` | Global concealment gap (probe − verbal) | +0.15 → +0.01 (inverts naive H2) |

`extension/probe/diagnose_outcome.py` produces a sample-rollout report including the confident-wrong examples for the qualitative figure.

---

## 11. Reproducibility / Code

All analysis is on GitHub at `Abraham-y/224r-project`. Key entry points:

```
extension/probe/cache_hidden_states.py            -- Modal job (per checkpoint)
extension/probe/analyze_probes.py                 -- per-cell probe AUROCs
extension/probe/robustness_probes.py              -- balanced subsample + bootstrap CIs
extension/probe/deeper_analyses.py                -- per-keyword + Cohen's d + per-layer
extension/probe/qualitative_matched_pairs.py      -- §2.2 matched-pair table
extension/probe/cross_checkpoint_transfer.py      -- §2.4 transfer matrix + heatmap
extension/probe/significance_and_baselines.py     -- §7 significance + LR/RF/MLP
extension/probe/analyze_concealment_gap.py        -- §2.5 / §5
extension/probe/analyze_dynamics.py               -- §6 per-snapshot AUROC table
extension/probe/make_figures.py                   -- all 7 figures
```

Probe cache (`extension/cache/probe_cache/`) and dynamics cache (`extension/cache/probe_cache_dynamics/`) are local mirrors of the Modal volume.

---

## 12. The five claims I'd put in the paper

1. **At 0.5B with outcome-RL on an exact-verifier task, the global concealment gap inverts** (closes from +0.15 on C_SFT to +0.01 on C_outcome under a binary verbal-keyword measurement). Outcome RL produces a globally calibrated verbal policy.
2. **At the *positions* of confidence verbalization, the internal correctness representation collapses** (probe AUROC 0.74 → 0.52 at assertion tokens; matched-pair within-prompt drops from 76% to 36% above diagonal; p < 0.005 in two independent statistical frameworks).
3. **Within-problem effect size drops 70%** (Cohen's d +1.26 → +0.38; Mann-Whitney p = 0.004), directly comparable to Yuan-et-al's headline metric at 1.5B+.
4. **Cross-checkpoint probe transfer fails in both directions** (off-diagonal AUROC 0.52–0.62 on both position kinds), indicating representation drift rather than signal suppression within a shared subspace.
5. **The trace-final representation is preserved across all 100 RLOO steps** (~0.80 throughout). The position-resolved collapse therefore reflects a *behavioral shift in where the model places confidence tokens*, not destruction of the underlying representation. The mechanism is **decoupling, not damage**.

**Headline framing.** The model has learned a globally calibrated *policy* for when to verbalize confidence, while the internal mechanism producing those verbalizations has decoupled from its own correctness representation. Behavioral calibration via mechanism decoupling — a more sophisticated form of concealment than the naive "the model says confident things while internally knowing they're wrong."
