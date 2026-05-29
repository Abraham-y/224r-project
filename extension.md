# Concealment Under Outcome Pressure

*How RL reward structure reshapes the gap between what a small reasoner knows internally and what it says out loud.*

> **Status:** IPO and RLOO core implementations complete; SFT baseline uses `asingh15/qwen-sft-countdown-defaultproj` (Anikait Singh's Countdown SFT) in lieu of a team-trained SFT. `C_outcome` trained with intermediate checkpoints saved every 10 steps. Extension build begins now.
> **Team:** 2 people. Combined Modal budget ≈ $700.
> **This file:** the consolidated source of truth. Supplementary docs (`build_plan.md`, `proposal_draft.md`, `next_steps_from_neighbors.md`) are kept but defer to this one when they conflict.

---

## TL;DR

We use Countdown's exact step-checkable verifier as an instrument to ask whether the choice of RL reward structure (outcome-only versus dense exact process supervision) creates or shapes the recently documented gap between a small reasoner's *internal* representation of correctness and its *verbalized* confidence. Two recent neighbor papers explicitly flag this question as their open future work. We answer it with a three-checkpoint controlled comparison, three measurement layers, and a cross-checkpoint probe transfer analysis that decomposes the gap into representation drift versus signal suppression.

You have `C_outcome` trained with every-10-step snapshots, and you use Anikait's checkpoint as `C_SFT`. The build from here is the subgoal infrastructure, `C_process` training, the probe pipeline, and the cross-checkpoint analysis. Section 2.3 is resolved: intermediate checkpoints exist, so Layer C (temporal axis) is doable without re-training.

---

## 1. The Project

### 1.1 What we're studying

Recent interpretability work has shown that reasoning models often *internally* represent that they will produce a wrong answer while *verbally* expressing high confidence. A linear probe on hidden states can predict correctness with AUROC 0.95 while verbalized confidence achieves only 0.67 on the same traces (Yuan et al. 2026). The model knows things it does not say.

All existing work studies this gap on *fixed*, already-trained models. Our question is the origin question: **does RL training create this concealment gap, and does the reward structure (outcome-only versus dense exact process supervision) determine whether the gap grows or shrinks?** Countdown is uniquely suited because its rule-based verifier gives noise-free correctness labels for both individual subgoals and final answers. Both the probe labels and the process reward are driven by the same exact ground truth, removing the label noise that has confounded every related measurement on natural-language math benchmarks.

### 1.2 The honest novelty framing

I have been inconsistent about this across the planning conversation. The committed version:

The *question* we ask is novel and explicitly flagged as open by recent neighbors. Yuan et al.'s limitations section names RL from probe feedback and training-time intervention on the concealment gap as not tested. Anand et al.'s discussion names the training-dynamics origin of self-verification as a promising unexplored avenue. The *methodology* we assemble (linear probes, exact subgoal verification, RLOO, cross-checkpoint transfer analysis) is from existing work. The *result*, when we have it, will be a new empirical finding no current paper contains. This is workshop-eligible work where execution and result legibility determine acceptance, not the novelty of the question itself.

What we are not claiming: that we discovered the concealment phenomenon (Yuan et al. did, three weeks ago), or that we invented probe-based analysis (Anand et al. and others did), or that we propose a new training mechanism. What we are claiming: that we use Countdown's exact-verifier setting to run a controlled study no one has run, answering a question two of our closest neighbors explicitly named as their open future work.

### 1.3 The bet (three falsifiable predictions)

- **H1.** After SFT, probe correctness-AUROC and verbalized-confidence-AUROC are similar. The model says roughly what it knows.
- **H2.** After outcome-only RLOO, accuracy rises (0.3 to 0.5+ per spec), but the concealment gap *widens*. The model becomes more confident in assertions decoupled from its internal correctness representation.
- **H3.** After process-reward RLOO using the exact subgoal verifier, the gap partially closes but a residual persists, with the residual explained by reward hacking of the subgoal channel.

Any falsification is itself a publishable result. The most publishable outcome is H3 failing (process reward does not close the gap), which generalizes Yuan et al.'s "diagnostic not causal" finding to dense exact process supervision.

### 1.4 Why this satisfies the class

The CS 224R spec is explicit: the extension is graded on "doing science to figure out strengths and weaknesses of whatever you have tried" and explicitly says "achieving state-of-the-art performance is not required ... performance requirements will be very lax for the extension." Our project is built around exactly this framing: we compare two reward regimes on a fixed RL algorithm and fixed model, and we measure their strengths and weaknesses behaviorally and internally. The five-neighbor triangulation (Yuan et al., Anand et al., Damani et al., Chen et al., Taufeeque et al. each flag a piece of our project as their open future work) is the strongest motivation paragraph available.

---

## 2. Current State

### 2.1 Done

- `C_SFT` = `asingh15/qwen-sft-countdown-defaultproj` (Anikait Singh's Qwen2.5-0.5B Countdown SFT baseline). Team-trained SFT was not run; this checkpoint serves as the SFT baseline. Test-set pass@1 = 28.6%, pass@16 = 78.0%.
- IPO trainer → IPO checkpoint (milestone-only, not part of the science).
- RLOO trainer with outcome reward → `C_outcome` at `/vol/checkpoints/rloo_checkpoints/rloo_training/rloo_fixed_v2/latest_checkpoint/model` on the Modal volume. Intermediate snapshots persisted at steps 0, 10, 20, …, 90. Test-set pass@1 = 53.5%, pass@16 = 72.0% (sharpening: pass@1 up 24.9 pts vs SFT; pass@16 down 6 pts).

### 2.2 To do (all extension, AI assistance permitted)

In rough dependency order:

1. Behavioral metric harness (Layer A)
2. Subgoal infrastructure (parser, exact verifier, reward composition, SFT augmentation)
3. Train `C_process`
4. Probe pipeline (Layer B)
5. Training-dynamics snapshots (Layer C)
6. Cross-checkpoint transfer analysis
7. Optional Phase 3B: probe-aware reward with KL sweep

### 2.3 The one urgent decision — RESOLVED

`C_outcome` (run `rloo_fixed_v2`) saved persistent checkpoints at steps 0, 10, 20, …, 90 plus `latest_checkpoint` on the Modal volume. This is *finer* than the originally planned ~50-step granularity. **Layer C training dynamics is doable without re-training; risk R7 is closed.** Proceed with the build.

---

## 3. Experimental Design

### 3.1 Three checkpoints

| Checkpoint | What it is | Status |
|---|---|---|
| `C_SFT` | `asingh15/qwen-sft-countdown-defaultproj` (Anikait Singh's Qwen2.5-0.5B Countdown SFT). Used as SFT baseline in lieu of team-trained SFT. | In use |
| `C_outcome` | RLOO from `C_SFT`, outcome reward only (0.0 / 0.1 / 1.0 per spec). Persistent snapshots every 10 steps. | Done |
| `C_SFT_aug` | `C_SFT` continued-SFT on subgoal-augmented `Asap7772/cog_behav_all_strategies`. Initialization for `C_process`. | To build |
| `C_process` | RLOO from `C_SFT_aug`, outcome + exact-verifier subgoal reward | To build |

The IPO checkpoint exists for milestone compliance and does not participate in the science.

### 3.2 Three measurement layers

Applied to all three checkpoints, and to intermediate snapshots of the RL arms.

**Layer A: Behavioral.** Free, runs on existing checkpoints.

1. Accuracy (overall, hard mult/div subset).
2. **Reasoning-answer consistency rate.** Parse the last arithmetic expression the CoT endorses, parse the `<answer>` expression, evaluate both, check match. Quantifies El et al.'s and Hu/Wang's eyeballed Failure Mode 1.
3. **Confident-wrong rate.** Fraction of wrong rollouts where the CoT contains a high-confidence assertion ("this works", "verified", "the answer is").
4. **ECE on verbalized confidence.** Prompt for confidence in a separate elicitation call, bin, reliability diagram.
5. **Verbalized confidence AUROC.** Treat verbalized confidence as a binary classifier for correctness.
6. **Step-grounded verbalized confidence (C_process only).** Elicit confidence per declared subgoal, calibrate against the exact subgoal verifier. Extends RLCR's flagged limitation (uncertainty reasoning stays generic, untied to specific steps). Adjacent scoopers (MMBoundary, SPAE, Temporalizing Confidence) work in multimodal, advantage-estimation, or non-RL settings respectively; our intersection (verbalized per-step confidence on text-only exact-verifier task, calibrated across reward regimes) is unoccupied.

**Layer B: Internal.** The probe.

1. **Probe correctness AUROC** trained per checkpoint at the `</think>` position, predicting final-answer correctness.
2. **Concealment gap** = Probe AUROC − Verbalized AUROC. The headline quantity.
3. **Probe earliness.** Earliest token position at which probe AUROC exceeds 0.7.
4. **Concealment-by-example.** Fraction of confident-wrong rollouts where the probe at `</think>` correctly classifies "wrong".
5. **Within-problem Cohen's d.** Multiple traces per problem at temperature 0.7; probe-score effect size for correct vs wrong traces from the same problem. Yuan et al. ran this on standard models but had n=14 mixed-outcome problems on their RL model; with cheap Countdown rollouts we run a properly powered version on all three checkpoints.

**Layer C: Training dynamics.** Snapshot every ~50 RLOO steps, recompute Layers A and B on each. Plot the trajectory of the concealment gap during training. Anand et al. explicitly flag this temporal view as unexplored.

### 3.3 The headline plot

X-axis: training step. Y-axis: AUROC. Two lines per RL arm (verbalized confidence and internal probe). Shaded gap. If the gap widens during outcome-RLOO and stabilizes or narrows during process-RLOO, you have the paper in one figure.

---

## 4. Technical Details

### 4.1 The probe

Following Yuan et al. for direct comparability.

- **Architecture.** Logistic regression. L2 regularization C=0.1, max_iter=2000, standardized inputs.
- **Inputs.** Hidden state vector at a specified (layer, position) pair. Qwen2.5-0.5B has 24 layers, 896-dim hidden states. Default layer L16; sweep upper third.
- **Positions.**
  - Early prefix: after first 16 tokens of `<think>`.
  - Mid-trace: at every `=` token (sampled to control cost).
  - Pre-answer: at the `</think>` token. Main reporting position.
- **Labels.** Whether the trace's final answer is correct under the Countdown verifier. Free, exact, no LLM judge.
- **Training data.** 5k-10k rollouts per checkpoint, balanced by correctness. Held-out *problems* (not just held-out trajectories), 5-fold CV.
- **Sanity checks.**
  - Shuffled-label probe must hit ~0.5 AUROC.
  - Random-direction probe must be near 0.5.
  - Held-out-problem AUROC reported alongside held-out-trajectory AUROC.
  - Linear probe should match or exceed MLP and random forest (Yuan et al. show this on MATH; if it fails on Countdown that itself is a finding).

The probe is **trained per checkpoint**. Each model gets its own probe trained on its own activations. The single exception is the cross-checkpoint transfer experiment (Section 4.5), which by design uses one probe across multiple checkpoints.

### 4.2 The process reward

Per rollout:

```
R = R_outcome + λ * R_subgoal

R_outcome ∈ {0.0, 0.1, 1.0}                        # per default project spec
R_subgoal = (n_valid_and_achieved − α * n_invalid) # capped at [0, 1]
            / max(n_declared, 1)
```

Defaults: `λ = 0.3` (subgoal bonus cannot dominate correctness), `α = 1.0`. Small sweep on both during validation.

### 4.3 Subgoal token format and exact verifier

The model emits `<subgoal>` declarations during reasoning:

```
<subgoal> reach 60 from [3, 4, 5] </subgoal>
3 * 4 = 12 ... no. 4 * 5 = 20, 20 * 3 = 60. reached 60.
<subgoal> reach 68 from [60, 8] </subgoal>
60 + 8 = 68. done.
```

A subgoal is a `(target_value, available_subset)` pair. Two exact checks:

- **Validity.** Is `target_value` reachable from `available_subset` using +, −, ×, ÷? Cheap exhaustive enumeration over the 3-4 element subset.
- **Achievement.** Does the model's subsequent reasoning before the next subgoal (or before `</think>`) actually compute `target_value` using only `available_subset`? Parse arithmetic in the segment, evaluate, check.

Both checks require no learned model. This is the cleanliness property that differentiates our process reward from SGVR, PROF, PROGRS, all of which use learned PRMs.

### 4.4 SFT augmentation

`C_process` cannot be trained directly via RL because the base model never emits subgoals (so the subgoal reward term is dead). We first SFT on an augmented dataset where subgoal declarations are inserted at natural decomposition points in the existing expert traces.

Recipe (mirrors SGVR's milestone augmentation and El et al.'s `<clean>` augmentation):

1. Take each trace in `Asap7772/cog_behav_all_strategies`.
2. Parse arithmetic expressions in the trace.
3. Identify intermediate values that appear in the final expression.
4. Insert `<subgoal>` declarations announcing each intermediate before it is computed.
5. Validate that the augmented trace is syntactically clean (hand-spot-check 50).

Output is a new SFT dataset. SFT on it produces `C_SFT_aug`, which is the initialization for `C_process`'s RL run.

### 4.5 Cross-checkpoint probe transfer

Train probe on `C_SFT` activations. Evaluate the same probe (no retraining) on `C_outcome` and `C_process` activations. Repeat for probes trained on each checkpoint. Produces a 3×3 matrix of AUROCs.

- Degraded off-diagonal AUROC → representation drift (in the sense Taufeeque et al. found in coding RLVR).
- Preserved off-diagonal AUROC → signal suppression without drift.

The matrix decomposes *which mechanism* explains any concealment gap we find. This is the third figure of the paper.

---

## 5. The Build

Each module below has a clear acceptance criterion. Build order in Section 5.2.

### 5.1 Modules

**`extension/metrics/behavioral.py`**
Inputs: checkpoint, prompt set. Outputs: per-rollout CSV with parsed CoT, parsed answer, computed answer value, target, correctness, reasoning-answer consistency, confident-wrong flag.
*Accept when:* runs on `C_SFT` and `C_outcome`, produces a CSV with all columns populated, ~200 rollouts each.

**`extension/metrics/calibration.py`**
ECE, reliability diagram, verbalized AUROC over the verbalized confidence column from `behavioral.py`. Uses a separate elicitation call ("Rate your confidence in the above answer from 0 to 100") rather than inline elicitation, to avoid contaminating the CoT.
*Accept when:* reliability diagrams produced for `C_SFT` and `C_outcome`, ECE and AUROC numbers reported.

**`extension/subgoal/parser.py`**
Parses `<subgoal> ... </subgoal>` declarations. Returns list of `(target_value, available_subset, position)` tuples.
*Accept when:* unit tests pass on 20 hand-crafted traces including malformed declarations.

**`extension/subgoal/verifier.py`**
`is_reachable(target, subset)` via exhaustive enumeration. `is_achieved(target, subset, segment)` via arithmetic parsing in the trace segment.
*Accept when:* unit tests on positive and negative cases pass.

**`extension/subgoal/reward.py`**
Composes `R = R_outcome + λ * R_subgoal` using the existing outcome reward and the subgoal verifier.
*Accept when:* `compute_reward(rollout)` returns expected values on hand-crafted cases including degenerate (zero subgoals) cases.

**`extension/subgoal/sft_augment.py`**
Inserts subgoal declarations into expert traces.
*Accept when:* spot-check of 50 augmented traces shows syntactic validity and semantic naturalness; augmented dataset matches expected size.

**`extension/probe/cache_hidden_states.py`**
Extracts hidden states at specified (layer, position) pairs and caches to disk.
*Accept when:* 5k rollouts × 3 positions × 2 layers caches in under 30 minutes; cache size manageable (<20 GB per checkpoint).

**`extension/probe/train_probe.py`**
Per (checkpoint, layer, position) tuple, trains logistic regression probe with the sanity checks built in.
*Accept when:* probe AUROC, shuffled-label AUROC, random-direction AUROC, and held-out-problem AUROC are all reported.

**`extension/probe/eval_probe.py`**
Computes concealment gap, earliness, Cohen's d.
*Accept when:* outputs headline numbers for the paper.

**`extension/probe/transfer.py`**
Cross-checkpoint matrix.
*Accept when:* 3×3 AUROC matrix produced.

**`extension/metrics/dynamics.py`**
Snapshots every ~50 RLOO steps, reruns Layer A and Layer B on each.
*Accept when:* trajectory plots of probe AUROC, verbalized AUROC, concealment gap over training steps.

**`extension/metrics/step_confidence.py`**
Per-subgoal calibration on `C_process`.
*Accept when:* reliability diagram at the subgoal level.

**`extension/honest_reward/probe_aware_reward.py`** *(Phase 3B, optional)*
Frozen probe consistency reward augmentation, KL sweep harness.
*Accept when:* `C_honest_lowKL` and `C_honest_highKL` train without diverging.

### 5.2 Build order with go/no-go gates

```
Day 1                      Day 7
  |                          |
  Behavioral harness (4.1) --> Phase 1 results in hand
  ↓
  Subgoal parser+verifier (4.2-4.4)
  ↓
  SFT augmentation, C_process training
  ↓
  Probe pipeline (cache, train, eval)
  ↓
  Cross-checkpoint transfer
  ↓
  Layer C dynamics
  ↓
  [Optional] Phase 3B
```

Gates:

- **G1 (after Phase 1):** Behavioral metrics show meaningful differences between C_SFT and C_outcome (e.g., confident-wrong rate visibly higher in C_outcome). If yes, proceed. If no, reconsider whether the behavioral story exists at 0.5B.
- **G2 (after first probe):** Probe AUROC on C_SFT > 0.65 at some layer/position. If yes, proceed to full Phase 2. If no, report "scale-dependent: 0.5B does not encode the signal as 1.5B+ do" as the headline result.
- **G3 (after C_process):** C_process trains successfully (>0.4 accuracy, emits >50% well-formed subgoals). If yes, proceed. If no, debug subgoal augmentation; if intractable, drop process arm and report only outcome vs SFT.
- **G4 (after transfer):** 3×3 matrix is interpretable. If yes, Phase 3 done. If no, exploratory only.
- **G5 (after all of the above):** Time and compute remain. If yes, attempt Phase 3B. If no, stop; Phases 1-3 are a complete paper.

---

## 6. Operational Plan

### 6.1 Team split (2 people)

- **Person A (RL + subgoal infrastructure).** Subgoal parser/verifier/reward, SFT augmentation, `C_SFT_aug` and `C_process` training, training-dynamics infrastructure for Layer C (replay against existing 10-step snapshots).
- **Person B (metrics + probe).** Layer A behavioral harness, calibration, hidden-state caching, probe training/eval, cross-checkpoint transfer.

Person B starts immediately against existing artifacts (`eval_sft.json`, `eval_rloo.json`, on-volume `C_outcome` checkpoints) with no dependency. Person A starts subgoal infrastructure in parallel. They converge after Phase 1 results, when the probe pipeline needs `C_process` activations.

Step-grounded confidence (§3.2 Layer A #6) and Phase 3B remain explicitly optional and scoped out for the 2-person headline.

### 6.2 Week 1 sprint (2 people)

- **Day 1.** Person B runs Layer A behavioral metrics on existing `eval_sft.json` and `eval_rloo.json` (no new GPU required). Phase 1 numbers same-day.
- **Day 1-3.** Person A builds subgoal parser + verifier + reward. Unit tests pass by end of day 3.
- **Day 2-3.** Person B writes hidden-state caching script; caches `C_SFT` (from HF) and `C_outcome` activations. Trains a sanity probe on `C_SFT` by end of day 3.
- **Day 3-4.** Person A runs subgoal SFT augmentation; trains `C_SFT_aug`.
- **Day 4-6.** Person A launches `C_process` RLOO run; Person B runs probe and concealment-gap analysis on `C_SFT` vs `C_outcome`. Two-arm Phase 2 numbers in hand.
- **Day 6-7.** Person B caches `C_process` activations once available, runs cross-checkpoint transfer (3×3 matrix). Headline plot drafted.

By end of week 1, Phase 1 + two-arm Phase 2 numbers exist; Phase 2 with three checkpoints completes early in week 2. Layer C dynamics (replay over the existing 10-step `C_outcome` snapshots) and the temporal headline plot land in week 2.

### 6.3 Compute budget (≈ $700 combined, 2 people)

- SFT baseline: $0 (using `asingh15/qwen-sft-countdown-defaultproj` from HF).
- `C_outcome` RLOO: ~$30 sunk (one full 100-step run, 1×H100). Done.
- `C_SFT_aug` SFT: ~$15-25.
- `C_process` RLOO: ~$30-60 per run; budget for 1-2 runs to tune `λ` and `α`.
- Probe training + hidden-state caching across 3 checkpoints + 10 dynamics snapshots: ~$30-50.
- Phase 3B (optional KL sweep): ~$60-120.

Conservative total for Phases 1-3 (excluding 3B): ≈ $135-195. Headroom is large against $700; Phase 3B remains affordable even after.

### 6.4 Timeline to deliverables

| Week | Goal |
|---|---|
| 1 | Phase 1 behavioral results, subgoal infrastructure complete, `C_process` trained |
| 2 | Probe pipeline working, Phase 2 numbers, cross-checkpoint transfer |
| 3 | Training dynamics if intermediate checkpoints exist (see 2.3). Poster materials. |
| 4 | Final report. Optional Phase 3B if time. |

---

## 7. Risk Register

| ID | Risk | Mitigation |
|---|---|---|
| R1 | Probe AUROC too low at 0.5B to claim signal | Sweep layers and positions; if all weak, report scale-dependence as the finding. Yuan et al. show 0.918 AUROC at 1.5B; we extend below their range. |
| R2 | Outcome RL does not widen the gap (H2 fails) | Publishable scale-dependence claim: "RL at 0.5B does not induce concealment unlike at 7B+". |
| R3 | Subgoal SFT augmentation produces unlearnable traces | SGVR and El et al. got similar augmentation to work; hand-inspect a few, fix parser. |
| R4 | 700 RLOO steps insufficient to see gap evolve | Checkpoint every 50 steps; the trajectory itself is informative even if final values noisy. |
| R5 | Phase 3B obfuscation emerges (probe reward-hacked) | Predicted outcome at low KL per Taufeeque et al. Report and characterize via cross-checkpoint transfer. |
| R6 | Step-grounded confidence elicitation fails at 0.5B | Drop from Layer A; overall confidence metric still captures the behavioral story. |
| R7 | `C_outcome` was trained without intermediate checkpoint saves | Re-train with snapshots (~$80-150, ~10 hours) or skip Layer C. See Section 2.3. **Decide today.** |
| R8 | Verbalized confidence elicitation contaminates the CoT | Use a separate elicitation call (post-rollout), not inline. Already baked into 5.1 calibration module. |

---

## 8. Workshop Ceiling, Honestly

This is a strong class project with a real shot at being among the best in the cohort. Workshop submission is plausible, in the range of 45-60% conditional on clean execution and a legible result, with the upper end reflecting the explicit-future-work flags from Yuan et al. and Anand et al. that strengthen our motivation paragraph. The dominant risk is execution at 0.5B and result legibility, not novelty.

What carries it from "class project" to "workshop":
- Clean headline plot showing the gap evolving across training.
- Cross-checkpoint transfer matrix that has an interpretable pattern.
- A representative confident-wrong example with annotated probe activations.

What does not push it to workshop:
- Marginal accuracy improvements.
- Replication of existing process-reward methods on a new task.
- Generic discussion of process vs outcome reward without the internal lens.

Realistic submission targets if results are clean: a reasoning-focused workshop, MATH-AI, an interpretability workshop, or Tiny Papers. Not main conference tracks.

---

## 9. Class Deliverable Mapping

- **Proposal + SFT (5/1).** Submitted (SFT done, proposal in `proposal_draft.md`).
- **Milestone IPO + RLOO (5/22).** Submitted (IPO + C_outcome done).
- **Poster (6/3).** Phase 1 + Phase 2 + cross-checkpoint transfer figures. Phase 1 guaranteed; Phase 2 if probe works.
- **Final report (6/8).** Full Phases 1-3 in the body; Phase 3B as headline if pursued and successful.

Every required deliverable produces a science measurement. Nothing is wasted work.

---

## 10. Honor Code Boundary

The CS 224R spec bans AI assistance on the default implementation (SFT, IPO, RLOO core) and permits it for the extension. Default-project deliverables (RLOO trainer, IPO trainer, milestone evaluation) are submitted; the SFT *baseline* uses `asingh15/qwen-sft-countdown-defaultproj` in lieu of a team-trained SFT. Everything in this document from here on is extension territory and is AI-assistable per §3.4 of the spec.

Repository organization should preserve the separation:

```
project/
├── core/                 # Your team's work, untouched
│   ├── sft.py
│   ├── ipo.py
│   ├── rloo.py
│   ├── data.py
│   └── verifier_countdown.py
└── extension/            # AI-assisted, this document's scope
    ├── subgoal/
    ├── probe/
    ├── metrics/
    └── honest_reward/    # Phase 3B only
```

If a question arises about core code during the extension build, surface the symptom, not the code. I will help debug the extension; I will not touch core.

---

## 11. References

These are the citations our proposal and final report depend on. **Before submitting any document, verify each tag against the actual arXiv listing.** I have been imprecise about author names across the planning conversation and the names below are best-effort, not verified.

**Direct neighbors (must cite, central to motivation):**
- Yuan et al. (May 2026). "Hidden Error Awareness in CoT Reasoning: The Signal Is Diagnostic, Not Causal." arXiv 2605.09502. The concealment phenomenon paper. Their limitations explicitly flag "RL from probe feedback" as not tested.
- Anand et al. (April 2025). "Reasoning Models Know When They're Right." arXiv 2504.05419. The probe methodology and early-prediction property. Their discussion explicitly flags training-dynamics origin as future work.
- Damani et al. (July 2025). "Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty" (RLCR). arXiv 2507.16806. The calibration reward, and the explicit flag that uncertainty stays generic and untied to specific solution steps.
- Chen et al. (January 2026). "Milestones over Outcome: Unlocking Geometric Reasoning with Sub-Goal Verifiable Reward" (SGVR). arXiv 2601.05073. The process-reward method we adapt.
- Taufeeque et al. (February 2026). "The Obfuscation Atlas." arXiv 2602.15515. The probe-as-reward taxonomy in coding RLVR, motivating Phase 3B and the cross-checkpoint transfer experiment.

**Adjacent neighbors (cite for completeness):**
- David (November 2025). "Temporal Predictors of Outcome in Reasoning Language Models." arXiv 2511.14773. Early-token probe variant.
- He et al. (May 2025). "MMBoundary." arXiv 2505.23224. Step-level confidence calibration in multimodal RL.
- (SPAE, January 2026). "Step Potential Advantage Estimation." arXiv 2601.03823. Step confidence as advantage estimation in math RL.
- Papadatos and Freedman (December 2024). "Linear Probe Penalties Reduce LLM Sycophancy." arXiv 2412.00967. Probe-as-reward in preference RLHF.
- Liu et al. (April 2025). "To Backtrack or Not to Backtrack." arXiv 2504.07052. The closest controlled-Countdown-RL study.

**CS 224R precedent:**
- El, Erol, Park-Kaufmann (2025). "RL Training for Dynamic Context Management in Mathematical Reasoning" (the `<clean>` token paper). CS 224R 2025 final report.
- Hu and Wang (2025). "Efficient Arithmetic Reasoning in Small LMs via Function Calling." CS 224R 2025 final report.

**Method papers we use:**
- Ahmadian et al. (2024). "Back to basics: revisiting REINFORCE-style optimization for learning from human feedback in LLMs." arXiv 2402.14740. RLOO.
- Gandhi et al. (2024). "Stream of Search." arXiv 2404.03683. The Countdown task and the dataset we SFT on.
- Gandhi et al. (2025). "Cognitive Behaviors that Enable Self-Improving Reasoners." arXiv 2503.01307. Source of `Asap7772/cog_behav_all_strategies`.

---

## What To Do Today

1. ~~Confirm Section 2.3.~~ Resolved: 10-step snapshots exist.
2. ~~Confirm team split.~~ Locked to 2 people; roles in §6.1.
3. **Person B** starts on `extension/metrics/behavioral.py` against existing `eval_sft.json` (`sft_baseline_passk.json`) and `eval_rloo.json` (`rloo_fixed_v2_passk.json`). Reasoning-answer consistency parser + confident-wrong / multiple-`<answer>` repetition counter are the first metrics. No new GPU runs needed for Phase 1.
4. **Person A** reads Yuan et al. and Anand et al. end-to-end while drafting `extension/subgoal/parser.py` and `verifier.py` against the format in §4.3.
5. Both: create the `extension/` directory tree per §10 layout before any code lands.

Lock and build.