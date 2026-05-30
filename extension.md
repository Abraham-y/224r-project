# Concealment Under Outcome Pressure

*How RL reward structure reshapes the gap between what a small reasoner knows internally and what it says out loud.*

> **Status:** Scope locked to probe-as-measurement layer on `C_SFT` and `C_outcome` (both already trained). `C_process` was attempted and is retained as a documented negative result; no further training planned. SFT baseline uses `asingh15/qwen-sft-countdown-defaultproj` (Anikait Singh's Countdown SFT) in lieu of a team-trained SFT. `C_outcome` trained with intermediate checkpoints saved every 10 steps.
> **Team:** 2 people. Combined Modal budget ≈ $700.
> **This file:** the consolidated source of truth. Supplementary docs (`build_plan.md`, `proposal_draft.md`, `next_steps_from_neighbors.md`) are kept but defer to this one when they conflict.

---

## TL;DR

We use Countdown's exact-verifier setting to measure the gap between what a small reasoner *internally* represents about correctness and what it *verbalizes*, on two RL checkpoints we already have (`C_SFT`, `C_outcome`). The headline measurement is a linear-probe concealment gap at the trace-final position plus a per-position probe analysis at confidence-asserting token sites ("Perfect!", "this works", "got it"). Cross-checkpoint probe transfer (2×2) decomposes the gap into representation drift versus signal suppression. A failed annotation-only process-reward arm (`C_process`) is documented as a negative result whose mechanism — surface declarations causally disconnected from computation — confirms a recent theoretical prediction.

No new training runs. No new tokens. No probe-aware reward, no introspection token, no functional tokens. The project is anchored entirely on the probe layer plus the behavioral evidence already collected. Section 2.3 (intermediate `C_outcome` checkpoints exist) means Layer C (temporal axis) is doable without re-training.

---

## 1. The Project

### 1.1 What we're studying

Recent interpretability work has shown that reasoning models often *internally* represent that they will produce a wrong answer while *verbally* expressing high confidence. A linear probe on hidden states can predict correctness with AUROC 0.95 while verbalized confidence achieves only 0.67 on the same traces (Yuan et al. 2026). The model knows things it does not say.

All existing work studies this gap on *fixed*, already-trained models. Our scoped question is the **outcome-RL origin question**: does outcome-only RLOO training create or amplify the concealment gap relative to the SFT initialization? We study this on `C_SFT` and `C_outcome` — both already trained — at the exact-verifier Countdown setting where correctness labels are noise-free at every position in the trace. The exact verifier gives us per-rollout ground truth at zero label noise, so the *measurement* of the gap is uncontaminated even though the *intervention* (outcome reward) is the standard one.

We additionally ask **where in the trace** the gap concentrates: at the trace-final position (the standard report), or at specific epistemic-confidence assertions in the chain of thought ("Perfect!", "this works", "got it") — positions where the surface expresses high confidence and where, if the gap is real, the divergence between internal probe and verbal assertion should be sharpest.

### 1.2 The honest novelty framing

I have been inconsistent about this across the planning conversation. The committed version:

The *question* we ask is novel and explicitly flagged as open by recent neighbors. Yuan et al.'s limitations section names RL from probe feedback and training-time intervention on the concealment gap as not tested. Anand et al.'s discussion names the training-dynamics origin of self-verification as a promising unexplored avenue. The *methodology* we assemble (linear probes, exact verification at the outcome level, RLOO, cross-checkpoint transfer analysis, per-position probe analysis at confidence-asserting tokens) is from existing work. The *result*, when we have it, will be a new empirical finding no current paper contains. This is workshop-eligible work where execution and result legibility determine acceptance, not the novelty of the question itself.

What we are not claiming: that we discovered the concealment phenomenon (Yuan et al. did), that we invented probe-based analysis (Anand et al. and others did), or that we propose a new training mechanism. What we are claiming: that we use Countdown's exact-verifier setting to measure the concealment gap on outcome-RLOO at small scale, with a position-resolved probe analysis at confidence-asserting tokens, and that we report a failed annotation-only process-reward arm as a complementary negative result.

**Why no new tokens, no new arms.** Three rounds of recursive literature search converged on the same pattern. Functional tokens (`<clean>`, `<verify>`, `<exit>`, `<commit>`, `<introspect>`) were a novel area in 2024 and are now a saturated subfield in 2026. Probe-as-reward is partially scooped by Papadatos & Freedman ("Linear Probe Penalties Reduce LLM Sycophancy") and Taufeeque et al. ("The Obfuscation Atlas"). Step-grounded confidence as a novel mechanism is scooped by MMBoundary, SPAE, and Temporalizing Confidence. Trajectory drift as a headline is heavily occupied by Huang et al. (ICLR 2024, "LLMs Cannot Self-Correct Reasoning Yet"), SCoRe, and three 2026 papers. The contribution this project makes lives at the *intersection* of (exact verifier × 0.5B scale × outcome-RL intervention × probe-based internal measurement × training dynamics × position-resolved probing at confidence-asserting tokens), and the probe-as-measurement layer is rich enough on its own. Adding more arms doesn't add proportional novelty and runs into engineering risk on a tight timeline. The negative result on `C_process` is itself a contribution: it confirms Strategic Information Allocation's prediction that annotation-only tokens cannot route capability into a small reasoner.

### 1.3 The bet (three falsifiable predictions)

- **H1.** On `C_SFT`, probe-correctness AUROC at the trace-final position and verbalized-confidence AUROC are similar. The SFT model says roughly what it internally represents.
- **H2.** On `C_outcome`, accuracy rises (28.6% → 53.5% measured), but the trace-final concealment gap (probe AUROC − verbalized AUROC) *widens*. The outcome-RL'd model becomes more confident in assertions decoupled from its internal correctness representation.
- **H3.** The concealment gap concentrates at epistemic-confidence token positions (locations of "Perfect!", "this works", "got it", "the answer is", "verified") more than at neutral positions, measurable as `probe_AUROC(at confidence-asserting token) > probe_AUROC(at neutral token)` and as a larger gap-magnitude at those positions vs. the trace-final default.

Any falsification is itself a publishable result. The most publishable single outcome is H3 holding: it would be the first position-resolved evidence that the concealment gap is *spatially localized* at verbalized confidence sites, not uniformly distributed across the trace.

### 1.4 Why this satisfies the class

The CS 224R spec is explicit: the extension is graded on "doing science to figure out strengths and weaknesses of whatever you have tried" and explicitly says "achieving state-of-the-art performance is not required ... performance requirements will be very lax for the extension." Our project is built around exactly this framing: we compare two reward regimes on a fixed RL algorithm and fixed model, and we measure their strengths and weaknesses behaviorally and internally. The five-neighbor triangulation (Yuan et al., Anand et al., Damani et al., Chen et al., Taufeeque et al. each flag a piece of our project as their open future work) is the strongest motivation paragraph available.

---

## 2. Current State

### 2.1 Done

- `C_SFT` = `asingh15/qwen-sft-countdown-defaultproj` (Anikait Singh's Qwen2.5-0.5B Countdown SFT baseline). Team-trained SFT was not run; this checkpoint serves as the SFT baseline. Test-set pass@1 = 28.6%, pass@16 = 78.0%.
- IPO trainer → IPO checkpoint (milestone-only, not part of the science).
- RLOO trainer with outcome reward → `C_outcome` at `/vol/checkpoints/rloo_checkpoints/rloo_training/rloo_fixed_v2/latest_checkpoint/model` on the Modal volume. Intermediate snapshots persisted at steps 0, 10, 20, …, 90. Test-set pass@1 = 53.5%, pass@16 = 72.0% (sharpening: pass@1 up 24.9 pts vs SFT; pass@16 down 6 pts).
- `C_process` was trained (RLOO from `C_SFT_aug` with composite outcome + annotation-only subgoal reward) and is retained on disk as data for the failed-intervention analysis. It **hurt performance relative to `C_outcome`** and produced rollouts where `<subgoal>` declarations were causally disconnected from the computation that followed them. See the Appendix for details. **No further training on this arm is planned.**

### 2.2 To do (all extension, AI assistance permitted)

In rough dependency order — anchored entirely on the probe layer plus already-collected behavioral evidence. No new training:

1. Behavioral metric harness (Layer A) on `C_SFT` and `C_outcome`.
2. Probe pipeline (Layer B): hidden-state caching, probe training, evaluation.
3. Per-position probe analysis at confidence-asserting token sites (new — supports H3).
4. Cross-checkpoint probe transfer (2×2: `C_SFT` ↔ `C_outcome`).
5. Training-dynamics snapshots (Layer C) over the saved `C_outcome` 10-step checkpoints.
6. Retrospective analysis of `C_process` rollouts for the failed-intervention appendix (no training, just measurement against existing eval JSON).

The subgoal infrastructure (`extension/subgoal/*`) and the process-reward launcher (`extension/training/process_rloo.py`) remain in the repository for retrospective traceability but are no longer development targets.

### 2.3 The one urgent decision — RESOLVED

`C_outcome` (run `rloo_fixed_v2`) saved persistent checkpoints at steps 0, 10, 20, …, 90 plus `latest_checkpoint` on the Modal volume. This is *finer* than the originally planned ~50-step granularity. **Layer C training dynamics is doable on `C_outcome` without re-training; risk R7 is closed.** Proceed with the build.

---

## 3. Experimental Design

### 3.1 Checkpoints

| Checkpoint | What it is | Role |
|---|---|---|
| `C_SFT` | `asingh15/qwen-sft-countdown-defaultproj` (Anikait Singh's Qwen2.5-0.5B Countdown SFT). Used as SFT baseline. | **Primary** — anchor for H1, baseline arm of probe layer. |
| `C_outcome` | RLOO from `C_SFT`, outcome reward only (0.0 / 0.1 / 1.0 per spec). Persistent snapshots every 10 steps. | **Primary** — anchor for H2 and H3, RL-trained arm of probe layer. |
| `C_process` | RLOO with composite outcome + annotation-only subgoal reward, initialized from `C_SFT_aug`. | **Secondary, failed-intervention analysis only.** Documented in the Appendix; not part of the headline. |

The IPO checkpoint exists for milestone compliance and does not participate in the science.

### 3.2 Three measurement layers

Applied to `C_SFT` and `C_outcome`, and to the saved 10-step `C_outcome` snapshots for Layer C.

**Layer A: Behavioral.** Free, runs on existing rollouts.

1. Accuracy (overall, hard mult/div subset).
2. **Reasoning-answer consistency rate.** Parse the last arithmetic expression the CoT endorses, parse the `<answer>` expression, evaluate both, check match. Quantifies El et al.'s and Hu/Wang's eyeballed Failure Mode 1.
3. **Confident-wrong rate.** Fraction of wrong rollouts where the CoT contains a high-confidence assertion ("this works", "verified", "got it", "perfect", "the answer is").
4. **ECE on verbalized confidence.** Prompt for confidence in a separate elicitation call, bin, reliability diagram.
5. **Verbalized confidence AUROC.** Treat verbalized confidence as a binary classifier for correctness.
6. **Step-grounded verbalized confidence (C_outcome only, as a single-arm measurement).** Elicit confidence per declared computational step in the CoT (not per `<subgoal>` tag), calibrate against the exact verifier. This is now a *single-arm characterization* of `C_outcome`, not a process-vs-outcome comparison. Adjacent work in multimodal (MMBoundary), advantage-estimation (SPAE), and non-RL settings (Temporalizing Confidence) means we cannot claim it as a *novel mechanism*; we report it only as an additional descriptive metric on the outcome arm.

**Layer B: Internal.** The probe.

1. **Probe correctness AUROC** trained per checkpoint at the `</think>` position, predicting final-answer correctness. Reported as the trace-final default.
2. **Concealment gap** = Probe AUROC − Verbalized AUROC. The headline quantity at the trace-final position.
3. **Probe earliness.** Earliest token position at which probe AUROC exceeds 0.7.
4. **Concealment-by-example.** Fraction of confident-wrong rollouts where the probe at `</think>` correctly classifies "wrong".
5. **Within-problem Cohen's d.** Multiple traces per problem at temperature 0.7; probe-score effect size for correct vs wrong traces from the same problem. Yuan et al. ran this on standard models but had n=14 mixed-outcome problems on their RL model; with cheap Countdown rollouts we run a properly powered version on `C_SFT` and `C_outcome`.
6. **Per-position probe analysis at confidence-asserting token sites (new — supports H3).** Locate every occurrence of an assertion in the CoT matching the keyword set `{"Perfect", "this works", "got it", "the answer is", "verified"}`. At each such position, extract the hidden state and apply the trained probe. Report: probe AUROC at confidence-asserting positions vs. matched neutral positions sampled from the same traces; gap-magnitude at confidence-asserting positions vs. trace-final; alignment between probe prediction at the assertion and the assertion's implied confidence ("got it" reads as confident-correct; the probe's correctness prediction at that token is the contrast).

**Layer C: Training dynamics.** Replay Layers A and B on each saved 10-step snapshot of `C_outcome`. Plot the trajectory of the trace-final concealment gap, of the gap at confidence-asserting positions, and of accuracy over training steps. Anand et al. explicitly flag this temporal view as unexplored.

### 3.3 The headline plot

X-axis: `C_outcome` training step (0, 10, …, 90, final). Y-axis: AUROC. Three lines: probe AUROC at `</think>`, verbalized-confidence AUROC, probe AUROC at confidence-asserting token positions. Shaded gap between the trace-final probe and verbalized lines. If (a) the trace-final gap widens during outcome-RLOO and (b) the confidence-asserting-position probe AUROC sits above the trace-final probe AUROC by an interpretable margin, the paper lands in one figure.

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

### 4.2 — 4.4

*Moved to the Appendix "Failed Intervention: Annotation-Only Process Reward". The original §§4.2 (process reward formula), 4.3 (subgoal token format and exact verifier), and 4.4 (SFT augmentation) are preserved there in full, alongside what we observed empirically and the mechanistic explanation.*

### 4.5 Cross-checkpoint probe transfer (2×2)

Train probe on `C_SFT` activations. Evaluate the same probe (no retraining) on `C_outcome` activations. Repeat with a probe trained on `C_outcome` evaluated on `C_SFT`. Produces a 2×2 matrix of AUROCs.

- Degraded off-diagonal AUROC → representation drift (in the sense Taufeeque et al. found in coding RLVR).
- Preserved off-diagonal AUROC → signal suppression without drift.

The matrix decomposes *which mechanism* explains the concealment gap we find. This is the third figure of the paper.

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

**`extension/probe/cache_hidden_states.py`**
Extracts hidden states at specified (layer, position) pairs and caches to disk.
*Accept when:* 5k rollouts × 3 positions × 2 layers caches in under 30 minutes; cache size manageable (<20 GB per checkpoint).

**`extension/probe/train_probe.py`**
Per (checkpoint, layer, position) tuple, trains logistic regression probe with the sanity checks built in.
*Accept when:* probe AUROC, shuffled-label AUROC, random-direction AUROC, and held-out-problem AUROC are all reported.

**`extension/probe/eval_probe.py`**
Computes concealment gap, earliness, Cohen's d at the trace-final position.
*Accept when:* outputs headline numbers for the paper.

**`extension/probe/eval_at_assertion_tokens.py`** *(new — supports H3)*
Locates every occurrence of an assertion in the CoT matching the keyword set `{"Perfect", "this works", "got it", "the answer is", "verified"}`. For each occurrence, extract the hidden state at that token position and apply the (already-trained) probe from `train_probe.py`. Compare against matched neutral-position samples drawn from the same traces.
*Accept when:* outputs (a) probe AUROC at confidence-asserting positions, (b) probe AUROC at matched neutral positions, (c) per-assertion alignment table (probe prediction vs. surface implied confidence) for `C_SFT` and `C_outcome`. Reports whether the gap concentrates at assertion positions vs. trace-final.

**`extension/probe/transfer.py`**
Cross-checkpoint matrix.
*Accept when:* 2×2 AUROC matrix produced (`C_SFT` ↔ `C_outcome`).

**`extension/metrics/dynamics.py`**
Replays Layer A and Layer B on each saved 10-step `C_outcome` snapshot.
*Accept when:* trajectory plots of probe AUROC at `</think>`, probe AUROC at confidence-asserting positions, verbalized AUROC, and concealment gap, all over training steps.

**`extension/metrics/step_confidence.py`** *(scope-reduced)*
Step-level verbalized-confidence calibration on `C_outcome`. Single-arm characterization; not a process-vs-outcome comparison.
*Accept when:* reliability diagram at the computational-step level on `C_outcome`.

*Note: the `extension/subgoal/*` modules (`parser.py`, `verifier.py`, `reward.py`, `sft_augment.py`) and `extension/training/process_rloo.py` remain in the repo for retrospective traceability of the failed-intervention appendix, but are no longer build targets in this section.*

### 5.2 Build order with go/no-go gates

```
Day 1                      Day 7
  |                          |
  Behavioral harness (Layer A) --> Phase 1 results in hand
  ↓
  Probe pipeline (cache, train, eval at </think>)
  ↓
  Per-position probe analysis at confidence-asserting tokens
  ↓
  Cross-checkpoint transfer (2x2)
  ↓
  Layer C dynamics over saved 10-step snapshots
  ↓
  Retrospective C_process analysis for the Appendix
```

Gates (G3 — the former `C_process` training gate — has been removed; subsequent gates have been renumbered):

- **G1 (after Phase 1):** Behavioral metrics show meaningful differences between `C_SFT` and `C_outcome` (e.g., confident-wrong rate visibly higher in `C_outcome`). If yes, proceed. If no, reconsider whether the behavioral story exists at 0.5B.
- **G2 (after first probe):** Probe AUROC on `C_SFT` > 0.65 at some layer/position. If yes, proceed to full Phase 2. If no, report "scale-dependent: 0.5B does not encode the signal as 1.5B+ do" as the headline result.
- **G3 (after per-position probe analysis):** Either (a) probe AUROC at confidence-asserting positions is interpretably above trace-final probe AUROC — H3 supported, headline; or (b) it is statistically indistinguishable — H3 falsified, report as null result alongside the trace-final gap, which still stands. Either way, proceed.
- **G4 (after transfer):** 2×2 matrix is interpretable. If yes, Phase 3 done. If no, exploratory only.

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
| R4 | 100 RLOO steps insufficient to see gap evolve | We already have all 10 snapshots; the trajectory itself is informative even if endpoint values noisy. |
| R7 | `C_outcome` was trained without intermediate checkpoint saves | **CLOSED.** Snapshots persisted every 10 steps. See Section 2.3. |
| R8 | Verbalized confidence elicitation contaminates the CoT | Use a separate elicitation call (post-rollout), not inline. Already baked into the calibration module. |
| R-NEW | Per-position probe analysis shows no concentration of the gap at confidence-asserting token positions (H3 falsifies) | Report the null result; the trace-final concealment-gap measurement (H1/H2) still stands. The null on H3 is itself a publishable finding: "the gap is spatially uniform, not localized at verbalized assertions." |

---

## 8. Workshop Ceiling, Honestly

Workshop submission is plausible, in the range of **45-60%** conditional on clean execution and a legible result. The upper end reflects the explicit-future-work flags from Yuan et al. and Anand et al. and the position-resolved probe analysis (H3) being a genuinely under-explored angle on the gap. The dominant risk is execution at 0.5B and result legibility, not novelty.

**The workshop pitch, after this scope decision:** *Controlled measurement of the concealment gap on an exact-verifier task at small scale, with per-position probe analysis at verbalized confidence assertions, and a complementary negative result from an annotation-only process-reward intervention that confirms a recent theoretical prediction (Strategic Information Allocation).*

What carries it from "class project" to "workshop":
- Clean headline plot showing the trace-final gap evolving across `C_outcome` training, with the confidence-asserting-position probe AUROC as a third line.
- 2×2 cross-checkpoint transfer matrix with an interpretable drift-vs-suppression pattern.
- A representative confident-wrong example with annotated probe activations at the assertion token.
- The `C_process` negative-result appendix, framed as an empirical confirmation of Strategic Information Allocation's prediction about annotation-only tokens at small scale.

What does not push it to workshop:
- Marginal accuracy improvements.
- Replication of existing process-reward methods on a new task.
- Generic discussion of process vs outcome reward without the probe-as-measurement layer.

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

**Theory the negative result confirms and the measurement approach relies on (added post-scope-decision; VERIFY arXiv IDs AND author lists before any submission):**
- Kim et al. (March 2026). "Strategic Information Allocation." arXiv 2603.15500. *Theoretical prediction that small models without native epistemic capacity cannot be RL'd into effective reasoners via annotation-only signals.* Motivates the `C_process` failed-intervention framing in the Appendix.  ⚠️ **VERIFY tag, authorship, and exact title before submission — added from secondary discussion, not personally read.**
- (Anon. or Kim et al.) (March 2026). "Epistemic Observability." arXiv 2603.20531. *Formal impossibility result for text-only RL producing honest behavior in models without an internal verification mechanism.* Motivates the probe-based (rather than reward-based) measurement approach.  ⚠️ **VERIFY tag, authorship, and exact title before submission — added from secondary discussion, not personally read.**
- Huang et al. (ICLR 2024). "Large Language Models Cannot Self-Correct Reasoning Yet." *Original characterization of the correct-to-wrong revision pattern in CoT reasoning; cited as the established behavioral phenomenon underlying our trace-level observations.*  ⚠️ **VERIFY arXiv tag and exact title before submission.**

---

## What To Do Today

1. ~~Confirm Section 2.3.~~ Resolved: 10-step snapshots exist.
2. ~~Confirm team split.~~ Locked to 2 people; roles in §6.1.
3. ~~Build subgoal infrastructure and train `C_process`.~~ Done and retired — `C_process` trained but underperformed; arm closed per Section 1.2 reasoning and documented in the Appendix.
4. **Both**: run the probe pipeline against `C_SFT` and `C_outcome` per §5.1. Person B drives `behavioral.py` → `cache_hidden_states.py` → `train_probe.py` → `eval_probe.py` → `eval_at_assertion_tokens.py` → `transfer.py` (2×2) → `dynamics.py`. Person A drives the retrospective `C_process` analysis for the Appendix using existing eval JSONs.

Lock and build.

---

## Appendix: Failed Intervention — Annotation-Only Process Reward

This appendix preserves the design of the `C_process` arm, what we observed empirically, and the mechanistic explanation. It is now framed as a documented negative result, not an in-flight experiment.

### A.1 The process reward (original §4.2)

Per rollout, the composite reward we attempted was:

```
R = R_outcome + λ * R_subgoal

R_outcome ∈ {0.0, 0.1, 1.0}                        # per default project spec
R_subgoal = (n_valid_and_achieved − α * n_invalid) # capped at [0, 1]
            / max(n_declared, 1)
```

Defaults used in the run: `λ = 0.3`, `α = 1.0`.

### A.2 Subgoal token format and exact verifier (original §4.3)

The model was trained to emit `<subgoal>` declarations during reasoning:

```
<subgoal> reach 60 from [3, 4, 5] </subgoal>
3 * 4 = 12 ... no. 4 * 5 = 20, 20 * 3 = 60. reached 60.
<subgoal> reach 68 from [60, 8] </subgoal>
60 + 8 = 68. done.
```

A subgoal is a `(target_value, available_subset)` pair. Two exact checks:

- **Validity.** Is `target_value` reachable from `available_subset` using +, −, ×, ÷? Exhaustive enumeration over the 3-4 element subset.
- **Achievement.** Does the model's subsequent reasoning before the next subgoal (or before `</think>`) actually compute `target_value` using only `available_subset`?

Both checks require no learned model — the cleanliness property meant to differentiate this process reward from SGVR, PROF, PROGRS, all of which use learned PRMs.

### A.3 SFT augmentation (original §4.4)

Because the base model never emits subgoals on its own, we first SFT on an augmented dataset:

1. Take each trace in `Asap7772/cog_behav_all_strategies`.
2. Parse arithmetic expressions in the trace.
3. Identify intermediate values that appear in the final expression.
4. Insert `<subgoal>` declarations announcing each intermediate before it is computed.
5. Validate that the augmented trace is syntactically clean.

This produced `C_SFT_aug`. RLOO from `C_SFT_aug` with the composite reward produced `C_process`.

### A.4 What was empirically observed

- `C_SFT_aug` learned the subgoal *grammar* (~92% of rollouts emit `<subgoal>` tags; ~78% emit more than one) but not its *semantics*.
- `C_process` underperformed `C_outcome` on accuracy and **did not produce a discernible composite-reward bonus** in practice: the vast majority of emitted `<subgoal>` tags were *invalid* (declared target unreachable from declared inputs) or *not achieved* (subsequent computation did not reach the declared target).
- Qualitatively, the model treated `<subgoal>` declarations as annotation that decorated reasoning it would have produced anyway, and frequently as filler in dead-end search branches. Removing the tags would not have changed the arithmetic.

### A.5 Mechanistic explanation and theoretical connection

The pattern matches the prediction of **Strategic Information Allocation** (Kim et al., arXiv 2603.15500, March 2026 — VERIFY): at small scale, a reasoner without native epistemic capacity cannot be RL'd into routing useful structure through annotation-only tokens, because the tokens carry no inference-time functional load. Compare to:

- **Functional tokens that *did* work in prior CS 224R projects.** El et al.'s `<clean>` triggered context management at inference; Hu & Wang's function-call tokens triggered external arithmetic. Both *changed inference behavior*. Ours did not.
- **DeepSeek-R1 `<think>`.** Structural; the surrounding training pattern enforces "long reasoning before answer." But this requires a larger base model and much more training; not directly comparable.

The retrospective framing: `C_process` is data for the claim "annotation-only process reward does not change inference at 0.5B, even when paired with an exact verifier" — which is a positive contribution at the *theoretical-confirmation* level.

---

## Decision Log

Three rounds of recursive literature scan led to the current scope. Recording here so future-us doesn't re-litigate.

**Round 1.** Probe-aware reward (Phase 3B) was scoped out early because Papadatos & Freedman ("Linear Probe Penalties Reduce LLM Sycophancy", Dec 2024) and Taufeeque et al. ("The Obfuscation Atlas", Feb 2026) already characterize the regime; the novelty surplus did not justify the engineering risk on this timeline.

**Round 2.** Step-grounded confidence as a *novel mechanism* (vs. as a descriptive metric) was scoped out because MMBoundary (multimodal), SPAE (advantage-estimation), and Temporalizing Confidence (non-RL) collectively occupy the adjacent ground. We retain it as a single-arm descriptive metric on `C_outcome` only.

**Round 3.** Functional / introspection tokens (`<introspect>`, `<verify>`, `<commit>`, etc.) were scoped out because the 2024-novelty has saturated by 2026 across multiple papers, including DeepSeek-R1, SCoRe, and three concurrent 2026 works on trajectory drift. Adding a new token would compete in a crowded subfield. The `C_process` annotation-only arm was attempted before this decision crystallized; rather than discard, we retain it as a *negative result* whose mechanism confirms Strategic Information Allocation (Kim et al., 2603.15500, March 2026).

**Anchor decision.** The probe-as-measurement layer is rich enough on its own:
- Trace-final concealment gap on `C_SFT` and `C_outcome` (H1, H2).
- Position-resolved gap at confidence-asserting tokens (H3 — under-explored, low-cost to test).
- 2×2 cross-checkpoint transfer matrix decomposing drift vs. suppression.
- Layer C temporal trajectory over already-saved snapshots.
- `C_process` negative result as a complementary contribution.

**Abandoned arms (for the record):** probe-aware reward; introspection token; functional tokens; step-grounded confidence as a novel mechanism; expansion of `C_process` to additional λ/α sweeps; block-design subgoals; sub-function reasoning tokens; any new training run on top of the existing checkpoints. None of these are part of the headline. Future-us should not reopen them without a concrete novelty argument against the post-2026 literature.