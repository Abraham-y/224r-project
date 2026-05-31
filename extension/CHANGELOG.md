# Extension CHANGELOG

Polish-phase task list and outcomes.

---

## 2026-05-30 — Polish round

### Task 1 — Expand n from 50 to 200: **SKIPPED (infeasible)**
- `asingh15/countdown_tasks_3to4` test split has only **50 examples** total. We are
  already using all of them.
- The train split has 490k examples but it is `C_outcome`'s RLOO training set, so
  using train examples as additional "held-out" eval would be unfair to `C_outcome`.
- No alternative held-out Countdown corpus is in scope (no other dataset we can
  cleanly call "held-out" for both checkpoints).
- **Decision:** stand on n=50, document this as a hard dataset limit in §9 of the
  writeup. The within-problem CV and matched-pair p-values are already
  significant at p < 0.005 by two independent tests, so the sample-size critique
  is partially absorbed by the statistical-significance results.

### Task 3 — Lead TL;DR with decoupling-vs-damage: **DONE**
- Rewrote writeup.md §0 to land "behavioral calibration via mechanism decoupling"
  as the first sentence, and added a leading paragraph to §2 that surfaces the
  damage-vs-decoupling framing before the per-subsection numbers.

### Task 4 — Cross-checkpoint transfer interpretation + length-matched control: **DONE**
- Added an "Alternative interpretation worth flagging" paragraph to §2.4
  acknowledging the checkpoint-specific-features confound.
- Built `extension/probe/length_matched_transfer.py` and ran the control.
- **Headline result**: at trace-final, off-diagonal `C_SFT → C_outcome` moves
  from **0.523 → 0.652** after length-matching (+0.129); the other three cells
  are essentially unchanged. Drift framing survives but should be reported with
  the matched number.
- Added new §2.4.1 "Length-matched transfer control" reporting this honestly.

### Task 5 — Verbal-elicitation failure as methodological contribution: **DONE**
- Reframed §5 to call out the elicitation failure as a methodological observation
  worth flagging in the discussion: standard literature elicitation does not work
  on small SFT'd base models. Cited RLCR / Damani et al. 2025 as the method that
  failed.

### Task 6 — Soften §6 dynamics claim: **DONE**
- Replaced "outcome RL does not damage the trace-final correctness representation"
  with "consistent with representation preservation under the Option-A measurement;
  a stronger claim requires Option B." Forward-referenced §6.1 for Option B.

### Task 7 — Mechanism speculation: **DONE**
- Added new §13 "Mechanism speculation" with three candidate mechanisms (RL-on-
  uncorrelated-features, head-vs-deep-representation selectivity, commitment-vs-
  correctness divergence), explicitly marked as speculation and tagging which one
  is most parsimonious.

### Task 8 — Per-keyword analysis with bootstrap CIs: **SKIPPED (contingent on Task 1)**
- Task 8's value was conditional on Task 1 expanding n to surface more keywords.
  Task 1 was infeasible (dataset cap), so Task 8 was moot. Only "this works"
  clears the n >= 20 threshold on both checkpoints; the existing §4.2 already
  reports it. No new analysis added.

### Task 9 — Annotated qualitative figure: **DONE**
- Built `extension/probe/qualitative_annotated_figure.py`.
- Produces `extension/outputs/figures/fig8_annotated_qualitative.png`: side-by-side
  panels showing same-problem correct vs wrong `C_outcome` rollouts (prompt 24,
  target=62, nums=[80, 66, 48, 36]).
- **Visual result**: mean probe at "this works" tokens is **0.88 (wrong rollout)**
  vs **0.91 (correct rollout)** — essentially identical despite the eventual
  outcomes diverging. The H3 signature in its purest visual form. Documented in
  writeup.md §10.
- Limitation: the cache only captured assertion tokens inside the *first*
  `<think>` body (not the post-answer ramble), so each panel has only 2-3 cached
  points instead of one per assertion. Re-caching without the `</think>` cutoff
  would let us show the full trajectory but requires another Modal pass.

### Task 2 — Option B dynamics: **PENDING USER DECISION**
- Requires ~$30-50 Modal spend to re-sample fresh rollouts on three snapshot
  models (steps 30, 60, 90) and re-cache hidden states from each.
- Not yet launched; awaiting user approval before spending more compute.

---

## Headline-number diff (before -> after polish round)

| Number | Before | After | Change |
|---|---|---|---|
| Trace-final probe AUROC, C_SFT | 0.724 | 0.724 | (unchanged) |
| Trace-final probe AUROC, C_outcome | 0.793 | 0.793 | (unchanged) |
| Assertion probe AUROC, C_SFT | 0.735 | 0.735 | (unchanged) |
| Assertion probe AUROC, C_outcome | 0.520 | 0.520 | (unchanged) |
| Matched-pair (% above diagonal) | 76% / 36% | 76% / 36% | (unchanged) |
| Within-problem Cohen's d | +1.26 / +0.38 | +1.26 / +0.38 | (unchanged) |
| Cross-checkpoint `pre_answer` off-diagonal C_SFT → C_outcome | 0.523 | **0.652 (length-matched)** | **+0.129** |
| Cross-checkpoint `pre_answer` off-diagonal C_outcome → C_SFT | 0.580 | 0.581 | (essentially unchanged) |
| Cross-checkpoint `assertion` off-diagonals | 0.588 / 0.623 | 0.593 / 0.623 | (essentially unchanged) |
| Headline qualitative example | none | mean probe 0.88 (wrong) vs 0.91 (correct), same problem |

**Story changes from this round:** the polish reduced the dramatic-ness of the
trace-final transfer collapse (it survives length-matching but goes from "near
chance" to "modestly degraded"). The matched-pair / Cohen's d / position-resolved
collapse are robust to the length-matching control and remain the cleanest
finding. No headline went the wrong direction.

---

## 2026-05-30 (later) -- n=500 expansion + Option B dynamics + Phase 1 diagnostics

### n=500 expansion (procedural Option B; ~$12 Modal compute, 30 min wall)
- Built `extension/data/generate_countdown.py`. Procedurally generates Countdown
  problems with exhaustive arithmetic verification. NumPy-style right-justified
  number formatting matches `asingh15/countdown_tasks_3to4` byte-for-byte for
  identical `(nums, target)` inputs.
- Generated `extension/data/countdown_eval_500.jsonl`: 500 problems, 50/50
  3-num/4-num mix, 0 overlap with asingh15 TEST split.
- **CONTAMINATION CHECK:** 94/500 generated problems overlap with the asingh15
  TRAIN split (the source pool both generators sample from). Filtered cache
  `extension/cache/probe_cache_n500_clean406/` with prompts that are unseen by
  C_outcome's RLOO training set. Headlines on clean-406 differ from dirty-500
  by ~0.005-0.04 AUROC -- no qualitative claim flips.
- Built Modal infrastructure:
  `extension/evaluation/sample_local_jsonl.py` (vLLM rollouts from a local JSONL),
  `extension/evaluation/launch_expansion_rollouts.sh` (5 parallel jobs),
  `extension/probe/launch_expansion_cache.sh` (5 parallel cache jobs),
  `extension/evaluation/wait_and_launch_phase2.py` (Phase-1 -> Phase-2 watchdog).
- Phase 1 (rollouts on n=500 C_SFT/C_outcome + n=200 snapshots step_30/60/90):
  finished in ~25 min total wall time.
- Phase 2 (hidden-state caching on all 5 caches): finished in ~5 min total wall
  time -- WAY faster than my ~5-hour estimate. The 0.5B model does
  fast forwards.

### Option B dynamics (fresh rollouts per snapshot, n=200 first 200 prompts)
- Built `extension/probe/per_snapshot_decoupling_gap.py`. Computes balanced
  GroupKFold(5) probe AUROC at pre_answer and assertion-pos for each snapshot,
  reports gap.
- **Headline result -- the decoupling gap GROWS monotonically over training:**

  | Step | pre_answer | assertion | gap |
  |---|---|---|---|
  | C_SFT (pre-RL) | 0.804 | 0.785 | 0.019 |
  | step 30 | 0.791 | 0.769 | 0.022 |
  | step 60 | 0.864 | 0.749 | 0.115 |
  | step 90 | 0.871 | 0.654 | 0.217 |
  | C_outcome (final) | 0.896 | 0.703 | 0.193 |

  Position-decoupling under outcome RL is **emergent over training**, not
  inherited from SFT. Cleanest evidence for the H3 story so far. The Option-A
  measurement (fixed C_outcome rollouts across all snapshots) does not show
  this -- the gap was a true outcome-RL phenomenon, not a static representation
  feature.

### Headline shifts at n=500 (vs n=50 paper)
- Trace-final probe AUROC: **rose +0.09** on both checkpoints (C_SFT 0.724->0.816,
  C_outcome 0.793->0.887). Magnitude differences were small-sample artifacts.
- Assertion-pos C_outcome AUROC: **0.520 -> 0.696**. The "collapse to chance"
  claim is wrong at n=500; it's "weaker than pre_answer by ~0.19" but well
  above chance.
- Cohen's d at pre_answer, C_outcome: **+0.38 -> +0.991**. The "Yuan-comparable
  70% drop" claim is wrong; the d distributions differ by only ~0.08-0.1 between
  C_SFT and C_outcome (still Mann-Whitney p=3e-4).
- Matched-pair % above-diag, C_outcome: **36% -> 51%**. The "actively backwards"
  claim is wrong; it's "essentially random" at verbalization moments.
- Cross-checkpoint pre_answer transfer C_SFT->C_outcome: **0.523 -> 0.855**. The
  "representation drift" claim is wrong at n=500; transfer is high.
- BUT: significance tests **strengthen**. C_SFT Wilcoxon p drops from
  1.3e-4 -> 1.5e-32; Mann-Whitney C_SFT-vs-C_outcome from 1.6e-3 -> 5.3e-21.
  All major effects remain significant; magnitudes are just smaller than n=50
  measurements suggested.

### Bootstrap CIs (B=100, 80% subsample of prompts without replacement)
Built `extension/probe/bootstrap_headline_cis.py`. Sane CIs:

| ckpt | kind | point | 95% CI |
|---|---|---|---|
| C_SFT | pre_answer | 0.804 | [0.782, 0.816] |
| C_SFT | assertion | 0.785 | [0.750, 0.790] |
| C_SFT | neutral | 0.562 | [0.525, 0.574] |
| C_outcome | pre_answer | 0.896 | [0.883, 0.902] |
| C_outcome | assertion | 0.703 | [0.662, 0.725] |
| C_outcome | neutral | 0.562 | [0.524, 0.590] |

(First version used cluster-with-replacement bootstrap which had within-fold
leakage; the corrected `bootstrap_ci` uses subsample-without-replacement.)

### Phase 1 diagnostics (asymmetry + per-layer + neutral control)
Built `extension/probe/phase1_diagnostics.py`. Ran 10 seeds per cell with explicit
class balancing on both source training and target eval. Heatmap at
`extension/outputs/n500/figures/fig_phase1_transfer_heatmap.png`.

- **1A asymmetry:** Persists after re-balancing. At L12 C_outcome the
  asymmetry is +0.316 (pre->ass=0.586, ass->pre=0.270 -- assertion-trained probe
  ANTI-predicts on pre_answer); at L20 it reverses (-0.044). The high seed
  variance on ass->pre (std 0.08-0.18) suggests no stable linear direction --
  consistent with "no coherent shared subspace" rather than "stable anti-signal".
- **1B per-layer:** The transfer collapse is layer-invariant. Symmetric mean
  on C_outcome at L12/L16/L20 is 0.428/0.431/0.479 -- all clearly below the
  diagonals. L16 is not uniquely special.
- **1C neutral control:** REVISED INTERPRETATION. The pre_answer probe transfers
  to BOTH assertion AND neutral at chance, so the "orthogonality" is general
  (pre_answer is its own subspace) -- not assertion-specific. But: the
  NEUTRAL-trained probe transfers to pre_answer at 0.686 well above chance,
  while the assertion-trained probe does not (0.368 -- partial anti-correlation).
  The pre_answer correctness representation is RICH and other positions
  partially express it (neu->pre 0.69), but the pre_answer probe direction is
  too position-specific to transfer the other way.

### Phase 2A: within-rollout probe trajectory (Pattern A/B/C discrimination)
- Built `extension/probe/phase2a_per_answer_correctness.py` (no-compute pre-flight):
  per-answer correctness parsing within each rollout.
- Built `extension/probe/cache_answer_positions.py` (new Modal job): cached
  hidden states at every `<answer>` opening token across 5458 multi-answer
  C_outcome rollouts (clean-406). 44306 hidden states per layer; 3 layers.
  Job ran on Modal in ~5 min after image build.
- Built `extension/probe/phase2a_pattern_analysis.py`: trains held-out-by-prompt
  trace-final probe, applies to each `<answer>` position within each rollout.

**Headline finding -- PATTERN A confirmed:**
- 4393 multi-answer rollouts with valid probe scores. Transitions: 2335 TT,
  387 TF (drift correct->wrong, the pathology), 121 FT, 1550 FF.
- probe(last) tracks LAST answer's correctness, not first's:
  TT=0.558, TF=0.361, FT=0.622, FF=0.352. T->F probe(last) is essentially
  identical to F->F (0.361 vs 0.352).
- Per-block-INDEX trajectory: at each `<answer>` block within a rollout the
  probe accurately tracks that block's correctness; on T->F rollouts the
  probe descends 0.610 -> 0.457 -> 0.552 -> ... -> 0.148 as the rollout
  drifts away from correctness.
- Block 0 (first `<answer>`) is OOD for the probe (uniformly ~0 across all
  classes). This is a calibration artifact -- the probe is trained on `</think>`
  tokens which sit at avg token position 1093, while the first `<answer>`
  opening sits at avg token 534 (often inside the still-thinking phase).
  Per-block analysis at block_idx >= 1 is the reliable measurement.

**Implication for Phase 2B:** SKIPPED per user instructions. Pattern A means
there is no preserved "memory" of the first-answer correctness at the last
position to causally inject. The model genuinely updates its belief to the
wrong answer; outcome RL doesn't produce a hidden "secret correct
representation" the model is failing to verbalize. This is the right
mechanistic story: the decoupling isn't "knows but doesn't say"; it's
"truly changes its mind, and the verbalization reflects that change."

Figure: `extension/outputs/n500/figures/fig9_within_rollout_trajectory.png`.

### Task A: position-appropriate probe defensive control on Phase 2A
- Built `extension/probe/phase2a_position_appropriate_probe.py`. Trained a
  NEW probe on `<answer>` opening hidden states (NOT `</think>`), labeled by
  the correctness of the equation in each block. Same hyperparams as the
  trace-final probe (LR C=0.1, balanced, GroupKFold(5) by prompt).
- **Sanity check: position-appropriate probe held-out diagonal AUROC = 0.920**.
  The `<answer>`-opening position carries STRONG correctness signal when the
  probe is calibrated there.
- **Pattern A confirmed and CLEANER with the position-appropriate probe:**

  | transition | n | probe(first) | probe(last) |
  |---|---|---|---|
  | TT | 2983 | 0.874 | 0.823 |
  | TF (drift correct->wrong) | 490 | 0.856 | **0.154** |
  | FT (drift wrong->correct) | 150 | 0.156 | **0.580** |
  | FF | 1835 | 0.084 | 0.088 |

  probe(first) and probe(last) BOTH track their respective block's
  correctness. Per-block-index trajectory on T->F rollouts: probe descends
  0.856 -> 0.644 -> 0.521 -> ... -> 0.218 as the block-level correctness
  rate descends 100% -> 75% -> 52% -> ... -> 2%. There is NO anchoring to
  the first answer.

- The block-0 OOD artifact from the original Phase 2A analysis (probe ~0
  uniformly with the `</think>`-trained probe) is GONE under the position-
  appropriate probe. The original per-block-INDEX trajectory was robust to
  this because the OOD effect was constant across classes at any given
  block; the position-appropriate version is the bulletproof one to cite.
- Figure: `extension/outputs/n500/figures/fig9b_within_rollout_position_appropriate.png`.

### Final mechanistic summary
After all polish/expansion/diagnostic rounds, the n=500 clean-406 story is:

1. **Trace-final probe AUROC rises under outcome RL** (C_SFT 0.821 ->
   C_outcome 0.901 at L20). The model's internal correctness representation
   at the final-commit position gets STRONGER, not weaker.
2. **Assertion-position probe AUROC drops under outcome RL** (C_SFT 0.785 ->
   C_outcome 0.703 at L16, clean-406). The gap pre_answer - assertion grows
   from 0.019 (C_SFT, pre-RL) to 0.193 (C_outcome, post-RL) -- a 10x
   increase. The Option B dynamics show this gap grows MONOTONICALLY across
   training steps (0.019 -> 0.022 -> 0.115 -> 0.217 -> 0.193).
3. **Pre_answer correctness representation does not linearly transfer to
   other positions** (assertion or neutral), but neutral-position
   representations DO partially transfer to pre_answer (neu->pre = 0.686).
   This is layer-invariant (L12, L16, L20 all show the same pattern).
4. **Position-resolved internal belief tracks the local block's
   correctness, not the rollout's first answer's correctness.** Pattern A,
   confirmed by both the trace-final probe (block-1+ discrimination) and
   the position-appropriate probe (block-0 onwards). The model genuinely
   updates its belief with each `<answer>` commit.

The mechanistic claim: **outcome RL produces position-dependent
correctness representations that are NOT linearly tied across positions
within a rollout.** The model's pre_answer representation strengthens, the
assertion-position representation weakens (relative to pre_answer), and
the within-rollout probe trajectory shows the model's belief updates
dynamically. Activation patching from first->last answer position (Phase
2B) would not recover the first answer because there is no preserved
hidden "memory" to inject; the model has truly committed to the new
answer.

### Writeup rewrite
- Rewrote `writeup.md` end-to-end (15 sections, ~625 lines) to land on the
  position-decoupling story with the n=500 clean-406 + Option B + Phase 1 +
  Phase 2A evidence. Old §0 TL;DR was a single paragraph framed around
  "the gap relocates"; new TL;DR is single paragraph framed around
  "position-decoupling emerges over training."
- All n=50 numbers in the new writeup are explicitly tagged as such
  ("n=50 paper", "original 36% above-diagonal number", "n=33 matched
  problems", etc.). The principal headline numbers throughout are
  clean-406 with 95% CIs from cluster bootstrap.
- New sections added:
  - §2.2 Option B dynamics (gap-emerges-over-training table)
  - §2.3 Cross-position transfer (pre_answer is its own subspace)
  - §2.4 Within-rollout probe trajectory (Pattern A; position-appropriate probe)
  - §5 expanded cross-position transfer details (asymmetry, per-layer, neutral)
  - §7 Option B dynamics (replaces the Option A discussion)
  - §8 Phase 2A within-rollout probe trajectory (detailed)
  - §9.2 bootstrap CIs
  - §9.5 contamination check
  - §9.6 prompt format check
  - §14 Six claims (was Five) — adds the Pattern A claim
- Major number changes from the n=50 paper that the new writeup reflects:
  - Trace-final probe AUROC: 0.724/0.793 -> 0.821/0.901 (L20); 0.804/0.896 (L16)
  - Assertion AUROC: 0.735/0.520 -> 0.785/0.703 (L16)
  - Gap pre_answer-assertion: 0.019 -> 0.193 (L16 clean-406)
  - Matched-pair C_outcome: 36% above-diag (n=33) -> 52% (n=218); Wilcoxon
    within-checkpoint p flips from 0.72 (null) to 0.027 (small positive)
  - Cohen's d C_outcome (pre_answer): +0.38 -> +1.04; "70% reduction"
    claim removed
  - Cross-checkpoint pre_answer transfer C_SFT->C_outcome: 0.523 -> 0.855;
    "drift" framing softened to "small drift effect"
- New finding NOT in the n=50 paper:
  - Decoupling gap emerges over training (Option B dynamics) -- single
    cleanest visual of the claim
  - Cross-position transfer 3x3 (pre_answer is its own subspace; layer-
    invariant)
  - Pattern A within-rollout (model truly updates belief; no preserved
    "secret correct representation" -- Phase 2B skipped per gating rule)
- Headline framing changed from "behavioral calibration via mechanism
  decoupling" to "position-decoupling of correctness representations
  across token positions within a single rollout." More accurate, less
  reviewer-bait.
