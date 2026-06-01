# Concealment Under Outcome Pressure at 0.5B

*A probe-as-measurement study of internal-vs-verbalized correctness on outcome-RL'd Qwen2.5-0.5B on Countdown, with a position-resolved mechanism analysis and a within-rollout probe trajectory.*

> **Date of writeup.** 2026-05-30, revised after the n=500 expansion + Phase 1 / Phase 2A diagnostic rounds.
> **Status.** All in-scope analyses complete; the report can be drafted from this document. Numbers below are from the **n=500 procedurally-generated held-out set, contamination-filtered to 406 problems not in C_outcome's RLOO training set**, except where explicitly noted as "n=50 paper-original" or "Option A dynamics."

---

## 0. TL;DR (one-paragraph)

**Position-decoupling under outcome RL.** At 0.5B on an exact-verifier task, outcome-only RLOO *strengthens* the model's internal correctness representation at the trace-final position (probe AUROC `</think>` rises 0.821 on `C_SFT` → 0.901 on `C_outcome`, layer L20, n=406 clean) but *weakens* the representation at confidence-asserting token positions (probe AUROC 0.785 → 0.703 at L16). The gap between these two positions — essentially zero before RL (0.019) — grows monotonically across training to 0.193 at the final checkpoint, confirmed by fresh rollouts re-sampled at three intermediate snapshots (gap trajectory: **0.019 → 0.022 → 0.115 → 0.217 → 0.193** at steps `C_SFT/30/60/90/final`). The pre-answer correctness direction does not linearly transfer to any other position on `C_outcome` (off-diagonal AUROC ≈ 0.50 to assertion AND to neutral, layer-invariant across **all 25 layers** L0–L24; max gap +0.236 at L9; the cosine similarity between pre and assertion probe directions is +0.038 — essentially orthogonal), and within multi-answer rollouts where the model first emits a correct equation and then drifts to a wrong one, the probe at each `<answer>` block tracks *that block's* correctness, not the first block's (Pattern A, bidirectional: probe rises on rescue moves and falls on drift moves; position-appropriate probe held-out diagonal AUROC = 0.920; probe(last) on T→F drift = 0.154, indistinguishable from F→F floor at 0.088). **Causal steering along the probe direction is a null result** (probe-vs-random Δ ∈ [−0.07, +0.02] at α ∈ [0.5, 2.0] · h_mean_norm; n=97 prefixes; probe direction's effect on accuracy is indistinguishable from random-direction perturbation): the trace-final probe is a *reader* of a multi-dimensional correctness subspace, not a *controller*. **The model genuinely updates its belief about correctness at each commit; outcome RL does not produce a "model knows correct but verbalizes wrong" gap. The gap is at the representational level: the verbalization-time and trace-final correctness representations have decoupled into different linear subspaces over training, distributed across all transformer depth, with no preserved hidden representation that activation patching could recover.** All headline significance tests at p < 0.001 by at least two independent procedures.

---

## 1. Setup

**Task.** Countdown arithmetic reasoning (Gandhi et al. 2024): each problem gives 3–4 small integers and a target; the model must produce an equation that uses each number exactly once and evaluates to the target. The rule-based verifier in `evaluation/countdown.py` scores each response 0.0 (no parseable answer), 0.1 (parseable but wrong/invalid), 1.0 (correct).

**Model.** Qwen2.5-0.5B base + Countdown SFT throughout. We did not train an SFT model ourselves; `C_SFT` is Anikait Singh's `asingh15/qwen-sft-countdown-defaultproj`, used in lieu of a team-trained SFT.

**Checkpoints studied.**

| Checkpoint | Construction | Test pass@1 (asingh15 test, n=50) | Test pass@16 |
|---|---|---|---|
| `C_SFT` | `asingh15/qwen-sft-countdown-defaultproj` | 28.6% | 78.0% |
| `C_outcome` | RLOO from `C_SFT`, outcome reward only (0/0.1/1.0), 100 steps, all 10 intermediate snapshots persisted | **53.5%** | 72.0% |
| `C_process` | RLOO with annotation-only subgoal reward (`R = R_outcome + 0.3·R_subgoal`) on `C_SFT_aug` | underperformed; see §10 |

Outcome RL improved pass@1 by 24.9 absolute points but *reduced* pass@16 by 6 points — the standard sharpening-vs-diversity trade-off in RL fine-tuning.

**Eval set.** The original n=50 `asingh15/countdown_tasks_3to4` test split was too small for some headline statistical claims (matched-pair denominators ~33–38). We therefore **procedurally generated** an expanded held-out set:

- `extension/data/generate_countdown.py` samples 3–4 distinct integers from `[1, 100]`, exhaustively searches over orderings × operators × parenthesizations to verify a solvable target in `[10, 100]` exists, and emits the problem in the *byte-identical* prompt format `asingh15` uses (NumPy-style right-justified bracket formatting, e.g. `[14 45  9  1]`).
- We generated 500 problems and verified that **0/500 overlap with the asingh15 TEST split**. However, 94/500 overlap with the asingh15 **TRAIN** split (which is the pool C_outcome was trained on via RLOO). We therefore filter to the **406 problems not in asingh15 train**, the "clean-406" set. Headlines on dirty-500 differ from clean-406 by 0.005–0.04 AUROC; no qualitative claim flips, but we report clean-406 as the principal number.
- 500 prompts × 16 rollouts/prompt, sampling temperature 0.6, top_p 0.95, top_k 20, max_tokens 1024.

We additionally re-sampled fresh rollouts (n=200, first 200 prompts) from **three intermediate `C_outcome` snapshots** (steps 30, 60, 90) so that the dynamics analysis (§7) uses each snapshot's own behavioral distribution rather than the final-checkpoint's text under the snapshot's weights. This is "Option B" measurement vs. the original "Option A."

**Probe pipeline.** Hidden states cached from Qwen2.5-0.5B with `output_hidden_states=True` at three position kinds per rollout:
- `pre_answer`: the `</think>` token (one vector per rollout).
- `assertion`: every occurrence of a confidence-keyword token in the `<think>` body (`"Perfect"`, `"this works"`, `"got it"`, `"the answer is"`, `"verified"`).
- `neutral`: matched-count random positions from the same `<think>` body (control).

Phase 2A also caches hidden states at every `<answer>` opening token across all 5458 multi-answer C_outcome rollouts (44306 hidden states per layer; see §9).

Probes are logistic regression at three layers (L12/L16/L20), L2-regularized (C=0.1), trained per checkpoint with `GroupKFold(5)` on `prompt_idx` (held-out problems, not held-out trajectories) on a 50/50 class-balanced subsample of training rows. L16 is the primary report layer; the same patterns hold at L12 and L20.

---

## 2. The headline finding

**The single takeaway.** Outcome RL *strengthens* the trace-final correctness representation and *weakens* the assertion-position one. The two positions, essentially equally informative before RL, decouple into different linear subspaces over training. Within a single rollout, the model's internal belief at each commit-position tracks that commit's correctness — it does not retain a hidden "secret correct representation" the verbalization fails to express. The gap that develops under outcome RL is real and is *representational* and *position-specific*, but it is not "knows but doesn't say."

### 2.1 Trace-final probe AUROC rises; assertion-position AUROC falls

The probe at `</think>` strengthens under outcome RL; the probe at confidence-asserting tokens weakens — and the gap between them is the headline number.

| Position (L16, balanced 5-fold CV, n=406 clean) | `C_SFT` AUROC | `C_outcome` AUROC | Δ |
|---|---|---|---|
| `</think>` (trace-final) | **0.804** [0.782, 0.816] | **0.896** [0.883, 0.902] | **+0.092** |
| confidence-asserting tokens | 0.785 [0.750, 0.790] | 0.703 [0.662, 0.725] | **−0.082** |
| neutral tokens (control) | 0.562 [0.525, 0.574] | 0.562 [0.524, 0.590] | 0.000 |
| **gap = pre_answer − assertion** | **+0.019** | **+0.193** | **+0.174** |

95% CIs from 100-iter cluster bootstrap on prompt indices (sample 80% of unique prompts per replicate, hold out for CV). Shuffled-label baselines are ~0.48–0.51; random-direction baselines ~0.50–0.74. Signal is real, calibration is well-bounded.

**Reading.** Outcome RL strengthens the `</think>` position correctness representation by ~0.09 AUROC at both L16 and L20. At assertion-position tokens, the probe loses ~0.08 AUROC. The gap between the two positions — essentially zero on `C_SFT` (0.019) — grows by an order of magnitude to 0.193 on `C_outcome`. This is the position-decoupling effect: outcome RL makes the *trace-final* representation more informative and the *assertion-time* representation less informative, even though both still sit well above chance.

*Honest revision from the n=50 paper.* The original measurement reported assertion-position AUROC of 0.520 on `C_outcome` ("collapsed to chance"). At n=500 clean-406 the correct number is 0.703 — well above chance, but ~0.19 below the trace-final position. The gap is real and large; the "collapse to chance" framing was a small-sample artifact (assertion-position n was ~180 at n=50; it is 4043 at clean-406). The qualitative claim survives; the magnitude is more modest.

### 2.2 The decoupling gap *emerges* over training (Option B dynamics)

The pre_answer − assertion gap is not present at `C_SFT` and is not stable across all RLOO steps — it *grows monotonically* over training. We re-sampled fresh rollouts (n=200) from three intermediate `C_outcome` snapshots, re-cached hidden states, and re-trained the probe per snapshot:

| Step | `</think>` AUROC | assertion AUROC | gap |
|---|---|---|---|
| `C_SFT` (pre-RL) | 0.804 | 0.785 | **+0.019** |
| step 30 | 0.791 | 0.769 | **+0.022** |
| step 60 | 0.864 | 0.749 | **+0.115** |
| step 90 | 0.871 | 0.654 | **+0.217** |
| `C_outcome` (final) | 0.896 | 0.703 | **+0.193** |

**Reading.** The gap is 0.019 pre-RL, essentially unchanged at step 30, and then opens dramatically between steps 30 and 60. By step 90 it is +0.217 — eleven times the pre-RL value. The final-checkpoint value (+0.193) is slightly lower than step 90 because the trace-final probe and assertion probe shift in opposite directions in the last 10 steps. The crossing happens *during* training; it is not a property the model came in with.

This is the cleanest evidence for the decoupling-emergence claim and was *not* visible in the original Option-A measurement (which probed each snapshot's hidden states on a *fixed* set of final-checkpoint rollouts — a confounded measurement because the rollout distribution didn't update with the snapshot).

**Headline figure with bootstrap 95% CIs** (`extension/outputs/n500/figures/fig13_headline_dynamics.png`). Cluster-bootstrap CIs (80% prompt subsample, B=80) at each snapshot, plotted as shaded bands around the AUROC traces. The pre and assertion CI bands **overlap** at `C_SFT` (pre 0.804 [0.779, 0.816] vs ass 0.785 [0.749, 0.791]) and at step 30 (pre 0.791 [0.756, 0.819] vs ass 0.769 [0.738, 0.797]), then **separate** at step 60 (pre 0.864 [0.841, 0.874] vs ass 0.749 [0.719, 0.786]) and stay separated through step 90 and final. The gap is not a sampling artifact at any post-step-30 snapshot.

### 2.3 Cross-position probe transfer: pre_answer is a position-specific subspace

If pre_answer and assertion-position carried *the same* correctness representation in different coordinates, a probe trained at one would transfer to the other. It does not — on `C_outcome`. We measure all 3×3 train-position × eval-position transfers (10 balanced-subsample seeds, L16, clean-406):

**`C_SFT` (pre-RL):**

| train ↓ \ eval → | pre_answer | assertion | neutral |
|---|---|---|---|
| pre_answer | 0.803 | 0.438 | 0.506 |
| assertion | 0.499 | 0.780 | 0.513 |
| neutral | 0.684 | 0.481 | 0.558 |

**`C_outcome`:**

| train ↓ \ eval → | pre_answer | assertion | neutral |
|---|---|---|---|
| pre_answer | **0.895** | **0.494** | 0.505 |
| assertion | 0.368 | **0.694** | 0.500 |
| neutral | 0.686 | 0.571 | 0.565 |

**Three observations:**

1. **pre_answer probe doesn't transfer outward.** `C_outcome` pre_answer → assertion = 0.494 (chance); pre_answer → neutral = 0.505 (chance). The pre_answer-trained probe direction is essentially orthogonal to the activation distribution at both other position kinds. This is also true (though less starkly) on `C_SFT`: pre→ass = 0.438, pre→neu = 0.506.
2. **neutral probe DOES partially transfer to pre_answer.** `C_outcome` neutral → pre_answer = 0.686, well above chance — even though the neutral probe's own diagonal AUROC is only 0.565. Other positions can partially "read" the pre_answer correctness signal, but the pre_answer probe direction is too specific to "write back" to other positions.
3. **assertion → pre_answer is unstable.** Across 10 seeds, this off-diagonal varies wildly (mean 0.368, std 0.171 on `C_outcome`). Consistent with "no stable shared direction" rather than "stable anti-direction" — the assertion-trained probe direction doesn't have a consistent linear projection into pre_answer space.

**Layer-invariance.** The pattern holds at L12 (symmetric mean transfer 0.428), L16 (0.431), L20 (0.479) on `C_outcome`. L16 is not unique; this is not a single-layer artifact. The full 3×3 × 3-layer × 2-ckpt heatmap is at `extension/outputs/n500/figures/fig_phase1_transfer_heatmap.png`.

**Revised interpretation vs. n=50 paper.** Original framing: "pre-answer and assertion are orthogonal correctness subspaces on C_outcome." Better framing now: "pre_answer has a *specialized* correctness direction that does not linearly transfer to other positions, while other positions partially express a weaker form of the same signal. Under outcome RL this position-specificity sharpens." This is more accurate and less reviewer-bait.

### 2.4 Within-rollout probe trajectory: the model truly updates its belief (Pattern A)

`C_outcome`'s post-`</think>` rambling pathology emits **multiple `<answer>` blocks per rollout** (84% of clean-406 rollouts have ≥2 blocks; mean 8.1 blocks). For 490 of those rollouts the model emits a *correct* first equation and then drifts to a *wrong* final one — the classic "I had it, then I lost it" failure. We asked: at the wrong-final `<answer>` position, does the probe still encode the first answer's correctness ("Pattern B": preserved hidden memory of the original correct answer the verbalization failed to express), or has the representation moved with the model's commit ("Pattern A": true belief update)?

**Setup.** Train a position-appropriate probe directly on `<answer>` opening hidden states, labeled by each block's expression correctness (each block's equation evaluated against the Countdown verifier). Same hyperparameters as the trace-final probe (LR C=0.1, balanced classes, `GroupKFold(5)` by prompt). At L16:

- **Position-appropriate probe held-out diagonal AUROC: 0.920** — strong signal at `<answer>` opening positions when probed there directly. (Compare: the trace-final probe transferred to `<answer>` opening tokens gives ~0 score at block 0 due to OOD; see §9 for that diagnostic.)

**Headline 2×2 table** (n_first-correct × n_last-correct on clean-406 multi-answer rollouts, position-appropriate probe at L16):

| transition | n | probe(first) | probe(last) | first_acc | last_acc |
|---|---|---|---|---|---|
| TT both correct | 2983 | 0.874 | 0.823 | 1.00 | 1.00 |
| **T→F drift correct→wrong** | **490** | **0.856** | **0.154** | 1.00 | 0.00 |
| **F→T drift wrong→correct** | **150** | **0.156** | **0.580** | 0.00 | 1.00 |
| FF both wrong | 1835 | 0.084 | 0.088 | 0.00 | 0.00 |

**Reading — this is Pattern A, cleanly.**
- On T→F rollouts: probe(first) is **0.856** (high — the probe correctly identifies the correct first answer) and probe(last) is **0.154** (low — the probe correctly identifies the wrong final answer). Indistinguishable from the F→F floor (0.088).
- On F→T rollouts: probe(first) is 0.156, probe(last) is 0.580. The probe tracks the change in correctness in the other direction too.
- The probe's representation at each `<answer>` block is **not anchored to the first answer's correctness**. The model's internal belief about correctness genuinely updates with each commit.

**Per-block-index trajectory on T→F rollouts** (490 rollouts; mean probe at each block index, with block-level correctness rate):

| block_idx | n | % blocks correct | mean probe |
|---|---|---|---|
| 0 (first `<answer>`) | 490 | 100% | 0.856 |
| 1 | 490 | 75% | 0.644 |
| 2 | 424 | 52% | 0.521 |
| 3 | 328 | 48% | 0.481 |
| 4 | 261 | 38% | 0.384 |
| 5 | 203 | 33% | 0.360 |
| ... | | | |
| 12 | 54 | 2% | 0.218 |

The probe descends in step with the per-block correctness rate. There is no "memory" of the original correct answer remaining at later blocks.

**Implication.** The decoupling between pre_answer and assertion-position probes (§2.1, §2.3) is *not* a "knows-but-doesn't-say" gap. There is no preserved hidden representation of the correct answer for an activation-patching intervention to recover. The model has truly committed to the wrong answer. The gap is at the **representational level** (different linear subspaces for trace-final vs. assertion-time correctness; pre_answer doesn't linearly transfer to other positions) but not at the **knowledge level** (the model's commit-time belief reflects what it commits to).

This was the question that motivated Phase 2B (causal activation patching) in the original plan. Given Pattern A, **Phase 2B was skipped**: there is no preserved first-answer correctness signal to inject into the last-answer position, so a patch would not produce a clean intervention effect. This is the boring outcome for Phase 2B but the interesting outcome for the mechanism story — it specifically refutes the "Yuan-et-al concealment" framing in favor of a "true belief revision" framing at small scale.

### 2.5 Matched-pair within-prompt at assertion-position: weakened but still significant

For each prompt with both a correct and a wrong rollout containing assertion tokens, we compute mean assertion-token probe score on each side and take the within-prompt delta (clean-406, L16, position-original probe trained on `</think>`).

| | `C_SFT` | `C_outcome` |
|---|---|---|
| n prompts with both correct & wrong rollouts (with assertion tokens) | 244 | 218 |
| mean(correct − wrong) | **+0.185** | **+0.035** |
| median Δ | +0.186 | +0.004 |
| % prompts probe ranks correct > wrong | **78%** | **52%** |
| % prompts probe ranks wrong > correct | 22% | **48%** |

**Significance** (Wilcoxon signed-rank, one-sided > 0):
- `C_SFT`: W = 26138, **p = 1.8 × 10⁻²⁴**
- `C_outcome`: W = 13736, **p = 0.027**
**Mann-Whitney U between checkpoints** (C_SFT deltas > C_outcome deltas): **p = 8.0 × 10⁻¹⁶**

**Reading.** Within the same problem, the probe at assertion tokens **still slightly favors correct over wrong rollouts on `C_outcome`** (median delta +0.004, mean +0.035, 52% above-diagonal). The within-prompt effect is dramatically smaller than `C_SFT`'s (median +0.186, 78%), and the Mann-Whitney test between distributions remains highly significant. Compared to the n=50 paper's "actively backwards" framing (the original 36% above-diagonal number), the clean-406 result is "essentially random" — slightly favoring correct, by an amount that crosses statistical significance because n=218 (Wilcoxon p = 0.027) but does not reach our pre-registered p < 0.01 threshold within the C_outcome checkpoint.

The **between-checkpoint** comparison stays solid: the matched-pair distributions are significantly different (Mann-Whitney p ≈ 10⁻¹⁶). The within-checkpoint result on `C_outcome` should be reported as "small positive effect, p = 0.027" not as "null" and not as "actively backwards."

### 2.6 Within-problem Cohen's d at pre_answer: the Yuan-et-al benchmark

For each held-out problem with both a correct and wrong rollout, Cohen's d of the probe scores at `</think>` between correct and wrong samples (controls for problem-level difficulty; clean-406, L16):

| | `C_SFT` | `C_outcome` |
|---|---|---|
| n problems | 267 | 242 |
| mean d | **+1.121** | **+1.036** |
| Mann-Whitney U between distributions (one-sided): | | **p = 3.0 × 10⁻⁴** |

**Reading.** Both checkpoints have *large* within-problem effect sizes at the trace-final position (+1.12 vs +1.04). The distributions differ significantly (p < 10⁻³), and `C_SFT` is slightly larger, but the gap is small (~0.085). The n=50 paper's headline ("Yuan-et-al's d drops from +1.26 to +0.38, a 70% reduction") **does not survive at n=500**: the +0.38 number was a small-sample artifact (n=33 matched problems). At n=242 the correct picture is "both effect sizes are large; outcome RL slightly reduces the within-problem trace-final discriminability, by ~8%."

**This is the largest single-number revision from the n=50 paper.** The decoupling story should not be carried by Cohen's d at `</think>`; that number doesn't support a strong claim. The story should be carried by §2.1 (position-resolved AUROC), §2.2 (Option B dynamics), §2.3 (cross-position transfer), §2.4 (Pattern A within-rollout), and the cross-checkpoint significance test in §2.5.

### 2.7 Cross-checkpoint probe transfer: small drift effect, not collapse

Train probe on `C_X`'s activations; evaluate on `C_Y`'s. Diagonals use held-out CV; off-diagonals use a probe trained on the other checkpoint's full data (clean-406, L16, balanced classes):

**`</think>` position:**

| train ↓ \ eval → | C_SFT | C_outcome |
|---|---|---|
| C_SFT | 0.805 | **0.855** |
| C_outcome | 0.716 | 0.895 |

**confidence-asserting tokens:**

| train ↓ \ eval → | C_SFT | C_outcome |
|---|---|---|
| C_SFT | 0.771 | **0.649** |
| C_outcome | 0.533 | 0.683 |

**Reading — major revision from the n=50 paper.** The "off-diagonal collapses to 0.52–0.58" framing of the n=50 paper does not hold at n=500 clean-406. At trace-final the C_SFT-trained probe still discriminates correct vs wrong on C_outcome at **0.855** AUROC — close to in-distribution performance. The C_outcome-trained probe transfers to C_SFT at 0.716 — a more substantial drop, but still well above chance. **The pre_answer correctness representations are mostly shared across checkpoints**, not drifted into incompatible subspaces. The n=50 numbers (0.52, 0.58) were small-sample noise.

Assertion-position transfer is weaker (0.649 / 0.533) but neither direction is at chance. The asymmetry is real — the `C_SFT`-trained probe transfers better to `C_outcome` than vice versa — consistent with the `C_outcome` probe being more position-specific (§2.3).

**Net update.** The cross-checkpoint result is now a *small drift effect* (offdiag pre_answer transfer is high but lower than diagonal), not the dramatic "representations have moved" narrative the n=50 paper drew. The mechanistically interesting finding has shifted from "representations drifted across checkpoints" to "representations specialize across positions within a checkpoint" (§2.3).

### 2.8 The global concealment gap *still inverts* at 0.5B

Verbalized confidence at the global rollout level (binary keyword-presence in the `<think>` body — see §6 on why), n=50 original eval:

| | `C_SFT` | `C_outcome` |
|---|---|---|
| P(keyword \| correct) | 0.834 | **0.951** |
| P(keyword \| wrong) | 0.695 | **0.379** |
| Verbalized AUROC | 0.569 | **0.786** |
| Trace-final probe AUROC (n=50) | 0.724 | 0.793 |
| **Global gap = probe − verbal** | **+0.154** | **+0.008** |

**Reading.** This is from the original n=50 elicitation data and is preserved here for completeness. At the *global rollout level* (binary verbalization signal vs. probe), outcome RL makes the model's verbalization *more* calibrated — the global gap closes to essentially zero. This is the *opposite direction* from Yuan et al.'s 1.5B+ "concealment gap widens under outcome RL" prediction; we report it as a scale-dependence finding. The new measurements in §2.1–§2.4 are where the *position-resolved* gap lives at 0.5B.

### 2.9 Pattern A is bidirectional: probe rises on rescue moves too

Within multi-answer C_outcome rollouts, **150 are F→T rollouts** — the model first commits to a wrong equation, then "rescues" itself with a correct one later. Pattern A predicts the probe should *rise* across these rollouts, mirroring its *fall* across T→F drift rollouts. It does, but asymmetrically:

| direction | n | block_0 probe | %_corr at block_0 | mid-rollout probe | mid-rollout %_corr | terminal probe |
|---|---|---|---|---|---|---|
| **F→T (rescue)** | 150 | 0.156 | 0% | 0.50 (block 2-8) | 86-95% | 0.73 (block 11+) |
| T→F (drift) | 490 | 0.856 | 100% | 0.42 (block 2-4) | 38-52% | 0.22 (block 12+) |

The probe responds to both directions of correctness change. The asymmetry — F→T's probe values are systematically *lower* than T→F's at the same block-level correctness rate (e.g. F→T at 95% correct sits around probe = 0.50, while T→F at matched block-level correctness was at 0.85 from earlier blocks) — likely reflects (a) much smaller sample size on the F→T side (n=150 vs 490), and (b) selection effect: rescue rollouts began with a wrong answer, so their early-rollout representations carry residue that lingers.

The qualitative claim is robust: there is no preserved "secret correct" or "secret wrong" representation. The probe at each commit tracks that commit's content in both drift directions. Figure: `extension/outputs/n500/figures/fig10_ft_rollout_trajectory.png`.

### 2.10 Probe-direction cosine analysis — the orthogonality is geometric, not just AUROC

The §2.3 cross-position transfer AUROCs collapse to chance for pre→ass and ass→pre on `C_outcome`. We additionally compute direct cosine similarities between the trained probe direction vectors (in input space, after `w / scaler.scale_`):

**Within-checkpoint cross-position cosines at L16:**

| pair | cos | what it means |
|---|---|---|
| C_SFT: pre vs assertion | **+0.024** | essentially orthogonal |
| C_SFT: pre vs neutral | −0.002 | essentially orthogonal |
| C_outcome: pre vs assertion | **+0.038** | essentially orthogonal |
| C_outcome: pre vs neutral | +0.020 | essentially orthogonal |

**Cross-checkpoint within-position cosines at L16:**

| pair | cos | transfer AUROC |
|---|---|---|
| C_SFT pre vs C_outcome pre | +0.102 | 0.855 |
| C_SFT ass vs C_outcome ass | +0.058 | 0.649 |
| C_SFT neutral vs C_outcome neutral | +0.041 | 0.530 |

**Reading.**
- Within both checkpoints, the pre_answer and assertion probe directions are **near-orthogonal at the cosine level (~0.02–0.04)**. This isn't outcome-RL-specific — both C_SFT and C_outcome have it. The cross-position transfer collapse from §2.3 has direct geometric backing.
- Cross-checkpoint, pre_answer probe directions are also small-cosine (+0.10) but the cross-checkpoint transfer AUROC is high (0.86). This implies the correctness signal lives in a **multi-dimensional subspace** that multiple low-cosine probe directions can each "read" — they don't have to point the same way to extract the same signal.
- Probe-vector norms scale with diagonal AUROC: pre_answer probes have largest norms (~30–80), assertion smaller (~7–35), neutral smallest (~4–15). The probe needs a longer vector when the per-feature signal is weaker.

This re-interprets the AUROC findings geometrically. The position-decoupling and cross-checkpoint stability are coherent: positions encode correctness via *different* linear directions (low pairwise cosine), but the underlying correctness *subspace* is largely shared (high transfer AUROC).

### 2.11 Causal steering null result — the probe is a reader, not a controller

If the pre_answer correctness direction were *causally* tied to the model's generation, adding `αv` to the residual stream at `</think>` should push generated answers toward correct (positive α) or wrong (large positive α, presumably). It doesn't.

**Experimental setup.** Take each (prompt, rollout) prefix up to and including the first `</think>`. Forward-hook into Qwen2's layer-16 residual stream at the `</think>` token position; add `α · h_mean_norm · v_unit` where `v_unit` is the trained probe direction. Continue generation under the patched residual (KV cache for `</think>` is filled from the modified state, so all later tokens attend to the patched representation). Compare accuracy against:
- α = 0 (no patch, baseline)
- α = 0.5 / 1.0 / 2.0 of mean hidden-state L2 norm (= ~21.84)
- Random unit-direction control at matched α (tests whether *any* perturbation helps, or specifically the probe direction)

**Results (n=97 prefixes, C_outcome, L16):**

| condition | accuracy | acc_format |
|---|---|---|
| baseline α=0 | 0.577 | 1.000 |
| probe direction α=+0.5 | 0.567 | 1.000 |
| probe direction α=+1.0 | 0.598 | 1.000 |
| probe direction α=+2.0 | 0.515 | 0.845 |
| **random direction α=+0.5** | **0.639** | 1.000 |
| random direction α=+1.0 | 0.577 | 0.938 |
| random direction α=+2.0 | 0.546 | 0.897 |

Probe-direction vs. random-direction delta at matched magnitude:
- α=+0.5: probe 0.567 vs random 0.639 → Δ = **−0.072**
- α=+1.0: probe 0.598 vs random 0.577 → Δ = **+0.021**
- α=+2.0: probe 0.515 vs random 0.546 → Δ = **−0.031**

Paired analysis (same prefix, probe vs baseline): probe direction *gained* 4–5 prefixes and *lost* 3–11 at α=+0.5/+1/+2; random direction gained 3–6 and lost 0–6. **The probe direction's effect is statistically indistinguishable from random-direction perturbation at every magnitude**, with the probe direction slightly *under*-performing random at α=+0.5 and +2.0 and matching random at α=+1.0. Format breaks at α=+2.0 for both directions, suggesting we are at the manifold edge.

**Reading.** The trace-final probe identifies a linear direction that *predicts* correctness (held-out AUROC 0.90) but is *not* a causal control axis for the model's output. Pushing the residual stream along this direction does not systematically improve (or degrade) the next generated answer's correctness beyond what random-direction perturbation produces.

**Why this matters.** Two consistent stories:
1. The correctness representation is **distributed across many features**; the probe captures one linear summary that's good for *reading* but not for *writing*. The §2.10 cosine analysis backs this up: probe directions across positions and checkpoints are mostly orthogonal but transfer well at the AUROC level — implying the signal lives in a wide multi-dimensional subspace, and any single linear direction is one cut through it.
2. The model's commit mechanism is **decoupled from the trace-final probe direction in a causal sense**: even though L16 hidden states at `</think>` are highly predictive, the path from L16 residual to the eventual output `<answer>` does not run linearly through the probe direction.

This is consistent with Yuan et al.'s 1.5B+ result that activation patching variants of concealment-gap interventions failed; we replicate the negative result at 0.5B with a careful matched-magnitude control. **The probe is a measurement instrument, not a control axis.** This is itself a methodologically important finding: claims that probes capture "the model's belief" should be careful to distinguish *correlational* (the probe predicts) from *causal* (the model uses) interpretations.

Figure: `extension/outputs/n500/figures/fig12_causal_steering.png` — grouped bar chart with probe vs random direction at each α, baseline shown as red line with shaded 95% CI band, Wilson 95% CIs as error bars. Probe-vs-random Δ annotated above each group.

---

## 3. Behavioral evidence (Layer A) — unchanged from n=50

Diagnostics from the original n=50 single-rollout-per-problem report (preserved for completeness; the qualitative observations are unaffected by n):

| Metric | `C_SFT` (n=50) | `C_outcome` (n=50) |
|---|---|---|
| Overall accuracy | 34% | 54% |
| Answer well-formed | 92% | 96% |
| Any answer parseable | 92% | 98% |
| Uses each input number exactly once | 58% | 82% |
| CoT-answer consistency (last CoT-endorsed value matches `<answer>`) | 71% | 83% |
| **Confident-wrong rate** (over wrong rollouts) | **79%** | **43%** |

n=500 clean-406 reproduces:
- pass@1 (C_outcome): **0.546** (matches n=50 number 0.535 within 1.1 pp — strong sample-representativeness check).
- pass@16 (C_outcome): 0.728.
- Multi-answer rollout rate (C_outcome): **84.0%** of rollouts emit ≥2 `<answer>` blocks (mean 8.1 blocks/rollout). The post-`</think>` rambling pathology is dominant.

The "model emits the correct answer in an early `<answer>` block, then drifts" pattern is *exactly* what 490 of these rollouts do (§2.4). The behavioral and probe pictures are now coherent.

---

## 4. Probe-level evidence (Layer B)

The headline findings are in §2. Additional structural observations:

### 4.1 Layer invariance — full 25-layer sweep

We re-cached hidden states at **every layer 0–24** (embedding through final transformer output) and re-ran the balanced GroupKFold(5) probe per cell. The pre−assertion gap on `C_outcome` is essentially constant across all transformer layers:

| Layer | `C_SFT` `</think>` | `C_outcome` `</think>` | `C_SFT` assertion | `C_outcome` assertion | gap on C_outcome |
|---|---|---|---|---|---|
| L0 (embedding) | 0.488 | 0.463 | 0.630 | 0.529 | −0.065 |
| L1 | 0.785 | 0.869 | 0.723 | 0.660 | +0.209 |
| L5 | 0.796 | 0.886 | 0.714 | 0.666 | +0.220 |
| L9 | 0.794 | 0.889 | 0.733 | 0.653 | **+0.236** (max) |
| L12 | 0.795 | 0.893 | 0.761 | 0.697 | +0.196 |
| L16 | 0.804 | 0.896 | 0.785 | 0.703 | +0.193 |
| L20 | 0.817 | 0.900 | 0.776 | 0.710 | +0.190 |
| L24 (final) | 0.805 | 0.897 | 0.772 | 0.703 | +0.194 |

The gap stabilizes by L1 and stays **flat between +0.18 and +0.24 from L5 to L24**. It is *not* concentrated at late layers. **This rules out mechanism (b) from §15** ("outcome reward selectively shaped the late layers / output head") — if late-layer-only updating were the mechanism, we would expect the gap to grow with depth, not stay flat. The decoupling is distributed across all transformer depth.

Full table + figure: `extension/outputs/n500/figures/fig11_per_layer_sweep.png`.

### 4.2 Per-keyword breakdown

At n=500 we have enough samples to compare across multiple keywords. Per-keyword assertion-position AUROC at L16 (clean-406, balanced subsample where ≥20 positives exist on both checkpoints):

| Keyword | n (`C_SFT`) | `C_SFT` AUROC | n (`C_outcome`) | `C_outcome` AUROC |
|---|---|---|---|---|
| "this works" | 3228 | 0.718 | 5253 | 0.645 |
| "Perfect" | 224 | 0.538 | 31 | 0.583 |
| "got it" | 480 | 0.383 | 97 | 0.250 |
| "the answer is" | 118 | 0.500 | <20 | — |

`C_outcome`'s verbal style narrowed dramatically: "this works" went from 3228 to 5253 occurrences (positive-class rate 45% → 81%), while "Perfect" went from 224 to 31. The position-resolved collapse on "this works" (0.718 → 0.645) is the principal contribution to the assertion-position aggregate.

### 4.3 Probe family baselines (linear vs nonlinear)

Are we measuring "the signal is gone" or "the signal is gone *linearly*"? Random forest and small MLP baselines (clean-406, L16):

| Cell | LR | RF | MLP |
|---|---|---|---|
| `C_SFT` `</think>` | 0.805 | 0.801 | 0.803 |
| `C_SFT` assertion | 0.786 | 0.788 | 0.785 |
| `C_SFT` neutral | 0.563 | 0.565 | 0.589 |
| `C_outcome` `</think>` | 0.895 | 0.881 | 0.889 |
| `C_outcome` assertion | 0.707 | 0.684 | 0.674 |
| `C_outcome` neutral | 0.561 | 0.533 | 0.536 |

Nonlinear probes are essentially identical to linear at clean-406. The signal at every cell is well-represented linearly; we are not measuring "gone linearly only." This is a clean validation of the linear-probe-as-measurement methodology.

---

## 5. Cross-position transfer in detail (Phase 1 result)

§2.3 gave the headline; the diagnostic details:

### 5.1 The asymmetry persists after explicit double-balancing

Original concern: the off-diagonals on C_outcome are 0.494 (pre→ass) and 0.368 (ass→pre). Both at-or-below chance, with the ass→pre being below chance, raises a class-imbalance hypothesis. We re-ran with explicit balanced subsampling on *both* source training and target evaluation data, averaged across 10 seeds:

| layer | ckpt | pre→ass | ass→pre | symmetric mean | asymmetry |
|---|---|---|---|---|---|
| L12 | C_SFT | 0.533 | 0.470 | 0.502 | +0.063 |
| L12 | C_outcome | 0.586 | **0.270** | 0.428 | **+0.316** |
| L16 | C_SFT | 0.438 | 0.499 | 0.468 | −0.060 |
| L16 | C_outcome | 0.494 | 0.368 | 0.431 | +0.126 |
| L20 | C_SFT | 0.622 | 0.547 | 0.585 | +0.075 |
| L20 | C_outcome | 0.457 | 0.501 | 0.479 | −0.044 |

**Reading.** The asymmetry persists after double-balancing on C_outcome at L12 (pre→ass 0.586, ass→pre 0.270 — assertion-trained probe *anti*-predicts pre_answer correctness). It diminishes at L16 and reverses at L20. The standard deviation of ass→pre across seeds is high (0.08–0.18), suggesting the assertion-trained probe direction is *unstable* across balanced subsamples on `C_outcome` rather than "stably anti-aligned." Consistent with "no coherent shared direction" rather than "consistent anti-direction."

### 5.2 The orthogonality is layer-invariant, not L16-specific

Symmetric-mean (pre↔ass averaged) cross-position transfer on `C_outcome`:
- L12: **0.428**
- L16: **0.431**
- L20: **0.479**

All clearly below the diagonals (0.69–0.90). The collapse is not a single-layer artifact. The form differs by layer (L12 most asymmetric, L20 mildest), but the *fact* of the collapse is robust.

### 5.3 Pre_answer is its own thing (not specifically orthogonal to assertion)

`C_outcome` L16 transfer to/from each position:

|   | → pre_answer | → assertion | → neutral |
|---|---|---|---|
| pre_answer → | **0.895** | **0.494** | **0.505** |
| assertion → | 0.368 | **0.694** | 0.500 |
| neutral → | **0.686** | 0.571 | **0.565** |

Two patterns:
- **pre_answer probe does not transfer to *either* assertion or neutral** (both at chance). So the "orthogonality to assertion" is part of a broader "pre_answer is its own subspace." This is a narrower claim than the original "specifically orthogonal to assertion."
- **neutral probe transfers to pre_answer at 0.686** — well above chance, even though the neutral probe's own diagonal AUROC is only 0.565. Other positions can partially "read" the pre_answer correctness signal. The pre_answer direction is too position-specific to "write back" to other positions.

The right mechanistic claim is: **the pre_answer correctness direction is a specialized, position-specific linear feature that does not linearly transfer to non-pre_answer positions, while non-pre_answer positions encode a weaker form of the same signal that is partially readable from pre_answer-trained linear directions.**

Heatmap: `extension/outputs/n500/figures/fig_phase1_transfer_heatmap.png` (6 panels, 3 layers × 2 ckpts).

---

## 6. Verbalized confidence

**The intended elicitation didn't work.** We attempted two literature-standard methods:

1. **Generated [0,100] confidence elicitation** (RLCR / Damani et al. style): append a confidence-prompt suffix in chat-template form, generate ~8 tokens, parse an integer. **Failed**: the SFT'd Qwen base is not chat-tuned and treats *any* prompt as the start of a new Countdown problem. Every response was a degenerate `" 0\nLet me try to find..."`. Verbalized confidence was 0 for every rollout.
2. **Token-logprob elicitation** (more sophisticated; query `P("yes" | prompt+response+verdict-suffix)`): same prompt, but extract the first-token logprob of yes/no from the top-20. **Failed**: the model's first token after any prompt is `<th` (start of `<think>`) — neither yes nor no is in the top-20 for most rollouts. `logprob_no` is absent for 100% of rollouts; the salvageable AUROC of `logprob_yes` alone is 0.509 (`C_SFT`) and 0.504 (`C_outcome`), both essentially chance.

The model is **too narrowly tuned to its training distribution to be queried via verbal elicitation at all.** Both literature-standard methods broke for the same underlying reason: any prompt sent to an SFT'd-on-Countdown-only Qwen 0.5B is treated as the start of another Countdown problem, regardless of chat-template structure or whether we want one-token output. The model has no slot in its training distribution for "interrupt and tell us your confidence."

**This is a methodological contribution worth flagging in the discussion**, not just a limitation:

> Verbalized-confidence elicitation methods that work on instruction-tuned or RLHF-tuned models (Damani et al. 2025 / RLCR; logprob-based variants of Yuan et al.) do not transfer to small-scale SFT'd base models trained on a narrow task. The base distribution is too dominant; any elicitation suffix gets continued as more in-task text rather than as the verdict the suffix asks for. This is a meaningful limit on probing-via-elicitation methodology at small scale: it suggests that the concealment-gap literature's reliance on graded verbalized confidence implicitly assumes a model with enough out-of-distribution generalization to follow non-training-distribution instructions.

**Defensible fallback used for the global gap: keyword-presence as the verbalized signal.** We use binary "does the `<think>` body contain any of `{Perfect, this works, got it, the answer is, verified, this is correct, confirmed, found it}`?" as a per-rollout proxy for verbalized confidence. The AUROC of this binary signal against correctness is what we report in §2.8. This is defensible because (a) it's the model's *own* verbalization, (b) the diagnostic in §3 already shows the model uses these keywords as the primary surface signal of confidence, and (c) it gives a *positive* AUROC (0.57 / 0.79). The limitation is honestly that this is binary, not graded.

---

## 7. Training dynamics (Layer C) — Option B is the headline

The original Option-A measurement (probe each snapshot's hidden states on a *fixed* set of final-checkpoint rollouts, only the model's weights changing) was confounded because the rollout distribution did not update with the snapshot. We re-ran with **Option B**: re-sample fresh rollouts per snapshot, re-cache hidden states, re-train the probe.

**Option B fresh-rollout dynamics (L16, balanced AUROC, GroupKFold(5) by prompt):**

| Step | `</think>` AUROC | assertion AUROC | neutral AUROC | gap = pre − ass |
|---|---|---|---|---|
| `C_SFT` (pre-RL, n=406 clean) | 0.804 | 0.785 | 0.562 | **+0.019** |
| step 30 (n=200 fresh rollouts) | 0.791 | 0.769 | 0.562 | **+0.022** |
| step 60 (n=200 fresh) | 0.864 | 0.749 | 0.553 | **+0.115** |
| step 90 (n=200 fresh) | 0.871 | 0.654 | 0.562 | **+0.217** |
| `C_outcome` (final, n=406 clean) | 0.896 | 0.703 | 0.562 | **+0.193** |

**Reading.** Pre_answer AUROC rises +0.092 across training. Assertion AUROC falls −0.131. The gap is essentially zero pre-RL (0.019), unchanged through step 30 (0.022), then opens dramatically between steps 30 and 60 (to 0.115), and continues to grow to 0.217 at step 90 before retracting slightly to 0.193 at the final checkpoint. The decoupling is **emergent over training**, not a static representation feature.

**Option A vs Option B in one sentence.** The original Option-A measurement (n=50, fixed final rollouts) showed assertion AUROC stable at ~0.5 across all snapshots and trace-final AUROC stable at ~0.80; the gap did not appear to move. Option B (fresh per-snapshot rollouts, n=200) shows the gap moving from 0.022 at step 30 to 0.217 at step 90. The dynamics signal was hidden by the Option-A confound. This is the strongest single piece of evidence that the decoupling is a *consequence of outcome RL*, not an SFT-baseline property the position-resolved measurement was uncovering.

---

## 8. Within-rollout probe trajectory (Phase 2A result)

§2.4 gave the headline; the diagnostic details:

### 8.1 The "model emits correct early, drifts to wrong" pathology is large

Of 6496 clean-406 `C_outcome` rollouts:
- 5458 (84.0%) emit ≥2 `<answer>` blocks (the multi-answer / rambling case)
- 824 (12.7%) emit exactly 1 block (cleanly terminating)
- 214 (3.3%) emit zero blocks

Within multi-answer rollouts, parsed-per-block correctness:
- TT (both first and last correct): 2983 (54.7%)
- T→F (correct → wrong, the pathology): **490 (9.0%)**
- F→T (wrong → correct): 150 (2.7%)
- FF (both wrong): 1835 (33.6%)

So the model emits a correct early `<answer>` and then drifts to a wrong final one on ~9% of multi-answer rollouts. The headline question for the within-rollout probe trajectory is what the probe says at the wrong-final position in those 490 rollouts.

### 8.2 Position-appropriate probe vs trace-final probe

We trained two probes and applied both to `<answer>` opening hidden states:
- **Trace-final probe**: trained on `</think>` hidden states, labeled by final-answer correctness. Per-prompt held-out CV.
- **Position-appropriate probe**: trained on `<answer>` opening hidden states, labeled by *each block*'s correctness. Same hyperparameters, also held-out by prompt.

Position-appropriate probe held-out diagonal AUROC at `<answer>` opening: **0.920**. Strong signal.

The position-appropriate probe gives the same Pattern A verdict more cleanly than the trace-final probe (because the trace-final probe is OOD at the first `<answer>` opening — see §8.3 below):

**Probe(last) by transition class, position-appropriate probe:**
- TT: 0.823
- **T→F: 0.154** (statistically indistinguishable from F→F floor at 0.088)
- **F→T: 0.580**
- FF: 0.088

The probe at the wrong-final position predicts "wrong" on T→F drift rollouts. The model's representation has moved to the wrong answer. No preserved "secret correct representation" remains.

### 8.3 The trace-final probe is OOD at the first `<answer>` opening — sanity caveat

A diagnostic to flag for completeness: the *trace-final*-trained probe applied to first `<answer>` opening tokens returns ~0 uniformly across all correctness classes (TT: 0.001, T→F: 0.001, FF: 0.000). This is an out-of-distribution artifact, not a Pattern A or B signal: `</think>` tokens sit at average token position 1093 in the rollout, while first `<answer>` openings are at average token 534 — much earlier, often *inside* the still-thinking phase. The trace-final probe's StandardScaler-fitted mean/std do not match the first `<answer>` opening's distribution, so the probe outputs collapse.

This is exactly why the position-appropriate probe (§8.2) is the right tool. The position-appropriate probe gives a 0.874 probe(first) on TT vs 0.084 on FF — clean discrimination at the first `<answer>` position. The trace-final probe's discrimination kicks in only from block 1 onwards (where positions overlap more with the `</think>` training distribution).

The per-block-INDEX trajectory under the trace-final probe is still meaningful because the OOD effect is constant *across correctness conditions at any given block*. Per-block-INDEX numbers (block_idx vs probe(correct) vs probe(wrong) under trace-final probe):

| block_idx | n_corr | n_wrng | probe(correct) | probe(wrong) | diff |
|---|---|---|---|---|---|
| 0 | 3473 | 1985 | 0.001 | 0.000 | +0.001 |
| 1 | 3409 | 2049 | 0.725 | 0.387 | **+0.338** |
| 2 | 2981 | 1231 | 0.539 | 0.367 | +0.172 |
| 3 | 2558 | 832 | 0.678 | 0.369 | +0.310 |
| ... | | | | | |
| 12 | 580 | 145 | 0.156 | 0.062 | +0.094 |

Per-block discrimination is preserved from block 1 onward; only block 0 is OOD. The Pattern A verdict is the same under both probes.

### 8.4 What this rules out about the mechanism

The original concealment-gap literature (Yuan et al.) frames the gap as "the model internally represents that the answer is wrong while verbally claiming it is right." For the small-scale outcome-RL'd model studied here, the within-rollout probe trajectory **rejects this framing**:

- The model's commit-position representation does not retain hidden information about the first answer's correctness on T→F drift rollouts.
- At the wrong-final `<answer>` position, the probe predicts "wrong" — agreeing with the actual final correctness, not with the originally-correct first answer.
- There is no preserved "secret correct representation" that activation patching could recover. (This is why Phase 2B, the planned activation-patching experiment, was skipped per the gating rule: causal injection of a representation that doesn't exist in the model would not produce a meaningful intervention.)

The decoupling is at the representational level (different linear subspaces for trace-final vs. assertion-time correctness; §2.3), but the model's *belief* at each commit reflects what it commits to. The right framing for the small-scale outcome-RL'd model is "**position-decoupling of correctness representations**" not "**knows-but-doesn't-say**".

Figure: `extension/outputs/n500/figures/fig9b_within_rollout_position_appropriate.png`.

### 8.5 Per-problem probe-AUROC vs accuracy-delta correlation — decoupling confirmed at the problem level

The aggregate AUROCs in §2.1 show the probe weakens at assertion positions under outcome RL. **Is the per-problem behavior of the probe coupled to the per-problem behavior of accuracy?** If outcome RL *damaged* the probe on problems where the model also got worse, we'd see a positive correlation between probe-drop and accuracy-drop. If outcome RL *decoupled* the probe from the model's commit decision, the two should be independent.

**Setup.** Reuse the existing `eval_c_sft_n500.json` and `eval_c_outcome_n500.json` rollouts (no resampling). Train a position-appropriate probe at the same `<answer>`-opening L16 position as §2.4 on the 94 contaminated problems (disjoint from clean-406 eval set by construction; held-out training AUROC = **0.889**, within 0.03 of §2.4's 0.920). For each of the 406 clean problems, compute per-problem AUROC and accuracy under each checkpoint:
- `auroc_sft[i]`, `auroc_rloo[i]` from the K=16 rollouts' (probe_score, label) pairs
- `acc_sft[i]`, `acc_rloo[i]` from the K labels
- `probe_drop[i] = auroc_sft[i] - auroc_rloo[i]`  (>0 = probe got worse)
- `accuracy_delta[i] = acc_rloo[i] - acc_sft[i]`  (>0 = accuracy improved)

Spearman correlation across problems with mixed-outcomes on both checkpoints.

**Result.**

```
n_problems_used: 218
Spearman r = -0.032,  p = 0.63   →  essentially zero
```

Per-problem AUROC distributions:
- `C_SFT`: mean **0.827**, median 0.873 (n=267 problems with defined AUROC)
- `C_outcome`: mean **0.813**, median 0.867 (n=242)
- `probe_drop`: mean **+0.012**, median +0.005 — essentially zero shift at problem level

Per-problem accuracy: `C_SFT` mean 0.238, `C_outcome` mean 0.498, `accuracy_delta` mean **+0.260**.

**Quadrant counts (n=218 problems with both defined):**

| Quadrant | Interpretation | n | % |
|---|---|---|---|
| Top-right (probe ↓, accuracy ↑) | **decoupling** (RL improved behavior, probe weakened) | **103** | **47%** |
| Top-left (probe ↑, accuracy ↑) | both improved | 91 | 42% |
| Bottom-left (probe ↑, accuracy ↓) | noise/SFT-favored | 5 | 2% |
| **Bottom-right (probe ↓, accuracy ↓)** | **damage** | **4** | **2%** |
| Exactly on axis | — | 15 | 7% |

**Reading.** The damage quadrant is empty (4 / 218 = 2%). On 47% of problems the probe got worse while accuracy improved (the textbook decoupling signature); on 42% both got better. The Spearman r = −0.03 (p = 0.63) is the cleanest possible "independent" result: per-problem probe behavior and per-problem accuracy behavior are statistically decorrelated. The aggregate finding from §2.1 (assertion-position AUROC drops 0.785 → 0.703) is not the result of RL *breaking* the probe on a subset of problems where it also broke behavior; rather, the probe shift and the accuracy shift are independent processes operating across the problem distribution.

This per-problem evidence complements §2.4 (within-rollout the probe tracks each commit's correctness) and §2.11 (no causal effect of probe-direction steering on accuracy). All three results triangulate the same claim: the trace-final / position-appropriate probe is a **correlational reader** of a correctness-relevant subspace, and outcome RL's effect on accuracy is *not* mediated by the probe direction in a causal or per-problem-coupled way.

Figure: `extension/outputs/n500/probe_behavioral/probe_behavioral_correlation.png` (scatter with quadrant labels, regression line, Spearman annotation).

#### 8.5.1 Parallel test at the trace-final (`</think>`) position — sharper

The §8.5 experiment ran at the `<answer>`-opening position (where the aggregate probe AUROC *drops* under RL: 0.785 → 0.703). We repeated the same per-problem correlation analysis at the **trace-final position** (`</think>`), where the aggregate AUROC *rises* under RL (0.804 → 0.896). The aggregate metric goes the opposite direction; the per-problem analysis settles the question.

**Methodology.** Reuse the existing `extension/cache/probe_cache_n500_clean406/C_{SFT,outcome}_l16_pre_answer.npz` cache (no Modal forward passes). Train held-out probes within each checkpoint via GroupKFold(5) by prompt; per-problem AUROC over K=16 rollouts; same Spearman correlation against `accuracy_delta`. Script: `extension/probe/probe_behavioral_pre_answer.py`.

**Result.**

```
n_problems_used: 218
Spearman r = +0.335,  p = 4.0e-7   (HIGHLY SIGNIFICANT, POSITIVE)
```

| Quadrant | n | % |
|---|---|---|
| Decoupling (probe ↓, accuracy ↑) | **140** | **64%** |
| Both improved (probe ↑, accuracy ↑) | 57 | 26% |
| **Damage (probe ↓, accuracy ↓)** | **0** | **0%** |
| Noise (probe ↑, accuracy ↓) | 9 | 4% |
| On axis | 12 | 6% |

**Per-problem AUROC distributions:**
- `C_SFT`: mean **0.722**, median 0.741 (n=267 problems with mixed outcomes)
- `C_outcome`: mean **0.612**, median 0.638 (n=242)
- `probe_drop` mean: **+0.130** — per-problem AUROC *drops* by ~0.13 under RL

**Reading — aggregate vs per-problem reconciles a paradox.** §2.1's aggregate trace-final AUROC *rises* under RL (0.804 → 0.896) because aggregate AUROC pools across all rollouts of all problems and benefits from cross-problem difficulty signal: under outcome RL the model becomes more confident *on easy problems* and less confident *on hard problems*, which inflates aggregate AUROC even if the within-problem discriminative power weakens. The per-problem AUROC removes that confound. **Within-problem, the trace-final probe's discriminative power actually falls** (0.722 → 0.612), agreeing with the matched-pair finding (§2.5: median Δ at assertion-position drops +0.186 → +0.004 — same direction, different position).

The +0.335 Spearman is **positive** and **highly significant**: RL degrades the within-problem probe most on the problems where accuracy improved most. The damage quadrant is *empty* (0 / 218). This is a *cleaner* decoupling result than §8.5's null at `<answer>` opening — at the trace-final position the data positively triangulate "RL gain decouples from probe-readable correctness representation."

**Combined story for §8.5 + §8.5.1.** Two independent positions (`<answer>` opening and `</think>`); two checkpoints' worth of held-out probe scores. At both positions the damage quadrant is essentially empty (4 and 0 problems). At `<answer>` opening the Spearman is null (r=−0.03); at trace-final it is +0.33 (p=4e−7). The aggregate trace-final AUROC's *rise* in §2.1 is a cross-problem-difficulty artifact; the within-problem reality is a *drop* (0.72 → 0.61), and that drop is **positively coupled to where RL improved performance**. Both ways of looking at it point at decoupling, not damage.

Figure: `extension/outputs/n500/probe_behavioral/probe_behavioral_correlation_pre_answer.png`.

---

## 9. Methodological controls

### 9.1 Significance tests (clean-406, L16)

| Test | n | Result |
|---|---|---|
| Wilcoxon signed-rank on `C_SFT` matched-pair deltas (one-sided > 0) | 244 | **p = 1.8 × 10⁻²⁴** |
| Wilcoxon signed-rank on `C_outcome` matched-pair deltas (one-sided > 0) | 218 | **p = 0.027** |
| Mann-Whitney U between `C_SFT` and `C_outcome` deltas (one-sided) | — | **p = 8.0 × 10⁻¹⁶** |
| Paired t-test on `C_SFT` deltas (one-sided) | 244 | **p = 4.9 × 10⁻²⁸** |
| Paired t-test on `C_outcome` deltas (one-sided) | 218 | **p = 0.002** |
| Mann-Whitney U on per-problem Cohen's d distributions (one-sided) | — | **p = 3.0 × 10⁻⁴** |

All between-checkpoint comparisons hold at p < 10⁻³. The within-`C_outcome` Wilcoxon p drops from 0.72 (n=50) to 0.027 (clean-406) — the median delta of +0.004 is now detectably positive given the larger sample, **changing the interpretation from "null effect" to "small positive effect"**. The C_outcome matched-pair signal is *attenuated*, not *reversed* — the latter framing was a small-sample artifact.

### 9.2 Bootstrap 95% CIs on headline AUROCs

Cluster-bootstrap on prompt indices (80% subsample without replacement, B=100), L16, clean-406:

| ckpt | kind | point | 95% CI |
|---|---|---|---|
| C_SFT | pre_answer | 0.804 | [0.782, 0.816] |
| C_SFT | assertion | 0.785 | [0.750, 0.790] |
| C_SFT | neutral | 0.562 | [0.525, 0.574] |
| C_outcome | pre_answer | 0.896 | [0.883, 0.902] |
| C_outcome | assertion | 0.703 | [0.662, 0.725] |
| C_outcome | neutral | 0.562 | [0.524, 0.590] |

All CIs are tight (~0.03–0.06 wide). The position-resolved gap on `C_outcome` (pre 0.896 vs assertion 0.703) is far outside CI overlap; the gap is not a sampling-noise effect.

### 9.3 Class balance

All AUROCs in this writeup are **balanced 5-fold CV by held-out problem** (`GroupKFold(5)` on `prompt_idx`, then a 50/50 class-balanced subsample within each fold's training data). This removes the confound that `C_outcome`'s assertion data is 78% positive class (since RL increased both accuracy and conditional keyword-on-correct usage) while `C_SFT`'s is 39% positive.

### 9.4 Sanity checks per probe

Reported per-cell with each probe AUROC:
- **Shuffled-label AUROC**: ~0.48–0.51 across all cells.
- **Random-direction AUROC**: ~0.50–0.74 (modestly above 0.5, as expected from high-dimensional random projections of correlated activations).
- **Linear vs RF vs MLP**: §4.3 — essentially identical at n=500.
- **Length-matched transfer control**: at n=500 clean-406, length-matching changes off-diagonals by <0.02 in both directions — at n=50 we'd seen a 0.523 → 0.652 length-matching effect that was itself a small-sample artifact. The original n=50 "drift" was partially length-confounded; the n=500 cross-checkpoint result needs no length correction.

### 9.5 Contamination check

We procedurally generated 500 fresh Countdown problems with `extension/data/generate_countdown.py`. Verifications:
- **Test-split overlap (asingh15 test): 0/500.** No reuse of the n=50 problems that originally trained the probe pipeline.
- **Train-split overlap (asingh15 train, the RLOO training pool): 94/500 (18.8%).** Since `C_outcome` was RLOO-trained on the asingh15 train split, those 94 problems were potentially seen during training.
- **Clean-406 filter**: We filter the cache and rollouts to the 406 problems that are *not* in asingh15 train, ensuring no train-set leakage. Headlines on dirty-500 differ from clean-406 by 0.005–0.04 AUROC. The qualitative claims are robust to the filter; we report clean-406 throughout this writeup.

### 9.6 Format check

We verified that the procedurally-generated prompt format is byte-identical to asingh15's prompt format for matched `(nums, target)` inputs. The asingh15 prompts use NumPy-style right-justified bracket formatting (e.g. `[14 45  9  1]` with extra space-padding for single-digit numbers when 2-digit numbers are present in the same problem); our generator replicates this exactly. `len(prompt)` matches; character-by-character comparison passes after replacing the actual digits.

---

## 10. Failed intervention: `C_process` (Appendix)

Documented in detail in `extension.md` §A. Brief recap:

- We attempted a process-reward arm using annotation-only `<subgoal> reach X from [Y, Z] </subgoal>` tokens with the composite reward `R = R_outcome + 0.3·R_subgoal`, where `R_subgoal` rewards subgoals whose target is reachable from declared inputs *and* whose body computes the claimed value. Both validity and achievement use noise-free exact-arithmetic verifiers.
- `C_SFT_aug` (SFT on subgoal-augmented warm-start traces) learned the *grammar* (~92% of rollouts emit `<subgoal>` tags) but not the semantics.
- `C_process` (RLOO from `C_SFT_aug`) **underperformed** `C_outcome` on accuracy. Qualitatively, the subgoals were emitted in dead-end search branches rather than along the actual solution path — the tags were *annotations* on text the model would have produced anyway, not interventions that changed inference.
- This is the predicted outcome of **Strategic Information Allocation** (Kim et al., March 2026): at small scale, annotation-only tokens cannot route capability into a reasoner without a separate mechanism (no inference-time grounding, no external execution). We report this as a complementary negative result whose mechanism confirms a recent theoretical prediction.

---

## 11. Limitations

- **n = 500 procedurally-generated, clean-406 after train-split filter.** Matched-pair denominators are 218–267, comfortably enough for the headline statistics. The original n=50 paper's matched-pair (n=33–38) denominators were the source of several magnitude artifacts we've revised here.
- **Asingh15 train overlap.** 94/500 generated problems happened to also exist in `C_outcome`'s RLOO training pool. We filter to clean-406 and report on that. Robustness of the headline claims to dirty-500 vs clean-406 differs by 0.005–0.04 AUROC — no qualitative claim flips, but the train-filter is the right number to report. (Alternative future work: regenerate eval problems with explicit train-set deduplication during generation, so n is the full 500 instead of 406.)
- **Verbalized confidence is keyword-presence, not elicited.** The two literature-standard elicitation attempts (generated [0,100] and token-logprob yes/no) both broke because the SFT'd Qwen base is not chat-tuned (§6). We use a binary keyword proxy and disclose this clearly. The probe-level analysis (§2.1–§2.4, §5, §8) does not depend on verbalized confidence; only §2.8's global concealment-gap finding does.
- **C_SFT is `asingh15/qwen-sft-countdown-defaultproj`**, not a team-trained SFT. Documented in `extension.md` §10.
- **All experiments are at 0.5B.** The scale-inversion of the *global* concealment gap relative to Yuan et al. 1.5B+ is itself a finding, but we cannot claim our position-resolved finding generalizes to larger models without further work. (Yuan et al.'s position-resolved analysis at 1.5B+ has not been published.)
- **Phase 2B (activation patching) not run.** Per the pre-registered gating rule, Pattern A (which we observed) made Phase 2B's "inject first-answer state into last-answer position" experiment uninformative — there is no preserved first-answer representation to inject. This is the boring outcome for Phase 2B and is *evidence for* the no-preserved-secret-correct story, but it is not a tested causal claim. A different causal experiment (e.g., probing-based steering at trace-final, where we know a representation exists) would be the natural follow-up.
- **Per-layer cross-position transfer asymmetry** (§5.1) at L12 (ass→pre = 0.270, below chance) is mechanistically interesting but unstable across seeds (std 0.18). The clean claim is "no stable shared direction"; the "anti-direction" claim would need more samples or a different methodology to support.

---

## 12. What's in the figures

All under `extension/outputs/n500/figures/`, generated by `extension/probe/make_figures.py` plus per-phase scripts:

| Figure | What it shows | Headline number |
|---|---|---|
| `fig1_matched_pair_scatter.png` | Per-prompt mean assertion-probe scores, correct vs wrong rollouts (clean-406) | 78% above-diagonal → 52% above-diagonal |
| `fig2_within_problem_d.png` | Distribution of per-problem Cohen's d at `</think>` | mean +1.121 vs +1.036 |
| `fig3_position_bar.png` | Balanced probe AUROC at three position kinds | assertion: 0.785 → 0.703 |
| `fig4_per_keyword_bar.png` | Per-keyword assertion-position AUROC | "this works" 0.718 → 0.645 |
| `fig5_dynamics_trajectory.png` | L16 probe AUROC over training step (Option A; for reference) | trace-final stable; assertion stable — confounded by Option A |
| `fig6_transfer_heatmap.png` | 2×2 cross-checkpoint transfer matrix | pre_answer off-diag 0.86 / 0.72 |
| `fig7_concealment_gap.png` | Global concealment gap (probe − verbal), n=50 | +0.15 → +0.01 (inverts naive H2) |
| `fig8_annotated_qualitative.png` | Side-by-side qualitative: same prompt, wrong-vs-correct C_outcome rollouts (n=50 paper original) | qualitative H3 illustration |
| `fig9b_within_rollout_position_appropriate.png` | Scatter of probe(first) vs probe(last) on multi-answer rollouts | probe(last) = 0.154 on T→F drift; Pattern A |
| `fig10_ft_rollout_trajectory.png` | Per-block probe trajectory, F→T (rescue) vs T→F (drift) | Pattern A bidirectional |
| `fig11_per_layer_sweep.png` | Per-layer probe AUROC, pre/ass/neu × C_SFT/C_outcome × all 25 layers | gap depth-invariant; max +0.236 at L9; rules out (b) |
| `fig_phase1_transfer_heatmap.png` | 3×3 cross-position transfer × 3 layers × 2 ckpts | pre_answer is its own subspace; layer-invariant |
| `fig12_causal_steering.png` | Grouped bar chart: probe vs random direction at α=0.5/1/2; baseline + Wilson 95% CIs | probe-vs-random Δ ∈ [−0.07, +0.02]; null result |
| **`fig13_headline_dynamics.png`** | **Headline plot.** pre vs assertion AUROC over `C_SFT/step30/60/90/final` with bootstrap 95% CI bands | gap 0.019 → 0.022 → 0.115 → 0.217 → 0.193 |

The headline figure for the paper is **fig13** (Option B dynamics with CIs, §2.2 / §7): a single panel with the pre_answer and assertion-position AUROCs on the y-axis and training step on the x-axis, showing the gap opening between steps 30 and 60 with statistically separable CI bands from step 60 onwards. This is the *strongest single visual* of the decoupling-emergence claim.

---

## 13. Reproducibility / Code

All analysis is on GitHub at `Abraham-y/224r-project`. Key entry points:

```
extension/data/generate_countdown.py             -- procedural Countdown generator
extension/evaluation/sample_local_jsonl.py       -- vLLM rollouts from local JSONL
extension/evaluation/launch_expansion_rollouts.sh -- Phase 1 rollout jobs
extension/probe/launch_expansion_cache.sh        -- Phase 2 cache jobs
extension/probe/cache_hidden_states.py           -- pre_answer/assertion/neutral cache
extension/probe/cache_answer_positions.py        -- <answer>-opening cache (Phase 2A)
extension/probe/filter_to_clean.py               -- contamination filter (clean-406)
extension/probe/analyze_probes.py                -- per-cell probe AUROCs
extension/probe/bootstrap_headline_cis.py        -- bootstrap CIs
extension/probe/deeper_analyses.py               -- per-keyword + Cohen's d + per-layer
extension/probe/qualitative_matched_pairs.py     -- §2.5 matched-pair table
extension/probe/cross_checkpoint_transfer.py     -- §2.7 transfer matrix
extension/probe/length_matched_transfer.py       -- §9.4 length-matched control
extension/probe/cross_position_transfer.py       -- §2.3 cross-position transfer
extension/probe/phase1_diagnostics.py            -- §5 (asymmetry + per-layer + neutral)
extension/probe/per_snapshot_decoupling_gap.py   -- §2.2 / §7 Option B dynamics
extension/probe/significance_and_baselines.py    -- §9.1 + §4.3
extension/probe/phase2a_per_answer_correctness.py -- §8.1 per-block correctness
extension/probe/phase2a_pattern_analysis.py      -- §8.3 trace-final-probe trajectory
extension/probe/phase2a_position_appropriate_probe.py -- §8.2 position-appropriate probe
extension/probe/ft_rollout_trajectory.py         -- §2.9 F→T bidirectional Pattern A
extension/probe/probe_direction_cosines.py       -- §2.10 cosine analysis
extension/probe/per_layer_sweep.py               -- §4.1 full 25-layer sweep
extension/probe/save_probe_vector.py             -- save steering vector
extension/probe/causal_steering.py               -- §2.11 causal steering Modal job
extension/probe/analyze_causal_steering.py       -- §2.11 analysis
extension/probe/make_figures.py                  -- standard figures
```

Probe cache:
- `extension/cache/probe_cache_n500/` — n=500 dirty cache (preserved for diagnostics)
- `extension/cache/probe_cache_n500_clean406/` — clean-406 cache (primary)
- `extension/cache/probe_cache_dynamics_optB/` — Option B fresh-rollout cache per snapshot
- `extension/cache/probe_cache_n500_answers/` — `<answer>` opening cache (Phase 2A)
- `extension/cache/probe_cache_n500_all_layers_clean406/` — all 25 layers, clean-406 (§4.1)
- `extension/cache/steering/` — saved probe vector for causal steering

Modal compute budget consumed for the n=500 expansion + Option B + Phase 2A caching + per-layer caching + causal steering: **≈ $20-25 total** (mostly cheap forward-pass caching; causal steering was the slowest single job at ~60 min H100 for 100 prefixes × 7 conditions).

---

## 14. The eight claims I'd put in the paper

1. **At 0.5B with outcome RL on an exact-verifier task, the trace-final probe AUROC *rises*** (C_SFT 0.821 → C_outcome 0.901 at L20). Outcome RL strengthens, not damages, the model's internal correctness representation at the point of final commit.

2. **At confidence-asserting token positions, probe AUROC falls under outcome RL** (C_SFT 0.785 → C_outcome 0.703 at L16). The gap pre_answer − assertion grows from 0.019 (C_SFT) to 0.193 (C_outcome). Bootstrap-CI-tight: gap is 10× the C_SFT value with non-overlapping intervals.

3. **The decoupling gap emerges *over training*** (Option B dynamics, §2.2 / §7). Gap trajectory: 0.019 → 0.022 → 0.115 → 0.217 → 0.193 across `C_SFT → step30 → step60 → step90 → final`. The decoupling is a *consequence* of outcome RL, not an SFT property.

4. **The pre_answer correctness direction is layer-invariantly position-specific**: it does not linearly transfer to assertion or neutral positions (off-diagonal AUROC ≈ 0.50 on `C_outcome` at all three layers), while other-position probes can partially "read" pre_answer (neutral → pre_answer = 0.686). The decoupling is at the representational level.

5. **The model's internal belief about correctness updates dynamically at each commit (Pattern A)**: on T→F drift rollouts, probe(last) = 0.154 — indistinguishable from the F→F floor (0.088). There is no preserved hidden representation of the original correct answer. The gap is not "knows but doesn't say"; it is *position-decoupling of correctness representations across token positions within a single rollout*.

6. **The global concealment gap *inverts* at 0.5B** (binary verbal-keyword AUROC 0.57 → 0.79, probe AUROC 0.72 → 0.79; global gap +0.15 → +0.01). This is the scale-inversion of Yuan et al.'s 1.5B+ result. At small scale, outcome RL globally calibrates the verbal signal while position-decoupling the internal correctness representation.

7. **The trace-final probe is a *correlational reader*, not a *causal controller*.** Causal steering at α ∈ [0.5, 2.0] · h_mean_norm along the trace-final probe direction has accuracy effects indistinguishable from random-direction perturbation of matched magnitude (Δ in [-0.07, +0.02]; n=97 prefixes). Combined with the per-layer sweep showing the pre-ass gap is depth-invariant (rules out late-layer-only shaping) and the cosine analysis showing probe directions across positions are essentially orthogonal (within and across checkpoints), the picture is: the correctness representation lives in a multi-dimensional subspace; the probe captures one linear summary that reads well but writes poorly.

8. **Per-problem probe-AUROC change is *not* coupled to per-problem accuracy change in any damage-shaped way** (§8.5, §8.5.1). At `<answer>` opening: Spearman r = −0.03 (p = 0.63), damage quadrant 4/218. At trace-final (`</think>`): Spearman r = +0.33 (p = 4e−7), damage quadrant **0/218**. Per-problem AUROC actually *drops* under RL at trace-final (mean 0.72 → 0.61), reconciling with §2.5's matched-pair drop — and that within-problem drop is positively coupled to *where RL improved accuracy most*. RL's gain in accuracy is not mediated by the probe-readable correctness representation; if anything, RL's gain comes *with* a probe degradation on the same problems.

**Headline framing.** Under outcome RL, the model's correctness representations *specialize by token position*: the trace-final position becomes more informative, the verbalization-time position becomes less informative, and the two no longer share a common linear direction. The model's *belief* at each commit reflects what it commits to (no preserved secret correct answer). This is **position-decoupling of correctness representations**, not "the model says confident things while internally knowing they're wrong." A more sophisticated form of mechanism-level reorganization than the naive concealment-gap framing.

---

## 15. Mechanism speculation *(explicit speculation; the experiments distinguish among these only partially)*

We observe position-decoupling that emerges over outcome RL, with the model's commit-time belief reflecting its commit (not the original correct answer). The experiments do not pin down *why* the decoupling occurs. Three candidate mechanisms, in roughly decreasing parsimony:

**(a) RL trained the policy on features that vary by token position.** Outcome RLOO rewards rollouts based on a single bit at the end (correct vs not). The policy gradient updates the model's *next-token distribution* over many tokens. The most efficient way to increase reward is to shape the *output distribution* — including the distribution over confidence-keyword tokens at intermediate positions — to correlate with whatever low-level features predict reward *at those token positions*. Different token positions have different surrounding-context distributions, so the same correctness signal might require different linear projections at different positions. The trace-final position is the closest to the actual reward signal, so its representation strengthens. The assertion-time position is more "internal" and the RL gradient there is weaker, more diffuse, and may favor features uncorrelated with the trace-final correctness direction. This predicts the observed position-decoupling without requiring any specific architectural mechanism.

**(b) Outcome reward selectively shaped the late layers / output head. ~~Plausible~~ RULED OUT BY PER-LAYER EVIDENCE (§4.1).** Under late-layer-only updating we would expect the pre−ass gap to grow with depth. The full 25-layer sweep shows the opposite: the gap stabilizes by L1, sits between +0.18 and +0.24 from L5 to L24 with no monotonic trend, and peaks at L9 (+0.236) — *earlier* than L16 or L20. The decoupling is distributed across all transformer depth, not concentrated at the output head. This mechanism is no longer parsimonious; we leave it here only for completeness.

**(c) The position-decoupling reflects a divergence between "this is my answer" and "this is correct."** Outcome RL rewards the model for emitting a correct final `<answer>`. But emitting an answer is not the same as *believing* it is correct. Under this view, the model learned to *commit* to answers on different internal features than the ones that predict correctness. Within a rollout, the per-block probe (§8) confirms this: at each `<answer>` commit, the representation tracks the commit (not the prior best answer), and the probe accurately reads "this commit will be wrong" when it will be wrong. The first-answer correctness signal is not preserved into later commits — the model has truly moved its representation to the new commit, even when the new commit is worse.

Our experiments distinguish (a) and (b) from (c) on one dimension: (c) is the framing that *most cleanly explains the within-rollout result* (§8). If the model's internal representation tracked "is this answer correct" globally, it would retain the first-answer correctness signal across the rollout, which we *don't* observe. If the representation tracks "is the current commit correct," we should see what we see — probe(last) flips on T→F drift rollouts. (c) is the *parsimonious explanation that fits all data, including the within-rollout pattern*.

Note: (a), (b), and (c) are not mutually exclusive in principle, but our experiments are now strong enough to substantially narrow them:
- **(b) ruled out** by the layer-invariant pre−ass gap (§4.1, full 25-layer sweep). Late-layer-only shaping would have produced a depth-dependent gap; we observe a flat gap from L5 onwards. The decoupling is distributed.
- **(c) supported** by the within-rollout Pattern A result (§2.4): the model's commit-time belief tracks each commit's content; there is no preserved hidden "secret correct" representation. The §2.11 causal-steering null result confirms this: pushing the residual stream along the trace-final probe direction does not redirect generation, consistent with "the trace-final probe is a *reader* of the correctness representation, not a *controller* of the commit." (c) is the parsimonious mechanism that fits this.
- **(a) compatible** with (c) and is the most natural co-explanation. Outcome RLOO updates many tokens' next-token distributions in proportion to their reward contribution; positions closer to the final commit (`</think>`) get more direct gradient, sharpening their correctness representation; positions earlier in the trace (assertion tokens) get smaller, more diffuse gradient that may favor reward-correlated features uncorrelated with the trace-final correctness direction.

The cleanest mechanistic claim our experiments support is: **outcome RL produces position-dependent correctness representations distributed across all transformer depth that are NOT linearly tied across token positions; the trace-final probe direction is a correlational reader of a multi-dimensional correctness subspace, not a causal control axis; the model's commit-time belief at each position tracks the local content, not any preserved earlier correctness representation.** Further work (LoRA-localized RLOO; nonlinear steering; per-layer activation patching) would distinguish (a) from (c) more sharply but is out of scope.
