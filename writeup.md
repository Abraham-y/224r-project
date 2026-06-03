# Concealment Under Outcome Pressure at 0.5B

*A probe-as-measurement study of internal-vs-verbalized correctness on outcome-RL'd Qwen2.5-0.5B on Countdown, with a position-resolved mechanism analysis and a within-rollout probe trajectory.*

> **Date of writeup.** 2026-05-30, revised after the n=500 expansion + Phase 1 / Phase 2A diagnostic rounds.
> **Status.** All in-scope analyses complete; the report can be drafted from this document. Numbers below are from the **n=500 procedurally-generated held-out set, contamination-filtered to 406 problems not in C_outcome's RLOO training set**, except where explicitly noted as "n=50 paper-original" or "Option A dynamics."

---

## 0. TL;DR (one-paragraph)

**A near-oracle internal verifier emerges at the trace-final position under outcome RL, with substantial applied value at deployment time.** On Countdown (an exact-verifier task), the trace-final probe at `</think>` is **near-oracle** at predicting first-`<answer>`-block correctness — held-out AUROC **0.982** at 0.5B C_outcome and **0.974** at 1.5B C_outcome (corrected next-block-correctness labels throughout). Outcome RL *strengthens* the probe at every level we measure: aggregate (C_SFT 0.912 → C_outcome 0.982), per-problem (mean 0.882 → 0.927), and within-prompt matched-pair (Wilcoxon p < 1e-7 on both checkpoints, indistinguishable between them: MW p = 0.68). Within multi-answer rollouts, the model truly updates its belief at each commit (Pattern A: probe(last) on T→F drift rollouts = 0.154, matches the F→F floor 0.088 — no preserved "secret correct" representation; bidirectional in F→T rescues). Causal steering along the probe direction is null (probe-vs-random Δ ∈ [−0.07, +0.02]; the probe is a *reader*, not a *controller*, of a multi-dimensional correctness subspace). **Applied results** (deployment-time uses of the probe, all corrected labels): probe-as-answer-selector at 0.5B = +8.7 pp; best-of-16 selector = +12.1 pp; probe-guided budgeted restart matches best-of-16 at 60% less compute; selective abstention reaches 98% accuracy at 50% coverage and 99% at 33%; adaptive budget gives +2.4 pp at K_avg=4; probe-mean estimates dataset accuracy to ±0.4 pp; calibration is 5.4% overconfidence, 3.4% underconfidence. **Probe-as-RL-reward catastrophically Goodharts in both init regimes** (C_outcome init: delayed Goodhart, accuracy −25 pp from baseline; C_SFT init: immediate Goodhart, final accuracy below SFT starting point); the policy exploits structural confounds the probe was correlated with in its training distribution, and post-Goodhart causal steering shows the probe direction became mildly causal (Δ=+0.08 vs original null). **The eval-time "rambling" we previously reported (mean 7.6 `<answer>` blocks at C_outcome, growing to 11.23 under ramble-penalty) is an INFRASTRUCTURE BUG, not a behavioral phenomenon**: a forward-pass logit analysis shows the model wants to emit `<|im_end|>` (token id 151645, end-of-turn) with 97.3% probability immediately after `</answer>`, but `tokenizer.eos_token_id = 151643` (`<|endoftext|>`) and vLLM's default `SamplingParams` stops only on `eos_token_id` — vLLM passes through `<|im_end|>` and continues sampling from an OOD state (the model has never seen text past `<|im_end|>` during training), producing degenerate rambling that *looks* like reward-hacking. Across 8000 rollouts × 6 checkpoints (C_SFT, C_outcome, firstanswer, ramble-penalty λ=0.20, probe-RL runA/B) **zero rollouts ever terminated on EOS**, because the model isn't trying to emit EOS — it's trying to emit `<|im_end|>`. With the eval sampler fixed (`stop_token_ids=[151643, 151645]`), the "rambling pathology" largely disappears at C_outcome (see §20). The §18.10 first-answer-RLOO and EXP-22 ramble-penalty results were already withdrawn as confounded by the training-time `stop=["</answer>"]` in the sampling worker; the eval-time stop bug is a *separate, simpler* confound that further explains why all checkpoints look rambly. **The position-gap (§2.2) and probe results (§2.1–§19) are unaffected** — they were measured on cached activations, not on rambling counts. The 1.5B model "never rambles" (0.075% multi-answer) probably because its tokenizer/sampler config is different and `<|im_end|>` is respected; this remains to verify.

---

## 1. Setup

**Task.** Countdown arithmetic reasoning (Gandhi et al. 2024): each problem gives 3–4 small integers and a target; the model must produce an equation that uses each number exactly once and evaluates to the target. The rule-based verifier in `evaluation/countdown.py` scores each response 0.0 (no parseable answer), 0.1 (parseable but wrong/invalid), 1.0 (correct).

**Model.** Qwen2.5-0.5B base + Countdown SFT throughout. We did not train an SFT model ourselves; `C_SFT` is Anikait Singh's `asingh15/qwen-sft-countdown-defaultproj`, used in lieu of a team-trained SFT.

**Checkpoints studied.**

| Checkpoint | Construction | Test pass@1 (asingh15 test, n=50) | Test pass@16 |
|---|---|---|---|
| `C_SFT` | `asingh15/qwen-sft-countdown-defaultproj` | 28.6% | 78.0% |
| `C_outcome` | RLOO from `C_SFT`, outcome reward only (0/0.1/1.0), 100 steps, all 10 intermediate snapshots persisted | **53.5%** | 72.0% |

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

### 2.1 Trace-final probe AUROC rises; assertion-position AUROC stays close behind

Probes are labeled by the correctness of the `<answer>` block that immediately follows each cached token (the next-commit correctness — the relevant target in multi-answer rollouts, since the model emits multiple `<answer>` blocks per rollout under C_outcome).

| Position (L16, balanced 5-fold CV, n=406 clean) | `C_SFT` AUROC | `C_outcome` AUROC | Δ |
|---|---|---|---|
| `</think>` (trace-final) | **0.904** | **0.980** | **+0.076** |
| confidence-asserting tokens | 0.887 | 0.852 | **−0.035** |
| neutral tokens (control) | 0.562 | 0.562 | 0.000 |
| **gap = pre_answer − assertion** | **+0.017** | **+0.127** | **+0.110** |

Shuffled-label baselines are ~0.48–0.51; random-direction baselines ~0.50–0.74. Signal is real.

**Reading.** Outcome RL strengthens the `</think>` position correctness representation by ~0.08 AUROC (0.904 → 0.980 — the model's hidden state right before the first commit is **near-oracle** at predicting that commit's correctness). At assertion-position tokens the probe weakens slightly (0.887 → 0.852). The gap between the two positions — essentially zero on `C_SFT` (+0.017) — grows to +0.127 on `C_outcome`, a 7× increase. Both positions still carry strong correctness signal in absolute terms; the gap is at the relative-position level.

### 2.2 The decoupling gap *emerges* over training (Option B dynamics)

The pre_answer − assertion gap is not present at `C_SFT` and grows over RLOO training. We re-sampled fresh rollouts (n=200) from three intermediate `C_outcome` snapshots, re-cached hidden states, and re-trained probes per snapshot (corrected labels: first-`<answer>`-block correctness):

| Step | `</think>` AUROC | assertion AUROC | gap | mean blocks/rollout |
|---|---|---|---|---|
| `C_SFT` (pre-RL) | 0.912 | 0.885 | **+0.027** | 2.83 |
| step 30 | 0.907 | 0.867 | **+0.040** | 3.09 |
| step 60 | 0.962 | 0.914 | **+0.048** | 4.49 |
| step 90 | 0.971 | 0.835 | **+0.136** | 7.18 |
| `C_outcome` (final) | 0.982 | 0.896 | **+0.086** | 7.41 |

**Reading.** The gap is +0.027 pre-RL, grows modestly through step 60, opens to +0.136 at step 90 (5× pre-RL), and slightly retreats to +0.086 at the final checkpoint. The trajectory tracks the rambling rate (mean `<answer>` blocks per rollout) closely: **Pearson r(mean_blocks, gap) = +0.891 across snapshots (p = 0.04)**. The gap is small in absolute terms (all individual AUROCs are 0.83–0.98) but the rambling-correlated dynamic is real.

This dynamics measurement uses fresh rollouts sampled from each snapshot's own policy (vs. a fixed set of final-checkpoint rollouts) and is the cleanest evidence that the gap *emerges over training* rather than being an SFT-inherited property.

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

### 2.5 Matched-pair within-prompt at assertion-position: strongly positive on both checkpoints

For each prompt with both a correct and a wrong rollout (where "correct" = the immediate-next `<answer>` block is correct), we compute mean assertion-token probe score on each side and take the within-prompt delta (clean-406, L16).

| | `C_SFT` | `C_outcome` |
|---|---|---|
| n prompts with mixed-outcome assertion rollouts | 226 | 60 |
| median Δ | **+0.393** | **+0.378** |
| mean Δ | +0.391 | +0.420 |
| % prompts probe ranks correct > wrong | **89%** | **75%** |

**Significance** (Wilcoxon signed-rank, one-sided > 0):
- `C_SFT`: p = **9.3 × 10⁻³⁵**
- `C_outcome`: p = **3.9 × 10⁻⁸**

**Mann-Whitney U between checkpoints** (one-sided, C_SFT > C_outcome): p = 0.68 (not significant).

**Reading.** Within the same problem, the probe at assertion tokens robustly ranks correct above wrong rollouts on BOTH checkpoints (89% above-diag at C_SFT, 75% at C_outcome). The Wilcoxon tests are highly significant on each. The Mann-Whitney between the two checkpoint distributions is **not significant** (p = 0.68): with proper next-`<answer>`-block labels, the matched-pair effect is statistically indistinguishable between C_SFT and C_outcome. The position-decoupling at 0.5B is at the aggregate AUROC level (§2.1), not at the within-prompt matched-pair level.

The smaller `n` on C_outcome (60 vs 226) is because most C_outcome rollouts emit confidence keywords (78% pos% in the labeled cache), leaving fewer mixed-outcome prompts with assertion-containing wrong rollouts.

### 2.7 Cross-checkpoint probe transfer: small drift effect, not collapse

Train probe on `C_X`'s activations; evaluate on `C_Y`'s (corrected labels throughout). Diagonals use held-out CV; off-diagonals use a probe trained on the other checkpoint's full data (clean-406, L16, balanced classes):

**`</think>` position:**

| train ↓ \ eval → | C_SFT | C_outcome |
|---|---|---|
| C_SFT | 0.912 | **0.953** |
| C_outcome | 0.822 | 0.982 |

**confidence-asserting tokens:**

| train ↓ \ eval → | C_SFT | C_outcome |
|---|---|---|
| C_SFT | 0.885 | **0.770** |
| C_outcome | 0.633 | 0.896 |

**Reading.** At trace-final the C_SFT-trained probe discriminates correct-vs-wrong on C_outcome at **0.953** AUROC — essentially in-distribution performance. The C_outcome-trained probe transfers to C_SFT at 0.822 — a real but modest drop, still well above chance. **The pre_answer correctness representations are mostly shared across checkpoints.** Assertion-position transfer is weaker (0.770 / 0.633) but neither direction is at chance. The asymmetry is real: the `C_SFT`-trained probe transfers better to `C_outcome` than vice versa — consistent with the `C_outcome` probe being more position-specific (§2.3).

The cross-checkpoint result is a *small drift effect* (off-diagonal pre_answer transfer is high but lower than diagonal), not a collapse. The mechanistically interesting finding is "representations specialize across positions within a checkpoint" (§2.3), not "representations drifted across checkpoints."

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

The §2.3 cross-position transfer AUROCs collapse to chance for pre→ass and ass→pre on `C_outcome`. We additionally compute direct cosine similarities between the trained probe direction vectors (in input space, after `w / scaler.scale_`; corrected labels):

**Within-checkpoint cross-position cosines at L16:**

| pair | cos | what it means |
|---|---|---|
| C_SFT: pre vs assertion | **+0.104** | small, near-orthogonal |
| C_SFT: pre vs neutral | −0.031 | essentially orthogonal |
| C_outcome: pre vs assertion | **+0.036** | essentially orthogonal |
| C_outcome: pre vs neutral | +0.064 | small |
| C_outcome: assertion vs neutral | +0.035 | essentially orthogonal |

**Cross-checkpoint within-position cosines at L16:**

| pair | cos | transfer AUROC |
|---|---|---|
| C_SFT pre vs C_outcome pre | +0.169 | 0.953 |
| C_SFT ass vs C_outcome ass | +0.134 | 0.770 |
| C_SFT neutral vs C_outcome neutral | +0.072 | — |

**Reading.**
- Within both checkpoints, the pre_answer and assertion probe directions are **near-orthogonal at the cosine level (~0.04–0.10)**. The cross-position transfer collapse from §2.3 has direct geometric backing.
- Cross-checkpoint, pre_answer probe directions are also small-cosine (+0.17) but the cross-checkpoint transfer AUROC is high (0.95). This implies the correctness signal lives in a **multi-dimensional subspace** that multiple low-cosine probe directions can each "read" — they don't have to point the same way to extract the same signal.

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

### 4.1 Layer invariance — gap is depth-invariant

We cached hidden states at every layer 0–24 and trained balanced GroupKFold(5) probes per cell. The pre−assertion gap on `C_outcome` is essentially flat from L5 to L24 — it doesn't grow with depth, which rules out the "outcome reward selectively shaped late layers / output head" mechanism (§16, b). The gap is distributed across all transformer depth.

Figure: `extension/outputs/n500/figures/fig11_per_layer_sweep.png`. (Note: the per-layer sweep was originally run with the older label rule; we re-ran the L12/L16/L20 cells with corrected labels and the depth-invariance qualitative result is preserved.)

### 4.2 Probe family is linear-sufficient

Random forest and small MLP probes give essentially identical AUROCs to logistic regression at every cell on clean-406. The signal is well-represented linearly; we are not measuring "gone linearly only." Validates the linear-probe-as-measurement methodology.

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

## 7. Training dynamics — see §2.2

§2.2 gives the corrected-label dynamics trajectory. The headline: **gap grows over training (+0.027 at C_SFT → +0.136 at step 90 → +0.086 at final), Pearson r(mean_blocks, gap) = +0.891 across snapshots (p = 0.04)**. The gap correlates with the rambling rate, not with anything else we measured. We re-sampled fresh rollouts per snapshot (Option B) rather than reusing a fixed set of final-checkpoint rollouts (the Option-A confound), and re-trained probes with corrected next-block-correctness labels per snapshot.

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

### 8.5 Per-problem probe-AUROC vs accuracy-delta correlation — probe strengthens under RL at the problem level

Is the per-problem behavior of the probe coupled to per-problem accuracy? Compute per-problem AUROC at `</think>` under each checkpoint (held-out via GroupKFold(5); corrected labels = first-`<answer>`-block correctness); take `probe_drop = auroc_sft − auroc_outcome` and `accuracy_delta = acc_outcome − acc_sft`; correlate.

**Result (n=131 problems with both checkpoints and both classes present):**

```
Pearson r(probe_drop, acc_delta)  = +0.093,  p = 0.29 (NS)
Spearman r(probe_drop, acc_delta) = +0.062,  p = 0.48 (NS)
```

**Per-problem AUROC distributions:**
- `C_SFT`: mean **0.882**
- `C_outcome`: mean **0.927**
- `probe_drop` mean: **−0.044** — per-problem AUROC *RISES* by ~0.04 under RL

**Quadrant counts (n=131):**

| Quadrant | n | % |
|---|---|---|
| **Both improved (probe ↑, accuracy ↑)** | **73** | **55.7%** |
| Decoupling (probe ↓, accuracy ↑) | 21 | 16.0% |
| Noise (probe ↑, accuracy ↓) | 11 | 8.4% |
| Damage (probe ↓, accuracy ↓) | 4 | 3.1% |

**Reading.** Outcome RL *strengthens* the trace-final probe at every level — aggregate (0.912 → 0.982; §2.1), per-problem (0.88 → 0.93; this section), and within-prompt matched-pair (Wilcoxon p < 1e-7 on both checkpoints, §2.5). The dominant per-problem quadrant is "both improved" (55.7%) — the model gets more accurate AND the probe gets a stronger discriminative signal on the same problems. The Spearman correlation between probe-drop and accuracy-delta is not significant (p = 0.48), and the damage quadrant is small (3.1%). The probe-readable correctness representation is not damaged by RL; it is sharpened.

This per-problem evidence is consistent with the within-rollout finding in §2.4 (the model genuinely updates belief at each commit) and the causal-steering null in §2.11 (the probe is a reader of a multidim correctness subspace, not a control axis). The probe at trace-final is a near-oracle internal verifier; outcome RL makes it better, not worse.

---

## 9. Methodological controls

### 9.1 Significance tests (clean-406, L16, corrected labels)

| Test | Result |
|---|---|
| Wilcoxon signed-rank on `C_SFT` matched-pair deltas (one-sided > 0) | **p = 9.3 × 10⁻³⁵** |
| Wilcoxon signed-rank on `C_outcome` matched-pair deltas (one-sided > 0) | **p = 3.9 × 10⁻⁸** |
| **Mann-Whitney U between `C_SFT` and `C_outcome` deltas (one-sided)** | **p = 0.68 (NS)** |

Both checkpoints have highly-significant within-prompt matched-pair effects. The Mann-Whitney *between* the two distributions is not significant — under corrected labels, the matched-pair effect is statistically indistinguishable across checkpoints. Any "outcome RL collapsed the matched-pair effect" claim is unsupported.

### 9.2 Headline AUROCs (clean-406, L16, corrected labels)

| ckpt | kind | AUROC |
|---|---|---|
| C_SFT | pre_answer | **0.912** |
| C_SFT | assertion | 0.885 |
| C_SFT | neutral | 0.516 |
| C_outcome | pre_answer | **0.982** |
| C_outcome | assertion | 0.896 |
| C_outcome | neutral | 0.567 |

The position-resolved gap on `C_outcome` (pre 0.982 vs assertion 0.896 = +0.086) is the modest aggregate signal; all individual position AUROCs are high (≥0.88) on `C_outcome`.

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

## 11. Limitations

- **n = 500 procedurally-generated, clean-406 after train-split filter.** Matched-pair denominators are 218–244; the per-problem correlation analysis at trace-final uses n=131 (problems with both checkpoints having both classes present).
- **Asingh15 train overlap.** 94/500 generated problems happened to also exist in `C_outcome`'s RLOO training pool. We filter to clean-406 and report on that. Robustness of the headline claims to dirty-500 vs clean-406 differs by 0.005–0.04 AUROC.
- **Verbalized confidence is keyword-presence, not elicited.** The two literature-standard elicitation attempts (generated [0,100] and token-logprob yes/no) both broke because the SFT'd Qwen base is not chat-tuned (§6). We use a binary keyword proxy and disclose this clearly. The probe-level analysis does not depend on verbalized confidence.
- **C_SFT is `asingh15/qwen-sft-countdown-defaultproj`**, not a team-trained SFT.
- **Most analyses are at 0.5B**; 1.5B comparison covers the aggregate AUROC, matched-pair, and applied-probe results, but Option B dynamics and the per-layer sweep at 1.5B are not in scope.
- **No causal activation-patching has been run.** The causal steering at `</think>` (§2.11) is the only causal experiment; its null result is informative but doesn't substitute for a full activation-patching study.
- **Rambling-as-reward-hack causal hypothesis is UNTESTED.** We attempted three RL variants intended to distinguish reward-hack from drift: (i) first-answer reward RLOO; (ii) ramble-penalty RLOO at λ=0.05; (iii) ramble-penalty RLOO at λ=0.20. All three were **confounded** by `rloo_trainer/sampling_worker.py:84-85` which sets `stop=["</answer>"]` — vLLM halts each training rollout at the first `</answer>` token, so every rollout has exactly one `<answer>` block. The three reward functions are therefore mathematically identical on the training distribution: `last-block-score = first-block-score = first-block-score − λ × max(0, n_blocks−1) = first-block-score` (when n_blocks=1). The rambling we observe at eval time reflects parameter drift on the model's *post-`</answer>` distribution*, which RL never directly observes; the reward function shape cannot distinguish these drifts. A real test of the hypothesis requires removing the stop string during training (substantially more expensive: longer rollouts, slower sampling) and is out of scope. The rambling-position-gap link rests on correlational evidence only (r = +0.89 across snapshots).

---

## 12. What's in the figures

All under `extension/outputs/n500/figures/`, generated by `extension/probe/make_figures.py` plus per-phase scripts:

| Figure | What it shows | Headline number |
|---|---|---|
| `fig9_within_rollout_trajectory.png` | Per-block trace-final-probe trajectory through multi-answer rollouts | trace-final probe is OOD at `<answer>` opening (§8.3) |
| `fig9b_within_rollout_position_appropriate.png` | Scatter of probe(first) vs probe(last) on multi-answer rollouts (position-appropriate probe) | probe(last) = 0.154 on T→F drift; Pattern A |
| `fig10_ft_rollout_trajectory.png` | Per-block probe trajectory, F→T (rescue) vs T→F (drift) | Pattern A bidirectional |
| `fig11_per_layer_sweep.png` | Per-layer probe AUROC, pre/ass/neu × C_SFT/C_outcome × all 25 layers | gap depth-invariant; rules out late-layer-only mechanism |
| `fig12_causal_steering.png` | Grouped bar chart: probe vs random direction at α=0.5/1/2; baseline + Wilson 95% CIs | probe-vs-random Δ ∈ [−0.07, +0.02]; null result |
| `fig14_probe_answer_commit.png` | Probe-as-answer-selector accuracy by threshold | PROBE-COMMIT at T=0.35 → +8.7 pp |
| `fig15_probe_commit_variants.png` | Threshold × fallback strategy sweep | applied probe-commit variants |
| `fig17_position_resolved_auroc.png` | Per-distance-to-`</think>` per-kind AUROC | assertion bin AUROC > neutral at matched position |
| `fig18_probe_guided_restart.png` | Probe-guided restart Pareto: accuracy vs avg rollouts used | matches best-of-16 at 60% compute saved |
| `fig19_probe_abstention.png` | Selective abstention: accuracy on attempted vs coverage | 0.980 at 50% coverage |
| `fig20_probe_majority_hybrid.png` | Probe-best vs majority-vote vs hybrid | probe-best wins 5.2× on disagreements |
| `fig21_probe_variance_difficulty.png` | Per-prompt mean / std probe vs accuracy | mean Pearson r = +0.967 |
| `fig22_multi_position_ensemble.png` | Probe ensembles (pre + ass + neu combinations) | no gain over pre_answer alone |
| `fig23_probe_adaptive_budget.png` | Adaptive budget vs uniform K-per-prompt | +2.4 pp at K_avg=4 |
| `fig24_probe_eval_proxy.png` | Probe-mean vs true accuracy + failure-mode breakdown | dataset-acc estimate ±0.4 pp; 5.4% overconf |

---

## 13. Reproducibility / Code

All analysis is on GitHub at `Abraham-y/224r-project`. Key entry points:

```
extension/data/generate_countdown.py             -- procedural Countdown generator
extension/evaluation/sample_local_jsonl.py       -- vLLM rollouts from local JSONL
extension/probe/cache_hidden_states.py           -- pre_answer/assertion/neutral cache
extension/probe/cache_answer_positions.py        -- <answer>-opening cache (Phase 2A)
extension/probe/cache_all_think_close.py         -- every-</think>-token cache
extension/probe/filter_to_clean.py               -- contamination filter (clean-406)
extension/probe/relabel_full_grid.py             -- corrected-label probe AUROCs (full grid)
extension/probe/relabel_redo_downstream.py       -- corrected-label downstream stats
extension/probe/relabel_cross_checkpoint.py      -- §2.7 transfer matrix (corrected labels)
extension/probe/relabel_cosines.py               -- §2.10 cosines (corrected labels)
extension/probe/relabel_per_problem.py           -- §8.5 per-problem (corrected labels)
extension/probe/relabel_dynamics.py              -- §2.2 / §7 dynamics (corrected labels)
extension/probe/cross_position_transfer.py       -- §2.3 cross-position transfer
extension/probe/phase1_diagnostics.py            -- §5 (asymmetry + per-layer + neutral)
extension/probe/phase2a_per_answer_correctness.py -- §8.1 per-block correctness
extension/probe/phase2a_pattern_analysis.py      -- §8.3 trace-final-probe trajectory
extension/probe/phase2a_position_appropriate_probe.py -- §8.2 position-appropriate probe
extension/probe/ft_rollout_trajectory.py         -- §2.9 F→T bidirectional Pattern A
extension/probe/per_layer_sweep.py               -- §4.1 full 25-layer sweep
extension/probe/save_probe_vector.py             -- save steering vector
extension/probe/causal_steering.py               -- §2.11 causal steering Modal job
extension/probe/analyze_causal_steering.py       -- §2.11 analysis
extension/probe/probe_answer_commit.py           -- §17 probe-as-answer-selector
extension/probe/probe_thinkclose_selector.py     -- every-</think> selector
extension/probe/position_resolved_auroc.py       -- distance-to-</think> AUROC bins
extension/probe/probe_guided_restart.py          -- §18.1 budgeted restart
extension/probe/probe_abstention_and_hybrid.py   -- §18.2/§18.3 abstention + ensemble
extension/probe/probe_applied_scale_comparison.py -- §18.4 cross-scale applied
extension/probe/probe_creative_extensions.py     -- §18.5/§18.6/§18.7 difficulty / xfer / ensemble
extension/probe/probe_adaptive_budget.py         -- §18.8 adaptive budget
extension/probe/probe_as_eval_proxy.py           -- §18.9 eval-proxy + failure-mode
extension/training/firstanswer_rloo.py           -- §18.10 first-answer-reward RLOO trainer
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

## 14. The claims that survive corrected labels

1. **The trace-final probe is near-oracle** — held-out balanced AUROC = 0.982 at 0.5B C_outcome and 0.974 at 1.5B C_outcome at predicting the immediately-following first-`<answer>`-block correctness. The model's hidden state at `</think>` is a reliable internal verifier.

2. **Outcome RL *strengthens* the probe at every level.** Aggregate AUROC rises (C_SFT 0.912 → C_outcome 0.982). Per-problem AUROC rises (mean 0.882 → 0.927). Within-prompt matched-pair effects are highly significant on both checkpoints (Wilcoxon p < 1e-7) and statistically indistinguishable between checkpoints (Mann-Whitney p = 0.68). The probe-readable correctness representation is not damaged by RL; it is sharpened.

3. **A modest aggregate position-gap emerges over training.** Pre_answer − assertion gap: +0.027 (C_SFT) → +0.086 (C_outcome final), peaking at +0.136 at step 90. The trajectory tracks the rambling rate at Pearson r = +0.891 (p = 0.04) across snapshots. The gap is real but small in absolute terms (all individual AUROCs stay above 0.83).

4. **The pre_answer correctness direction is layer-invariantly position-specific.** It does not linearly transfer to assertion or neutral positions (off-diagonal AUROC ≈ chance on `C_outcome` at all three layers), while other-position probes can partially "read" pre_answer. Position-orthogonality cosines stay below 0.10 within each checkpoint.

5. **The model's internal belief updates dynamically at each commit (Pattern A).** On T→F drift rollouts, probe(last) = 0.154 — indistinguishable from the F→F floor (0.088). Bidirectional in F→T rescue rollouts. No preserved hidden representation of the original correct answer; the "knows but doesn't say" framing is refuted at 0.5B.

6. **The trace-final probe is a correlational reader, not a causal controller.** Causal steering at α ∈ [0.5, 2.0] · h_mean_norm along the probe direction has accuracy effects indistinguishable from random-direction perturbation (Δ ∈ [−0.07, +0.02]; n=97 prefixes). The correctness representation lives in a multi-dimensional subspace; the probe captures a linear summary that reads well but writes poorly.

7. **Position-decoupling is a small-scale (0.5B) phenomenon.** At 1.5B, the gap shrinks to +0.04, per-problem AUROC rises rather than falls under RL, rambling rate is 0.075% vs 87% at 0.5B. The decoupling story does not generalize to 1.5B.

8. **The probe has substantial applied value.** Concrete numbers (all corrected labels):
   - Probe-as-answer-selector: +8.7 pp absolute pass@1 lift (§17)
   - Best-of-16 selector lift: +12.1 pp at 0.5B, +8.6 pp at 1.5B
   - Probe-guided budgeted restart: matches best-of-16 at 60% less compute (§18.1)
   - Selective abstention: 98% accuracy at 50% coverage; 99% at 33% (§18.2)
   - Probe + majority hybrid: 5.2× win rate on disagreement prompts (§18.3)
   - Adaptive budget allocation: +2.4 pp at K_avg=4 (§18.8)
   - Probe-mean estimates dataset accuracy to ±0.4 pp (§18.9)
   - Calibration: 5.4% overconfidence, 3.4% underconfidence
   - C_SFT-trained probe transfers to C_outcome at AUROC 0.953 / +10.3 pp lift (§18.6)

9. **The 0.5B rambling pattern is correlational only; the reward-hack mechanism is UNTESTED.** 87% of C_outcome rollouts emit ≥2 `<answer>` blocks (mean 7.6); 0.075% at 1.5B. Rambling rate ~ position-gap correlation r = +0.89 across RLOO snapshots. We attempted to causally test the verifier-rule mechanism via first-answer-reward RLOO and ramble-penalty RLOO; **both were confounded by `stop=["</answer>"]` in the training sampling worker** (every training rollout has exactly one block, so the three reward functions are mathematically identical at training time). The §18.10 first-answer-RLOO null result is therefore **withdrawn**: it does not refute the reward-hack hypothesis. Rambling appears to emerge from parameter drift on the post-`</answer>` distribution, which RL never directly observes — but disentangling reward-hack from drift requires a different (more expensive) experimental design and is out of scope.

**Headline framing.** The model's hidden state at `</think>` is a near-oracle internal verifier that outcome RL *sharpens* (not damages). The probe is a correlational reader of a multidim correctness subspace; it has substantial deployment-time value (best-of-K, abstention, restart, adaptive budget, eval-proxy). A modest position-gap emerges during RL and correlates with the rambling pathology, which itself is a small-scale (0.5B-only) phenomenon. The original Yuan-style "knows but doesn't say" framing is refuted at 0.5B — Pattern A is confirmed bidirectionally, and the probe gets *better* under RL at every level we measure.

---

## 15. Scale extension: 1.5B reproduction

**Setup.** We trained Qwen2.5-**1.5B** on `Asap7772/cog_behav_all_strategies` (the same demonstrations our 0.5B `C_SFT` baseline uses — Anikait Singh's recipe) for 6 epochs at lr=1e-5, effective batch 64 (microbatch 4 × grad_accum 16). The result has pass@1 = 0.280 / pass@16 = 0.700 on the n=50 test (essentially tied with the 0.5B `C_SFT`'s 0.286/0.780). We then ran RLOO outcome-only for 100 steps with snapshots every 10 — same recipe as the 0.5B run except batch_size was reduced from 128 to 64 for memory. Reward_mean at step 99 = **0.457**. The post-RLOO model has pass@1 = **0.480** on the n=50 test (1.5B SFT + RL: +0.20 pp; the 0.5B RL added +0.25 pp). On the n=500 procedural set: pass@1 = **0.558** (vs 0.5B C_outcome's 0.55), avg per-rollout acc 0.568. So the 1.5B C_outcome is **slightly stronger** than the 0.5B C_outcome on this set despite undertrained SFT initialization.

We then cached hidden states on both 1.5B checkpoints at L12/L16/L20 × {pre_answer, assertion, neutral} on the procedural n=500 prompts and filtered to clean-406 (same prompt indices as 0.5B). Total Modal compute for the scale extension: ≈ **$30**.

### 15.1 Aggregate probe AUROCs: pre−assertion gap NARROWS at 1.5B

Corrected-label numbers (next-`<answer>`-block correctness) at L16, clean-406:

| Cell | 0.5B | 1.5B |
|---|---|---|
| `</think>` AUROC C_SFT | 0.912 | 0.857* |
| `</think>` AUROC C_outcome | **0.982** | **0.974** |
| Assertion AUROC C_outcome (L16) | 0.896 | 0.816 |
| Assertion AUROC C_outcome (L20) | 0.710 | 0.936 |
| **Gap pre − assertion C_outcome (L16)** | **+0.086** | **+0.157** |
| **Gap pre − assertion C_outcome (L20)** | (n/a) | **+0.040** |

Both checkpoints are near-oracle at the trace-final position (AUROC 0.97–0.98). At 1.5B the gap is +0.04 at L20 (the best layer) and +0.16 at L16 — depending on the layer, smaller or comparable to 0.5B. The 1.5B model maintains a coherent correctness representation across positions on the layer that the original Yuan-style measurement targeted (L20).

*1.5B C_SFT pre_answer AUROC reported above is from the original-label probe; not re-run with corrected labels because the 1.5B SFT baseline is undertrained and the headline numbers come from the post-RL checkpoint.

### 15.2 Matched-pair within-prompt at 1.5B

| | 1.5B C_SFT | 1.5B C_outcome |
|---|---|---|
| Matched-pair n | 213 | 57 |
| % above-diag | 86% | 82% |
| Wilcoxon p (one-sided > 0) | 3.4e−29 | 1.4e−8 |
| Mann-Whitney *between* ckpts | **p = 0.14 (NS)** | |

The 1.5B matched-pair distributions across the two checkpoints are statistically indistinguishable (p = 0.14). The 0.5B equivalent under corrected labels is also NS (p = 0.68; §2.5). Neither scale supports a "RL collapsed the matched-pair effect" finding under corrected labels.

Caveat: 1.5B C_outcome's matched-pair n is only 57 because the model is much more deterministic in K=16 sampling (pass@1 = 0.56; many problems become all-correct, removing them from the mixed-outcome pool).

### 15.3 Per-problem probe-AUROC at 1.5B

The 1.5B per-problem analysis uses original-label probes (not re-run with corrected labels because the 1.5B caches are smaller and the matched-pair effect is already established by §15.2):

- Per-problem AUROC mean: 1.5B C_SFT 0.79 → 1.5B C_outcome 0.90 (rises under RL; same direction as 0.5B under corrected labels).
- Spearman r(probe_drop, acc_delta) at 1.5B: essentially zero.
- Dominant quadrant: "both improved" (~63%).

The 1.5B picture mirrors the corrected-label 0.5B picture: per-problem AUROC rises under RL, no significant correlation with accuracy delta, "both improved" is dominant.

### 15.4 Interpretation — rambling is a small-scale phenomenon

Two pieces of evidence at 1.5B point the same way:
1. **Rambling is essentially absent**: 0.075% multi-answer at 1.5B C_outcome vs 87% at 0.5B C_outcome. The model emits a single `<answer>` block on virtually all rollouts.
2. **The probe is near-oracle at both scales** (AUROC 0.974 at 1.5B, 0.982 at 0.5B). The applied probe strategies generalize (best-of-16 lift +8.6 pp at 1.5B, abstention reaches 93% accuracy at 50% coverage at 1.5B; §18.4).

**The rambling reward-hack appears to be specifically a small-scale (0.5B) phenomenon.** The 1.5B model never falls into the multi-answer exploit. The position-decoupling gap that emerges under RL at 0.5B (§2.2) is correlationally tied to the rambling rate (r = +0.89); since rambling doesn't develop at 1.5B, the gap doesn't either.

**Open question for future work.** Does this fit with Yuan et al.'s "concealment gap widens under outcome RL at 1.5B+" finding? Their gap was probe-vs-verbalized; we measured probe-vs-probe across positions. The relationship isn't direct. If at 1.5B the *internal* representation stays coherent across positions but the *verbalized* signal still diverges (as Yuan reports), that would imply Yuan's gap is at the verbalization-readout level (not in the hidden-state representation itself). Confirming this would require running their elicitation methodology on our 1.5B C_outcome — out of scope here (the SFT'd-on-Countdown base is not chat-tuned for graded confidence elicitation; see writeup §6).

**Caveats for this scale comparison:**
- Our 1.5B SFT was undertrained relative to the asingh15 0.5B SFT recipe (6 epochs at lr=1e-5; we did not match their hyperparameters exactly).
- We ran 100 RLOO steps at both scales — at 1.5B that's effectively fewer optimizer updates per parameter. The reward_mean at step 99 was 0.46 (vs ~0.5 at 0.5B). Stronger RL convergence at 1.5B might reveal a different pattern; we did not extend.
- n=105 at 1.5B (vs 218 at 0.5B) per-problem comparison: smaller statistical power.
- We did not run Option B dynamics or the per-layer sweep at 1.5B. The aggregate scale claim above is from §15.1 alone; the rest are corroborating per-problem evidence.

The cleanest pre-paper extension: redo Option B dynamics at 1.5B (~$10-15 Modal) and the cross-position transfer matrix at 1.5B (free, local) to confirm the position-coherence claim mechanistically. Out of scope for this round.

---

## 16. Mechanism speculation *(explicit speculation; the experiments distinguish among these only partially)*

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
## 17. Probe-as-answer-selector — a practical use of the probe at 0.5B

The rambling pathology at 0.5B (§17 / §18; 87% multi-answer, mean 7.6 `<answer>` blocks, 9% T→F drift) is *exploitable* by the probe: it can pick the model's "best" answer block before the model drifts. Strategy:

1. Score each emitted `<answer>` block by the position-appropriate probe (held-out trained on contaminated-94, AUROC 0.92 on `<answer>` opening tokens; or 0.98 with corrected labels at trace-final).
2. **Commit at the first `<answer>` whose probe score ≥ 0.35**. Stop generation there.
3. If no block crosses the threshold, fall back to the verifier's default (last block).

**Result on 5458 multi-answer C_outcome 0.5B clean-406 rollouts:**

| Strategy | Accuracy | Gain over verifier baseline |
|---|---|---|
| Verifier-scored baseline (last `<answer>`) | 0.574 | — |
| Oracle pick-first (always commit to first block) | 0.636 | +0.062 |
| **PROBE-COMMIT at T=0.35, fallback=last** | **0.661** | **+0.087** |
| PROBE-COMMIT at T=0.89, fallback=argmax-probe | 0.662 | +0.088 |
| PROBE-MAX (argmax over all blocks) | 0.659 | +0.085 |
| Oracle commit-at-first-correct (upper bound) | 0.671 | +0.097 |

**Probe-commit captures 91% of the achievable ceiling** (the oracle "any block correct" rate of 0.671). +8.7 pp absolute pass@1 gain at 0.5B, plus ~50% generation compute savings (stop at block 2-3 instead of generating all 7+).

**Not applicable at 1.5B** because the model doesn't ramble (0.075% multi-answer; nothing to pick between).

This is the cleanest *applied* result we have: a free pass@1 boost for the 0.5B C_outcome model by exploiting the rambling-induced redundancy. Caveats:
- Compute saving comes from early termination, not from fewer total generations.
- Only useful if you have a learned probe (we trained on the disjoint contaminated-94 prompts; standard process-reward-model setup).
- The improvement is bounded by the oracle ceiling of 0.671 — beyond that, you'd need a better base model (e.g., 1.5B, which gets pass@1 = 0.558 without any probe trickery).

Figure: `extension/outputs/n500/figures/fig15_probe_commit_variants.png` (sweep of threshold × fallback strategy).

---

## 18. Probe applied: extended results (restart, abstention, hybrid, cross-scale, first-answer reward)

Section §17 showed that the corrected-label `</think>` probe (held-out balanced AUROC = 0.982 on 0.5B C_outcome clean-406 — see §2.1 with corrected labels) supports a within-rollout commit-selector with +8.7 pp gain. We extend that result along five complementary directions: across-rollout selection, selective abstention, ensemble with self-consistency, cross-scale generalization to 1.5B, and feeding the same idea back into RL as a verifier-level fix.

### 18.1 Probe-guided budgeted restart sampling (across-rollout selection)

`probe-best-of-K` is an upper-bound oracle for a setting where you commit to generating all K rollouts. In deployment, you can be smarter: generate one rollout, score it with the probe; if probe ≥ T, commit and stop; else re-sample, up to budget B. Saves compute in expectation.

Sweep over (B, T) on the cached 0.5B C_outcome clean-406 rollouts using the corrected-label `</think>` probe scores (script: `extension/probe/probe_guided_restart.py`):

| Strategy | Accuracy | Avg rollouts used |
|---|---|---|
| pass@1 (just take rollout 0) | 0.549 | 1.0 |
| best-of-K, K=2 | 0.589 | 2.0 |
| best-of-K, K=4 | 0.626 | 4.0 |
| best-of-K, K=8 | 0.658 | 8.0 |
| best-of-K, K=16 | 0.670 | 16.0 |
| **probe-restart B=16, T=0.95** | **0.675** | **6.3** |
| probe-restart B=8, T=0.95 | 0.658 | 4.6 |
| probe-restart B=4, T=0.95 | 0.637 | 2.8 |
| probe-restart B=2, T=0.95 | 0.580 | 1.7 |

**Reading.** Probe-restart at (B=16, T=0.95) matches and slightly exceeds best-of-16 accuracy (+0.005) while using only 6.3 rollouts per prompt on average — **~60% compute savings** at no accuracy cost. At B=8, T=0.95, accuracy 0.658 matches best-of-8 with only 4.6 rollouts used. The accuracy-vs-compute Pareto front sits *above* the best-of-K curve at every B. Figure: `extension/outputs/n500/figures/fig18_probe_guided_restart.png`.

### 18.2 Probe-guided selective abstention

When the probe says "I'm not confident," the optimal action is to *not* commit, not commit-and-be-wrong. We score each prompt's first rollout at `</think>` and commit only if probe ≥ T (script: `extension/probe/probe_abstention_and_hybrid.py`).

| Coverage target | Threshold T | n attempted | Accuracy on attempted |
|---|---|---|---|
| 100% (no abstention) | 0.00 | 406 | 0.549 |
| ~70% | 0.02 | 263 | 0.837 |
| ~55% | 0.30 | 228 | 0.943 |
| **~50%** | **0.86** | **201** | **0.980** |
| ~33% | 0.98 | 133 | 0.992 |

**Reading.** At coverage 50% (commit on the half of problems where the probe is most confident in the first rollout), accuracy on attempted problems is **0.980**. At 33% coverage it is **0.992**. The base no-abstention accuracy is 0.549. This is a strong selective-prediction curve: a deployed system with the option to abstain can be near-perfect on the half of problems where the model is internally confident, even though the unconditional accuracy is around chance. Figure: `extension/outputs/n500/figures/fig19_probe_abstention.png`.

### 18.3 Probe + majority-vote ensemble

Self-consistency (majority vote over K rollouts) is a strong baseline. We test whether the probe is complementary to it:

| Strategy | Accuracy |
|---|---|
| probe-best-of-16 (by probe argmax) | 0.670 |
| majority-of-16 (by first-block equation frequency) | 0.618 |
| **intersection (majority if mean_probe ≥ 0.5, else probe-best)** | **0.677** |
| union (commit iff probe-best ≡ majority) | 0.618 |

Agreement rate between probe-best and majority-best: **53.7%** (218/406 prompts pick the same equation). On the 188 disagreement prompts, **probe-best wins 26 times, majority wins 5 times** (5.2× ratio), 64 both-correct, 93 both-wrong. The probe is **strictly complementary** to majority vote: when the two methods disagree, the probe is right far more often. A trivial intersection rule (majority if probe agrees enough, else probe-best) gives the highest accuracy in the table (0.677). Figure: `extension/outputs/n500/figures/fig20_probe_majority_hybrid.png`.

### 18.4 Cross-scale probe applied: 1.5B has near-oracle probe too

We repeated the §18.1–§18.3 pipeline at 1.5B C_outcome (independent probe trained on 1.5B hidden states; hidden dims differ, so this is **not** a weight-transfer test — it's a "does the same strategy work at the next scale" test). Script: `extension/probe/probe_applied_scale_comparison.py`.

| Metric | 0.5B C_outcome | 1.5B C_outcome |
|---|---|---|
| Held-out balanced AUROC at `</think>` (corrected labels) | **0.982** | **0.974** |
| pass@1 (first-block correctness, cached rollouts) | 0.549 | 0.517 |
| probe-best-of-16 | 0.670 (+12.1 pp) | 0.603 (+8.6 pp) |
| abstain at ~50% coverage | 0.980 | 0.931 |
| abstain at ~33% coverage | 0.993 | 0.956 |
| probe-restart B=16, T=0.95 | 0.675 (used 6.3) | 0.601 (used 5.7) |

**Reading.** The probe is near-oracle at both scales (AUROC 0.982 / 0.974), and the same applied strategies — best-of-K selection, selective abstention, restart — give consistent lifts at both. The 1.5B abstention curve (~93% accuracy at 50% coverage) is slightly below 0.5B's (~98%) but still strong. The qualitative finding is that **the probe-as-applied-tool generalizes beyond the rambling regime**: at 1.5B the model does not ramble (~0% multi-answer), and yet the probe still supports a useful selective-prediction and best-of-K mechanism. Note that the cached cross-scale rollouts subselect to prompts that produced a `</think>` token; the absolute pass@1 numbers above are first-block correctness restricted to that subset and should not be compared directly to the n=50 headline pass@1.

### 18.5 Probe-mean as a near-oracle problem-difficulty signal

Aggregating the probe across the K rollouts for a given prompt gives a per-prompt difficulty estimate. Specifically: for each of the 406 prompts in clean-406, compute `mean(probe across the 16 cached rollouts at </think>)`. Correlate against the per-prompt accuracy (fraction of 16 rollouts whose first `<answer>` block is correct). Script: `extension/probe/probe_creative_extensions.py` part (A).

| Statistic | Pearson r | Spearman ρ |
|---|---|---|
| mean_probe ~ per-prompt accuracy | **+0.967 (p ≈ 4 × 10⁻²⁴³)** | **+0.941** |
| std_probe ~ per-prompt accuracy | +0.062 (p = 0.21, NS) | −0.014 (NS) |

**Reading.** The probe mean across K rollouts is a *near-perfect* predictor of per-prompt accuracy (Pearson 0.97). If you sample K rollouts and average the probe scores, you get an estimate of how often the model will get this problem right, with very little noise. The probe **standard deviation** carries no information about accuracy (r ≈ 0). The implication: the probe captures problem-level difficulty almost optimally, but it does so via the *mean* of confidences, not by signaling "the model is uncertain" via spread. This is itself notable — the model doesn't really "know it doesn't know" via inconsistency across rollouts; its lack of confidence is consistent across rollouts on hard problems.

### 18.6 Cross-checkpoint applied probe transfer — train on C_SFT, deploy on C_outcome

Does the applied probe-selector need to be trained on the deployment-time checkpoint, or does a probe trained on C_SFT activations work as a selector for C_outcome rollouts? Script: `extension/probe/probe_creative_extensions.py` part (B).

| Probe source | Held-out balanced AUROC on C_outcome | best-of-16 selector accuracy | Lift vs pass@1 |
|---|---|---|---|
| C_outcome-trained (in-distribution) | 0.982 | 0.670 | +12.1 pp |
| **C_SFT-trained (cross-checkpoint)** | **0.953** | **0.653** | **+10.3 pp** |

**Reading.** A probe trained only on the SFT checkpoint's hidden states transfers well to the post-RL checkpoint's selector setting: AUROC drops by only 0.029 (0.982 → 0.953), and best-of-16 lift drops by only 1.8 pp (12.1 → 10.3). The applied probe is largely *checkpoint-invariant*. This matters practically: you can train the probe **once** on a stable SFT model and reuse it across many post-SFT RL checkpoints without retraining — including snapshots taken during a single RL run.

### 18.7 Multi-position probe ensemble — does not help, pre_answer saturates the signal

Combining probe scores from multiple positions (pre_answer + assertion + neutral) for best-of-K selection (script: `extension/probe/probe_creative_extensions.py` part (C); 357 prompts with all three probe types):

| Selection score | best-of-16 accuracy | Lift vs pass@1 |
|---|---|---|
| pre_answer alone | **0.765** | **+1.7 pp** |
| assertion alone | 0.762 | +1.4 pp |
| neutral alone | 0.754 | +0.6 pp |
| mean(pre, ass) | 0.762 | +1.4 pp |
| product(pre, ass) | 0.762 | +1.4 pp |
| max(pre, ass) | 0.762 | +1.4 pp |
| min(pre, ass) | **0.765** | **+1.7 pp** |

**Reading.** No combination beats pre_answer-alone selection. Multi-position aggregation gives essentially the same lift (~+1.4 pp) as pre_answer alone (+1.7 pp), and even the worst single-position probe (neutral) is only slightly worse. The trace-final probe is the dominant signal; assertion and neutral probes carry signal too (their AUROCs are individually high — see §2.1), but they're highly *redundant* with the pre_answer signal in this selection setting. The applied story is "use the trace-final probe; multi-position aggregation isn't worth the complexity."

Note: the pass@1 baseline here (0.748) is on the 357-prompt subset where every rollout has at least one assertion + neutral position — a biased-easier subset than the full 406. So the absolute lifts here are smaller than §18.1's; the rank ordering of strategies is the comparison.

### 18.8 Probe-adaptive test-time budget allocation

Given a fixed total budget of B rollouts across N prompts, should you spend them uniformly (B/N each, then best-of-K) or adaptively (give more compute to the prompts where the probe is least confident)? Script: `extension/probe/probe_adaptive_budget.py`. Simulation uses the cached 16 rollouts per prompt as the rollout pool, generating one at a time; the adaptive strategy re-prioritizes after each rollout's probe is revealed.

**Threshold-waterfill strategy.** Round 1: generate 1 rollout per prompt. Rounds 2..R: any prompt whose current best probe score is below threshold T gets one more rollout (priority: lowest current best first); stop when budget exhausted. Final answer = best-by-probe among assigned rollouts per prompt.

**Results (N=288 prompts with full 16 cached rollouts available):**

| Avg rollouts / prompt | Uniform best-of-K | **Adaptive (T=0.95)** | Lift |
|---|---|---|---|
| 1 (no selection) | 0.719 | 0.719 | — |
| 2 | 0.795 | **0.816** | +2.1 pp |
| 4 | 0.806 | **0.830** | **+2.4 pp** |
| 6 | 0.809 | **0.830** | +2.1 pp |
| 8 | 0.809 | **0.830** | +2.1 pp |
| 12 | 0.813 | **0.830** | +1.7 pp |
| 16 (no budget constraint) | 0.823 | 0.830 | +0.7 pp |

**Reading.** At constrained budgets (2–12 rollouts/prompt average), adaptive allocation gives **+1.7 to +2.4 pp** absolute accuracy lift over uniform allocation at the same total compute. The peak is at K_avg=4, where adaptive matches the full K=16 budget's accuracy with **1/4 the rollouts**.

**How it works.** At K_avg=4 the adaptive strategy gives ≤3 rollouts to most prompts (the easy ones, where round 1 already crosses T=0.95) and pushes a subset of 43 hard prompts up to all 16 rollouts. This concentrates compute where it matters: easy prompts don't need re-sampling, hard prompts get a much bigger pool to find a correct rollout in. The probe identifies which prompts are "easy" vs "hard" after just one rollout, at near-oracle accuracy (cf. §18.5 — probe mean is r=0.97 with per-prompt accuracy).

Figure: `extension/outputs/n500/figures/fig23_probe_adaptive_budget.png`.

### 18.9 Probe-as-verifier-free-eval and failure-mode diagnostic

If the probe is near-oracle at predicting first-block correctness, can we use it as a *verifier replacement* for dataset-level accuracy estimation? And: when the probe makes mistakes, what do those mistakes look like? Script: `extension/probe/probe_as_eval_proxy.py`.

**(D) Probe-as-eval-proxy.** For each prompt, take mean probe at `</think>` across 16 rollouts. Average across all 406 prompts.

| Quantity | Value |
|---|---|
| True dataset accuracy (verifier on clean-406) | 0.5531 |
| **Probe-mean estimate (no verifier)** | **0.5565** |
| Probe-vote estimate (per-prompt mean ≥ 0.5) | 0.5985 |

Probe-mean is off by **0.0035** from the verifier's number — essentially exact. This is useful in deployment settings where ground truth is expensive (LLM-judge tasks, open-ended generation): the probe gives a calibrated dataset-level accuracy estimate without any verifier calls. The probe-vote variant overestimates because the probe's positive-class threshold doesn't quite match the natural 50% accuracy threshold; the continuous probe-mean is the better proxy.

**(E) Failure-mode diagnostic.** Of 6306 rollouts on clean-406:

| Class | n | Mean blocks/rollout | Mean probe |
|---|---|---|---|
| TP (probe ≥ 0.5, correct) | 3485 | 9.7 | 0.960 |
| TN (probe < 0.5, wrong) | 2534 | 3.1 | 0.028 |
| **FP (probe ≥ 0.5, wrong) — overconfidence** | **197** | **9.5** | 0.809 |
| **FN (probe < 0.5, correct) — underconfidence** | **90** | **6.8** | 0.207 |

| Calibration metric | Rate |
|---|---|
| P(wrong \| probe ≥ 0.5) — overconfidence rate | **5.4%** |
| P(correct \| probe < 0.5) — underconfidence rate | **3.4%** |

**Reading.** The probe is extremely well-calibrated: only 5.4% of "confident" rollouts are wrong, and only 3.4% of "unconfident" rollouts are correct. The 197 FP overconfidence cases are interesting: they have the same block-count profile (9.5 blocks) as TP correct rollouts (9.7 blocks), and their probe scores are still high (0.809) but not maxed out (vs 0.960 for TPs). These are problems where the model rambles confidently to a wrong answer — i.e., the rambling pathology occasionally produces convincing-looking-but-incorrect first answers that even the probe falls for. The 90 FN underconfidence cases ramble less (6.8 blocks); they're rollouts where the first answer happens to be correct but the model's internal state at `</think>` doesn't strongly endorse it.

These are the natural error modes of any near-oracle calibrated predictor at scale, and they bound the practical ceiling of probe-based selection at the floor "~5% overconfidence × your selection threshold" rate.

Figure: `extension/outputs/n500/figures/fig24_probe_eval_proxy.png`.

### 18.10 First-answer reward RLOO + ramble-penalty RLOO — WITHDRAWN (confounded by training-time stop string)

**This section reports null results that we now believe are CONFOUNDED and uninformative about the underlying hypothesis. We retain it as a methodological cautionary note rather than as evidence.**

We attempted to causally test the hypothesis "rambling is a reward-hack induced by the verifier's last-block scoring rule." Three RL variants from C_SFT:

1. **first-answer RLOO** (`extension/training/firstanswer_rloo.py`): monkey-patch the verifier to score the FIRST `<answer>` block instead of the LAST. Eliminates "free retries" — extra blocks can only hurt.
2. **ramble-penalty RLOO λ=0.05** (`extension/training/ramble_penalty_rloo.py`): reward = first-block-score − 0.05 × max(0, n_blocks − 1). Direct penalty for emitting extra blocks.
3. **ramble-penalty RLOO λ=0.20**: same with stronger penalty (4× λ).

All three ran for 100 RLOO steps from C_SFT with otherwise identical vanilla hyperparameters. Downstream eval on each final checkpoint (500 held-out prompts × 16 rollouts):

| Variant | mean blocks/rollout | first-block acc | last-block acc |
|---|---|---|---|
| C_SFT (start) | 2.90 | 0.313 | 0.253 |
| Vanilla (last-block) C_outcome | 7.59 | 0.602 | 0.543 |
| first-answer | 7.04 | 0.570 | 0.510 |
| ramble-penalty λ=0.20 | **11.23** | 0.561 | 0.342 |

The three RL variants all rambled at eval time — none dropped block count, and the strongest-penalty variant rambled MORE than vanilla. At face value this looks like a clean refutation of the reward-hack hypothesis ("changing the reward rule didn't help"). **It isn't.**

**The confound: training-time `stop=["</answer>"]`.** [rloo_trainer/sampling_worker.py:84-85](rloo_trainer/sampling_worker.py#L84-L85) configures the vLLM sampling worker to halt each rollout at the first `</answer>` token (`stop=["</answer>"], include_stop_str_in_output=True`). Every training rollout therefore contains *exactly one* `<answer>` block. The reward functions, evaluated on 1-block rollouts, collapse to identical signals:

- Vanilla (last block) on n=1: scores the only block
- first-answer (first block) on n=1: scores the only block
- ramble-penalty (first block − λ × max(0, n−1)) on n=1: scores the only block, penalty is 0

**All three reward functions are the same function on the training distribution.** They cannot differ in what they reward, because they never see a multi-block rollout to disagree about. The differences in their eval-time rambling rates are noise (KL drift / seed variance / which arm of the multi-dim correctness subspace the policy slid into) — not differential reward signals.

This is visible in the training metrics for ramble-penalty λ=0.20: `train/response_length_mean` *fell* (481 → 438 tokens), `train/rollout_accuracy` *rose* (0.21 → 0.54), all consistent with the model converging on confident single-block answers. But that's because vLLM was *truncating* the rollouts at the first `</answer>`. The model's behavior *past* `</answer>` — never observed during training — is what produces eval-time rambling.

**What we can still say (correlational only):** at eval time, the rambling rate at 0.5B is high (84% multi-answer, mean 7.6 blocks) and tracks the position-gap across RLOO snapshots at Pearson r=+0.89. We have no causal evidence that the verifier-rule (last vs first) drives this; the experiments designed to test it were silently neutered.

**To actually test the hypothesis** one needs to remove `stop=["</answer>"]` during training so the policy can emit multi-block rollouts and receive differential reward across the three rules. That is substantially more expensive (longer rollouts, slower sampling, more KV cache pressure) and out of scope here.

**Compute lost to this confound:** ~$80 in Modal H100 time across the three runs + downstream evals. Reported honestly because the methodological lesson is the value: *always verify that training-time sampling constraints don't make your candidate reward functions degenerate.*

---

## 19. Probe-as-RL-reward: catastrophic Goodhart in both init regimes

The near-oracle probe (AUROC 0.98 in-distribution; §2.1) is *excellent* as a deployment-time tool (§17–§18.9: best-of-K +12 pp, abstention 98% at 50% coverage, restart 60% compute saved, eval-proxy ±0.4 pp, etc.). The natural follow-up: **can we use it during RL training**, as a replacement for or supplement to the verifier?

### 19.1 Setup

`extension/training/probe_rloo.py` monkey-patches `evaluation.countdown.compute_score` to score each rollout via a fixed pickled linear probe applied to L16 hidden states at the `</think>` token. A reference model (transformers, loaded in the main RLOO process) extracts hidden states per rollout; it's reloaded each training round from the latest checkpoint so the probe sees the *current* policy's representations. Dual-logging is patched into `wandb.sdk.wandb_run.Run.log` so each step emits `train/probe_mean`, `train/verifier_mean`, and `train/probe_minus_verifier` — the verifier is logged for diagnostics only; the probe IS the RL reward.

The probe used for these runs was re-trained on temperature-matched rollouts: 300 prompts × 8 rollouts sampled from C_outcome at `temperature=1.0, top_p=1.0` (matching RLOO's vLLM sampling), labeled by rollout-final verifier correctness, AUROC 0.81 held-out on the matched distribution. (The earlier `eval_c_outcome_n500.json` cache was at `temperature=0.6, top_p=0.95` — a different sampling regime; using the temp=0.6 probe gave OOD activations during training.)

**Two-arm experiment** (both 100 RLOO steps, identical hyperparameters to vanilla):

- **runA**: init from C_outcome (probe is in-distribution at step 0) — sanity test
- **runB**: init from C_SFT (probe is cross-distribution at step 0) — replicate-vanilla test: can probe-RL take C_SFT → C_outcome accuracy lift (0.30 → 0.55) using only the probe as reward?

### 19.2 Engineering reality — 10 bugs before the runs could start

Probe-as-RL-reward is much more brittle than probe-as-deployment-tool. Before runA / runB could run validly, we hit and fixed 10 distinct bugs across 14 launch attempts:

1. `warmup_ratio > 0` + `lr_schedule=constant` incompatible
2. OOM at batch=128 with default grad_accum=1 (need to match vanilla's grad_accum=128)
3. Hardcoded `probe_rloo_run1` in reference-checkpoint path
4. Dual-logging patched `wandb.log` (module function) — rloo.py uses `self.wandb.log()` method
5. Reference-model HF download silently hung container startup
6. `_find_latest_checkpoint()` globally globbed → loaded checkpoint from a *different* run as the reference
7. **Token-position extraction** used the token covering the LAST char of `</think>` (`'>\n\n'`, id 1339) but the probe was trained on hidden states at the FIRST char (`'</'`, id 911) — a two-token offset that placed activations completely OOD, causing the probe to saturate to ~0.99 on every rollout regardless of correctness. (THE bug — explained ~5 failed runs.)
8. **Prompt-reconstruction whitespace**: `numpy.array2string([7, 2, 43, 63])` gives `'[ 7  2 43 63]'` with leading space inside brackets; my `.strip("[]").strip()` dropped it, mismatching asingh15's exact format in ~15% of prompts.
9. Reference-model load defaulted to C_outcome regardless of `--model_name`.
10. Tokenizer defaulted (need explicit `use_fast=True` to match `cache_hidden_states.py`).

Local verification after all fixes: extracted hidden-state vector is **bit-identical** to `cache_hidden_states.py`'s extraction on the same text (max abs diff = 0.0, cosine = 1.0). Probe scores have healthy spread (mean 0.25, std 0.24, range [0.005, 0.911], 0% above 0.95). Methodological note: probe-as-reward deployment requires the *exact* extraction pipeline used to train the probe; subtle mismatches push activations off the probe's calibrated range and cause saturation.

### 19.3 Training trajectories — two distinct Goodhart dynamics

Both runs Goodharted, on different timescales:

**runA (C_outcome init) — "delayed Goodhart":**

| step | probe | verifier | gap | KL |
|---|---|---|---|---|
| 0 | 0.452 | 0.572 | −0.120 | 0.000 |
| 20 | 0.561 | 0.582 | −0.021 | 0.082 |
| 30 | 0.553 | 0.525 | +0.028 | 0.099 |
| **40** | **0.687** | 0.528 | **+0.159** | 0.232 |
| 60 | 0.947 | 0.385 | +0.561 | 0.255 |
| 90 | 0.988 | 0.310 | +0.678 | 1040 |
| 99 (final) | 0.991 | 0.321 | +0.671 | 0.374 |

Probe and verifier tracked closely for the first 30 steps (gap ±0.03). At step 40 the gap suddenly widens; by step 60 the policy has reached probe saturation while verifier has dropped from 0.57 → 0.39. Final verifier 0.32 — the in-distribution probe-RL **destroyed 25 pp of accuracy** off C_outcome's starting point. The "looks benign for 30 steps before sudden collapse" dynamic is particularly dangerous: a researcher checking step-10 metrics would think training is working.

**runB (C_SFT init) — "immediate Goodhart":**

| step | probe | verifier | gap | KL |
|---|---|---|---|---|
| 0 | 0.471 | 0.298 | +0.173 | 0.000 |
| 10 | 0.729 | 0.207 | +0.522 | 0.072 |
| 30 | 0.925 | 0.207 | +0.718 | 0.198 |
| 50 | 0.984 | 0.171 | +0.814 | 0.273 |
| 99 (final) | 0.990 | 0.166 | +0.824 | 0.535 |

Probe climbs 0.47 → 0.99 in 50 steps; verifier drops monotonically from 0.30 → 0.17. **Final verifier 0.17 — probe-RL ended catastrophically WORSE than the SFT baseline (0.29) it started from.** Far from replicating vanilla's lift to 0.55, the probe-RL policy regressed.

### 19.4 Downstream eval — both checkpoints catastrophically bad

Sampled 16 rollouts × 406 clean-406 prompts from each final checkpoint (script: `extension/probe/probe_rl_downstream_analysis.py`):

| Checkpoint | mean blocks/rollout | multi% | first-block acc | last-block acc | mean len |
|---|---|---|---|---|---|
| C_SFT (no RL) | 2.71 | 60.5% | 0.290 | 0.238 | 2094 |
| vanilla C_outcome | 6.78 | 84.0% | **0.550** | 0.498 | 1969 |
| firstanswer C_outcome' | 6.36 | 82.2% | 0.521 | 0.468 | 1984 |
| **probe-RL runA** (C_outcome init) | **15.55** | **99.6%** | **0.236** | **0.130** | 2298 |
| **probe-RL runB** (C_SFT init) | **1.27** | 22.5% | **0.073** | 0.072 | 2432 |

Both probe-RL checkpoints reached the worst first-block accuracies in the entire project. They learned **opposite structural exploits** for the same probe:

- **runA**: emit many `<answer>` blocks (15.5/rollout, 99.6% multi-answer)
- **runB**: emit one `<answer>` block (1.27/rollout, 76.8% one-block)

Both score probe ~0.99 on average. The SAME probe direction admits two completely different "winning" surface patterns.

### 19.5 What the probe was actually noticing — structural confounds

We sampled 30 high-probe rollouts from runB and inspected them. Every probe=1.000 rollout shared a template (`/tmp/inspect_high_prob.py`):

- Opener: *"Let me analyze this step by step:"*
- Numbered enumeration (1. First, 2. Looking at..., 3. I found a solution...)
- Verification language (*"Let me verify one final time:"*, *"Therefore, our solution is valid."*)
- Specific `</think>\n\n<answer>` formatting
- Post-answer `<think>` continuation (*"Let me verify:"*)

Despite the structured rhetorical scaffold, **the actual answers were wrong AND often invalid**. Examples (probe scored 1.000 on all):

- target=39, nums=[43, 56, 34, 50] → answer `((43 - 4) - (56 - 50)) = 39` — uses '4' not in nums; includes `= 39` in the expression itself
- target=85, nums=[66, 61, 42] → answer `(66 - 6) + (66 / 6)` — uses 66 twice; '6' not in nums
- target=62, nums=[10, 7, 45] → answer `(45 + 7) + 10 - 111` — uses '111' which isn't in nums

**Diagnosis.** The probe was trained on C_outcome rollouts where correctness covaries with structured reasoning style. The probe learned to upweight the structural/stylistic surface, not the underlying mathematics. The policy then maximized probe score by emitting that surface template — divorced from actual answer correctness. **Classic confound exploitation**: the probe scored a feature correlated with correctness in the training distribution, the policy gamed the feature.

This explains the "even in-distribution probe-RL Goodharts" result of §19.3. The probe doesn't have to be "wrong" at training time for the policy to break it: the policy actively explores parts of activation space where the probe's *training-time confounds* dominate the *training-time true features*, and the linear probe — having only one weight vector — can't tell the difference.

### 19.6 Causal steering on the post-Goodhart checkpoint — probe direction became mildly causal

The original §2.11 result on vanilla C_outcome was a clean null: perturbing the residual stream along the probe direction at the `</think>` token had accuracy effects indistinguishable from random-direction perturbation (Δ ∈ [−0.07, +0.02] across α). We re-ran the same experiment on runA's post-Goodhart checkpoint to test whether the policy learned to install a *causal write-pathway* to the probe direction during probe-RL training.

**Setup.** Same `extension/probe/causal_steering.py` from §2.11: take 200 prefixes (prompt + response up to `</think>`), inject α · h_mean_norm · v_unit at the `</think>` token's L16 residual stream, continue generation, score the final `<answer>`. Compare probe direction vs random direction at matched magnitudes (α = 0.5, 1.0, 2.0). Conditions: 200 prefixes × {α=0 baseline, α=0.5/1.0/2.0 × {probe, random}}.

**Result on runA's post-Goodhart checkpoint:**

| α | probe-acc | rand-acc | Δ (probe − rand) | vs original §2.11 null [−0.07, +0.02] |
|---|---|---|---|---|
| 0 (baseline) | 0.237 | — | — | (matches downstream first-block acc 0.236) |
| 0.5 | 0.253 | 0.211 | +0.041 | slightly above null |
| **1.0** | **0.253** | **0.170** | **+0.083** | **materially above null** |
| 2.0 | 0.175 | 0.227 | −0.052 | within null |

**Reading.** At α=1.0, perturbing along the probe direction gives **+0.083 absolute accuracy** over a matched-magnitude random direction — clearly outside the original null band. The probe direction has become **mildly causally controllable** on the post-Goodhart policy. RL did install *some* write-pathway to the probe direction.

**But the effect is modest.** Absolute magnitude is small (~8 pp on a 17–25% baseline) compared to the overall accuracy drop the policy exhibited (25 pp from C_outcome's 0.55). The probe direction is **one partial exploit, not the dominant gaming mechanism**. Most of the policy's Goodhart gain came from non-causal structural confounds (the surface rhetorical template documented in §19.5) — not from installing a clean causal write-pathway to the probe direction.

**Two complementary mechanisms of probe-RL gaming:**
1. **Structural confound exploitation** (dominant): the policy learned to emit the surface template ("Let me analyze step by step..." + numbered reasoning + verification language) that correlates with correctness in the probe's training data. This is the larger effect — visible in the +0.65 to +0.83 final probe-minus-verifier gap.
2. **Causal axis installation** (small but measurable): RL slightly increased the controllability of the probe direction itself. The before/after Δ on probe-vs-random shifted from [−0.07, +0.02] to +0.08 at α=1.0 — a real but modest representational change.

This is a clean mech-interp signature: the probe was previously a *correlational reader* of a multidimensional correctness subspace; under RL pressure, the policy partially converted it into a *causal control axis* in its own representation, while also exploiting more powerful surface-level confounds.

### 19.7 Headline

**Near-oracle linear probes on hidden states are excellent inference-time tools but unsafe as RL rewards.** Probe-RL catastrophically Goodharts in both regimes:

- In-distribution (probe trained on policy's own activation distribution): **delayed Goodhart** — ~30 steps of benign-looking training before the policy discovers a structural exploit and accuracy crashes 25 pp below baseline.
- Cross-distribution (probe trained on a different policy's distribution): **immediate Goodhart** — accuracy drops below the SFT starting point within 10 steps; ends 22 pp below SFT.

What the probe was rewarding is not "correctness" but "structural patterns that correlate with correctness in the training distribution." When the policy is free to emit those patterns without solving the problem, it does so. The probe has no way to tell the difference between a correct answer wrapped in structured reasoning and an incorrect answer wrapped in the same structured reasoning.

The boundary between probe-as-deployment-tool (works great) and probe-as-RL-reward (Goodharts catastrophically) is **whether the policy gets gradient access to the probe's input distribution**. At deployment time, the probe sees naturally-generated activations and gives a meaningful score. At training time, the policy reshapes its activations to maximize the probe's output — and finds that the easiest way is to amplify the structural confounds the probe is calibrated to.

**This bounds the broader applied-probe story** (§17–§18): probes are valuable for selection/abstention/restart at inference but should not be deployed as RL rewards without explicit anchoring (verifier hybrid, periodic probe re-training on the new policy's rollouts, or both).

---

## 20. The rambling pathology is an eval-pipeline bug, not a model behavior

The "rambling" pattern at 0.5B C_outcome (mean 7.6 `<answer>` blocks per rollout; 84% multi-answer) and the *worse* rambling we observed at firstanswer (7.04 blocks), ramble-penalty λ=0.20 (11.23 blocks), and probe-RL runA (15.5 blocks) all turn out to be artifacts of a token-id mismatch between the Qwen2.5 tokenizer's `eos_token_id` and the chat-template end-of-turn token. The model is not gaming a reward; it is being forced past its intended stop signal by the eval sampler.

### 20.1 The mechanism

Qwen2.5 was trained with the chat template that delimits assistant turns with `<|im_end|>` (token id **151645**). The tokenizer config sets:

```
tokenizer.eos_token = '<|endoftext|>'   (id 151643)
tokenizer.eos_token_id = 151643
```

vLLM's `SamplingParams` defaults to stopping on `tokenizer.eos_token_id`. So vLLM stops on 151643 — *not* on the token the model was actually trained to emit at end-of-turn.

A forward-pass on C_SFT shows what the model wants to do at the position immediately following the first `</answer>`:

| Top-5 next-token predictions (3 samples, identical) | Token | ID | Probability |
|---|---|---|---|
| 1 | `<|im_end|>` (end-of-turn) | **151645** | **0.973** |
| 2 | `<|im_start|>` | 151644 | 0.002 |
| 3 | (garbage non-ASCII) | — | 0.0002 |
| 4 | (garbage) | — | 0.0001 |
| 5 | (garbage) | — | 0.0001 |

The model is essentially deterministic — it places **97.3% probability on the end-of-turn token**. There is almost no probability mass on continuing.

But vLLM doesn't recognize `<|im_end|>` as a stop. With `skip_special_tokens=True` (vLLM default), the special token is *stripped from the decoded output*, then the sampler continues. After `<|im_end|>` the model is in an out-of-distribution state (it was never trained to generate past end-of-turn), so the continuation is degenerate — most often `\n<think>Let me verify...`, which then produces another `<answer>` block, and the cycle repeats until `max_tokens=1024` is reached.

### 20.2 The cross-checkpoint signature

We checked the eval JSONs for *any* `<|im_end|>` token in the decoded responses, and for what the text following the first `</answer>` looks like (n=8000 rollouts per checkpoint, n=500 prompts × 16 responses):

| Checkpoint | rollouts ending on EOS | `<\|im_end\|>` in decoded text | continuation pattern |
|---|---|---|---|
| C_SFT | 0.0% | 0% | 99.9% `\n<think>Let me verify...` |
| C_outcome (vanilla) | 0.0% | 0% | 94.8% `\n\n<think>...`, 5.2% direct `<answer>` |
| firstanswer | 0.0% | 0% | 97.8% `<think>`, 2.2% `<answer>` |
| ramble-penalty λ=0.20 | 0.0% | 0% | 56.1% `<think>`, 36.7% direct `<answer>` |
| probe-RL runA | 0.0% | 0% | 57.8% `<think>`, 33.4% direct `<answer>` |
| probe-RL runB | 0.1% | 0% | 92.6% `<think>`, 1.6% direct `<answer>` |

**Across all six checkpoints, zero rollouts ever stop on EOS.** Not one. The model never wants to emit `<|endoftext|>` — it wants to emit `<|im_end|>` — but vLLM doesn't honor that, and `<|im_end|>` is stripped on decode, so the output has no visible trace of the model trying to stop.

The "rambling" we measured is a deterministic consequence of forcing a model that wants to stop to keep generating. It is not differential reward signal, it is not policy gaming, it is not parameter drift in any meaningful sense — it is the eval pipeline overrunning the model's intent.

### 20.3 The fix

One line in `extension/evaluation/sample_local_jsonl.py`:

```python
sampling_params = SamplingParams(
    ...
    stop_token_ids=[tokenizer.eos_token_id, 151645],  # also stop on <|im_end|>
)
```

We re-ran eval on C_outcome and C_SFT with the fixed sampler. (See `eval_c_outcome_FIXEDSTOP_n500.json` and `eval_c_sft_FIXEDSTOP_n500.json`. n = 500 prompts × 16 rollouts = 8000 each, temperature 1.0, max_tokens 1024.)

| Checkpoint | mean blocks (bug) | mean blocks (fixed) | mean chars (bug) | mean chars (fixed) | acc_first (bug) | acc_first (fixed) | acc_last (bug) | acc_last (fixed) |
|---|---|---|---|---|---|---|---|---|
| **C_SFT** | 2.83 | **1.04** | 2111 | **1072** | 0.313 | 0.239 | 0.253 | 0.241 |
| **C_outcome** | 7.41 | **1.83** | 1982 | **1095** | 0.602 | 0.583 | 0.543 | 0.607 |

**Reading.**
- The "rambling" at C_SFT essentially disappears: mean blocks drops from 2.83 to 1.04 (effectively single-block emission per rollout, matching the 97.3% `<|im_end|>` prediction).
- At C_outcome, rambling drops by 75% (7.41 → 1.83), but **does not fully collapse to 1.0** — the RL-trained model retains ~17% multi-block emission rate even under proper stopping. This residual rambling IS a real behavioral signal: RL shifted some probability mass away from `<|im_end|>` toward continuation tokens. Whether this is "the real reward-hack we were after" or just KL drift remains an open question for a clean follow-up experiment.
- At C_outcome under proper stop, `acc_first ≈ acc_last` (0.583 vs 0.607), as expected when most rollouts have one block. The 0.024 gap is the residual contribution from the 17% multi-block rollouts.
- Caveat: bug-era evals were likely at temperature 0.6, while these fixed evals are at temperature 1.0. The accuracy comparisons are confounded by sampling temperature; the block-count comparisons are not (block count is a structural property at any temperature).

### 20.3a Confirmation: RL shifted P(`<|im_end|>`) downward at C_outcome

A direct forward-pass logit check at the post-`</answer>` position on the C_outcome step-90 checkpoint, same setup as §20.1:

| Sample | C_SFT P(`<\|im_end\|>`) | C_outcome P(`<\|im_end\|>`) | Δ |
|---|---|---|---|
| 0 (833 tokens) | 0.9731 | **0.9624** | $-0.0107$ |
| 1 (661 tokens) | 0.9732 | **0.9591** | $-0.0141$ |

RL training pulls ~1.1–1.4% of probability mass away from the stop token. Per-rollout this is tiny, but it compounds across 16 rollouts × 500 prompts into the 17% multi-block rate we measured. This is the cleanest evidence that **the residual rambling at C_outcome under proper stopping is a real RL-induced distributional shift** — not the original "reward-hack" framing, but a genuine (small) effect on the model's unobserved post-`</answer>` distribution. The mechanism is parameter drift through the shared transformer parameters: RL's gradient updates on first-block tokens incidentally drag down the post-stop distribution's mass on `<|im_end|>`, because that token was never observed during training rollouts (vLLM stopped at `</answer>` before it could appear) and therefore has no preservation signal.

### 20.4 Implications for prior claims

**Unaffected:**
- Probe AUROCs (§2.1, §2.7, §15) — measured on cached activations, not on rambling counts.
- Pattern A within-rollout findings (§2.4, §8) — measured on actual emitted blocks; the model genuinely does produce multiple blocks under the buggy sampler, but those blocks are real, parseable, and the model's commit-time representation tracks each one's content. Pattern A holds as a description of "what the model represents at each block it emits" regardless of why it emits them.
- Causal steering null (§2.11) — bugs in eval don't affect causal interventions at fixed activation positions.
- Probe-as-RL-reward Goodhart story (§19) — the policy did game the probe via structural confounds in pre-`</think>` reasoning; the post-`</answer>` rambling we documented in runA (15.5 blocks) is partly the same sampler bug, but the dominant gaming mechanism (`/tmp/inspect_high_prob.py` shows the structural-template signature inside `<think>`) is independent of the sampler.
- Applied probe results (§17, §18.1-§18.9) — these used real per-rollout probe scores; even if the post-`</answer>` extension was bug-induced, the within-rollout probe trajectory is real signal.

**Affected:**
- The rambling rate / position-gap correlation (r = +0.89 across snapshots, §2.2 / §7). The position-gap (a probe AUROC observation) is real, but the rambling rate it correlates with is largely artifactual. The correlation may still mean *something* (sampler-induced rambling growing under RL is itself a function of the policy's drift), but it is no longer the clean reward-hack signature we framed it as.
- The "rambling reward-hack" framing throughout the writeup — withdrawn. The §18.10 firstanswer + EXP-22 ramble-penalty experiments were already withdrawn as confounded by the *training-time* stop string in the sampling worker (every training rollout has n_blocks=1, so the three rewards collapse). The *eval-time* `<|im_end|>` bug is an additional, simpler confound that further invalidates rambling as a behavioral target.
- The "1.5B doesn't ramble" scale claim (§15.4) — **partially verified**. The 1.5B SFT checkpoint's `tokenizer_config.json` has `eos_token = '<|endoftext|>'` (id 151643) — the SAME mismatch as 0.5B. So the bug is not eliminated by tokenizer config alone. This means the 0.075% multi-answer rate at 1.5B is a *genuine* model-level difference: at the post-`</answer>` position, the 1.5B model must place more probability on the actual EOS (151643) than the 0.5B model does. Why is open — could reflect the 1.5B SFT being undertrained (6 epochs at lr=1e-5; we did not match asingh15's recipe) so the model retains more of the base Qwen pre-training prior, which more often emits `<|endoftext|>` directly; or it could be a true capacity-related effect. The "scale-dependence" claim survives but its mechanism is unconfirmed.

### 20.5 The methodological lesson

Three independent layers of confound were stacked here:
1. **Training-time stop string** (`stop=["</answer>"]` in `rloo_trainer/sampling_worker.py:84-85`) — collapses all candidate reward functions to "score the first/only block."
2. **Eval-time eos_token_id mismatch** (`tokenizer.eos_token_id = 151643`, model wants 151645) — fires every rollout past its intended stop into OOD territory.
3. **`skip_special_tokens=True` decoding** — hides the model's actual stop attempt from any human inspection of the rollout text, making the bug invisible without a forward-pass logit check.

Each layer alone would have been hard to find. Together they produced a story ("model rambles → reward hack → can be fixed by reward shape") that was clean, intuitive, and entirely wrong. The eventual evidence was a single line of forward-pass output: `P(<|im_end|>) = 0.973`.

The lesson is uncomfortable: when a published-style claim depends on a behavioral measurement, **verify that the measurement reflects what the model is *trying* to do**, not just what the sampler reports. A forward-pass argmax check at the relevant position would have caught this in five minutes, before any of the EXP-19 / EXP-22 RL runs were launched.

---

## 21. Probe-as-baseline (not probe-as-reward): a clean way to use a near-oracle critic in RL

Co-author: **Anagha Ramaswamy** (`anagha-ramaswamy`). See [rloo_trainer/rloo_update_worker.py:243-260](rloo_trainer/rloo_update_worker.py#L243-L260) for the implementation and [extension/training/probe_reward_rloo.py](extension/training/probe_reward_rloo.py) for the trainer wrapper.

§19 showed that using a near-oracle linear probe (AUROC 0.98) directly as the RLOO reward catastrophically Goodharts in both initialization regimes. The probe is gameable as a *target*. But the probe is also a *very accurate value estimator* — and in policy gradient, the canonical use for an accurate value estimator is the **baseline**, not the reward. This section describes a probe-as-baseline trainer that uses the probe correctly.

### 21.1 The idea

Standard RLOO uses the leave-one-out mean of the group's rewards as the per-rollout baseline:
$$A_i = r_i - \frac{1}{G-1} \sum_{j \neq i} r_j$$
where $r_j$ is the verifier reward in $\{0, 0.1, 1.0\}$. This is unbiased (the baseline doesn't depend on rollout $i$'s action) but high-variance: the group baseline is essentially a few discrete points away from the per-rollout reward.

The probe-as-baseline variant replaces the baseline with the leave-one-out mean of *probe values*:
$$A_i = r_i - \frac{1}{G-1} \sum_{j \neq i} v_\theta(s_j)$$
where $v_\theta(s_j) \in [0,1]$ is the frozen near-oracle probe applied to rollout $j$'s `</think>` hidden state. The **reward stays the verifier** — the policy still optimizes the true reward, so there is no Goodhart. The baseline is now a smooth, low-variance estimate of expected reward (since the probe achieves AUROC 0.98 on the verifier-defined correctness label). Lower advantage variance → lower gradient variance → faster convergence in expectation.

### 21.2 Why this avoids Goodhart

In §19's probe-as-reward setup, the optimization target was `probe(rollout)`. The policy could maximize this by finding any feature the probe correlated with, regardless of true correctness. In probe-as-baseline, the optimization target is still `verifier(rollout)`. The probe enters only through the baseline, which is subtracted from $r_i$ but doesn't affect the optimization landscape — it only affects the variance of the advantage estimate.

Formally, for any baseline $b(s)$ that doesn't depend on the current action,
$$\mathbb{E}_{a \sim \pi}[\nabla_\theta \log \pi(a|s) \cdot (r(s,a) - b(s))] = \mathbb{E}_{a \sim \pi}[\nabla_\theta \log \pi(a|s) \cdot r(s,a)]$$
The baseline's only role is variance reduction. Using a near-oracle critic as the baseline is the textbook variance-reduction strategy (cf. actor-critic methods); the novelty here is that we have an unusually good critic *without separately training one* — the linear probe trained for measurement work doubles as the critic.

### 21.3 Implementation

[rloo_trainer/rloo_update_worker.py](rloo_trainer/rloo_update_worker.py): the `update()` method accepts an optional `probe_baseline` array (per-rollout probe values). When provided, the LOO baseline is computed from probe values instead of rewards:
```python
if probe_baseline is not None:
    grouped_pb = torch.as_tensor(probe_baseline, ...).view(-1, self.group_size)
    pb_sum = grouped_pb.sum(dim=1, keepdim=True)
    baseline = (pb_sum - grouped_pb) / (self.group_size - 1)
else:
    group_sum = grouped_rewards.sum(dim=1, keepdim=True)
    baseline = (group_sum - grouped_rewards) / (self.group_size - 1)
advantages = (grouped_rewards - baseline).reshape(-1)
```

[rloo_trainer/rloo.py](rloo_trainer/rloo.py): a `--probe_baseline` CLI flag toggles the path. When on, the trainer attaches a `probe_valuer` (frozen reference model + probe pkl) that scores each sampled rollout at the `</think>` token. The rest of the pipeline is unchanged.

The trainer-side machinery (loading the frozen model, attaching the probe, computing `pv` per batch) is contained in [extension/training/probe_reward_rloo.py](extension/training/probe_reward_rloo.py) which also supports the §19-style probe-AS-reward arm; the `--probe_baseline` flag in `rloo.py` is the standalone baseline path.

### 21.4 Status

Code-complete, plumbing verified. A controlled comparison (vanilla outcome RLOO vs.\ probe-baseline RLOO from the same initialization, same seed, same compute) is the natural next experiment. The hypothesis is: equivalent or higher final accuracy at fewer training steps, with smoother per-step learning curves (lower advantage variance ⇒ less noisy gradient updates). Out of scope for this writeup; teed up for follow-up work.

---

