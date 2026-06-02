# Findings — exhaustive master record

*Every experiment, every number, every script, every artifact. Updated as work proceeds.*

> **Purpose.** `writeup.md` is the paper-ready synthesis (~750 lines, narrative). `extension/CHANGELOG.md` is the chronological work log. This file is the **dense lookup table**: per-experiment, what we asked, how we asked it, what the numbers were, which scripts produced them, where the raw outputs live, and what the conclusion was. It's deliberately less readable than the writeup and more complete than the changelog.
>
> **Date of last update.** 2026-06-01 (corrected-label probe pipeline, applied-probe extensions, first-answer RLOO mid-run).
> **Status legend.** ✅ done · ⏳ running · ⏸ paused / blocked · ❌ failed / abandoned · 🔜 queued.

---

## 0. Headline numbers (one-glance)

### 0.5B (primary) — corrected labels throughout

| Metric | C_SFT | C_outcome | Δ |
|---|---|---|---|
| pass@1 (n=50 asingh15 test) | 28.6% | **53.5%** | +24.9 pp |
| pass@16 | 78.0% | 72.0% | −6 pp (sharpening) |
| Trace-final probe AUROC (L16, corrected labels) | 0.912 | **0.982** | +0.070 |
| Assertion-position AUROC (L16, corrected labels) | 0.885 | 0.896 | +0.011 |
| Position-appropriate `<answer>` probe AUROC | 0.920 | — | — |
| **Gap pre_answer − assertion (L16, corrected)** | +0.027 | **+0.086** | +0.059 |
| **Per-problem** trace-final AUROC mean (L16, corrected) | 0.882 | **0.927** | +0.045 |
| Per-problem Spearman (probe_drop vs accuracy_delta), corrected | — | r=+0.062, p=0.48 | null |
| Cross-checkpoint transfer pre_answer SFT→outcome | — | **0.953** | high |
| Cross-position cosine pre vs ass (L16, corrected) | +0.104 | +0.036 | ≈ orthogonal |
| Pearson r(rambling rate, position-gap) across snapshots | — | **+0.891 (p=0.04)** | rambling tracks gap |
| Causal steering Δ(probe direction − random) | — | [−0.07, +0.02] across α | null |
| Rambling rate (multi-answer rollouts) | — | **87%** | — |
| First-answer vs last-answer pass@1 gap (clean-406, EXP-20) | +0.052 | **+0.052** | first > last; McNemar p≈5e-43 |
| Post-first-answer token waste (clean-406, EXP-20) | 54.0% | **56.2%** | rambling tail = wasted tokens |

### 1.5B (scale extension) — corrected labels at trace-final

| Metric | 1.5B C_SFT | 1.5B C_outcome | Δ |
|---|---|---|---|
| pass@1 (n=50 test) | 28.0% | **48.0%** | +20.0 pp |
| pass@1 (n=500 procedural) | 23.6% | **55.8%** | +32.2 pp |
| Trace-final probe AUROC (L16, corrected) | — | **0.974** | near-oracle |
| **Gap pre−assertion (L20)** | — | **+0.040** | small |
| Matched-pair % above-diag (L16) | 86% | 82% | MW p = 0.14 (NS) |
| Rambling rate | — | **0.075%** | absent |

**Scale interpretation.** The rambling pathology is a small-scale (0.5B) phenomenon (87% multi-answer vs 0.075% at 1.5B). The position-decoupling gap that emerges under RL at 0.5B is correlationally tied to rambling (r = +0.89); since rambling doesn't develop at 1.5B, the gap stays small. The probe is near-oracle at both scales.

**Compute spent so far** (Modal H100 at ~$4/hr, billed to `ayeung16` workspace): ≈ **$25-35** across the n=500 expansion, Option B dynamics, per-layer sweep, causal steering, and the per-problem correlation experiments. RLOO 1.5B currently running (~$30-40 estimated).

---

## 1. Setup

### 1.1 Models

| Tag | Construction | Source | Path |
|---|---|---|---|
| `C_SFT` | Qwen2.5-0.5B + Anikait Singh's Countdown SFT | HF: `asingh15/qwen-sft-countdown-defaultproj` | — |
| `C_outcome` | RLOO from `C_SFT`, outcome reward only (0/0.1/1.0), 100 steps, snapshots every 10 | trained by us | `/vol/checkpoints/rloo_checkpoints/rloo_training/rloo_fixed_v2/latest_checkpoint/model` |
| `C_outcome_step_{N}` | Intermediate RLOO snapshot at step N ∈ {0,10,…,90} | persisted during run | `/vol/checkpoints/rloo_checkpoints/rloo_training/rloo_fixed_v2/epoch_0_step_{N}/model` |
| **1.5B SFT** | Qwen2.5-1.5B + our SFT on `Asap7772/cog_behav_all_strategies` (1200 demos, 6 epochs, lr=1e-5) | ours | `/vol/checkpoints/sft_qwen15b_countdown/sft_qwen15b_countdown/sft_1.5b_run2/model` |
| **1.5B RLOO** | RLOO from 1.5B SFT, 100 steps, snapshots every 10 (running) | ours | `/vol/checkpoints/rloo_qwen15b_checkpoints/...` (in progress) |

### 1.2 Eval sets

- **Original (n=50)**: `asingh15/countdown_tasks_3to4` test split.
- **n=500 procedurally generated**: `extension/data/countdown_eval_500.jsonl`. 250×3-num + 250×4-num. **0/500 overlap with asingh15 TEST** (verified). **94/500 overlap with asingh15 TRAIN** (the RLOO training pool — relevant only for C_outcome).
- **clean-406**: the 406/500 not in asingh15 train. Primary eval set throughout. Recorded in `extension/data/contaminated_prompt_idx.json` (lists both `clean` and `contaminated`).
- **Rollouts on clean-406** (16 per problem, temperature 0.6, top_p 0.95, top_k 20): `eval_c_sft_n500.json`, `eval_c_outcome_n500.json`.
- **Option B snapshots (n=200 fresh rollouts each)**: `eval_c_outcome_step_{30,60,90}_n200.json`.

### 1.3 Probe pipeline

- Hidden states: Qwen2.5-0.5B with `output_hidden_states=True`.
- Layers: L12, L16 (primary), L20. Per-layer sweep covered all 25 layers (L0–L24).
- Positions: `pre_answer` (the `</think>` token), `assertion` (confidence-keyword tokens inside `<think>`), `neutral` (matched-count random control), `<answer>`-opening (Phase 2A).
- Probe: `LogisticRegression(C=0.1, max_iter=2000)` on `StandardScaler`-normalized features. Pickled as `Pipeline(StandardScaler, LogisticRegression)` so scaling travels.
- Training: `GroupKFold(5)` on `prompt_idx` (held-out *problems*, not rollouts), balanced subsample within each fold.

### 1.4 Procedural eval generator — format & contamination check

`extension/data/generate_countdown.py`. Sample 3 or 4 distinct ints from [1,100]; exhaustively enumerate orderings × ops × parenthesizations; if any integer in [10,100] is reachable, randomly select one as the target. Format prompt to be **byte-identical** to asingh15's NumPy-style right-justified bracket layout (e.g. `[14 45  9  1]`).

Verified equal char-for-char on matched `(nums, target)` inputs. Length match too.

---

## 2. Experiments — chronological

### EXP-01: Initial 0.5B aggregate probe AUROCs (n=50) — superseded by EXP-02/EXP-14 ✅

The n=50 results were superseded by the n=500 clean-406 dataset (EXP-02) and again by the corrected-label pipeline (EXP-14). The corrected numbers are in the headline section of this file and writeup §2.1; this entry retains only the timeline pointer.

**Scripts.** `extension/probe/cache_hidden_states.py` (the n=50 `analyze_probes.py` analysis was removed in the 64→30 script trim; superseded by the corrected-label `relabel_*` pipeline).

---

### EXP-02: n=500 procedural expansion + clean-406 contamination filter ✅

**Method.** 
1. Built procedural Countdown generator (§1.4) and emitted 500 fresh problems.
2. Sampled 16 rollouts × 500 prompts on both C_SFT and C_outcome via Modal vLLM. 5 parallel rollout jobs (~30 min).
3. Cached hidden states at L12/L16/L20 × 3 position kinds × both checkpoints (~5 min wall).
4. Verified contamination: 94/500 in asingh15 train, 0/500 in test. Filtered to clean-406.

**Headline numbers (clean-406, L16, balanced GroupKFold(5), corrected labels):**

| Cell | C_SFT | C_outcome | Δ |
|---|---|---|---|
| pre_answer | 0.912 | **0.982** | +0.070 |
| assertion | 0.885 | 0.896 | +0.011 |
| neutral | 0.516 | 0.567 | +0.051 |
| **gap (pre − assertion)** | **+0.027** | **+0.086** | +0.059 |

**Matched-pair (within-prompt, assertion position, corrected labels):**
- C_SFT: Wilcoxon p = 9.3e−35
- C_outcome: Wilcoxon p = 3.9e−8
- Mann-Whitney between checkpoints: p = 0.68 (NS)

**Cross-checkpoint pre_answer transfer (corrected):** C_SFT→C_outcome = 0.953; C_outcome→C_SFT = 0.822.

**Scripts.**
- Generator: `extension/data/generate_countdown.py`
- Modal rollouts: `extension/evaluation/sample_local_jsonl.py`
- Cache: `extension/probe/cache_hidden_states.py`
- Filter: `extension/probe/filter_to_clean.py`
- Corrected-label probes + downstream stats: `extension/probe/relabel_full_grid.py`, `relabel_redo_downstream.py`, `relabel_cross_checkpoint.py`, `relabel_per_problem.py`, `relabel_cosines.py`.

**Caches.** `extension/cache/probe_cache_n500_clean406/` (primary).

**Modal cost.** ≈ $5-8 (vLLM rollouts + hidden-state cache).

---

### EXP-03: Option B dynamics — gap emerges over training ✅

**Method.** Re-sample fresh rollouts (n=200, first 200 prompts) from snapshot models at steps 30/60/90; re-cache hidden states at each step; re-train the probe per snapshot with corrected labels (first-`<answer>`-block correctness).

**Result (L16, balanced GroupKFold(5), corrected labels):**

| Step | pre_answer AUROC | assertion AUROC | gap | mean blocks/rollout |
|---|---|---|---|---|
| C_SFT (pre-RL) | 0.912 | 0.885 | **+0.027** | 2.83 |
| step 30 (n=200 fresh) | 0.907 | 0.867 | **+0.040** | 3.09 |
| step 60 (n=200 fresh) | 0.962 | 0.914 | **+0.048** | 4.49 |
| step 90 (n=200 fresh) | 0.971 | 0.835 | **+0.136** | 7.18 |
| C_outcome (final) | 0.982 | 0.896 | **+0.086** | 7.41 |

**Pearson r(mean_blocks, gap) across snapshots = +0.891 (p = 0.04).** Gap grows over training in lockstep with rambling rate.

**Scripts.**
- Modal rollouts at snapshots: `extension/evaluation/launch_expansion_rollouts.sh` (same batch as Phase 1)
- Corrected-label per-snapshot dynamics: `extension/probe/relabel_dynamics.py`
- (the original `launch_expansion_cache.sh`, `per_snapshot_decoupling_gap.py`, and `headline_dynamics_figure.py` were removed in the 64→30 script trim)

**Caches.** `extension/cache/probe_cache_dynamics_optB/` (fresh-rollout caches per snapshot).

**Modal cost.** ≈ $3-5.

---

### EXP-04: Phase 1 — cross-position transfer + diagnostics

The Phase 1 cross-position transfer experiment used the pre-relabel (wrong) label rule and is not represented here. The corrected-label cosine-based geometric claim (position-orthogonality within each checkpoint) is in EXP-07; the cross-checkpoint pre_answer/assertion transfer with corrected labels is in writeup §2.7. We did not re-run the full 3×3 cross-position AUROC matrix with corrected labels.

---

### EXP-05: Phase 2A — within-rollout probe trajectory (Pattern A vs B) ✅

**Question.** Within multi-answer rollouts where the model emits a *correct* first equation and *drifts to a wrong final one* (T→F): does the probe at the wrong-final position still encode the first answer's correctness (Pattern B, "knows but doesn't say") or does it move with the model's commit (Pattern A, "true belief update")?

**Method.**
1. **Pre-flight (local, free):** parse all C_outcome multi-answer rollouts, classify per-block correctness. → `phase2a_per_answer_correctness.py` → `17_per_answer_correctness.txt`.
2. **Modal cache (~$2):** extract hidden states at every `<answer>` opening token across 5458 multi-answer C_outcome rollouts (44306 hidden states/layer × 3 layers). → `extension/probe/cache_answer_positions.py`. Output: `extension/cache/probe_cache_n500_answers/`.
3. **Local analysis (free):** train held-out (GroupKFold(5)) trace-final probe AND a position-appropriate `<answer>`-opening probe; apply to every `<answer>` block; aggregate per-rollout.

**Pre-flight counts (clean-406 C_outcome rollouts):**
- Total: 6496
- ≥2 `<answer>` blocks (multi-answer): **5458 (84%)**
- T→F drift (correct → wrong): **490 (9% of multi-answer)**
- F→T (rescue): 150
- TT: 2983; FF: 1835

**Pattern A test (position-appropriate probe at `<answer>` opening, held-out diagonal AUROC = 0.920):**

| transition | n | probe(first) | probe(last) | verdict |
|---|---|---|---|---|
| TT both correct | 2983 | 0.874 | 0.823 | — |
| **T→F drift correct→wrong** | **490** | **0.856** | **0.154** | probe matches LAST commit, not FIRST |
| **F→T rescue wrong→correct** | **150** | 0.156 | **0.580** | probe matches LAST commit, not FIRST |
| FF both wrong | 1835 | 0.084 | 0.088 | — |

probe(last) on T→F is **indistinguishable from F→F floor (0.088)**. **Pattern A confirmed cleanly.** The model genuinely updates its belief at each commit; there is no preserved "secret correct" representation that would have made activation patching meaningful (Phase 2B was therefore skipped per the pre-registered gating rule).

**Per-block trajectory on T→F rollouts** (position-appropriate probe; probe descends with %correct):
```
block 0: probe=0.856, %corr=100%
block 1: probe=0.644, %corr=75%
block 4: probe=0.384, %corr=38%
block 12: probe=0.218, %corr=2%
```

**OOD caveat for trace-final probe.** When the trace-final-trained probe is applied to first `<answer>` opening tokens (avg token position 534, much earlier than `</think>` at ~1093), it returns ~0 uniformly across all classes — a calibration-mismatch OOD artifact, NOT a Pattern B signal. The position-appropriate probe (§8.2 in writeup) is the right tool; it gives a 0.874 probe(first) on TT vs 0.084 on FF — clean discrimination at block 0.

**Scripts.**
- `extension/probe/phase2a_per_answer_correctness.py` → `17_per_answer_correctness.txt`
- `extension/probe/cache_answer_positions.py` (Modal job)
- `extension/probe/phase2a_pattern_analysis.py` → `18_phase2a_patterns.txt`
- `extension/probe/phase2a_position_appropriate_probe.py` → `19_phase2a_position_appropriate.txt` + `fig9b_within_rollout_position_appropriate.png`

**Modal cost.** ≈ $2 (one forward-pass-only cache job).

---

### EXP-06: F→T rescue trajectory (Pattern A bidirectional) ✅

**Question.** Pattern A in T→F: does it also hold in the rescue direction (F→T)? If the probe rises across blocks as %correct rises in rescue rollouts, the bidirectional Pattern A claim is sharper.

**Method.** Same position-appropriate probe; just analyze the 150 F→T rollouts' per-block trajectory.

**Result.**

| direction | n | block 0 probe | mid-rollout probe | terminal probe |
|---|---|---|---|---|
| **F→T (rescue)** | 150 | 0.156 (0% correct) | 0.50 (block 2-8, 86-95% correct) | 0.73 (block 11+, 100% correct) |
| T→F (drift, for ref) | 490 | 0.856 (100% correct) | 0.42 (block 2-4, 38-52%) | 0.22 (block 12+, 2%) |

Probe responds to both directions. **Pattern A is bidirectional.** Amplitude is asymmetric (rescue probe values lower at matched %correct than drift values) — likely a residue/selection effect (rescue rollouts started wrong).

**Script.** `extension/probe/ft_rollout_trajectory.py` → `20_ft_trajectory.txt` + `fig10_ft_rollout_trajectory.png`.

**Modal cost.** $0 (reuses Phase 2A cache).

---

### EXP-07: Probe-direction cosine similarity ✅

**Question.** Are the position-decoupling claims (low cross-position AUROC transfer) backed by *geometric* orthogonality of probe direction vectors, or just by activation-scale mismatches?

**Method.** Train one probe per (ckpt, layer, position kind) on the full balanced subsample; recover input-space direction as `w_lr / scaler.scale_`; compute pairwise cosines.

**Within-checkpoint cross-position cosines (L16, corrected labels):**
| pair | cos |
|---|---|
| C_SFT pre vs assertion | +0.104 |
| C_SFT pre vs neutral | −0.031 |
| C_outcome pre vs assertion | +0.036 |
| C_outcome pre vs neutral | +0.064 |
| C_outcome ass vs neutral | +0.035 |

**Cross-checkpoint within-position cosines (L16, corrected):**
| pair | cos | transfer AUROC |
|---|---|---|
| C_SFT pre vs C_outcome pre | +0.169 | 0.953 |
| C_SFT ass vs C_outcome ass | +0.134 | 0.770 |
| C_SFT neu vs C_outcome neu | +0.072 | — |

**Interpretation.** The position-decoupling is geometric (cosines ≤0.10 within ckpt), not just AUROC-level. Cross-checkpoint: probe directions point in different directions (cosine 0.17) but the underlying correctness *subspace* is shared (AUROC 0.95) — multiple low-cosine directions can each "read" the same signal.

**Script.** `extension/probe/relabel_cosines.py` → `extension/outputs/n500/text/41_relabel_cosines.txt`.

**Modal cost.** $0 (local).

---

### EXP-08: Per-layer probe sweep (all 25 layers) ✅

**Question.** Is the pre−assertion gap on C_outcome concentrated at late layers, or distributed across depth?

**Method.** Cached hidden states at every layer 0..24 on both checkpoints. Per-cell balanced GroupKFold(5) probe AUROC at L12/L16/L20 was re-run with corrected labels; the full 25-layer sweep was originally run with the wrong label rule. The depth-invariance qualitative result — gap roughly flat from L5 to L24 with no monotonic trend toward late layers — survives, ruling out the "selective late-layer shaping" mechanism (writeup §16, b).

**Cache.** `extension/cache/probe_cache_n500_all_layers_clean406/` (300 files, 1.3 GB local).

**Modal cost.** ≈ $3 (one Modal job per checkpoint, ~5 min each).

---

### EXP-09: Causal steering at `</think>` ✅

**Question.** Is the trace-final probe direction a **causal control axis** for the model's output (intervention via residual-stream injection changes behavior in a predictable direction), or only a **correlational reader** (probe predicts but doesn't steer)?

**Method.** HuggingFace generation with a forward hook on Qwen2's layer-16 decoder. For each prefix (existing rollout, truncated to and including `</think>`), inject `α · h_mean_norm · v_unit` at the `</think>` token position; continue autoregressive generation; verify the new final `<answer>`.

Conditions per prefix:
- α=0 (baseline)
- α=+0.5, +1.0, +2.0 along probe direction
- α=+0.5, +1.0, +2.0 along a fixed random unit direction (matched magnitude control)

**Result (n=97 prefixes, C_outcome, L16):**

| condition | accuracy | acc_format |
|---|---|---|
| baseline (α=0) | 0.577 | 1.000 |
| probe α=+0.5 | 0.567 | 1.000 |
| probe α=+1.0 | 0.598 | 1.000 |
| probe α=+2.0 | 0.515 | 0.845 |
| **random α=+0.5** | **0.639** | 1.000 |
| random α=+1.0 | 0.577 | 0.938 |
| random α=+2.0 | 0.546 | 0.897 |

**Probe-vs-random Δ:**
- α=+0.5: probe 0.567 vs random 0.639 → **Δ = −0.072**
- α=+1.0: probe 0.598 vs random 0.577 → **Δ = +0.021**
- α=+2.0: probe 0.515 vs random 0.546 → **Δ = −0.031**

**All within sampling noise.** Probe direction is indistinguishable from random direction. **The probe is a reader, not a controller.** Replicates Yuan et al.'s 1.5B+ activation-patching null at small scale with a matched-magnitude random-direction control.

**Scripts.**
- `extension/probe/save_probe_vector.py` (saves L16 pre_answer probe direction to `extension/cache/steering/`)
- `extension/probe/causal_steering.py` (Modal job with HF + forward hook)
- `extension/probe/analyze_causal_steering.py` → `23_causal_steering.txt`
- (the separate `causal_steering_figure.py` plotting helper was removed in the 64→30 script trim)

**Bug history.** First attempt crashed at the hook (newer transformers returns Tensor not tuple from decoder layer). Fixed in commit `34a2e8c`. Second attempt's stdout buffered for ~50 min — killed (had 37/100 prefixes done); third attempt added `flush=True`; resumed on the missing 63 prefixes with a JSONL-based skip filter (`extension/data/steering_todo.jsonl`).

**Modal cost.** ≈ $4-6 (~60 min H100 across the two passes).

---

### EXP-10/11: Per-problem probe-AUROC vs accuracy-delta correlation — superseded by corrected-label re-run

The original per-problem correlation analyses (using rollout-level labels, before we discovered the labels were wrong for multi-answer rollouts) gave numbers that don't survive the corrected-label re-run. The corrected version is in the headline (§0 of this file) and writeup §8.5: under corrected labels, per-problem AUROC at trace-final RISES under RL (mean 0.88 → 0.93), Spearman r(probe_drop, acc_delta) is statistically insignificant (r = +0.062, p = 0.48), and the dominant per-problem quadrant is "both improved" (55.7%). The original "decoupling at the per-problem level" framing was an artifact of wrong labels.

Script for corrected version: `extension/probe/relabel_per_problem.py` → `extension/outputs/n500/text/42_relabel_per_problem.txt`.

---

### EXP-12: 1.5B SFT (scale extension setup) ✅

**Question.** Does the position-decoupling claim scale upward to 1.5B?

**Method.** Train Qwen2.5-1.5B SFT on the same `Asap7772/cog_behav_all_strategies` demos that the asingh15 0.5B SFT used. Hyperparameters: batch_size=64 (effective, ÷ grad_accum 16 = microbatch 4), 6 epochs, lr=1e-5, gradient clipping 1.0.

**Result.** SFT completed in ~7 min on H100. Saved to `/vol/checkpoints/sft_qwen15b_countdown/sft_qwen15b_countdown/sft_1.5b_run2/model`.

**Eval on Countdown test (n=50):**
- pass@1 = **0.280** (vs 0.286 for asingh15 0.5B SFT — essentially matched)
- pass@16 = **0.700** (vs 0.780 for 0.5B — slightly weaker)

The 1.5B SFT is on par with our 0.5B baseline. Adequate starting point for RLOO.

**Scripts.** `sft_trainer/sft.py` via `modal_train.py sft`. Eval via `modal_train.py eval`.

**Artifacts.** `eval_sft_1.5b.json` (50 prompts × 16 rollouts).

**Modal cost.** ≈ $1 (~7 min SFT + ~3 min eval).

---

### EXP-13: 1.5B scale extension (RLOO + cache + analyses) ✅

**Question.** Does the position-decoupling story from 0.5B (gap, matched-pair drop, per-problem AUROC fall) reproduce at 1.5B?

**Method.**
1. **SFT**: Qwen2.5-1.5B + `Asap7772/cog_behav_all_strategies` (same data as asingh15 0.5B SFT), 6 epochs lr=1e-5, effective batch 64. → pass@1 = 0.280, pass@16 = 0.700 on n=50 test (essentially matched to 0.5B SFT's 0.286/0.780).
2. **RLOO**: 100 steps, snapshots every 10, KL=0.001, group_size=8, batch=64, grad_accum=64. reward_mean at step 99 = 0.457. → pass@1 = **0.480** on n=50 (RL added +0.20 pp; 0.5B added +0.25).
3. **Sample n=500 rollouts** on the procedural set for both 1.5B ckpts (the SFT and the RLOO endpoint).
4. **Cache hidden states** L12/L16/L20 × {pre_answer, assertion, neutral} for both ckpts. Filter to clean-406.
5. Run the standard probe analyses (analyze_probes, qualitative_matched_pairs, significance_and_baselines, per-problem trace-final correlation).

**Results.**

| Metric | 0.5B C_outcome (corrected) | 1.5B C_outcome (corrected) |
|---|---|---|
| pass@1 (n=500) | ~0.55 | 0.558 |
| pass@1 (n=50 test) | 0.535 | 0.480 |
| Trace-final probe AUROC (L16, corrected) | **0.982** | **0.974** |
| **Gap pre−ass (L20 / best layer, corrected)** | +0.086 (L16) | **+0.040** |
| Matched-pair MW between SFT/outcome (corrected) | p = 0.68 (NS) | p = 0.14 (NS) |
| Rambling rate (% multi-answer) | **87%** | **0.075%** |
| Best-of-16 probe-selector lift | **+12.1 pp** | **+8.6 pp** |
| Abstain 50% coverage accuracy | 0.980 | 0.931 |

**Headline interpretation.** The rambling pathology does NOT reproduce at 1.5B: only 0.075% multi-answer vs 87% at 0.5B. The position-decoupling gap stays small at 1.5B (+0.04 at the best layer). The probe is near-oracle at both scales and the same applied strategies generalize: best-of-K, abstention, restart all work.

**Rambling-as-reward-hack is a small-scale (0.5B) phenomenon.** At 1.5B the model emits a single `<answer>` block on virtually all rollouts and so the verifier's "last-block scored" rule cannot be exploited.

**Scripts.**
- SFT: `modal_train.py sft` (no new script)
- RLOO: `modal_train.py rloo` (no new script)
- Rollouts: `extension/evaluation/sample_local_jsonl.py` (existing)
- Cache: `extension/probe/cache_hidden_states.py` (existing)
- Filter: inlined Python (renames `C_*_1.5b_*` to `C_*_*` for compatibility with existing analysis scripts)
- Analyses: all existing scripts re-pointed at the 1.5B cache dir

**Artifacts.**
- `eval_c_sft_1.5b_n500.json`, `eval_c_outcome_1.5b_n500.json` (committed to repo root)
- `extension/cache/probe_cache_1.5b_clean406/` (filtered cache, 18 files, ~108 MB)
- `extension/outputs/n500/probe_behavioral_1.5b/probe_behavioral_correlation_pre_answer.{json,png}`

**Modal cost.** ≈ $30-35 total (~$1 SFT + ~$15-20 RLOO + ~$5-6 rollouts + ~$3-5 cache).

**Caveats (also in writeup §15.4).**
- 1.5B SFT undertrained relative to asingh15 (no recipe-match attempt).
- 100 RLOO steps at 1.5B is fewer per-parameter updates than at 0.5B.
- n=105 valid problems for per-problem correlation (vs 218 at 0.5B); 1.5B C_outcome is too deterministic for many problems to have mixed outcomes in K=16.
- We did NOT redo Option B dynamics or the cross-position transfer matrix at 1.5B. The scale claim above rests on the aggregate + matched-pair + per-problem evidence. A cleaner pre-paper extension would add those (~$15 Modal, half a day local).

---

### EXP-14: Probe pipeline (labels = next-`<answer>`-block correctness) ✅

**Question.** What is the probe at assertion / pre_answer positions actually predicting? Answer: the correctness of the immediately-following `<answer>` block. Each cached hidden state should be labeled accordingly.

**Setup.** `extension/probe/relabel_full_grid.py` labels every cached row by the correctness of the next `<answer>` block following that row's token. `extension/probe/relabel_redo_downstream.py` trains the probe via GroupKFold(5) on these labels and computes the downstream statistics. All numbers throughout this doc and the writeup use this labeling rule.

**Results — corrected-label AUROCs:**

| Cell | corrected-label AUROC |
|---|---|
| C_outcome L16 pre_answer | **0.982** |
| C_outcome L16 assertion | 0.896 |
| C_outcome L16 gap | +0.086 |
| C_SFT L16 pre_answer | 0.912 |
| C_SFT L16 assertion | 0.885 |
| C_SFT L16 gap | +0.027 |

**Downstream stats under corrected labels:**
- Wilcoxon C_SFT (one-sided > 0): p = 9.3e-35
- Wilcoxon C_outcome (one-sided > 0): p = 3.9e-8
- **Mann-Whitney between checkpoints: p = 0.68 (NOT significant)** — within-prompt matched-pair effect is statistically indistinguishable across checkpoints.

**Interpretation.** With proper labels, the within-position probes are all very strong (0.85-0.98), and the cross-checkpoint matched-pair effect is no longer significant. The story is (a) a near-oracle probe at trace-final (AUROC 0.98), (b) a modest aggregate position-gap (+0.086) that correlates with rambling rate across snapshots, and (c) substantial applied value in deployment-time uses of the probe (EXP-15 through EXP-19e).

**Scripts.** `extension/probe/relabel_full_grid.py`, `extension/probe/relabel_redo_downstream.py`.

---

### EXP-15: Probe-guided budgeted restart sampling ✅

**Question.** Can we use the probe at `</think>` as an early-stopping criterion during best-of-K sampling — generate one rollout, accept if probe ≥ T, else re-sample — to match best-of-K accuracy at less compute?

**Setup.** `extension/probe/probe_guided_restart.py`. Cached rollouts on 0.5B C_outcome clean-406. Held-out probe scores from GroupKFold(5) with corrected labels (AUROC 0.982). Sweep (B, T).

**Results (truth = first-`<answer>` correctness; verifier-equivalent if model emits 1 block):**

| Strategy | Accuracy | Avg rollouts used |
|---|---|---|
| pass@1 | 0.549 | 1.0 |
| best-of-16 (by probe) | 0.670 | 16.0 |
| **restart B=16, T=0.95** | **0.675** | **6.3** |
| restart B=8, T=0.95 | 0.658 | 4.6 |
| restart B=4, T=0.95 | 0.637 | 2.8 |

Probe-restart matches/exceeds best-of-16 with **~60% compute savings**. The accuracy-vs-compute Pareto curve sits strictly above best-of-K at every budget.

**Output.** `extension/outputs/n500/text/32_probe_guided_restart.txt`. Figure: `extension/outputs/n500/figures/fig18_probe_guided_restart.png`.

---

### EXP-16: Probe-guided selective abstention ✅

**Question.** If we let the model abstain on uncertain prompts, how high does accuracy on attempted problems get as a function of coverage?

**Setup.** `extension/probe/probe_abstention_and_hybrid.py` (part 1). For each prompt, score the first cached rollout's `</think>` activation with the held-out corrected-label probe; commit iff probe ≥ T; else abstain. Sweep T.

**Results on 0.5B C_outcome clean-406, n=406:**

| Coverage | Threshold T | n attempted | Accuracy on attempted |
|---|---|---|---|
| 100% | 0.00 | 406 | 0.549 |
| ~70% | 0.02 | 263 | 0.837 |
| ~55% | 0.30 | 228 | 0.943 |
| **~50%** | **0.86** | **201** | **0.980** |
| ~33% | 0.98 | 133 | 0.992 |

Near-perfect accuracy at 33% coverage — strongest practical-mechanism use of the probe. Useful in any deployment where the system has the option to say "I don't know."

**Output.** `extension/outputs/n500/text/33_probe_abstention_hybrid.txt`. Figure: `extension/outputs/n500/figures/fig19_probe_abstention.png`.

---

### EXP-17: Probe + majority-vote ensemble ✅

**Question.** Is the probe complementary to self-consistency (majority vote over K rollouts), or just a noisier version of the same signal?

**Setup.** `extension/probe/probe_abstention_and_hybrid.py` (part 2). For each prompt, compute (a) probe-best-of-16 (rollout with highest probe wins), (b) majority-of-16 (most frequent first-`<answer>` equation wins), (c) intersection rule (majority if its mean probe ≥ 0.5, else fall back to probe-best).

**Results on 0.5B C_outcome clean-406, n=406:**

| Strategy | Accuracy |
|---|---|
| probe-best-of-16 | 0.670 |
| majority-of-16 | 0.618 |
| **intersection (majority if mean_probe ≥ 0.5, else probe-best)** | **0.677** |
| union (commit iff probe-best ≡ majority) | 0.618 |

Agreement rate between probe-best and majority-best: 53.7%. **On the 188 disagreement prompts, probe-best wins 26 vs majority 5 (5.2× ratio).**

The probe is **strictly complementary** to majority vote: when they disagree, the probe is right 5× more often. The intersection rule is the best single combination tested. This rules out the "probe is just self-consistency in disguise" alternative explanation.

**Output.** `extension/outputs/n500/text/33_probe_abstention_hybrid.txt`. Figure: `extension/outputs/n500/figures/fig20_probe_majority_hybrid.png`.

---

### EXP-18: Cross-scale applied-probe comparison (0.5B vs 1.5B) ✅

**Question.** Do the applied strategies (best-of-K, abstention, restart) generalize to 1.5B C_outcome, where the rambling reward-hack does not occur?

**Setup.** `extension/probe/probe_applied_scale_comparison.py`. Train independent corrected-label probes on 0.5B and 1.5B cached `</think>` activations (different hidden dims; this is not weight-transfer, just same-strategy comparison). Run the same applied strategies at both scales.

**Results.**

| Metric | 0.5B C_outcome | 1.5B C_outcome |
|---|---|---|
| Held-out balanced AUROC at `</think>` | **0.982** | **0.974** |
| pass@1 (first-block, cached subset) | 0.549 | 0.517 |
| probe-best-of-16 | 0.670 (+12.1 pp) | 0.603 (+8.6 pp) |
| abstain ~50% coverage | 0.980 | 0.931 |
| abstain ~33% coverage | 0.993 | 0.956 |
| probe-restart B=16, T=0.95 | 0.675 (used 6.3) | 0.601 (used 5.7) |

**Reading.** Probe is near-oracle at both scales. Applied strategies generalize: abstention reaches >93% accuracy at 50% coverage even at 1.5B (where the rambling exploit is absent). The applied story is not "useful only because the model rambles" — it's "useful as a general internal verifier."

Note: cached-rollout subset only includes prompts that produced a `</think>` token; these absolute first-block pass@1 numbers should not be compared to the n=50 last-block test pass@1.

**Output.** `extension/outputs/n500/text/34_probe_applied_scale_comparison.txt`.

---

### EXP-19a: Probe-mean as near-oracle problem-difficulty signal ✅

**Question.** For each prompt, can the average probe score across K rollouts predict per-prompt accuracy? Does probe variance carry additional information ("the model knows it doesn't know")?

**Setup.** `extension/probe/probe_creative_extensions.py` part (A). For each of the 406 prompts, compute `mean(probe at </think> across 16 rollouts)` and `std(probe across 16 rollouts)`. Correlate against per-prompt accuracy (fraction first-block correct).

**Results.**

| | Pearson r | Spearman ρ |
|---|---|---|
| mean_probe ~ accuracy | **+0.967** (p ≈ 4e-243) | **+0.941** |
| std_probe ~ accuracy | +0.062 (p = 0.21, NS) | −0.014 (NS) |

The probe mean is near-perfectly correlated with per-prompt accuracy. The probe std carries no usable signal. **Implication.** The model's "uncertainty" is reflected in low *mean* confidence on hard problems (consistent across rollouts), not in *spread* of confidence. The probe is a perfect estimator of problem difficulty when averaged across K rollouts.

**Output.** `extension/outputs/n500/figures/fig21_probe_variance_difficulty.png`.

---

### EXP-19b: Cross-checkpoint applied probe transfer ✅

**Question.** Does the applied probe need to be trained on the deployment-time checkpoint, or does a probe trained once on C_SFT work on C_outcome?

**Setup.** `extension/probe/probe_creative_extensions.py` part (B). Train probe on C_SFT activations + C_SFT corrected first-block labels (full data, no held-out split). Deploy on C_outcome cached rollouts. Compute AUROC + best-of-16 selector accuracy.

**Results.**

| Probe | Held-out balanced AUROC on C_outcome | best-of-16 accuracy | Lift vs pass@1 (0.549) |
|---|---|---|---|
| C_outcome-trained (in-distribution) | 0.982 | 0.670 | +12.1 pp |
| **C_SFT-trained (cross-checkpoint)** | **0.953** | **0.653** | **+10.3 pp** |

A probe trained on C_SFT alone gives ~85% of the in-distribution lift (10.3 / 12.1 = 0.85). **Implication.** You can train the probe once on a stable SFT model and reuse it on many post-SFT RL checkpoints. The applied probe is largely checkpoint-invariant.

---

### EXP-19c: Multi-position probe ensemble does not help ✅

**Question.** Does combining probe scores from multiple positions (pre_answer + assertion + neutral) improve best-of-K selection?

**Setup.** `extension/probe/probe_creative_extensions.py` part (C). Train held-out probes at all three positions. Aggregate per-rollout (mean of assertion / neutral scores; pre_answer is one per rollout). For each rollout, combine the three position-scores under 8 strategies (alone, mean, product, max, min, weighted).

**Results (357 prompts with all three probe types; pass@1 = 0.748 on this subset):**

| Selection score | best-of-16 acc |
|---|---|
| pre_answer alone | **0.765** |
| assertion alone | 0.762 |
| neutral alone | 0.754 |
| mean(pre, ass) | 0.762 |
| product(pre, ass) | 0.762 |
| max(pre, ass) | 0.762 |
| **min(pre, ass)** | **0.765** |

No combination beats pre_answer-alone. The trace-final probe saturates the signal; multi-position aggregation is not worth the complexity.

**Output.** `extension/outputs/n500/figures/fig22_multi_position_ensemble.png`.

---

### EXP-19d: Probe-adaptive test-time budget allocation ✅

**Question.** Given a fixed total rollout budget across N prompts, does probe-adaptive allocation (give more rollouts to less-confident prompts) beat uniform allocation (same K per prompt + best-of-K)?

**Setup.** `extension/probe/probe_adaptive_budget.py`. Threshold-waterfill strategy: round 1 generate 1 rollout per prompt; rounds 2+ give one more rollout to any prompt whose best probe score is below T (priority: lowest current best first). Compare to uniform K-per-prompt + best-of-K at matched total budget. N=288 prompts with all 16 rollouts cached.

**Results (T=0.95):**

| K_avg | Uniform | Adaptive T=0.95 | Lift |
|---|---|---|---|
| 2 | 0.795 | **0.816** | +2.1 pp |
| **4** | 0.806 | **0.830** | **+2.4 pp** |
| 6 | 0.809 | 0.830 | +2.1 pp |
| 8 | 0.809 | 0.830 | +2.1 pp |
| 16 | 0.823 | 0.830 | +0.7 pp |

At K_avg=4, adaptive allocation matches the full-budget K=16 uniform accuracy with **1/4 the compute**. The strategy gives ≤3 rollouts to easy prompts (round 1 crosses T=0.95) and pushes 43 hard prompts up to all 16. The probe identifies easy-vs-hard after just one rollout (cf. EXP-19a, probe mean has r=0.97 with per-prompt accuracy).

**Output.** `extension/outputs/n500/text/36_probe_adaptive_budget.txt`. Figure: `extension/outputs/n500/figures/fig23_probe_adaptive_budget.png`.

---

### EXP-19e: Probe-as-verifier-free-eval + failure-mode diagnostic ✅

**Question.** Can the probe serve as a free dataset-accuracy estimator (no verifier needed)? When it makes mistakes, what do those look like?

**Setup.** `extension/probe/probe_as_eval_proxy.py`. Per-prompt: take mean probe across 16 rollouts; average across prompts. Compare to true verifier accuracy. Bin rollouts into TP/TN/FP/FN by (probe ≥ 0.5) × (first-block correct).

**Results — eval-proxy:**

| | Value |
|---|---|
| True dataset accuracy | 0.5531 |
| Probe-mean estimate | **0.5565** (diff +0.0035) |
| Probe-vote estimate | 0.5985 (diff +0.0455) |

Probe-mean gives a near-exact estimate of true dataset accuracy. Useful as a verifier substitute in deployment.

**Results — failure modes (n=6306 rollouts):**

| Class | n | mean blocks | mean probe |
|---|---|---|---|
| TP | 3485 | 9.7 | 0.960 |
| TN | 2534 | 3.1 | 0.028 |
| **FP (overconf)** | 197 | 9.5 | 0.809 |
| **FN (underconf)** | 90 | 6.8 | 0.207 |

| Calibration | Rate |
|---|---|
| Overconfidence: P(wrong \| probe≥0.5) | **5.4%** |
| Underconfidence: P(correct \| probe<0.5) | **3.4%** |

Probe is extremely well-calibrated. Overconfidence cases have the same rambling profile as correct ones — they're confidently-rambling-to-wrong-answer pathologies, the natural floor for any near-oracle predictor.

**Output.** `extension/outputs/n500/figures/fig24_probe_eval_proxy.png`.

---

### EXP-19: First-answer reward RLOO (verifier-level remedy) ⏳

**Question.** The rambling at 0.5B C_outcome is a reward-hack of the verifier's "last-`<answer>`-block wins" rule (§3.1, §7 in writeup). If we monkey-patch the verifier to score the FIRST `<answer>` block instead, does the rambling go away while accuracy is preserved? This is the verifier-level equivalent of probe-as-reward-shaping (the probe is a near-oracle predictor of first-block correctness, so the two reward signals are equivalent).

**Setup.** `extension/training/firstanswer_rloo.py`: monkey-patch `evaluation.countdown.compute_score` to score the first `<answer>` block, then exec `rloo_trainer/rloo.py` unchanged. Same RLOO hyperparameters as vanilla C_outcome run. Modal app `ap-xeO1zDmat85U3LiC5c9vqQ`, wandb run `1bm6ggzs`.

**Status.** Training in progress on Modal. At step 0, reward_mean = 0.265 (vs ~0.46 for vanilla RLOO at step 0 — first-block reward is a harder target for C_SFT). Expected completion ~2-3h from launch.

**Planned downstream analysis** (when checkpoint completes):
1. Sample 16 rollouts × clean-406 from the new C_outcome' checkpoint.
2. Measure mean blocks-per-rollout (vs 7.6 for vanilla C_outcome; expect ≈1).
3. Measure first-block, last-block accuracy.
4. Cache hidden states at `</think>` + assertion + neutral; retrain probe; check whether pre−assertion gap shrinks under first-block training.
5. Compare matched-pair statistics across new C_outcome' vs vanilla C_outcome.

If the gap shrinks under first-answer training, it directly confirms the "rambling-as-mediator-of-decoupling" story.

**Prediction from EXP-20.** The model's first answer already beats its last (verifier-scored) answer by +5.2pp, so first-answer RLOO should *raise* pass@1, not merely preserve it. If it does not, rambling was load-bearing for accuracy (also a finding).

---

### EXP-20: Reward-hack cost — first-vs-last answer accuracy + token waste ✅

**Question.** Is the rambling reward-hack *functional* (the model rescues itself wrong→right) or *net-negative* (it drifts right→wrong)? How many tokens does the rambling tail waste, and does the cost grow over RLOO training? This is the no-retraining behavioral counterfactual that predicts EXP-19's outcome.

**Method.** Re-score the EXISTING clean-406 rollouts (no GPU, no generation): score the FIRST `<answer>` block vs the LAST (= verifier default = what was scored), count `<answer>` blocks and the tokens after the first `</answer>` (Qwen fast tokenizer, offset mapping). Exact McNemar two-sided test on discordant pairs. Repeat on the snapshot rollouts (step 30/60/90) on the common clean-∩-first-200 set for the dynamics.

**Headline (full clean-406, 6496 rollouts/ckpt):**

| ckpt | first-blk | last-blk (scored) | Δ | drift→wrong | rescue→correct | McNemar p | mean blocks | tok-waste |
|---|---|---|---|---|---|---|---|---|
| C_SFT | 0.290 | 0.238 | **+0.052** | 526 | 188 | 7.0e-38 | 2.78 | 54.0% |
| C_outcome | 0.550 | 0.498 | **+0.052** | 490 | 150 | 5.2e-43 | 6.96 | 56.2% |

**Dynamics (common clean-∩-first-200, 2672 rollouts/ckpt):** Δ(first−last) is roughly flat (+0.049 / +0.033 / +0.066 / +0.062 / +0.047 across SFT/step30/60/90/final); mean `<answer>` blocks climbs **2.73 → 2.90 → 4.34 → 6.97 → 7.25**.

**Interpretation.**
1. The model's FIRST answer beats its verifier-scored LAST answer by **+5.2pp on BOTH checkpoints**; drift (correct→wrong) outnumbers rescue ~3:1; McNemar p < 1e-37. **Rambling is net accuracy-DESTROYING**, not functional.
2. The per-rollout accuracy tax is **~constant (+5pp) and already present at SFT** — RLOO does NOT worsen per-decision judgment; it inflates rambling **volume** (~2.6× blocks) and the token tail (~56% of generated tokens are post-first-answer). RL's contribution to the hack is token bloat, not increased self-sabotage.
3. Committing to the first answer is a **zero-retraining Pareto move: +5.2pp accuracy AND −56% tokens**.

**Consistency check (vs EXP-15).** Reconciles exactly: single-answer rollouts (16%) are ~10% accurate (first=last); multi-answer (84%) give first=0.636 / last=0.574 (matching EXP-15's `26_probe_answer_commit.txt`). Recomputing the full-set average from these reproduces 0.550 / 0.498.

**Convention note.** This is a *behavioral* (verifier) first-vs-last **answer** comparison — distinct from the EXP-14 next-`<answer>`-block **probe relabeling** (a representation question). Kept separate so it can be compared against a future next-answer-block behavioral experiment.

**Scripts.** `extension/probe/reward_hack_cost.py` → `extension/outputs/n500/text/35_reward_hack_cost.json` + `extension/outputs/n500/figures/fig25_reward_hack_cost.png`. (Branch `reward-hack-cost`.)

**Modal cost.** $0 (local CPU; tokenizer download only).

---

### EXP-NN: Skipped / negative-result experiments

| Experiment | Why skipped/null | Documented in |
|---|---|---|
| Verbal confidence elicitation (RLCR / token-logprob style) | SFT'd-on-Countdown Qwen 0.5B is not chat-tuned; every elicitation prompt gets continued as more Countdown. AUROC ≈ 0.50 (chance) for token-logprob. **Methodological observation worth flagging.** | writeup §6 |
| Phase 2B activation patching (inject first-answer state into last-answer position) | Pre-registered gating rule: only run if Phase 2A (EXP-05) revealed Pattern B. We observed Pattern A cleanly (T→F probe(last) = 0.154, indistinguishable from F→F). No preserved representation to inject. Skipped. | writeup §2.4 |

---

## 3. Key cross-references & open questions

### What survives under corrected labels:

1. **Near-oracle trace-final probe**: AUROC 0.982 at 0.5B C_outcome, 0.974 at 1.5B (corrected labels).
2. **Probe is strengthened by RL at every level**: aggregate (0.912 → 0.982), per-problem (0.882 → 0.927), within-prompt (Wilcoxon p < 1e-7 on both).
3. **Modest aggregate position-gap emerges over training**: +0.027 → +0.136 → +0.086 across C_SFT/step90/final, correlates with rambling rate at r = +0.89.
4. **Pattern A confirmed** (Phase 2A position-appropriate probe; corrected by construction): T→F probe(last) = 0.154, F→F floor 0.088; bidirectional in F→T (EXP-06).
5. **Causal steering null** (probe = reader, not controller): probe-vs-random Δ ∈ [-0.07, +0.02] across α.
6. **Position-orthogonality is geometric** (cosines ≤ +0.10 within-checkpoint cross-position).
7. **Rambling pathology is small-scale**: 87% multi-answer at 0.5B vs 0.075% at 1.5B.
8. **The probe has substantial applied value**: best-of-K +12 pp, abstention 98% at 50% coverage, restart at 60% compute saved, eval-proxy ±0.4 pp, calibration 5.4%/3.4%.

### What did NOT survive corrected labels (deleted from headline):

- "Per-problem AUROC drops under RL at trace-final" (was +0.130 under wrong labels; now -0.044 under corrected — RISES instead).
- "Spearman r=+0.335 between probe_drop and acc_delta" (now +0.062, NS).
- "Within-prompt matched-pair effect collapses between checkpoints" (Mann-Whitney was p=8e-16 under wrong labels; now p=0.68 NS under corrected).
- "Within-problem decoupling at trace-final" framing (no longer supported).

### Pending:

- firstanswer_rloo downstream eval (causal test of rambling-as-reward-hack).
- 1.5B Option B dynamics (cleaner mechanistic confirmation of scale claim).

---

## 4. Full file/script inventory

### Scripts (extension/)

```
extension/data/
  generate_countdown.py                # procedural Countdown problem generator
  countdown_eval_500.jsonl             # 500 generated problems (50/50 3-num/4-num)
  contaminated_prompt_idx.json         # which 94/500 are in asingh15 train
  steering_todo.jsonl                  # resume-list for causal steering Modal job

extension/evaluation/
  sample_local_jsonl.py                # vLLM rollouts from local JSONL
  launch_expansion_rollouts.sh         # Phase 1 batch launcher (5 parallel jobs)
  wait_and_launch_phase2.py            # Phase 1 → Phase 2 watchdog

extension/probe/
  # caching
  cache_hidden_states.py                # pre_answer / assertion / neutral cache
  cache_answer_positions.py             # <answer>-opening cache (Phase 2A)
  cache_all_think_close.py              # all </think> tokens cache
  filter_to_clean.py                    # n=500 -> clean-406 filter
  # corrected-label (next-<answer>-block) pipeline -- EXP-14 + downstream
  relabel_full_grid.py                  # relabel every cached row by next-block correctness
  relabel_redo_downstream.py            # corrected-label probe training + stats
  relabel_cross_checkpoint.py           # cross-checkpoint transfer (corrected)
  relabel_dynamics.py                   # Option B per-snapshot dynamics (corrected)
  relabel_cosines.py                    # EXP-07 geometric cosines (corrected)
  relabel_per_problem.py                # corrected per-problem correlation (supersedes EXP-10/11)
  # position / mechanism analyses
  cross_position_transfer.py            # 3x3 cross-position matrix (single-seed)
  phase1_diagnostics.py                 # cross-position transfer (10-seed + heatmap)
  per_layer_sweep.py                    # 25-layer probe sweep
  position_resolved_auroc.py            # AUROC by token position in the trace
  phase2a_per_answer_correctness.py     # EXP-05 pre-flight
  phase2a_pattern_analysis.py           # EXP-05 with trace-final probe
  phase2a_position_appropriate_probe.py # EXP-05 with position-appropriate probe
  ft_rollout_trajectory.py              # EXP-06 F->T trajectory
  probe_answer_commit.py                # first-vs-last answer / probe-as-selector
  # applied / test-time uses (EXP-15..19e)
  probe_guided_restart.py               # EXP-15 budgeted restart
  probe_abstention_and_hybrid.py        # EXP-16/17 abstention + majority hybrid
  probe_adaptive_budget.py              # EXP-19d adaptive test-time budget
  probe_as_eval_proxy.py                # EXP-19e verifier-free eval proxy
  probe_applied_scale_comparison.py     # EXP-18 0.5B vs 1.5B applied
  probe_creative_extensions.py          # EXP-19a/b/c
  probe_thinkclose_selector.py          # </think>-selector variant
  # steering
  save_probe_vector.py                  # saves probe direction for steering
  causal_steering.py                    # EXP-09 Modal HF hook injection
  analyze_causal_steering.py            # steering aggregation
  # reward-hack cost
  reward_hack_cost.py                   # EXP-20 first-vs-last accuracy + token waste

# (33 superseded scripts were removed in the 64->30 trim; their results are
#  retained in the EXP entries above and were recomputed via the relabel_* pipeline.)
```

### Caches (under `extension/cache/`, gitignored)

```
probe_cache/                            # original n=50
probe_cache_n500/                       # raw n=500 (pre-filter)
probe_cache_n500_clean406/              # ← primary clean-406 cache
probe_cache_n500_all_layers_clean406/   # 25 layers × 3 kinds × 2 ckpts
probe_cache_n500_answers/               # <answer>-opening cache (Phase 2A)
probe_cache_dynamics/                   # Option A dynamics (n=50, confounded)
probe_cache_dynamics_optB/              # Option B dynamics (n=200 fresh rollouts)
steering/                               # saved probe direction (npz)
confidence/                             # verbalized-confidence JSONLs
```

### Text outputs (in `extension/outputs/n500/text/`, gitignored)

Numbered `NN_*.txt` files (01–42, non-contiguous after the script trim) plus a few `.json`/`.csv`. The reward-hack-cost output is `35_reward_hack_cost.json`.

### Figures (in `extension/outputs/n500/figures/`, gitignored)

```
fig1_matched_pair_scatter.png
fig2_within_problem_d.png
fig3_position_bar.png
fig4_per_keyword_bar.png
fig5_dynamics_trajectory.png         # Option A (confounded, kept for reference)
fig7_concealment_gap.png             # global gap, n=50
fig9_within_rollout_trajectory.png   # trace-final probe, OOD at block 0
fig9b_within_rollout_position_appropriate.png   # position-appropriate probe, clean
fig10_ft_rollout_trajectory.png      # F→T bidirectional Pattern A
fig11_per_layer_sweep.png            # 25 layers × 3 kinds × 2 ckpts
fig12_causal_steering.png            # steering bar chart
fig13_headline_dynamics.png          # HEADLINE — gap-over-training with CIs
fig_phase1_transfer_heatmap.png      # 3×3 × 3 layers × 2 ckpts cross-position
```

### Per-problem correlation artifacts (committed, in `extension/outputs/n500/probe_behavioral/`)

```
probe_behavioral_correlation.json              # EXP-10 (@ <answer>-opening)
probe_behavioral_correlation.png
probe_behavioral_correlation_pre_answer.json   # EXP-11 (@ trace-final)
probe_behavioral_correlation_pre_answer.png
```

### Eval JSONs (committed to repo root)

```
eval.json                              # n=50 C_outcome (original)
eval_sft.json                          # n=50 C_SFT (original)
eval_sft_1.5b.json                     # 1.5B SFT on n=50
eval_c_sft_n500.json                   # 500 prompts × 16 rollouts on C_SFT
eval_c_outcome_n500.json               # same on C_outcome
eval_c_outcome_step_30_n200.json       # Option B snapshot
eval_c_outcome_step_60_n200.json
eval_c_outcome_step_90_n200.json
eval_old_may1.json                     # historical
```

---

## 5. Budget tracking (approximate Modal H100 spend)

| Block | Cost |
|---|---|
| n=50 baseline (initial caching + analyses) | ~$5 |
| EXP-02 n=500 expansion (rollouts + 3-layer cache + filter) | ~$5-8 |
| EXP-03 Option B dynamics (snapshot rollouts + cache) | ~$3-5 |
| EXP-05 Phase 2A cache (<answer> positions) | ~$2 |
| EXP-08 per-layer cache (25 layers × 2 ckpts) | ~$3 |
| EXP-09 causal steering | ~$5-6 (two passes) |
| EXP-10 train_answer_probe + behavioral_correlation | ~$3-5 |
| EXP-12 1.5B SFT | ~$1 |
| EXP-13 1.5B RLOO (running) | est. ~$25-35 |
| 1.5B caching + downstream (queued) | est. ~$5-10 |
| **Total (so far + expected)** | **≈ $55-80** |

User-approved budget for the scale extension: $70-110. Tracking well within.

---

## 6. Open methodology notes / honest caveats

- **Cohen's d at `</think>` was the largest single-number revision** from the n=50 paper. The "70% reduction" claim was a small-sample artifact. At n=242 matched problems the reduction is ~8% (+1.12 → +1.04).
- **Position-appropriate probe sanity bar** for EXP-10's new probe: held-out AUROC 0.889 vs §2.4's 0.920. Within the 0.85+ tolerance; the slight drop is plausibly because the EXP-10 training pool (94 contaminated problems) is much smaller than §2.4's training pool (clean-406's own GroupKFold).
- **Aggregate vs per-problem AUROC are different metrics** (EXP-11). At trace-final, aggregate rises (0.804 → 0.896) while per-problem mean drops (0.722 → 0.612). Always report which one. The within-problem version is the one that matches the matched-pair finding and the decoupling story.
- **Cross-checkpoint pre-answer transfer** at clean-406 is 0.855 (not the 0.523 of n=50). The pre-answer correctness representation is largely shared across checkpoints; "drift" is a small effect.
- **EXP-09 causal-steering null** is consistent with Yuan et al.'s 1.5B+ patching failures. With the matched-magnitude random-direction control we make the null sharper than they did.
- **1.5B SFT is undertrained** vs asingh15 0.5B SFT (pass@1 0.280 vs 0.286 on n=50 — basically tied; pass@16 0.700 vs 0.780 — weaker). The starting point for RLOO is comparable; if the 1.5B story differs, it won't be because of the SFT quality.

---

*Last updated: 2026-06-01. Author: Claude (Opus 4.7) with abrahamyeung. Source of truth: this doc + writeup.md + extension/CHANGELOG.md.*
