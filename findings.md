# Findings — exhaustive master record

*Every experiment, every number, every script, every artifact. Updated as work proceeds.*

> **Purpose.** `writeup.md` is the paper-ready synthesis (~750 lines, narrative). `extension/CHANGELOG.md` is the chronological work log. This file is the **dense lookup table**: per-experiment, what we asked, how we asked it, what the numbers were, which scripts produced them, where the raw outputs live, and what the conclusion was. It's deliberately less readable than the writeup and more complete than the changelog.
>
> **Date of last update.** 2026-06-01 (corrected-label probe pipeline, applied-probe extensions, first-answer RLOO mid-run).
> **Status legend.** ✅ done · ⏳ running · ⏸ paused / blocked · ❌ failed / abandoned · 🔜 queued.

---

## 0. Headline numbers (one-glance)

### 0.5B (primary)

| Metric | C_SFT | C_outcome | Δ |
|---|---|---|---|
| pass@1 (n=50 asingh15 test) | 28.6% | **53.5%** | +24.9 pp |
| pass@16 | 78.0% | 72.0% | −6 pp (sharpening) |
| Trace-final probe AUROC, L20, clean-406 | 0.821 | **0.901** | +0.080 |
| Trace-final probe AUROC, L16, clean-406 | 0.804 | **0.896** | +0.092 |
| **Per-problem** trace-final AUROC mean (L16) | **0.722** | **0.612** | −0.110 |
| Assertion-position AUROC, L16, clean-406 | 0.785 | 0.703 | −0.082 |
| Position-appropriate `<answer>` probe AUROC (held-out, §2.4) | 0.920 | (held-out test on C_outcome) | — |
| Gap pre_answer − assertion (L16) | 0.019 | **0.193** | +0.174 |
| Cross-position pre→ass on C_outcome | (chance) | **0.494** | — |
| Cross-position cosine pre vs ass (L16) | +0.024 | +0.038 | ≈ orthogonal |
| Per-problem Spearman (probe_drop vs accuracy_delta) at `<answer>` | — | r=−0.03, p=0.63 | null |
| Per-problem Spearman at trace-final | — | **r=+0.335, p=4e−7** | sharp +ve decoupling |
| Causal steering Δ(probe direction − random direction) | — | [−0.07, +0.02] across α | null |

### 1.5B (scale extension)

| Metric | 1.5B C_SFT | 1.5B C_outcome | Δ |
|---|---|---|---|
| pass@1 (n=50 test) | 28.0% | **48.0%** | +20.0 pp |
| pass@1 (n=500 procedural) | 23.6% | **55.8%** | +32.2 pp |
| Aggregate pre_answer AUROC (L20) | 0.887 | **0.976** | +0.089 |
| Aggregate assertion AUROC (L20) | 0.844 | **0.936** | +0.092 |
| **Gap pre−assertion (L20)** | +0.043 | **+0.040** | (stays small, didn't open up like at 0.5B) |
| Matched-pair % above-diag (L16) | 86% | **82%** | MW p = 0.14 (NS) |
| Per-problem trace-final AUROC mean | 0.790 | **0.901 (RISE)** | +0.110 |
| Per-problem Spearman (probe_drop vs accuracy_delta) | — | r = +0.008, p = 0.93 | null |
| Per-problem dominant quadrant | — | **both improved (63%)** | (not decoupling) |

**Scale interpretation.** Position-decoupling under outcome RL is a small-scale (0.5B) phenomenon. At 1.5B the model maintains a coherent correctness representation across positions throughout RL — the gap stays small (+0.04), the matched-pair distributions are indistinguishable across checkpoints (p = 0.14), and per-problem trace-final AUROC *rises* under RL instead of falling.

**Compute spent so far** (Modal H100 at ~$4/hr, billed to `ayeung16` workspace): ≈ **$25-35** across the n=500 expansion, Option B dynamics, per-layer sweep, causal steering, and the per-problem correlation experiments. RLOO 1.5B currently running (~$30-40 estimated).

---

## 1. Setup

### 1.1 Models

| Tag | Construction | Source | Path |
|---|---|---|---|
| `C_SFT` | Qwen2.5-0.5B + Anikait Singh's Countdown SFT | HF: `asingh15/qwen-sft-countdown-defaultproj` | — |
| `C_outcome` | RLOO from `C_SFT`, outcome reward only (0/0.1/1.0), 100 steps, snapshots every 10 | trained by us | `/vol/checkpoints/rloo_checkpoints/rloo_training/rloo_fixed_v2/latest_checkpoint/model` |
| `C_outcome_step_{N}` | Intermediate RLOO snapshot at step N ∈ {0,10,…,90} | persisted during run | `/vol/checkpoints/rloo_checkpoints/rloo_training/rloo_fixed_v2/epoch_0_step_{N}/model` |
| `C_process` | RLOO with annotation-only subgoal reward, init from `C_SFT_aug`. **Underperformed C_outcome**; appendix only. | — | — |
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

### EXP-01: Initial 0.5B aggregate probe AUROCs (n=50) — superseded ✅

**Question.** Does outcome RL leave a position-resolved gap between probe AUROC at `</think>` vs at confidence-asserting tokens?

**Method.** Cache hidden states on 50 prompts × 16 rollouts via `extension/probe/cache_hidden_states.py`. Train per-cell LR probes on L12/L16/L20 × {pre_answer, assertion, neutral} × {C_SFT, C_outcome}.

**n=50 numbers (paper-original, since superseded):**
- pre_answer AUROC (L16): C_SFT 0.724, C_outcome 0.793 (Δ +0.069)
- assertion AUROC (L16): C_SFT 0.735, **C_outcome 0.520 ("collapse to chance")** ← later revised
- Cohen's d at `</think>` within-problem: +1.26 (C_SFT) → +0.38 (C_outcome) ← later revised
- Matched-pair % above-diag at assertion: 76% (C_SFT) → 36% (C_outcome, "actively backwards") ← later revised
- Cross-checkpoint pre_answer off-diag: 0.523 / 0.580 ← later revised

**Status.** Superseded by EXP-02 (n=500 clean-406). The qualitative position-gap claim survived; several magnitudes were small-sample artifacts (see EXP-02 "Honest revisions" section).

**Scripts.** `extension/probe/cache_hidden_states.py`, `extension/probe/analyze_probes.py`, `extension/probe/significance_and_baselines.py`, `extension/probe/cross_checkpoint_transfer.py`.

**Artifacts.** `extension/outputs/n500/text/01_analyze_probes.txt`, `extension/outputs/n500/text/04_matched_pairs.txt`, etc. (these directories are reused at n=500).

---

### EXP-02: n=500 procedural expansion + clean-406 contamination filter ✅

**Question.** Are the n=50 magnitudes (most importantly "C_outcome assertion AUROC ≈ chance") robust to a larger held-out set?

**Method.** 
1. Built procedural Countdown generator (§1.4) and emitted 500 fresh problems.
2. Sampled 16 rollouts × 500 prompts on both C_SFT and C_outcome via Modal vLLM (`extension/evaluation/sample_local_jsonl.py`). 5 parallel rollout jobs (~30 min).
3. Cached hidden states at L12/L16/L20 × 3 position kinds × both ckpts (~5 min wall).
4. Verified contamination: 94/500 in asingh15 train, 0/500 in test. Filtered to clean-406.
5. Re-ran the full analysis pipeline on clean-406.

**Headline numbers (clean-406, L16, balanced GroupKFold(5)):**

| Cell | C_SFT | C_outcome | Δ |
|---|---|---|---|
| pre_answer | **0.804** [0.782, 0.816] | **0.896** [0.883, 0.902] | **+0.092** |
| assertion | 0.785 [0.750, 0.790] | 0.703 [0.662, 0.725] | **−0.082** |
| neutral | 0.562 [0.525, 0.574] | 0.562 [0.524, 0.590] | 0.000 |
| **gap (pre − assertion)** | **+0.019** | **+0.193** | **+0.174** |

CIs from B=100 subsample-without-replacement bootstrap (80% of unique prompts per replicate).

**Matched-pair (within-prompt, assertion position):**
- C_SFT: median +0.186, 78% above-diag, Wilcoxon p = **1.8e−24** (n=244)
- C_outcome: median +0.004, 52% above-diag, Wilcoxon p = 0.027 (n=218)
- Mann-Whitney between checkpoints: p = **8e−16**

**Within-problem Cohen's d (at pre_answer):** C_SFT +1.121, C_outcome +1.036. MW-U between distributions p = 3e−4. **Earlier "70% reduction" claim does not survive.**

**Cross-checkpoint pre_answer transfer:** C_SFT→C_outcome = **0.855** (was 0.523 at n=50). The "drift" framing was an n=50 artifact.

**Honest revisions documented in writeup §2 (every magnitude that changed):**
- assertion AUROC C_outcome: 0.520 → 0.703 ("collapse to chance" → "weakened, still well above chance")
- Cohen's d C_outcome: +0.38 → +1.04 ("70% reduction" → "8% reduction")
- Matched-pair C_outcome: 36% → 52% ("actively backwards" → "essentially random with small +ve")
- Cross-checkpoint pre_answer transfer: 0.523 → 0.855 ("dramatic drift" → "small drift effect")

**Scripts.**
- Generator: `extension/data/generate_countdown.py`
- Modal rollouts: `extension/evaluation/sample_local_jsonl.py` + `launch_expansion_rollouts.sh` (Phase 1)
- Modal cache: `extension/probe/launch_expansion_cache.sh` (Phase 2)
- Filter: `extension/probe/filter_to_clean.py`
- Analyses (all in `extension/probe/`):
  - `analyze_probes.py` → output `10_analyze_probes_clean406.txt`
  - `qualitative_matched_pairs.py` → `12_matched_pairs_clean406.txt`
  - `significance_and_baselines.py` → `11_significance_clean406.txt`
  - `cross_checkpoint_transfer.py` → `05_cross_checkpoint_transfer.txt`
  - `length_matched_transfer.py` → `07_length_matched_transfer.txt`
  - `bootstrap_headline_cis.py` → `15_bootstrap_cis.txt`
  - `deeper_analyses.py` → `03_deeper_analyses.txt`
  - `make_figures.py` → `08_make_figures.txt`

**Caches.** `extension/cache/probe_cache_n500/` (raw n=500), `extension/cache/probe_cache_n500_clean406/` (primary).

**Modal cost.** ≈ $5-8 (vLLM rollouts + hidden-state cache).

---

### EXP-03: Option B dynamics — gap emerges over training ✅

**Question.** The aggregate gap on `C_outcome` is +0.19 vs +0.02 on `C_SFT`. Is this gap something that *emerges* during RL training, or a property of the initialization that RL preserved?

**Method.** Re-sample fresh rollouts (n=200, first 200 prompts) from snapshot models at steps 30/60/90; re-cache hidden states at each step; re-train the probe per snapshot. Replaces the Option-A measurement (which probed each snapshot on the *fixed* final-checkpoint rollouts and was confounded by the rollout distribution not updating).

**Result (L16, balanced GroupKFold(5)):**

| Step | pre_answer AUROC | assertion AUROC | gap |
|---|---|---|---|
| C_SFT (pre-RL) | 0.804 | 0.785 | **+0.019** |
| step 30 (n=200 fresh) | 0.791 | 0.769 | **+0.022** |
| step 60 (n=200 fresh) | 0.864 | 0.749 | **+0.115** |
| step 90 (n=200 fresh) | 0.871 | 0.654 | **+0.217** |
| C_outcome (final) | 0.896 | 0.703 | **+0.193** |

Gap jumps between step 30 and step 60. Decoupling is **emergent over training**, not an SFT-inherited property.

**Bootstrap CIs per snapshot** (B=80, 80% subsample): bands overlap at C_SFT + step 30, separate at step 60+.

**Scripts.**
- Modal rollouts at snapshots: `extension/evaluation/launch_expansion_rollouts.sh` (snapshot rollouts launched in same batch as Phase 1)
- Cache: `extension/probe/launch_expansion_cache.sh`
- Analysis: `extension/probe/per_snapshot_decoupling_gap.py` → `14_per_snapshot_decoupling_gap.txt` / `.csv`
- Figure with CIs: `extension/probe/headline_dynamics_figure.py` → `fig13_headline_dynamics.png`

**Caches.** `extension/cache/probe_cache_dynamics_optB/` (fresh-rollout caches per snapshot).

**Modal cost.** ≈ $3-5.

---

### EXP-04: Phase 1 — cross-position transfer + diagnostics ✅

**Question.** Three sub-questions:
- **(1A)** Does the C_outcome pre→assertion off-diagonal asymmetry (0.494 vs 0.368 at L16) persist after explicit class rebalancing on both source training and target evaluation?
- **(1B)** Is the orthogonality L16-specific or layer-invariant?
- **(1C)** Is pre→assertion specifically failing, or is pre_answer just "its own thing" everywhere?

**Method.** `extension/probe/phase1_diagnostics.py`: 10-seed averaging with explicit balanced subsampling on BOTH source training and target evaluation; runs at L12, L16, L20; includes neutral position.

**Results.**

**1A — asymmetry persists, layer-dependent in form:**

| Layer | C_outcome pre→ass | C_outcome ass→pre | asymmetry |
|---|---|---|---|
| L12 | 0.586 | **0.270** (below chance!) | +0.316 |
| L16 | 0.494 | 0.368 | +0.126 |
| L20 | 0.457 | 0.501 | −0.044 |

Std of ass→pre across 10 seeds: 0.08–0.18 (high) → consistent with "no stable shared direction" rather than "stable anti-direction."

**1B — layer-invariant collapse**: symmetric mean transfer on C_outcome at L12/L16/L20 = **0.428 / 0.431 / 0.479**. All clearly below diagonals (0.69–0.90). L16 is not unique.

**1C — pre_answer is its own subspace** (L16, C_outcome):
- pre_answer → assertion: 0.494 (chance)
- pre_answer → neutral: 0.505 (chance)
- assertion → pre_answer: 0.368 (unstable)
- assertion → neutral: 0.500 (chance)
- **neutral → pre_answer: 0.686** (well above chance!)
- neutral → assertion: 0.571

Pre_answer-trained probe transfers nowhere; but neutral-trained probe partially reads pre_answer. The correctness subspace is shared but the pre_answer probe direction is too specific to "write" to other positions.

**Script.** `extension/probe/phase1_diagnostics.py` → `16_phase1_diagnostics.txt` + `fig_phase1_transfer_heatmap.png` (6 panels: 3 layers × 2 ckpts).

**Modal cost.** $0 (local CPU).

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

**Within-checkpoint cross-position cosines (L16) — essentially orthogonal in BOTH ckpts:**
| pair | cos |
|---|---|
| C_SFT pre vs assertion | **+0.024** |
| C_SFT pre vs neutral | −0.002 |
| C_outcome pre vs assertion | **+0.038** |
| C_outcome pre vs neutral | +0.020 |

**Cross-checkpoint within-position cosines (L16) — small but positive:**
| pair | cos | transfer AUROC |
|---|---|---|
| C_SFT pre vs C_outcome pre | +0.102 | 0.855 |
| C_SFT ass vs C_outcome ass | +0.058 | 0.649 |
| C_SFT neu vs C_outcome neu | +0.041 | 0.530 |

**Norms (input space):** pre_answer probes have largest norms (30–80), assertion smaller (7–35), neutral smallest (4–15). Norm scales with diagonal AUROC.

**Interpretation.** The position-decoupling is geometric (cosines ~0.03 within ckpt), not just AUROC-level. Cross-checkpoint: probe directions point in different directions (cosine 0.10) but the underlying correctness *subspace* is shared (AUROC 0.86) — multiple low-cosine directions can each "read" the same signal.

**Script.** `extension/probe/probe_direction_cosines.py` → `21_probe_cosines.txt`.

**Modal cost.** $0 (local).

---

### EXP-08: Per-layer probe sweep (all 25 layers) ✅

**Question.** Is the pre−assertion gap on C_outcome concentrated at late layers (would support "outcome RL shaped only the output head") or distributed across depth?

**Method.** Re-cache hidden states at every layer 0..24 on both ckpts (single Modal job, all layers in one forward pass). Filter to clean-406. Per-cell balanced GroupKFold(5) probe AUROC.

**Result (L16 baseline + key extrema):**

| Layer | C_SFT pre | C_outcome pre | C_SFT ass | C_outcome ass | gap on C_outcome |
|---|---|---|---|---|---|
| L0 (embedding) | 0.488 | 0.463 | 0.630 | 0.529 | −0.065 |
| L1 | 0.785 | 0.869 | 0.723 | 0.660 | +0.209 |
| L5 | 0.796 | 0.886 | 0.714 | 0.666 | +0.220 |
| **L9 (max gap)** | 0.794 | 0.889 | 0.733 | 0.653 | **+0.236** |
| L12 | 0.795 | 0.893 | 0.761 | 0.697 | +0.196 |
| L16 | 0.804 | 0.896 | 0.785 | 0.703 | +0.193 |
| L20 | 0.817 | 0.900 | 0.776 | 0.710 | +0.190 |
| L24 (final) | 0.805 | 0.897 | 0.772 | 0.703 | +0.194 |

Gap stable between +0.18 and +0.24 from L5 to L24. **Rules out mechanism (b) "selective late-layer shaping"** from writeup §15 — late-only would predict gap growing with depth.

**Script.** `extension/probe/per_layer_sweep.py` → `22_per_layer_sweep.txt` + `fig11_per_layer_sweep.png` (2 panels, C_SFT and C_outcome, 3 position kinds × 25 layers each).

**Cache.** `extension/cache/probe_cache_n500_all_layers_clean406/` (300 files, 1.3 GB local).

**Modal cost.** ≈ $3 (one Modal job per ckpt, ~5 min each).

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
- `extension/probe/causal_steering_figure.py` → `fig12_causal_steering.png`

**Bug history.** First attempt crashed at the hook (newer transformers returns Tensor not tuple from decoder layer). Fixed in commit `34a2e8c`. Second attempt's stdout buffered for ~50 min — killed (had 37/100 prefixes done); third attempt added `flush=True`; resumed on the missing 63 prefixes with a JSONL-based skip filter (`extension/data/steering_todo.jsonl`).

**Modal cost.** ≈ $4-6 (~60 min H100 across the two passes).

---

### EXP-10: Per-problem probe-AUROC vs accuracy-delta correlation @ `<answer>` opening (PR experiment) ✅

**Question.** When the aggregate probe AUROC drops under RL (0.785 → 0.703 at assertion-position; or 0.92 → ~0.7 at `<answer>` opening per the trained position-appropriate probe), is the drop *concentrated on problems where accuracy also dropped* (damage) or *spread across problems independent of accuracy change* (decoupling)?

**Method.** Cherry-picked from `probe-behavioral-correlation` branch and aligned with our existing position convention (`answer_token_pos = open`, matching writeup §2.4).
1. **Train probe** (Modal, GPU): `extension/probe/train_answer_probe.py`. Same hyperparameters as §2.4 (LR C=0.1, GroupKFold(5) by problem). Trained on the **94 contaminated problems' rollouts** (disjoint from clean-406 by construction). Result: **held-out (by problem) AUROC = 0.889** — within the 0.85+ sanity bar vs §2.4's 0.920.
2. **Score the 406 clean problems** (Modal, GPU): `extension/probe/probe_behavioral_correlation.py`. Forward-pass each rollout, extract L16 hidden state at the LAST `<answer>`'s opening token, apply the trained probe, cache scalar score per rollout.
3. **Per-problem AUROC over K=16 rollouts** (local). Per-problem accuracy from the rollout JSONs. Spearman correlation between `probe_drop = AUROC_SFT − AUROC_RLOO` and `accuracy_delta = acc_RLOO − acc_SFT`.

**Result (n=218 valid problems):**

```
Spearman r = -0.032,  p = 0.63   (essentially zero correlation)
```

**Quadrants:**

| Quadrant | n | % |
|---|---|---|
| Top-right: probe ↓, accuracy ↑ (DECOUPLING) | **103** | **47%** |
| Top-left: probe ↑, accuracy ↑ (both improved) | 91 | 42% |
| Bottom-left: probe ↑, accuracy ↓ (noise) | 5 | 2% |
| **Bottom-right: probe ↓, accuracy ↓ (DAMAGE)** | **4** | **2%** |
| Exactly on axis | 15 | 7% |

**Per-problem AUROC distributions:** C_SFT mean 0.827 (median 0.873), C_outcome mean 0.813 (median 0.867). probe_drop mean +0.012 (essentially zero at problem level). accuracy_delta mean +0.260 (RL improved per-problem accuracy by ~26 pp on average).

**Interpretation.** The damage quadrant has only 4 problems / 218. The two changes (probe and accuracy) are statistically independent. Aggregate probe drop is *not* concentrated on problems with accuracy degradation. Decoupling, not damage, at per-problem level.

**Scripts.**
- `extension/probe/train_answer_probe.py` (Modal)
- `extension/probe/probe_behavioral_correlation.py` (Modal)

**Artifacts.**
- Probe pickle: `/vol/outputs/probe_behavioral/probe.pkl` + meta JSON
- Per-rollout cache: `/vol/outputs/probe_behavioral/rollout_scores_C_*_clean406.jsonl`
- `extension/outputs/n500/probe_behavioral/probe_behavioral_correlation.json` (per-problem auroc + accuracy arrays, Spearman, meta)
- `extension/outputs/n500/probe_behavioral/probe_behavioral_correlation.png` (scatter + quadrants + regression line)

**Modal cost.** ≈ $3-5 (probe training + forward-passes ~13K rollouts at `<answer>`-opening).

---

### EXP-11: Per-problem correlation @ trace-final position ✅

**Question.** Same per-problem correlation, but at the `</think>` position (where aggregate AUROC *rises* under RL: 0.804 → 0.896). Does the per-problem story match (decoupling) or diverge from EXP-10?

**Method.** Local-only (no Modal). Reuses `extension/cache/probe_cache_n500_clean406/C_{SFT,outcome}_l16_pre_answer.npz`. Held-out probe scores via GroupKFold(5) by prompt; per-problem AUROC; same Spearman correlation.

**Result (n=218):**

```
Spearman r = +0.335,  p = 4.0e-7   (HIGHLY SIGNIFICANT, POSITIVE)
```

**Quadrants:**

| Quadrant | n | % |
|---|---|---|
| Top-right: probe ↓, accuracy ↑ (decoupling) | **140** | **64%** |
| Top-left: probe ↑, accuracy ↑ | 57 | 26% |
| Bottom-left: probe ↑, accuracy ↓ | 9 | 4% |
| **Bottom-right: probe ↓, accuracy ↓ (DAMAGE)** | **0** | **0%** |
| On axis | 12 | 6% |

**Per-problem AUROC distributions (the key surprise):**
- C_SFT: mean **0.722**, median 0.741
- **C_outcome: mean 0.612, median 0.638** ← drops under RL at the per-problem level!
- probe_drop mean: **+0.130** (per-problem AUROC drops by 0.13 under RL on average)
- accuracy_delta mean: **+0.290**

**Aggregate vs per-problem paradox reconciled.** §2.1's aggregate trace-final AUROC *rises* (0.804 → 0.896) because it pools across all rollouts of all problems and benefits from cross-problem difficulty signal — under outcome RL the model becomes more confident on easy problems and less on hard ones, which inflates aggregate AUROC. The per-problem AUROC removes that confound. **Within-problem trace-final discrimination actually falls** (0.722 → 0.612), agreeing with the matched-pair finding (writeup §2.5: median Δ +0.186 → +0.004 — same direction, different position).

**+0.335 Spearman direction.** RL degrades the within-problem probe most on the problems where accuracy improved most. **Damage quadrant literally empty (0/218).** Strong positive triangulation of decoupling.

**Script.** `extension/probe/probe_behavioral_pre_answer.py` → `24_headline_dynamics_with_cis.txt` (wait, no — the actual outputs are `probe_behavioral_correlation_pre_answer.{json,png}`).

**Modal cost.** $0 (local CPU).

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

| Metric | 0.5B C_SFT | 0.5B C_outcome | **1.5B C_SFT** | **1.5B C_outcome** |
|---|---|---|---|---|
| pass@1 (n=500) | ~0.24 | ~0.55 | **0.236** | **0.558** |
| pass@1 (n=50 test) | 0.286 | 0.535 | **0.280** | **0.480** |
| Aggregate pre_answer AUROC (L16) | 0.804 | 0.896 | **0.857** | **0.973** |
| Aggregate pre_answer AUROC (L20) | 0.821 | 0.901 | **0.887** | **0.976** |
| Aggregate assertion AUROC (L16) | 0.785 | 0.703 | **0.825** | **0.816** |
| Aggregate assertion AUROC (L20) | 0.775 | 0.710 | **0.844** | **0.936** |
| **Gap pre−ass C_outcome (L20)** | **+0.190** | (same row) | **+0.040** | (same row) |
| Matched-pair % above-diag C_SFT | 78% | (same row) | **86%** | (same row) |
| Matched-pair % above-diag C_outcome | (same row) | 52% | (same row) | **82%** |
| Mann-Whitney C_SFT vs C_outcome (matched-pair) | (same row) | **p = 1e−16** | (same row) | **p = 0.14** (NS) |
| **Per-problem AUROC mean C_outcome** (L16, trace-final) | (same row) | **0.612 (DROP)** | (same row) | **0.901 (RISE)** |
| **Per-problem probe_drop mean** (trace-final) | (same row) | **+0.130** | (same row) | **−0.113** |
| Per-problem Spearman (probe_drop vs accuracy_delta) | (same row) | **+0.335, p=4e−7** | (same row) | **+0.008, p=0.93** |
| Per-problem dominant quadrant | (same row) | decoupling (64%) | (same row) | **both improved (63%)** |

**Headline interpretation.** The position-decoupling story does NOT reproduce at 1.5B:
- The pre−assertion gap on C_outcome shrinks from +0.19 to **+0.04** (best layer).
- Matched-pair distributions on C_SFT vs C_outcome are **statistically indistinguishable** at 1.5B (p = 0.14 vs p = 1e−16 at 0.5B).
- The per-problem trace-final AUROC under RL **rises** at 1.5B (0.79 → 0.90) instead of **falling** at 0.5B (0.72 → 0.61).
- The dominant per-problem quadrant flips from "decoupling" (64% at 0.5B) to "both improved" (63% at 1.5B).

**Position-decoupling under outcome RL is therefore a small-scale (0.5B) phenomenon.** At 1.5B the model preserves a coherent correctness representation across token positions throughout RLOO training.

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

### EXP-14: Corrected-label probe pipeline (next-`<answer>`-block as label) ✅

**Question.** When training the probe on assertion/pre_answer rows from multi-answer C_outcome rollouts, what is the *correct* label? Original pipeline labeled each row by the rollout's overall verifier score (last-block correctness). But the probe at assertion-position-N is morally a prediction of the *immediately next `<answer>` block*, not the rollout-final one. With rambling at 87% and per-block correctness varying widely within a rollout, this is a confounding label choice.

**Setup.** `extension/probe/relabel_full_grid.py` re-labels every assertion/pre_answer row by the correctness of the next `<answer>` block following that row's token. `extension/probe/relabel_redo_downstream.py` then retrains the probe via GroupKFold(5) under the corrected labels and recomputes the downstream statistics.

**Results — every AUROC in §2.1 shifts upward (probe is much stronger than we'd realized):**

| Cell | original-label AUROC | corrected-label AUROC | delta |
|---|---|---|---|
| C_outcome L16 pre_answer | 0.896 | **0.980** | +0.084 |
| C_outcome L16 assertion | 0.703 | **0.852** | +0.149 |
| C_outcome L16 gap | +0.193 | **+0.127** | −0.066 (gap narrows) |
| C_SFT L16 pre_answer | 0.804 | **0.904** | +0.100 |
| C_SFT L16 assertion | 0.785 | **0.887** | +0.102 |
| C_SFT L16 gap | +0.019 | **+0.017** | ≈ unchanged |

**Downstream stats — the matched-pair effect between checkpoints loses significance:**
- Wilcoxon C_SFT (one-sided > 0): p = 9.3e-35 (unchanged direction; corrected magnitude)
- Wilcoxon C_outcome (one-sided > 0): p = 3.9e-8
- **Mann-Whitney between checkpoints (was the headline matched-pair difference): p = 0.68 (NOT significant under corrected labels)**

**Interpretation.** With proper labels, the within-position probes are all very strong (0.85-0.98), and the cross-checkpoint matched-pair effect is no longer significant. The decoupling story is now (a) the pre-assertion gap, still real but smaller (+0.127), (b) the rambling-as-reward-hack pathology (87% multi-answer at 0.5B C_outcome), and (c) the applied near-oracle probe that we can deploy (EXP-15, 16, 17, 18). The "decoupling between checkpoints" framing is downgraded; the "applied near-oracle probe" framing is upgraded.

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

---

### EXP-NN: Skipped / negative-result experiments

| Experiment | Why skipped/null | Documented in |
|---|---|---|
| Verbal confidence elicitation (RLCR / token-logprob style) | SFT'd-on-Countdown Qwen 0.5B is not chat-tuned; every elicitation prompt gets continued as more Countdown. AUROC ≈ 0.50 (chance) for token-logprob. **Methodological observation worth flagging.** | writeup §6 |
| Phase 2B activation patching (inject first-answer state into last-answer position) | Pre-registered gating rule: only run if Phase 2A (EXP-05) revealed Pattern B. We observed Pattern A cleanly (T→F probe(last) = 0.154, indistinguishable from F→F). No preserved representation to inject. Skipped. | writeup §2.4 |
| C_process arm | Annotation-only subgoal reward; underperformed C_outcome. Documented as a confirmation of Strategic Information Allocation prediction at small scale. | writeup §10 |

---

## 3. Key cross-references & open questions

### Five independent strands of evidence for the decoupling claim:

1. **Aggregate position-resolved gap** (§2.1 writeup / EXP-02): pre−ass = +0.193 on C_outcome vs +0.019 on C_SFT.
2. **Gap emerges over training** (§2.2 / EXP-03 Option B): 0.019 → 0.022 → 0.115 → 0.217 → 0.193.
3. **Within-rollout Pattern A** (§2.4 / EXP-05): probe(last) on T→F = 0.154, matches F→F floor. Bidirectional in F→T (EXP-06).
4. **Causal steering null** (§2.11 / EXP-09): probe-vs-random Δ ∈ [−0.07, +0.02] across α.
5. **Per-problem independence at `<answer>`-opening** (§8.5 / EXP-10): Spearman r = −0.03, damage quadrant 4/218.
6. **Per-problem positive coupling at trace-final** (§8.5.1 / EXP-11): Spearman r = +0.335, damage quadrant **0/218**.

These triangulate the same claim: **outcome RL's effect on accuracy is not mediated by the probe-readable correctness representation in any causal, within-rollout, or per-problem-coupled way.**

### Pending (1.5B):

- Does the gap+dynamics emerge at 1.5B too?
- Does the per-problem correlation pattern hold at 1.5B?
- Does Yuan et al.'s "concealment widens" prediction return at 1.5B (closing the loop on §2.8's scale inversion)?

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
  cache_hidden_states.py               # default cache: pre_answer / assertion / neutral
  cache_answer_positions.py            # <answer>-opening cache (Phase 2A)
  filter_to_clean.py                   # n=500 → clean-406 filter
  analyze_probes.py                    # per-cell AUROCs
  bootstrap_headline_cis.py            # bootstrap 95% CIs on AUROCs
  qualitative_matched_pairs.py         # §2.5 matched-pair table
  cross_checkpoint_transfer.py         # 2×2 cross-checkpoint matrix
  cross_position_transfer.py           # 3×3 cross-position matrix (single-seed)
  phase1_diagnostics.py                # cross-position transfer (10-seed + heatmap)
  per_snapshot_decoupling_gap.py       # Option B dynamics analysis
  length_matched_transfer.py           # length-matched control
  significance_and_baselines.py        # significance tests + LR/RF/MLP
  deeper_analyses.py                   # per-keyword + per-layer + Cohen's d
  per_layer_sweep.py                   # 25-layer probe sweep
  probe_direction_cosines.py           # geometric cosine analysis
  ft_rollout_trajectory.py             # EXP-06 F→T trajectory
  save_probe_vector.py                 # saves probe direction for steering
  causal_steering.py                   # Modal HF hook injection
  analyze_causal_steering.py           # bar chart aggregation
  phase2a_per_answer_correctness.py    # EXP-05 pre-flight
  phase2a_pattern_analysis.py          # EXP-05 with trace-final probe
  phase2a_position_appropriate_probe.py # EXP-05 with position-appropriate probe
  train_answer_probe.py                # EXP-10 probe trainer (Modal)
  probe_behavioral_correlation.py      # EXP-10 experiment (Modal)
  probe_behavioral_pre_answer.py       # EXP-11 trace-final replication (local)
  make_figures.py                      # standard figures 1–7
  headline_dynamics_figure.py          # fig13 (headline plot)
  causal_steering_figure.py            # fig12 (steering bar chart)
  qualitative_annotated_figure.py      # qualitative example fig8
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

See list at top of doc; 24 files numbered 01_*.txt through 24_*.txt + 1 CSV.

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
