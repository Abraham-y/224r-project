# Code audit — correctness pass over the probe codebase

Scope: every `.py` on the results path (`extension/`, `rloo_trainer/`, `evaluation/`,
`scripts/`, `modal_train.py`). For each, I compared what the docstring/README claims the
code does against what it actually does, and where the data was available locally I re-ran
the computation to check.

Findings are ordered: **A** = code does not do what it says, **B** = result cannot be
regenerated, **C** = code is correct but the result should be read differently than the
paper reads it, **D** = checked and clean.

Everything under C1, C2, C3 and C6 below was verified numerically against the files in this
repo; the commands are at the bottom.

---

## Status as of 2026-08-12

This audit was written 2026-07-30. Most of section A has since been fixed, and
section B is largely stale because the deleted scripts were restored. Current
state, verified by re-reading the code:

| item | state |
|---|---|
| A1 `relabel_cross_checkpoint.py` NameError | **fixed** — constants hoisted above the parser |
| A2 assertion rows on a different label rule | **open** — still two conventions; see `REVISION_PACK.md` §D1 |
| A3 `union` selector is a no-op | **fixed** — row removed, reason left in a comment |
| A4 steering injects at the wrong token | **fixed** — `--steer_position probe_read` is the default |
| A5 `probe_augmented_rloo.py` silent λ=0 | **fixed** — explicit ctor arg, effective λ logged |
| A6 two scripts write the same figure | **partly fixed** — hardcoded figure functions removed |
| A7 dead constant | trivial |
| B1 twelve deleted scripts | **mostly stale** — `probe_usefulness_suite.py`, `firstanswer_rloo.py` and `phase2a_per_answer_correctness.py` are back |
| B2 README CLI examples wrong | **fixed** |
| C1–C3 | **addressed in the paper** — `structural_baselines.py` now computes all of it and the numbers are in the deployment table, the noise floor, and Limitations |
| C4 thresholds selected on eval data | **fixed in the paper** — see the correction below; this row previously said "open, moves no headline number" and that was wrong |
| C5, C7, C8, C10–C12 | **open** — checked against the current draft, and the paper quotes none of them |
| C6 causal steering read too strongly | **resolved by withdrawal** — the section is cut |
| C9 top-M at M/G effective lr | **fixed** — `--probe_topk_renormalize`, off by default so old runs reproduce |
| A8 anonymiser identity list incomplete | **fixed 2026-08-13** — new; see A8 below |

**One defect this audit missed**, found 2026-08-12: `surface_battery.py`
compared the frozen model's score on the held-out prompt half against the true
accuracy over *all* rows (0.4980 where 0.5173 belongs). Fixed; the corrected
population widens the section's central gap rather than narrowing it.

**A second, still live until 2026-08-12:** `causal_steering.py` hooked
`model.layers[L]`, whose output is `hidden_states[L+1]` — one transformer block
past the `hidden_states[L]` the probe reads. This is the layer half of the
off-by-one A4 covers the token half of, and it is why A4's note that the null
was "measured one to three tokens away" understated the problem. Now
`--layer_convention hidden_state` by default.

### Correction to this audit's own C4 verdict (2026-08-13)

This banner said C4 was open and moved no headline number. **That was wrong, and
it is the second time a "does not affect a headline number" judgement in this
file has failed.** Two of C4's three scripts supply figures the paper prints, and
in both cases the script *already computes the held-out version and labels it*:

```
BEST (IN-SAMPLE, (B,T) maximised on these same prompts -- optimistic):
  B=16, T=0.95: acc=0.6749, n_used=6.27
BEST (HELD-OUT, 2-fold by prompt) -- report this:
  acc=0.6675, n_used=5.54
```

The draft quoted the line above the one that says "report this."

| figure | was quoted | corrected |
|---|---|---|
| probe-guided restart | 0.675 at 6.27 rollouts | **0.6675 at 5.54** |
| within-rollout probe-commit | 0.6607 | **0.6601** |

Both now use the held-out figures. Selective abstention is **not** affected — its
threshold targets a *coverage* level, not the accuracy it reports, so it is not
selected on the quantity it prints. Full write-up in `REVISION_PACK.md` §D2.

**The generalisable lesson**, and it is the same one `followup/PAPER_NOTES.md`
already drew: in this project the artifacts have been right and the prose has
drifted from them. Twice now the producing script printed the correct value, in
capital letters, next to the one that got transcribed. A
regenerate-tables-from-artifacts step is worth more than any individual fix in
this file.

### One more, same class, found the same day

`writeup_workshop.tex` asserted *"within-checkpoint cross-position cosines ≤ 0.10
while cross-checkpoint within-position transfer AUROC stays ≥ 0.95."* Neither
bound holds: the cosine is **0.1042** (`41_relabel_cosines.txt`), and transfer is
**asymmetric** — 0.953 for C_SFT→C_outcome but **0.822** the other way
(`40_relabel_cross_checkpoint.txt`). The sentence quoted the favourable direction
of a two-sided quantity. Corrected; `REVISION_PACK.md` §D3. Note this is *not*
C5 — C5 concerns leakage in the cross-*position* AUROC table, which the paper
does not quote.

---

## A. Confirmed bugs

### A1. `relabel_cross_checkpoint.py` cannot run — `NameError` at line 30
```python
_ap.add_argument("--eval_sft", default=EVAL_SFT_PATH)   # line 30
...
EVAL_SFT_PATH = _args.eval_sft                          # line 34 — assigned AFTER use
```
`python extension/probe/relabel_cross_checkpoint.py` → `NameError: name 'EVAL_SFT_PATH' is
not defined`. Confirmed.

This matters because its committed output
`extension/outputs/n500/text/40_relabel_cross_checkpoint.txt` is where the headline
`0.912 → 0.982` AUROC pair comes from. That file was produced by a version of the script
that is no longer in the repo. Fix: hoist the two path constants above the parser.

### A2. `relabel_cross_checkpoint.py` labels assertion rows differently than every sibling script
`next_block_labels_assertion()` (lines 64–87) never uses `tok_idx`. It labels **every**
assertion-position row by the correctness of the **first** `<answer>` block in the rollout.
The comment says so ("Simplification: take the FIRST `<answer>` block").

`relabel_full_grid.py:132-141` and `relabel_redo_downstream.py:92-97` do the intended thing:
map `tok_idx → char offset`, then take the first block whose `char_open > char_pos`. So the
assertion row of the cross-checkpoint transfer table is on a different label definition than
the assertion row of the full-grid table. They are not comparable.

### A3. `probe_abstention_and_hybrid.py` — the "union" selector is a no-op
```python
if first_eq[(p, best[0])] == top_eq:
    union_acc.append(majority_label)   # both agree
else:
    union_acc.append(majority_label)   # disagree — use majority
```
Both branches append the same thing. `union` is identically `majority`, yet it is printed
and written to `33_probe_abstention_hybrid.txt` as a fourth method. Either implement the
intended abstain-on-disagreement rule or delete the row.

### A4. `causal_steering.py` injects at a different token than the probe reads
```python
think_tok_pos = input_ids.shape[1] - 1   # line 165
```
The prefix is built to end exactly at `</think>`, so this is the **last** subtoken (`>`).
The probe direction was fit on the hidden state at the token containing the **first**
character of `</think>` (`cache_hidden_states.py:181` → `char_to_token_index`), which for
this tokenizer is 2–3 positions earlier.

`probe_rloo.py:228-231` documents this exact trap as a bug they already fixed on the RL
side ("Earlier bug: was using last-char which is 2-3 tokens later — different hidden state,
OOD probe → saturation"). The steering script still has it.

Second, smaller mismatch: because the prefix terminates at `>`, that `>` tokenizes on its
own, whereas in the cache it merges with the following `\n\n` into one token. Same issue the
docstring of `probe_rloo._probe_score` warns about.

This weakens the null result specifically: "steering along the probe direction does nothing"
was measured one to three tokens away from where the probe reads.

### A5. `probe_augmented_rloo.py` silently falls back to λ=0
`lambda_mix` reaches the Ray actor only through `os.environ["PROBE_AUG_LAMBDA"]`
(`probe_augmented_rloo.py:95`), read in the worker as
```python
_lam = float(_os.environ.get("PROBE_AUG_LAMBDA", "0.0"))   # rloo_update_worker.py:274
```
If the env var does not reach the worker process, λ silently becomes 0 — the *pure probe
baseline*, a different experiment — with no error and nothing in the logs. It probably does
propagate (the var is set before `ray.init()`), but nothing verifies it. Log the effective
λ from inside `update()` on the first call, or pass it as a constructor arg like
`probe_topk_M` already is.

### A6. Two scripts write the same figure, one from data and one from hardcoded numbers
- `scripts/make_poster_figures.py:150` writes `figures/poster_post_goodhart_delta.pdf` from
  the literals `deltas = [0.021, 0.083]`.
- `extension/probe/plot_causal_steering.py:73` writes **the same path** by bootstrapping the
  actual JSONLs.

Whichever runs last wins. Also `fig_probe_auroc` hardcodes `0.904 / 0.980` where every text
artifact says `0.912 / 0.982`, and assigns `neutral` the identical value `0.562` to both
checkpoints (looks like a copy-paste). Delete the hardcoded figure functions or have them
read the same JSON the text reads.

### A7. Dead constant
`probe_guided_restart.py:41` defines `PAC = ".../per_answer_correctness.jsonl"` and never
uses it. Harmless, but the producing script is also gone (see B1).

### A8. The anonymiser's identity list matched the owner but not the repo — added 2026-08-13

`scripts/make_submission_tex.py` builds the double-blind submission and gates it
on a list of identity patterns, exiting non-zero if any survives. The list had
`github\.com/Abraham-y` but no pattern for the repository *name*. So the check
verified the owner path was gone and would have passed a document that still
said `reader-writer-probe-rl` anywhere the URL rewrite did not reach.

That is not hypothetical here. The repo has already been renamed once —
`cs224r-project` → `reader-writer-probe-rl` — and the rename is what exposed it:
GitHub announces it on push, the URL rewrite in the script was keyed to one
spelling, and a future rename would produce exactly the surviving-string case the
check was supposed to catch.

**Severity is the point.** Every other item in this file costs accuracy. This one
costs the submission: a de-anonymised double-blind paper is a desk reject, and it
fails silently, at the one moment nobody re-reads the PDF. A gate whose failure
mode is "passes when it should fail" is worse than no gate, because it is trusted.

Fixed: both repository names are now patterns in their own right (13 checked, not
11), with a comment saying why the name and not just the owner. The check also
runs against the *rewritten* output rather than being trusted to have handled
things, and the rendered PDF is scanned separately — extracted text and metadata
both — because LaTeX can put an author string in `/Author` that no source grep
would see.

**Related, and worth stating because I got it backwards once:** the URL in the
paper was never dead. `cs224r-project` redirects to the new name, so both resolve;
the *canonical* one is `reader-writer-probe-rl`, which is what the paper had
originally and what it has again. A redirect is not a broken link, and "this link
404s" should be checked by following it rather than by comparing it against a
stale `git remote`.

---

## B. Results that cannot be regenerated from this repo

### B1. Twelve referenced scripts were deleted in `cd8cd5e`
`modal_train.py` still routes to, and other scripts still depend on:

| missing file | consumed by |
|---|---|
| `extension/probe/probe_usefulness_suite.py` | **`probe_usefulness_suite_results_n406.json` at repo root — supplies 3 README headline numbers** |
| `extension/training/firstanswer_rloo.py` | `eval_c_firstanswer_n500.json`, read by `probe_rl_downstream_analysis.py:75` |
| `extension/probe/phase2a_per_answer_correctness.py` | `per_answer_correctness.jsonl` → `cache_answer_positions.py`, `causal_steering.py --rollouts_jsonl` |
| `extension/probe/train_answer_probe.py` | the `<answer>`-position probe pickles |
| `extension/probe/cache_all_think_close.py` | `extension/cache/probe_cache_n500_all_thinkclose/` |
| `extension/probe/prompt_difficulty_probe.py` | — |
| `extension/probe/probe_behavioral_correlation.py` | — |
| `extension/probe/build_probe_filtered_sft.py`, `sft_probe_filtered.py` | — |
| `extension/probe/elicit_verbalized_confidence.py`, `elicit_token_logprob_confidence.py` | — |
| `extension/training/ramble_penalty_rloo.py` | (marked WITHDRAWN) |

The README's applied-probe row ("Best-of-16 `+12.1 pp`; abstention `98%` at `50%` coverage;
probe-mean within `±0.4 pp`") is entirely `probe_usefulness_suite.py` output. Restore that
one file at minimum.

### B2. README §5 CLI examples do not match the scripts
```bash
python extension/training/probe_reward_rloo.py --init C_outcome --mode reward
```
Neither flag exists. The real ones are `--probe_model` and
`--reward_mode probe|probe_gated|blend|mult`. `_pop_value` would silently swallow
`--init C_outcome` and pass `--mode reward` through to `rloo.py`'s argparse, which errors.
Same for `--K 8 --top_M 4 --init C_SFT` in §6 (`probe_augmented_rloo.py` takes
`--lambda_mix`; top-M lives on `rloo.py` as `--probe_topk_M` and requires `--probe_baseline`).

---

## C. Correct code, but the result means less than the paper says

### C1. The probe's headline numbers are heavily confounded by *when* `</think>` is emitted — verified

This is the one I would fix before anything else.

The probe reads the residual stream **at the `</think>` token**. That token's *position* is
itself a strong correctness signal: this model rambles when it is wrong. Measured on the
shipped caches and evals:

| | C_SFT | C_outcome |
|---|---|---|
| probe, 896-d L16 activation, held-out balanced AUROC | 0.912 | **0.982** |
| **one scalar**: index of the `</think>` token, same CV, same folds | 0.790 | **0.915** |
| probe AUROC stratified within `</think>`-position deciles | 0.852 | **0.939** |
| (sanity) position AUROC within those deciles | 0.510 | 0.564 |

Two conclusions, one good and one not:

- **Good**: the probe survives the control. At matched think-length the probe still reaches
  0.939 on C_outcome, and the "outcome RL strengthens the probe" delta actually *grows*
  under the control (0.852 → 0.939, +0.087, vs +0.070 uncontrolled). The claim that the
  probe reads something beyond rambling is defensible — but it currently has no supporting
  control in the repo.
- **Not good**: the *applied* result is majority-explained by a signal that needs no probe.
  On the identical population the suite used (clean-406, C_outcome, K=16 — my random-pick
  and oracle numbers match `probe_usefulness_suite_results_n406.json` exp3 to 4 decimals):

  | selector | accuracy | lift vs random pick | share of oracle gap |
  |---|---|---|---|
  | random pick (pass@1) | 0.5531 | — | — |
  | **pick the rollout with the shortest `<think>` body** (no probe, no forward pass) | 0.6379 | **+8.5 pp** | 64% |
  | probe best-of-16 | 0.6626 | +10.9 pp | 83% |
  | oracle pass@16 | 0.6847 | +13.2 pp | 100% |

  So the probe buys +2.5 pp over a one-line heuristic. That is still a real result, but
  "+11 pp from a near-oracle internal verifier" should become "+11 pp, of which +8.5 pp is
  available from response length alone; the probe adds +2.5 pp."

  Same story at C_SFT: shortest-`<think>` gives +14.8 pp there.

**Recommended fix**: add `-len(think_body)` (or `tok_idx`) as a baseline column to every
applied-probe table, and add the length-stratified AUROC row to the AUROC table. Both are
cheap and both are numbers the paper is stronger for having.

### C2. `probe_as_eval_proxy.py` — mostly a calibration identity, but not entirely

**Corrected after running the null on the real cache.** My first pass used a *synthetic*
null (i.i.d. Bernoulli labels, no per-prompt clustering) and concluded a zero-signal probe
matches dataset accuracy *better* than the real one. That does not hold on the actual data,
because the real labels are strongly clustered by prompt and GroupKFold makes the noise
probe's per-fold intercept track the held-out base rate less well.

The null control is now built into the script (same folds, same LR, activations replaced by
Gaussian noise of the same shape). On `C_outcome` clean-406:

| | AUROC | dataset-accuracy estimate | \|error\| |
|---|---|---|---|
| noise features (zero signal) | 0.521 | 0.5673 | **1.43 pp** |
| real probe | 0.982 | 0.5566 | **0.35 pp** |
| truth | — | 0.5531 | — |

So the honest reading: a probe with **no signal at all** already lands within ~1.4 pp,
because logistic regression with an intercept is calibrated in the mean by construction.
The real probe's 0.35 pp is a genuine improvement on that, but the right comparison for
"±0.4 pp" is the **1.4 pp noise floor, not zero**. Report it against the null, or drop the
claim — it is by far the weakest item in the applied-probe table.

### C3. Rollouts with no `</think>` are silently dropped, and they are 100% wrong — verified

`cache_hidden_states.py:182` only caches a `pre_answer` row when `</think>` is locatable.
Measured on the eval JSONs:

| | dropped | accuracy of dropped rollouts |
|---|---|---|
| C_SFT | 673 / 6496 (10.4%) | **0.000** |
| C_outcome | 190 / 6496 (2.9%) | **0.000** |

Three consequences:
1. Every AUROC in the paper is computed on a population from which the trivially-detectable
   negatives have been removed, so it is not the deployment AUROC.
2. The `0.912 → 0.982` comparison is between two *differently* filtered populations — C_SFT
   loses 3.6× as large a share. Some of the reported improvement is a filtering artifact.
3. In `probe_bestofk_offline.py:91` those rollouts get `probe_score = -1`, so the probe never
   picks them — a free win — while the `random_pick` baseline (`yy.mean()`, line 117)
   includes them. That's worth +1.6 pp at C_SFT and +0.3 pp at C_outcome of the reported
   lift before the probe does anything.

Fix: report the deployment rule as "no `</think>` ⇒ score 0" and score the full population,
for both the probe and the baselines.

### C4. Thresholds are selected on the evaluation data
- `probe_answer_commit.py:143-151`: `best_acc = max` over the 19-point threshold sweep, then
  reported as the headline with its "GAIN".
- `probe_guided_restart.py:184`: `best = max(results, key=acc)` over the 20-point (B,T) grid.
- `probe_abstention_and_hybrid.py:198`: `intersection_T = 0.5` is fixed (fine), but the
  reference coverage points are read off the same curve.

None of these hold out a threshold. Split by prompt: pick T on half the prompts, report on
the other half.

### C5. `cross_position_transfer.py` puts two different estimators in one table
- Diagonal (line 110–119): GroupKFold(5) by `prompt_idx` — properly held out.
- Off-diagonal (line 122): `train_and_eval(X_tr, y_tr, X_te, y_te)` — trains on **all** of
  position kind A, tests on **all** of kind B, with **no group separation**. The same
  rollouts appear on both sides carrying the same rollout-level label.

The leakage inflates the off-diagonals, so the stated conclusion ("diagonals > off-diagonals
⇒ distinct subspaces") is conservative rather than wrong — but the numbers in the table are
not comparable to each other. Use the same GroupKFold folds for both: train on kind A rows
of the training prompts, test on kind B rows of the held-out prompts.

### C6. Causal steering — the analysis code is right, the reading of it is too strong

I re-ran `causal_steering_stats.py` on the shipped JSONLs. The data is clean: 1358 rows per
file, all 7 conditions × 194 prefixes present, `patch_applied=True` on every steered row, no
duplicates. The headline reproduces exactly: post-Goodhart Δ = **+0.082** at α=1.0, assertion
control Δ = **−0.015**. The bootstrap and exact-McNemar implementations are both correct.

Four things the write-up should say and currently doesn't:

1. **The vanilla checkpoint is not "causally inert" at every α.** At α=0.5 the vanilla Δ is
   **−0.072, CI [−0.138, −0.010], McNemar p=0.065** — a significant *negative* effect. The
   README reports the interval `[-0.07, +0.02]` as if −0.07 were noise. It is the one vanilla
   α that most clearly is not noise.
2. **The null band is circular.** `NULL_BAND_UPPER = 0.02` (`causal_steering_stats.py:30`) is
   the max of the vanilla Δs. `p(D>band)` then tests the post-RL Δ against a threshold read
   off the same three measurements. It is not a calibrated test and should not be reported
   as a p-value.
3. **No dose-response.** Post-RL Δ across α is `+0.041, +0.082, −0.052` — non-monotone, with
   the largest magnitude effect flipping sign at α=2. A "the direction became a control axis"
   story predicts monotonicity over some range.
4. **Multiplicity.** One of three α values reaches McNemar p<0.05 (p=0.017); Bonferroni over
   three α gives 0.051. Pre-register α=1.0 or report the correction.

### C7. `relabel_dynamics.py` — an n=5 correlation across mixed populations
`Pearson r(mean_blocks, gap) = +0.891, p=0.043, n=5` (`43_relabel_dynamics.txt`). The five
points come from two different eval sets — `C_SFT`/`C_outcome` from the n500 clean-406 caches,
`step_30/60/90` from `eval_c_outcome_step_*_n200.json` (different 200 prompts). Within each
snapshot, `mean_blocks` (line 88) averages over **all** prompts in the eval file while the
AUROC is on the clean-406 subset. Don't report a p-value on this.

### C8. "majority-of-16" means two different things
- `probe_applied_scale_comparison.py:113` and `probe_guided_restart.py:194`:
  `int(sum(true_labels) > K/2)` — an **oracle** majority. It needs ground truth and is not a
  deployable baseline.
- `probe_abstention_and_hybrid.py:213-219`: votes on the normalized equation string — the
  real self-consistency baseline.

Both are labelled "majority-of-K" in output. Beating the first is not beating self-consistency.

### C9. `probe_topk_M` is not compared at matched effective step size
`rloo_update_worker.py:287` zeroes (G−M) of G advantages, but line 298 still divides by G:
```python
advantages = advantages * topk_mask
...
pg_loss = -(importance_weights * advantages * seq_log_probs).mean()
```
so the top-M arm takes roughly M/G of vanilla's gradient magnitude at the same learning rate.
The "+1.5 pp" is confounded with a smaller effective lr. Either divide by `topk_mask.sum()`
or run vanilla at lr·M/G as the control.

### C10. `probe_rloo.py` scores against a stale policy
`_ensure_reference_model` (line 137) reloads from `latest_checkpoint`, but `rloo.py:330-335`
writes to `epoch_{e}_step_{s}` instead of `latest_checkpoint` on every `save_every_n_steps`-th
step. So on those steps the reference model is one update behind, and between saves it is up
to `save_every_n_steps−1` updates behind. (Lower priority — this script is marked WITHDRAWN
in `modal_train.py:227`.)

Related, in `rloo.py:330`: the checkpoint written at `global_step=N` is the state **after**
N+1 updates, so `epoch_0_step_30` is 31 steps of training, not 30.

### C11. `per_layer_sweep.py` uses different labels than everything else
It reads `y` straight from the npz (`load_cell`, line 32), i.e. the **rollout-final verifier**
label baked in at cache time. Every applied-probe and relabel script uses corrected
**first-block** labels. The per-layer figure and the headline AUROC table are therefore on
different label definitions.

### C12. `probe_answer_commit.py` compares against a different denominator than it reports
`verifier_acc` comes from the eval JSON (verifier on the last `<answer>` in the raw text);
`last_acc` and the probe-commit fallback come from the last **cached** block. Where the cache
dropped a trailing block these differ, and "GAIN vs verifier" is partly that difference. The
comment at line 114 flags it; the headline number doesn't.

---

## D. Checked and clean

Worth recording so these don't get re-audited:

- **RLOO update math** (`rloo_update_worker.py`). LOO baseline `(sum − self)/(G−1)` is right;
  group flattening is prompt-major on both the reward side (`rloo.py:203-206`) and the probe
  side (line 242), so `rewards.view(-1, group_size)` rows really are groups; the response
  mask `is_response_token[:,1:] * attention_mask[:,1:]` correctly excludes pads and correctly
  *includes* the first response token; importance weights are detached so gradient flows only
  through `seq_log_probs`; the KL is the standard k3 estimator and is differentiable.
- **No prompt-level leakage in the applied-probe results.** Every one of
  `probe_bestofk_offline`, `probe_guided_restart`, `probe_abstention_and_hybrid`,
  `probe_adaptive_budget`, `probe_applied_scale_comparison`, `probe_as_eval_proxy`,
  `probe_answer_commit`, `phase2a_position_appropriate_probe` uses `GroupKFold(5)` by
  `prompt_idx` for the held-out scores. This is done right.
- **Probe direction recovery.** `w / scaler.scale_` is the correct input-space direction for
  `Pipeline(StandardScaler, LogisticRegression)`; consistent in
  `save_probe_direction_temp1.py:25` and `relabel_cosines.py:75`.
- **Prompt reconstruction.** `generate_countdown.format_prompt` reproduces numpy's
  `array2string(separator=" ")` including the leading-space padding case
  (`[ 7  2 43 63]`), matching `probe_rloo._reconstruct_prompt`.
- **Verifier.** `countdown.validate_equation`'s sorted-multiset check correctly enforces
  "each number exactly once"; `evaluate_equation`'s character whitelist is sound.
- **`causal_steering_stats.py`.** Cluster bootstrap resamples `prompt_idx` (the right unit);
  exact two-sided McNemar is implemented correctly.
- **Monkey-patch ordering.** `probe_reward_rloo._exec_rloo` patches
  `evaluation.countdown.compute_score` and `RLOODataset.collate_fn` *before* exec'ing
  `rloo.py`, so `rloo.py:25`'s `from evaluation.countdown import compute_score` picks up the
  patched function. This works, though it is fragile.
- **`<answer>` block index alignment** in `cache_answer_positions.py`. The open-tag regex
  (`<answer>`) can in principle drift out of sync with the closed-pair list that produced
  `per_block_correct`. Measured: mid-stream misalignment occurs in **2/8000** C_outcome
  rollouts and **3/8000** C_SFT rollouts (0.03%). Not a problem.
- **Position ordering.** `<answer>` never precedes `</think>` in any of the 16000 rollouts
  checked, so the "pre_answer" naming is accurate.

Minor, non-result-affecting: `sft_trainer/sft.py:100` masks with `is_response_token[:,1:]`
without `& attention_mask[:,1:]`, so response-side pad tokens enter the SFT loss. Off the
results path (SFT was not retrained) but worth fixing if it is ever rerun.

---

## Reproducing the checks in section C

All run on CPU from the repo root against files already present.

```bash
# C1 — length confound: structural baseline AUROC + stratified probe AUROC
python3 - <<'PY'
import json,re,warnings,numpy as np
from sklearn.pipeline import Pipeline; from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression; from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ANS=re.compile(r"<answer>(.*?)</answer>",re.DOTALL)
def cb(eq,t,nums):
    try:
        n=[int(x) for x in re.findall(r"\d+",eq.strip())]
        if sorted(n)!=sorted([int(x) for x in nums]): return False
        if not re.match(r"^[\d+\-*/().\s]+$",eq.strip()): return False
        return abs(eval(eq.strip(),{"__builtins__":None},{})-int(t))<1e-5
    except Exception: return False
def bal(y,s,seed=0):
    p=np.where(y==1)[0];n=np.where(y==0)[0];k=min(len(p),len(n));r=np.random.RandomState(seed)
    i=np.concatenate([r.choice(p,k,0),r.choice(n,k,0)]); return float(roc_auc_score(y[i],s[i]))
for ck,ev in [("C_SFT","eval_c_sft_n500.json"),("C_outcome","eval_c_outcome_n500.json")]:
    X=np.load(f"extension/cache/probe_cache_n500_clean406/{ck}_l16_pre_answer.npz")["X"]
    meta=json.load(open(f"extension/cache/probe_cache_n500_clean406/{ck}_l16_pre_answer.meta.json"))
    g=np.array([int(m["prompt_idx"]) for m in meta]); tok=np.array([int(m["tok_idx"]) for m in meta],float)
    rows=[json.loads(l) for l in open(ev) if l.strip()]; lab={}
    for p,r in enumerate(rows):
        t=int(r["target"]);nums=list(r["nums"])
        for ri,resp in enumerate(r["response"]):
            m=ANS.search(resp); lab[(p,ri)]=int(bool(m) and cb(m.group(1),t,nums))
    y=np.array([lab[(int(m["prompt_idx"]),int(m["resp_idx"]))] for m in meta],int)
    def ho(F):
        s=np.full(len(y),np.nan)
        for tr,te in GroupKFold(5).split(F,y,g):
            s[te]=Pipeline([("sc",StandardScaler()),("lr",LogisticRegression(C=0.1,max_iter=2000))]).fit(F[tr],y[tr]).predict_proba(F[te])[:,1]
        return s
    sp,sl=ho(X),ho(tok.reshape(-1,1))
    q=np.quantile(tok,np.linspace(0,1,11)); b=np.clip(np.digitize(tok,q[1:-1]),0,9)
    def st(s):
        num=den=0
        for k in range(10):
            m=b==k
            if len(set(y[m]))<2: continue
            n1=int(y[m].sum());n0=int((1-y[m]).sum()); num+=roc_auc_score(y[m],s[m])*n1*n0; den+=n1*n0
        return num/den
    print(ck,"probe",round(bal(y,sp),3),"| tok_idx only",round(bal(y,sl),3),
          "| probe stratified",round(st(sp),3),"| tok_idx stratified",round(st(sl),3))
PY

# C1 — shortest-<think> best-of-16 vs the reported probe best-of-16
# C2 — noise-feature null test for the eval-proxy claim
# C3 — accuracy of rollouts with no </think>
# C6 — reproduce the steering table
python3 extension/probe/causal_steering_stats.py --bootstrap 10000
```

---

## FIX LOG — what was actually changed

Everything below is applied and compile-checked; every analysis with local data was re-run.

### Bugs fixed
| # | File | Change |
|---|---|---|
| A1 | `extension/probe/relabel_cross_checkpoint.py` | Hoisted the path constants above the argparse block. Script runs again; `pre_answer` reproduces the committed 0.912 / 0.982 / 0.953 / 0.822 exactly. |
| A2 | same | `next_block_labels_assertion` now maps `tok_idx → char offset` and takes the *next* `<answer>` block, matching `relabel_full_grid.py` / `relabel_redo_downstream.py`. Was labelling every row by the *first* block. |
| A3 | `extension/probe/probe_abstention_and_hybrid.py` | Removed `union_acc` (both branches were identical → it was a duplicate of `majority`). Replaced with an agreement-gated **selective** metric that reports accuracy *and* coverage. Comment explains why no always-commit version of that idea can be non-degenerate. |
| A4 | `extension/probe/causal_steering.py` | Injects at the token containing the **first** char of `</think>` — the position the probe was fitted on — instead of the last token of the prefix (2–3 tokens later). `--steer_position last_token` reproduces the old runs. Random-direction dim now derives from the steer vector instead of a hardcoded `896` (so 1.5B works). Per-row `steer_position` / `steer_tok_pos` written to the JSONL. Docstring corrected re: which layers' KV actually change. |
| A5 | `probe_augmented_rloo.py`, `rloo.py`, `rloo_update_worker.py` | `lambda_mix` is now an explicit CLI flag → constructor arg. The worker resolves it once, records the source, and prints it on the first update. A lost env var can no longer silently turn the run into λ=0. |
| A6 | `scripts/make_poster_figures.py` | No longer writes `poster_post_goodhart_delta.pdf` (that path is now solely `plot_causal_steering.py`'s). All bar heights read from `extension/outputs/poster_numbers.json`, generated by the new `scripts/collect_poster_numbers.py`. |
| A7 | `extension/probe/probe_guided_restart.py` | Dropped the unused `PAC` constant. |
| C5 | `extension/probe/cross_position_transfer.py` | Off-diagonals now use the **same** prompt-level GroupKFold(5) holdout as the diagonals (new `grouped_transfer_auroc`). Previously they trained and tested on the same rollouts with no group separation, putting a leaky estimator in the same table as a clean one. |
| C8 | `probe_applied_scale_comparison.py`, `probe_guided_restart.py` | `majority-of-16` → `ORACLE majority-of-16 (needs answer key)`. It consumes ground truth and is not self-consistency; the name invited exactly that misreading. |
| C9 | `rloo_update_worker.py`, `rloo.py` | Documented that top-M gating leaves the arm at ~M/G of vanilla's effective lr, and added `--probe_topk_renormalize` to correct it. Default off so existing runs reproduce. |
| C11 | `extension/probe/per_layer_sweep.py` | Defaults to corrected first-block labels, matching the rest of the paper, instead of the rollout-final labels baked into the `.npz`. `--labels cached` restores the old behaviour. |
| — | `sft_trainer/sft.py` | Loss/accuracy masks now `& attention_mask`, so right-padding no longer enters the SFT loss (3 sites). |
| — | `extension/probe/cache_answer_positions.py` | Enumerates closed `<answer>…</answer>` pairs, so block indices align with `per_block_correct` by construction rather than by luck; warns loudly if they ever don't. |

### Restored from git (`cd8cd5e`)
`extension/probe/probe_usefulness_suite.py` (supplies three README headline numbers),
`extension/probe/phase2a_per_answer_correctness.py` (produces the `per_answer_correctness.jsonl`
that `cache_answer_positions.py` and `causal_steering.py --rollouts_jsonl` consume), and
`extension/training/firstanswer_rloo.py` (produces `eval_c_firstanswer_n500.json`, read by
`probe_rl_downstream_analysis.py`). The other nine deleted files were retracted experiments
with no live consumer and were left deleted.

### New
- **`extension/probe/structural_baselines.py`** — the C1 control, as a first-class experiment.
  Reports position-only AUROC, position-stratified probe AUROC, shortest-`<think>` best-of-K
  next to probe-best-of-K, and the excluded no-`</think>` population. Output:
  `extension/outputs/n500/text/60_structural_baselines.{txt,json}`.
- **`scripts/collect_poster_numbers.py`** — every poster number from data into one JSON,
  reporting *both* held-out AUROC estimators the codebase uses so the 0.904-vs-0.912
  discrepancy is visible instead of hidden.

### Reporting changes (in-sample → held-out)
- `probe_answer_commit.py` and `probe_guided_restart.py` now also report a **held-out**
  threshold / (B,T), chosen 2-fold by prompt. Both in-sample and held-out are printed, with
  the held-out one flagged as the reportable number.
- `probe_as_eval_proxy.py` now runs the noise-feature null control inline and prints the
  comparison every time.
- `causal_steering_stats.py` now documents that `p(D>band)` is not a calibrated p-value and
  prints the Bonferroni factor for the alpha sweep.

### Numbers that changed as a result
| Claim | Before | After |
|---|---|---|
| poster `neutral` AUROC | 0.562 for both checkpoints (unsourced) | 0.596 (C_SFT) → 0.620 (C_outcome) — it *rises*, it is not flat |
| best-of-16 lift, C_outcome | `+12.1 pp` vs nothing | `+11.7 pp`, of which `+8.5 pp` is free from `<think>` length; probe adds `+3.2 pp` |
| probe-commit gain | `+0.0867` (in-sample threshold) | `+0.0861` (held-out) — holds up |
| cross-position off-diagonals | leaky, inflated | near-chance (0.45–0.58) vs diagonals 0.70–0.90 — conclusion strengthened |
| eval-proxy `±0.4 pp` | vs an implicit zero floor | vs a **1.4 pp** noise floor |

### Verified unchanged
`relabel_cross_checkpoint` pre_answer (0.912/0.982/0.953/0.822) and the causal-steering
headline (+0.082 post-Goodhart, −0.015 assertion control) both reproduce bit-for-bit after
the refactors.

### Round 2 — the remaining code items
| # | File | Change |
|---|---|---|
| C1/C3 | `probe_bestofk_offline.py` | The probe was getting a free win the baseline did not: rollouts with no `</think>` score `-1` so the probe never picks one (they are ~always wrong), while `random_pick` averaged over all K. Now reports **three** baselines — random-over-all-K, random-over-those-with-`</think>` (the honest denominator, same free filter the probe gets), and shortest-`<think>` best-of-K — plus the probe's gain over each. `np.zeros(896)` → `model.config.hidden_size`. |
| C3 | `structural_baselines.py` | New section (4): **deployment AUROC** over the full population under the rule "no `</think>` ⇒ predict wrong", alongside the cached-subset AUROC the paper reports. |
| B1 | `modal_train.py` | `_require_script()` + a pre-launch local check, so the 8 routes whose scripts were pruned fail immediately with the exact `git show` command to restore them, instead of an opaque subprocess error minutes into a provisioned GPU container. |
| C7 | `relabel_dynamics.py` | `mean_blocks` now computed over the same rollouts as the AUROC (was averaging the full n500 while the AUROC was on clean-406). Added Spearman, an **exact permutation** p-value (scipy's asymptotic p returns ~1e-24 for rho=1.0 at n=5, which is nonsense — only 120 orderings exist, so the floor is 2/120 = 0.017), and a printed health warning that n is snapshots, that they span two eval sets, and that they are successive checkpoints of one run. |
| C10 | `probe_rloo.py` | `_find_latest_checkpoint` now globs `epoch_*_step_*/model` as well as `latest_checkpoint/model`. It previously went stale on exactly the `save_every_n_steps`-th steps, scoring rollouts with pre-save weights. |
| C10 | `rloo.py` | Writes a `training_state.json` sidecar per checkpoint recording `updates_completed` (= `global_step + 1`), since `epoch_0_step_30` actually holds 31 updates. Renaming the dirs would break every downstream glob, so the count is recorded instead. Also added the missing `import json` this needed. |
| C12 | `probe_answer_commit.py` | Reports the cache-last vs verifier-last disagreement count and a like-for-like "GAIN vs cache-last" next to "GAIN vs verifier-last". |
| — | `extension/metrics/calibration.py` | Was a stub whose two functions raised `NotImplementedError` while being listed in the README layout. Implemented `ece`, `mce`, `brier`, `reliability_table`, `plot_reliability`, with input validation and correct closed-top-bin handling. Verified against a synthetic calibrated set (ECE 0.007) and a +0.3-shifted overconfident one (ECE 0.254). |

### Two findings from round 2
- **C12 was a false alarm.** Cache-last and verifier-last disagree on **0 / 5458** rollouts, so the gain is `+0.0861` either way. That concern is empirically dead — no correction needed.
- **C3 was real and quantified.** Scoring the full population with the free "no `</think>` ⇒ wrong" rule: C_SFT `0.912 → 0.923`, C_outcome `0.982 → 0.983`. The reported `+0.070` improvement is `+0.060` on the honest deployment population, i.e. roughly **15% of the headline delta was a filtering artifact** — the dropped rollouts are 9.4% of C_SFT but only 2.5% of C_outcome, so the filter flattered the weaker checkpoint less.

### Not done (needs a GPU / a training run)
- Re-running causal steering at the corrected `--steer_position probe_read`. The code is
  fixed; the *result* in the paper is still from the old position. **This is the one
  outstanding item that could change a published conclusion.**
- The assertion half of `relabel_cross_checkpoint.py` (needs `transformers`, not installed
  locally).
- Any top-M-vs-vanilla rerun at matched effective lr (`--probe_topk_renormalize`).

---

## Suggested order of work

1. **C1** — add the `</think>`-position baseline to the AUROC table and the length-based
   selector to every applied-probe table. Biggest effect on what the paper can claim, and
   the AUROC claim survives it.
2. **C3** — score the full population with "no `</think>` ⇒ wrong" and rerun the applied
   numbers. Cheap, removes a free win from the probe's column.
3. **C2** — pull the eval-proxy claim or reframe it.
4. **B1** — restore `probe_usefulness_suite.py` (and ideally `firstanswer_rloo.py` and
   `phase2a_per_answer_correctness.py`).
5. **A1/A2** — fix `relabel_cross_checkpoint.py` and regenerate `40_*.txt`, since the headline
   AUROC pair currently traces to a script that cannot run.
6. **A4** — rerun steering at the probe's actual read position before publishing the null.
7. **C6** — soften the vanilla "inert" wording, drop the circular null-band p-value, report
   the α multiplicity.
8. **A3, A5, A6, C4, C5, C8, C9, C11** — mechanical fixes.
