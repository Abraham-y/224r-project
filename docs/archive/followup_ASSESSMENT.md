# Assessment of `followup/CLAUDE.md`

Written 2026-07-30, after reading the plan, the existing codebase
(`extension/`, `rloo_trainer/`, `evaluation/`), `findings.md`, and the Modal
volume; and after running the parts of the plan that could be run for free.

**Verdict: the plan is sound and worth doing.** The research question is the
right next one, the hypotheses are the right hypotheses, and the engineering
conventions are stricter than the original project's, which is the correct
direction. Roughly 80% of it can be built on existing code rather than written
fresh.

Five substantive problems, listed worst-first. Four are fixed in this commit;
one needs your decision.

---

## 1. Phase 0 is planned as a re-run. It doesn't need to be. (fixed)

CLAUDE.md Phase 0 says: add dense checkpointing, add activation capture, re-run
the collapse experiment. Cost ≈ $160 and ~10 h per ladder.

All 11 archival checkpoints from **both** published Goodhart runs are still on
the Modal volume, verified 2026-07-30:

```
checkpoints/rloo_probe_checkpoints/rloo_probe_0.5b/
  probe_rloo_runA_coutcome_FINAL/epoch_0_step_{0,10,...,90} + latest_checkpoint
  probe_rloo_runB_csft_FINAL/    epoch_0_step_{0,10,...,90} + latest_checkpoint
```

They span the entire collapse (runA probe−verifier gap −0.120 → +0.671, verifier
0.572 → 0.321). What is missing is not checkpoints but **activations off** those
checkpoints, which is a forward-pass job, not a training job.

Phase 0 is therefore reframed as a **harvest**: ~$35–50 of GPU rather than
~$320 for two ladders, and hours instead of days. `harvest_ladder.py` +
`modal_fragility.py` implement it.

Cost of the reframe: 11 checkpoints per ladder instead of the 30–50 CLAUDE.md
asks for. I judged that acceptable to start, because the published trajectory
puts runA's cliff between steps 30 and 60 and the existing ladder brackets it
with three points. Densifying is a one-flag change (`--save_every_n_steps 3`,
34 checkpoints) and `dense_probe_rloo.py` is written and ready — but it should
be paid for only if the K=10 curve turns out to be under-resolved where the
transition actually happens.

## 2. H1 and H2 do not exhaust the hypothesis space, and the missing one is the leading one (fixed)

CLAUDE.md offers H1 evasion (info moved off the probe direction, preserved
elsewhere) and H2 corruption (info degraded). The original paper's own diagnosis
is **neither**. `findings.md` EXP-21:

> The probe direction picked up the structural/stylistic features, not the
> underlying mathematics. […] Classic confound exploitation.

Call it **H0**. It matters because H0 is *observationally confusable with H1*:
if the probe scored a surface correlate, then post-RL a retrained probe still
separates correctness (the information was never removed) while the frozen probe
fails (its boundary sat on the now-decoupled confound). That is exactly the H1
signature — a large positive `evasion_gap` — with a different mechanism and a
very different safety implication. "The model learned to hide" and "your probe
had a shortcut" are not the same paper.

Running the analysis with only H1/H2 in mind would produce a large positive
evasion gap and a write-up claiming evasion. A reviewer would ask this question
immediately.

Fixed by: `fragility_core/confounds.py` (a full H0 measurement suite),
`auroc_frozen_cond_template` wired through Phase 1, and an H0 column in the
Phase-2 pre-registration with the one measurement that actually discriminates —
does steering along the probe direction change the *writing style* without
changing correctness? Nobody has taken that measurement.

**The free control run has put partial numbers on H0** (see `PAPER_NOTES.md`):
on C_outcome, a 7-regex template score alone predicts correctness at AUROC
**0.824** (unbalanced), up from 0.587 at C_SFT, and the frozen probe's Spearman
correlation with that score rises 0.19 → 0.63 over vanilla RL. That trend is
solid. The residualisation test is **not** yet: it drops linear-probe AUROC
0.903 → 0.596, but the "matched placebo costs 0.001" comparison it was reported
against is invalid — the placebo was matched on column count, not removed
variance (template removes 31.4% of activation variance, placebo 2.0%), and three
of the placebo's seven columns are constant. A variance-matched PC control gives
0.895 vs 0.596 at C_outcome but the reverse at C_SFT. The magnitude of the
confound's contribution is open, not established.

## 3. The `dir_drift` metric as specified is close to uninformative (fixed)

CLAUDE.md defines `dir_drift(t)` as the cosine between the t=0 probe direction
and the checkpoint-t retrained direction. Measured on the control ladder:

| | cosine |
|---|---|
| two probes fit on **disjoint halves of the same checkpoint** | **0.05 – 0.18** |
| frozen probe vs retrained probe at t=100 | 0.55 |
| frozen probe vs retrained probe at t=0 | 0.13 |

Two probes fit on the *same data* agree at cos ≈ 0.1 while both score AUROC
≈ 0.85. The residual stream's dimensions are correlated enough that many
directions implement nearly the same classifier, so a small cosine does not mean
the probes disagree. Reporting a raw drift of 0.13 as "the direction rotated
away" would be wrong.

Fixed by two additions:
- `dir_drift_self_baseline` — the refit-noise floor, measured per checkpoint, so
  the interpretable quantity is `dir_drift / self_baseline`;
- `score_corr_frozen_retrained` — Spearman correlation between the two probes'
  **outputs**, which is well determined and is what "the monitor and a fresh
  monitor now disagree" actually means.

## 4. The Phase-2 α scale is off by ~30×, and the steering site was wrong twice (fixed)

**α.** CLAUDE.md specifies α ∈ {2,4,8} × "activation std at that layer". The
published steering results use α × `h_mean_norm` = α × 21.86 at L16. The mean
per-dimension std at L16 is ≈ 0.116 at C_outcome (and drifts down to ≈ 0.093 by
runA step 99, so it is not even a fixed reference). So α=8 × act_std injects ≈ 4%
of the paper's α=1.0 magnitude — it would return a null by construction and the
null would look like a finding. Fixed: the scale reference is an explicit config
field defaulting to `h_mean_norm` (comparable to the published [−0.07, +0.02]
null band and the +0.083 post-Goodhart result), with `proj_std` swept as a
secondary convention and the injected L2 norm recorded for every condition.

**Layer — fixed.** The probe reads `hidden_states[16]`, which in HuggingFace is
the output of `model.model.layers[15]`. The original `causal_steering.py` hooks
`model.model.layers[16]`, whose output is `hidden_states[17]`. The published
steering therefore injected the probe direction one block *downstream* of where
the probe reads. This does not invalidate the published null, but "steering at
the probe's layer" is not literally what was done. Phase 2 runs both conventions
(`hook_offset` 0 and −1) and reports them side by side.

**Token position — a bug the layer fix introduced, since fixed.** The first
Phase-2 steering implementation injected at the last token of the prefix. Since
`</think>` tokenises as three tokens, that site is **two positions downstream** of
where the probe reads, and its token identity differs from the calibration
sequence. This is the same class of error as `findings.md` EXP-21 bug #7, and it
returns a null by construction just as surely as the α mistake would have. The
current `run_steering.py` locates the injection site with
`char_to_token_index(offs, close_char)` — the probe's own helper — skips rollouts
whose `</think>` is truncated away rather than falling back to an arbitrary token,
and asserts the decoded token at the injection site on the first batch. So the
site is now correct at both the layer and the position, but the assessment should
record that it took two passes.

## 5. Where the code should live — a decision I made, flag if you disagree

CLAUDE.md's repo layout (`experiments/fragility/`, `fragility_core/`,
`configs/fragility/`, `results/fragility/`, `PAPER_NOTES.md` "at repo root") does
not say whether "root" is the paper repo or `followup/`.

I put everything under `followup/`, preserving the specified relative layout, and
treat `followup/` as this project's root. Reasons: the paper repo was just
refactored for public release (commit cd8cd5e) and adding four top-level
directories would undo that; and the follow-up imports the paper's code as a
read-only dependency, which is cleaner across a directory boundary. Nothing in
the original tree was modified.

Two smaller judgement calls in the same family:

- **Metric schema.** CLAUDE.md mandates `(run_id, phase, checkpoint_step, metric,
  value, seed)`. Those six are present and are the first six columns, but a
  metric is not identified without knowing its layer, arm, and probe, so
  `layer`, `arm`, `probe_id`, `extra_json` were added. A superset, not a change.
- **Model track.** CLAUDE.md says to port the RL setup to a 1.5–3B model "for
  iteration". No port is needed: the existing collapse is on Qwen2.5-**0.5B**,
  which is smaller and cheaper than the suggested track, and its checkpoints
  already exist. 0.5B *is* the small track; 1.5B (which also already has SFT and
  RLOO checkpoints on the volume) is the promotion target.

---

## What I disagree with, and did not change

**Phase 3 is priced like the cheap phase and is actually the expensive one.**
13 probes × 1 pilot RL run × ~$30 ≈ $390, more than the whole original project's
compute spend ($265). CLAUDE.md schedules the transfer experiment last and calls
it cheap; I think it should be the **gate**. The transfer matrix runs entirely
off cached activations for $0 and can answer "does probe choice buy robustness?"
in the direction that matters:

- every column collapses → the attack transfers, probe choice buys nothing,
  don't spend $390 rediscovering that;
- spread across columns → that *is* the H3 signal, and the RL runs are then
  worth paying for, starting from the extremes.

**As recorded, it does not answer that yet.** The only transfer results in
`results/fragility/metrics/` are over the **vanilla control ladder** (steps 0 and
100 — C_SFT and C_outcome), where all 13 probes score 0.79–0.90 at step 100
simply because verifier RL improves them. That is a sanity check on the zoo, not
a transfer test: no probe-RL checkpoint has been run through it. The attack-
transfer question needs the matrix evaluated on runA's post-collapse activations,
which are already harvested, so it remains a $0 job — it just has not been run.

`transfer_matrix.py` prints the decision rule above at the end of its run.
I have not reordered the phases in CLAUDE.md — that is your call — but the code
is built so the cheap evidence comes first.

**Three seeds for Phase 3 pilots may be unaffordable.** CLAUDE.md says 3 seeds
minimum for paper claims, 1 for pilots. With 13 probes that is 13 pilots + 3×N
follow-ups. The configs encode `seeds_pilot: [0]` and `seeds_paper: [0,1,2]`
separately so the escalation is a deliberate, visible step.

**The Phase-1 headline figure may not be the deciding figure.** If H0 is right,
Phase 1 produces a large positive evasion gap that looks like clean H1 evidence.
Two of the three hypotheses share that signature. What separates them is Phase
2's Δtemplate-under-steering measurement and Phase 1's
`auroc_frozen_cond_template`. Worth knowing before the figure is drawn.

---

## What is built and verified

Run for real, on this machine, at zero compute cost — see `PAPER_NOTES.md` for
the numbers:

- `fragility_core/` — 11 modules
- the vanilla-RLOO **control ladder** ingested from existing caches (5
  checkpoints × 3 layers) and the full Phase-1 suite run over it, 3 seeds
- 6 paper figures generated from parquet
- the runA probe-RL ladder: all 11 checkpoints harvested, Phase 1 run at 3 seeds
  across 5 layers (see `PAPER_NOTES.md`, 2026-07-31); runB at 2 checkpoints, 1 seed
- the 13-probe zoo built with pre-hoc features
- the residualisation test with two controls (a placebo feature set and random
  projections). The placebo turned out not to be a valid control — it is matched
  on column count, not removed variance — so the test currently has one usable
  control and an open question. Two further rows quoted in an early draft
  (shuffled template columns, random binary columns) were never computed; they
  existed only in a docstring and have been removed from the write-up.

Written, compiles, needs GPU/Modal: `harvest_ladder.py`, `run_steering.py`,
`run_patching.py`, `dense_probe_rloo.py`, `modal_fragility.py`.

Not started: Phase 4 (correctly — it is conditional on Phase 3 finding a
predictor).

## What I'd do next, in order

*(Superseded 2026-07-31 by the scoping harvest — see below. The original ordering
put runB first because its collapse is faster. That is true of the dynamics and
irrelevant to the mechanism: runB initialises from C_SFT, where both the probe
and the confound are weak before the attack starts, so there is little room to
measure a decline. runA is the primary ladder.)*

1. ~~**Harvest runA's full ladder**~~ **done** — all 11 checkpoints harvested.
2. ~~Run Phase 1 across the ladder, 3 seeds~~ **done** — `evasion_gap` read
   against the control's measured null band, **[−0.021, +0.041]** unbalanced and
   **[−0.024, +0.041]** balanced. Result: a two-phase collapse in which reward
   hacking precedes monitor failure by ~40 steps (`PAPER_NOTES.md`, 2026-07-31).
3. Run the transfer matrix on the probe-RL checkpoints ($0) before committing to
   Phase 2 or 3. The version on disk covers only the vanilla control ladder and
   does not answer the transfer question.
4. Phase 2 only after that, and only after re-reading `phase2_predictions.md`.
5. runB's full ladder last, as a robustness check, if the budget allows.
6. A variance-matched control for the template residualisation test, at both
   C_SFT and C_outcome, before any H0 magnitude goes in the paper.
