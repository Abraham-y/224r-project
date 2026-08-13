# STATUS — read this first

Single entry point. Everything else is either the deliverable, an active work
item, or archive. Last updated 2026-08-06.

---

## What this project is

A linear probe on Qwen2.5-0.5B layer-16 `</think>` hidden states predicts whether
a Countdown rollout will be correct. The finding is an asymmetry:

- **As a read-only selector it is excellent.** Best-of-16 by probe score reaches
  0.6700 against a random-pick-of-16 baseline of 0.5531 and an oracle ceiling of
  0.6847 — **89% of all available headroom**. It holds AUROC 0.939 with trace
  length stratified out, against 0.564 for a pure length detector.
  *(Corrected 2026-08-08: previously 0.6798 / 96% / 0.981. 0.6798 does not
  reproduce — the held-out figure is 0.6700 — the 0.5493 baseline came from a
  different rollout sample, and 0.981 was an in-sample number. See
  `REVISION_PACK.md` §B1–B2; all of it now comes out of
  `extension/probe/structural_baselines.py`.)*
- **As an RL reward it is catastrophic.** Probe score rises while true accuracy
  falls ~31 pp.

Same artefact, opposite outcomes. Reading a representation and writing to it are
different privileges, and probe accuracy licenses only the first.

**Target:** JUDGe workshop (backup: Verify-Agents). Deadline **2026-08-29**.

---

## Where it stands

**Updated 2026-08-12.** The revision is now *applied*, not just planned. Until
today `writeup_workshop.md` and `.tex` were still the pre-correction draft and
contained none of `REVISION_PACK.md`'s fixes. They now do:

- **Section 3 is cut**, replaced by an explicit withdrawal in the paper's own
  Limitations and a `\subsection` that states all three reasons (cosine 0.163
  not 1.000, hook one block + 2-3 tokens off the read site, p = 0.063 / 0.080,
  zero shared prompts).
- **The false novelty sentence is gone.** Baker, Gupta & Jenner, Taufeeque and
  Bailey are cited; the delta is stated as penalty-vs-positive-reward.
- **All three unbacked "statistically indistinguishable" claims are replaced**
  with tests or with softened wording.
- **The deployment table is one population** with the free structural baseline
  as a row, +11.69 pp against the correct denominator, and the estimator
  noise floor stated.
- **The surface decomposition and both pre-registered arms are now IN the
  paper** as the new Section 3, including the Arm B rollout in a box.
- **-25 pp is -31.4 pp** everywhere, and the ladder table now shows read-only
  selection beating every training-time construction.
- `writeup_workshop.tex` compiles clean (16 pp, `article` 11pt).

Two code defects found in the 2026-08-12 audit are fixed:

- `surface_battery.py` compared a score on the held-out half against an accuracy
  over all rows. Corrected to 0.5173; the centrepiece delta widens to -0.387.
  Artifacts regenerated.
- `causal_steering.py` hooked `layers[L]`, one block downstream of
  `hidden_states[L]`. Now `--layer_convention hidden_state` by default, with
  `legacy_block` to reproduce the shipped runs.

Reproduction check: `structural_baselines.py`, `surface_battery.py` and
`quantify_structural_confound.py` were each re-run end-to-end and diffed against
their committed JSON. **Zero differences**, including the 10k-resample bootstraps.

Assessed ~6.5 / 7.5 / 5.5 (novelty / interest / usefulness) — defensible weak
accept, up from a Reject verdict at the start of this work.

### Arm A/B are now verifiable (2026-08-12)

Pulled `arm{A,B}_*_step100.json` off `default-proj-training`
(`evaluation/eval_results/`). All four published accuracies recompute **exactly**
— 0.5306 / 0.2361 / 0.1678 / 0.0000 — and the four contrast CIs agree to the
third digit. `extension/probe/verify_residual_arms.py` is now a standing check:
it prints the published value beside each recomputed one and flags disagreement.

Two things the verification turned up:

- **The arms were evaluated at 8 responses per prompt, the references at 16.**
  3,248 rollouts against 6,496. The bootstrap pairs on prompts and uses
  per-prompt means so the contrast is fine, but the pack never said it.
- **"1095 → 2394" mixed populations** — a clean-406 baseline against an all-500
  Arm B. On one population it is **1098 → 2390**.

The Arm B rollout the paper quotes is real, and the full four-line block is
sharper than the two-line excerpt that was in the draft. Both the `.md` and
`.tex` now carry it verbatim with the input and target stated.

### Still open

1. **Format.** 16 pages of `\documentclass[11pt]{article}` against a workshop
   limit of 4-9 pages in the venue's own style. `neurips_2024.sty` is in the
   repo. This is a content decision (what to cut), not a mechanical one.
2. **Push.** Everything is committed locally; nothing has been pushed.

### What changed earlier (2026-08-08 audit)

- **§3 must be cut.** Its table claims the steered vector is the direction RL
  optimised at "cosine 1.000, AUROC 0.982". Measured: **cosine 0.163**, AUROC
  0.896. Steering was also applied one transformer block and 2–3 tokens
  downstream of the probe's read site (already flagged in `CODE_AUDIT.md:503-505`),
  and neither key contrast is significant (p = 0.063, p = 0.080). Not fixable —
  the checkpoints are gone.
- **§1 and §4 verify exactly.** The drift table reproduces to the digit
  (2983/490/150/1835); §4's +1.5 pp and 1.83→0.91 reproduce to the decimal.
- **Line 57 was wrong and its replacement is better.** "Statistically
  indistinguishable from the F→F floor" is false (+0.065, CI [+0.023, +0.105]),
  but the residue is only 8.7% of the way from floor to ceiling — a measurable,
  almost entirely erased trace. That is a positive result where there was a null.
- **Line 24's novelty claim is false.** Gupta & Jenner study exactly what it says
  nobody studies. Must be rewritten; the real delta is that prior work uses the
  monitor as a *penalty* (evasion is the argmax) while this uses it as a *positive
  reward* with no evasion incentive — and it breaks anyway.
- **New centrepiece (§G2).** A 7-feature logistic regression — no network, no
  forward pass — reproduces the probe's Goodhart signature: its score rises
  0.506 → 0.832 while true accuracy falls 0.498 → 0.130.
  *(Corrected 2026-08-08: previously 0.588 → 0.932 and 0.588 → 0.276. Until now
  no script in the repo computed §G2 at all — `n_backtrack` and `has_therefore`
  existed in no source file — so the old figures were unreproducible. It is now
  `extension/probe/surface_battery.py`; the claim holds, the magnitudes moved,
  and the per-feature shifts turn out to be negative, not positive.)*

Full detail and every corrected table: **`REVISION_PACK.md`**.

---

## The experiment

Testing the remediation the paper's own line 179 proposes as an open question:
does a probe that cannot read cheap surface features resist Goodharting?

| arm | reward probe | read-only AUROC | accuracy @ step 100 |
|---|---|---|---|
| reference | C_outcome (verifier RL) | — | 0.5306 |
| reference | published probe-as-reward | 0.978 | 0.2361 |
| **A** | surface-residualised `LR(h − s@B)` | 0.834 | **0.1678** |
| **B** | surface features ONLY, ignores activation | 0.925 | **0.0000** |

**Arm A falsified its own pre-registered prediction.** Residualising made the
collapse **worse**: −6.83 pp vs the published run, CI [−8.85, −4.79]. The
remediation the paper proposes does not work and backfires.

The "weaker probes collapse less" confound is dead — Arm A is weaker *and*
collapsed harder, so weakness cannot explain a result in that direction.

**Arm B CONFIRMED its pre-registered prediction.** A reward with no access to the
model's internals destroyed the task outright (0.0000, −23.6 pp vs the published
run, CI [−26.4, −20.9]). §G2 is now a tested claim. The failure is not degenerate:
99.9% of rollouts still emit `<answer>`, but the block contains reasoning prose
instead of an equation, and mean length doubled. See `REVISION_PACK.md` §G5.

Total cost ~$90. Both arms complete.

---

## Active files

| file | what it is |
|---|---|
| `writeup_workshop.md` | **the deliverable** — the paper being submitted |
| `REVISION_PACK.md` | **the work list** — every corrected number, table, and citation, with populations labelled |
| `STATUS.md` | this file |
| `followup/experiments/fragility/residual_probe/PREREGISTRATION.md` | **do not edit** — predictions committed before the runs |
| `followup/experiments/fragility/residual_probe/HANDOFF.md` | commands, gates, and the five deployment bugs already hit |
| `followup/RELATED_WORK.md` | literature position; Baker, Gupta & Jenner, Obfuscation Atlas |

## Reference (not working docs)

`findings.md`, `CODE_AUDIT.md`, `writeup.md`, `README.md`,
`RUNBOOK_CLEAN_REPLICATION.md`, `followup/CLAUDE.md`, `followup/README.md`,
`followup/PAPER_NOTES.md` (carries explicit retractions — worth reading before
re-asserting any earlier claim).

Superseded process docs are under `docs/archive/`.

---

## Two standing traps

1. **Do not let §G2 or §G4 become mechanism claims.** The measured facts are that
   a surface model reproduces the Goodhart signature, and that residualising
   coincided with a worse collapse. *Why* is unestablished. "Less-constrained
   boundary, more room to exploit" is a hypothesis. Asserting it as a finding is
   exactly how §3 became indefensible.
2. **"Statistically indistinguishable" has appeared three times in this paper
   without a test** (§3's null band, line 88, line 57). At least two were wrong.
   Search for the phrase before submitting.

## One methodological note worth keeping

Five deployment defects were hit during the experiment — credentials, Ray
`PYTHONPATH`, sklearn version, `load_dataset` on a local file, container path.
**All five lived outside what in-process gates can see, and the gates stayed green
and correct throughout.** In-process correctness and deployment correctness are
close to disjoint. A 5-step smoke run (~$2, ~10 min) caught every one of them
before a 5.5-hour job did.
