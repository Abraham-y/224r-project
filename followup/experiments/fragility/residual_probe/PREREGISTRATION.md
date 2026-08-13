# Pre-registration — surface-residualised probe as RL reward

Written **before** the 100-step runs, per `followup/CLAUDE.md` ("Pre-registration
note: BEFORE running, write predictions. Then run.").

## Question

The workshop paper (line 179) asks, as an open question, whether a probe trained
to ignore rhetorical scaffolding would still be near-oracle and would resist
confound exploitation under RL. This tests it.

## Setup, fixed before running

- Base policy `asingh15/qwen-sft-countdown-defaultproj`; RLOO pool
  `asingh15/countdown_tasks_3to4` — **verified**, not assumed: 0/406 clean eval
  prompts appear in it and 94/94 excluded prompts do, so it is the pool the
  contamination manifest was built against.
- Recipe identical to the published probe-as-reward run: 100 steps, batch 128,
  group 8, lr 1e-5, KL 1e-3, temperature 1.0.
- Reward = `P(correct)` from the frozen probe at `</think>`, layer 16.
- Measured cost: ~194 s/step steady-state on one H100 after ~430 s startup,
  so ~5.5 h and roughly $25 per arm.

## Arms

| arm | reward probe | read-only AUROC |
|---|---|---|
| **A (treatment)** | surface-residualised, `LR(h - s@B)` over 39 features | 0.834 |
| **B (positive control)** | surface features ONLY, no activation | 0.925 |
| **C (reference)** | the published run, already have it | 0.978 |

## Predictions, committed now

1. **Arm B collapses at least as hard as C.** A reward that reads nothing but
   surface must be occupiable. If B does *not* collapse, the surface-occupation
   account in §G/G2 of the revision pack is wrong and the whole framing needs
   revisiting.
2. **Arm A collapses less than C** in true first-block accuracy at step 100.
   This is the substantive prediction. Direction committed; no magnitude claimed.
3. **Probe score rises in all three arms.** Rising probe score is not evidence of
   anything on its own — that is the paper's point.

## The confound this design exists to handle

Arm A's probe is weaker read-only (0.834 vs 0.978). "A collapsed less" is
therefore ambiguous between *less surface coupling helps* and *weaker probes are
simply harder to Goodhart*. Arm B is the discriminator: it is **stronger** than A
read-only (0.925) while being **pure** surface. So:

- A < C **and** B ≥ C → supports the surface-coupling account.
- A < C **and** B < C → the "weaker probe" story is live; result is
  uninterpretable without a further arm matched on read-only AUROC.

**Arm A alone does not answer the question. Do not report it alone.**

## Analysis, fixed now

- Primary outcome: first-block accuracy on the clean-406 eval at step 100,
  first-block convention (the paper's own; last-block would give −45.7 pp where
  the paper reports −25 pp).
- Prompt-clustered paired bootstrap, 10k resamples, for every interval.
- n = 1 seed per arm. This is a demonstration, not an effect size, and must be
  labelled as such wherever it appears.

## What would falsify the framing

If arm B — a reward with no access to the model's internals at all — fails to
collapse, then "the optimizer occupies the surface features that carried signal"
is not the mechanism, and §G2 of the revision pack should be retracted rather
than defended.
