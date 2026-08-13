# Revision pack — corrected numbers for `writeup_workshop.md`

Everything below is measured from the released artifacts on this machine. Every
number states its population. Nothing here is interpolated or carried over from a
previous draft.

Written 2026-08-04. Prose is deliberately absent: this is the numbers layer only.

---

## A. DELETE: Section 3 (lines 118–140), the mech-interp signature

Three independent problems, any one of which is disqualifying. None is fixable
without a rerun, and the runA/runB checkpoints needed for a rerun are deleted.

**A1 — the table's central claim is false.** Line 129 reports the steered vector
as the optimized direction, `cosine w/ optimized 1.000 | AUROC 0.982`. Measured:

| quantity | paper | measured |
|---|---|---|
| cosine(steered, the probe RL actually optimized) | 1.000 | **+0.163** |
| AUROC of the steered vector | 0.982 | **0.896** (from its own `.npz`) |
| AUROC of the probe RL optimized (`..._temp1.pkl`) | — | **0.810** |
| cosine(steered, assertion control) | 0.038 | +0.038 ✓ |

The assertion cosine matching to three decimals confirms which vector was steered:
`C_outcome_l16_pre_answer_direction.npz`, the eval-regime probe. RL optimized
`probe_pipeline_C_outcome_l16_pre_answer_temp1.pkl`, a different probe fit under
sampling temperature 1.0. 0.163 is 4.9σ in 896 dimensions — non-random, but not
identity. The experiment is real; its label is wrong.

**A2 — the steering site is not the read site.** `hidden_states[16]` is the
*output* of `model.model.layers[15]`, so a hook on `layers[16]` writes one block
downstream of where the probe reads; the write also lands 2–3 tokens after the
`</think>` position the probe scores. This is the same off-by-one the authors
found and fixed on the RL side (`findings.md:717`, bug #7) and never fixed in
`causal_steering.py`. `CODE_AUDIT.md:503-505` already records it as *"the one
outstanding item that could change a published conclusion."*

**A3 — neither required contrast is significant.** Recomputed with the validated
signed-contrast estimator, prompt-clustered bootstrap, 10k resamples:

| contrast | Δ | 95% CI | p |
|---|---|---|---|
| probe direction, post-RL | +0.080 | [+0.020, +0.140] | 0.010 |
| assertion direction, post-RL | −0.015 | [−0.090, +0.065] | 0.744 |
| **probe − assertion** (the "sharp contrast", line 132) | +0.095 | SE 0.050 | **0.063** |
| **post-RL − vanilla** (the "becomes causal" claim) | — | — | **0.080** |

Line 132's "kills the alternative explanation" rests on a difference between a
significant and a non-significant result, which is itself not significant
(Gelman–Stern). The three steering runs also share **zero prompts**, so they are
not paired.

**Replacement:** two sentences in Limitations. Something like — *we attempted a
causal test of whether RL makes the probe direction causal; the steering hook was
placed one block and 2–3 tokens downstream of the probe's read site, the steered
vector has cosine 0.163 with the direction actually optimized, and the key
contrast reaches only p = 0.063. We report it as inconclusive.* Volunteering this
is worth more than defending it: it is discoverable from the released artifacts in
under an hour.

---

## B. REPLACE: the deployment table (Section 2, lines 65–79)

Line 65's "all numbers on the C_outcome held-out 406" is not true of the table as
printed — the adaptive-budget row is a 288-prompt subset with pass@1 = 0.719,
printed under a 0.549 header. Table numbers also disagree with
`probe_usefulness_suite_results_n406.json`.

### B1. Best-of-16 selection — identical 406 prompts, identical 6,306 rollouts

This is the table to lead with. It is new, it is free, and it makes the paper
stronger rather than weaker.

**CORRECTED 2026-08-08.** An earlier version of this table read `probe argmax
0.6798` against `pass@1 0.5493`, for **96%** of oracle headroom, and those two
figures are the ones STATUS.md and section §B2/§9 quoted. Neither survives
recomputation, and they are not from the same population as each other:

- **0.6798 is not reproducible.** Recomputed from
  `extension/cache/probe_cache_n500_clean406/C_outcome_l16_pre_answer` on the same
  GroupKFold(5)-by-prompt recipe, probe best-of-16 is **0.6700**. Fitting the probe
  in-sample and selecting with it gives 0.6773 — closer to 0.6798, and leaky.
  0.6700 is what `extension/probe/structural_baselines.py` computes and what
  `extension/outputs/n500/text/60_structural_baselines.json` has held all along.
- **The baseline was from a different rollout sample.** `pass@1 = 0.5493` comes
  from `probe_usefulness_suite_results_n406.json`, which sampled its own rollouts;
  the oracle (0.6847) and shortest-`<think>` (0.6379) rows come from the 6,306-rollout
  cache, where first-rollout pass@1 is **0.5443** and the expected accuracy of one
  of the 16 is **0.5531**. Mixing them inflated both the lift and the share.

All rows below are now from the cache, one population, held-out scores, with
prompt-clustered paired-bootstrap CIs (10,000 resamples, n = 406 prompts) —
recomputed, not rescaled, since the previous intervals were built on the
withdrawn point estimate:

| selector | accuracy | 95% CI | lift vs random-of-16 | 95% CI (pp) | % of oracle headroom |
|---|---|---|---|---|---|
| random pick of 16 (expected accuracy of one) | 0.5531 | [0.5110, 0.5956] | — | — | 0% |
| first-rollout pass@1, same prompts | 0.5443 | — | — | — | — |
| **shortest `<think>` — free, no forward pass** | 0.6379 | [0.5911, 0.6847] | **+8.49 pp** | [+6.18, +10.80] | **64%** |
| 39-feature surface model — free | 0.6379 | [0.5911, 0.6847] | +8.49 pp | [+6.14, +10.90] | 64% |
| **probe argmax** | **0.6700** | [0.6232, 0.7143] | **+11.69 pp** | [+9.23, +14.21] | **89%** |
| oracle pass@16 (ceiling) | 0.6847 | [0.6404, 0.7291] | +13.17 pp | [+10.89, +15.61] | 100% |

**probe − shortest-`<think>` = +3.20 pp, 95% CI [+0.99, +5.42], p = 0.0076.**
Significant, and it is the number README.md has always quoted. **probe − the
39-feature surface model = +3.20 pp, [+0.99, +5.42], p = 0.0042** — also
significant, so the deployment claim clears the strongest free baseline available
and not just trace length.

**The share of the oracle headroom is 88.8%, 95% CI [78.9%, 96.5%]** — bootstrapped
as a ratio, because the denominator is estimated on the same 406 prompts as the
numerator and moves with it. Note that the withdrawn "96%" sits at the very top of
that interval: the corrected point estimate is not merely lower, the old value was
never a central estimate of this quantity.

Two caveats that belong in the caption:

- **K is a ceiling, not a constant.** Rollouts with no locatable `</think>` are not
  in the cache, so a prompt offers at most the rollouts that were cached: mean
  effective K **15.53**, minimum 8, and only **70.9%** of prompts have the full 16.
  Every arm — random, shortest, probe, oracle — is computed on the same reduced
  candidate set, so the contrast is internally consistent, but a reader comparing
  against an uncached best-of-16 needs to know.
- **The CIs above are the recomputed ones.** The intervals in the previous version
  of this table were built on the withdrawn 0.6798 and are not carried forward;
  these come from `structural_baselines.py` section (2b) and are in
  `60_structural_baselines.json` under `bootstrap` / `contrasts`.

Two things this table still buys, on the corrected numbers:

1. **A stronger headline than AUROC.** The probe captures **89% of the gain any
   selector could achieve** on these rollouts. "Near-oracle" stops being a gloss on
   0.982 and becomes a measured operational fact — a weaker one than 96%, and one
   that survives being checked.
2. **It pre-empts the obvious attack.** A reviewer *will* ask whether the probe is
   reading trace length. Answering before they ask, with the baseline in the table,
   converts the paper's biggest vulnerability into a contribution. Note the honest
   shape of that answer: the free structural baseline already captures 64% of the
   headroom, and the probe adds the remaining 25 points of it.

### B2. The length confound, closed

**CORRECTED 2026-08-08.** The previous version of this table quoted 0.995 and
0.981, which are **in-sample** figures — probe fit and scored on the same 6,306
rows — set against a one-scalar baseline that needs no fitting at all. That
flatters the probe twice, and 0.981 is the number §G and STATUS.md leaned on.
Held-out (GroupKFold(5) by prompt, the same recipe as everything else here):

| | AUROC | |
|---|---|---|
| `</think>` character index alone (one scalar, no model) | 0.913 | |
| `</think>` token index alone, cross-validated | 0.915 | |
| probe | **0.982** | (in-sample: 0.995 — do not quote) |
| **probe, stratified within `</think>`-position deciles** | **0.939** | (in-sample: 0.981 — do not quote) |
| position, stratified the same way (sanity: should be ~0.5) | 0.564 | |

Length is a strong correlate of correctness in this setup — worth stating plainly,
it is a finding — and the probe keeps most, not essentially all, of its
discrimination when length is held fixed. **The probe is still not a length
detector**: 0.939 is far above the 0.564 a pure length detector scores under the
same stratification. But the honest cost of stratifying is **−0.043** (0.982 →
0.939), not the −0.014 the in-sample pair implied — three times larger.

`extension/probe/structural_baselines.py` now prints the held-out and in-sample
columns side by side, labelled, so the two cannot be confused again.

**But length is not the surface.** Residualising the activations against the full
**39**-feature surface battery (fit on held-out-by-prompt training rows, applied to
held-out prompts) is far more costly to the probe than stratifying on length alone.
*(39, not 23 — this was mislabelled. The battery is 15 structural features plus 24
reasoning-marker word counts; **23** is the count that survives dropping the
near-constant columns, and it applies only to the dose-response in §G2, never to
the residualisation. See the note at the top of
`followup/experiments/fragility/residual_probe/surface_residual_probe.py`.)*

| probe trained on | held-out AUROC | corr. with the surface model's score |
|---|---|---|
| raw activations | 0.9775 | +0.858 |
| **surface-residualised activations** | **0.8363** | +0.255 |

So the honest two-part claim, and it is more interesting than either half:

> The probe is not a *length* detector — it survives length-stratification at 0.939,
> against 0.564 for a pure length detector. But a large share of its signal is
> surface-explainable in the broader sense: strip 39 cheap textual features out of
> the activations and it falls from 0.978 to 0.836.

This partially **rehabilitates** the paper's original intuition (line 97) while
still refuting the specific version it stated. Note the standard caveat: correctness
genuinely causes shorter, denser traces, so residualising against 39 correlated
features removes some real signal too. 0.836 is a lower bound on the probe's
non-surface content, not a point estimate.

**Selector check, full surface battery** (the strongest free baseline available).
Also corrected: the previous version paired a `0.6429 / 0.9252` surface row with an
in-sample probe. 0.9252 is the *shipped surface-only probe's* held-out AUROC under a
different label rule and a different split — not this population. All rows below are
now one population, one label rule (first `<answer>` block), one set of folds, and
they come out of `structural_baselines.py` rather than an unrecorded session:

| best-of-16 selector, clean-406 | accuracy | AUROC |
|---|---|---|
| random pick of 16 | 0.5531 | — |
| shortest `<think>` (one scalar, free) | 0.6379 | 0.915 |
| 39-feature surface model (free, held-out) | 0.6379 | 0.926 |
| **probe** | **0.6700** | **0.982** |
| oracle | 0.6847 | — |

**probe − full surface model = +3.20 pp.** The deployment claim survives against a
strong free baseline, not just against trace length.

*(The two free baselines landing on exactly 0.6379 is a coincidence, and it was
checked rather than assumed: they select the same rollout on only 170 of 406
prompts — 41.9% — and happen to get 259/406 right either way. Do not read it as a
copy-paste error, and do not read it as the two baselines being the same thing.)*

*(Every AUROC in this section is on the 6,306-rollout clean-406 cache, held out by
prompt. Label the population and the fitting regime whenever you quote one; mixing
an in-sample probe with a cross-validated baseline is what this correction is
about.)*

### B3. Rows to correct against `probe_usefulness_suite_results_n406.json`

**These are corrections to the paper's Section-2 table only.** They are NOT the §B1
figures: the suite sampled its own rollouts, so its numbers describe a different
population from the 6,306-rollout cache §B1 uses. State which population each row
belongs to, and never subtract one from the other.

| paper | suite artifact (its own rollouts) | cache (§B1's population) |
|---|---|---|
| 0.670 | 0.6626 | 0.6700 |
| 0.980 | 0.9703 | — |
| ±0.4 pp | ±0.575 pp | — |
| AUROC 0.982 | 0.9777 | 0.982 (held out) |
| adaptive-budget 0.830 | different 288-prompt population, pass@1 = 0.719 — move out of this table or give it its own header | — |

That the paper's 0.670 and the cache's 0.6700 agree is a coincidence of two
different rollout samples, not a confirmation. Pick one population per table.

### B4. Dropped rollouts — a footnote, not a problem

Rollouts with no `</think>` are excluded from probe scoring. **Recomputed
2026-08-08 on the clean-406 population** — the previous counts were `n/8000`, i.e.
over all 500 prompts, while every AUROC they footnote is on the contamination-
filtered 406 (6,496 rollouts). Same filter, correct denominator:

| | rollouts | dropped (no `</think>`) | accuracy of the dropped |
|---|---|---|---|
| C_outcome | 6,496 | 190 (2.9%) | 0.000 |
| C_SFT | 6,496 | 673 (10.4%) | 0.000 |

Small on C_outcome, *not* small on C_SFT — 10.4% is an order of magnitude more than
the "2.1%" previously printed, and it is the checkpoint the probe is weakest on. All
of them are wrong, so scoring them as wrong is free and makes the monitor better:
full-population AUROC 0.982 → 0.983 on C_outcome and 0.912 → 0.924 on C_SFT. The
filter therefore flatters C_SFT's *reported* AUROC more than C_outcome's, which
slightly inflates the C_SFT → C_outcome gain as printed (+0.070 filtered, +0.059
full-population). Zero rollouts are dropped for any other reason (truncation
removes none at this sequence length), which is now asserted in section (4) of
`structural_baselines.py` rather than assumed. One footnote disclosing the rule and
these counts is sufficient — but disclose it, because an undisclosed filter that
correlates with the label reads badly even when it is harmless.

---

## C. STRENGTHEN: the multiplicative-shaping result (line 146)

The paper reports `+2.8 pp … z ≈ 3.6, p < 0.001 on the rollout-pooled estimator`.
The rollout-pooled estimator is the wrong one — it treats 16 rollouts from one
prompt as 16 independent observations. The correct prompt-paired, prompt-clustered
bootstrap is **tighter**, not looser:

| | value |
|---|---|
| base → multiplicative | 0.5827 → 0.6111 |
| paired delta | **+2.84 pp** |
| 95% CI (prompt-clustered, 10k resamples, n = 500 prompts) | **[+1.85, +3.85]** |

The claim survives, better supported than as printed. **The real limitation is
different and must be stated: n = 1 training run.** A single-seed RL delta of
+2.8 pp is a demonstration, not an effect size. Say so.

---

## D. Arithmetic and scope fixes

| line | printed | correct |
|---|---|---|
| 94 | "0.236 (vs C_outcome's 0.55 — a −25 pp drop)" | 0.236 vs 0.550 = **−31.4 pp** |
| 10 | "replicated at 1.5B" | qualify: which result, which n, and that the probe was refit |
| 88 | "statistically indistinguishable from random-direction perturbation at every tested magnitude" | false at one point — vanilla α = 0.5 gives Δ = −0.072, CI [−0.141, −0.010], excludes zero. Either drop "every" or report the exception |
| 10 | "near-oracle" as a gloss on AUROC 0.982 | now defensible on its own terms — 89% of oracle selection headroom (§B1) |

### D2. Two applied figures were the in-sample ones — and the artifacts said so

**NEW 2026-08-13.** `CODE_AUDIT.md` §C4 flags three scripts that select a
threshold on the evaluation data. Two of those numbers are in the paper, and in
both cases the producing script **already computes the held-out version and
prints "report this" next to it**. The draft quoted the line above it.

| figure | in-sample (was quoted) | held-out (correct) |
|---|---|---|
| probe-guided restart | 0.675 at 6.27 avg rollouts | **0.6675 at 5.54** |
| within-rollout probe-commit | 0.6607 | **0.6601** (+8.61 pp over baseline) |

Both are in `32_probe_guided_restart.txt` and `26_probe_answer_commit.txt` under
explicit `BEST (IN-SAMPLE ... optimistic)` / `BEST (HELD-OUT ...) -- report this`
headers. The gap is small — the objective is flat over the grid — but "we tuned
two hyperparameters on the test set" is not a sentence worth risking for 0.7 pp,
and the fix reads better anyway: **restart is now level with best-of-16 (0.6675
vs 0.6700) at −64% of the sampling compute**, which is a cleaner claim than
beating it by 0.5 pp.

Selective abstention is **not** affected. Its threshold is chosen to hit a
*coverage* target, not to maximise the reported accuracy, so the 0.980-at-50%
figure is not selected on the quantity it reports.

### D1. One AUROC table, three estimator conventions

**NEW 2026-08-12.** Not previously in this pack, and it is the one remaining
defect a careful reviewer finds without leaving the paper — the numbers disagree
with each other on the page.

`extension/outputs/poster_numbers.json` stores **two** held-out AUROC estimators
per cell and its own `_note` field says *"Pick ONE for the paper and say which."*
The paper picked neither; it uses both, plus a third convention in §3.

| position | `writeup_workshop.tex` §4 table | `balance_then_cv` | `cv_then_balance` | abstract / README / text artifacts |
|---|---|---|---|---|
| `</think>`, C_SFT → C_outcome | 0.904 → 0.980 | 0.9044 → 0.9796 | 0.9121 → 0.9820 | 0.912 → 0.982 |
| assertion, C_SFT → C_outcome | 0.887 → 0.852 | 0.8867 → 0.8523 | 0.8852 → 0.8955 | — |

Two things follow, and the second is worse than the first:

1. **The abstract and the table disagree.** 0.982 in one place, 0.980 in the
   other, for the same quantity. Small, but it is on the first page.
2. **The assertion row changes sign between conventions**: Δ = −0.035 under
   `balance_then_cv`, **+0.011** under `cv_then_balance`. The table's claim that
   outcome RL *weakens* the assertion probe is an artefact of which estimator was
   used, not a finding.

There is a third convention in play. §3's specificity control quotes the assertion
probe at **AUROC 0.70**, which is the *original-label* figure from
`13_cross_position_transfer.txt`; the corrected-label value is 0.852 or 0.896
depending on the estimator. So §3's contrast table pairs a corrected-label probe
AUROC (0.982) against an original-label assertion AUROC (0.70) inside a single
row. (§3 is being cut, so this one dies with it — but do not carry 0.70 into the
Limitations replacement without saying which label rule it is on.)

**Fix.** Use `cv_then_balance` everywhere: it is what the abstract, the README,
`40_relabel_cross_checkpoint.txt` and every other text artifact already report, so
it is the smaller edit and the one that makes the most existing numbers correct.
Regenerate the §4 position table from it, and add one sentence to Setup naming the
estimator and the fold rule. Also note `followup/PAPER_NOTES.md`'s standing
warning: balanced AUROC moves by up to ~0.015 on re-analysis because the
subsample is seed-dependent, so do not quote a balanced figure to three decimals
where a 0.015 jitter would change the reading.

---

## E. Citations

**Add**, with the delta stated explicitly:

- **Baker et al., arXiv 2503.11926** — CoT monitorability under optimization
  pressure. Note the distinction: their monitor-score drop conflates "the agent
  evaded" with "the monitor stopped discriminating."
- **Gupta & Jenner, "RL-Obfuscation", arXiv 2506.14261** — latent-space probes
  evaded by RL; roughly 70/30 behaviour/representation decomposition.
- **Taufeeque et al., "The Obfuscation Atlas", arXiv 2602.15515** (ICML 2026 Oral).

**Restore: Bailey et al.**, cited at `report.tex:63` in the earlier version and
dropped from the workshop draft. Dropping a citation between versions is the kind
of thing that gets noticed.

**The delta to claim.** Both Baker and Gupta & Jenner use the probe as a
**penalty** — the policy is directly incentivised to evade. Here the probe is a
**positive reward** with no evasion term anywhere in the objective, and the monitor
still collapses. That is a different and arguably more alarming mechanism: you do
not need to be attacking the monitor to break it. State this in one sentence; do
not oversell it.

---

## F. Reproducibility

**RESOLVED 2026-08-12.** This previously read that nothing in Section 2 or 3
could be regenerated from the repo. That is no longer true:

- the removed scripts are back (`probe_usefulness_suite.py`,
  `firstanswer_rloo.py`, `phase2a_per_answer_correctness.py`);
- §B, §G and §G2 are computed by `structural_baselines.py` and
  `surface_battery.py`, both of which re-run to a zero diff against their
  committed JSON;
- §G4/§G5 are computed by `verify_residual_arms.py` off rollouts now fetched
  from the Modal volume;
- the small `.txt`/`.json` outputs are no longer gitignored, so the numbers are
  checkable without rerunning anything.

What remains unreproducible is exactly one thing, and the paper says so in
Limitations: the withdrawn §3 causal-steering result, whose checkpoints are
deleted. The two arms' rollout JSONs (~10 MB each) stay out of git and are
fetched with the two `modal volume get` commands in
`verify_residual_arms.py`'s docstring.

---

## G. REWRITE: the mechanism claim (lines 97, 179) — it is currently wrong

Line 97 (*"picked up the structural surface, not the underlying mathematics"*) and
line 179 (*"the rhetorical surface is fused with correctness in the learned
direction"*) rest on ~30 hand-inspected rollouts, and **§B2 argues against them**:
a pure surface detector would be fixed by regressing out length, and the probe
retains AUROC **0.939** with length held fixed (held out; 0.981 was the in-sample
figure — see the §B2 correction), against 0.564 for a pure length detector under
the same stratification. It is not a surface detector at rest.

But the intuition is not simply wrong either. Running the **frozen length-detector
on the attacked policy's own rollouts** — a detector we know reads nothing but
surface — measures how much of the collapse is surface:

**REGENERATED 2026-08-08** by `extension/probe/surface_battery.py`, like §G2 —
this table had no script either. Clean-406, `last_block`, verifier =
`evaluation.countdown`:

| | baseline C_outcome | runA post-RL |
|---|---|---|
| true accuracy | 0.4980 | 0.1301 (**−36.8 pp**) |
| median `</think>` position | 773 chars | **625 chars** |
| length-detector AUROC | 0.8553 | **0.5694** |
| precision at a frozen threshold (562 chars) | **83.1%** | **16.0%** |
| fraction flagged correct | 30.1% | 29.7% |

Threshold frozen at the median `</think>` position of *correct baseline* rollouts,
so the detector is not refit to the attack. (Withdrawn figures: 0.5876 → 0.2756,
736 → 620 chars, AUROC 0.8899 → 0.6722, precision 92.5% → 41.9% at 573 chars.
Same story, different population; these come out of a script.)

Three things this establishes, all measured:

1. **The policy did move along the surface axis.** Median reasoning length fell
   773 → 625. The shift the paper guessed at from 30 rollouts is real and
   quantified.
2. **A purely-surface detector breaks too, and mostly in precision.** It goes on
   flagging the same share of rollouts (30.1% → 29.7%) while the share of those
   that are actually correct collapses **83.1% → 16.0%**. Ranking degrades less
   (0.855 → 0.569). That precision collapse is far steeper than the withdrawn
   figures suggested, and it is the sharpest version of the section's point.
3. **The probe breaks worse than the surface detector does.** Probe score *rises*
   while accuracy falls; the surface detector stays above chance. So surface
   exploitation is part of the mechanism and not all of it.

**The defensible sentence**, replacing lines 97 and 179:

> The attacked policy moved toward the surface signature the probe partly rides on
> — median reasoning length fell from 773 to 625 characters — and a detector
> reading that surface alone lost almost all of its precision, from 83% to 16% at
> a frozen threshold, while flagging the same fraction of rollouts. But the probe
> reads more than the surface: it retains AUROC 0.939 with length held fixed, and
> it fails worse than the surface detector does. Surface exploitation is part of
> this collapse, not the whole of it.

This is also a **method worth stating in its own right**: a free, known-surface
detector run against the attacked policy is a cheap instrument for decomposing how
much of a monitor's failure is confound occupation. It costs nothing and it is not
in the prior work.

**A caution on §4 the judge is right about.** *"Less gradient access is better"*
(line 157) has a degenerate optimum at zero, and the paper's own numbers point
there: read-only is +13.05 pp, multiplicative shaping is +2.84 pp, top-k is +1.5 pp
single-seed. §4 does not currently demonstrate any training-time use that beats not
using the probe in training at all. Either state that plainly as the finding, or
drop the ladder framing and keep §4 as "here is the invariant that makes a
training-time use safe, and here is how little it buys."

Also unaddressed: the 1.5B replication gives best-of-16 **+8.6 pp** against +12–13 pp
at 0.5B. The effect shrinks with scale, which cuts against §5's frontier-relevance
argument (line 163). Say something about it before a reviewer does.

---

## G2. THE SURFACE BATTERY — this should be the paper's centrepiece

**REGENERATED 2026-08-08.** Until now nothing in the repository computed this
section. `n_backtrack` and `has_therefore` appeared in no source file, so the table
below could not be reproduced, checked, or corrected — which is precisely the
standard `followup/PAPER_NOTES.md` applies to itself ("computed once in an unsaved
session and is not reproducible from any artefact, so it has been removed rather
than quoted"). It is now computed by **`extension/probe/surface_battery.py`**, and
every number here comes out of that script.

**The central claim survives.** The magnitudes have all moved, because the original
run's population is unrecoverable; these are clean-406, `last_block`, held out by
prompt, verifier = `evaluation.countdown`.

Seven hand-coded surface features, logistic regression, frozen on baseline, no
forward pass anywhere: `len_think`, `n_equals`, `n_lines`, `n_backtrack`, `n_ops`,
`has_therefore`, `ans_len`. (`n_backtrack` is defined in the script as the count of
ten self-correction markers; the lost original definition is not recoverable, so
this is a fresh one stated in code.)

**It reproduces the probe's Goodhart signature.**

**CORRECTED 2026-08-12 — population mismatch inside the table.** The baseline
score was the frozen model's mean P(correct) on the **held-out half** (3,072 rows,
correctly, so it is not inflated by having been fit there), while the accuracy on
the line beneath it was `yb.mean()` over **all 6,496** rows. Two different
populations, one comparison. On the rows the score actually describes, baseline
accuracy is **0.5173**, not 0.4980 — so the frozen monitor was mildly
*under*-reading at baseline rather than sitting on top of the truth. Fixed in
`surface_battery.py` (`run_battery`), which now reports both and labels them:

| | baseline (C_outcome, held-out 3,072) | runA post-RL (6,496) |
|---|---|---|
| joint surface model, mean P(correct) | 0.5060 | **0.8317** (+0.326) |
| true accuracy | **0.5173** | **0.1301** (**−0.387**) |
| joint surface model AUROC | 0.8690 | 0.5613 (−0.308) |

*(Baseline accuracy over all 6,496 rows is 0.4980, and that is the figure §G's
length-detector table quotes — that table fits nothing, so it has no held-out
split and correctly uses the full population. The two are not in conflict; label
which is which whenever both appear.)*

A monitor with **no access to the model's internals at all** inflates by +0.33
while true accuracy falls by **−0.39**. That is the whole point of the section, it
holds, and the corrected population makes the gap slightly wider rather than
narrower. Under the `first_block` rule the same run gives 0.5422 → 0.8923 score
against **0.5605** → 0.2361 accuracy, so the result is not an artefact of the
label rule.

Per-feature, the shift under attack broadly tracks how much signal the feature
carried:

| feature | AUROC base | AUROC hacked | mean shift (SD) |
|---|---|---|---|
| n_equals | 0.1384 | 0.3804 | **−1.04** |
| n_ops | 0.1421 | 0.3845 | **−1.06** |
| n_lines | 0.1416 | 0.4092 | **−0.81** |
| len_think | 0.1447 | 0.4306 | **−0.57** |
| ans_len | 0.3972 | 0.3905 | −0.67 |
| n_backtrack | 0.4563 | 0.4869 | +0.04 |
| has_therefore | near-constant at baseline | 0.4997 | — (dropped) |

Note the sign, which the previous version of this table had backwards: these
features are *negatively* associated with correctness (AUROC ≈ 0.14 means longer,
more equation-dense traces are **wrong**), and the attacked policy moved them
**down**. The mechanism is the same either way — the policy travelled along the
axes that carried signal — but the direction has to be stated correctly.
Dose-response over the 6 usable features: Spearman ρ = **+0.771**, p = 0.072;
terciles 0.35 / 0.82 / 0.92 SD.

**Scaled up to 39 features** (adding punctuation/word counts and 24 reasoning-marker
word frequencies), keeping the **23** that are not near-constant — a rule now
applied in code (drop any feature whose modal baseline value covers ≥99% of
rollouts) rather than by eye, and it selects exactly the 23 the original reported.
The trend holds but is a moderate correlation, not a law:

| signal tercile at baseline (\|AUROC − 0.5\|) | mean \|shift\| under attack |
|---|---|
| low (n=7) | 0.76 SD |
| mid (n=9) | 0.88 SD |
| **high (n=7)** | **0.90 SD** |

Spearman ρ = **0.575**, p = 0.004 — the correct statistic here, since the claim is
monotone and ordinal. Pearson r = 0.196, p = 0.37, and is not robust: several
features have near-zero baseline variance, which makes a standardized shift
explode (`w_yes` reads +12.4 SD before filtering). **Do not report the Pearson.**

**The "roughly twice as far" phrasing does not survive.** The regenerated tercile
spread is 0.90 vs 0.76 SD — a factor of **1.2**, not 2. The ordering is monotone and
the rank correlation is solid (p = 0.004), so state it as ordering, not as a ratio:
*"features that carried correctness signal moved further than features that did not
(0.90 vs 0.76 SD, ρ = 0.58, p = 0.004),"* not *"in proportion to"* and not
*"roughly twice as far."*

Figure: `fig_dose_response.png`, **regenerated 2026-08-08** by
`surface_battery.py --fig`. The previous version was not merely stale, it argued
against its own caption and should not have shipped:

- it plotted all 39 features with no near-constant filter, so one point sat at
  |shift| ≈ 125,000 SD and every real point was flattened onto the axis;
- its title read `r = -0.14` — the **Pearson**, the statistic this section says
  not to report — under the heading "The optimizer moves what predicts";
- it drew a least-squares fit line, sloping *downward*, asserting a linear
  relationship in the opposite direction to the claim.

The replacement plots the 23 usable features, states the excluded count on the
figure, carries Spearman ρ, and draws the tercile means as the ordinal summary
instead of a fit line. The `ans_paren` outlier — low signal, 3.7 SD of movement —
is directly labelled rather than hidden, since it is what holds ρ down.

**What this licenses the paper to claim** — and it is stronger than anything
currently in it:

> The collapse is not concealment. A policy optimized against the probe moves the
> cheap surface features that carried correctness signal further than the ones that
> did not, and a seven-feature logistic regression with no forward pass reproduces
> the probe's failure signature — its score rising from 0.51 to 0.83 while true
> accuracy falls 37 points. Any monitor whose signal correlates with those features
> inherits the failure, whether or not it is the monitor being optimized against.

**The prescription this yields** (§C usefulness fix — a real Monday-morning rule):
before wiring any monitor into a gradient, measure how much of its AUROC survives
conditioning on cheap surface features. Here: 0.982 → 0.939 stratified on `</think>`
position (held out), so the probe carries real signal beyond the surface — and it
still collapsed, because the *policy's* accessible axes are the surface ones.
Read-only quality is necessary and not sufficient; the question is what the
optimizer can reach, not what the monitor can see.

---

## G3. Audit of the sections not previously checked — §1 and §4 are CLEAN

Two of the paper's tables turned out wrong (§2, §3), so §1 and §4 were audited
against the artifacts as well. Both hold.

**§1, the multi-answer drift table (lines 50–55), verifies exactly** on clean-406:

| cell | paper | measured |
|---|---|---|
| TT | 2,983 | **2,983** |
| T→F | 490 | **490** |
| F→T | 150 | **150** |
| FF | 1,835 | **1,835** |
| total | 5,458 | **5,458** |

All four cells to the digit.

**CORRECTED 2026-08-12 — the paper's "84%" is right and the correction printed
here was wrong.** This previously read that line 46's "84% have ≥2 `<answer>`
blocks" measures 86.9% and told you to fix the number. Acting on that would have
introduced an error. The four cells sum to **5,458** multi-block rollouts, and
5458 / **6,496** (all clean-406 rollouts) = **84.02%**. The 86.9% figure is
5458 / 6,306, i.e. against the `</think>`-cached subset — a different denominator,
and not the one the sentence is about. **Keep the number; state the denominator.**

**§4, probe-best-of-K (line 148), verifies to the decimal:**

| | paper | measured |
|---|---|---|
| first-block accuracy gain | +1.5 pp | **+1.52 pp** (0.5827 → 0.5979) |
| mean `<answer>` blocks/rollout | 1.83 → 0.91 | **1.825 → 0.909** |

**One new discrepancy, minor but worth pre-empting.** Line 146 says multiplicative
shaping produced "no rambling explosion." Measured, its mean `<answer>` blocks per
rollout goes **1.825 → 2.316, +27%** — the opposite direction from probe-best-of-K,
which halves them. The accuracy gain is real and the traces are not exploding, but
"no rambling explosion" is the wrong phrase for a 27% increase. Say that
multiplicative shaping buys accuracy *without* the block-count reduction that
best-of-K delivers, and note the two constructions trade differently.

### G3a. Line 57 RESOLVED — the claim is false as written, and the truth is better

Pipeline gated against all nine published numbers before any new quantity was
computed. Out-of-fold, 5-fold GroupKFold over prompts, class-balanced, `None`
interior blocks dropped (that rule is not a guess — it is the only one that
reproduces the published counts; coercing them to wrong gives 2335/1138/121/1864):

| cell | n / paper | probe(first) / paper | probe(last) / paper |
|---|---|---|---|
| TT | 2983 / 2983 | 0.884 / 0.874 | 0.833 / 0.823 |
| T→F | 490 / 490 | 0.864 / 0.856 | **0.154 / 0.154** |
| F→T | 150 / 150 | 0.153 / 0.156 | 0.603 / 0.580 |
| FF | 1835 / 1835 | 0.081 / 0.084 | 0.089 / 0.088 |

Diagonal AUROC **0.9224** vs the paper's 0.920. Gate passes.

**The test the paper never ran:**

| | value |
|---|---|
| T→F probe(last) | 0.1539 (n = 490) |
| FF probe(last) | 0.0891 (n = 1835) |
| difference | **+0.0647** |
| 95% CI, prompt-clustered, 10k resamples | **[+0.0229, +0.1046]** — excludes zero |

**Line 57 is false as written.** The two are distinguishable. This is the third
unbacked "statistically indistinguishable" in the paper (with §3's null band and
line 88); treat the phrase as a red flag wherever it appears.

**But the corrected finding is stronger than the claim it replaces.** The residue
sits **8.7% of the way** from the FF floor to the TT level. So: a small, real,
statistically detectable trace of the earlier correct answer survives at the
wrong-final commit — and ~91% of it is gone. Write that. "There is a measurable but
almost entirely erased residue" is a positive result with a number and a CI;
"indistinguishable from floor" was a null with neither, and it was wrong.

The §1 reading (line 59) does not change: this is nowhere near a preserved hidden
correct answer, and the concealment framing still fails at this scale. Only the
strength of the claim changes, from "no signal" to "9% of the signal."

**Still unverified — none.** *(Previously flagged here: line 57, now resolved
above.)* The earlier note read:

> Line 57 asserts
probe(last) = 0.154 on T→F drift is "statistically indistinguishable" from the
F→F floor of 0.088, with **no test reported**. The `<answer>`-opening activation
caches are present locally, so this is a CPU-only computation. Either run it and
report the interval, or soften the wording — an unbacked "statistically
indistinguishable" is exactly the phrase a reviewer stops on, and it is the same
error class as §3's null band.

---

## G4. NEW RESULT — the remediation line 179 proposes makes things WORSE

Line 179 ends the paper by proposing, as an open question, that probes trained on
de-stylized or contrastively-balanced rollouts might resist confound exploitation.
That question is now answered, and the answer is no.

**Arm A**: the same RL recipe as the published probe-as-reward run (100 steps,
batch 128, group 8, lr 1e-5, KL 1e-3, from C_SFT), with the reward probe replaced
by a surface-residualised one — `LR(h - s@B)` over 39 cheap textual features,
held-out AUROC 0.834 vs the raw probe's 0.978.

| | first-block accuracy, clean-406 |
|---|---|
| C_outcome (verifier RL) | 0.5306 |
| published probe-as-reward | **0.2361** |
| **Arm A, surface-residualised probe** | **0.1678** |

- **Arm A − published probe-RL = −6.83 pp**, 95% CI [−8.85, −4.79]
- Arm A − C_outcome = −36.28 pp, 95% CI [−39.36, −33.19]

Prompt-clustered paired bootstrap, 10k resamples. **Pre-registered before launch**
(`followup/experiments/fragility/residual_probe/PREREGISTRATION.md`), which
predicted the opposite direction. The prediction is falsified outside the interval.

**Pipeline validation:** this analysis puts the published probe-RL run at 0.2361
against the paper's own stated 0.236. Exact agreement, so the comparison is not an
artefact of a differently-computed baseline.

**The obvious confound is dead, in a useful direction.** The worry was that a
favourable Arm A result would be ambiguous — a weaker probe (0.834) might simply be
harder to Goodhart. But Arm A is weaker *and* collapsed harder, so probe weakness
cannot explain a result pointing this way. Arm A is interpretable largely on its own.

**What to claim, carefully.** The measured fact is that stripping cheap surface
features out of the probe's input did not slow the collapse and coincided with a
worse one. The paper should report exactly that. What it should NOT claim without
Arm B is a mechanism for *why* — "removing surface leaves a less-constrained
boundary with more room to exploit" is a hypothesis, not a finding.

**Status of the remaining arm.** Arm B (a reward reading surface features ONLY, no
activations) is not yet built. Its role has changed: it is no longer disambiguating
a favourable Arm A, it is the falsifier for §G2's surface-occupation account. If a
pure-surface reward does not collapse, §G2 should be retracted rather than defended.

**Cost, measured:** ~194 s/step on one H100 (~5.5 h, ~$25 per 100-step arm);
~$4 per 500-prompt × 8-rollout eval.

---

## G5. ARM B — §G2 is now a TESTED claim, and the failure mode is quotable

Arm B replaced the reward with a probe reading the **39 surface features only**,
never the activation (gated: identical scores for zero vs random `X`). Read-only
AUROC 0.9252 — a genuinely good correctness predictor. Same recipe, 100 steps.

| | first-block accuracy, clean-406 |
|---|---|
| C_outcome (verifier RL) | 0.5306 |
| published probe-as-reward | 0.2361 |
| Arm A, surface-residualised | 0.1678 |
| **Arm B, surface-only** | **0.0000** |

- Arm B − published probe-RL: **−23.61 pp**, 95% CI [−26.43, −20.89]
- Arm B − C_outcome: −53.06 pp, CI [−57.04, −48.97]

**Pre-registered prediction CONFIRMED.** §G2 survives its falsifier and is no
longer an observed correlation: a reward with *no access to the model's internals*
destroyed the task completely.

**The failure is not degenerate output — that is what makes it good.** 100% of
rollouts still emit `</think>`, 99.9% still emit an `<answer>` block. The block
simply contains reasoning prose where an equation belongs:

```
<answer>
Let me verify:
1. 17 - 10 = 7
2. 63 - 7 = 56
```

Mean rollout length **1098 → 2390** on clean-406. *(Corrected 2026-08-12: the
previous "1095 → 2394" paired a clean-406 baseline with an all-500 Arm B. Same
story, one population.)* The policy learned to fill the answer slot with more
correct-looking work, because `ans_len`, `ans_ops`, `n_equals` and `n_digits` all
reward precisely that. It produced the appearance of careful arithmetic and zero
arithmetic.

**VERIFIED 2026-08-12.** The arm rollouts were pulled off the Modal volume
(`evaluation/eval_results/arm{A,B}_*_step100.json`) and all four published
accuracies recompute exactly — 0.5306 / 0.2361 / 0.1678 / 0.0000 — with the
contrast CIs agreeing to the third digit (bootstrap seed). This is now a standing
check: `extension/probe/verify_residual_arms.py` prints the published value beside
each recomputed one and flags disagreement. Note one population fact the pack did
not previously state: **the arms were evaluated at 8 responses per prompt and the
references at 16**, so an arm contributes 3,248 rollouts against a reference's
6,496. The bootstrap pairs on prompts and uses per-prompt means, so the contrast
is well defined — but say it, because the raw counts differ.

This is the sharpest artefact in the project. Put a rollout in a box, verbatim.
The one the paper quotes is real and the full block is stronger than the excerpt;
for input `[29, 17, 63, 38]`, target `68`:

```
<answer>
Let me verify:
1. 17 - 10 = 7
2. 63 - 7 = 56
3. 29 * 38 = 1102
4. 56 - (29 * 38 / 38) = 68
</answer>
```

Step 1 uses `10`, not in the input set. Step 3 is correct and irrelevant. Step 4
reuses `38` and `29` and evaluates to 27 — while *asserting* 68, the target, in
the answer position. Quote all four lines, not two.

**What this licenses, and what it does not.** Licensed: occupying cheap textual
structure is *sufficient* to destroy both a monitor and the task — no internal
access required. NOT licensed: that this is *the* mechanism of the published
probe's collapse. Arm A cuts against the simple version — removing surface access
made things worse, not better. The honest statement is that surface occupation is
sufficient to produce this failure, and that the published probe's collapse
resembles it, without claiming identity.

---

## H. Line 24's novelty claim is false and must be rewritten

> *"almost none of it studies internal probes on the policy's own activations as
> the reward signal."*

Gupta & Jenner study exactly that. This is checkable in five minutes and it is the
single most damaging sentence in the paper — not because the error is large, but
because it is the kind that ends a review. The replacement claim, which is true and
is the paper's actual delta:

> In prior work the monitor enters as a **penalty**, so evasion is the argmax — the
> policy is paid to produce activations that read clean. Here the probe is a
> **positive reward with no evasion term anywhere in the objective**. The policy is
> not concealing anything; it is complying, and it finds a region where the probe's
> decision boundary was never constrained by training data. Monitor collapse under
> RL therefore does not require adversarial pressure, so auditing for adversarial
> pressure is not sufficient.

---

## I. What the paper claims after these edits

Section 3 is gone, so the mechanistic claim goes with it. What remains is
behavioural, and it is clean.

**CORRECTED 2026-08-12.** This section is the outline the prose gets drafted from,
and until now it still carried four figures its own §B2 and §G had already
withdrawn: "AUROC 0.98 with trace length held fixed" (§B2: **0.939**), "736 → 620"
(§G: **773 → 625**), and "92.5% → 41.9%" (§G: **83.1% → 16.0%**). Drafting from it
would have re-introduced the corrections as errors. Every figure below is now the
one its own section computes.

- A linear probe on one layer's `</think>` hidden state captures **89% of the
  headroom** available to any best-of-16 selector (88.8%, CI [78.9%, 96.5%]), and
  retains held-out AUROC **0.939** with `</think>` position stratified out, against
  **0.564** for a pure position detector under the same stratification. It is an
  excellent read-only selector, and it is not a length detector.
- Used as a **positive** RL reward — with no evasion incentive in the objective —
  probe score rises while true accuracy falls **31 pp** (0.550 → 0.236, first
  block).
- The same signal is therefore near-oracle in one direction and actively harmful
  in the other. Read-only monitor quality does not transfer to gradient-facing use,
  and AUROC does not predict which one you are in.
- Constructions that keep the probe useful without putting it in the gradient:
  best-of-K selection, multiplicative shaping (+2.8 pp, single run), abstention.

- Part of that collapse is the policy occupying a surface correlate the probe
  partly rides on (§G): median `</think>` position falls **773 → 625** characters,
  and a known-surface detector's precision at a frozen threshold falls
  **83.1% → 16.0%** while it goes on flagging the same share of rollouts
  (30.1% → 29.7%). But only part — the probe fails worse than the surface
  detector, which stays above chance, and the probe is not a surface detector at
  rest.

The full mechanism is deliberately left open and is what the follow-up work
addresses. §G is a bound on one component, not an explanation. Do not gesture at
more than that here.

**Ordering note.** The strongest version of this paper puts §B1 (89% of oracle
headroom, with the free baseline in the table) and §G (the surface decomposition)
adjacent, because together they make one argument that neither makes alone: the
same probe is near-oracle read-only *and* the surface it partly rides on is exactly
what the optimizer occupies. That is a sharper story than "the probe was a template
detector," and it is what the data supports.
