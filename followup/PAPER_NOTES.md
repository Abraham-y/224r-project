# PAPER_NOTES

One paragraph per figure-worthy result, the day it is produced: claim, evidence,
caveats. Append-only.

Target: NeurIPS 2026 interp/safety workshop (fallback), ICLR 2027 main (stretch).

**Metric conventions.** AUROC is recorded in two variants: unbalanced (all val
rows) and balanced (positive/negative subsampled to equal size, seed-dependent).
They are different numbers and must not be mixed inside one table. Every table
below states which variant it uses. Metrics with no balanced variant (`sep`,
`score_corr_frozen_retrained`, Spearman correlations) are computed on the val
split; `verifier_acc` and block counts are computed on all harvested rollouts.
The control null band exists in both variants:
**[−0.021, +0.041] unbalanced, [−0.024, +0.041] balanced.** Compare a gap only
against the band of its own variant.

Balanced AUROC depends on which rows the subsample draws, and re-analysing the
same activations moves it by up to ~0.015 (measured by re-running the control
ladder's seed-0 analysis: every unbalanced value was bit-identical, every balanced
one moved). The balanced band's lower edge moves with it — a second pass gives
−0.021 rather than −0.024 — but **its upper edge is +0.041 in every pass**, and the
upper edge is the one every evasion claim is tested against. The unbalanced band
is exactly reproducible. Quote balanced figures to three decimals only when the
comparison tolerates that much jitter.

---

## 2026-08-08 — AUDIT PASS: three runA tables regenerated, four code bugs fixed

**What was wrong.** The metric store held two generations of runA Phase-1 rows
(2026-08-01 pre-relabel, 2026-08-02 post-relabel). `metrics_io.latest()` was
serving the corrected ones correctly the whole time — but three tables in this
file were transcribed from the earlier generation and never revisited, while the
2026-08-01 and 2026-08-02 entries below used the corrected values. So this file
carried two mutually inconsistent 3-seed tables, neither of which matched the
store. All three are now regenerated from parquet and marked.

**What changed in the conclusions:**

| claim | was | is |
|---|---|---|
| gap exits the control band at | step 70 (one table) / step 50 (the other) | **step 50**, back at the edge at 60, clear from 70 |
| step 90 | "an outlier on every measure", gap 0.055 | not an outlier; gap 0.108, between step 80 and 99 |
| early-ladder gap | small and positive | **negative** at steps 10–30 (−0.005, −0.014, −0.032), well inside the band |
| frozen AUROC through step 40 | flat | flat-to-rising, **peaks at step 30** (0.814) |
| `sep` rises after step 40 | two transitions | **one** (70→80) |
| self-consistency cosine range | 0.052–0.184 | **−0.006 to 0.240** |

The headline ordering result — reward hacking runs ~40 steps before the monitor's
discrimination moves at all — is unaffected, and the step-90 correction actually
removes a caveat rather than adding one.

**Code fixed in the same pass** (details in the sub-entry under the transfer-matrix
retraction): the residualised probe was scored on raw activations and its
residualiser was fit on train+val; the ensemble probe's `layer = -1` sentinel was
routed through a truthiness check; derived CSVs were rewritten in place, which had
already cost the on-policy transfer-matrix rows; and `wandb` log mirroring would
have raised `TypeError` on any positional call. None of these touched a number in
this file except the transfer matrix.

**The lesson worth keeping.** Every one of these was invisible to
`verify_invariants.py`, which passes 18/18 and did so throughout — it checks that
the *data* is consistent, not that the *prose* still matches the data. A
regenerate-tables-from-parquet step belongs in the workflow, not a transcription.

---

## 2026-08-02 — RETRACTED: the transfer-matrix conclusion below does not follow

> The entry that follows claims "no probe resists the attack" and uses it to
> cancel the Phase-3 sweep. Adversarial review refuted it. **Do not quote it.**
> The matrix itself is correctly wired — no leakage, right step, right layers for
> every real probe — but the inference is unsupported and two headline numbers
> are contaminated.
>
> **1. The zoo does not span "monitors".** It contains only frozen directions fit
> at t=0. On the SAME step-99 rows where all zoo probes read 0.46–0.57:
>
> | monitor | balanced AUROC @99 |
> |---|---|
> | best zoo probe (L16_C0.001) | 0.573 |
> | **retrained linear probe (fit on step-99 train rows)** | **0.742** |
> | **`resp_chars` — response length, no activations at all** | **0.613** |
> | PC3 of step-99 activations | 0.687 |
>
> Response length alone beats every probe in the zoo. What collapsed is
> *"directions fit before the attack"*, not *"monitors"*. The correctness signal
> is still there and still linearly decodable — which is the fixed-text result
> arriving by another route, and it makes "probe choice buys no robustness" a
> claim the data does not support.
>
> **2. `spread` was read where it has least power.** AUROC is squeezed against
> the 0.5 floor, so max−min compresses mechanically when everything is near
> chance. Normalised by distance from chance, on-policy@99 has the *largest*
> relative dispersion (2.52) not the smallest (fixed-text@99: 1.14). At step 50 —
> before the floor — the on-policy spread is 0.222 and the rank correlation
> between on-policy and fixed-text robustness is **rho = 0.758, p = 0.002**.
> Probe identity predicts robustness strongly; the measurement was taken after
> that information had been floored out.
>
> **3. The gate's own criterion was not met.** It requires "spread ~0". Prompt-level
> bootstrap on runA@99 gives spread 0.118, 95% CI [0.076, 0.197] — excludes zero.
> The STOP branch was taken by eyeballing "all near 0.5".
>
> **4. runB has no baseline and is mostly not the attack.** The zoo scores
> 0.545–0.657 on runB@**0**, before any RL, because it was fit on runA's init
> (C_outcome) while runB starts from C_SFT. Roughly half the quoted "collapse" is
> cross-init transfer failure. The step-0 row was never computed.
>
> **5. There is no diagonal.** runA was attacked against the original paper probe,
> not any zoo member, so every cell is off-diagonal. And frozen-monitor AUROC is
> not the sweep's dependent variable anyway: the sweep measures Δ true-accuracy
> when probe B *is the reward*. A probe that fails as a passive monitor can still
> be hard to hack as a target. Nothing here estimates that.
>
> **6. Two numbers are wrong.** `L16_resid_template` is scored on raw activations
> though it was fit on template-residualised ones (the residualisation is not part
> of the pickle) — 0.749 reported where the like-for-like figure is 0.606. And
> "fifteen probes" is **fourteen**: the ensemble records `layer = -1` (its spec has
> `layers`, not `layer`) and is dropped by the layer check, so the ensemble arm of
> objective-variation was never tested.
>
> **Revised decision.** Do not spend the $390 yet — but H3 is NOT closed. Three
> free re-analyses decide it: re-run the matrix at steps 30–60 where probes are
> off the floor and gate on the fixed-text/on-policy rank correlation instead of
> raw spread; add retrained-linear and `resp_chars` as reference columns to every
> row; add runB's step-0 baseline and report Δ from each row's own baseline.

### 2026-08-08 — items 4 and 6 of that retraction are now FIXED IN CODE

Both were live bugs, not just reporting errors, and both are repaired:

- **Item 6a, the residualised probe.** `build_probe_zoo.py` now persists the
  fitted residualiser next to the pickle (`{probe_id}.residualiser.npz`) and
  `transfer_matrix.py` replays it before scoring, raising rather than falling
  back to raw activations. Separately, the residualiser used to be fit on
  train+val; it is now fit on train only, which is what `confounds.residualise`
  had documented all along. On the corrected path `L16_resid_template` reads
  **0.677** at runA@0 (not 0.749), and its t=0 val AUROC rises 0.605 → **0.675**
  once the leak is removed.
- **Item 6b, the ensemble.** `prehoc_features.csv` now records `single_layer`
  and `layers` explicitly, and `transfer_matrix.py` no longer routes the -1
  sentinel through a truthiness check. The ensemble is still uncomputable on a
  single-layer snapshot — that part was right — but it is now reported as
  "multi-layer probe; cannot be evaluated here" instead of silently becoming a
  NaN that looked like a missing cache.
- **Item 4, runB's missing baseline.** Computed. The zoo scores **0.545–0.657**
  on runB@0 and **0.487–0.613** on runB@99, so the drop from runB's own baseline
  is roughly 0.05, against the ~0.15 the uncorrected reading implied. The
  retraction's suspicion is confirmed: most of runB's apparent collapse is
  cross-init transfer failure, not the attack.

The matrix now covers steps 0/50/99 on runA (both arms) and 0/99 on runB, and
`results/fragility/transfer_matrix.csv` holds all eight rows. It previously held
only the three fixed-text rows, because the CSV was rewritten in place by each
run; derived CSVs now keep a timestamped copy in a `.archive/` sibling.

## 2026-08-02 — TRANSFER MATRIX: no probe resists the attack, and H3 is answered for $0

**Claim.** Fifteen probes fit at t=0 — varying layer (12/14/16/18/20),
regularisation (C = 0.001 / 0.1 / 10), rank (PCA 8 / 32 / 128), objective
(logistic, mean-difference, MLP, ensemble), and two H0-motivated variants
(template-adversarial, template-residualised) — **all collapse together** on
runA's post-RL rollouts. At step 99 every one scores 0.460–0.573 (spread 0.113,
all near chance). The same uniform collapse holds on runB (0.487–0.615).

**Probe choice buys no robustness against this attack.** That is the H3 question,
answered off already-harvested activations at zero marginal cost, and it says the
~$390 probe-zoo RL sweep should NOT be bought.

**But the fixed-text arm shows they are not damaged.** The identical probes on the
identical weights, evaluated on step-0 text, mostly still work:

| probe | on-policy @99 | fixed-text @99 |
|---|---|---|
| L16_pca8 | 0.555 | **0.804** |
| L16_C0.001 | 0.573 | **0.803** |
| L16_pca32 | 0.550 | **0.798** |
| L16_adv_template | 0.548 | **0.787** |
| L16_C0.1 | 0.526 | 0.735 |
| L16_C10 | 0.540 | 0.544 |
| L20_C0.1 | 0.460 | 0.755 |

Spread at step 99: **0.113 on-policy, 0.260 fixed-text.** Fifteen structurally
different probes fail *identically* on new text and *differently* on old text.
That is what a distribution-shift failure looks like and what a
representation-damage failure does not: if the weights had destroyed the
correctness code, the fixed-text column would collapse too.

**A real but secondary H3 signal, in the representational component only.**
Within the 16% of failure that IS representational, probe geometry matters a lot:
on fixed text, the weakly-regularised L16_C10 loses 0.203 (0.747 → 0.544) while
the heavily-regularised L16_C0.001 loses 0.008 and the rank-8 PCA probe loses
0.002. Regularisation and rank restriction confer near-immunity to the weight
change — and nothing at all against the text change. There is also a *timing*
effect on-policy: at step 50, L16_C0.001 still reads 0.762 and pca8 0.752 while
L16_C0.1 has fallen to 0.587 and L16_C10 to 0.540. Robust probes buy **delay**,
not immunity.

**Caveats.** The zoo was refit on runA's own step 0 (C_outcome at temp 1.0)
specifically so that sampling-regime shift is not charged to the attack; the
earlier eval-regime zoo is kept but is not comparable to these ladders. The
ensemble probe is absent from every row (it expects concatenated layers and the
matrix evaluates one layer at a time) — recorded as NaN, not imputed. Balanced
AUROC, seed 0, one trajectory.

**Decision.** Phase 3's RL sweep is not worth its cost: the thing it would
measure (which probe survives) has a uniform negative answer already. What
remains worth doing in Phase 3 is the pre-hoc feature correlation against the
*fixed-text* degradation, which is free and which the numbers above suggest is
strongly predicted by regularisation strength and effective rank.

---

## 2026-08-02 — THE MISSING CELL: pre-registered prediction confirmed; the attribution is now identified

**The 2×2 is complete.** L16, `last_block`, val split, seed 0:

| | step-0 text | step-99 text |
|---|---|---|
| **step-0 weights** | 0.796 | **0.604** ← measured today |
| **step-99 weights** | 0.759 | 0.578 |

All four cells are val-split `auroc_frozen` from the parquet store, L16, seed 0.

**The prediction was pre-committed before the run** (see the retraction below,
written before the job launched):

| | predicted AUROC(w0,t99) | predicted mean probe | observed |
|---|---|---|---|
| additive / distribution shift | ~0.615 | ~0.999 | — |
| interaction / co-adaptation | 0.75–0.80 | ~0.5 | — |
| **actual** | | | **0.604, mean probe 0.9945** |

Essentially a bullseye on the additive prediction and nowhere near the
interaction alternative. **The unoptimised step-0 model scores step-99's text at
mean probe 0.9945, with 99.0% of rollouts above 0.95.** That text saturates the
monitor without ever passing through the weights that produced it.

**The identification gap is closed.** Both paths through the square now agree:

- text effect, weights held at step 0: (0.796 − 0.604)/(0.796 − 0.578) = **88.1%**
- representational effect, text held at step 0: 0.037/0.218 = **17.0%**
- residual interaction: **−5.1%**

Before this cell the behavioural share was free in **[0%, 136%]**; it is now
pinned near 90% with a small negative interaction. The dose-response across text
sets is monotone and confirms it:

| text | AUROC(w0, t) | AUROC(wt, t) | mean probe under w0 | share via this cell |
|---|---|---|---|---|
| t50 | 0.714 | 0.722 | 0.859 | 111.5% |
| t70 | 0.560 | 0.575 | 0.972 | 106.9% |
| t99 | 0.604 | 0.578 | 0.994 | **88.1%** |

**CORRECTED 2026-08-02.** An earlier version of this table read 0.693 / 0.572 /
0.600 with shares 139% / 101% / 90%. Those three cells came from
`sanity_check_probe`, which computes AUROC over **all rows**, while the cells
they were subtracted from came from the Phase-1 store, which uses the **val
split only** — two different estimators in one table, matching to 4 dp against
each source. All six cells are now val-split figures read from the parquet store
(`run_phase1.py --run_id phase0_harvest_runA__frozen_w0 --arm frozen_weights`).
The t99 headline survives (90% -> 88.1%); the t50 claim does not.

At t50 and t70 the share exceeds 100%: old weights do *worse* on the new text
than the co-adapted weights do, i.e. the weight update partially **compensates**
for the text shift. On the corrected figures that recovery is **0.0084** AUROC at
t50, not the 0.029 first reported — a 3.5x overstatement produced entirely by the
estimator mix. The interaction is real and has the sign the adversarial review
predicted from the logit analysis, but it is small.

**What this does NOT rescue.** Two of the three refutations below still stand:

1. **"The representation barely moved" remains false on threshold metrics.** On
   fixed text the weight change still moves TPR 0.837 → 0.424 and flips 15.9% of
   rollout pairs. AUROC is invariant to monotone recalibration by construction,
   so a 0.037 AUROC drop is consistent with a large operating-point shift. The
   new cell identifies the *AUROC* decomposition; it says nothing about that.
2. **The decomposition is still ruler-dependent.** 17% on AUROC, 46% on Youden's
   J, −266% on TPR. This cell pins the AUROC-scale answer; it does not make AUROC
   the right scale. Any write-up must name the ruler.

Resolved: refutation 3 (no discriminating power). The cross-cell design *does*
discriminate — the interaction alternative predicted 0.75–0.80 and the
measurement returned 0.600.

**Standing caveats.** One seed, one trajectory. No fixed-text control arm on the
vanilla ladder or runB, so the 0.037 still has no benign-RL null.

**Cost.** ~$1.50, three forward passes. It was the single highest-value
measurement in the project and it sat unrun because the original design
specified only the on-policy arm.

---

## 2026-08-02 — RETRACTED IN PART: "the representation barely moved" is false, and the 83/17 split is unidentified

> Adversarial review of the entry below. **Every number reproduces exactly** —
> 0.7957 at (w0,t0), 0.7587 at (w99,t0), 0.5781 at (w99,t99), CKA 0.992635. This
> is not a reporting error. It is a reasoning error, in three places.
>
> ### What survives
> Six alternative explanations were tested and eliminated: class imbalance
> (rebalancing on-policy@99 to step-0's positive rate moves AUROC by 0.002),
> length/composition shift (conditional AUROC on n_blocks, resp_chars, tok_idx,
> template_score, and within-prompt all fail to recover anything), the label rule
> (`first_block` gives 18%/82%, same split), score saturation, the `</think>`
> selection filter, and "fixed-text stability is forced because the probe was fit
> there" (within-prompt conditional AUROC strips prompt difficulty and the ratio
> survives). **The behavioural component is real and large.** That much holds.
>
> ### REFUTED 1 — "the representation barely moved"
> On the fixed-text arm's OWN data, identical text, weights only, step 0 → 99:
>
> | quantity | change |
> |---|---|
> | mean frozen-probe logit | **−1.17** (against a correct-vs-wrong gap of only 2.42) |
> | per-row \|Δlogit\| | sd 1.215; **44.3% of rollouts move by more than half the entire signal** |
> | Spearman(z₀, z₉₉) | 0.857 — **15.9% of rollout pairs have their ordering flipped by weights alone** |
> | TPR at the deployed threshold | **0.837 → 0.424** |
> | flag rate | 0.515 → 0.251 |
>
> **The weight change halves the monitor's sensitivity on text it has already
> seen.** AUROC reports 0.037 because the change is a near-monotone
> recalibration, and AUROC is invariant to monotone recalibration *by
> construction*. `sep` is invariant to it too. The two statistics offered as
> evidence are precisely the two that are blind to what happened.
>
> ### REFUTED 2 — the 83/17 split is a property of the ruler
> The path decomposition is additive by construction on any scale, so the answer
> is whatever metric you pick:
>
> | ruler | representational share |
> |---|---|
> | AUROC | +17% |
> | sep | +18% |
> | **Youden's J at the deployed threshold** | **+46%** |
> | probe logit (the RL reward's own scale) | −18% |
> | flag rate | −55% |
> | **TPR** | **−266%** |
>
> Range −266% to +46%. The negative shares are the important part: the weight
> effect and the text effect have **opposite signs** on every operational metric —
> weights push the probe score *down* while RL was rewarding it *up*. That is
> prima facie a weights×text **interaction**, which an additive decomposition
> assumes away. "17%" without "on the AUROC scale" is not defensible.
>
> ### REFUTED 3 — the arm has no power against the hypotheses it claims to refute
> RL only updates on sequences the policy sampled. A representational change
> concentrated on the *on-policy* distribution — which is what H1 and H2 would
> both look like under RL — produces **exactly** the observed fixed-text
> stability. The arm's stability is predicted by all three hypotheses, so it
> discriminates none of them. It also has no control: there is no fixed-text arm
> on the vanilla ladder or runB, so 0.037 has no null. (The band quoted below is
> an on-policy evasion-gap band from a different ladder at a different sampling
> regime — not a valid reference.)
>
> And 89% of the fixed-text logit drift (−1.06 of −1.17) happens between steps 0
> and 10, while on-policy AUROC is still 0.794 and accuracy is intact. Whatever
> the weight effect is, it mostly *precedes* the Goodhart.
>
> ### The missing cell decides everything
> Behavioural share via the other path is (A−C)/(A−D). With C = AUROC(w0, t99)
> unmeasured and free in [0.5, 0.796], the share spans **[0%, 136%]** — every
> possible conclusion. The additive model makes a sharp falsifiable prediction:
> **AUROC(w0,t99) ≈ 0.615**, with the *unoptimised* step-0 model already scoring
> step-99 text at mean probe ≈ 0.999. The interaction alternative predicts ≈
> 0.75–0.80 and mean score near 0.5. One forward pass settles it.
>
> ### What the original Phase-1 result still says
> Untouched by any of this: on-policy at step 99 the evasion gap is **+0.164**
> (frozen 0.578, retrained linear 0.742, MLP 0.685), four times the null band,
> with frozen-vs-retrained score correlation collapsing 0.730 → 0.170. And for a
> policy whose entire action space IS text, "the policy found inputs on which the
> monitor fails" *is* evasion. Showing the mechanism is not weight-mediated
> **narrows** H1; it does not refute it.
>
> **Required before any of this is written up:** (1) run (w0, t99); (2) a
> fixed-text control arm on the vanilla ladder and runB; (3) report the
> decomposition on a threshold-respecting metric alongside AUROC, naming the
> ruler; (4) the CKA sentence stays deleted.

## 2026-08-01 — FIXED-TEXT ARM: the representation barely moved. The collapse is distribution shift. [SEE RETRACTION ABOVE]

**Claim.** Holding the text fixed and varying only the weights, the frozen probe
loses **0.037** AUROC across the entire ladder (0.796 → 0.759) and a retrained
probe loses **0.007** (0.805 → 0.798). On the policy's own rollouts the same
frozen probe loses **0.218** (0.796 → 0.578) — roughly a **6× difference**.

So the correctness-reading machinery is essentially intact at step 99. The probe
fails because the policy moved to a region of *text* space where its decision
rule does not apply — not because the information moved off the direction (H1) or
degraded (H2). Both of those are representational claims, and at the
representational level almost nothing happened.

**Evidence.** Same 3216 rollouts (step-0 text, labels constant by construction)
forward-passed through all 11 checkpoints. L16, seed 0, unbalanced:

| step | frozen (on-policy) | frozen (fixed-text) | retrained (on) | retrained (fixed) | sep (on) | sep (fixed) | CKA |
|---|---|---|---|---|---|---|---|
| 0 | 0.796 | 0.796 | 0.805 | 0.805 | 1.164 | 1.164 | 1.000 |
| 40 | 0.803 | 0.786 | 0.818 | 0.800 | 1.142 | 1.141 | 0.995 |
| 50 | 0.722 | 0.785 | 0.786 | 0.799 | 0.682 | 1.128 | 0.994 |
| 70 | 0.575 | 0.777 | 0.657 | 0.797 | 0.237 | 1.072 | 0.993 |
| 99 | 0.578 | 0.759 | 0.742 | 0.798 | 0.213 | 0.994 | 0.993 |

Step 0 agrees to **0.0000** between arms, which it must: the fixed text at step 0
IS that checkpoint's own text through its own weights, so the activations are
bit-identical. That identity is now asserted by `verify_invariants.py`. An earlier
version of this table did NOT satisfy it (0.787 vs 0.796) because the on-policy
metrics predated a prompt-index relabel and the two arms were therefore split on
different val sets. The numbers above are recomputed on a common split.

**The decomposition, with its uncertainty.** Prompt-clustered bootstrap (2000
draws; the 1003 val rows come from only 126 prompts, so clustering matters):

| quantity | point | 95% CI |
|---|---|---|
| fixed-text drop 0→99 | 0.037 | [0.017, 0.059] |
| on-policy drop 0→99 | 0.218 | [0.152, 0.284] |
| **representational share** | **17%** | **[7%, 29%]** |

Both drops are individually significant, so the DIRECTION is solid: weight drift
explains a small minority of the failure. The two-significant-figure "83/17"
split is not supported — the interval spans 4×. State it as "5–15× smaller", not
as a percentage point estimate.

`sep` tells the same story: −0.951 on-policy against −0.170 fixed-text.

**RETRACTED: the CKA evidence.** Earlier versions of this entry cited "layer-16
CKA never falls below 0.993" as evidence the representation is unchanged. CKA
cannot support that claim. It is variance-weighted and dominated by top principal
components, while the probe direction carries **2.2e-4** of L16 variance.
Demonstrated on the real step-0 activations:

| perturbation | CKA vs X₀ | frozen AUROC |
|---|---|---|
| project the probe direction completely OUT | 0.9999999 | 0.796 → **0.500** |
| INVERT the probe direction | 1.0000000 | 0.796 → **0.500** |
| what was actually observed at step 99 | 0.9926 | 0.796 → 0.759 |

Totally destroying the monitor yields *higher* CKA than what we observed. The
statistic carries zero information about the quantity it was cited for. (It was
also misquoted: the minimum is 0.99264, i.e. below 0.993.) The probe-relevant
evidence is `sep`, which is computed along the actual probe direction and does
move: 1.164 → 0.994.

**The evasion gap stays inside the control null band at every fixed-text step**
(0.004 to 0.040, band [−0.021, +0.041]), while the on-policy gap reaches 0.168.
Whatever "evasion" means here, it is not visible when the text is held constant.

**This reframes the two-phase result rather than replacing it.** The on-policy
two-phase structure is real and reproducible — but both phases are now
attributable mostly to behaviour. The steps 50–60 "general degradation" is not
the representation degrading; it is the text distribution moving far enough that
even a freshly-fit probe on the new distribution does worse (retrained on-policy
falls 0.088 while retrained fixed-text falls 0.007).

**Caveats — important.** The fixed-text arm evaluates late checkpoints on text
they would no longer produce, so "the representation is unchanged" means
precisely "the representation OF THIS TEXT is unchanged". It cannot speak to how
step-99 represents step-99's own outputs — by construction nothing can, since
that text does not exist at step 0. The residual drift is small but real and
roughly monotone (frozen −0.037, sep −0.170, CKA 1.000 → 0.993), so the honest
statement is "an order of magnitude smaller than the behavioural effect", not
"zero". Single seed on this arm so far, and runA only.

**Consequence for the paper.** The headline is no longer "evasion vs corruption".
It is: *a monitor can fail completely while the thing it monitors is unchanged,
because optimisation moves the input distribution out from under it.* That is a
sharper and more general safety claim than either H1 or H2, and the fixed-text
arm is what licenses it. Running only the on-policy arm — which is what the
original plan specified — would have produced a confident and wrong mechanism.

---

## 2026-07-31 — Post-audit regeneration: all metrics recomputed with corrected code

Everything below this entry was recomputed after a four-way audit found bugs in
the metric library. Two results changed materially; the rest tightened.

**1. The residualisation finding is now properly controlled, and it survives at
roughly half its former size.** The earlier "placebo costs 0.001" comparison was
invalid — the placebo matched the template on column *count*, not on removed
*variance* (2.0% vs 31.4%), and three of its seven columns are constant. The
projection was also fit on train+val, leaking. With the fit restricted to train,
constant columns dropped, and a control matched on **removed variance**:

| condition | variance removed | AUROC |
|---|---|---|
| baseline linear probe | — | 0.903 |
| **residualise on template indicators** (rank 6) | **31.4%** | **0.718** |
| **variance-matched PC control** | **31.6%** | **0.898** |
| column-count-matched placebo (the old, invalid control) | 2.0% | 0.902 |
| random projections of the activations | — | 0.664 |

**Template-specific effect: +0.180** (`resid_template_minus_matched`). Removing
the template-predictable subspace costs 0.185 AUROC; removing an arbitrary
subspace of the *same destructive power* costs 0.005. The finding stands — it is
not a generic consequence of deleting activation variance — but at 0.180, not the
0.307 originally reported. Status: **established**, superseding the "OPEN" note in
the entry below.

**2. Most of the apparent seed noise was balanced-subsample draw noise.**
`auroc_balanced` now averages 25 draws instead of 1. Across runA's 11 checkpoints
the 3-seed standard deviations fell from 0.001–0.036 (median 0.019) to
0.000–0.006 (median 0.003), roughly a 6× reduction, with the means essentially
unchanged. The wiggle in the earlier curves was measurement artefact, not
run-to-run variability.

**3. The phase boundary moves earlier and gets cleaner.**

> **REGENERATED 2026-08-08 from the metric store.** The table originally printed
> here was computed BEFORE the prompt-index relabel (see the 2026-08-01 entry),
> so its by-prompt val split differed from the one every current number uses. It
> read frozen 0.788 … 0.542 and a step-99 gap of 0.179. Those are the
> pre-relabel figures and are withdrawn. `metrics_io.latest()` had been serving
> the corrected values all along; only this table was stale.

runA L16, balanced, mean ± sd over 3 seeds, against the control band
[−0.021, +0.041]:

| step | frozen | retrained | gap | vs band |
|---|---|---|---|---|
| 0 | 0.797 ± 0.000 | 0.807 ± 0.001 | 0.010 ± 0.001 | inside |
| 10 | 0.794 ± 0.005 | 0.789 ± 0.002 | −0.005 ± 0.004 | inside |
| 20 | 0.800 ± 0.001 | 0.786 ± 0.002 | −0.014 ± 0.002 | inside |
| 30 | 0.814 ± 0.001 | 0.782 ± 0.001 | −0.032 ± 0.002 | inside |
| 40 | 0.803 ± 0.003 | 0.818 ± 0.004 | 0.014 ± 0.003 | inside |
| 50 | 0.723 ± 0.000 | 0.786 ± 0.001 | **0.064 ± 0.001** | **outside** |
| 60 | 0.652 ± 0.006 | 0.693 ± 0.002 | 0.041 ± 0.004 | at the edge |
| 70 | 0.575 ± 0.002 | 0.657 ± 0.004 | 0.082 ± 0.004 | outside |
| 80 | 0.612 ± 0.005 | 0.741 ± 0.003 | 0.130 ± 0.003 | outside |
| 90 | 0.582 ± 0.005 | 0.690 ± 0.008 | 0.108 ± 0.004 | outside |
| 99 | 0.583 ± 0.002 | 0.742 ± 0.007 | 0.158 ± 0.005 | outside |

The gap exits the band **at step 50** (0.064 ± 0.001), dips back to the band's
top edge at 60 (0.041 ± 0.004), and is clear of it from 70 onward. Frozen AUROC
is flat-to-rising through step 40 — it *peaks* at step 30 (0.814) — and falls
only from 50.

**Two conclusions the pre-relabel table supported and this one does not:**

- **Step 90 is not an outlier.** The old figures put its gap at 0.070 against
  0.150 and 0.179 either side, which the entry below explains with a
  single-batch KL spike. On the corrected numbers the step-90 gap is 0.108,
  sitting between step 80 (0.130) and step 99 (0.158). The KL spike is still in
  the training log; it no longer needs a metric anomaly to explain.
- **The gap is NEGATIVE through steps 10–30**, not merely small (−0.005, −0.014,
  −0.032): a probe retrained on those checkpoints does slightly *worse* than the
  frozen one. That is well inside the control band, so it is not a finding — but
  "the gap stays small and positive early" is not what the data says.

**The headline ordering claim is unaffected.** Through step 40 the gap sits
inside the band at every checkpoint with sd ≤ 0.004, while the probe score
climbs 0.460 → 0.677 and blocks per rollout go 5.0 → 13.6. Reward hacking runs
for ~40 steps before the monitor's discrimination degrades at all.

**Also confirmed unchanged:** the control null band, [−0.021, +0.041] unbalanced
and [−0.021, +0.041] balanced after 25-draw averaging (the two now agree, where a
single draw put the balanced lower edge at −0.024).

---

## 2026-07-30 — Control ladder: the evasion-gap null band is measured, not assumed

**Claim.** Under vanilla verifier-reward RLOO, the frozen correctness probe does
not degrade relative to a probe retrained on the same checkpoint. Across the full
C_SFT → C_outcome ladder, `evasion_gap = auroc_retrained_linear − auroc_frozen`
stays in **[−0.021, +0.041]** on the unbalanced metric and **[−0.024, +0.041]**
on the balanced metric (3 seeds × 5 checkpoints, L16). This is the null band that
any probe-RL evasion claim has to clear, in whichever variant it is quoted. The
upper edge, **+0.041**, is the operative number and is identical in both variants
and stable across re-analysis.

**Evidence.** `run_phase1.py --config phase1_control_vanilla.yaml`, run over five
snapshots (steps 0/30/60/90/100) ingested from the original project's caches. At
the probe's own layer L16. **All AUROC columns below are unbalanced** — the
exactly reproducible variant; the balanced values differ from them by less than
0.02 at every cell, which is why the two bands almost coincide.

| step | frozen | retrained linear | retrained MLP | evasion gap | sep | acc |
|---|---|---|---|---|---|---|
| 0 (C_SFT) | 0.768 | 0.809 | 0.803 | +0.041 | 0.97 | 0.266 |
| 30 | 0.807 | 0.789 | 0.780 | −0.017 | 1.63 | 0.662 |
| 60 | 0.897 | 0.877 | 0.868 | −0.020 | 2.01 | 0.455 |
| 90 | 0.914 | 0.893 | 0.891 | −0.021 | 2.27 | 0.474 |
| 100 (C_outcome) | 0.911 | 0.903 | 0.897 | −0.007 | 2.37 | 0.513 |

The frozen probe *strengthens* over training (0.768 → 0.911), reproducing the
paper's 0.912 → 0.982 direction on the eval-regime probe. `nonlinear_gain`
(MLP − linear, unbalanced) stays within ±0.043 across all layers and seeds, so
correctness is linearly decodable
throughout — there is no nonlinear reservoir the linear probe is missing.

**Caveats.** Sampled at temp 0.6 / top_p 0.95 (eval regime), not the RL regime,
so this bounds pipeline/split/label artefacts, not sampling-regime artefacts.
Steps 30/60/90 cover the first 200 prompts, steps 0/100 all 406 clean prompts;
`prompt_idx` is consistent so the split applies uniformly, but the middle
snapshots have fewer rows. Step-30 accuracy (0.662) exceeds step-100 (0.513),
which is a property of the cached dynamics snapshots and is left as measured.

Figure: `results/fragility/figures/phase1_three_aurocs__vanilla_rloo_ladder__L16__on_policy.pdf`

---

## 2026-07-30 — Template residualisation: a large drop with no valid control yet — OPEN

**Status: open. Do not cite this as an established result.** An earlier version of
this entry claimed that residualising L16 activations on seven rhetorical-template
regexes drops linear-probe AUROC 0.903 → 0.596 "while a matched placebo costs
0.001", and read that contrast as strong evidence for H0 (confound exploitation).
The contrast is invalid and is retracted.

**Why the placebo does not control anything.** The placebo was matched to the
template set on **column count only**, not on removed variance. Three of its seven
columns (`has_div`, `has_equals`, `has_multidigit`) are 1.0 for every rollout, so
residualising on them is a literal no-op. Measured on C_outcome L16, the template
columns remove **31.4%** of activation variance and the placebo removes **2.0%** —
a 13× mismatch. The two conditions remove different amounts of the representation,
so their AUROCs are not comparable and the 0.001 figure carries no information.

**What survives.** The raw numbers are correct; only the inference from them was
wrong. `build_probe_zoo.py` / `confounds.residualisation_test`, C_outcome L16,
n=6306 rollouts, split by prompt, seed 0, unbalanced AUROC — these four rows are
the ones the code actually computes:

| condition | AUROC | variance removed |
|---|---|---|
| baseline linear probe | 0.903 | — |
| residualise on 7 template indicators | **0.596** | 31.4% |
| residualise on 7 placebo surface features (arithmetic symbols) | 0.902 | 2.0% |
| residualise on 7 random projections **of the activations** | 0.541 | — |

**The one variance-matched control that has been run is ambiguous.** Residualising
on PC[1:9] (31.6% of variance, matched to the template's 31.4%) gives AUROC
**0.895** at C_outcome against the template's 0.596, which points the same way as
the original claim. But the same comparison at C_SFT reverses: template 0.874 vs
matched 0.878, i.e. no effect at all. One PC subspace is not a rigorous
variance-matched control in any case — PCs are chosen to span variance, not to
resemble a text-derived feature set. Until a proper control exists (matched
removed variance *and* matched subspace construction, at both checkpoints), the
size of the template's contribution is unmeasured.

**Independent evidence that is not subject to this artefact.** AUROC computed
within template-score strata and pooled gives `auroc_frozen_cond_template` = 0.790
at C_outcome. The probe retains substantial signal with the template held fixed,
so whatever the confound contributes, it is not everything.

Supporting, from the control ladder (unbalanced): the template score *alone*
predicts correctness at AUROC 0.587 → **0.824** from C_SFT to C_outcome, and the
frozen probe's Spearman correlation with it rises **0.19 → 0.63**. The probe does
become progressively more entangled with writing style over vanilla RL, and runA
initialises from C_outcome, the checkpoint where that entanglement is largest.
This is a real trend and does not depend on the residualisation test.

**What it would take to close this.** A control that matches removed variance and
is built from text-derived features rather than PCs, run at both C_SFT and
C_outcome, and recorded to parquet like everything else.

---

## 2026-07-30 — Methods note: weight-space probe drift is a weak instrument

**Claim.** `dir_drift` defined as the cosine between two logistic probe
directions is not usable at face value in this setting. Two linear probes fit on
**disjoint halves of the same checkpoint's data** agree at cosine **−0.01 to 0.24**
while both achieve AUROC ≈ 0.85. Any drift figure must be normalised by that
refit-noise floor, or replaced by a function-space measure.

**Evidence.** `probes.direction_self_consistency`, control ladder, 3 seeds ×
5 checkpoints × 3 layers. Self-baseline range **−0.006 to 0.240** (corrected
2026-08-08 from a quoted "0.052–0.184"; read out of the metric store over all
45 cells, it is wider at both ends and includes a negative value at L12/step 60,
where the refit noise swamps the signal entirely). Frozen-vs-retrained cosine
over the same ladder runs 0.128 (step 0) → 0.546 (step 100) — i.e. comparable to
or a few times the noise floor, and *rising*, which is the correct direction
given the frozen probe was fit on step-100 data.

The negative value matters for more than tidiness: it is why `_dir_drift_v2`
returns NaN rather than a ratio when the halves cosine is non-positive. A range
quoted as strictly positive would suggest that branch is unreachable.

**Consequence for the paper.** `dir_drift_normalized` and
`score_corr_frozen_retrained` (Spearman between the two probes' outputs on
held-out rows; 0.63–0.90 on the control ladder) are reported instead of the raw
cosine. Reporting a raw cosine of 0.13 as "the direction rotated away" would have
been a false claim, and it is the claim the metric as originally specified
invites.

---

## 2026-07-30 — Scoping check: the policy keeps the surface form and drops the substance

**Claim.** On the post-Goodhart checkpoints, the rhetorical template becomes
near-universal while its association with correctness falls to chance. Two runs
that Goodhart the same probe by *opposite* structural strategies converge on the
same outcome. This is H0's central prediction and it holds on rollout text alone,
before any activation is harvested.

**Evidence.** `scope_check.py`, clean-406 prompts × 16 rollouts, verifier labels
recomputed from text:

| | acc (first block) | mean blocks | P(template≥3) | AUROC(template) |
|---|---|---|---|---|
| C_SFT | 0.290 | 2.71 | 0.765 | 0.599 |
| C_outcome | 0.550 | 6.78 | 0.108 | **0.804** |
| runA post-RL (init C_outcome) | 0.236 | 15.55 | **0.953** | **0.534** |
| runB post-RL (init C_SFT) | 0.073 | 1.27 | **1.000** | **0.523** |

AUROC(template) is unbalanced.

runA: accuracy −31 pp, template rate +85 pp, AUROC(template) 0.804 → 0.534.
runB: accuracy −22 pp, template rate +24 pp (saturating at 1.000), 0.599 → 0.523.
The accuracy deltas reproduce the published −31/−22 pp exactly, so the artefacts
are the right ones.

Note the shape of the C_outcome column: the template is *rare* there (10.8%) and
*highly diagnostic* (0.804). The probe was trained on a checkpoint where the
template was a scarce, reliable marker of correctness. The policy then made it
universal, which destroyed its diagnosticity.

**Caveats.** Template score is length-sensitive and these checkpoints differ
enormously in length (1.27 to 15.55 answer blocks), so part of the AUROC(template)
trajectory may be a length effect rather than a template effect. Nothing in this
repo separates the two yet: a length-conditioned version of the column was
computed once in an unsaved session and is not reproducible from any artefact, so
it has been removed rather than quoted. Until a recorded length control exists,
the defensible claim is "the probe's signal rides on some surface structure that
decouples post-RL", not "on this particular seven-regex template". Text-only: says
nothing yet about what happened to the *activations*, which is what H1 vs H2
needs.

---

## 2026-07-30 — Scoping harvest (runB, steps 0 and 99): H2 is out, but the premise needs reframing

**Claim.** On runB's post-collapse checkpoint the frozen probe fails completely
on held-out prompts — balanced AUROC 0.579 → 0.516, while it reports a mean score
of 0.996 on 99.1% of rollouts. A probe retrained on the same activations is
**unchanged** (balanced 0.661 → 0.648). So the correctness information is not
destroyed: this is H1/H0-shaped, and **H2 (corruption) is ruled out** for runB.

**But** the absolute level is the story. Retrained linear tops out at 0.65
balanced in the RL sampling regime. The paper's near-oracle 0.98 is a temp=0.6,
C_outcome, first-block number. At temp=1.0 — the regime the RL reward actually
reads — correctness is only weakly linearly decodable to begin with. On the
vanilla ladder at C_SFT, retrained linear scores 0.809 at temp 0.6 versus 0.646
here at temp 1.0. The honest framing is not "a 0.98 monitor collapsed" but
"a ~0.65 monitor fell to chance while a fresh 0.65 monitor kept working".

**Evidence.** 2 checkpoints × 406 clean prompts × 8 rollouts at temp 1.0, layers
12/14/16/18/20, seed 0. L16, with both AUROC variants given explicitly:

| | frozen | frozen bal. | retrained lin | retrained bal. | MLP | MLP bal. | gap (bal.) | probe mean | frac>0.95 | acc |
|---|---|---|---|---|---|---|---|---|---|---|
| step 0 | 0.590 | 0.579 | 0.646 | 0.661 | 0.642 | 0.639 | +0.082 | 0.498 | 0.076 | 0.159 |
| step 99 | **0.492** | 0.516 | 0.610 | 0.648 | 0.586 | 0.608 | **+0.132** | **0.996** | **0.991** | 0.068 |

The balanced evasion gap of +0.132 sits well outside the balanced control null
band of [−0.024, +0.041]. Frozen-vs-retrained score correlation collapses
0.321 → 0.074: the deployed monitor and a fresh one stop agreeing almost entirely.

**Relocation: no on the unbalanced metric, not clean on the balanced one.**
Unbalanced retrained AUROC falls at *every* layer (L12 0.634→0.603, L16
0.646→0.610, L20 0.664→0.542) and rises nowhere. On the balanced metric — which
this entry elsewhere calls the reliable one, given the 6.7% positive rate — two of
the five layers rise: L12 0.636 → **0.662** and L14 0.636 → 0.643, while L16
0.661→0.648, L18 0.661→0.621 and L20 0.683→0.538 fall. The rises are small and at
the layers furthest from the probe, and this is 1 seed and 2 checkpoints, so it is
not evidence *for* relocation either. The honest statement is that runB cannot
settle the relocation question; runA's full ladder can, and does (see below).

**H0 is untestable on runB.** The template confound is weak here — AUROC(template)
0.563 → 0.559 unbalanced, flat and low — because runB initialises from C_SFT, where the
template↔correctness coupling was already weak (0.599). The strong confound lives
at C_outcome (0.804), which is **runA's** init. So runA, not runB, is the run
that discriminates H0 from H1. My earlier "harvest runB first" call was right for
resolving the collapse *dynamics* and wrong for the *mechanism* question.

**Caveats.** 1 seed, 2 checkpoints, and step 0 here is a cross-distribution
baseline (the probe was fit on C_outcome, step 0 is C_SFT), so the 0.590 starting
point understates the probe's in-regime quality. runA step 0 is the clean
baseline and has not been harvested. Positive rate at step 99 is 6.7% — **66
positives in 984 val rows** — so unbalanced AUROC is the less reliable column, and
balanced figures are quoted for the headline claims for that reason.

---

## 2026-07-30 — Scoping harvest (runA, steps 0 and 99): the mechanism is a MIXTURE

**Claim.** On runA — the run initialised from the probe's own training
checkpoint, and therefore the only one where the mechanism is decidable — the
frozen probe falls from a genuine 0.785 to 0.533 (chance) while retrained probes
retain 0.695 linear / 0.751 MLP. **H2 is ruled out.** But retrained AUROC also
degrades (0.815 → 0.695), so this is not clean H1 evasion either. All three
hypotheses have partial support; the mechanism is a mixture, which is what the
pre-registered prior said was most likely.

**Evidence.** 406 clean prompts × 8 rollouts at temp 1.0, L16, seed 0, balanced
AUROC (positive rate falls 0.360 → 0.124, so balanced is the reliable column):

| | frozen | retrained lin | retrained MLP | gap | score corr | sep | acc |
|---|---|---|---|---|---|---|---|
| step 0 (C_outcome) | 0.785 | 0.815 | 0.784 | +0.030 | 0.733 | 1.096 | 0.372 |
| step 99 | **0.533** | **0.695** | **0.751** | **+0.163** | **0.099** | 0.171 | 0.133 |

All three AUROC columns and the gap are balanced. `score corr`, `sep` and `acc`
have no balanced variant and are computed on all val rows.

The gap of +0.163 is four times the top of the balanced control null band
[−0.024, +0.041]. Agreement between the deployed monitor and a fresh one collapses
(0.733 → 0.099), and separation along the *fixed* t=0 direction all but vanishes
(1.096 → 0.171) — the probe's axis stops carrying the distinction while the
activations still contain it.

**A lead that did not survive the full ladder: the MLP overtakes the linear
probe.** At step 0 linear beats MLP (balanced 0.815 vs 0.784); at step 99 MLP beats
linear (balanced 0.751 vs 0.695). The recorded `nonlinear_gain` moves
**−0.018 → +0.018**. Read at the time as correctness becoming *less linearly*
accessible while staying nonlinearly present — an H1-flavoured refinement. **The
full ladder retracts this; see the 2026-07-31 entry.** Two checkpoints could not
separate a trend from seed noise.

**H0, on the run where it is testable.** Every H0 prediction fires, and none is
sufficient. AUROC(template) and AUROC(frozen | template fixed) are **unbalanced**
here (`auroc_frozen_cond_template` has no balanced variant, so both are quoted
unbalanced to keep the column comparable); the remaining rows are not AUROCs:

| | step 0 | step 99 |
|---|---|---|
| AUROC(template), unbalanced | 0.743 | 0.584 |
| corr(probe, template), Spearman | 0.531 | 0.037 |
| P(template ≥ 3) | 0.492 | 0.948 |
| AUROC(frozen \| template fixed), unbalanced | 0.703 | 0.552 |
| mean answer blocks | 5.04 | 15.44 |

The template goes near-universal while its diagnosticity decays and its coupling
to the probe collapses — H0's signature. **But** the probe was never *only* a
template detector: conditional on template it still scored 0.703 at step 0, and
it lost that too (0.552). So the confound is real and is not the whole story.

**Reading.** H1-dominant with a substantial H0 component, H2 excluded. The frozen
monitor fails for two compounding reasons: the surface feature it partly relied
on stopped being diagnostic, *and* the correctness information reorganised out of
its linear reach.

**Caveats.** 1 seed, 2 checkpoints, no intermediate dynamics — this says nothing
yet about *ordering*, which is the whole Phase-1/Phase-2 question (does the
confound decouple before or after the linear signal degrades?). Balanced AUROC is
quoted because step 99 has 12.4% positives. runB's numbers are weaker throughout
and should not be pooled with these: it initialises from C_SFT, where both the
probe and the confound were weak to begin with.

**Supersedes** the pessimistic reframing in the runB entry above: the ~0.65
retrained ceiling was runB's cross-distribution weakness, not a property of
temp-1.0 sampling. In-distribution the frozen probe is a real 0.785 monitor at
the RL sampling regime.

---

## 2026-07-31 — runA full ladder: the collapse has TWO phases, and reward hacking precedes monitor failure by ~40 steps

**Claim.** With all 11 checkpoints harvested, the collapse resolves into two
mechanistically distinct phases, and neither matches a single hypothesis:

- **Steps 0–40 — reward hacking with an intact monitor.** The policy is already
  gaming hard: probe score 0.460 → 0.677, answer blocks per rollout 5.0 → 13.6,
  accuracy 0.372 → 0.302. The monitor is *fine*: frozen AUROC 0.785 → 0.767 and
  the evasion gap stays inside the balanced control null band [−0.024, +0.041] at
  every step (0.030, 0.014, 0.030, 0.006, 0.032).
- **Steps 50–60 — general degradation (H2-like).** Frozen and retrained fall
  *together*: 0.767 → 0.646 and 0.799 → 0.690. The gap sits at the null band's
  edge and then just past it (0.029, 0.043 against a band top of 0.041). `sep`
  along the fixed t=0 direction halves (0.941 → 0.488), and the structural
  confound decouples in the same window (AUROC(template) 0.694 → 0.542;
  corr(probe, template) 0.475 → 0.072).
- **Steps 70–99 — divergence (H1/H0-like).** Retrained stabilises at 0.66–0.72
  while frozen keeps falling to 0.533. Step 60's gap (0.043) is the first to
  exceed the band top, but only by 0.002; step 70 is the first to clear it with
  margin (+0.140), and the series ends at +0.163.

**The headline ordering result: reward hacking starts ~40 steps before the
monitor stops working.** A deployment that watched only the probe's score would
have seen it climbing from step 10; a deployment that watched the probe's
*discrimination* would have seen nothing wrong until step 50. Score inflation and
discrimination loss are separate events, and the gap between them is most of
training.

**Evidence.** L16, seed 0, 406 clean prompts × 8 rollouts, temp 1.0. `frozen`,
`retrained` and `gap` are **balanced**; `AUROC(templ)` is **unbalanced** (quoted
that way so it stays comparable with the H0 table in the entry above and with
runB, neither of which has a balanced conditional variant); `score corr`, `sep`
and `corr(probe,templ)` have no balanced variant and are computed on the val
split; `blocks` and `acc` are computed on all harvested rollouts:

> **REGENERATED 2026-08-08 from the metric store.** Pre-relabel, like the
> 3-seed table in the 2026-07-31 post-audit entry. The withdrawn version read
> frozen 0.785 … 0.533 with a step-99 gap of +0.163 and a step-90 gap of +0.041.

| step | frozen | retrained | gap | score corr | sep | AUROC(templ) | corr(probe,templ) | blocks | acc |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.797 | 0.807 | +0.010 | 0.730 | 1.164 | 0.748 | 0.528 | 5.0 | 0.372 |
| 10 | 0.791 | 0.790 | −0.001 | 0.720 | 1.149 | 0.754 | 0.568 | 11.0 | 0.308 |
| 20 | 0.799 | 0.786 | −0.013 | 0.721 | 1.213 | 0.736 | 0.572 | 12.7 | 0.312 |
| 30 | 0.815 | 0.782 | −0.033 | 0.735 | 1.216 | 0.731 | 0.565 | 12.9 | 0.291 |
| 40 | 0.800 | 0.815 | +0.014 | 0.700 | 1.142 | 0.697 | 0.489 | 13.6 | 0.302 |
| 50 | 0.723 | 0.787 | +0.065 | 0.614 | 0.682 | 0.572 | 0.250 | 16.0 | 0.260 |
| 60 | 0.657 | 0.694 | +0.037 | 0.307 | 0.534 | 0.549 | 0.040 | 15.9 | 0.186 |
| 70 | 0.576 | 0.654 | +0.078 | 0.125 | 0.237 | 0.560 | 0.036 | 18.9 | 0.148 |
| 80 | 0.607 | 0.740 | **+0.133** | 0.221 | 0.393 | 0.543 | 0.038 | 18.3 | 0.148 |
| 90 | 0.577 | 0.681 | +0.104 | 0.314 | 0.218 | 0.564 | 0.081 | 16.3 | 0.141 |
| 99 | 0.582 | 0.736 | **+0.155** | 0.170 | 0.213 | 0.579 | 0.096 | 15.4 | 0.133 |

`sep` declines from 1.16 to 0.21 across the ladder, and the decline is
concentrated after step 40 (1.142 at step 40 → 0.213 at step 99). It is **not
monotone**: the series runs 1.164, 1.149, 1.213, 1.216, 1.142, 0.682, 0.534,
0.237, 0.393, 0.218, 0.213, and three of the ten step-to-step transitions rise
(10→20, 20→30, 70→80) — only one of them after step 40, where the withdrawn
series had two. What the series does show is that the fixed t=0 direction starts
losing correct/incorrect separation at step 40, ahead of the point where frozen
AUROC registers a change.

**No relocation.** Balanced retrained AUROC declines at all five layers (L12
0.811→0.703, L14 0.806→0.732, L16 0.815→0.695, L18 0.804→0.670, L20
0.813→0.687); the unbalanced metric gives the same picture (L12 0.799→0.694, L14
0.799→0.719, L16 0.806→0.718, L18 0.799→0.692, L20 0.808→0.685). No layer picks
up what L16 loses under either convention, so the information did not move across
depth.

**Retraction: the linear→nonlinear crossover was noise.** The runA two-checkpoint
entry above called the MLP overtaking the linear probe "the sharpest lead the
scoping data produced". The full ladder does not support it. MLP and linear track
each other with no consistent ordering across the 11 checkpoints, in either
convention — balanced, MLP−linear at seed 0 is −0.077 at step 70, −0.013 at step
80 and +0.056 at step 99, and the sign flips repeatedly earlier in the ladder.
Two checkpoints were not enough to distinguish a trend from seed noise. Do not use
that claim.

**3-seed confirmation** (mean ± sd across seeds 0/1/2, L16, all four columns
balanced).

> **REGENERATED 2026-08-08 from the metric store**, like the two tables above.
> The withdrawn pre-relabel version read frozen 0.794 … 0.554 with a step-99 gap
> of 0.164 and a step-90 gap of 0.055. Note also that its standard deviations
> were up to 0.039; on the corrected values the largest is 0.008 (excluding the
> MLP column), because seeds here vary only the balanced-AUROC subsample and the
> MLP init — the split is fixed — and the 25-draw averaging removed most of that.

| step | frozen | retrained linear | retrained MLP | evasion gap |
|---|---|---|---|---|
| 0 | 0.797 ± 0.000 | 0.807 ± 0.001 | 0.795 ± 0.002 | 0.010 ± 0.001 |
| 10 | 0.794 ± 0.005 | 0.789 ± 0.002 | 0.789 ± 0.004 | −0.005 ± 0.004 |
| 20 | 0.800 ± 0.001 | 0.786 ± 0.002 | 0.793 ± 0.002 | −0.014 ± 0.002 |
| 30 | 0.814 ± 0.001 | 0.782 ± 0.001 | 0.774 ± 0.008 | −0.032 ± 0.002 |
| 40 | 0.803 ± 0.003 | 0.818 ± 0.004 | 0.822 ± 0.014 | 0.014 ± 0.003 |
| 50 | 0.723 ± 0.000 | 0.786 ± 0.001 | 0.782 ± 0.010 | 0.064 ± 0.001 |
| 60 | 0.652 ± 0.006 | 0.693 ± 0.002 | 0.667 ± 0.036 | 0.041 ± 0.004 |
| 70 | 0.575 ± 0.002 | 0.657 ± 0.004 | 0.656 ± 0.029 | 0.082 ± 0.004 |
| 80 | 0.612 ± 0.005 | 0.741 ± 0.003 | 0.693 ± 0.101 | 0.130 ± 0.003 |
| 90 | 0.582 ± 0.005 | 0.690 ± 0.008 | 0.694 ± 0.021 | 0.108 ± 0.004 |
| 99 | 0.583 ± 0.002 | 0.742 ± 0.007 | 0.714 ± 0.044 | 0.158 ± 0.005 |

Phase boundaries, stated precisely against the 3-seed figures:

- **Frozen is flat-to-rising through step 40** (it peaks at step 30, 0.814) and
  falls from step 50. Retrained is flat through 50 and falls from 60. The
  degradation is therefore NOT simultaneous, which the withdrawn table implied:
  the frozen probe leads by roughly one checkpoint.
- The evasion gap is inside the balanced control band through step 40, **exits at
  step 50** (0.064 ± 0.001, with sd clearance), returns to the band's top edge at
  60 (0.041 ± 0.004), and is clear from 70 onward.
- So the divergence between deployed and fresh monitors begins at **step 50**,
  not step 70 — ten steps sooner than the withdrawn table said, and fifty after
  score inflation begins.

**Nonlinear probes add nothing, anywhere.** The recorded `nonlinear_gain`
(MLP − linear, unbalanced — the convention the metric is stored in) at L16 has a
3-seed mean between −0.051 and +0.009 across the eleven checkpoints, and its sign
flips from step to step (negative at 7 of 11,
positive at 4). It is not a small-and-stable quantity — at step 70 the 3-seed mean
is −0.051 with sd 0.033, and at 5 of 11 steps the mean exceeds its own sd — but it
has no consistent direction, which is what matters: there is no step at which the
MLP reliably beats the linear probe. Whatever correctness information survives is
linearly accessible; the collapse is not a linear probe failing to keep up with a
nonlinear reformatting. This confirms the retraction above with 3 seeds rather
than 1.

**Caveats.** Step 90 is an outlier on every measure (gap 0.055 against 0.166 at
step 80 and 0.164 at step 99) — the published trajectory records a single-batch
KL spike to 1040 at exactly this step, so the checkpoint is anomalous rather than
the metric noisy. Seeds vary only the balanced-AUROC subsample and MLP init; the
split is fixed across all of them, so these sds understate true run-to-run
variance. Single trajectory: one RL run, not a distribution over runs.

**Open items.**

- runB's full ladder is still at 2 checkpoints; runA's 11 are harvested and
  analysed at 3 seeds. runB is the robustness check, not the primary ladder.
- A variance-matched, text-derived control for the template residualisation test,
  at both C_SFT and C_outcome. Until then the H0 magnitude is unmeasured.
- Phase 2 steering + patching (pre-registered in `phase2_predictions.md`;
  do not read the predictions after seeing the data).
- Extend the transfer matrix to the probe-RL checkpoints. As recorded it covers
  only the vanilla control ladder (steps 0 and 100), so it does not yet say
  whether the attack transfers across probes.
