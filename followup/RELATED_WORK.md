# Related work and novelty assessment

Compiled 2026-08-02 from six parallel literature sweeps. **Everything here comes
from agent summaries, not primary reading.** Before committing to any framing,
read at minimum: Gupta & Jenner §5, Taufeeque et al. Eq. 3 and Table 4, Beigi et
al. (PRIME) §timing, and Yudelson's control-run methodology.

---

## The headline: what got claimed while this project was being planned

`CLAUDE.md` was written to answer "what happens mechanistically when a policy is
optimised against a correctness probe." **That question was answered by other
people between June 2025 and May 2026.**

| paper | venue | what it owns |
|---|---|---|
| **Gupta & Jenner, RL-Obfuscation** ([2506.14261](https://arxiv.org/abs/2506.14261)) | Jun 2025 | The weights×text decomposition, **quantified: ~70% policy / ~30% representation**, by the same token-forcing logic we used |
| **Taufeeque, Heimersheim, Gleave & Cundy, The Obfuscation Atlas** ([2602.15515](https://arxiv.org/abs/2602.15515)) | **ICML 2026 (Oral)** | The formal three-term decomposition (Eq. 3) and the *obfuscated policy* vs *obfuscated activations* taxonomy |
| **Yudelson, Held-out Monitors Sometimes Degrade** ([LessWrong](https://www.greaterwrong.com/posts/APkFfRp2AicL9RqvT/)) | Jul 2026 | The policy term measured with the representation term structurally zeroed, plus a **no-proxy control run** |

Prior to those, the mechanism was already named: **Manheim & Garrabrant (2018)**
call it *Extremal Goodhart (Model Insufficiency)* — "selection pressure moves the
metric away from the region in which the relationship is most accurate so that
the relationship collapses." **Gao, Schulman & Hilton (ICML 2023)** assert it
empirically and use **answer length** as the worked example. **Laidlaw et al.
(ICLR 2025)** define reward hacking *as* distribution shift. **Karwowski et al.
(ICLR 2024)** give the occupancy-measure geometry.

---

## Scorecard

| our finding | status | who owns it |
|---|---|---|
| Probe-as-reward collapses monitor while accuracy falls | **partly open** | Nobody has shown it for a *correctness* probe as *positive* reward. The Atlas used a deception-probe *penalty* and accuracy went **up** (3.1% → 48%) — opposite sign |
| weights×text decomposition (~90/17) | **scooped** | Gupta & Jenner (70/30). Ours differs in domain and reward sign |
| the 4th cell: **tuned weights × base text** | **ours** | Both prior papers hold text at the *tuned* policy's outputs. Nobody scores base text under tuned weights |
| **~40-step lag: hacking onset → monitor discrimination loss** | **NOVEL** | Confirmed absent at survey level by [Fudan survey 2604.13602](https://arxiv.org/abs/2604.13602). Gao et al. never plot proxy accuracy vs KL at all — only validation loss on a fixed t=0 set |
| retrained probe recovers | **contested** | Recovery: Wehner & Fritz, Duan (0.981). **Against**: Gupta & Jenner (near-null), Neural Chameleons (post-hoc probes on frozen post-evasion weights still fail), Atlas (continual retraining → no difference) |
| length/surface beats probe | **fragments** | Dutta (0.98 vs 0.998, 0.5B toy); Singhal (70–90% of RLHF gain is length); McKenzie (bag-of-words 88.9%) |
| regularisation buys delay | **contested** | Delay: Gao (KL ≈ early stopping), WARM ("delays the collapse"), Wolf. **Prevent**: Coste, Atlas (β=1e-1 → 100% honest). Neutral: MacDermott (KL has no consistent effect, sign flips) |

---

## The one strong result, and why nobody else has it

**Reward hacking precedes monitor discrimination failure by ~40 RL steps.**

Verified to be measured in the strong form: `auroc_frozen(t)` uses checkpoint
*t*'s own freshly sampled rollouts (each step reads its own `step_t.jsonl` from
its own model), on **126 held-out prompts**, with the frozen probe trained on a
**disjoint 300-prompt set** (overlap = 0, verified). This defeats the objection
that a static t=0 audit distribution would trivially fail to move
(Lamparth et al. Thm 3.9).

**Why the gap exists is structural and should be stated as part of the
contribution**: RM accuracy on the current policy distribution requires fresh
preference labels at every checkpoint. A *verifiable* correctness probe gives
AUROC free at every checkpoint. The affordance is the reason the measurement was
available to us and not to the RLHF literature.

**Positioning required:**
- **Baker et al. ([2503.11926](https://arxiv.org/abs/2503.11926))** conflate the two
  events — their "recall falls to near zero" mixes *the agent evades* with *the
  monitor can no longer discriminate*, because they never hold out separate data.
  This is the cleanest one-sentence statement of our delta.
- **PRIME ([2606.09711](https://arxiv.org/abs/2606.09711))** reports a **~40-step
  lag of the same magnitude in the opposite direction** (internal precursor leads
  behaviour). Three separate reviewers flagged that readers will conflate them.
  Pre-empt in the intro.

---

## Three things that must be fixed before writing up

**1. No no-pressure control.** The Atlas finds representation drift occurs under
RLVR **with or without** a detector penalty. Yudelson found most monitors degrade
even in a control run and *subtracts* that degradation. Without a fixed-text arm
on the vanilla-RLOO ladder at matched KL, our 17% representational term is not
attributable to optimisation pressure. Flagged independently by three reviewers.
~$8.

**2. H2b — concept drift — was never in the hypothesis set.** The Atlas explains
its 0.991 → 0.376 collapse as *the model's beliefs about what counts as
deceptive changed* (undesired-answer rate 10–40% → 45–65%). That is neither H1
(information moved) nor H2 (information degraded). It is a third mechanism, and
`CLAUDE.md`'s protocol forbids adding hypotheses after seeing data. Record it as
a post-hoc addition, honestly labelled.

**3. Aggregation may dominate the Phase 3 zoo.** Gupta & Jenner: attention probes
and max-pooling are **robust (0.0–0.02 evasion)** while mean/median-pooled linear
and MLP probes are evadable (0.59–0.89). We tested only mean-pooled
single-position probes. Any H3 predictor must reproduce this or it is confounded.
ODIN's result is the one-line argument for why: decorrelating length moves
reward–length ρ from 0.451 → −0.05 at an RM accuracy cost of only 70.1% → 69.2%
— **in-distribution AUROC does not predict robustness.**

---

## Contrast cases worth confronting rather than hiding

- **Parekh, Drop the Act** ([2605.11467](https://arxiv.org/html/2605.11467v1)) — a
  frozen gated-attention probe used as an advantage *filter* holds **AUROC 1.000
  before, during and after** training. Filter-vs-reward and attention-vs-linear
  both plausibly explain the difference from our collapse. Separating them would
  itself be a result.
- **Wu & Tang** ([2604.01476](https://arxiv.org/html/2604.01476)) — concept
  directions at cosine **≥0.99** with base after RL. Independent support for a
  small representation term.
- **Anthropic, Recommendations for Technical AI Safety Research Directions** —
  verbatim: *"there is substantial risk that activation monitors are not robust to
  distributional shifts."* They flag the worry and propose no experiment. Our
  decomposition is a direct answer to a named open problem.

---

## The methodological claims, checked separately

A seventh sweep covered concept erasure, probing methodology, mech interp, fMRI
confound regression, and econometrics.

**Variance-matched residualisation controls: the currency change is genuinely
absent — but the phenomenon is not.**

Every removal control found in every field is matched on **rank, count, degrees
of freedom, spectrum, or norm**. Never on removed variance. Zero hits for
"variance-matched control" or equivalents in arXiv metadata.

But **do not write "nobody noticed this"** — it is falsifiable and will be
punished. That random removal costs little is in print repeatedly and unremarked:
LEACE ("erasing a randomly selected subspace has little to no effect"), Tigges
(<1%), ITI (0.7pp), and Amnesic Probing's own Table 1 at low rank. Bright &
Murphy (2015, *NeuroImage*) state the failure mode outright: *"any group of
regressors that randomly sample variance may remove highly structured 'signal'
as well as 'noise'."*

**The near-scoop, which must be cited**: Haghighatkhah et al. (EMNLP 2022,
[2212.04273](https://arxiv.org/abs/2212.04273)) sample random directions
**variance-weighted from the data's PC spectrum**, run 500 draws with CIs, and
conclude *"the improvement in similarity scores are due to reducing dimensions in
general rather than removing (partial) representations of gender"* — our
conclusion, in word embeddings, with a count-matched design.

**Elazar et al. flagged the gap themselves, twice**: *"the number of classes in
those experiments might be a factor… While not part of this study, it would be
interesting to control for this factor in future work."* They frame the
alternative as *rank*-reduction, never variance.

Defensible formulation: **report removed-variance fraction alongside every
ablation, and match the control on that quantity rather than on rank.** A
protocol fix, not a discovery. Distinguish explicitly from three things reviewers
will raise: LEACE's minimality theorem (a better *treatment*, not a null),
causal scrubbing's "ablations remove variation" (about the treatment), and
Rozanova et al.'s "run more random draws" (fixes the control's *variance*, not
its *calibration*). Cite **Elsayed & Cunningham (2017, Nat. Neuro.)** — nulls for
high-dimensional population data must be matched on second-order statistics — so
the idea reads as a principled import rather than an ad-hoc complaint.

**The other methodological claims:**

- **CKA blindness — drop it.** It is Kornblith et al.'s own Eq. 14, presented as
  a *design goal*, and published as a critique by Ding et al. (NeurIPS 2021),
  who use probing accuracy as the readout. Reframe our CKA number as the
  *global-drift null* it was always meant to be.
- **Frozen-threshold TPR — keep, reframed.** TPR@1%FPR is already the field
  convention (Goldowsky-Dill et al. 2025). The surviving point is sharper: *every*
  ROC-family metric, including that one, is equally invariant to monotone
  recalibration, because it re-derives its threshold from the current
  evaluation's negatives. Only a **numerically frozen** t=0 threshold catches it.
- **Refit-noise baseline for direction cosine — keep.** No established practice.
  Frame by analogy to the RSA **noise ceiling** (Nili et al. 2014). Expect
  pushback from papers reporting seed-refit cosines ≥0.998 — say plainly that
  optimizer-seed stability and data-resampling stability are different baselines.

## Revised spine

1. **Setting**: a correctness probe as the *positive* RL reward — no evasion
   incentive, pure Goodhart. Distinguishes us from every prior paper, all of
   which use the probe as a penalty.
2. **Headline**: the ~40-step detection window, measured on-policy on held-out
   prompts, against Baker et al.'s conflation.
3. **Mechanism**: the completed 2×2 including the unclaimed fourth cell, reported
   with its ruler named (AUROC-scale; the split runs +46% on Youden's J and
   −266% on TPR) and its interaction term explicit.
4. **Methodological**: frozen-threshold TPR (every ROC-family metric, including
   the field-standard TPR@1%FPR, is equally invariant to monotone recalibration);
   the refit-noise baseline for direction drift (an RSA *noise ceiling* by
   analogy); variance-matched residualisation controls (Elazar et al.'s canonical
   control is count-matched, and they flagged the gap as future work in 2021).

**Honest scale**: a workshop paper, not the ICLR submission `CLAUDE.md` targeted.
One seed, one 0.5B model, one task.
