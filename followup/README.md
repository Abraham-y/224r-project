# Probe Fragility Under Optimization Pressure

Follow-up to the CS 224R correctness-probe project. The paper showed a linear
correctness probe is near-oracle as a read-only deployment selector (AUROC 0.98)
and Goodharts catastrophically as an RL reward (−25 pp accuracy). That result is
behavioural. **This project asks what happens mechanistically, and whether
monitor robustness is predictable in advance.**

Plan and conventions: `CLAUDE.md`. Critique of that plan, and what was changed:
`ASSESSMENT.md`. Results as they land: `PAPER_NOTES.md`.

## Layout

```
fragility_core/            shared library — reads caches, never runs a model
  registry.py              config YAML -> content-hash run_id; append-only registry
  metrics_io.py            append-only parquet metric store
  labels.py                verifier labels, template features, the global split
  activations.py           acts/{run_id}/{step}/{layer}.npy + labels.parquet
  probes.py                frozen eval, retrain-linear, retrain-MLP, directions
  geometry.py              dir_drift, sep, CKA, participation ratio
  confounds.py             H0: is the probe reading a writing style?
  steering.py              add-vector steering + patching hooks
  checkpoint_logging.py    ladder discovery; dense checkpointing for new runs
  figures.py               one function per paper figure, parquet-only

experiments/fragility/
  phase0_replicate/        NOTES.md (read the trainer), ingest, harvest
  phase1_evasion_vs_corruption/   run_phase1.py, make_figures.py
  phase2_causal_status/    phase2_predictions.md (PRE-REGISTERED), steering, patching
  phase3_predict_robustness/  probe zoo, transfer matrix, dense probe-RLOO

configs/fragility/         one YAML per run; no hardcoded hyperparameters
results/fragility/         parquet + figures + registry (append-only)
acts/                      activation cache (gitignored; rebuild from configs)
modal_fragility.py         Modal launcher (separate from the paper's modal_train.py)
```

The original repo one level up is treated as **read-only**. Nothing in
`extension/`, `rloo_trainer/`, or `evaluation/` was modified. Activation
extraction is *imported* from `extension/probe/cache_hidden_states.py` rather
than reimplemented — two historical bugs came from a second implementation
drifting from the cache's token conventions, and both failed silently.

## Run it

Zero-cost, works on a laptop. Reproduces everything in `PAPER_NOTES.md`:

```bash
cd followup

# Vanilla-RLOO control ladder from the original project's caches
python experiments/fragility/phase0_replicate/ingest_existing_cache.py \
    --config ingest_vanilla_ladder.yaml

# Phase 1 over it (3 seeds). This measures the evasion-gap null band.
python experiments/fragility/phase1_evasion_vs_corruption/run_phase1.py \
    --config phase1_control_vanilla.yaml --run_id vanilla_rloo_ladder

python experiments/fragility/phase1_evasion_vs_corruption/make_figures.py \
    --run_id vanilla_rloo_ladder --layer 16 --layers 12,16,20

# 13-probe zoo + the residualisation test with its controls
python experiments/fragility/phase3_predict_robustness/build_probe_zoo.py \
    --config phase3_probe_zoo.yaml
```

Needs a GPU (Modal). `set -a; source ../.env; set +a` first:

```bash
# Phase 0: harvest activations off the EXISTING probe-RL checkpoint ladders.
# ~$20 each. runB first — its collapse is immediate, so the ladder resolves it better.
modal run --detach modal_fragility.py harvest -- --config phase0_harvest_runB.yaml
modal run --detach modal_fragility.py harvest -- --config phase0_harvest_runA.yaml

# Phase 1 on the harvested ladder (no GPU needed, but the acts live on the volume)
modal run modal_fragility.py analyse -- \
    --config phase1_default.yaml --run_id phase0_harvest_runB

# Transfer matrix — free, and it gates all Phase-3 spend. Run before Phase 2 or 3.
python experiments/fragility/phase3_predict_robustness/transfer_matrix.py \
    --ladders phase0_harvest_runA:99 phase0_harvest_runB:99

# Phase 2 — READ phase2_predictions.md FIRST. It is pre-registered.
modal run --detach modal_fragility.py steering -- --config phase2_default.yaml --ladder runA
modal run --detach modal_fragility.py patching -- --config phase2_default.yaml --ladder runA
```

## The three hypotheses

| | frozen AUROC | retrained AUROC | mechanism |
|---|---|---|---|
| **H1 evasion** | collapses | recovers | info moved off the probe direction |
| **H2 corruption** | collapses | also collapses | the information itself degraded |
| **H0 confound** | collapses | recovers | the probe scored a writing style that RL decoupled from correctness |

H0 is not in `CLAUDE.md` and is the leading hypothesis; see `ASSESSMENT.md` §2.
H0 and H1 share the Phase-1 signature (large positive `evasion_gap`), so the
headline figure does not decide between them. What decides it:
`auroc_frozen_cond_template` (Phase 1) and Δ-template-under-steering (Phase 2).

## Rules that are enforced, not just documented

- **Missing is missing.** A checkpoint with no cached snapshot produces no
  parquet rows. Nothing is interpolated or stubbed. Figures show gaps.
- **Append-only.** Every analysis run writes a new parquet shard; the metric
  store is never overwritten. `metrics_io.latest()` deduplicates for plotting.
  Derived CSVs (`transfer_matrix.csv`, `scope_check.csv`, `prehoc_features.csv`)
  keep a canonical path so figures and readers can reference it, and every write
  also drops a timestamped copy in `<stem>.archive/` — they used to be rewritten
  in place, which is how the CSV on disk came to hold only a later fixed-text run
  while the on-policy rows quoted in `PAPER_NOTES.md` survived solely in parquet.
- **Analysis choices are global.** Splits, layers, label rule, and α live in the
  phase config and are fixed before the phase runs — never tuned per checkpoint.
- **Split by prompt.** Rollouts of one problem never straddle train and val, and
  the split is a deterministic hash of `prompt_idx`, identical at every
  checkpoint even though the on-policy arm resamples rollouts.
- **Every figure ships its CSV.** `foo.pdf` always has `foo.csv` beside it, so a
  quoted number can never drift from the plot.
- **Interventions assert they fired.** A steering hook that silently no-ops
  produces a perfect null; `run_steering.py` raises instead.

## Two arms, and why both are needed

- **on_policy** — rollouts resampled from checkpoint *t*. What the monitor
  actually sees. Cannot separate "the representation moved" from "the policy
  writes different text now".
- **fixed_text** — the same text at every *t*, forward-passed through checkpoint
  *t*. Labels constant by construction, so movement is representation drift.
  The only arm where CKA is defined (it needs row correspondence).
