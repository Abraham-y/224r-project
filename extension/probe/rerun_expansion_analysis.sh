#!/bin/bash
# Re-run the full probe-analysis pipeline on the n=500 expanded eval set.
# Prereq: Phase 2 cache jobs have completed and their outputs have been
# downloaded into:
#   extension/cache/probe_cache_n500/         (C_SFT, C_outcome at n=500)
#   extension/cache/probe_cache_dynamics_optB/ (C_outcome_step_{30,60,90} at n=200)
# Plus eval JSONs locally:
#   eval_c_sft_n500.json
#   eval_c_outcome_n500.json
#   eval_c_outcome_step_{30,60,90}_n200.json
#
# All analyses are local (CPU); takes ~5-10 min total.
# Outputs go to extension/outputs/n500/*.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "${PROJECT_ROOT}"

CACHE="extension/cache/probe_cache_n500"
DYN_CACHE="extension/cache/probe_cache_dynamics_optB"
SFT_EVAL="eval_c_sft_n500.json"
OUT_EVAL="eval_c_outcome_n500.json"
OUT_DIR="extension/outputs/n500"
mkdir -p "${OUT_DIR}/figures" "${OUT_DIR}/text"

echo "=============================================================="
echo "[1/8] analyze_probes (per-cell AUROCs, GroupKFold by prompt)"
echo "=============================================================="
python3 extension/probe/analyze_probes.py \
    --cache_dir "${CACHE}" \
    | tee "${OUT_DIR}/text/01_analyze_probes.txt"

echo
echo "=============================================================="
echo "[2/8] robustness_probes (balanced subsample, regularization scan)"
echo "=============================================================="
python3 extension/probe/robustness_probes.py \
    --cache_dir "${CACHE}" \
    | tee "${OUT_DIR}/text/02_robustness_probes.txt"

echo
echo "=============================================================="
echo "[3/8] deeper_analyses (multi-layer, multi-position)"
echo "=============================================================="
python3 extension/probe/deeper_analyses.py \
    --cache_dir "${CACHE}" \
    | tee "${OUT_DIR}/text/03_deeper_analyses.txt"

echo
echo "=============================================================="
echo "[4/8] qualitative_matched_pairs (within-problem matched-pair)"
echo "=============================================================="
python3 extension/probe/qualitative_matched_pairs.py \
    --cache_dir "${CACHE}" \
    --layer 16 \
    --sft_eval "${SFT_EVAL}" \
    --outcome_eval "${OUT_EVAL}" \
    | tee "${OUT_DIR}/text/04_matched_pairs.txt"

echo
echo "=============================================================="
echo "[5/8] cross_checkpoint_transfer (transfer matrix)"
echo "=============================================================="
python3 extension/probe/cross_checkpoint_transfer.py \
    --cache_dir "${CACHE}" \
    --layer 16 \
    | tee "${OUT_DIR}/text/05_cross_checkpoint_transfer.txt"

echo
echo "=============================================================="
echo "[6/8] significance_and_baselines (Wilcoxon, MW-U, paired t, RF/MLP)"
echo "=============================================================="
python3 extension/probe/significance_and_baselines.py \
    --cache_dir "${CACHE}" \
    --layer 16 \
    | tee "${OUT_DIR}/text/06_significance_baselines.txt"

echo
echo "=============================================================="
echo "[7/8] length_matched_transfer control (with n=500)"
echo "=============================================================="
python3 extension/probe/length_matched_transfer.py \
    --cache_dir "${CACHE}" \
    --layer 16 \
    --sft_eval "${SFT_EVAL}" \
    --outcome_eval "${OUT_EVAL}" \
    | tee "${OUT_DIR}/text/07_length_matched_transfer.txt"

echo
echo "=============================================================="
echo "[8/8] make_figures (fig1..fig7, fig8 done separately)"
echo "=============================================================="
python3 extension/probe/make_figures.py \
    --cache_dir "${CACHE}" \
    --layer 16 \
    --out_dir "${OUT_DIR}/figures" \
    --sft_confidence extension/cache/confidence/C_SFT_confidence.jsonl \
    --outcome_confidence extension/cache/confidence/C_outcome_confidence.jsonl \
    2>&1 | tee "${OUT_DIR}/text/08_make_figures.txt" || echo "[warn] make_figures had errors -- check log"

echo
echo "All analyses complete. Outputs in ${OUT_DIR}/."
