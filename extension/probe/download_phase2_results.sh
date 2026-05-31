#!/bin/bash
# Pull Phase 2 cache files + Phase 1 eval JSONs from the Modal volume to
# the local repo so the analysis scripts can read them.
#
# Modal vol structure (default-proj-training):
#   /probe_cache_n500/*.{npz,meta.json}                 (90 files)
#   /probe_cache_dynamics_optB/*.{npz,meta.json}         (54 files)
#   /evaluation/eval_results/eval_c_*_n*.json            (5 files)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VOL="${MODAL_VOLUME_NAME:-default-proj-training}"

CACHE_LOCAL="${PROJECT_ROOT}/extension/cache"

mkdir -p "${CACHE_LOCAL}"

echo "Downloading probe_cache_n500/ ..."
modal volume get --force "${VOL}" probe_cache_n500/ "${CACHE_LOCAL}/"

echo "Downloading probe_cache_dynamics_optB/ ..."
modal volume get --force "${VOL}" probe_cache_dynamics_optB/ "${CACHE_LOCAL}/"

echo "Downloading expansion eval JSONs ..."
for fname in eval_c_sft_n500 eval_c_outcome_n500 \
             eval_c_outcome_step_30_n200 eval_c_outcome_step_60_n200 \
             eval_c_outcome_step_90_n200; do
    src="evaluation/eval_results/${fname}.json"
    dest="${PROJECT_ROOT}/${fname}.json"
    echo "  ${VOL}:${src} -> ${dest}"
    modal volume get --force "${VOL}" "${src}" "${dest}"
done

echo
echo "Local files now present:"
ls -lh "${CACHE_LOCAL}" | head -10
echo "..."
ls -lh "${PROJECT_ROOT}" | grep "eval_c_"
