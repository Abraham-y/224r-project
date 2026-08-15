#!/usr/bin/env bash
# Every gate this project has, in one command. Run it after any edit to the paper
# or the analysis code, and especially after a prose rewrite -- rewriting is when
# numbers get retyped, and retyping is how every defect in the August audit got in.
#
#   bash scripts/check_everything.sh
#
# Exits non-zero if anything disagrees. Each verifier prints the published value
# beside the recomputed one, so a failure tells you which number moved.
#
# Requires the two Arm A/B rollout JSONs, which are ~10 MB each and not in git:
#   modal volume get default-proj-training \
#     evaluation/eval_results/armA_residual_step100.json ./eval_armA_residual_step100.json
#   modal volume get default-proj-training \
#     evaluation/eval_results/armB_surface_step100.json ./eval_armB_surface_step100.json

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

fail=0
run() {
  local name="$1"; shift
  printf '%-46s' "$name"
  if out=$("$@" 2>&1); then
    echo "PASS"
  else
    echo "FAIL"
    echo "$out" | tail -12 | sed 's/^/      /'
    fail=1
  fi
}

echo "=== analysis: does the code still produce the published numbers? ==="
run "structural baselines (selection, length)" \
    python -W ignore extension/probe/structural_baselines.py --out /tmp/_sb.txt
run "surface battery (the decomposition)" \
    python -W ignore extension/probe/surface_battery.py --out /tmp/_sbat.txt
run "template confound (section 3 table)" \
    python -W ignore scripts/quantify_structural_confound.py
run "pre-registered arms A/B + output shape" \
    python -W ignore extension/probe/verify_residual_arms.py
run "the 40-step lag + scope condition" \
    python followup/experiments/fragility/phase0_replicate/verify_lag_result.py
# Needs the cached activations under followup/acts/ (gitignored, ~56 MB/checkpoint).
if [ -d followup/acts/phase0_harvest_runA/50 ]; then
  run "change-point test (prompt-clustered)" \
      python -W ignore followup/experiments/fragility/phase0_replicate/changepoint_lag.py --n_boot 500 --out /tmp/_cp.txt
else
  printf '%-46s%s\n' "change-point test (prompt-clustered)" "SKIP (acts not cached)"
fi

echo
echo "=== submission: is the thing you are about to upload safe? ==="
if [ -f writeup_judge.tex ]; then
  run "anonymity scan of writeup_judge.tex" \
      python scripts/make_submission_tex.py --check writeup_judge.tex
else
  printf '%-46s%s\n' "anonymity scan of writeup_judge.tex" "SKIP (not built)"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "All gates pass."
else
  echo "SOMETHING DISAGREES -- see above. Do not submit until this is green."
fi
exit "$fail"
