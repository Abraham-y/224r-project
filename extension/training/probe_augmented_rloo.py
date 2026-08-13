"""RLOO with a probe-augmented LOO baseline (lambda-mix control variate).

This is the principled use of a near-oracle correctness probe in policy
gradient: the verifier remains the REWARD (so no Goodhart -- the optimization
target is unchanged from vanilla RLOO), and the probe enters only through the
advantage BASELINE as a control variate. Specifically:

  A_i = R_i  -  (1 / (K-1)) * sum_{j != i}  [ lambda * R_j  +  (1 - lambda) * probe_j ]

  lambda = 1.0  -> vanilla RLOO (LOO-over-rewards baseline)
  lambda = 0.0  -> Anagha's pure probe-baseline (LOO-over-probe-values)
  lambda in (0,1) -> a smooth interpolation; the probe contributes some
                     variance reduction while the empirical reward keeps the
                     baseline calibrated to the actual reward scale.

Suggested sweep: lambda in {0.3, 0.5, 0.7}. The high-lambda end (closer to
vanilla) is safer when the probe's calibration is suspect; the low-lambda end
exploits the probe's near-oracle predictive power more aggressively.

Architecture mirrors firstanswer_rloo.py / probe_reward_rloo.py: this wrapper
just configures the probe-valuer and the lambda, then defers to rloo.py for
the actual RLOO loop. It uses Anagha's existing --probe_baseline plumbing in
rloo_trainer/rloo.py / rloo_update_worker.py and adds a single line of
arithmetic (the lambda-mix at baseline-construction time, see
rloo_update_worker.py:lambda-mix branch).

CLI (wrapper flags consumed here; everything else passes through to rloo.py):
  --probe PATH            precomputed probe pkl (Pipeline) or npz direction.
                          If omitted, the probe is trained at startup from
                          --train_rollouts.
  --probe_model PATH      frozen model used to extract </think> hidden states
                          (default: --model_name).
  --probe_layer N         hidden-state layer to read (default 16).
  --train_rollouts PATH   C_SFT rollouts JSONL for at-startup probe training
                          (default: eval_c_sft_n500.json).
  --probe_train_max N     cap on rollouts used to train the probe (default 3000).
  --lambda_mix FLOAT      0 = pure probe baseline, 1 = vanilla. (default 0.5)

Eval-sampler fix: the rloo.py the wrapper exec's uses the existing sampler
config; for downstream eval rollouts that use sample_local_jsonl.py, pass
--stop_strings '</answer>' to that script to match the training-time stop.

Example:
  modal run modal_train.py probe_augmented_rloo -- \\
      --wandb_project rloo_probe_aug_0.5b --wandb_name probe_aug_lam05_v1 \\
      --num_training_steps 100 --save_every_n_steps 10 \\
      --lambda_mix 0.5
"""

from __future__ import annotations

import os
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RLOO_DIR = os.path.join(_REPO_ROOT, "rloo_trainer")
_DATA_DIR = os.path.join(_REPO_ROOT, "extension", "data")
for path in (_RLOO_DIR, _REPO_ROOT, _DATA_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def _pop_value(name: str, default: str | None = None) -> str | None:
    out = []
    found = default
    skip = False
    for i, tok in enumerate(sys.argv):
        if skip:
            skip = False
            continue
        if tok == f"--{name}" and i + 1 < len(sys.argv):
            found = sys.argv[i + 1]
            skip = True
            continue
        out.append(tok)
    sys.argv[:] = out
    return found


def main() -> None:
    # Consume wrapper-only flags.
    probe_pkl     = _pop_value("probe", None)
    probe_model   = _pop_value("probe_model", None)
    probe_layer   = int(_pop_value("probe_layer", "16"))
    train_rolls   = _pop_value("train_rollouts", "eval_c_sft_n500.json")
    probe_max     = int(_pop_value("probe_train_max", "3000"))
    lambda_mix    = float(_pop_value("lambda_mix", "0.5"))

    # The plumbing for "compute probe per rollout and pass into update worker"
    # already exists via the --probe_baseline path in rloo.py. We turn it on and
    # pass lambda_mix through as an explicit CLI flag, which rloo.py forwards as
    # a constructor argument to the Ray actor.
    #
    # The env var is still set for older launchers, but it is NO LONGER the
    # transport: it used to be, and if it failed to reach the separately-spawned
    # Ray worker the run silently degraded to lambda=0 (the pure probe baseline
    # -- a different experiment) with no error. The worker now logs the lambda
    # it actually resolved, and its source, on the first update.
    os.environ["PROBE_AUG_LAMBDA"] = f"{lambda_mix:.4f}"
    if "--probe_aug_lambda" not in sys.argv:
        sys.argv += ["--probe_aug_lambda", f"{lambda_mix:.4f}"]

    # Now hand off the remaining argv to rloo.py with --probe_baseline enabled.
    # The probe-value piping (probe_value_model / probe_value_pkl etc.) is the
    # existing flag set; we map our wrapper names to those.
    if "--probe_baseline" not in sys.argv:
        sys.argv.append("--probe_baseline")
    if probe_pkl is not None:
        sys.argv += ["--probe_value_pkl", probe_pkl]
    if probe_model is not None:
        sys.argv += ["--probe_value_model", probe_model]
    sys.argv += ["--probe_value_layer", str(probe_layer)]
    sys.argv += ["--probe_value_rollouts", train_rolls]
    sys.argv += ["--probe_value_train_max", str(probe_max)]

    print(
        f"[probe_augmented_rloo] ACTIVE: A_i = R_i - LOO_{{j!=i}}["
        f"{lambda_mix:.3f} * R_j + {1 - lambda_mix:.3f} * probe_j]. "
        f"Reward unchanged (verifier).",
        flush=True,
    )

    rloo_path = os.path.join(_RLOO_DIR, "rloo.py")
    sys.argv[0] = rloo_path
    with open(rloo_path) as f:
        code = compile(f.read(), rloo_path, "exec")
    exec(code, {"__name__": "__main__", "__file__": rloo_path})


if __name__ == "__main__":
    main()
