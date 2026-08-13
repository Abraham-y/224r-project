"""Phase 3 — probe-as-reward RLOO with a DENSE checkpoint ladder.

Wraps `extension/training/probe_rloo.py` without editing it, using the repo's
own patch-then-exec idiom. Adds three things and changes nothing else:

  1. `--save_every_n_steps K` forced to the dense value (K=3 over 100 steps
     gives 34 checkpoints, the CLAUDE.md 30-50 target);
  2. every W&B scalar mirrored to `train_log.jsonl` beside the ladder, so the
     run is interpretable without a live W&B account;
  3. optimizer/scheduler state pruned from archival checkpoints one step behind
     the front, which is what makes K=3 affordable on disk. rloo.py resumes the
     optimizer from the PREVIOUS directory, so the two newest are never touched.

The reward, the model, and the optimisation are byte-for-byte `probe_rloo.py`.
That is the point: the probe under attack is the only thing that varies across
the zoo.

    modal run --detach modal_fragility.py dense_probe_rloo -- \
        --dense_every 3 \
        --probe_pkl /vol/fragility/probe_zoo/L16_pca8.pkl \
        --model_name asingh15/qwen-sft-countdown-defaultproj \
        --wandb_project rloo_zoo_0.5b --wandb_name zoo_L16_pca8 \
        --save_dir /vol/checkpoints/rloo_zoo_checkpoints \
        --batch_size 128 --group_size 8 --gradient_accumulation_steps 128 \
        --num_training_steps 100 --warmup_ratio 0 --lr_schedule constant
"""

from __future__ import annotations

import atexit
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fragility_core import checkpoint_logging, paths  # noqa: E402

PAPER_ROOT = paths.PAPER_ROOT
TARGET = PAPER_ROOT / "extension" / "training" / "probe_rloo.py"


def _pop(name: str, default: str | None = None) -> str | None:
    out, val, skip = [], default, False
    for i, tok in enumerate(sys.argv):
        if skip:
            skip = False
            continue
        if tok == f"--{name}" and i + 1 < len(sys.argv):
            val, skip = sys.argv[i + 1], True
            continue
        out.append(tok)
    sys.argv[:] = out
    return val


def _peek(name: str, default: str | None = None) -> str | None:
    for i, tok in enumerate(sys.argv):
        if tok == f"--{name}" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main() -> None:
    every = int(_pop("dense_every", "3"))
    prune = _pop("prune_optimizer", "1") != "0"

    sys.argv[:] = checkpoint_logging.dense_checkpoint_argv(sys.argv, every_k=every)

    save_dir = _peek("save_dir", "/vol/checkpoints/rloo_zoo_checkpoints")
    project = _peek("wandb_project", "rloo_zoo_0.5b")
    name = _peek("wandb_name", "zoo")
    run_root = Path(save_dir) / project / name
    run_root.mkdir(parents=True, exist_ok=True)

    writer = checkpoint_logging.TrainLogWriter(run_root / "train_log.jsonl")
    checkpoint_logging.patch_wandb_log_to_jsonl(writer)

    print(
        f"[dense_probe_rloo] ACTIVE: save_every_n_steps={every} "
        f"(~{100 // every + 1} checkpoints over 100 steps); "
        f"train log -> {writer.path}; optimizer pruning={'on' if prune else 'off'}",
        flush=True,
    )

    if prune:
        # Runs on exit (including on failure): reclaims optimizer/scheduler state
        # from every archival checkpoint except the two newest.
        def _cleanup() -> None:
            try:
                removed = checkpoint_logging.prune_stale_optimizer_state(run_root, keep_last=2)
                print(f"[dense_probe_rloo] pruned {len(removed)} optimizer/scheduler files",
                      flush=True)
            except Exception as exc:
                print(f"[dense_probe_rloo] prune skipped: {exc}", flush=True)

        atexit.register(_cleanup)

    if not TARGET.exists():
        raise SystemExit(f"cannot find {TARGET}")
    sys.argv[0] = str(TARGET)
    os.chdir(PAPER_ROOT)
    with open(TARGET) as f:
        code = compile(f.read(), str(TARGET), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(TARGET)})


if __name__ == "__main__":
    main()
