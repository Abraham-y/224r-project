"""Checkpoint ladders: discovering existing ones, and densifying future runs.

Two jobs.

1. DISCOVERY. The original probe-as-reward runs (findings.md EXP-21, runA/runB)
   already wrote a checkpoint every 10 steps and those directories still exist on
   the Modal volume:

     checkpoints/rloo_probe_checkpoints/rloo_probe_0.5b/
       probe_rloo_runA_coutcome_FINAL/epoch_0_step_{0,10,...,90}/model
       probe_rloo_runB_csft_FINAL/epoch_0_step_{0,10,...,90}/model

   So Phase 0 does not need a fresh training run to get a ladder across the
   collapse — it needs activations harvested off the ladder that exists.
   `discover_ladder` (inside a Modal container, volume mounted) and
   `discover_ladder_via_cli` (from a laptop) enumerate it.

2. DENSIFICATION, for the runs Phase 3 will launch. `rloo.py` already supports
   `--save_every_n_steps`; K=3 over 100 steps yields 34 checkpoints, satisfying
   the 30-50 target with a flag rather than a code change. The only thing missing
   is disk hygiene: each archival directory also holds optimizer.pt + scheduler.pt
   (~2x the model), and rloo.py needs the PREVIOUS step's optimizer to resume, so
   they can only be pruned one step late. `prune_stale_optimizer_state` does that.
   `TrainLogWriter` records step / reward_mean / KL next to the ladder, since
   those otherwise live only in W&B.

Nothing here modifies rloo.py. Densification is applied by a wrapper script that
patches and then execs it, the same idiom the original repo already uses
(extension/training/probe_reward_rloo.py).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_CKPT_RE = re.compile(r"^epoch_(\d+)_step_(\d+)$")


@dataclass(frozen=True)
class Checkpoint:
    step: int
    epoch: int
    path: str          # directory containing model/ (+ optimizer.pt, scheduler.pt)
    model_path: str    # the HF model directory to load

    @property
    def name(self) -> str:
        return f"epoch_{self.epoch}_step_{self.step}"


def parse_checkpoint_name(name: str) -> tuple[int, int] | None:
    """"epoch_0_step_40" -> (epoch, step); None for anything else."""
    m = _CKPT_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def discover_ladder(run_root: str | Path, *, require_model: bool = True) -> list[Checkpoint]:
    """Enumerate epoch_*_step_* checkpoints under a run directory, step-ordered.

    Use inside a Modal container where the volume is mounted (e.g. /vol/...).
    `latest_checkpoint` is deliberately excluded: it is the rolling scratch
    directory, whose step number is not recoverable from the path.
    """
    root = Path(run_root)
    if not root.exists():
        raise FileNotFoundError(f"run root does not exist: {root}")
    out: list[Checkpoint] = []
    for child in sorted(root.iterdir()):
        parsed = parse_checkpoint_name(child.name)
        if parsed is None or not child.is_dir():
            continue
        epoch, step = parsed
        model_dir = child / "model"
        if require_model and not (model_dir / "config.json").exists():
            continue
        out.append(Checkpoint(step=step, epoch=epoch, path=str(child), model_path=str(model_dir)))
    return sorted(out, key=lambda c: c.step)


def discover_ladder_via_cli(
    volume: str,
    run_root: str,
    *,
    mount_prefix: str = "/vol",
) -> list[Checkpoint]:
    """Same, but from a laptop via `modal volume ls` (read-only, no GPU).

    `run_root` is volume-relative (no leading /vol). Paths in the returned
    Checkpoints are rewritten to `mount_prefix` so they are directly usable by a
    job running inside the container.
    """
    cmd = ["modal", "volume", "ls", volume, run_root]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"`{' '.join(cmd)}` failed:\n{res.stderr.strip()}")
    out: list[Checkpoint] = []
    for line in res.stdout.splitlines():
        entry = line.strip()
        if not entry:
            continue
        name = Path(entry).name
        parsed = parse_checkpoint_name(name)
        if parsed is None:
            continue
        epoch, step = parsed
        full = f"{mount_prefix.rstrip('/')}/{entry.lstrip('/')}"
        out.append(Checkpoint(step=step, epoch=epoch, path=full, model_path=f"{full}/model"))
    return sorted(out, key=lambda c: c.step)


def resolve_ladder(ladder_cfg: dict[str, Any]) -> list[Checkpoint]:
    """Full ladder from a config's `ladder:` block, including the final checkpoint.

    `discover_ladder` deliberately skips `latest_checkpoint` because its step
    number is not recoverable from the path. Every caller then needs the same
    fixup — map it to `final_step` — and when only one caller had it, Phase 2
    silently found zero checkpoints for step 99 and wrote no rows at all. Shared
    here so the special case exists exactly once.
    """
    run_root = f"{ladder_cfg['mount_prefix'].rstrip('/')}/{ladder_cfg['run_root']}"
    ladder = discover_ladder(run_root)
    if ladder_cfg.get("include_final"):
        final = Path(run_root) / "latest_checkpoint"
        step = int(ladder_cfg.get("final_step", 99))
        if (final / "model" / "config.json").exists():
            if any(c.step == step for c in ladder):
                raise ValueError(
                    f"final_step {step} collides with an existing epoch_*_step_{step} "
                    "directory; pick a distinct final_step"
                )
            ladder.append(
                Checkpoint(step=step, epoch=0, path=str(final),
                           model_path=str(final / "model"))
            )
    return sorted(ladder, key=lambda c: c.step)


def ladder_summary(ladder: Iterable[Checkpoint]) -> dict[str, Any]:
    steps = [c.step for c in ladder]
    if not steps:
        return {"n_checkpoints": 0, "steps": []}
    gaps = sorted({b - a for a, b in zip(steps, steps[1:])})
    return {
        "n_checkpoints": len(steps),
        "steps": steps,
        "step_min": min(steps),
        "step_max": max(steps),
        "step_gaps": gaps,
    }


# ---------------------------------------------------------------------------
# densification for future runs
# ---------------------------------------------------------------------------


def prune_stale_optimizer_state(run_root: str | Path, *, keep_last: int = 2) -> list[str]:
    """Delete optimizer.pt / scheduler.pt from all but the newest `keep_last` steps.

    rloo.py resumes the optimizer from the previous step's directory, so the two
    newest must stay intact. Archival checkpoints only ever get loaded for
    inference, where optimizer state is dead weight (~2x the model per step).

    Never touches model/ and never removes a checkpoint directory. Returns the
    files deleted.
    """
    ladder = discover_ladder(run_root, require_model=False)
    removed: list[str] = []
    for ckpt in ladder[: max(len(ladder) - keep_last, 0)]:
        for fname in ("optimizer.pt", "scheduler.pt"):
            f = Path(ckpt.path) / fname
            if f.exists():
                f.unlink()
                removed.append(str(f))
    return removed


def dense_checkpoint_argv(argv: list[str], *, every_k: int) -> list[str]:
    """Force --save_every_n_steps to K, replacing any existing value."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--save_every_n_steps":
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out + ["--save_every_n_steps", str(int(every_k))]


class TrainLogWriter:
    """Append-only JSONL of per-step training scalars alongside the ladder.

    W&B holds these already, but a ladder that can only be interpreted with a
    live W&B account is not reproducible. One line per step:
    {step, reward_mean, kl, probe_mean, verifier_mean, ...}
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step: int, **scalars: Any) -> None:
        row = {"step": int(step)}
        for k, v in scalars.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                row[k] = v
        with open(self.path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(l) for l in f if l.strip()]


def patch_wandb_log_to_jsonl(writer: TrainLogWriter) -> None:
    """Mirror every self.wandb.log() call into `writer`.

    rloo.py logs through `self.wandb.log`, i.e. the bound Run.log method — so the
    patch target is `wandb.sdk.wandb_run.Run.log`, not the module-level
    `wandb.log`. (findings.md EXP-21 bug #4 was exactly this mistake.)
    """
    import wandb
    from wandb.sdk.wandb_run import Run

    orig = Run.log

    def log(self, data, step=None, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            scalars = {
                k: v for k, v in data.items() if isinstance(v, (int, float)) and k != "_step"
            }
            writer.log(step if step is not None else -1, **scalars)
        except Exception as exc:  # never let logging break training
            print(f"[checkpoint_logging] mirror failed: {exc}", flush=True)
        # `*args` must precede the `step=` keyword. Written the other way round,
        # any positional argument after `data` binds to `step` positionally AND
        # is passed again as a keyword -> TypeError: got multiple values for
        # argument 'step', inside a wrapper whose whole job is to be invisible.
        return orig(self, data, *args, step=step, **kwargs)

    Run.log = log
    print(f"[checkpoint_logging] mirroring wandb scalars to {writer.path}", flush=True)


def copy_model_only(src_ckpt_dir: str | Path, dst_dir: str | Path) -> Path:
    """Copy just the model/ subtree of a checkpoint (for cheap archival copies)."""
    src = Path(src_ckpt_dir) / "model"
    dst = Path(dst_dir) / "model"
    if not src.exists():
        raise FileNotFoundError(f"no model/ under {src_ckpt_dir}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return dst
