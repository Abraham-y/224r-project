"""Modal launcher for the probe-fragility follow-up.

Separate from the original `modal_train.py` on purpose: followup/CLAUDE.md says
not to modify the paper's scripts in place, and the original launcher's function
list is part of the published artefact.

The image mounts the WHOLE repo (the follow-up imports `evaluation.countdown`,
`extension.probe.cache_hidden_states`, and `extension/evaluation/sample_local_jsonl.py`
from it) and reuses `modal_requirements.txt` unchanged.

    modal run --detach modal_fragility.py harvest  -- --config phase0_harvest_runA.yaml
    modal run --detach modal_fragility.py steering -- --config phase2_default.yaml --ladder runA
    modal run --detach modal_fragility.py patching -- --config phase2_default.yaml --ladder runA
    modal run          modal_fragility.py analyse  -- --config phase1_default.yaml --run_id phase0_harvest_runA

`analyse` needs no GPU; it is here only so Phase 1 can run against activations
that live on the volume without pulling them to a laptop first.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).resolve().parent.parent   # the original repo root
REMOTE_ROOT = "/root/default_proj"
REMOTE_VOL = "/vol"
FOLLOWUP = f"{REMOTE_ROOT}/followup"

APP_NAME = os.environ.get("MODAL_APP_NAME", "default-proj-training")
GPU_CONFIG = os.environ.get("MODAL_GPU", "H100!")
TIMEOUT_SECONDS = int(os.environ.get("MODAL_TIMEOUT_SECONDS", "86400"))
STARTUP_TIMEOUT_SECONDS = int(os.environ.get("MODAL_STARTUP_TIMEOUT_SECONDS", "1800"))
CPU_COUNT = int(os.environ.get("MODAL_CPU_COUNT", "8"))
VOLUME_NAME = os.environ.get("MODAL_VOLUME_NAME", "default-proj-training")
PIP_EXTRA_INDEX_URL = os.environ.get(
    "MODAL_PIP_EXTRA_INDEX_URL", "https://download.pytorch.org/whl/cu128"
)

TRAINING_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _secrets() -> list[modal.Secret]:
    vals = {
        k: os.environ[k]
        for k in ("HF_TOKEN", "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_USERNAME", "WANDB_USER_EMAIL")
        if os.environ.get(k)
    }
    return [modal.Secret.from_dict(vals)] if vals else []


image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_dir(str(LOCAL_ROOT), remote_path=REMOTE_ROOT, copy=True)
    .run_commands(
        f"cd {shlex.quote(REMOTE_ROOT)} && python -m pip install --upgrade "
        "pip==25.3 setuptools==80.10.2 wheel==0.46.3",
        f"cd {shlex.quote(REMOTE_ROOT)} && python -m pip install "
        f"--extra-index-url {shlex.quote(PIP_EXTRA_INDEX_URL)} "
        f"-r {shlex.quote(REMOTE_ROOT)}/modal_requirements.txt",
        f"cd {shlex.quote(REMOTE_ROOT)} && python -m pip install --no-deps -e .",
    )
)

app = modal.App(APP_NAME, image=image)


def _run(script: str, args: list[str]) -> str:
    vol = Path(REMOTE_VOL)
    (vol / "cache" / "huggingface" / "datasets").mkdir(parents=True, exist_ok=True)
    (vol / "fragility").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    hf_home = vol / "cache" / "huggingface"
    env.setdefault("HF_HOME", str(hf_home))
    env.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Unbuffered stdout. Without it Python buffers when stdout is not a tty, so a
    # multi-hour job's progress lines only appear when the buffer fills or the
    # process exits — which makes a hung job indistinguishable from a quiet one.
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Activations, metrics and the registry land on the volume. Without this the
    # container's repo copy is ephemeral and every artefact dies with the job.
    env.setdefault("FRAGILITY_ACTS_DIR", str(vol / "fragility" / "acts"))
    env.setdefault("FRAGILITY_RESULTS_DIR", str(vol / "fragility" / "results"))

    cmd = ["python", script, *args]
    print(f"Executing: {shlex.join(cmd)}")
    print(f"Artefacts persist to Modal volume {VOLUME_NAME!r} under {REMOTE_VOL}/fragility.")
    try:
        subprocess.run(cmd, cwd=FOLLOWUP, env=env, check=True)
    finally:
        TRAINING_VOLUME.commit()
    return f"Finished {script}."


_GPU_KW = dict(
    gpu=GPU_CONFIG,
    cpu=CPU_COUNT,
    timeout=TIMEOUT_SECONDS,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    volumes={REMOTE_VOL: TRAINING_VOLUME},
    secrets=_secrets(),
)
_CPU_KW = dict(
    cpu=CPU_COUNT,
    timeout=TIMEOUT_SECONDS,
    volumes={REMOTE_VOL: TRAINING_VOLUME},
    secrets=_secrets(),
)


@app.function(**_GPU_KW)
def run_harvest(args: list[str]) -> str:
    """Phase 0: sample + cache activations across an existing checkpoint ladder."""
    return _run("experiments/fragility/phase0_replicate/harvest_ladder.py", args)


@app.function(**_GPU_KW)
def run_cross_cell(args: list[str]) -> str:
    """The missing weights x text cell: step-0 weights on step-t text."""
    return _run("experiments/fragility/phase0_replicate/harvest_cross_cell.py", args)


@app.function(**_GPU_KW)
def run_steering(args: list[str]) -> str:
    """Phase 2: activation-addition steering per checkpoint."""
    return _run("experiments/fragility/phase2_causal_status/run_steering.py", args)


@app.function(**_GPU_KW)
def run_patching(args: list[str]) -> str:
    """Phase 2: activation patching between correct and incorrect runs."""
    return _run("experiments/fragility/phase2_causal_status/run_patching.py", args)


@app.function(**_GPU_KW)
def run_dense_probe_rloo(args: list[str]) -> str:
    """Phase 3: probe-RL with a dense checkpoint ladder (wraps probe_rloo.py)."""
    return _run("experiments/fragility/phase3_predict_robustness/dense_probe_rloo.py", args)


@app.function(**_GPU_KW)
def run_residual_rl(args: list[str]) -> str:
    """Probe-as-reward RL with the surface-residualised probe (text-aware fork)."""
    return _run("experiments/fragility/residual_probe/launch_residual_rl.py", args)


@app.function(**_GPU_KW)
def run_eval_local(args: list[str]) -> str:
    """Eval a checkpoint on the paper's local 500-prompt file (forked harness)."""
    return _run("experiments/fragility/residual_probe/countdown_eval_local.py", args)


@app.function(**_CPU_KW)
def run_analyse(args: list[str]) -> str:
    """Phase 1 analysis against volume-resident activations. No GPU."""
    return _run("experiments/fragility/phase1_evasion_vs_corruption/run_phase1.py", args)


_TARGETS = {
    "harvest": run_harvest,
    "cross_cell": run_cross_cell,
    "steering": run_steering,
    "patching": run_patching,
    "dense_probe_rloo": run_dense_probe_rloo,
    "residual_rl": run_residual_rl,
    "eval_local": run_eval_local,
    "analyse": run_analyse,
}


@app.local_entrypoint()
def main(*raw: str) -> None:
    ap = argparse.ArgumentParser(description="Launch a follow-up job on Modal.")
    ap.add_argument("job", choices=sorted(_TARGETS))
    ap.add_argument("job_args", nargs=argparse.REMAINDER)
    args = ap.parse_args(list(raw))

    job_args = list(args.job_args)
    if job_args[:1] == ["--"]:
        job_args = job_args[1:]

    fn = _TARGETS[args.job]
    if args.job == "analyse":
        print(fn.remote(job_args))
        return
    call = fn.spawn(job_args)
    print(f"Spawned {args.job}. Function call ID: {call.object_id}")
    print("Runs on Modal even if this client disconnects.")
