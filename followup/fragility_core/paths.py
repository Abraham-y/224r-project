"""Canonical paths for the follow-up project.

Two roots matter:

  PAPER_ROOT     the original CS 224R repo (read-mostly). Holds `evaluation/`,
                 `extension/`, `rloo_trainer/`, and the original cached
                 activations under `extension/cache/`.
  FOLLOWUP_ROOT  this project (`<PAPER_ROOT>/followup`). Everything we write
                 goes here.

Importing this module puts PAPER_ROOT on sys.path so that `evaluation.countdown`
and `extension.probe.*` import cleanly from follow-up code. That is the only
side effect.

Override the activation cache location with FRAGILITY_ACTS_DIR (useful on Modal,
where it should point into the mounted volume).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FOLLOWUP_ROOT = Path(__file__).resolve().parent.parent
PAPER_ROOT = FOLLOWUP_ROOT.parent

# Original-paper artefacts we read.
PAPER_EXTENSION = PAPER_ROOT / "extension"
PAPER_CACHE = PAPER_EXTENSION / "cache"
PAPER_STEERING = PAPER_CACHE / "steering"
PAPER_DATA = PAPER_EXTENSION / "data"

# Follow-up artefacts we write.
#
# RESULTS is env-overridable for the same reason the activation cache is: inside
# a Modal container the repo lives on ephemeral disk, so metrics and the
# append-only registry have to be steered onto the mounted volume or they vanish
# when the job exits.
CONFIGS = FOLLOWUP_ROOT / "configs" / "fragility"
RESULTS = Path(os.environ["FRAGILITY_RESULTS_DIR"]) if os.environ.get(
    "FRAGILITY_RESULTS_DIR"
) else (FOLLOWUP_ROOT / "results" / "fragility")
METRICS = RESULTS / "metrics"
FIGURES = RESULTS / "figures"
REGISTRY_PATH = RESULTS / "registry.jsonl"
EXPERIMENTS = FOLLOWUP_ROOT / "experiments" / "fragility"


def acts_root() -> Path:
    """Root of the activation cache. Env-overridable for Modal volumes."""
    env = os.environ.get("FRAGILITY_ACTS_DIR")
    return Path(env) if env else (FOLLOWUP_ROOT / "acts")


def ensure_paper_on_syspath() -> None:
    """Make `evaluation.*` / `extension.*` importable from follow-up code."""
    for p in (str(PAPER_ROOT), str(FOLLOWUP_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


def mkdirs() -> None:
    for p in (RESULTS, METRICS, FIGURES, acts_root()):
        p.mkdir(parents=True, exist_ok=True)


def commit_volume(tag: str = "") -> bool:
    """Flush the Modal volume so finished work survives a later crash/timeout.

    `modal_fragility._run` commits in a `finally`, i.e. once at job end, so a job
    killed by the Modal timeout can lose every checkpoint it finished. Long loops
    should call this after each unit of work, as `harvest_ladder` does.

    Returns True if a commit actually happened. No-op (returns False, prints why)
    outside a Modal container, so the same code runs locally.
    """
    try:
        import modal

        modal.Volume.from_name(
            os.environ.get("MODAL_VOLUME_NAME", "default-proj-training")
        ).commit()
        print(f"{tag}volume committed", flush=True)
        return True
    except Exception as exc:  # not on Modal, or the volume is unknown
        print(f"{tag}volume commit skipped ({exc})", flush=True)
        return False


ensure_paper_on_syspath()
