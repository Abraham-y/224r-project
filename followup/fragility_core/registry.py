"""Experiment registry: config loading, content-hash run IDs, seeding.

Convention (followup/CLAUDE.md): every run gets a config YAML and a run ID that
is a content hash of that config. Nothing runs from hardcoded hyperparameters.

    cfg = load_config("configs/fragility/phase1_default.yaml")
    rid = run_id(cfg)                       # e.g. "phase1_default_9f2c1ab30e7d"
    register(cfg, phase="phase1", note="...")

The content hash covers the config AND the command-line arguments passed to
`register` as keyword extras, because those arguments select what a run computes
and where it writes.

Scripts whose artefact is named by something other than the hash (most of them:
`acts/{run_id}/` is keyed on `cfg["name"]` or a derived string) must pass that
name as `register(..., run_id=...)`, or `lookup()` cannot find the row again.

`register` appends one JSON line to results/fragility/registry.jsonl. The file is
append-only; nothing here ever rewrites or deletes a prior line.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import paths

# Keys excluded from the content hash: they describe *where* results go or how
# chatty the run is, not what is computed.
_HASH_EXCLUDE = frozenset({"note", "notes", "description", "out_dir", "verbose", "wandb"})

# Extras that NAME the run rather than describe the computation. Folding these
# into the hash would make the id depend on itself.
_EXTRA_HASH_EXCLUDE = frozenset({"run_id", "run_id_override", "run_id_source"})


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def hashable_extras(extra: dict[str, Any]) -> dict[str, Any]:
    """The subset of `register`'s keyword extras that belongs in the content hash.

    Callers pass two different kinds of thing through `**extra`: run INPUTS that
    came off the command line (`weights_step=0`, `text_steps="99"`,
    `ladder="phase0_harvest_runA"`, `steps=[0, 10, 20]`) and run OUTPUTS
    (`report=[...]`, `ladder={...summary...}`, `n_probes=15`). Inputs must change
    the id; outputs must not, or the id could not be computed before the run.

    The split is by shape: JSON scalars and flat sequences of scalars are treated
    as inputs; dicts and nested containers are treated as reports and dropped.
    That matches every call site in this repo — argparse produces exactly scalars
    and flat lists, while the summaries are dicts or lists of dicts.
    """
    out: dict[str, Any] = {}
    for k, v in extra.items():
        if k.startswith("_") or k in _HASH_EXCLUDE or k in _EXTRA_HASH_EXCLUDE:
            continue
        if _is_scalar(v):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(_is_scalar(x) for x in v):
            out[k] = list(v)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config. Relative paths resolve against configs/fragility/."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = paths.CONFIGS / p
    with open(p) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{p}: config must be a YAML mapping, got {type(cfg).__name__}")
    cfg.setdefault("name", p.stem)
    cfg["_config_path"] = str(p)
    return cfg


def canonical(cfg: dict[str, Any], extra: dict[str, Any] | None = None) -> str:
    """Deterministic JSON for hashing: sorted keys, private/excluded keys dropped.

    `extra` holds the command-line arguments that select what a run computes.
    They are nested under a reserved "__args__" key so an arg can never collide
    with (or silently shadow) a config key of the same name.
    """
    clean: dict[str, Any] = {
        k: v
        for k, v in cfg.items()
        if not k.startswith("_") and k not in _HASH_EXCLUDE
    }
    if extra:
        clean["__args__"] = dict(sorted(extra.items()))
    return json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: dict[str, Any], n: int = 12, *, extra: dict[str, Any] | None = None) -> str:
    return hashlib.sha256(canonical(cfg, extra).encode()).hexdigest()[:n]


def _content_run_id(cfg: dict[str, Any], extra: dict[str, Any] | None = None) -> str:
    return f"{cfg.get('name', 'run')}_{config_hash(cfg, extra=extra)}"


def run_id(cfg: dict[str, Any], *, extra: dict[str, Any] | None = None) -> str:
    """`{name}_{content-hash}`. Stable across machines; changes iff the inputs do.

    Pass the run's command-line arguments as `extra` when they change what is
    computed. Without them, `harvest_cross_cell --weights_step 0` and
    `--weights_step 50` hash identically while writing to different caches.
    """
    return _content_run_id(cfg, extra)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(paths.PAPER_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def register(
    cfg: dict[str, Any],
    phase: str,
    *,
    run_id: str | None = None,
    **extra: Any,
) -> str:
    """Append a registry row and return the run_id actually recorded.

    `run_id` names the artefact this row describes — the run_dir under `acts/`,
    the `run_id` column in the metric shards. Pass it whenever the artefact is
    named by something other than the config hash, which in this repo is most of
    the time: `harvest_ladder` writes to `cfg["name"]`, `harvest_cross_cell` to
    `f'{name}__frozen_w{weights_step}'`, `ingest_existing_cache` to
    `cfg.get("run_id")`. This function used to ignore any caller-supplied name
    and always record the content hash, so `lookup()` returned nothing for every
    cached run in this repo.

    Name resolution, first hit wins:
        1. `run_id=`            explicit argument
        2. `run_id_override=`   accepted alias (what the call sites pass today)
        3. `cfg["run_id"]`      config-declared name, as ingest_vanilla_ladder.yaml uses
        4. the content hash

    Keyword extras that look like command-line arguments (scalars and flat
    scalar sequences — see `hashable_extras`) are folded into `config_hash`, so
    two invocations that differ only in their arguments no longer share an id.
    Report- and summary-shaped extras are recorded but not hashed.
    """
    override = (
        run_id
        if run_id is not None
        else extra.get("run_id_override") or cfg.get("run_id")
    )
    args = hashable_extras(extra)
    rid = str(override) if override else _content_run_id(cfg, args)
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    # Extras go down first so they can never overwrite an identity field.
    row: dict[str, Any] = dict(extra)
    row.update(
        {
            "run_id": rid,
            "phase": phase,
            "name": cfg.get("name"),
            # Identifies the computation (config + args), independently of
            # whatever the artefact is named.
            "config_hash": config_hash(cfg, extra=args),
            "run_id_source": "override" if override else "config_hash",
            "hashed_args": args,
            "config_path": cfg.get("_config_path"),
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "git_sha": _git_sha(),
            "registered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    with open(paths.REGISTRY_PATH, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return rid


def lookup(run_id_: str) -> list[dict[str, Any]]:
    """All registry rows for a run_id, oldest first ([] if unknown)."""
    if not paths.REGISTRY_PATH.exists():
        return []
    rows = []
    with open(paths.REGISTRY_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_id") == run_id_:
                rows.append(row)
    return rows


def set_all_seeds(seed: int) -> None:
    """Seed python/numpy (and torch if importable). Call once per run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
