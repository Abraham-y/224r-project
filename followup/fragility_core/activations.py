"""Activation cache: write once per checkpoint, read many times.

Layout (followup/CLAUDE.md):

    acts/{run_id}/{checkpoint_step}/{layer}.npy      float32 (N, d_model)
    acts/{run_id}/{checkpoint_step}/labels.parquet   N rows, row-aligned with every .npy
    acts/{run_id}/{checkpoint_step}/manifest.json    provenance for this snapshot
    acts/{run_id}/manifest.json                      provenance for the whole ladder

`labels.parquet` row i corresponds to row i of every `{layer}.npy` in the same
directory. That invariant is checked on load; a mismatch raises rather than
silently truncating.

All probe and geometry analysis reads this cache. No analysis code re-runs the
model.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import paths

# Sibling-directory suffixes used by the write path. None of these are valid step
# names (`list_steps` accepts only all-digit names), so they never masquerade as
# snapshots.
_PARTIAL = ".partial"   # staging: the new snapshot, still being built
_ASIDE = ".aside."      # the previous snapshot, moved out of the way mid-swap
_ORPHAN = ".orphan."    # a preserved directory we were not permitted to delete


def run_dir(run_id: str) -> Path:
    return paths.acts_root() / run_id


def step_dir(run_id: str, step: int) -> Path:
    return run_dir(run_id) / str(int(step))


def acts_path(run_id: str, step: int, layer: int) -> Path:
    return step_dir(run_id, step) / f"{int(layer)}.npy"


def labels_path(run_id: str, step: int) -> Path:
    return step_dir(run_id, step) / "labels.parquet"


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


def save_snapshot(
    run_id: str,
    step: int,
    layer_to_X: dict[int, np.ndarray],
    labels: pd.DataFrame,
    *,
    manifest: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Persist one checkpoint's activations + labels.

    Refuses to clobber an existing snapshot unless `overwrite=True` — caches are
    expensive and a silent overwrite destroys the audit trail. "Existing
    snapshot" means a directory that holds at least one .npy; a directory with
    no .npy is a half-written leftover, and this function is allowed to get past
    it. It is NOT allowed to delete it: without `overwrite` the leftover is
    renamed to `{step}.orphan.{token}` and kept. (It used to be rmtree'd
    unconditionally, one line below a guard that had just decided not to
    overwrite anything.)

    The swap is done by rename, never by delete-then-rename:

        1. build the new snapshot in  {step}.partial
        2. move any existing {step} to {step}.aside.{token}
        3. rename {step}.partial -> {step}          <- the atomic step
        4. only now dispose of the aside directory

    Nothing is ever deleted before its replacement is fully written and in
    place. Step 2->3 is the one window where {step} does not exist; the old
    snapshot is intact in the aside directory throughout it, and `recover_run`
    (called at the top of this function and by `list_steps`) puts it back. The
    previous version deleted {step} and then renamed, so a crash in that window
    destroyed a valid snapshot outright, leaving only a `{step}.partial` that no
    reader looks at.
    """
    d = step_dir(run_id, step)
    recover_run(run_id)
    if d.exists() and any(d.glob("*.npy")) and not overwrite:
        raise FileExistsError(f"{d} already holds a snapshot; pass overwrite=True to replace")

    # Write to a sibling directory and swap it in. A crash midway through the
    # older in-place version left .npy files with no labels, and `list_steps`
    # (which only looked for a .npy) then reported the corrupt step as valid —
    # whereupon the analysis caught the missing labels and logged it as
    # "absent". A corrupt snapshot silently became a missing one.
    staging = d.with_name(d.name + _PARTIAL)
    if staging.exists():
        # Unverified leftover from an interrupted write. The real snapshot, if
        # there is one, is at `d` or in an aside directory recovered above.
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    n = len(labels)
    for layer, X in sorted(layer_to_X.items()):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        if X.shape[0] != n:
            shutil.rmtree(staging)
            raise ValueError(
                f"layer {layer}: {X.shape[0]} activation rows but {n} label rows"
            )
        np.save(staging / f"{int(layer)}.npy", X)

    labels.reset_index(drop=True).to_parquet(staging / "labels.parquet", index=False)

    meta = {
        "run_id": run_id,
        "checkpoint_step": int(step),
        "n_rows": int(n),
        "layers": sorted(int(k) for k in layer_to_X),
        "d_model": int(next(iter(layer_to_X.values())).shape[1]) if layer_to_X else None,
    }
    if manifest:
        meta.update(manifest)
    with open(staging / "manifest.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # Replace wholesale: `overwrite` must also clear stale layers from a previous
    # write with a different layer list, or `load_acts` would happily serve them.
    # Renaming the whole directory aside achieves that without deleting anything
    # until the replacement is in place.
    aside = None
    if d.exists():
        aside = d.with_name(f"{d.name}{_ASIDE}{uuid.uuid4().hex[:12]}")
        d.rename(aside)
    # One syscall takes `d` from absent to complete; there is no instant at which
    # a reader can see a partially populated `d`.
    staging.rename(d)
    if aside is not None:
        if overwrite:
            shutil.rmtree(aside)
        else:
            # Reached only when `d` existed with no .npy, i.e. a half-written
            # leftover. The caller did not authorise a delete, so keep it under a
            # name `recover_run` will not resurrect.
            aside.rename(d.with_name(f"{d.name}{_ORPHAN}{uuid.uuid4().hex[:12]}"))
    return d


def recover_run(run_id: str) -> list[int]:
    """Undo an interrupted swap. Returns the steps whose snapshot was restored.

    A `{step}.aside.{token}` directory exists only between steps 2 and 3 of
    `save_snapshot`, so finding one while `{step}` is missing means a write was
    interrupted in that window. The aside directory is the last known-good
    snapshot: rename it back.

    If `{step}` does exist, the swap completed and the aside is stale leftover
    from a crash between steps 3 and 4; it is removed, not restored.
    """
    d = run_dir(run_id)
    if not d.exists():
        return []
    restored: list[int] = []
    for child in sorted(d.glob(f"*{_ASIDE}*")):
        if not child.is_dir():
            continue
        name = child.name.split(_ASIDE, 1)[0]
        if not name.lstrip("-").isdigit():
            continue
        target = d / name
        if target.exists():
            shutil.rmtree(child)  # swap finished; this copy is superseded
        else:
            child.rename(target)
            restored.append(int(name))
    return sorted(restored)


def save_run_manifest(run_id: str, manifest: dict[str, Any]) -> Path:
    d = run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    out = d / "manifest.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return out


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


def list_steps(run_id: str, *, recover: bool = True) -> list[int]:
    d = run_dir(run_id)
    if not d.exists():
        return []
    if recover:
        # Enumeration is where an interrupted write would otherwise show up as a
        # silently missing checkpoint. No-op unless a `.aside.` directory exists.
        recover_run(run_id)
    steps = []
    for child in d.iterdir():
        if not (child.is_dir() and child.name.lstrip("-").isdigit()):
            continue
        # A snapshot counts as present only if it is COMPLETE. Requiring just a
        # .npy let a half-written directory pass as valid.
        if any(child.glob("*.npy")) and (child / "labels.parquet").exists():
            steps.append(int(child.name))
    return sorted(steps)


def list_layers(run_id: str, step: int) -> list[int]:
    d = step_dir(run_id, step)
    if not d.exists():
        return []
    return sorted(int(p.stem) for p in d.glob("*.npy") if p.stem.lstrip("-").isdigit())


def load_acts(run_id: str, step: int, layer: int) -> np.ndarray:
    p = acts_path(run_id, step, layer)
    if not p.exists():
        raise FileNotFoundError(f"no cached activations at {p}")
    return np.load(p)


def load_labels(run_id: str, step: int) -> pd.DataFrame:
    p = labels_path(run_id, step)
    if not p.exists():
        raise FileNotFoundError(f"no labels at {p}")
    return pd.read_parquet(p)


def load_snapshot(run_id: str, step: int, layer: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Activations + row-aligned labels, with the alignment invariant enforced."""
    X = load_acts(run_id, step, layer)
    y = load_labels(run_id, step)
    if X.shape[0] != len(y):
        raise ValueError(
            f"{run_id}/{step}/{layer}: {X.shape[0]} activation rows vs {len(y)} label rows"
        )
    return X, y


def load_manifest(run_id: str, step: int | None = None) -> dict[str, Any]:
    p = (step_dir(run_id, step) if step is not None else run_dir(run_id)) / "manifest.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def describe(run_id: str) -> pd.DataFrame:
    """One row per cached checkpoint: step, n_rows, layers, positive rate."""
    rows = []
    for step in list_steps(run_id):
        lab = load_labels(run_id, step)
        rec: dict[str, Any] = {
            "checkpoint_step": step,
            "n_rows": len(lab),
            "layers": ",".join(str(x) for x in list_layers(run_id, step)),
        }
        for rule in ("first_block", "last_block"):
            if rule in lab.columns:
                rec[f"pos_rate_{rule}"] = float(lab[rule].mean())
        rows.append(rec)
    return pd.DataFrame(rows)
