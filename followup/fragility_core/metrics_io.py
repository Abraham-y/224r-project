"""Append-only parquet metric store.

Schema (followup/CLAUDE.md mandates the first six columns; the rest are added
dimensions that the phase definitions require — a metric is only meaningful
given its layer / arm / probe):

    run_id           str
    phase            str
    checkpoint_step  int64  (nullable: -1 for run-level metrics)
    metric           str
    value            float64 (NaN = measured-and-undefined; absent rows = not measured)
    seed             int64
    layer            int64  (nullable, -1 when not layer-specific)
    arm              str    ("on_policy" | "fixed_text" | "" )
    probe_id         str    (which probe the metric is about; "" when N/A)
    extra_json       str    (free-form provenance; never parsed by figures)

Every `flush()` writes a NEW file. Nothing is ever overwritten or deleted, so a
re-run of an analysis leaves the earlier numbers auditable.

Missing measurements are represented by ABSENT ROWS, not interpolation. If a
checkpoint has no activation cache, no row is written for it and the figures
show a gap.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from . import paths

COLUMNS = [
    "run_id",
    "phase",
    "checkpoint_step",
    "metric",
    "value",
    "seed",
    "layer",
    "arm",
    "probe_id",
    "extra_json",
    # Real write time, stamped in `flush()` and therefore identical for every row
    # in a shard. `latest()` used to rely on shard filename order, but the shard
    # counter reuses freed indices: delete any shard and the next write takes its
    # number, which sorts BEFORE later shards, so "keep last" would retain a
    # stale row. Filenames also collide when the Modal-side and local-side
    # results directories are merged.
    #
    # It is stamped at flush, not at add(), because add() times are spread over
    # however long the job ran. Two overlapping re-runs then interleave: a row
    # added early by the LATER job carries an older stamp than a row added late
    # by the EARLIER job, and `latest()` keeps the stale one. Shard-uniform
    # timestamps make the comparison "which analysis finished later", which is
    # the question `latest()` is actually asking.
    "written_utc",
]

_NULL_STEP = -1
_NULL_LAYER = -1


class MetricWriter:
    """Accumulate rows in memory, then write one parquet file per flush."""

    def __init__(self, run_id: str, phase: str, seed: int):
        self.run_id = run_id
        self.phase = phase
        self.seed = int(seed)
        self._rows: list[dict[str, Any]] = []
        self._last_flush: datetime | None = None

    def add(
        self,
        metric: str,
        value: float | None,
        *,
        checkpoint_step: int | None = None,
        layer: int | None = None,
        arm: str = "",
        probe_id: str = "",
        **extra: Any,
    ) -> None:
        self._rows.append(
            {
                "run_id": self.run_id,
                "phase": self.phase,
                "checkpoint_step": _NULL_STEP if checkpoint_step is None else int(checkpoint_step),
                "metric": metric,
                "value": float("nan") if value is None else float(value),
                "seed": self.seed,
                "layer": _NULL_LAYER if layer is None else int(layer),
                "arm": arm or "",
                "probe_id": probe_id or "",
                "extra_json": json.dumps(extra, sort_keys=True, default=str) if extra else "",
                # written_utc is deliberately NOT set here; flush() stamps it.
            }
        )

    def add_many(self, metrics: dict[str, float | None], **kwargs: Any) -> None:
        for name, value in metrics.items():
            self.add(name, value, **kwargs)

    def __len__(self) -> int:
        return len(self._rows)

    def to_frame(self, written_utc: str | None = None) -> pd.DataFrame:
        """Pending rows. `written_utc` is NaN unless a stamp is supplied — the
        real stamp is applied by `flush()`, which is the moment the rows exist
        for any other process to read."""
        df = pd.DataFrame(self._rows, columns=COLUMNS)
        df["written_utc"] = pd.NA if written_utc is None else written_utc
        return df

    def _stamp(self) -> str:
        """Flush timestamp, strictly increasing within this writer.

        Microsecond resolution makes a collision between two flushes unlikely but
        not impossible (a flush of a handful of rows can be fast). Two shards
        sharing a stamp would leave `latest()` to break the tie on concat order,
        which is random — shard filenames carry a uuid.
        """
        now = datetime.now(timezone.utc)
        if self._last_flush is not None and now <= self._last_flush:
            now = self._last_flush + timedelta(microseconds=1)
        self._last_flush = now
        return now.isoformat(timespec="microseconds")

    def flush(self) -> Path | None:
        """Write accumulated rows to a fresh parquet file. Returns its path.

        Every row in the shard is stamped with this flush's time, not the time
        `add()` happened to be called (see COLUMNS).
        """
        if not self._rows:
            return None
        paths.METRICS.mkdir(parents=True, exist_ok=True)
        # Unique suffix rather than a reused counter: two results directories
        # (Modal volume + local) can be merged without name collisions, and a
        # deleted shard never has its index recycled.
        stem = f"{self.run_id}__{self.phase}__seed{self.seed}"
        out = paths.METRICS / f"{stem}__{uuid.uuid4().hex[:12]}.parquet"
        self.to_frame(self._stamp()).to_parquet(out, index=False)
        self._rows.clear()
        return out


def write_table(df: pd.DataFrame, path: Path, *, index: bool = True) -> Path:
    """Write a derived CSV to `path` AND keep an immutable archival copy.

    `results/` is documented as append-only, and the parquet metric store is.
    The derived CSVs were not: `transfer_matrix.csv`, `scope_check.csv` and
    `prehoc_features.csv` were rewritten in place on every run, so the on-policy
    transfer rows quoted in PAPER_NOTES survived only in the parquet store while
    the CSV on disk held a later fixed-text-only run. The canonical path stays
    (readers and figures reference it); the archival sibling makes the previous
    contents recoverable, which is what append-only is for.

    Archive layout: `<stem>.archive/<stem>__<utc>__<uuid>.csv`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    archive = path.parent / f"{path.stem}.archive"
    archive.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    df.to_csv(archive / f"{path.stem}__{stamp}__{uuid.uuid4().hex[:8]}{path.suffix}", index=index)
    df.to_csv(path, index=index)
    return path


def read_metrics(
    *,
    phase: str | None = None,
    run_id: str | None = None,
    metrics: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Concatenate every parquet shard, optionally filtered. Figures read only this."""
    if not paths.METRICS.exists():
        return pd.DataFrame(columns=COLUMNS)
    frames = [pd.read_parquet(p) for p in sorted(paths.METRICS.glob("*.parquet"))]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    if phase is not None:
        df = df[df["phase"] == phase]
    if run_id is not None:
        df = df[df["run_id"] == run_id]
    if metrics is not None:
        df = df[df["metric"].isin(list(metrics))]
    return df.reset_index(drop=True)


def latest(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate to the last-written row per measurement key.

    Shards are append-only, so a re-analysis produces a second row for the same
    key. Figures want the newest, decided by the recorded write timestamp — not
    by filename order, which is not reliably monotonic (see COLUMNS).
    """
    key = ["run_id", "phase", "checkpoint_step", "metric", "seed", "layer", "arm", "probe_id"]
    if df.empty:
        return df.reset_index(drop=True)
    if "written_utc" in df.columns and df["written_utc"].notna().any():
        # Rows from before this column existed sort first, which is correct.
        df = df.sort_values("written_utc", kind="stable", na_position="first")
    return df.drop_duplicates(subset=key, keep="last").reset_index(drop=True)


def pivot_curve(
    df: pd.DataFrame,
    metric_names: Iterable[str],
    *,
    layer: int | None = None,
    arm: str | None = None,
) -> pd.DataFrame:
    """checkpoint_step-indexed wide frame of the named metrics (for plotting)."""
    d = latest(df)
    d = d[d["metric"].isin(list(metric_names))]
    if layer is not None:
        d = d[d["layer"] == layer]
    if arm is not None:
        d = d[d["arm"] == arm]
    if d.empty:
        return pd.DataFrame()
    return (
        d.pivot_table(index="checkpoint_step", columns="metric", values="value", aggfunc="mean")
        .sort_index()
    )
