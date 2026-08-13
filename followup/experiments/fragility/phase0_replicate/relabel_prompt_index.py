"""Repair the prompt-index namespace in caches harvested before the index fix.

Early harvests numbered prompts 0..405 (their position in the contamination-
filtered subset) instead of 0..499 (their index in the original eval file). The
sampler rebuilds output rows from a fixed field list, so the original index that
`build_prompt_subset` attached at the top level was silently dropped, and
`_orig_index` fell back to the local position.

Nothing about the ACTIVATIONS or the rollout text is wrong — only the label
column. And the mapping is exactly recoverable, because the subset is a
deterministic filter that preserves order: local index i is the i-th clean
prompt. Verified by prompt-text equality before anything is written.

This matters because `labels.prompt_split` hashes `prompt_idx`, so a cache in
the wrong namespace lands the same problem in a different train/val fold than
the control ladder does — making per-prompt paired comparisons across ladders
silently wrong — and because applying `clean_prompt_idx()` to a 0..405 cache
filters the wrong rows.

WHICH NAMESPACE A SNAPSHOT IS IN IS NOT INFERABLE FROM ITS INDEX RANGE. The
previous guard was `if max(prompt_idx) > max(subset_map)`, i.e. "> 405". A cache
harvested from a 200-prompt eval file is already correct at 0..199, and it
passes that guard (199 < 405) *and* the "any index outside the map" guard
(199 is inside 0..405) — so the repair would have fired on a correct cache and
mapped it through the 406-prompt map, irreversibly. The namespace is therefore
read from the snapshot manifest's `prompt_idx_namespace` field. A snapshot that
does not declare one is SKIPPED unless the operator declares it explicitly with
--assume_namespace; the flag never overrides a namespace the manifest states.

`--rollouts` is required: the mapping is only trustworthy if the rollout file's
i-th prompt really is the i-th clean prompt of the original eval file, and that
is checked by text equality (and by row count) before any snapshot is touched.

Writes go to `labels.parquet.partial` / `manifest.json.partial` and are renamed
into place, so an interrupted run cannot leave a half-written labels file as the
only copy.

    python experiments/fragility/phase0_replicate/relabel_prompt_index.py \
        --run_id phase0_harvest_runA --rollouts /vol/.../step_0.jsonl --dry_run
    python experiments/fragility/phase0_replicate/relabel_prompt_index.py \
        --run_id phase0_harvest_runA --rollouts /vol/.../step_0.jsonl \
        --assume_namespace subset_406_clean
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fragility_core import activations, labels as labels_mod, paths  # noqa: E402

# The two namespaces `prompt_idx` can be in. Recorded in each snapshot's
# manifest.json; `harvest_cross_cell.py` imports these names so the strings
# cannot drift between the writer and this repair tool.
NAMESPACE_ORIG = "orig_500"          # index into extension/data/countdown_eval_500.jsonl
NAMESPACE_SUBSET = "subset_406_clean"  # position within the contamination-filtered subset
KNOWN_NAMESPACES = (NAMESPACE_ORIG, NAMESPACE_SUBSET)

# The contamination filter keeps 406 of the 500 eval prompts. A different count
# means the mapping this script derives is not the one the cache was built with.
EXPECTED_SUBSET_N = 406


def build_mapping() -> dict[int, int]:
    """local subset position -> original countdown_eval_500 index."""
    clean = labels_mod.clean_prompt_idx()
    if clean is None:
        raise SystemExit("contamination manifest missing; mapping cannot be derived")
    if len(clean) != EXPECTED_SUBSET_N:
        raise SystemExit(
            f"contamination manifest lists {len(clean)} clean prompts, expected "
            f"{EXPECTED_SUBSET_N}; the caches this tool repairs were built from the "
            "clean-406 subset, so refusing to guess."
        )
    return {i: orig for i, orig in enumerate(sorted(clean))}


def verify_mapping(mapping: dict[int, int], rollouts: Path) -> None:
    """Refuse to relabel unless prompt TEXT confirms the mapping."""
    orig_rows = labels_mod.load_eval_rows(paths.PAPER_DATA / "countdown_eval_500.jsonl")
    roll_rows = labels_mod.load_eval_rows(rollouts)
    if len(roll_rows) != len(mapping):
        raise SystemExit(
            f"mapping rejected: {rollouts} has {len(roll_rows)} prompts but the "
            f"clean subset has {len(mapping)}. These rollouts were not sampled from "
            "the subset this mapping describes."
        )
    bad = [
        i for i in range(len(roll_rows))
        if roll_rows[i]["prompt"] != orig_rows[mapping[i]]["prompt"]
    ]
    if bad:
        raise SystemExit(
            f"mapping rejected: {len(bad)} prompt-text mismatches (first: {bad[:5]}). "
            "The cache must be re-harvested rather than relabelled."
        )
    print(f"[relabel] mapping verified against {len(roll_rows)} prompts by text equality")


# ---------------------------------------------------------------------------
# per-snapshot state
# ---------------------------------------------------------------------------


def declared_namespace(run_id: str, step: int) -> str | None:
    """`prompt_idx_namespace` from the snapshot manifest, or None if unrecorded."""
    ns = activations.load_manifest(run_id, step).get("prompt_idx_namespace")
    return str(ns) if ns is not None else None


def stale_partials(run_id: str, step: int) -> list[str]:
    """Leftover `.partial` files from an interrupted earlier run."""
    d = activations.step_dir(run_id, step)
    return sorted(p.name for p in d.glob("*.partial")) if d.exists() else []


def _atomic_replace(path: Path, write) -> None:
    """Write via `<name>.partial`, then rename onto `path`."""
    tmp = path.with_name(path.name + ".partial")
    write(tmp)
    os.replace(tmp, path)


def commit_relabel(
    run_id: str, step: int, lab: pd.DataFrame, *, rollouts: Path
) -> None:
    """Swap in the relabelled labels, then record the new namespace.

    Labels first: the manifest is what the next run reads, so marking it
    NAMESPACE_ORIG before the labels were actually replaced would make an
    interrupted run look finished.
    """
    lab_path = activations.labels_path(run_id, step)
    man_path = activations.step_dir(run_id, step) / "manifest.json"

    meta = activations.load_manifest(run_id, step)
    meta.setdefault("run_id", run_id)
    meta.setdefault("checkpoint_step", int(step))
    meta.update({
        "prompt_idx_namespace": NAMESPACE_ORIG,
        "prompt_idx_relabelled_from": NAMESPACE_SUBSET,
        "prompt_idx_relabel_tool": Path(__file__).name,
        "prompt_idx_relabel_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt_idx_relabel_verified_against": str(rollouts),
    })

    _atomic_replace(lab_path, lambda p: lab.to_parquet(p, index=False))
    _atomic_replace(
        man_path,
        lambda p: p.write_text(json.dumps(meta, indent=2, default=str)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--rollouts", required=True,
                    help="one rollout JSONL from this run, for text verification")
    ap.add_argument("--assume_namespace", choices=list(KNOWN_NAMESPACES), default=None,
                    help="namespace to assume for snapshots whose manifest does not "
                         "record one. Never overrides a recorded namespace.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    mapping = build_mapping()
    verify_mapping(mapping, Path(args.rollouts))

    steps = activations.list_steps(args.run_id)
    if not steps:
        raise SystemExit(f"no snapshots under {activations.run_dir(args.run_id)}")

    relabelled: list[int] = []
    skipped: dict[str, list[int]] = {}

    def skip(step: int, reason: str, msg: str) -> None:
        skipped.setdefault(reason, []).append(step)
        print(f"[relabel] step {step:>4}: SKIPPED — {msg}")

    for step in steps:
        partials = stale_partials(args.run_id, step)
        if partials:
            skip(step, "interrupted_earlier_run",
                 f"leftover {partials} from an interrupted run; inspect by hand")
            continue

        ns = declared_namespace(args.run_id, step)
        source = "manifest"
        if ns is None:
            ns, source = args.assume_namespace, "--assume_namespace"
        if ns is None:
            skip(step, "namespace_undeclared",
                 "manifest records no prompt_idx_namespace; pass --assume_namespace "
                 f"{NAMESPACE_SUBSET} if this cache really is in the subset namespace")
            continue
        if ns not in KNOWN_NAMESPACES:
            skip(step, "namespace_unknown", f"unknown prompt_idx_namespace {ns!r}")
            continue
        if ns == NAMESPACE_ORIG:
            skip(step, "already_original",
                 f"already {NAMESPACE_ORIG} (per {source})")
            continue

        p = activations.labels_path(args.run_id, step)
        lab = pd.read_parquet(p)
        lo, hi = int(lab["prompt_idx"].min()), int(lab["prompt_idx"].max())
        unknown = set(lab["prompt_idx"].unique()) - set(mapping)
        if unknown:
            # The declared namespace and the data disagree. Refuse either way.
            skip(step, "outside_subset",
                 f"declared {ns} (per {source}) but {len(unknown)} indices fall "
                 f"outside 0..{max(mapping)} (range [{lo},{hi}])")
            continue
        if hi != max(mapping):
            # A cache in the subset namespace was harvested from all 406 clean
            # prompts, so its top local index is 405. Anything less is either a
            # different prompt set (the 200-prompt case above) or a cache that
            # lost its last prompt entirely; neither can be repaired by this map.
            skip(step, "coverage_mismatch",
                 f"declared {ns} (per {source}) but the top index is {hi}, not "
                 f"{max(mapping)}; this cache does not cover the clean-{len(mapping)} subset")
            continue

        new = lab["prompt_idx"].map(mapping).astype(lab["prompt_idx"].dtype)
        print(
            f"[relabel] step {step:>4}: [{lo},{hi}] -> "
            f"[{int(new.min())},{int(new.max())}]  ({len(lab)} rows)"
            + ("  (dry run)" if args.dry_run else "")
        )
        relabelled.append(step)
        if not args.dry_run:
            lab["prompt_idx"] = new
            commit_relabel(args.run_id, step, lab, rollouts=Path(args.rollouts))

    n_skipped = sum(len(v) for v in skipped.values())
    verb = "would be relabelled" if args.dry_run else "relabelled"
    print(f"\n[relabel] {len(relabelled)}/{len(steps)} snapshots {verb} "
          f"to the {NAMESPACE_ORIG} namespace: {relabelled}")
    for reason, ss in sorted(skipped.items()):
        print(f"[relabel]   skipped ({reason}): {len(ss)} — {ss}")
    if n_skipped + len(relabelled) != len(steps):
        raise SystemExit(
            f"internal accounting error: {len(relabelled)} relabelled + {n_skipped} "
            f"skipped != {len(steps)} snapshots"
        )
    if args.dry_run:
        print("[relabel] dry run: nothing written")
    elif relabelled:
        print("[relabel] re-run Phase 1 so the by-prompt split is recomputed")


if __name__ == "__main__":
    main()
