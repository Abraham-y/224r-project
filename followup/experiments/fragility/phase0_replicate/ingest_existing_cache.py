"""Ingest the original project's .npz activation caches into the follow-up layout.

The original project cached activations as
`{prefix}_l{layer}_{position}.npz` (X, y) plus a `.meta.json` carrying
(prompt_idx, resp_idx, tok_idx) per row. This script re-keys those rows against
freshly recomputed labels and writes them as an `acts/{run_id}/{step}/` snapshot
that Phase 1 can read like any harvested ladder.

Why re-derive the labels instead of using the cached `y`:
  - the cached `y` is one label rule (and, for probe_cache_dynamics_optB,
    a rule that was later corrected — see findings.md EXP-14);
  - Phase 1 needs BOTH rules plus structural descriptors (n_blocks,
    template_score) on the same rows.
The cached `y` is kept as `y_cached` so any disagreement stays visible.

    python experiments/fragility/phase0_replicate/ingest_existing_cache.py \
        --config ingest_vanilla_ladder.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fragility_core import activations, labels as labels_mod, paths, registry  # noqa: E402


def _resolve(p: str) -> Path:
    """Config paths are written relative to the ORIGINAL repo root."""
    path = Path(p)
    return path if path.is_absolute() else paths.PAPER_ROOT / path


def ingest_snapshot(
    *,
    run_id: str,
    step: int,
    cache_prefix: Path,
    eval_json: Path,
    layers: list[int],
    position: str,
    clean: set[int] | None,
    label_extra: dict,
    overwrite: bool,
) -> dict:
    """Build and write one checkpoint snapshot. Returns a status record."""
    status = {"step": step, "cache_prefix": str(cache_prefix), "eval_json": str(eval_json)}

    meta_path = Path(f"{cache_prefix}_l{layers[0]}_{position}.meta.json")
    if not meta_path.exists():
        status["status"] = "missing_cache"
        return status
    if not eval_json.exists():
        status["status"] = "missing_eval_json"
        return status

    with open(meta_path) as f:
        meta = json.load(f)
    rows = pd.DataFrame(
        [{"prompt_idx": int(m["prompt_idx"]), "resp_idx": int(m["resp_idx"]),
          "tok_idx": int(m.get("tok_idx", -1))} for m in meta]
    )

    # Recompute labels from the rollout text and join onto the cached rows.
    lab = labels_mod.build_label_table(eval_json)
    # validate="one_to_one" for the same reason harvest_ladder.build_labels does
    # it: a duplicate key silently lengthens `merged`, and `keep` is then applied
    # as a boolean mask to activation arrays it no longer matches.
    merged = rows.merge(lab, on=["prompt_idx", "resp_idx"], how="left",
                        indicator=True, validate="one_to_one")
    n_unmatched = int((merged["_merge"] != "both").sum())
    if n_unmatched:
        status["unmatched_rows"] = n_unmatched
    keep = (merged["_merge"] == "both").to_numpy()
    if clean is not None:
        keep &= merged["prompt_idx"].isin(clean).to_numpy()
    merged = merged.drop(columns=["_merge"])

    # Load every requested layer, dropping the same rows from each.
    layer_to_X: dict[int, np.ndarray] = {}
    y_cached: np.ndarray | None = None
    for layer in layers:
        npz = Path(f"{cache_prefix}_l{layer}_{position}.npz")
        if not npz.exists():
            status.setdefault("missing_layers", []).append(layer)
            continue
        with np.load(npz) as d:
            X = d["X"]
            if y_cached is None and "y" in d.files:
                y_cached = d["y"]
        if X.shape[0] != len(rows):
            status["status"] = "row_mismatch"
            status["detail"] = f"L{layer}: {X.shape[0]} act rows vs {len(rows)} meta rows"
            return status
        layer_to_X[layer] = X[keep]
    if not layer_to_X:
        status["status"] = "no_layers"
        return status

    out_labels = merged[keep].reset_index(drop=True)
    if y_cached is not None and len(y_cached) == len(rows):
        out_labels["y_cached"] = y_cached[keep]

    activations.save_snapshot(
        run_id,
        step,
        layer_to_X,
        out_labels,
        manifest={
            "source": "ingested from original-project npz cache",
            "cache_prefix": str(cache_prefix),
            "eval_json": str(eval_json),
            "position": position,
            "clean_filter": clean is not None,
            "n_before_filter": len(rows),
            "n_after_filter": int(keep.sum()),
            **label_extra,
        },
        overwrite=overwrite,
    )
    status.update(
        {
            "status": "ok",
            "n_rows": int(keep.sum()),
            "layers": sorted(layer_to_X),
            "pos_rate_last_block": float(out_labels["last_block"].mean()),
            "pos_rate_first_block": float(out_labels["first_block"].mean()),
            "mean_blocks": float(out_labels["n_blocks"].mean()),
        }
    )
    if y_cached is not None and "y_cached" in out_labels:
        agree_first = float((out_labels["y_cached"] == out_labels["first_block"]).mean())
        agree_last = float((out_labels["y_cached"] == out_labels["last_block"]).mean())
        status["cached_y_agrees_first_block"] = agree_first
        status["cached_y_agrees_last_block"] = agree_last
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = registry.load_config(args.config)
    registry.set_all_seeds(int(cfg.get("seed", 0)))
    paths.mkdirs()

    run_id = cfg.get("run_id") or registry.run_id(cfg)
    layers = [int(x) for x in cfg["layers"]]
    position = cfg.get("position", "pre_answer")
    clean = labels_mod.clean_prompt_idx() if cfg.get("clean_only", True) else None
    if cfg.get("clean_only", True) and clean is None:
        print("[ingest] WARNING: clean-406 manifest not found; ingesting ALL prompts")

    print(f"[ingest] run_id={run_id}  layers={layers}  position={position}")
    print(f"[ingest] acts root: {paths.acts_root()}")

    report = []
    for snap in cfg["snapshots"]:
        step = int(snap["step"])
        st = ingest_snapshot(
            run_id=run_id,
            step=step,
            cache_prefix=_resolve(snap["cache_prefix"]),
            eval_json=_resolve(snap["eval_json"]),
            layers=layers,
            position=position,
            clean=clean,
            label_extra={"label": snap.get("label", ""), "arm": cfg.get("arm", "on_policy")},
            overwrite=args.overwrite,
        )
        report.append(st)
        if st["status"] == "ok":
            print(
                f"[ingest] step {step:>4} {snap.get('label',''):<20} "
                f"n={st['n_rows']:>5}  acc_last={st['pos_rate_last_block']:.3f}  "
                f"acc_first={st['pos_rate_first_block']:.3f}  blocks={st['mean_blocks']:.2f}"
            )
        else:
            # An absent snapshot stays absent. Nothing is interpolated.
            print(f"[ingest] step {step:>4} SKIPPED: {st['status']} {st.get('detail','')}")

    activations.save_run_manifest(
        run_id,
        {
            "run_id": run_id,
            "source": "ingest_existing_cache.py",
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "snapshots": report,
        },
    )
    registry.register(cfg, phase=cfg.get("phase", "phase0"), run_id_override=run_id, report=report)

    ok = [r for r in report if r["status"] == "ok"]
    print(f"\n[ingest] {len(ok)}/{len(report)} snapshots written to {activations.run_dir(run_id)}")
    if len(ok) < len(report):
        print("[ingest] missing snapshots are absent from the cache, not filled in.")


if __name__ == "__main__":
    main()
