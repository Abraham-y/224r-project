"""Phase 3 — the transfer matrix. Free, and it gates the expensive part.

Rows: a policy that was optimised against probe A (i.e. a harvested ladder's
final checkpoint). Columns: every probe in the zoo, frozen. Cell: that probe's
AUROC on that policy's activations.

Runs entirely off already-harvested activations — no GPU, no RL. CLAUDE.md
schedules this as a nice-to-have; it is better treated as the gate on Phase 3's
budget:

  - a column that stays high across every row is a probe that resisted an attack
    it was not the target of. That is the H3 signal, obtained for free.
  - if EVERY column collapses in every row, the attack transfers completely, and
    probe choice does not buy robustness. Paying ~$390 for 13 RL runs to
    rediscover that would be a waste.
  - if every column stays high except the diagonal, the collapse is entirely
    probe-specific, which is itself a strong (and cheap) result.

    python experiments/fragility/phase3_predict_robustness/transfer_matrix.py \
        --zoo_dir results/fragility/probe_zoo \
        --ladders phase0_harvest_runA:99 phase0_harvest_runB:99 \
        --layer 16 --label_rule last_block
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fragility_core import (  # noqa: E402
    activations,
    figures,
    labels as labels_mod,
    metrics_io,
    paths,
    probes,
)


def load_residualiser(zoo_dir: Path, pid: str) -> dict | None:
    """The fit-time activation transform for `pid`, if `build_probe_zoo` saved one.

    A probe fit on transformed activations must be SCORED on the same transform.
    Without this, `L16_resid_template` — fit on template-residualised rows — was
    scored on raw ones and reported 0.749 where the like-for-like figure is
    0.606: a basis mismatch printed as a robustness result.
    """
    p = zoo_dir / f"{pid}.residualiser.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as d:
        return {
            "kind": str(d["kind"]),
            "columns": [str(c) for c in d["columns"]],
            "beta": np.asarray(d["beta"], dtype=np.float64),
        }


def apply_residualiser(X: np.ndarray, lab: pd.DataFrame, resid: dict) -> np.ndarray:
    """Replay the fit-time transform. Raises rather than falling back to raw X."""
    missing = [c for c in resid["columns"] if c not in lab.columns]
    if missing:
        raise KeyError(f"label table is missing residualiser columns {missing}")
    T = lab[list(resid["columns"])].to_numpy(dtype=np.float64)
    Td = np.column_stack([np.ones(len(T)), T])
    return np.asarray(X, dtype=np.float64) - Td @ resid["beta"]


def load_zoo(zoo_dir: Path) -> dict[str, dict]:
    """{probe_id: {"model": ..., "layer": int|None, "resid": dict|None}}.

    Each probe is evaluated at the layer it was FIT on, through the transform it
    was fit on. Scoring an L12 probe against L16 activations is a basis mismatch,
    not a weakened monitor — it would report a spurious collapse for every
    off-layer probe in the zoo.

    `layer` is None for a probe with no single layer (the ensemble, whose spec
    carries `layers`). It must stay None rather than the -1 sentinel the CSV
    stores: `entry["layer"] or args.layer` treated -1 as truthy, so the ensemble
    was looked up at layer -1, found nothing, and every one of its cells came
    back NaN — which is how a 15-probe zoo silently became a 14-probe matrix.
    """
    feats = zoo_dir / "prehoc_features.csv"
    layer_of: dict[str, int | None] = {}
    if feats.exists():
        tbl = pd.read_csv(feats)
        has_single = "single_layer" in tbl.columns
        for r in tbl.itertuples():
            single = bool(getattr(r, "single_layer")) if has_single else int(r.layer) >= 0
            layer_of[str(r.probe_id)] = int(r.layer) if single else None
    else:
        print(f"[transfer] WARNING: {feats} missing; falling back to --layer for every probe")

    zoo = {}
    for p in sorted(zoo_dir.glob("*.pkl")):
        with open(p, "rb") as f:
            zoo[p.stem] = {
                "model": pickle.load(f),
                # `.get` returns None both for "not in the table" and for "no
                # single layer"; only the first should fall back to --layer, so
                # the two are distinguished by membership, not by value.
                "layer": layer_of.get(p.stem),
                "known": p.stem in layer_of,
                "resid": load_residualiser(zoo_dir, p.stem),
            }
    return zoo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo_dir", default=None)
    ap.add_argument("--ladders", nargs="+", required=True,
                    help="run_id:step pairs, e.g. phase0_harvest_runA:99")
    ap.add_argument("--layer", type=int, default=16,
                    help="fallback layer when prehoc_features.csv does not name one")
    ap.add_argument("--label_rule", default="last_block")
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balanced", action="store_true",
                    help="report class-balanced AUROC (post-collapse rows are very imbalanced)")
    args = ap.parse_args()

    paths.mkdirs()
    zoo_dir = Path(args.zoo_dir) if args.zoo_dir else paths.RESULTS / "probe_zoo"
    zoo = load_zoo(zoo_dir)
    if not zoo:
        raise SystemExit(f"no probes in {zoo_dir}; run build_probe_zoo.py first")
    print(f"[transfer] {len(zoo)} probes: {sorted(zoo)}")

    writer = metrics_io.MetricWriter("transfer_matrix", "phase3", args.seed)
    cells: dict[str, dict[str, float]] = {}

    for spec in args.ladders:
        run_id, _, step_s = spec.partition(":")
        step = int(step_s) if step_s else max(activations.list_steps(run_id) or [0])
        available = activations.list_layers(run_id, step)
        if not available:
            # A ladder that was never harvested contributes no row. Not zeros.
            print(f"[transfer] {run_id}@{step}: no cached activations; row omitted")
            continue

        row_key = f"{run_id}@{step}"
        cells[row_key] = {}
        lab_cache: dict[int, tuple] = {}

        for pid, entry in zoo.items():
            probe = entry["model"]
            if entry["known"] and entry["layer"] is None:
                # No single layer (the ensemble expects concatenated layers).
                # Stated, not silently NaN'd via an out-of-range layer lookup.
                print(f"[transfer]   {pid}: multi-layer probe; a single-layer "
                      "snapshot cannot evaluate it. Cell left empty.")
                cells[row_key][pid] = float("nan")
                continue
            layer = args.layer if entry["layer"] is None else entry["layer"]
            if layer not in available:
                print(f"[transfer]   {pid}: L{layer} not cached for this snapshot; cell empty")
                cells[row_key][pid] = float("nan")
                continue
            if layer not in lab_cache:
                X, lab = activations.load_snapshot(run_id, step, layer)
                lab = labels_mod.attach_split(lab, val_frac=args.val_frac, seed=args.split_seed)
                val = (lab["split"] == "val").to_numpy()
                lab_cache[layer] = (X, lab[args.label_rule].to_numpy().astype(int)[val], val, lab)
            X, y, val, lab = lab_cache[layer]

            # Replay the fit-time activation transform, if the probe had one.
            # A failure here leaves the cell EMPTY; it must never quietly fall
            # through to raw activations, which is the basis mismatch this
            # branch exists to prevent.
            if entry["resid"] is not None:
                try:
                    X = apply_residualiser(X, lab, entry["resid"])
                except Exception as exc:
                    print(f"[transfer]   {pid}: cannot replay its {entry['resid']['kind']} "
                          f"transform ({exc}); cell left empty")
                    cells[row_key][pid] = float("nan")
                    continue

            w = probes.probe_direction(probe)
            if w is not None and w.size != X.shape[1]:
                print(f"[transfer]   {pid}: dim {w.size} != {X.shape[1]}; cell left empty")
                cells[row_key][pid] = float("nan")
                continue
            try:
                s = probes.probe_scores(probe, X[val])
            except Exception as exc:
                print(f"[transfer]   {pid}: scoring failed ({exc}); cell left empty")
                cells[row_key][pid] = float("nan")
                continue
            a = (
                probes.auroc_balanced(y, s, seed=args.seed)
                if args.balanced
                else probes.auroc(y, s)
            )
            cells[row_key][pid] = a
            writer.add("transfer_auroc", a, checkpoint_step=step, layer=layer,
                       arm=run_id, probe_id=pid)
        n_val = int(next(iter(lab_cache.values()))[2].sum()) if lab_cache else 0
        print(f"[transfer] {row_key}: {n_val} val rows, {len(cells[row_key])} probes scored")

    if not cells:
        raise SystemExit("no ladders available; harvest at least one first")

    matrix = pd.DataFrame(cells).T.sort_index(axis=1)
    out_csv = paths.RESULTS / "transfer_matrix.csv"
    # Archival copy alongside the canonical path; see metrics_io.write_table.
    metrics_io.write_table(matrix, out_csv)
    writer.flush()

    print("\n[transfer] AUROC of each frozen probe on each attacked policy:")
    print(matrix.round(3).to_string())

    # The gate. Spread across columns is what Phase 3's budget is buying.
    spread = matrix.max(axis=1) - matrix.min(axis=1)
    print("\n[transfer] per-policy spread across probes (max - min AUROC):")
    for k, v in spread.items():
        print(f"  {k}: {v:.3f}")
    print(
        "\n[transfer] Read this before launching probe-zoo RL runs:\n"
        "  spread ~0 and all LOW   -> the attack transfers; probe choice buys nothing. STOP.\n"
        "  spread ~0 and all HIGH  -> nothing generalises from the attacked probe. Re-check the harvest.\n"
        "  spread large            -> some probes resist. THIS is the H3 signal; the RL runs are\n"
        "                             then worth paying for, starting with the extremes."
    )

    figures.fig_transfer_matrix(matrix)
    print(f"\n[transfer] wrote {out_csv}")


if __name__ == "__main__":
    main()
