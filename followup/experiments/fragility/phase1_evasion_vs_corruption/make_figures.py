"""Regenerate every Phase-0/1 figure from parquet alone. No model, no cache.

    python experiments/fragility/phase1_evasion_vs_corruption/make_figures.py \
        --run_id vanilla_rloo_ladder --layer 16 --layers 12,16,20 --arm on_policy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fragility_core import figures, paths  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--layer", type=int, default=16, help="the probe's layer")
    ap.add_argument("--layers", default="12,16,20", help="layers for the relocation panel")
    ap.add_argument("--arm", default="on_policy")
    args = ap.parse_args()

    paths.mkdirs()
    layers = [int(x) for x in args.layers.split(",")]

    made = [
        figures.fig_phase0_collapse(args.run_id, layer=args.layer, arm=args.arm),
        figures.fig_phase1_three_aurocs(args.run_id, layer=args.layer, arm=args.arm),
        figures.fig_phase1_per_layer(args.run_id, layers=layers, arm=args.arm),
        figures.fig_phase1_geometry(args.run_id, layer=args.layer, arm=args.arm),
        figures.fig_h0_confound(args.run_id, layer=args.layer, arm=args.arm),
        figures.fig_phase2_causal(args.run_id, layer=args.layer),
    ]
    n = sum(1 for m in made if m)
    print(f"\n[figures] {n}/{len(made)} figures written to {paths.FIGURES}")
    print("[figures] figures with no underlying metrics were skipped, not stubbed.")


if __name__ == "__main__":
    main()
