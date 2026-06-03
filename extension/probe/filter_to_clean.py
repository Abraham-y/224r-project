"""Filter the n=500 cache to the 406 problems that are NOT in asingh15's train split.

Reads the contamination index from extension/data/contaminated_prompt_idx.json
and writes a parallel cache dir extension/cache/probe_cache_n500_clean406/ with
filtered .npz + .meta.json files. Same naming convention.

Then any of the analysis scripts can be re-run by pointing --cache_dir at the
clean dir.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="extension/cache/probe_cache_n500",
                    help="Source cache directory (output of cache_hidden_states.py).")
    ap.add_argument("--dst_dir", default="extension/cache/probe_cache_n500_clean406",
                    help="Filtered output cache directory.")
    ap.add_argument("--contam_path", default="extension/data/contaminated_prompt_idx.json")
    args = ap.parse_args()

    src = Path(args.src_dir)
    dst = Path(args.dst_dir)
    contam_path = Path(args.contam_path)

    if not src.exists():
        print(f"missing {src}", file=sys.stderr); sys.exit(1)
    if not contam_path.exists():
        print(f"missing {contam_path}", file=sys.stderr); sys.exit(1)

    cont = json.load(open(contam_path))
    clean_set = set(cont["clean"])
    contaminated_set = set(cont["contaminated"])
    print(f"clean prompt_idx: {len(clean_set)} (will keep)")
    print(f"contaminated prompt_idx: {len(contaminated_set)} (will drop)")

    dst.mkdir(parents=True, exist_ok=True)

    for npz_path in sorted(src.glob("*.npz")):
        meta_path = src / (npz_path.stem + ".meta.json")
        if not meta_path.exists():
            print(f"  WARN: missing meta for {npz_path.name}")
            continue
        d = np.load(npz_path)
        X, y = d["X"], d["y"]
        meta = json.load(open(meta_path))
        assert len(meta) == len(X), f"meta/X length mismatch: {len(meta)} vs {len(X)} in {npz_path.name}"

        keep = np.array([int(m["prompt_idx"]) in clean_set for m in meta], dtype=bool)
        n_keep = int(keep.sum())
        n_drop = int(len(keep) - n_keep)

        new_X = X[keep].astype(X.dtype)
        new_y = y[keep].astype(y.dtype)
        new_meta = [meta[i] for i in range(len(meta)) if keep[i]]

        out_npz = dst / npz_path.name
        out_meta = dst / meta_path.name
        np.savez_compressed(out_npz, X=new_X, y=new_y)
        with open(out_meta, "w") as f:
            json.dump(new_meta, f)
        print(f"  {npz_path.name}: kept {n_keep} / dropped {n_drop} (pos% = {new_y.mean():.1%})")

    print(f"\nWrote filtered cache to {dst}")


if __name__ == "__main__":
    main()
