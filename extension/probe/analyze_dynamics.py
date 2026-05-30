"""Layer C dynamics analysis.

Reads .npz files from --cache_dir whose checkpoint names look like
'C_outcome_step_NN'. For each (step, layer, position) it computes:
  - probe AUROC (held-out-problem GroupKFold 5-fold CV)
  - balanced 50/50 subsample AUROC (the leakage-free comparison)
  - n_pos, n_neg

Writes a long-form CSV ready for plotting, and prints the trajectory
of the headline pre_answer-vs-assertion split at the best layer.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from extension.probe.analyze_probes import parse_filename, load_groups, probe_auc_cv
from extension.probe.robustness_probes import balanced_subsample_auroc


_STEP_RE = re.compile(r"C_outcome_step_(?P<step>\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True,
                        help="Directory with the per-snapshot .npz files.")
    parser.add_argument("--out_csv", default="extension/outputs/dynamics_auroc.csv")
    parser.add_argument("--summary_layer", type=int, default=16,
                        help="Which layer to use for the printed trajectory summary.")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache_dir, "*.npz")))
    rows: list[dict] = []
    for npz_path in files:
        name = os.path.basename(npz_path)
        parsed = parse_filename(name)
        if not parsed:
            continue
        ckpt, layer, kind = parsed
        step_match = _STEP_RE.match(ckpt)
        if not step_match:
            continue  # only process step-indexed dynamics files
        step = int(step_match.group("step"))
        meta_path = npz_path.replace(".npz", ".meta.json")
        if not os.path.exists(meta_path):
            continue
        with np.load(npz_path) as data:
            X = data["X"]; y = data["y"]
        groups = load_groups(meta_path)

        probe_auc = probe_auc_cv(X, y, groups, n_splits=5)
        balanced_auc = balanced_subsample_auroc(X, y, groups)

        rows.append({
            "step": step, "layer": layer, "kind": kind,
            "n": int(X.shape[0]),
            "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
            "probe_auc": probe_auc,
            "balanced_auc": balanced_auc,
        })
        print(f"  done step={step:02d} L{layer} {kind} "
              f"probe={probe_auc:.3f} balanced={balanced_auc:.3f}", flush=True)

    # Write CSV.
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    field_names = ["step", "layer", "kind", "n", "n_pos", "n_neg",
                   "probe_auc", "balanced_auc"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for r in sorted(rows, key=lambda x: (x["step"], x["layer"], x["kind"])):
            writer.writerow(r)
    print(f"\nWrote {args.out_csv} ({len(rows)} rows)")

    # --- Printed trajectory summary -----------------------------------
    print()
    print("=" * 70)
    print(f"Layer-{args.summary_layer} trajectory (balanced AUROC):")
    print("=" * 70)
    header = f"{'step':>6}  {'pre_answer':>12}  {'assertion':>12}  {'neutral':>12}  {'asserts-neutral':>18}"
    print(header)
    print("-" * len(header))
    by_step: dict[int, dict[str, float]] = {}
    for r in rows:
        if r["layer"] != args.summary_layer:
            continue
        by_step.setdefault(r["step"], {})[r["kind"]] = r["balanced_auc"]
    for step in sorted(by_step):
        d = by_step[step]
        pre = d.get("pre_answer", float("nan"))
        ass = d.get("assertion", float("nan"))
        neu = d.get("neutral", float("nan"))
        gap = ass - neu if (not np.isnan(ass) and not np.isnan(neu)) else float("nan")
        print(f"{step:>6}  {pre:>12.3f}  {ass:>12.3f}  {neu:>12.3f}  {gap:>+18.3f}")


if __name__ == "__main__":
    main()
