"""Extract the temp1 probe's direction in input space for causal steering.

Output: extension/cache/steering/C_outcome_temp1_l16_direction.npz
  - v_unit: unit-norm probe direction in input space (w / scaler.scale_, normalized)
  - h_mean_norm: typical L2 norm of hidden states at the </think> position
  - probe_auroc: held-out balanced AUROC
"""

from __future__ import annotations
import json, os, pickle
import numpy as np

PROBE = "extension/cache/steering/probe_pipeline_C_outcome_l16_pre_answer_temp1.pkl"
META = "extension/cache/steering/probe_pipeline_temp1_meta.json"
CACHE = "extension/cache/probe_cache_temp1/C_outcome_temp1_l16_pre_answer.npz"
OUT = "extension/cache/steering/C_outcome_temp1_l16_direction.npz"


def main():
    with open(PROBE, "rb") as f: probe = pickle.load(f)
    with np.load(CACHE) as d: X = d["X"]
    scaler = probe.named_steps["sc"]
    lr = probe.named_steps["lr"]
    # Direction in input space: w_input = w_lr / scaler.scale_
    w_input = lr.coef_[0] / scaler.scale_
    v_unit = w_input / np.linalg.norm(w_input)
    h_norm = float(np.mean([np.linalg.norm(x) for x in X]))
    auroc = float(json.load(open(META))["auroc_heldout_balanced"])
    np.savez(OUT,
             v_unit=v_unit.astype(np.float32),
             v_input_raw=w_input.astype(np.float32),
             h_mean_norm=np.float32(h_norm),
             probe_auroc=np.float32(auroc))
    print(f"v_unit shape: {v_unit.shape}; ||w_input||: {np.linalg.norm(w_input):.2f}")
    print(f"h_mean_norm: {h_norm:.2f}, probe AUROC: {auroc:.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
