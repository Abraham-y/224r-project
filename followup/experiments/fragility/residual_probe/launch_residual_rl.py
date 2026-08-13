"""Run probe-as-reward RL with the surface-residualised probe.

Wraps `extension/training/probe_reward_rloo.py` rather than editing it
(followup/CLAUDE.md). Two module globals are swapped before `main()` runs:

    _load_probe        -> loads the v1 dict artefact, falling back to the
                          original loader for .npz / legacy pickles
    _make_probe_reward -> the fork that passes rollout text to text-aware probes

Everything else — arg parsing, frozen-model loading, the </think> extraction,
the collate patch, the handoff to rloo.py — is the original code path.

    python launch_residual_rl.py --probe <artefact.pkl> --reward_mode probe [rloo args...]
"""
from __future__ import annotations
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

# rloo.py starts Ray, and a Ray actor is a FRESH process: it inherits the
# environment but NOT runtime sys.path edits. `_run` invokes this script with
# cwd=followup/, so the repo root holding `rloo_trainer` is on the path only
# because of the insert above -- which the SamplingWorker never sees, and it
# dies with `ModuleNotFoundError: No module named 'rloo_trainer'` after the
# model has already loaded. PYTHONPATH is the one that propagates to children.
import os  # noqa: E402

_pp = [p for p in (str(_ROOT), str(_HERE), os.environ.get("PYTHONPATH", "")) if p]
os.environ["PYTHONPATH"] = os.pathsep.join(_pp)

import surface_residual_probe as srp          # noqa: E402
import residual_reward as rr                  # noqa: E402
import extension.training.probe_reward_rloo as prr  # noqa: E402

_orig_load = prr._load_probe


def _load_probe(path: str):
    """v1 residual artefact if it is one, otherwise the original loader."""
    if str(path).endswith(".npz"):
        return _orig_load(path)
    try:
        p = srp.load_any(path)
        print(f"[residual] loaded surface-residualised probe: "
              f"{type(p).__name__}, {len(p.keys)} surface features, "
              f"heldout AUROC {p.provenance.get('report',{}).get('auroc_residual_heldout') or p.provenance.get('report',{}).get('auroc_heldout')}",
              flush=True)
        return p
    except ValueError:
        print(f"[residual] {path} is not a v1 residual artefact; using the "
              "original loader (probe will be called WITHOUT text)", flush=True)
        return _orig_load(path)


prr._load_probe = _load_probe
prr._make_probe_reward = rr.make_probe_reward

if __name__ == "__main__":
    print("[residual] reward path: text-aware fork ACTIVE", flush=True)
    prr.main()
