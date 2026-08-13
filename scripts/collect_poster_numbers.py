"""Collect every number the poster/report figures need, from data, into one JSON.

Motivation: `make_poster_figures.py` used to hardcode its bar heights as Python
literals. Three problems followed from that:

  1. Its probe-AUROC bars said 0.904 / 0.980 while the README said 0.912 / 0.982.
     Both are real numbers -- they come from two DIFFERENT held-out estimators
     that the codebase uses interchangeably (see `auroc_balance_then_cv` vs
     `auroc_cv_then_balance` below). Hardcoding hid the discrepancy.
  2. Its `neutral` bar was 0.562 for BOTH checkpoints, which no artifact in the
     repo supports.
  3. It wrote `figures/poster_post_goodhart_delta.pdf` -- the same path
     `extension/probe/plot_causal_steering.py` writes from the actual JSONLs --
     so whichever ran last won.

This script computes everything reproducibly and writes
`extension/outputs/poster_numbers.json`. `make_poster_figures.py` reads that.
The post-Goodhart delta figure is now owned solely by plot_causal_steering.py.

Pure CPU. Needs the local caches + eval JSONs.

    python scripts/collect_poster_numbers.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import warnings
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

CACHE_DIR = "extension/cache/probe_cache_n500_clean406"
EVALS = {"C_SFT": "eval_c_sft_n500.json", "C_outcome": "eval_c_outcome_n500.json"}
STEERING_VANILLA = "extension/outputs/n500/causal_steering_full.jsonl"
OUT = "extension/outputs/poster_numbers.json"

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def check_block(eq: str, target: int, nums: list[int]) -> bool:
    eq = eq.strip()
    try:
        n = [int(x) for x in re.findall(r"\d+", eq)]
        if sorted(n) != sorted([int(x) for x in nums]):
            return False
        if not re.match(r"^[\d+\-*/().\s]+$", eq):
            return False
        return abs(eval(eq, {"__builtins__": None}, {}) - int(target)) < 1e-5
    except Exception:
        return False


def first_block_labels(eval_path: str) -> dict[tuple[int, int], int]:
    labs = {}
    for p, row in enumerate(json.loads(l) for l in open(eval_path) if l.strip()):
        t = int(row["target"]); nums = list(row["nums"])
        for r_i, resp in enumerate(row["response"]):
            m = _ANSWER_RE.search(resp)
            labs[(p, r_i)] = int(bool(m) and check_block(m.group(1), t, nums))
    return labs


def _pipe():
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=0.1, max_iter=2000))])


def auroc_balance_then_cv(X, y, groups, seed: int = 0) -> float:
    """Balance the classes first, then GroupKFold(5) on the balanced subsample.

    This is what relabel_full_grid.py / per_layer_sweep.py / relabel_dynamics.py do.
    """
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    nb = min(len(pos), len(neg))
    if nb < 5:
        return float("nan")
    idx = np.concatenate([rng.choice(pos, nb, replace=False),
                          rng.choice(neg, nb, replace=False)])
    Xs, ys, gs = X[idx], y[idx], groups[idx]
    preds = np.full(len(ys), np.nan)
    for tr, te in GroupKFold(5).split(Xs, ys, gs):
        preds[te] = _pipe().fit(Xs[tr], ys[tr]).predict_proba(Xs[te])[:, 1]
    return float(roc_auc_score(ys, preds))


def auroc_cv_then_balance(X, y, groups, seed: int = 0) -> float:
    """GroupKFold(5) on ALL rows, then balance only for the AUROC computation.

    This is what relabel_cross_checkpoint.py / probe_guided_restart.py /
    probe_abstention_and_hybrid.py do, and it is the source of the 0.912 / 0.982
    pair quoted in the README. It trains on more data, so it reads slightly higher.
    """
    scores = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        scores[te] = _pipe().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    nb = min(len(pos), len(neg))
    if nb < 5:
        return float("nan")
    idx = np.concatenate([rng.choice(pos, nb, replace=False),
                          rng.choice(neg, nb, replace=False)])
    return float(roc_auc_score(y[idx], scores[idx]))


def probe_auroc_table(layer: int = 16) -> dict:
    out: dict = {}
    for ckpt, eval_path in EVALS.items():
        labs = first_block_labels(eval_path)
        out[ckpt] = {}
        for kind in ("neutral", "assertion", "pre_answer"):
            npz = os.path.join(CACHE_DIR, f"{ckpt}_l{layer}_{kind}.npz")
            if not os.path.exists(npz):
                print(f"  [skip] missing {npz}")
                continue
            with np.load(npz) as d:
                X = d["X"]
            meta = json.load(open(npz.replace(".npz", ".meta.json")))
            groups = np.array([int(m["prompt_idx"]) for m in meta])
            y = np.array([labs.get((int(m["prompt_idx"]), int(m["resp_idx"])), 0)
                          for m in meta], dtype=np.int32)
            a_bc = auroc_balance_then_cv(X, y, groups)
            a_cb = auroc_cv_then_balance(X, y, groups)
            out[ckpt][kind] = {
                "auroc_balance_then_cv": a_bc,
                "auroc_cv_then_balance": a_cb,
                "n_rows": int(len(y)),
                "pos_frac": float(y.mean()),
            }
            print(f"  {ckpt:>10} L{layer} {kind:<11} "
                  f"balance->CV {a_bc:.3f}   CV->balance {a_cb:.3f}   (n={len(y)})")
    return out


def steering_curve(path: str) -> dict:
    if not os.path.exists(path):
        print(f"  [skip] missing {path}")
        return {}
    rows = [json.loads(l) for l in open(path) if l.strip()]
    acc: dict[tuple[float, str], list[int]] = defaultdict(list)
    for r in rows:
        acc[(float(r["alpha"]), r["direction"])].append(int(r["new_score"] == 1.0))
    alphas = sorted({a for a, _ in acc})
    base = float(np.mean(acc[(0.0, "zero")])) if (0.0, "zero") in acc else float("nan")
    out = {"baseline_acc": base, "n_rows": len(rows), "alphas": [], "probe": [], "rand": []}
    for a in alphas:
        if a == 0.0:
            out["alphas"].append(0.0); out["probe"].append(base); out["rand"].append(base)
            continue
        if (a, "probe") not in acc or (a, "rand") not in acc:
            continue
        out["alphas"].append(a)
        out["probe"].append(float(np.mean(acc[(a, "probe")])))
        out["rand"].append(float(np.mean(acc[(a, "rand")])))
    print(f"  steering baseline acc={base:.3f}; alphas={out['alphas']}")
    print(f"    probe={[round(v, 3) for v in out['probe']]}")
    print(f"    rand ={[round(v, 3) for v in out['rand']]}")
    return out


# The runA training trajectory is a W&B time series with no artifact in this
# repo, so it stays a literal -- but an explicitly sourced and labelled one,
# rather than an anonymous list buried in a plotting function.
GOODHART_RUNA = {
    "_source": "W&B run rloo_probe_reward/runA (init from C_outcome), probe-as-reward. "
               "No local artifact; re-export from W&B if these need to change.",
    "steps":    [0, 10, 20, 30, 40, 50, 60, 70, 90, 99],
    "probe":    [0.452, 0.447, 0.561, 0.553, 0.687, 0.809, 0.947, 0.978, 0.988, 0.991],
    "verifier": [0.572, 0.490, 0.582, 0.525, 0.528, 0.479, 0.385, 0.337, 0.310, 0.321],
}


def main() -> None:
    print("[poster] probe AUROC table (L16, corrected first-block labels)")
    aurocs = probe_auroc_table(16)
    print("[poster] causal steering curve (vanilla C_outcome)")
    steering = steering_curve(STEERING_VANILLA)

    payload = {
        "_generated_by": "scripts/collect_poster_numbers.py",
        "_note": "Two held-out AUROC estimators are reported per cell because the "
                 "codebase uses both. balance_then_cv is what relabel_full_grid.py "
                 "reports; cv_then_balance is what the README's 0.912/0.982 quotes. "
                 "Pick ONE for the paper and say which.",
        "probe_auroc_l16": aurocs,
        "causal_steering_vanilla": steering,
        "goodhart_runA": GOODHART_RUNA,
    }
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[poster] wrote {OUT}")


if __name__ == "__main__":
    main()
