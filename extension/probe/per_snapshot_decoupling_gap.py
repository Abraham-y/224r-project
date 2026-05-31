"""Per-snapshot decoupling gap over training.

Using Option B fresh-rollout caches at steps {30, 60, 90} (n=200) plus the
final C_SFT and C_outcome caches (n=500), compute the position-dependent
AUROC gap = pre_answer_AUROC - assertion_AUROC at each step. A growing gap
over training is the signature of decoupling emerging during outcome RL.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")


def load_cell(npz_path: str):
    if not os.path.exists(npz_path):
        return None
    with np.load(npz_path) as d:
        X = d["X"]; y = d["y"]
    meta_path = npz_path.replace(".npz", ".meta.json")
    meta = json.load(open(meta_path))
    groups = np.array([m["prompt_idx"] for m in meta])
    return X, y, groups


def balanced_groupkfold_auroc(X, y, groups, seed=0):
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n = min(len(pos), len(neg))
    if n < 5:
        return float("nan")
    idx = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
    X, y, groups = X[idx], y[idx], groups[idx]
    preds = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        preds[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    mask = ~np.isnan(preds)
    if len(np.unique(y[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(y[mask], preds[mask]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n500_cache", default="extension/cache/probe_cache_n500_clean406")
    parser.add_argument("--dyn_cache", default="extension/cache/probe_cache_dynamics_optB")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--out", default="extension/outputs/n500/text/14_per_snapshot_decoupling_gap.txt")
    args = parser.parse_args()

    rows: list[dict] = []

    # C_SFT (step "0" - reference for pre-RL)
    c = load_cell(f"{args.n500_cache}/C_SFT_l{args.layer}_pre_answer.npz")
    a = load_cell(f"{args.n500_cache}/C_SFT_l{args.layer}_assertion.npz")
    rows.append({
        "step": "C_SFT (pre-RL)",
        "pre_answer": balanced_groupkfold_auroc(*c) if c else float("nan"),
        "assertion":  balanced_groupkfold_auroc(*a) if a else float("nan"),
        "n": "clean406",
    })

    # Option B snapshots 30/60/90 (n=200 fresh rollouts each)
    for step in (30, 60, 90):
        c = load_cell(f"{args.dyn_cache}/C_outcome_step_{step}_l{args.layer}_pre_answer.npz")
        a = load_cell(f"{args.dyn_cache}/C_outcome_step_{step}_l{args.layer}_assertion.npz")
        rows.append({
            "step": f"step {step}",
            "pre_answer": balanced_groupkfold_auroc(*c) if c else float("nan"),
            "assertion":  balanced_groupkfold_auroc(*a) if a else float("nan"),
            "n": "n=200 fresh",
        })

    # C_outcome final
    c = load_cell(f"{args.n500_cache}/C_outcome_l{args.layer}_pre_answer.npz")
    a = load_cell(f"{args.n500_cache}/C_outcome_l{args.layer}_assertion.npz")
    rows.append({
        "step": "C_outcome (final)",
        "pre_answer": balanced_groupkfold_auroc(*c) if c else float("nan"),
        "assertion":  balanced_groupkfold_auroc(*a) if a else float("nan"),
        "n": "clean406",
    })

    out_lines = [
        f"Per-snapshot decoupling gap over training (L{args.layer})",
        f"= balanced GroupKFold(5) probe AUROC at pre_answer minus at assertion-pos",
        "",
        f"{'step':<22}{'pre_answer':>12}{'assertion':>12}{'gap':>10}   {'n':<14}",
        "-" * 70,
    ]
    for r in rows:
        gap = r["pre_answer"] - r["assertion"]
        out_lines.append(
            f"{r['step']:<22}{r['pre_answer']:>12.3f}{r['assertion']:>12.3f}{gap:>10.3f}   {r['n']:<14}"
        )

    txt = "\n".join(out_lines)
    print(txt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(txt + "\n")

    # Also save as a CSV so make_figures can pick it up.
    csv_path = args.out.replace(".txt", ".csv")
    with open(csv_path, "w") as f:
        f.write("step,pre_answer,assertion,gap,n\n")
        for r in rows:
            f.write(f"{r['step']},{r['pre_answer']:.4f},{r['assertion']:.4f},"
                    f"{r['pre_answer'] - r['assertion']:.4f},{r['n']}\n")


if __name__ == "__main__":
    main()
