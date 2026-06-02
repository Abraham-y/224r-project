"""Per-problem AUROC correlation with CORRECTED labels.

Re-runs §8.5.1. For each prompt with both correct and wrong rollouts
(under corrected first-block correctness labels), compute the
within-prompt probe AUROC at the </think> position. probe_drop =
auroc_C_SFT(p) - auroc_C_outcome(p). accuracy_delta = acc_C_outcome(p) -
acc_C_SFT(p). Then Spearman r between probe_drop and accuracy_delta.

Reports per-problem statistics and quadrant breakdown.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

CACHE_DIR = "extension/cache/probe_cache_n500_clean406"
OUT = "extension/outputs/n500/text/42_relabel_per_problem.txt"
_ANSWER_OPEN_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def check_block(eq, target, nums):
    try:
        nums_in_eq = [int(n) for n in re.findall(r"\d+", eq.strip())]
        if sorted(nums_in_eq) != sorted([int(x) for x in nums]):
            return False
        if not re.match(r"^[\d+\-*/().\s]+$", eq.strip()):
            return False
        result = eval(eq.strip(), {"__builtins__": None}, {})
        return abs(result - int(target)) < 1e-5
    except Exception:
        return False


def first_block_labels(eval_path):
    rows = [json.loads(l) for l in open(eval_path) if l.strip()]
    labs = {}
    for p, row in enumerate(rows):
        target = int(row["target"]); nums = list(row["nums"])
        for r_i, resp in enumerate(row["response"]):
            m = _ANSWER_OPEN_RE.search(resp)
            labs[(p, r_i)] = int(bool(m) and check_block(m.group(1), target, nums))
    return labs


def load_cache(name):
    with np.load(f"{CACHE_DIR}/{name}.npz") as d: X = d["X"]
    m = json.load(open(f"{CACHE_DIR}/{name}.meta.json"))
    g = np.array([int(x['prompt_idx']) for x in m])
    return X, m, g


def held_out_scores(X, y, groups):
    sc = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
        pipe.fit(X[tr], y[tr])
        sc[te] = pipe.predict_proba(X[te])[:, 1]
    return sc


def main():
    out_lines = ["Per-problem AUROC correlation with CORRECTED labels",
                 "Probe: pre_answer L16; labels: first-<answer>-block correctness", ""]

    # Load caches + labels
    X_s, m_s, g_s = load_cache("C_SFT_l16_pre_answer")
    X_o, m_o, g_o = load_cache("C_outcome_l16_pre_answer")
    labs_s = first_block_labels("eval_c_sft_n500.json")
    labs_o = first_block_labels("eval_c_outcome_n500.json")
    y_s = np.array([labs_s[(int(x['prompt_idx']), int(x['resp_idx']))] for x in m_s], dtype=np.int32)
    y_o = np.array([labs_o[(int(x['prompt_idx']), int(x['resp_idx']))] for x in m_o], dtype=np.int32)

    # Held-out probe scores for each ckpt
    print("[per_problem] training held-out probes...", flush=True)
    sc_s = held_out_scores(X_s, y_s, g_s)
    sc_o = held_out_scores(X_o, y_o, g_o)

    # Per-prompt AUROC + accuracy
    def per_prompt_auroc_and_acc(meta, sc, y):
        by_p = defaultdict(list)
        for i, e in enumerate(meta):
            if np.isnan(sc[i]): continue
            by_p[int(e['prompt_idx'])].append((sc[i], y[i]))
        per_auc = {}; per_acc = {}; per_n = {}
        for p, vals in by_p.items():
            if len(vals) < 4: continue
            scs = np.array([t[0] for t in vals]); labs = np.array([t[1] for t in vals])
            per_acc[p] = float(labs.mean())
            per_n[p] = len(vals)
            # AUROC needs both classes present
            if len(set(labs)) < 2:
                per_auc[p] = float("nan")
            else:
                per_auc[p] = float(roc_auc_score(labs, scs))
        return per_auc, per_acc, per_n

    auc_s, acc_s, n_s = per_prompt_auroc_and_acc(m_s, sc_s, y_s)
    auc_o, acc_o, n_o = per_prompt_auroc_and_acc(m_o, sc_o, y_o)

    common = sorted(set(auc_s) & set(auc_o))
    common = [p for p in common if not (np.isnan(auc_s[p]) or np.isnan(auc_o[p]))]
    print(f"[per_problem] n_problems with both ckpts and both classes present: {len(common)}")

    probe_drop = np.array([auc_s[p] - auc_o[p] for p in common])
    acc_delta  = np.array([acc_o[p] - acc_s[p] for p in common])

    r_p, p_p = pearsonr(probe_drop, acc_delta)
    r_s, p_s = spearmanr(probe_drop, acc_delta)
    print(f"  Pearson r = {r_p:+.3f} (p={p_p:.2e})")
    print(f"  Spearman r = {r_s:+.3f} (p={p_s:.2e})")

    # Quadrant breakdown
    n_decoupling = int(((probe_drop > 0) & (acc_delta > 0)).sum())  # probe DOWN under RL, acc UP
    n_both_up    = int(((probe_drop < 0) & (acc_delta > 0)).sum())  # probe UP, acc UP
    n_damage     = int(((probe_drop > 0) & (acc_delta < 0)).sum())  # probe DOWN, acc DOWN
    n_noise      = int(((probe_drop < 0) & (acc_delta < 0)).sum())
    N = len(common)
    print(f"  n={N}")
    print(f"  decoupling (probe drop, acc rise): {n_decoupling} ({100*n_decoupling/N:.1f}%)")
    print(f"  both improved (probe rise, acc rise): {n_both_up} ({100*n_both_up/N:.1f}%)")
    print(f"  damage (probe drop, acc drop): {n_damage} ({100*n_damage/N:.1f}%)")
    print(f"  noise (probe rise, acc drop): {n_noise} ({100*n_noise/N:.1f}%)")

    print(f"  per-problem AUROC mean: C_SFT {np.mean([auc_s[p] for p in common]):.3f}, "
          f"C_outcome {np.mean([auc_o[p] for p in common]):.3f}, drop {probe_drop.mean():+.3f}")

    out_lines += [
        f"  n_problems: {N}",
        f"  Pearson r(probe_drop, acc_delta) = {r_p:+.3f} (p={p_p:.2e})",
        f"  Spearman r(probe_drop, acc_delta) = {r_s:+.3f} (p={p_s:.2e})",
        f"  Quadrants:",
        f"    decoupling: {n_decoupling}/{N} ({100*n_decoupling/N:.1f}%)",
        f"    both improved: {n_both_up}/{N} ({100*n_both_up/N:.1f}%)",
        f"    damage: {n_damage}/{N} ({100*n_damage/N:.1f}%)",
        f"    noise: {n_noise}/{N} ({100*n_noise/N:.1f}%)",
        f"  Per-problem AUROC mean: C_SFT={np.mean([auc_s[p] for p in common]):.3f}, "
        f"C_outcome={np.mean([auc_o[p] for p in common]):.3f}",
        f"  Per-problem probe_drop mean: {probe_drop.mean():+.3f}",
    ]

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f: f.write("\n".join(out_lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
