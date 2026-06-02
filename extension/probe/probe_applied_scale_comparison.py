"""Cross-scale probe-applied comparison: 0.5B vs 1.5B.

For each model scale, evaluate:
  1. base pass@1 (first-block accuracy)
  2. held-out probe AUROC at </think>
  3. probe-best-of-16 lift (vs pass@1)
  4. probe-guided abstention: accuracy at 50% coverage
  5. probe-guided budgeted restart: (B=16, T=0.95)

Question: does probe-applied selection help when the base model is already
strong? At 0.5B (~55% first-block acc, 87% multi-answer), the probe was a
big lift. At 1.5B (~85% acc, ~0% multi-answer), the question is whether
the probe can still help (i.e., abstain on the hard 15%).

Note: per-scale held-out probes trained INDEPENDENTLY — this is not a
weight-transfer experiment (hidden dims differ across scales).
"""

from __future__ import annotations

import json
import os
import re
import warnings
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

EVAL_05B = "eval_c_outcome_n500.json"
EVAL_15B = "eval_c_outcome_1.5b_n500.json"
CACHE_05B = "extension/cache/probe_cache_n500_clean406/C_outcome_l16_pre_answer.npz"
META_05B = CACHE_05B.replace(".npz", ".meta.json")
CACHE_15B = "extension/cache/probe_cache_1.5b_clean406/C_outcome_l16_pre_answer.npz"
META_15B = CACHE_15B.replace(".npz", ".meta.json")
OUT_TXT = "extension/outputs/n500/text/34_probe_applied_scale_comparison.txt"

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


def evaluate_scale(eval_path, cache_path, meta_path, label):
    print(f"\n===== {label} =====")
    with np.load(cache_path) as d:
        X = d["X"]
    meta = json.load(open(meta_path))
    groups = np.array([int(m["prompt_idx"]) for m in meta])
    rows = [json.loads(l) for l in open(eval_path) if l.strip()]

    first_correct = {}
    for p_idx, row in enumerate(rows):
        target = int(row["target"]); nums = list(row["nums"])
        for r_idx, resp in enumerate(row["response"]):
            m = _ANSWER_OPEN_RE.search(resp)
            if m is None:
                first_correct[(p_idx, r_idx)] = False
            else:
                first_correct[(p_idx, r_idx)] = check_block(m.group(1), target, nums)

    y_new = np.array([int(first_correct.get((int(m["prompt_idx"]), int(m["resp_idx"])), 0)) for m in meta], dtype=np.int32)
    print(f"  n_rows_cached: {len(y_new)}   positive (first-block correct): {y_new.mean():.3f}")

    # Held-out probe
    scores = np.full(len(y_new), np.nan)
    for tr, te in GroupKFold(5).split(X, y_new, groups):
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
        pipe.fit(X[tr], y_new[tr])
        scores[te] = pipe.predict_proba(X[te])[:, 1]

    pos = np.where(y_new == 1)[0]; neg = np.where(y_new == 0)[0]
    nb = min(len(pos), len(neg))
    rng = np.random.RandomState(0)
    if nb > 0:
        idx = np.concatenate([rng.choice(pos, nb, replace=False), rng.choice(neg, nb, replace=False)])
        bal_auc = float(roc_auc_score(y_new[idx], scores[idx]))
    else:
        bal_auc = float("nan")
    print(f"  held-out balanced AUROC (corrected labels): {bal_auc:.3f}")

    # Group by prompt
    by_p = defaultdict(list)
    for i, m in enumerate(meta):
        p = int(m["prompt_idx"]); r = int(m["resp_idx"])
        if np.isnan(scores[i]): continue
        by_p[p].append((r, float(scores[i]), int(first_correct.get((p, r), 0))))
    for p in by_p: by_p[p].sort()
    prompts = sorted(by_p.keys())
    n = len(prompts)

    # Pass@1 = resp_idx 0 first-block
    pass1 = float(np.mean([by_p[p][0][2] for p in prompts if by_p[p]]))
    # Best-of-16 (by probe)
    bok = float(np.mean([max(by_p[p][:16], key=lambda t: t[1])[2] for p in prompts if by_p[p]]))
    # Majority vote (true-correctness majority) for reference
    maj = float(np.mean([int(sum(t[2] for t in by_p[p][:16]) > len(by_p[p][:16])/2) for p in prompts if by_p[p]]))
    # Abstention at 50% coverage
    first_probes = [(p, by_p[p][0][1], by_p[p][0][2]) for p in prompts if by_p[p]]
    first_probes.sort(key=lambda t: -t[1])
    half = first_probes[:n // 2]
    abst50 = float(np.mean([t[2] for t in half])) if half else float("nan")
    # Abstention at 30% coverage
    third = first_probes[:n // 3]
    abst33 = float(np.mean([t[2] for t in third])) if third else float("nan")
    # Restart B=16, T=0.95
    def restart(B, T):
        acc = 0; nu = 0; nused = 0
        for p in prompts:
            v = by_p[p][:B]
            if not v: continue
            best = None; picked = None; n_u = 0
            for r, sc, lbl in v:
                n_u += 1
                if best is None or sc > best[1]: best = (lbl, sc)
                if sc >= T: picked = lbl; break
            if picked is None: picked = best[0]
            acc += picked; nu += 1; nused += n_u
        return acc / nu, nused / nu
    r_acc, r_used = restart(16, 0.95)

    print(f"  pass@1 (first-block of rollout 0)         : {pass1:.4f}")
    print(f"  best-of-16 (by probe)                     : {bok:.4f}   (+{100*(bok-pass1):+.1f} pp)")
    print(f"  majority-of-16 (by first-block label)     : {maj:.4f}")
    print(f"  abstain at coverage~50% (top-half probe)  : {abst50:.4f}")
    print(f"  abstain at coverage~33% (top-third)       : {abst33:.4f}")
    print(f"  probe-guided restart B=16 T=0.95          : acc={r_acc:.4f}, n_used={r_used:.2f}")

    return {
        "label": label,
        "n_prompts": n,
        "pos_frac": float(y_new.mean()),
        "auroc": bal_auc,
        "pass1": pass1,
        "best_of_16": bok,
        "majority": maj,
        "abstain50": abst50,
        "abstain33": abst33,
        "restart_acc": r_acc,
        "restart_used": r_used,
    }


def main():
    r05 = evaluate_scale(EVAL_05B, CACHE_05B, META_05B, "0.5B C_outcome")
    r15 = evaluate_scale(EVAL_15B, CACHE_15B, META_15B, "1.5B C_outcome")

    print("\n===== Cross-scale summary =====")
    print(f"  {'metric':<28} {'0.5B':>10} {'1.5B':>10}   delta")
    for k in ["pass1", "best_of_16", "majority", "abstain50", "abstain33", "restart_acc", "auroc"]:
        a = r05[k]; b = r15[k]
        print(f"  {k:<28} {a:>10.4f} {b:>10.4f}   {b-a:+.4f}")

    # Save
    os.makedirs(os.path.dirname(OUT_TXT) or ".", exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write("Cross-scale probe-applied comparison: 0.5B vs 1.5B C_outcome\n\n")
        for r in (r05, r15):
            f.write(f"=== {r['label']} ===\n")
            for k, v in r.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
        f.write("Summary table:\n")
        f.write(f"  {'metric':<28} {'0.5B':>10} {'1.5B':>10}   delta\n")
        for k in ["pass1", "best_of_16", "majority", "abstain50", "abstain33", "restart_acc", "auroc"]:
            a = r05[k]; b = r15[k]
            f.write(f"  {k:<28} {a:>10.4f} {b:>10.4f}   {b-a:+.4f}\n")
    print(f"\nwrote {OUT_TXT}")


if __name__ == "__main__":
    main()
