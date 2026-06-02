"""Probe analysis on the firstanswer C_outcome' checkpoint.

Compute corrected-label AUROCs at pre_answer, assertion, neutral × L16.
Compare position-gap (pre - assertion) to vanilla C_outcome.

Vanilla C_outcome corrected-label AUROCs (from §2.1):
  pre_answer: 0.982
  assertion:  0.896
  gap:       +0.086
  rambling rate: 87%

Firstanswer eval (this work): mean blocks/rollout 6.36 (vs vanilla 6.78).
Question: did the position-gap change?
"""

from __future__ import annotations
import json, re
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

CACHE_DIR = "extension/cache/probe_cache_firstanswer"
EVAL = "eval_c_firstanswer_n500.json"
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


def main():
    clean = set(json.load(open("extension/data/contaminated_prompt_idx.json"))["clean"])
    labs = first_block_labels(EVAL)
    print(f"clean-406 prompts: {len(clean)}, labeled rollouts: {len(labs)}")

    results = {}
    for kind in ["pre_answer", "assertion", "neutral"]:
        cache = f"{CACHE_DIR}/C_firstanswer_l16_{kind}"
        with np.load(f"{cache}.npz") as d: X = d["X"]
        meta = json.load(open(f"{cache}.meta.json"))
        groups = np.array([int(m["prompt_idx"]) for m in meta])
        # Filter to clean-406
        mask = np.array([int(m["prompt_idx"]) in clean for m in meta])
        Xf = X[mask]; gf = groups[mask]; metaf = [meta[i] for i in range(len(meta)) if mask[i]]
        y = np.array([labs.get((int(x['prompt_idx']), int(x['resp_idx'])), 0) for x in metaf], dtype=np.int32)
        # GroupKFold(5) held-out scores
        sc = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(Xf, y, gf):
            pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
            pipe.fit(Xf[tr], y[tr])
            sc[te] = pipe.predict_proba(Xf[te])[:, 1]
        # Balanced AUROC
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
        nb = min(len(pos), len(neg))
        if nb == 0:
            auc = float("nan")
        else:
            rng = np.random.RandomState(0)
            idx = np.concatenate([rng.choice(pos, nb, replace=False), rng.choice(neg, nb, replace=False)])
            auc = float(roc_auc_score(y[idx], sc[idx]))
        print(f"  {kind:12} AUROC (L16, corrected, clean-406): {auc:.3f}  (n={len(y)}, pos%={y.mean():.3f})")
        results[kind] = auc

    gap = results["pre_answer"] - results["assertion"]
    print(f"\nGap pre_answer - assertion (firstanswer): {gap:+.4f}")
    print(f"Vanilla C_outcome gap (from §2.1):        +0.086")
    print(f"Delta vs vanilla:                          {gap - 0.086:+.4f}")


if __name__ == "__main__":
    main()
