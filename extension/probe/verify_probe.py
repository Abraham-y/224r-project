"""Sanity check: does the pickled probe actually discriminate correct from wrong
on the natural rollout distribution? Apply it to cached hidden states + show
example rollouts at low and high probe scores.

We're worried the probe might be saturated or broken. This script:
  1. Loads the cached C_outcome pre_answer L16 hidden states
  2. Loads the pickled probe
  3. Computes scores
  4. Sorts rollouts by score
  5. Prints the lowest-scored and highest-scored rollouts with their actual
     first-<answer>-block correctness (computed locally via verifier rules).
"""

from __future__ import annotations
import json
import pickle
import re

import numpy as np

CACHE_NPZ = "extension/cache/probe_cache_n500_clean406/C_outcome_l16_pre_answer.npz"
META = CACHE_NPZ.replace(".npz", ".meta.json")
EVAL = "eval_c_outcome_n500.json"
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


def show_rollout_excerpt(text, max_chars=400):
    text = text.replace("\n", "\\n")
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def main(probe_path, label):
    print(f"\n========== {label} ==========")
    with open(probe_path, "rb") as f:
        probe = pickle.load(f)
    with np.load(CACHE_NPZ) as d: X = d["X"]
    meta = json.load(open(META))
    rows = [json.loads(l) for l in open(EVAL) if l.strip()]

    # Score every cached hidden state
    scores = probe.predict_proba(X)[:, 1]
    print(f"n_cached: {len(scores)}")
    print(f"score distribution: min={scores.min():.4f} max={scores.max():.4f}")
    print(f"  mean={scores.mean():.4f} std={scores.std():.4f}")
    print(f"  quantiles: 5%={np.quantile(scores, 0.05):.3f}, 25%={np.quantile(scores, 0.25):.3f}, 50%={np.quantile(scores, 0.50):.3f}, 75%={np.quantile(scores, 0.75):.3f}, 95%={np.quantile(scores, 0.95):.3f}")
    print(f"  fraction with score>0.95: {(scores > 0.95).mean():.3f}")
    print(f"  fraction with score<0.05: {(scores < 0.05).mean():.3f}")

    # Compute first-block + rollout-final correctness for each meta entry
    first_correct = []
    last_correct = []
    rollout_texts = []
    for entry in meta:
        p = int(entry["prompt_idx"]); r = int(entry["resp_idx"])
        row = rows[p]
        target = int(row["target"]); nums = list(row["nums"])
        resp = row["response"][r]
        rollout_texts.append(resp)
        matches = list(_ANSWER_OPEN_RE.finditer(resp))
        first_correct.append(int(bool(matches) and check_block(matches[0].group(1), target, nums)))
        last_correct.append(int(bool(matches) and check_block(matches[-1].group(1), target, nums)))
    first_correct = np.array(first_correct); last_correct = np.array(last_correct)
    print(f"first-block accuracy on cached: {first_correct.mean():.3f}")
    print(f"last-block (rollout-final) accuracy on cached: {last_correct.mean():.3f}")

    # AUROC sanity
    from sklearn.metrics import roc_auc_score
    auc_first = roc_auc_score(first_correct, scores) if 0 < first_correct.mean() < 1 else float("nan")
    auc_last = roc_auc_score(last_correct, scores) if 0 < last_correct.mean() < 1 else float("nan")
    print(f"probe AUROC vs first-block: {auc_first:.3f}")
    print(f"probe AUROC vs last-block:  {auc_last:.3f}")

    # Sort by score; show extremes
    order = np.argsort(scores)
    print("\n--- 5 LOWEST probe scores ---")
    for i in order[:5]:
        sc = scores[i]
        p = int(meta[i]["prompt_idx"]); r = int(meta[i]["resp_idx"])
        first_eq = ""
        ms = list(_ANSWER_OPEN_RE.finditer(rollout_texts[i]))
        if ms: first_eq = ms[0].group(1).strip()
        target = rows[p]["target"]; nums = rows[p]["nums"]
        print(f"  score={sc:.4f}  first_eq={first_eq[:50]!r}  target={target}  nums={nums}")
        print(f"    first_correct={first_correct[i]}  last_correct={last_correct[i]}  n_blocks={len(ms)}")
        print(f"    excerpt: {show_rollout_excerpt(rollout_texts[i], 250)}")

    print("\n--- 5 HIGHEST probe scores ---")
    for i in order[-5:]:
        sc = scores[i]
        p = int(meta[i]["prompt_idx"]); r = int(meta[i]["resp_idx"])
        first_eq = ""
        ms = list(_ANSWER_OPEN_RE.finditer(rollout_texts[i]))
        if ms: first_eq = ms[0].group(1).strip()
        target = rows[p]["target"]; nums = rows[p]["nums"]
        print(f"  score={sc:.4f}  first_eq={first_eq[:50]!r}  target={target}  nums={nums}")
        print(f"    first_correct={first_correct[i]}  last_correct={last_correct[i]}  n_blocks={len(ms)}")


if __name__ == "__main__":
    main("extension/cache/steering/probe_pipeline_C_outcome_l16_pre_answer.pkl",
         "FIRST-BLOCK probe (AUROC 0.98)")
    main("extension/cache/steering/probe_pipeline_C_outcome_l16_pre_answer_lastblock.pkl",
         "LAST-BLOCK probe (AUROC 0.90)")
