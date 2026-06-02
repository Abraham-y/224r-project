"""Mean blocks/rollout + first-block accuracy diagnostic on firstanswer_rloo vs vanilla C_outcome.

The single load-bearing test of the rambling-as-reward-hack causal hypothesis.
"""

from __future__ import annotations
import json, re, os
from collections import Counter

import numpy as np

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


def analyze(eval_path, label, clean_idx=None):
    rows = [json.loads(l) for l in open(eval_path) if l.strip()]
    n_prompts = 0
    block_counts = []
    first_correct_count = 0
    last_correct_count = 0
    n_rollouts = 0
    block_count_hist = Counter()
    n_zero_block = 0
    n_one_block = 0
    n_multi_block = 0
    lengths = []

    for p_idx, row in enumerate(rows):
        if clean_idx is not None and p_idx not in clean_idx: continue
        n_prompts += 1
        target = int(row["target"]); nums = list(row["nums"])
        for r_idx, resp in enumerate(row["response"]):
            n_rollouts += 1
            lengths.append(len(resp))
            matches = list(_ANSWER_OPEN_RE.finditer(resp))
            n_blocks = len(matches)
            block_counts.append(n_blocks)
            block_count_hist[n_blocks] += 1
            if n_blocks == 0:
                n_zero_block += 1
            elif n_blocks == 1:
                n_one_block += 1
            else:
                n_multi_block += 1
            # First and last block correctness
            if matches:
                first_correct_count += int(check_block(matches[0].group(1), target, nums))
                last_correct_count += int(check_block(matches[-1].group(1), target, nums))

    block_counts = np.array(block_counts)
    print(f"\n=== {label} (n_prompts={n_prompts}, n_rollouts={n_rollouts}) ===")
    print(f"  mean blocks/rollout: {block_counts.mean():.3f}")
    print(f"  median blocks: {int(np.median(block_counts))}")
    print(f"  max blocks: {block_counts.max()}")
    print(f"  zero-block rollouts: {n_zero_block} ({100*n_zero_block/n_rollouts:.1f}%)")
    print(f"  one-block (clean) rollouts: {n_one_block} ({100*n_one_block/n_rollouts:.1f}%)")
    print(f"  multi-block rollouts: {n_multi_block} ({100*n_multi_block/n_rollouts:.1f}%)")
    print(f"  block count distribution: {dict(sorted(block_count_hist.items())[:10])}{'...' if max(block_count_hist) > 9 else ''}")
    print(f"  first-block accuracy: {first_correct_count/n_rollouts:.4f}")
    print(f"  last-block accuracy: {last_correct_count/n_rollouts:.4f}")
    print(f"  mean response length: {np.mean(lengths):.0f}")
    return {
        "label": label,
        "mean_blocks": float(block_counts.mean()),
        "pct_multi": 100 * n_multi_block / n_rollouts,
        "pct_one": 100 * n_one_block / n_rollouts,
        "pct_zero": 100 * n_zero_block / n_rollouts,
        "first_acc": first_correct_count / n_rollouts,
        "last_acc": last_correct_count / n_rollouts,
        "mean_len": float(np.mean(lengths)),
    }


# Load clean-406 index
clean = set(json.load(open("extension/data/contaminated_prompt_idx.json"))["clean"])
print(f"clean-406 size: {len(clean)}")

r_vanilla = analyze("eval_c_outcome_n500.json", "vanilla C_outcome (last-block reward)", clean_idx=clean)
r_first = analyze("eval_c_firstanswer_n500.json", "firstanswer C_outcome' (first-block reward)", clean_idx=clean)
r_sft = analyze("eval_c_sft_n500.json", "C_SFT (no RL)", clean_idx=clean)

print("\n=== SUMMARY ===")
print(f"{'metric':<28} {'C_SFT':>10} {'vanilla':>10} {'firstans':>10}")
for k in ["mean_blocks", "pct_multi", "pct_one", "first_acc", "last_acc", "mean_len"]:
    print(f"  {k:<28} {r_sft[k]:>10.4f} {r_vanilla[k]:>10.4f} {r_first[k]:>10.4f}")
