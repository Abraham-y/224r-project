"""Procedural Countdown problem generator.

Generates fresh Countdown (numbers game) problems whose validity is guaranteed
by an exhaustive arithmetic search (same pattern as Stream of Search / TinyZero).

Rules of the game (matching `asingh15/countdown_tasks_3to4`):
  - Given 3 or 4 distinct positive integers in [1, 100], a target in [10, 100].
  - The target must be reachable using each number EXACTLY once with the four
    binary operations +, -, *, / (intermediate divisions must be exact).

Algorithm
=========
1. Sample `n_numbers` distinct integers uniformly from [1, 100].
2. Enumerate all reachable real values using the recursive 24-game-style
   reducer (pick two numbers, apply each op, recurse). Collect the integer
   results that fall inside [10, 100].
3. If any reachable integers exist, pick the target uniformly at random
   from that set.
4. Otherwise, reject and resample.
5. Deduplicate the final problem set against an "already seen" set passed
   as input (the 50 problems in asingh15's test split).

Default outputs `extension/data/countdown_eval_500.jsonl` (n=500). Each row:
    {"nums": [a, b, c[, d]], "target": t, "prompt": "...", "ground_truth": {...}}

The `prompt` is built with the same template as asingh15's test split so
downstream evaluation code can ingest the JSONL drop-in.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Iterable


# Same wrapper as asingh15/countdown_tasks_3to4. Verified by reading
# eval.json's prompt[0] field.
PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the "
    "answer.\nUser: Using the numbers {nums_str}, create an equation that "
    "equals {target}. You can use basic arithmetic operations (+, -, *, /) "
    "and each number can only be used once. Show your work in <think> "
    "</think> tags. And return the final answer in <answer> </answer> "
    "tags, for example <answer> (1 + 2) / 3 </answer>.\n"
    "Assistant: Let me solve this step by step.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def reachable_integers(nums: list[float]) -> set[int]:
    """Return every integer value reachable from `nums` using each number once.

    Standard 24-game style brute force: pick two numbers, combine with each of
    the four ops, recurse on the reduced list. We track only the integer
    results inside [-1000, 1000] to keep the recursion bounded.
    """
    if len(nums) == 1:
        v = nums[0]
        return {int(round(v))} if abs(v - round(v)) < 1e-9 else set()

    out: set[int] = set()
    n = len(nums)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = nums[i], nums[j]
            rest = [nums[k] for k in range(n) if k != i and k != j]
            # commutative-symmetric: only do a+b once
            if i < j:
                for combined in (a + b, a * b):
                    out |= reachable_integers(rest + [combined])
            # non-commutative: do both orders
            for combined in (a - b,):
                out |= reachable_integers(rest + [combined])
            if abs(b) > 1e-9 and abs((a / b) - round(a / b)) < 1e-9:
                out |= reachable_integers(rest + [a / b])
    return out


def sample_problem(rng: random.Random, n_numbers: int,
                    target_min: int = 10, target_max: int = 100) -> tuple[list[int], int] | None:
    """One generate-and-validate attempt. Returns None if no valid target."""
    nums = rng.sample(range(1, 101), n_numbers)
    reachable = reachable_integers([float(x) for x in nums])
    valid = sorted(t for t in reachable if target_min <= t <= target_max)
    if not valid:
        return None
    target = rng.choice(valid)
    return nums, target


def already_seen_keys(rows: Iterable[dict]) -> set[tuple[tuple[int, ...], int]]:
    keys: set[tuple[tuple[int, ...], int]] = set()
    for r in rows:
        nums = tuple(sorted(int(x) for x in r["nums"]))
        keys.add((nums, int(r["target"])))
    return keys


def format_prompt(nums: list[int], target: int) -> str:
    # asingh15/countdown_tasks_3to4 renders `nums` with NumPy's default
    # array2string (right-justified to the widest number). Examples from
    # eval.json: "[44 19 35]", "[14 45  9  1]", "[83 78  1 39]". We
    # reproduce that exactly so the model sees the same input distribution.
    width = max(len(str(n)) for n in nums)
    nums_str = "[" + " ".join(f"{n:>{width}d}" for n in nums) + "]"
    return PROMPT_TEMPLATE.format(nums_str=nums_str, target=target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="number of problems to emit")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--three_to_four_split", type=float, default=0.5,
                        help="fraction of problems with len(nums)=3")
    parser.add_argument("--seen_jsonl", default="eval.json",
                        help="existing eval JSON whose (nums, target) tuples must NOT be reused")
    parser.add_argument("--out", default="extension/data/countdown_eval_500.jsonl")
    parser.add_argument("--target_min", type=int, default=10)
    parser.add_argument("--target_max", type=int, default=100)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.seen_jsonl and os.path.exists(args.seen_jsonl):
        with open(args.seen_jsonl) as f:
            seen_rows = [json.loads(l) for l in f if l.strip()]
        seen = already_seen_keys(seen_rows)
        print(f"[gen] loaded {len(seen)} (nums, target) tuples to avoid from {args.seen_jsonl}")
    else:
        seen = set()
        print(f"[gen] no seen-jsonl supplied / file missing -- not deduplicating")

    # Target counts for the 3/4-number mix.
    n_three = int(round(args.n * args.three_to_four_split))
    n_four = args.n - n_three
    print(f"[gen] sampling {n_three} 3-number problems and {n_four} 4-number problems")

    out_rows: list[dict] = []
    seen_emit: set[tuple[tuple[int, ...], int]] = set(seen)

    for n_numbers, want in ((3, n_three), (4, n_four)):
        emitted = 0
        attempts = 0
        rejects_no_target = 0
        rejects_dup = 0
        while emitted < want:
            attempts += 1
            if attempts > want * 50:
                print(f"[gen] WARNING: {attempts} attempts to emit {emitted}/{want} "
                      f"n={n_numbers}-number problems; stopping")
                break
            result = sample_problem(rng, n_numbers, args.target_min, args.target_max)
            if result is None:
                rejects_no_target += 1
                continue
            nums, target = result
            key = (tuple(sorted(nums)), int(target))
            if key in seen_emit:
                rejects_dup += 1
                continue
            seen_emit.add(key)
            out_rows.append({
                "nums": list(nums),
                "target": int(target),
                "prompt": format_prompt(nums, target),
                "ground_truth": {"numbers": list(nums), "target": int(target)},
            })
            emitted += 1
        print(f"[gen]   n_numbers={n_numbers}: emitted {emitted}/{want} "
              f"(attempts={attempts}, no-target rejects={rejects_no_target}, "
              f"dedup rejects={rejects_dup})")

    rng.shuffle(out_rows)

    # Distribution check vs. seen set.
    seen_targets = [k[1] for k in seen]
    new_targets = [r["target"] for r in out_rows]
    print(f"\n[gen] target distribution comparison")
    print(f"  seen:  n={len(seen_targets)}  min={min(seen_targets) if seen_targets else 'NA'}  "
          f"max={max(seen_targets) if seen_targets else 'NA'}  "
          f"mean={sum(seen_targets) / max(1, len(seen_targets)):.1f}")
    print(f"  new:   n={len(new_targets)}  min={min(new_targets)}  max={max(new_targets)}  "
          f"mean={sum(new_targets) / len(new_targets):.1f}")

    seen_nums_lens = Counter(len(r["nums"]) for r in seen_rows) if args.seen_jsonl and os.path.exists(args.seen_jsonl) else Counter()
    new_nums_lens = Counter(len(r["nums"]) for r in out_rows)
    print(f"  seen len(nums) dist: {dict(seen_nums_lens)}")
    print(f"  new  len(nums) dist: {dict(new_nums_lens)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n[gen] wrote {len(out_rows)} problems to {args.out}")


if __name__ == "__main__":
    main()
