"""Quick sanity scan over an eval JSON: how often does C_SFT_aug emit subgoals?

Run after `modal run modal_train.py eval ...` finishes and you've pulled the JSON down.
"""
from __future__ import annotations

import argparse
import json
import re


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to eval JSON (e.g. sft_aug_sanity.json).")
    args = parser.parse_args()

    text = open(args.path).read().strip()
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]

    n_rollouts = sum(len(r["response"]) for r in rows)
    with_sg = sum(1 for r in rows for resp in r["response"] if "<subgoal>" in resp)
    multi_sg = sum(1 for r in rows for resp in r["response"] if resp.count("<subgoal>") > 1)
    correct = sum(1 for r in rows for s in r["scores"] if s == 1.0)
    correct_with_sg = sum(
        1
        for r in rows
        for resp, s in zip(r["response"], r["scores"])
        if s == 1.0 and "<subgoal>" in resp
    )
    unicode_op = sum(
        1 for r in rows for resp in r["response"] if re.search(r"[−×÷]", resp)
    )

    print(f"Total rollouts: {n_rollouts}")
    print(f"  With <subgoal>:           {with_sg} ({100 * with_sg / n_rollouts:.0f}%)")
    print(f"  With >1 <subgoal>:        {multi_sg} ({100 * multi_sg / n_rollouts:.0f}%)")
    print(f"  Correct answer:           {correct} ({100 * correct / n_rollouts:.0f}%)")
    print(f"  Correct AND has subgoal:  {correct_with_sg}")
    print(f"  Unicode minus/times/div:  {unicode_op}")


if __name__ == "__main__":
    main()
