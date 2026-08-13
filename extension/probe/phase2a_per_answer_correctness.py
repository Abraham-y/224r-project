"""Phase 2A pre-flight: for each multi-answer C_outcome rollout, parse every
`<answer>...</answer>` block, validate it (exact-numbers usage), evaluate it,
and record per-answer correctness within each rollout.

Output:
  - extension/outputs/n500/text/17_per_answer_correctness.txt: aggregate stats
  - extension/outputs/n500/per_answer_correctness.jsonl: one row per rollout
    {prompt_idx, resp_idx, target, nums, n_blocks, per_block_correct: [bool],
     first_correct, last_correct, scored_correct}

No new compute. Reads existing eval JSONs.
"""

from __future__ import annotations

import json
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from evaluation.countdown import validate_equation, evaluate_equation


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def check_block(eq: str, target: int, nums: list[int]) -> bool:
    """True iff equation uses each `nums` exactly once AND evaluates to `target`."""
    eq = eq.strip()
    if not validate_equation(eq, nums):
        return False
    res = evaluate_equation(eq)
    if res is None:
        return False
    return abs(res - target) < 1e-5


def main():
    contam = json.load(open("extension/data/contaminated_prompt_idx.json"))
    clean = set(int(i) for i in contam["clean"])
    rows = [json.loads(l) for l in open("eval_c_outcome_n500.json")]

    out_rows: list[dict] = []
    n_multi_ans = 0
    n_total = 0
    n_no_answer = 0
    n_single_answer = 0
    n_clean_multi = 0
    transitions = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for p_idx, row in enumerate(rows):
        if p_idx not in clean:
            continue
        target = int(row["target"])
        nums = [int(x) for x in row["nums"]]
        for r_idx, resp in enumerate(row["response"]):
            n_total += 1
            blocks = _ANSWER_RE.findall(resp)
            if not blocks:
                n_no_answer += 1
                continue
            if len(blocks) == 1:
                n_single_answer += 1
                # also note the single-answer case
                ok = check_block(blocks[0], target, nums)
                out_rows.append({
                    "prompt_idx": p_idx, "resp_idx": r_idx,
                    "target": target, "nums": nums,
                    "n_blocks": 1,
                    "per_block_correct": [ok],
                    "first_correct": ok, "last_correct": ok,
                    "scored_correct": bool(row["scores"][r_idx] == 1.0),
                })
                continue
            n_multi_ans += 1
            n_clean_multi += 1
            per_block = [check_block(b, target, nums) for b in blocks]
            first_ok, last_ok = per_block[0], per_block[-1]
            transitions[(first_ok, last_ok)] += 1
            out_rows.append({
                "prompt_idx": p_idx, "resp_idx": r_idx,
                "target": target, "nums": nums,
                "n_blocks": len(blocks),
                "per_block_correct": per_block,
                "first_correct": first_ok, "last_correct": last_ok,
                "scored_correct": bool(row["scores"][r_idx] == 1.0),
            })

    summary = [
        "Phase 2A pre-flight: per-answer correctness within C_outcome multi-answer rollouts",
        f"  source: eval_c_outcome_n500.json filtered to clean-406 prompts",
        "",
        f"Total C_outcome rollouts (clean-406): {n_total}",
        f"  zero answer blocks:         {n_no_answer}  ({100*n_no_answer/n_total:.1f}%)",
        f"  single answer block:        {n_single_answer}  ({100*n_single_answer/n_total:.1f}%)",
        f"  multi-answer (>=2 blocks):  {n_multi_ans}  ({100*n_multi_ans/n_total:.1f}%)",
        "",
        "FIRST-vs-LAST answer correctness transitions (multi-answer rollouts only):",
        f"  first=T, last=T (both correct):                          {transitions[(True, True)]}",
        f"  first=T, last=F (started correct, DRIFTED TO WRONG):     {transitions[(True, False)]}",
        f"  first=F, last=T (started wrong, drifted TO CORRECT):     {transitions[(False, True)]}",
        f"  first=F, last=F (both wrong):                            {transitions[(False, False)]}",
        "",
        "Interpretation:",
        f"  (T->F) rate is the post-think drift pathology rate.",
        f"  Of {n_multi_ans} multi-answer rollouts, "
        f"{transitions[(True, False)]} ({100*transitions[(True, False)]/n_multi_ans:.1f}%) "
        f"drift away from a correct first answer.",
    ]

    txt = "\n".join(summary)
    print(txt)
    os.makedirs("extension/outputs/n500/text", exist_ok=True)
    with open("extension/outputs/n500/text/17_per_answer_correctness.txt", "w") as f:
        f.write(txt + "\n")
    os.makedirs("extension/outputs/n500", exist_ok=True)
    with open("extension/outputs/n500/per_answer_correctness.jsonl", "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
