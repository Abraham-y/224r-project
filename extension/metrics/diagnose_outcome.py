"""C_outcome diagnostic. One rollout per problem.

Reads an eval JSON (produced by evaluation/countdown_eval.py), picks response[0]
per prompt, and reports a per-rollout breakdown plus a sample of rollouts that
span the failure modes the user wants to inspect.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.countdown import evaluate_equation, extract_solution, validate_equation


CONFIDENT_PATTERNS = (
    "this works", "it works", "this is correct", "verified", "confirmed",
    "the answer is", "got it", "found it", "i've found", "found the solution",
    "solution found", "final answer", "perfect", "exactly", "this gives",
    "we have it", "i found it", "solved",
)


# ---------------------------------------------------------------------------
# Per-rollout features.
# ---------------------------------------------------------------------------


@dataclass
class RolloutFeatures:
    target: int
    nums: list[int]
    response: str
    has_answer: bool
    answer_raw: str | None              # raw text inside the last <answer>...</answer>
    well_formed: bool                   # parses as a Python arithmetic AST
    answer_value: float | None          # numerical value if well-formed
    uses_each_once: bool                # exact-multiset Countdown constraint
    correct: bool                       # uses_each_once AND value == target
    ends_cleanly: bool                  # response stopped near a </answer>, not mid-token
    cot_endorsed_value: float | None    # value of the last endorsed expression in <think>
    cot_answer_match: bool | None       # None if either side unparsable
    has_confident_assertion: bool       # any of CONFIDENT_PATTERNS appears in <think>
    answer_tag_count: int               # detects multi-answer / repetition


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_EQ_TO_TARGET_RE = re.compile(r"([-+*/().\s\d]+?)=\s*(-?\d+(?:\.\d+)?)")


def parse_arithmetic_ast(text: str) -> ast.AST | None:
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return None
    # Reject anything that's not pure arithmetic on numeric constants.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp,
                             ast.Constant, ast.Add, ast.Sub, ast.Mult,
                             ast.Div, ast.USub, ast.UAdd)):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            continue
        return None
    return tree


def safe_eval(expr: str) -> float | None:
    if not re.match(r"^[\d+\-*/().\s]+$", expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": None}, {}))
    except Exception:
        return None


def extract_think(response: str) -> str:
    m = _THINK_RE.search(response)
    if m:
        return m.group(1)
    # No closing </think>? Return everything up to first <answer>.
    if "<answer>" in response:
        return response[: response.index("<answer>")]
    return response


def extract_last_answer_block(response: str) -> tuple[str | None, int]:
    matches = list(_ANSWER_RE.finditer(response))
    if not matches:
        return None, response.count("<answer>")
    return matches[-1].group(1).strip().rstrip("."), len(matches)


def ends_cleanly(response: str) -> bool:
    rstrip = response.rstrip()
    return rstrip.endswith("</answer>") or rstrip[-32:].count("</answer>") >= 1


def find_cot_endorsed_value(think_text: str, target: int) -> tuple[str | None, float | None]:
    """Find the last expression inside <think> that evaluates to the target.

    Heuristic: scan all 'expr = stated' patterns; if expr evaluates to the
    target (regardless of stated), treat that expr as endorsed.
    """
    last_expr: str | None = None
    last_value: float | None = None
    for match in _EQ_TO_TARGET_RE.finditer(think_text):
        expr = match.group(1).strip()
        if not any(op in expr for op in "+-*/"):
            continue
        value = safe_eval(expr)
        if value is None:
            continue
        if abs(value - target) < 1e-5:
            last_expr = expr
            last_value = value
    return last_expr, last_value


def multiset_uses_each_once(nums_in_expr: list[int], available: list[int]) -> bool:
    return Counter(nums_in_expr) == Counter(available)


def extract_int_literals(expr: str) -> list[int]:
    # Match positive integer literals only (Countdown has no negatives in inputs).
    return [int(n) for n in re.findall(r"\b\d+\b", expr)]


def featurize(target: int, nums: list[int], response: str) -> RolloutFeatures:
    answer_raw, answer_tag_count = extract_last_answer_block(response)
    has_answer = answer_raw is not None and len(answer_raw) > 0

    well_formed = False
    answer_value: float | None = None
    uses_each_once = False
    correct = False
    if has_answer:
        if parse_arithmetic_ast(answer_raw) is not None:
            v = safe_eval(answer_raw)
            if v is not None:
                well_formed = True
                answer_value = v
                literals = extract_int_literals(answer_raw)
                uses_each_once = multiset_uses_each_once(literals, list(nums))
                correct = uses_each_once and abs(v - target) < 1e-5

    think = extract_think(response)
    cot_expr, cot_value = find_cot_endorsed_value(think, target)
    if cot_value is None or answer_value is None:
        cot_answer_match: bool | None = None
    else:
        cot_answer_match = abs(cot_value - answer_value) < 1e-5

    low = think.lower()
    has_confident_assertion = any(p in low for p in CONFIDENT_PATTERNS)

    return RolloutFeatures(
        target=target,
        nums=list(nums),
        response=response,
        has_answer=has_answer,
        answer_raw=answer_raw,
        well_formed=well_formed,
        answer_value=answer_value,
        uses_each_once=uses_each_once,
        correct=correct,
        ends_cleanly=ends_cleanly(response),
        cot_endorsed_value=cot_value,
        cot_answer_match=cot_answer_match,
        has_confident_assertion=has_confident_assertion,
        answer_tag_count=answer_tag_count,
    )


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def load_rollouts(path: str) -> list[RolloutFeatures]:
    rows = [json.loads(line) for line in open(path).read().strip().splitlines() if line.strip()]
    out: list[RolloutFeatures] = []
    for row in rows:
        # One rollout per problem: take response[0].
        target = int(row["target"])
        nums = list(row["nums"])
        response = row["response"][0]
        out.append(featurize(target, nums, response))
    return out


def print_headline(rollouts: list[RolloutFeatures]) -> None:
    n = len(rollouts)
    n_correct = sum(r.correct for r in rollouts)
    n_well_formed = sum(r.well_formed for r in rollouts)
    n_has_answer = sum(r.has_answer for r in rollouts)
    n_uses_each_once = sum(r.uses_each_once for r in rollouts)
    n_cot_consistent = sum(1 for r in rollouts if r.cot_answer_match is True)
    n_cot_compared = sum(1 for r in rollouts if r.cot_answer_match is not None)
    n_confident_wrong = sum(1 for r in rollouts if not r.correct and r.has_confident_assertion)

    def pct(num: int, den: int) -> str:
        if den == 0:
            return "n/a"
        return f"{num}/{den} ({100 * num / den:.0f}%)"

    print("=" * 60)
    print(f"C_outcome diagnostic — 1 rollout per problem ({n} problems)")
    print("=" * 60)
    print(f"  Overall accuracy:                {pct(n_correct, n)}")
    print(f"  Answer well-formed:              {pct(n_well_formed, n)}")
    print(f"  Any answer at all:               {pct(n_has_answer, n)}")
    print(f"  Uses each number exactly once:   {pct(n_uses_each_once, n)}")
    print(f"  CoT-answer consistency:          {pct(n_cot_consistent, n_cot_compared)}   "
          f"(denom = rollouts where both sides parsed)")
    print(f"  Confident-wrong rollouts:        {pct(n_confident_wrong, n - n_correct)}   "
          f"(over wrong rollouts)")


def is_degenerate(r: RolloutFeatures) -> bool:
    """Wrong + ugly: many <answer> tags, ran off the end, or hit token limit."""
    if r.answer_tag_count >= 3:
        return True
    if not r.ends_cleanly:
        return True
    return False


def sample_for_report(rollouts: list[RolloutFeatures]) -> dict[str, list[RolloutFeatures]]:
    correct = [r for r in rollouts if r.correct]
    wrong = [r for r in rollouts if not r.correct]
    wrong_degenerate = [r for r in wrong if is_degenerate(r)]
    wrong_coherent = [r for r in wrong if not is_degenerate(r)]
    confident_wrong = [r for r in wrong if r.has_confident_assertion]
    return {
        "correct": correct[:5],
        "wrong_coherent": wrong_coherent[:3],
        "wrong_degenerate": wrong_degenerate[:2],
        "confident_wrong": confident_wrong[:5],
    }


def dump_rollout(r: RolloutFeatures, label: str) -> None:
    print()
    print("-" * 78)
    print(f"[{label}]  target={r.target}  nums={r.nums}")
    print(f"           answer_raw={r.answer_raw!r}  answer_value={r.answer_value}")
    print(f"           well_formed={r.well_formed}  uses_each_once={r.uses_each_once}  "
          f"correct={r.correct}")
    print(f"           cot_endorsed_value={r.cot_endorsed_value}  "
          f"cot_match={r.cot_answer_match}  confident={r.has_confident_assertion}  "
          f"answer_tag_count={r.answer_tag_count}  ends_cleanly={r.ends_cleanly}")
    print("-" * 78)
    print(r.response)
    print("-" * 78)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", default="eval.json")
    args = parser.parse_args()

    rollouts = load_rollouts(args.eval_json)
    print_headline(rollouts)

    buckets = sample_for_report(rollouts)
    print()
    print("=" * 60)
    print("Samples (raw rollouts)")
    print("=" * 60)
    for i, r in enumerate(buckets["correct"]):
        dump_rollout(r, f"CORRECT {i + 1}/5")
    for i, r in enumerate(buckets["wrong_coherent"]):
        dump_rollout(r, f"WRONG-COHERENT-COT {i + 1}/3")
    for i, r in enumerate(buckets["wrong_degenerate"]):
        dump_rollout(r, f"WRONG-DEGENERATE-COT {i + 1}/2")
    print()
    print("=" * 60)
    print("Confident-wrong (wrong answer + high-confidence assertion in <think>)")
    print("=" * 60)
    if not buckets["confident_wrong"]:
        print("(none)")
    for i, r in enumerate(buckets["confident_wrong"]):
        dump_rollout(r, f"CONFIDENT-WRONG {i + 1}/5")


if __name__ == "__main__":
    main()
