"""Inject <subgoal> declarations into the warm-start SFT traces.

Pipeline (per extension.md §4.4):

    1. Load Asap7772/cog_behav_all_strategies.
    2. Parse the query to recover the available numbers and target.
    3. Parse the completion's <answer>...</answer> equation into an AST.
    4. Walk the AST bottom-up; each BinOp becomes a candidate subgoal
         (target = value of that subtree,
          inputs = values of its two operands -- which can themselves be
                   intermediates produced by prior subgoals).
    5. For each candidate, locate the line in the completion where that
       arithmetic step is explicitly written ("36 - 32 = 4"), and inject
       a "<subgoal> reach <value> from [<inputs>] </subgoal>" line just
       before it.
    6. Validate that:
         a) the answer evaluates to the stated target,
         b) every subgoal we emit is reachable + uses only allowed numbers,
         c) every subgoal we emit can be located in the completion text.
       Drop traces where any of these fail.

The output is a JSONL file with the same {query, completion} schema as the
input, plus an optional HuggingFace push.

Person A. Day 3. Pure-CPU; runs on a laptop in a few minutes for the full
~490k-row dataset.
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
from typing import Iterable, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from extension.subgoal.verifier import is_reachable


# ---------------------------------------------------------------------------
# Query / completion parsing.
# ---------------------------------------------------------------------------

_NUMS_RE = re.compile(r"Using the numbers\s*\[([\d,\s\-]+)\]")
_TARGET_RE = re.compile(r"equals\s*(-?\d+)")
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def parse_query(query: str) -> tuple[list[int], int] | tuple[None, None]:
    nums_m = _NUMS_RE.search(query)
    target_m = _TARGET_RE.search(query)
    if not nums_m or not target_m:
        return None, None
    try:
        nums = [int(x.strip()) for x in nums_m.group(1).split(",") if x.strip()]
        target = int(target_m.group(1))
    except ValueError:
        return None, None
    return nums, target


def extract_answer_equation(completion: str) -> str | None:
    m = _ANSWER_RE.search(completion)
    if not m:
        return None
    eq = m.group(1).strip().rstrip(".")
    # ast.parse rejects unbalanced parens etc; surface those upstream.
    return eq or None


# ---------------------------------------------------------------------------
# AST walk -> ordered list of intermediates.
# ---------------------------------------------------------------------------

@dataclass
class Intermediate:
    value: int            # target of this subgoal
    inputs: tuple[int, ...]  # ints used (problem nums or prior intermediates)


_OP_MAP = {
    ast.Add: ("+", lambda a, b: a + b),
    ast.Sub: ("-", lambda a, b: a - b),
    ast.Mult: ("*", lambda a, b: a * b),
    ast.Div: ("/", lambda a, b: a / b if b != 0 else None),
}


def _collect_intermediates(node: ast.expr) -> tuple[float, list[Intermediate]]:
    """Bottom-up walk. Returns (subtree_value, list of intermediates emitted).

    Intermediates are appended in the order they're computed (deepest first).
    Each entry's `inputs` are the immediate-operand values of its BinOp, which
    may themselves be values of prior intermediates -- this mirrors the
    "reach 68 from [60, 8]" pattern in §4.3.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value), []
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v, sub = _collect_intermediates(node.operand)
        return -v, sub
    if isinstance(node, ast.BinOp):
        left_val, left_sub = _collect_intermediates(node.left)
        right_val, right_sub = _collect_intermediates(node.right)
        op_type = type(node.op)
        if op_type not in _OP_MAP:
            raise ValueError(f"unsupported op {op_type.__name__}")
        _, fn = _OP_MAP[op_type]
        val = fn(left_val, right_val)
        if val is None:
            raise ValueError("division by zero")
        intermediates = left_sub + right_sub
        intermediates.append(Intermediate(
            value=_to_int(val),
            inputs=(_to_int(left_val), _to_int(right_val)),
        ))
        return val, intermediates
    raise ValueError(f"unsupported node {type(node).__name__}")


def _to_int(x: float) -> int:
    """Snap near-integers to int; bail with a ValueError on truly fractional values."""
    if abs(x - round(x)) > 1e-6:
        raise ValueError(f"non-integer intermediate value {x}")
    return int(round(x))


# ---------------------------------------------------------------------------
# Locate the textual line that computes each intermediate.
# ---------------------------------------------------------------------------

def _build_eq_pattern(value: int, inputs: tuple[int, ...]) -> re.Pattern[str]:
    """Match `<a> <op> <b> = <value>` where {a, b} == set(inputs), any operator."""
    a, b = inputs
    a_or_b = rf"(?:{re.escape(str(a))}|{re.escape(str(b))})"
    return re.compile(
        rf"({a_or_b})\s*([+\-*/])\s*({a_or_b})\s*=\s*{re.escape(str(value))}\b"
    )


def _locate_intermediate(completion: str, inter: Intermediate) -> int | None:
    """Return the char offset to the start of the line that computes `inter`,
    or ``None`` if no such line is present."""
    expected = sorted(inter.inputs)
    pattern = _build_eq_pattern(inter.value, inter.inputs)
    candidates: list[int] = []
    for m in pattern.finditer(completion):
        captured = sorted([int(m.group(1)), int(m.group(3))])
        if captured != expected:
            continue
        # Snap to the start of the line containing this match.
        line_start = completion.rfind("\n", 0, m.start()) + 1
        candidates.append(line_start)
    if not candidates:
        return None
    return min(candidates)


def _format_subgoal(inter: Intermediate) -> str:
    inputs_str = ", ".join(str(x) for x in inter.inputs)
    return f"<subgoal> reach {inter.value} from [{inputs_str}] </subgoal>\n"


# ---------------------------------------------------------------------------
# Per-trace augmentation.
# ---------------------------------------------------------------------------

@dataclass
class AugmentResult:
    augmented_completion: str | None
    status: str           # "ok" or a short failure code
    n_subgoals: int       # how many tags we injected (0 on failure)


def _multiset_subset(needle: Iterable[int], haystack: Iterable[int]) -> bool:
    n, h = Counter(needle), Counter(haystack)
    return all(h[k] >= v for k, v in n.items())


def augment_trace(query: str, completion: str) -> AugmentResult:
    nums, target = parse_query(query)
    if nums is None:
        return AugmentResult(None, "query_parse_failed", 0)

    eq_text = extract_answer_equation(completion)
    if not eq_text:
        return AugmentResult(None, "no_answer", 0)

    try:
        tree = ast.parse(eq_text, mode="eval").body
    except SyntaxError:
        return AugmentResult(None, "answer_parse_failed", 0)

    try:
        eq_value, intermediates = _collect_intermediates(tree)
    except ValueError:
        return AugmentResult(None, "ast_walk_failed", 0)

    if abs(eq_value - target) >= 1e-5:
        return AugmentResult(None, "answer_does_not_hit_target", 0)
    if not intermediates:
        return AugmentResult(None, "no_intermediates", 0)  # single-number answer

    # Validate every intermediate against (problem nums U prior intermediate values).
    allowed = list(nums)
    for inter in intermediates:
        if not _multiset_subset(inter.inputs, allowed):
            return AugmentResult(None, "intermediate_uses_unknown_input", 0)
        if not is_reachable(inter.value, list(inter.inputs)):
            return AugmentResult(None, "intermediate_not_reachable", 0)
        allowed.append(inter.value)

    # Locate the explanatory line for each intermediate in the completion.
    located: list[tuple[int, Intermediate]] = []
    for inter in intermediates:
        pos = _locate_intermediate(completion, inter)
        if pos is None:
            return AugmentResult(None, "intermediate_not_located", 0)
        located.append((pos, inter))

    # Insert tags from latest position to earliest so prior offsets stay valid.
    augmented = completion
    for pos, inter in sorted(located, key=lambda x: -x[0]):
        augmented = augmented[:pos] + _format_subgoal(inter) + augmented[pos:]
    return AugmentResult(augmented, "ok", len(located))


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

@dataclass
class AugmentStats:
    n_seen: int
    n_augmented: int
    mean_subgoals: float
    failure_counts: dict[str, int]


def augment_dataset(
    input_dataset: str,
    output_jsonl: str,
    output_dataset: str | None,
    limit: int | None,
    push: bool,
    print_samples: int,
) -> AugmentStats:
    from datasets import Dataset, load_dataset  # local import; keeps top of file lazy

    ds = load_dataset(input_dataset, split="train")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    rows: list[dict] = []
    failures: dict[str, int] = {}
    n_seen = 0
    n_subgoal_total = 0
    samples_printed = 0

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    for ex in ds:
        n_seen += 1
        result = augment_trace(ex["query"], ex["completion"])
        if result.status != "ok":
            failures[result.status] = failures.get(result.status, 0) + 1
            continue
        rows.append({"query": ex["query"], "completion": result.augmented_completion})
        n_subgoal_total += result.n_subgoals
        if samples_printed < print_samples:
            print(f"--- sample {samples_printed} ({result.n_subgoals} subgoals) ---")
            print(result.augmented_completion[:600])
            print("...")
            samples_printed += 1

    with open(output_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    if push and output_dataset is not None:
        out = Dataset.from_list(rows)
        out.push_to_hub(output_dataset)

    n_augmented = len(rows)
    return AugmentStats(
        n_seen=n_seen,
        n_augmented=n_augmented,
        mean_subgoals=(n_subgoal_total / n_augmented) if n_augmented else 0.0,
        failure_counts=failures,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", default="Asap7772/cog_behav_all_strategies")
    parser.add_argument("--output_jsonl", default="extension/outputs/cog_behav_subgoal_augmented.jsonl")
    parser.add_argument("--output_dataset", default="prismane16/cog_behav_subgoal_augmented",
                        help="HF repo to push to (only used with --push). Default is the team repo.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only this many rows (useful for dry runs).")
    parser.add_argument("--push", action="store_true",
                        help="Push the result to --output_dataset. Requires HF_TOKEN.")
    parser.add_argument("--print_samples", type=int, default=3,
                        help="Print this many successfully-augmented samples for spot-checking.")
    args = parser.parse_args()

    stats = augment_dataset(
        input_dataset=args.input_dataset,
        output_jsonl=args.output_jsonl,
        output_dataset=args.output_dataset,
        limit=args.limit,
        push=args.push,
        print_samples=args.print_samples,
    )
    print()
    print(f"Seen:       {stats.n_seen}")
    print(f"Augmented:  {stats.n_augmented} "
          f"({100 * stats.n_augmented / max(stats.n_seen, 1):.1f}%)")
    print(f"Mean subgoals per augmented trace: {stats.mean_subgoals:.2f}")
    if stats.failure_counts:
        print("Failure reasons:")
        for k, v in sorted(stats.failure_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {k}: {v}")
    print(f"Wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
