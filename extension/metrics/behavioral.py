"""Layer A behavioral metrics over an eval JSON.

Reads the eval JSONs produced by `evaluation/countdown_eval.py` and computes:

  * accuracy (score == 1.0)
  * format-only rate (score == 0.1)
  * no-answer rate (score == 0.0)
  * reasoning-answer consistency rate
  * confident-wrong rate
  * multi-answer (repetition) rate
  * truncation rate (CoT does not end cleanly with </answer>)
  * response length statistics

Person B, Day 1. Pure-Python; no GPU. Runs on a laptop in seconds.

Usage:
    python extension/metrics/behavioral.py \\
        --sft_json eval_sft.json --rloo_json eval_rloo.json \\
        --out_dir extension/outputs/phase1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

# Reuse the existing default-project verifier utilities.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from evaluation.countdown import (
    extract_solution,
    evaluate_equation,
    validate_equation,
)


# ---------------------------------------------------------------------------
# Heuristics. Tune in one place so partners can see what counts as "confident".
# ---------------------------------------------------------------------------

# Phrases that count as a high-confidence assertion in the chain of thought.
# Matched case-insensitively. Add to this list as you find more in the data.
CONFIDENT_ASSERTIONS: tuple[str, ...] = (
    "this works",
    "it works",
    "this is correct",
    "verified",
    "confirmed",
    "the answer is",
    "got it",
    "found it",
    "i've found",
    "found the solution",
    "solution found",
    "final answer",
    "yes!",
    "perfect",
    "exactly",
    "this gives",
    "we have it",
    "i found it",
)

ANSWER_OPEN_TAG = "<answer>"
ANSWER_CLOSE_TAG = "</answer>"
THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"

# Approximate "arithmetic expression" used for reasoning-answer consistency:
# a maximal run of digits, operators, parentheses, and whitespace.
_ARITH_RE = re.compile(r"[-+*/().\s\d]{3,}")


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_eval_json(path: str) -> list[dict]:
    """Load an eval JSON. Handles both JSON-lines (HF default) and a JSON array."""
    with open(path) as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Per-rollout features.
# ---------------------------------------------------------------------------

def count_answer_tags(response: str) -> int:
    return response.count(ANSWER_OPEN_TAG)


def ends_cleanly(response: str) -> bool:
    """True if the response stops cleanly (ends shortly after </answer>).

    The vLLM generator emits text up to max_tokens; if the model didn't stop on
    its own, the response usually ends mid-token or with degenerate repetition
    well past the first </answer>. We call it "clean" if the closing tag
    appears within the last ~32 characters.
    """
    if ANSWER_CLOSE_TAG not in response:
        return False
    return response.rstrip().endswith(ANSWER_CLOSE_TAG) or (
        response.rstrip()[-64:].count(ANSWER_CLOSE_TAG) >= 1
        and len(response) - response.rfind(ANSWER_CLOSE_TAG) < 32
    )


def contains_confident_assertion(text: str) -> bool:
    low = text.lower()
    return any(phrase in low for phrase in CONFIDENT_ASSERTIONS)


def reasoning_section(response: str) -> str:
    """Return the <think>...</think> body if present, else everything before the first <answer>."""
    if THINK_OPEN_TAG in response and THINK_CLOSE_TAG in response:
        start = response.index(THINK_OPEN_TAG) + len(THINK_OPEN_TAG)
        end = response.index(THINK_CLOSE_TAG)
        return response[start:end]
    if ANSWER_OPEN_TAG in response:
        return response[: response.index(ANSWER_OPEN_TAG)]
    return response


def last_arithmetic_value(text: str) -> float | None:
    """Evaluate the last arithmetic-looking expression in `text`, returning its value.

    Used for reasoning-answer consistency: do the equations the model "endorses"
    in the CoT actually evaluate to the same value as the equation in <answer>?
    """
    matches = list(_ARITH_RE.finditer(text))
    for match in reversed(matches):
        candidate = match.group(0).strip()
        # Skip purely numeric or trivially short matches.
        if not any(op in candidate for op in "+-*/") or len(candidate) < 3:
            continue
        value = evaluate_equation(candidate)
        if value is not None:
            return float(value)
    return None


def reasoning_answer_consistent(response: str, available_numbers: Sequence[int]) -> bool | None:
    """Check that the last arithmetic in <think> evaluates to the same value as <answer>.

    Returns:
        True  -> CoT-endorsed value matches the <answer> equation's value
        False -> they differ
        None  -> one side could not be parsed (don't count this rollout)
    """
    answer = extract_solution(response)
    if not answer:
        return None
    if not validate_equation(answer, list(available_numbers)):
        return None
    ans_value = evaluate_equation(answer)
    if ans_value is None:
        return None
    cot_value = last_arithmetic_value(reasoning_section(response))
    if cot_value is None:
        return None
    return abs(cot_value - ans_value) < 1e-5


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------

@dataclass
class BehavioralMetrics:
    name: str
    n_prompts: int
    n_rollouts: int
    accuracy: float            # mean(score == 1.0)
    format_only_rate: float    # mean(score == 0.1)
    no_answer_rate: float      # mean(score == 0.0)
    consistency_rate: float    # over rollouts where both CoT and answer parsed
    consistency_n: int         # denominator for consistency_rate
    confident_wrong_rate: float  # over wrong rollouts only
    confident_wrong_n: int     # number of wrong rollouts
    multi_answer_rate: float   # fraction of rollouts with >1 <answer> tag
    mean_answer_tag_count: float
    truncation_rate: float     # fraction that did not end cleanly
    mean_chars: float
    median_chars: float


def compute(name: str, rows: Iterable[dict]) -> BehavioralMetrics:
    rows = list(rows)
    n_prompts = len(rows)
    n_rollouts = 0
    n_correct = n_format = n_no_answer = 0
    n_multi_answer = 0
    n_truncated = 0
    consistent = consistency_total = 0
    confident_wrong = wrong_total = 0
    tag_counts: list[int] = []
    char_lens: list[int] = []

    for row in rows:
        nums = row.get("nums") or row.get("ground_truth", {}).get("numbers", [])
        for response, score in zip(row["response"], row["scores"]):
            n_rollouts += 1
            char_lens.append(len(response))
            tag_n = count_answer_tags(response)
            tag_counts.append(tag_n)
            if tag_n > 1:
                n_multi_answer += 1
            if not ends_cleanly(response):
                n_truncated += 1

            if score == 1.0:
                n_correct += 1
            elif score == 0.1:
                n_format += 1
            else:
                n_no_answer += 1

            consistent_flag = reasoning_answer_consistent(response, nums)
            if consistent_flag is not None:
                consistency_total += 1
                if consistent_flag:
                    consistent += 1

            # Confident-wrong only meaningful on wrong rollouts.
            if score < 1.0:
                wrong_total += 1
                if contains_confident_assertion(reasoning_section(response)):
                    confident_wrong += 1

    def safe(num: int, den: int) -> float:
        return num / den if den else float("nan")

    return BehavioralMetrics(
        name=name,
        n_prompts=n_prompts,
        n_rollouts=n_rollouts,
        accuracy=safe(n_correct, n_rollouts),
        format_only_rate=safe(n_format, n_rollouts),
        no_answer_rate=safe(n_no_answer, n_rollouts),
        consistency_rate=safe(consistent, consistency_total),
        consistency_n=consistency_total,
        confident_wrong_rate=safe(confident_wrong, wrong_total),
        confident_wrong_n=wrong_total,
        multi_answer_rate=safe(n_multi_answer, n_rollouts),
        mean_answer_tag_count=statistics.mean(tag_counts) if tag_counts else float("nan"),
        truncation_rate=safe(n_truncated, n_rollouts),
        mean_chars=statistics.mean(char_lens) if char_lens else float("nan"),
        median_chars=statistics.median(char_lens) if char_lens else float("nan"),
    )


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

_DISPLAY_FIELDS: tuple[tuple[str, str, str], ...] = (
    # (attribute, label, format)
    ("n_prompts", "prompts", "{:d}"),
    ("n_rollouts", "rollouts", "{:d}"),
    ("accuracy", "accuracy (score=1.0)", "{:.3f}"),
    ("format_only_rate", "format-only (score=0.1)", "{:.3f}"),
    ("no_answer_rate", "no-answer (score=0.0)", "{:.3f}"),
    ("consistency_rate", "CoT-answer consistency", "{:.3f}"),
    ("confident_wrong_rate", "confident-wrong rate", "{:.3f}"),
    ("multi_answer_rate", "multi-<answer> rate", "{:.3f}"),
    ("mean_answer_tag_count", "mean <answer> count", "{:.2f}"),
    ("truncation_rate", "truncation rate", "{:.3f}"),
    ("mean_chars", "mean chars", "{:.0f}"),
    ("median_chars", "median chars", "{:.0f}"),
)


def print_table(metrics: Sequence[BehavioralMetrics]) -> None:
    name_col = 28
    val_col = max(14, max(len(m.name) for m in metrics) + 2)
    header = "metric".ljust(name_col) + "".join(m.name.rjust(val_col) for m in metrics)
    print(header)
    print("-" * len(header))
    for attr, label, fmt in _DISPLAY_FIELDS:
        row = label.ljust(name_col)
        for m in metrics:
            value = getattr(m, attr)
            row += fmt.format(value).rjust(val_col)
        print(row)


def write_csv(metrics: Sequence[BehavioralMetrics], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    field_names = list(asdict(metrics[0]).keys())
    with open(path, "w") as f:
        f.write(",".join(field_names) + "\n")
        for m in metrics:
            d = asdict(m)
            f.write(",".join(str(d[k]) for k in field_names) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_json", required=True)
    parser.add_argument("--rloo_json", required=True)
    parser.add_argument("--out_dir", default="extension/outputs/phase1")
    args = parser.parse_args()

    sft_rows = load_eval_json(args.sft_json)
    rloo_rows = load_eval_json(args.rloo_json)

    sft = compute("SFT (C_SFT)", sft_rows)
    rloo = compute("RLOO (C_outcome)", rloo_rows)

    print_table([sft, rloo])
    csv_path = os.path.join(args.out_dir, "phase1_metrics.csv")
    write_csv([sft, rloo], csv_path)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
