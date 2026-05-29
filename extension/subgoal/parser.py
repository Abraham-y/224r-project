"""Parse <subgoal>...</subgoal> declarations out of a reasoning trace.

Grammar (per extension.md §4.3):

    <subgoal> reach 60 from [3, 4, 5] </subgoal>

Any well-formed declaration produces a :class:`Subgoal` with the target value,
the list of allowed numbers, and the (start, end) character span. Malformed
declarations (unmatched tags, no "reach", no parseable list) are silently
skipped — the assumption is that subgoals come from the model, not from us, so
we expect a long tail of garbage and we shouldn't crash on it.

Person A, Day 1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


SUBGOAL_OPEN = "<subgoal>"
SUBGOAL_CLOSE = "</subgoal>"

# Body grammar: "reach <number> from [<n>, <n>, ...]"
# Tolerant about whitespace and parentheses.
_BODY_RE = re.compile(
    r"reach\s*(-?\d+)\s*from\s*[\[\(]([\d,\s\-]+)[\]\)]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Subgoal:
    target: int
    available: tuple[int, ...]   # immutable so Subgoals are hashable
    start: int                   # char offset of opening tag
    end: int                     # char offset just past closing tag


def _parse_body(body: str) -> tuple[int, tuple[int, ...]] | None:
    match = _BODY_RE.search(body)
    if not match:
        return None
    target = int(match.group(1))
    try:
        nums = tuple(int(x.strip()) for x in match.group(2).split(",") if x.strip())
    except ValueError:
        return None
    if not nums:
        return None
    return target, nums


def parse_subgoals(text: str) -> list[Subgoal]:
    """Return the well-formed subgoals in `text` in order of appearance."""
    out: list[Subgoal] = []
    cursor = 0
    while True:
        open_idx = text.find(SUBGOAL_OPEN, cursor)
        if open_idx == -1:
            return out
        close_idx = text.find(SUBGOAL_CLOSE, open_idx + len(SUBGOAL_OPEN))
        if close_idx == -1:
            return out
        body = text[open_idx + len(SUBGOAL_OPEN) : close_idx]
        parsed = _parse_body(body)
        if parsed is not None:
            target, nums = parsed
            out.append(Subgoal(target=target, available=nums,
                               start=open_idx, end=close_idx + len(SUBGOAL_CLOSE)))
        cursor = close_idx + len(SUBGOAL_CLOSE)


# ---------------------------------------------------------------------------
# Smoke test. Run with: python extension/subgoal/parser.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases: list[tuple[str, int]] = [
        # (text, expected number of subgoals)
        ("<subgoal> reach 60 from [3, 4, 5] </subgoal>", 1),
        ("noise <subgoal>reach 60 from [3,4,5]</subgoal> more noise", 1),
        ("<subgoal> reach 60 from (3, 4, 5) </subgoal>", 1),
        # Two in a row:
        ("<subgoal> reach 60 from [3, 4, 5] </subgoal> 4*5=20 ... 20*3=60. "
         "<subgoal> reach 68 from [60, 8] </subgoal> 60+8=68.", 2),
        # Malformed: no "reach":
        ("<subgoal> compute 60 from [3, 4, 5] </subgoal>", 0),
        # Malformed: no closing tag:
        ("<subgoal> reach 60 from [3, 4, 5]", 0),
        # Empty list:
        ("<subgoal> reach 60 from [] </subgoal>", 0),
        # Negative target:
        ("<subgoal> reach -7 from [1, 8] </subgoal>", 1),
        # Nested-looking text (we don't support nesting; outer wins):
        ("<subgoal> reach 60 from [3, 4, 5] <subgoal>noise</subgoal> </subgoal>", 1),
        # Garbage:
        ("totally unrelated text", 0),
    ]
    failed = 0
    for text, expected in cases:
        got = len(parse_subgoals(text))
        marker = "ok " if got == expected else "FAIL"
        if got != expected:
            failed += 1
        print(f"{marker}  expected {expected}, got {got}  --  {text[:60]!r}")
    if failed:
        raise SystemExit(f"{failed} case(s) failed")
    print("\nAll parser smoke tests passed.")
