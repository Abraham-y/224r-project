"""Exact subgoal verifier: is_reachable + is_achieved.

Both checks are noise-free over Countdown's small (3-4 element) subsets.

  * ``is_reachable(target, subset)``  -- is ``target`` reachable from ``subset``
    via +, -, *, / (using each number exactly once)?
    Enumerated exhaustively; cached on (target, sorted(subset)).

  * ``is_achieved(target, subset, segment)`` -- does the trace segment contain
    an arithmetic computation that evaluates to ``target`` while using only
    integers from ``subset`` (and each at most once)?

Person A, Day 1. Pure-Python, no learned model.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Sequence

OPS = ("+", "-", "*", "/")


# ---------------------------------------------------------------------------
# is_reachable -- exhaustive enumeration.
# ---------------------------------------------------------------------------

def _combine(a: float, b: float, op: str) -> float | None:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if abs(b) < 1e-12:
            return None
        return a / b
    raise ValueError(f"unknown op {op!r}")


def _reachable_values(values: tuple[float, ...]) -> set[float]:
    """All values reachable by binary-combining the multiset ``values``."""
    if len(values) == 1:
        return {values[0]}
    out: set[float] = set()
    n = len(values)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rest = tuple(v for k, v in enumerate(values) if k not in (i, j))
            for op in OPS:
                combined = _combine(values[i], values[j], op)
                if combined is None:
                    continue
                # Recurse on the new multiset.
                for v in _reachable_values(rest + (combined,)):
                    out.add(v)
    return out


@lru_cache(maxsize=4096)
def _reachable_cached(target: int, subset_key: tuple[int, ...]) -> bool:
    floats = tuple(float(x) for x in subset_key)
    for v in _reachable_values(floats):
        if abs(v - target) < 1e-5:
            return True
    return False


def is_reachable(target: int, subset: Sequence[int]) -> bool:
    """True iff ``target`` is reachable from ``subset`` via +,-,*,/.

    >>> is_reachable(60, [3, 4, 5])
    True
    >>> is_reachable(61, [3, 4, 5])
    False
    """
    key = tuple(sorted(int(x) for x in subset))
    return _reachable_cached(int(target), key)


# ---------------------------------------------------------------------------
# is_achieved -- does the trace segment actually compute target?
# ---------------------------------------------------------------------------

# Match equations of the form "<expr> = <number>".
_EQ_RE = re.compile(r"([\d+\-*/().\s]+?)=\s*(-?\d+(?:\.\d+)?)")


def _expr_uses_only(expr: str, subset: Sequence[int]) -> bool:
    """All integer literals in ``expr`` must come from ``subset`` (with multiplicity)."""
    nums_in_expr = [int(n) for n in re.findall(r"-?\d+", expr)]
    if not nums_in_expr:
        return False
    available = list(subset)
    for n in nums_in_expr:
        if n in available:
            available.remove(n)
        else:
            return False
    return True


def _safe_eval(expr: str) -> float | None:
    if not re.match(r"^[\d+\-*/().\s]+$", expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": None}, {}))
    except Exception:
        return None


def is_achieved(target: int, subset: Sequence[int], segment: str) -> bool:
    """True if ``segment`` chains equations to compute ``target`` using each subset number once.

    Walks the equations in ``segment`` in order, maintaining a pool of unused
    subset numbers and previously-computed intermediates. Each equation's
    literals must come from the pool; the equation's stated RHS must equal the
    actual arithmetic. We return True the first time the running result equals
    ``target`` with the subset exhausted. Dead-end equations (those whose
    literals can't be drawn from the current pool, or whose stated RHS is
    arithmetically wrong) are skipped without disturbing the pool, so a
    correct chain interleaved with abandoned attempts still counts.

    Examples
    --------
    >>> is_achieved(60, [3, 4, 5], "4 * 5 = 20, then 20 * 3 = 60.")
    True
    >>> is_achieved(60, [3, 4, 5], "3 + 4 = 7, 7 * 5 = 35.")
    False
    >>> is_achieved(60, [3, 4, 5], "10 * 6 = 60.")
    False
    """
    subset_pool = list(subset)
    prev_values: list[float] = []
    for match in _EQ_RE.finditer(segment):
        expr = match.group(1).strip()
        try:
            stated = float(match.group(2))
        except ValueError:
            continue
        actual = _safe_eval(expr)
        if actual is None or abs(actual - stated) >= 1e-5:
            continue  # arithmetic mistake -> skip, don't disturb pool

        literals = [int(n) for n in re.findall(r"-?\d+", expr)]
        trial_subset = list(subset_pool)
        trial_prev = list(prev_values)
        ok = True
        for n in literals:
            if n in trial_subset:
                trial_subset.remove(n)
                continue
            fn = float(n)
            for k, v in enumerate(trial_prev):
                if abs(v - fn) < 1e-5:
                    del trial_prev[k]
                    break
            else:
                ok = False
                break
        if not ok:
            continue  # equation introduces a number we don't have -> skip

        subset_pool = trial_subset
        prev_values = trial_prev
        prev_values.append(stated)
        if abs(stated - target) < 1e-5 and not subset_pool:
            return True
    return False


# ---------------------------------------------------------------------------
# Smoke test. Run with: python extension/subgoal/verifier.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    reach_cases = [
        # (target, subset, expected)
        (60, [3, 4, 5], True),       # 4*5*3
        (61, [3, 4, 5], False),
        (24, [3, 4, 6, 8], True),    # standard "24" puzzle
        (28, [95, 11, 56], True),    # (95-56)-11
        (-7, [1, 8], True),
        (100, [1, 2], False),
    ]
    achieved_cases = [
        # (target, subset, segment, expected)
        (60, [3, 4, 5], "4 * 5 = 20, then 20 * 3 = 60. reached 60.", True),
        (60, [3, 4, 5], "3 + 4 = 7, 7 * 5 = 35.", False),
        # Uses a number not in subset:
        (60, [3, 4, 5], "10 * 6 = 60.", False),
        # Right number, wrong arithmetic on RHS:
        (60, [3, 4, 5], "3 + 4 + 5 = 60", False),
        (28, [95, 11, 56], "95 - 56 = 39\n39 - 11 = 28 (works)", True),
    ]
    failed = 0
    for target, subset, expected in reach_cases:
        got = is_reachable(target, subset)
        marker = "ok " if got == expected else "FAIL"
        if got != expected:
            failed += 1
        print(f"{marker}  is_reachable({target}, {subset}) -> {got}  (expected {expected})")
    print()
    for target, subset, segment, expected in achieved_cases:
        got = is_achieved(target, subset, segment)
        marker = "ok " if got == expected else "FAIL"
        if got != expected:
            failed += 1
        print(f"{marker}  is_achieved({target}, {subset}, ...) -> {got}  (expected {expected})")
    if failed:
        raise SystemExit(f"{failed} case(s) failed")
    print("\nAll verifier smoke tests passed.")
