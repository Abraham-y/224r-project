"""RLOO with first-answer reward + per-extra-block penalty.

Reward = verifier(first <answer>) − λ * max(0, n_blocks − 1)

This directly punishes the model for emitting extra <answer> blocks beyond
the first. Tests the hypothesis: is rambling at 0.5B C_outcome a load-bearing
mechanism for the accuracy RL gives the model, or is it SFT inertia that
gradient pressure can suppress without losing accuracy?

  Outcome (a): rambling drops, accuracy preserved → rambling was inertia
  Outcome (b): rambling drops, accuracy drops → rambling is functional
  Outcome (c): rambling doesn't drop → something deeper than inertia

Default λ = 0.05 (so 5 extra blocks = −0.25 penalty, comparable to giving the
rollout the no-answer score of 0).

Same architecture as extension/training/firstanswer_rloo.py: monkey-patch
evaluation.countdown.compute_score, then exec rloo_trainer/rloo.py unchanged.

CLI:
  --ramble_penalty FLOAT   penalty per extra block beyond first (default 0.05)
"""

from __future__ import annotations
import os, re, sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RLOO_DIR = os.path.join(_REPO_ROOT, "rloo_trainer")
for path in (_RLOO_DIR, _REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _pop_arg(name: str, default: str | None = None) -> str | None:
    out = []; found = default; skip = False
    for i, tok in enumerate(sys.argv):
        if skip: skip = False; continue
        if tok == f"--{name}" and i + 1 < len(sys.argv):
            found = sys.argv[i + 1]; skip = True; continue
        out.append(tok)
    sys.argv[:] = out
    return found


def _install_ramble_penalty_reward(lam: float) -> None:
    """Replace compute_score with first-answer + ramble-penalty reward."""
    import evaluation.countdown as countdown

    validate_equation = countdown.validate_equation
    evaluate_equation = countdown.evaluate_equation

    def ramble_penalty_compute_score(solution_str, ground_truth, method="strict",
                                      format_score=0.1, score=1.0):
        """Reward = first-block-score − λ * max(0, n_blocks − 1)."""
        target = ground_truth["target"]
        numbers = ground_truth["numbers"]

        matches = list(_ANSWER_RE.finditer(solution_str))
        n_blocks = len(matches)
        penalty = lam * max(0, n_blocks - 1)

        if not matches:
            return 0 - penalty  # no answer, full ramble-penalty
        first_answer = matches[0].group(1).strip()

        if not validate_equation(first_answer, numbers):
            return format_score - penalty
        try:
            result = evaluate_equation(first_answer)
            if result is None:
                return format_score - penalty
            if abs(result - target) < 1e-5:
                return score - penalty
            return format_score - penalty
        except Exception:
            return format_score - penalty

    ramble_penalty_compute_score.__name__ = "compute_score"
    countdown.compute_score = ramble_penalty_compute_score
    print(
        f"[ramble_penalty_rloo] ACTIVE: reward = first_block_score − {lam} * max(0, n_blocks − 1)",
        flush=True,
    )


def main() -> None:
    lam = float(_pop_arg("ramble_penalty", "0.05"))
    _install_ramble_penalty_reward(lam)

    rloo_path = os.path.join(_RLOO_DIR, "rloo.py")
    sys.argv[0] = rloo_path
    with open(rloo_path) as f:
        code = compile(f.read(), rloo_path, "exec")
    exec(code, {"__name__": "__main__", "__file__": rloo_path})


if __name__ == "__main__":
    main()
