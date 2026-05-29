"""Launch the RLOO trainer with the composite (outcome + subgoal) reward.

Architecture (per extension.md §10 -- don't modify rloo_trainer/rloo.py):

    1. Import evaluation.countdown.
    2. Wrap its compute_score with a composite that adds the subgoal reward
       term, and write the wrapped version back to the module attribute.
    3. Use runpy to execute rloo_trainer.rloo as a script. Its
       `from evaluation.countdown import compute_score` is executed *after*
       step 2, so it picks up the wrapped function.

The wrapper preserves the existing CLI: every flag accepted by rloo.py works
here too. Adds three extra flags scoped to the process reward:

    --subgoal_lambda 0.3       weight on R_subgoal in R = R_outcome + lam * R_subgoal
    --subgoal_alpha  1.0       penalty weight on invalid subgoals
    --subgoal_disable          run as plain RLOO (composite reward not applied;
                               useful for A/B sanity)

These are stripped from sys.argv before rloo's argparse runs.

Person A. Day 4-ish.
"""

from __future__ import annotations

import os
import runpy
import sys

# Make sibling imports work whether launched as `python -m extension.training.process_rloo`
# or `python extension/training/process_rloo.py`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Pop our extra flags out of sys.argv before rloo's argparse sees them.
# ---------------------------------------------------------------------------

def _pop_flag(name: str, default: str | None) -> str | None:
    """Extract --name VALUE (or --name=VALUE) from sys.argv. Returns the value."""
    val: str | None = default
    out: list[str] = []
    i = 0
    while i < len(sys.argv):
        token = sys.argv[i]
        if token == f"--{name}":
            if i + 1 < len(sys.argv):
                val = sys.argv[i + 1]
                i += 2
                continue
            raise SystemExit(f"--{name} requires a value")
        if token.startswith(f"--{name}="):
            val = token.split("=", 1)[1]
            i += 1
            continue
        out.append(token)
        i += 1
    sys.argv[:] = out
    return val


def _pop_switch(name: str) -> bool:
    """Extract a bare --name switch; return True if present."""
    present = False
    out: list[str] = []
    for token in sys.argv:
        if token == f"--{name}":
            present = True
            continue
        out.append(token)
    sys.argv[:] = out
    return present


def _install_composite_reward(lam: float, alpha: float) -> None:
    """Replace evaluation.countdown.compute_score with the composite version."""
    import evaluation.countdown as countdown

    from extension.subgoal.reward import compute_reward

    original_compute_score = countdown.compute_score

    def composite_compute_score(
        solution_str, ground_truth, method="strict",
        format_score=0.1, score=1.0,
    ):
        outcome = original_compute_score(
            solution_str, ground_truth,
            method=method, format_score=format_score, score=score,
        )
        problem_nums = ground_truth.get("numbers", [])
        return compute_reward(outcome, solution_str, problem_nums,
                              lam=lam, alpha=alpha)

    composite_compute_score.__name__ = "compute_score"
    composite_compute_score.__wrapped__ = original_compute_score  # for introspection
    countdown.compute_score = composite_compute_score


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def main() -> None:
    lam_str = _pop_flag("subgoal_lambda", "0.3")
    alpha_str = _pop_flag("subgoal_alpha", "1.0")
    disable = _pop_switch("subgoal_disable")

    if not disable:
        lam = float(lam_str if lam_str is not None else 0.3)
        alpha = float(alpha_str if alpha_str is not None else 1.0)
        _install_composite_reward(lam=lam, alpha=alpha)
        print(
            f"[process_rloo] composite reward active: "
            f"R = R_outcome + {lam} * R_subgoal  (alpha={alpha})",
            flush=True,
        )
    else:
        print("[process_rloo] --subgoal_disable set; using plain outcome reward.", flush=True)

    # Hand off to the existing RLOO trainer as if it had been invoked directly.
    runpy.run_module("rloo_trainer.rloo", run_name="__main__")


if __name__ == "__main__":
    main()
