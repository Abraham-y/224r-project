"""ECE, reliability diagrams, verbalized-confidence AUROC.

Person B. Day 4-ish.

Verbalized confidence is elicited with a *separate* prompt after the rollout
(see extension.md §5.1), so this module:

    1. Loads per-rollout text (from the existing eval JSONs),
    2. For each rollout, calls the model with a "Rate your confidence 0-100"
       prompt and parses the number,
    3. Bins confidences vs binary correctness (score == 1.0),
    4. Computes ECE, plots a reliability diagram, reports AUROC.

The elicitation call needs the same model that produced the rollout, so this
step is the only one in metrics/ that *does* use GPU (briefly).

TODO:
    * Implement load_or_elicit_confidences(...) with a JSON cache so we never
      re-call the model on the same rollout.
    * Use a stable prompt template (commit it to the file as a constant so it's
      reproducible).
    * Implement ece(probs, correct, n_bins=10) and plot_reliability(...).
"""

from __future__ import annotations

CONFIDENCE_PROMPT = (
    "Below is a Countdown problem and a candidate answer. "
    "Rate your confidence from 0 to 100 that the candidate answer is correct. "
    "Reply with only an integer between 0 and 100.\n\n"
    "Problem: {problem}\n"
    "Candidate answer: {answer}\n"
    "Confidence (0-100):"
)


def ece(probs, correct, n_bins: int = 10) -> float:
    """Expected Calibration Error. TODO: implement."""
    raise NotImplementedError


def plot_reliability(probs, correct, out_path: str, n_bins: int = 10) -> None:
    """TODO: implement."""
    raise NotImplementedError
