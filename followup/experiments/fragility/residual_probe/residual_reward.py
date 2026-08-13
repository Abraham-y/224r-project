"""Forked reward builder that can pass rollout text to the probe.

`followup/CLAUDE.md`: "Do not modify the original paper's experiment scripts in
place; wrap or fork them." So this is a fork of `_make_probe_reward` from
`extension/training/probe_reward_rloo.py`, not an edit of it. Every helper it
needs (`_think_close_hidden`, `_reconstruct_prompt`, `_ANSWER_RE`) is imported
from the original, so the hidden-state extraction path — the part with the
history of position bugs — is shared code, not a copy that can drift.

The ONLY behavioural difference from the original:

    if getattr(probe, "needs_text", False):
        p = probe.predict_proba(h[None, :], text=solution_str)[0, 1]
    else:
        p = probe.predict_proba(h[None, :])[0, 1]          # original call, verbatim

A probe without `needs_text` — i.e. every probe that exists today — is called
with the identical signature, so this fork is a no-op for the published arms.
`test_gate.py` asserts that rather than assuming it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from extension.training.probe_reward_rloo import (  # noqa: E402
    _ANSWER_RE,
    _reconstruct_prompt,
    _think_close_hidden,
)


def make_probe_reward(tok, model, probe, layer, mode, blend, orig_compute_score):
    """Same contract as the original `_make_probe_reward`.

    `mode` semantics are unchanged: probe | probe_gated | blend | mult.
    """
    text_aware = bool(getattr(probe, "needs_text", False))

    def probe_compute_score(solution_str, ground_truth, method="strict",
                            format_score=0.1, score=1.0):
        prompt = ground_truth.get("_prompt") if isinstance(ground_truth, dict) else None
        if prompt is None:
            prompt = _reconstruct_prompt(ground_truth) or ""
        h = _think_close_hidden(tok, model, prompt, solution_str, layer)
        if h is None:
            return 0.0  # no </think> -> no readable internal state -> no reward
        if text_aware:
            p = float(probe.predict_proba(h[None, :], text=solution_str)[0, 1])
        else:
            p = float(probe.predict_proba(h[None, :])[0, 1])
        if mode == "probe":
            return p
        if mode == "probe_gated":
            return p if _ANSWER_RE.search(solution_str) else 0.0
        if mode == "blend":
            v = float(orig_compute_score(solution_str, ground_truth))
            return (1.0 - blend) * p + blend * v
        if mode == "mult":
            v = float(orig_compute_score(solution_str, ground_truth))
            return v * p
        return p

    probe_compute_score.__name__ = "compute_score"
    probe_compute_score.__wrapped__ = orig_compute_score
    return probe_compute_score
