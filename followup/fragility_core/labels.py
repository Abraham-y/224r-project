"""Rollout labelling, structural-template features, and the global train/val split.

Labels come from the ORIGINAL verifier (`evaluation.countdown`), not a
reimplementation, so probe labels here are bit-identical to the RL reward's
notion of correctness.

Two label rules are always computed and stored side by side:

  first_block  correctness of the FIRST <answer> block. The paper's headline
               probe-AUROC rule (a trace-final hidden state predicts what the
               model is about to commit to).
  last_block   correctness of the LAST <answer> block == `compute_score == 1.0`.
               This is what vanilla RLOO rewards, and it is the rule the
               probe-as-reward pickle was trained on (see
               extension/cache/steering/probe_pipeline_temp1_meta.json).

Which one is the headline for a phase is a CONFIG choice (`label_rule`), fixed
once per phase — never per checkpoint.

The train/val split is a deterministic function of prompt_idx only. That matters
because the on-policy arm resamples rollouts at every checkpoint, so row
identity is not stable across t; prompt identity is. Splitting by prompt also
prevents within-prompt leakage between train and val.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import paths

paths.ensure_paper_on_syspath()

from evaluation.countdown import evaluate_equation, validate_equation  # noqa: E402

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

LABEL_RULES = ("first_block", "last_block")

# Structural / rhetorical template features. These are the surface patterns the
# original paper identified as the probe's confound (findings.md EXP-21: every
# probe=1.000 post-Goodhart rollout shared this scaffold). Kept identical to
# scripts/quantify_structural_confound.py so numbers are comparable.
TEMPLATE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\blet me\b", "let_me"),
    (r"\bstep by step\b", "step_by_step"),
    (r"\btherefore\b", "therefore"),
    (r"\bso the answer is\b|\bthe answer is\b", "the_answer_is"),
    (r"\bverified\b|\bvalid\b|\bthis works\b|\bgot it\b|\bperfect\b", "verification"),
    (r"^\s*\d+\.\s", "numbered_step"),
    (r"\bfirst\b.*\bthen\b|\bnext\b|\bafter that\b", "sequential_markers"),
)
_COMPILED_TEMPLATE = tuple(
    (re.compile(pat, re.MULTILINE), name) for pat, name in TEMPLATE_PATTERNS
)

# PLACEBO features: same count, same kind of thing (surface properties of the
# same text, so equally correlated with activation structure), but carrying no
# rhetorical-confidence semantics. These exist because residualising activations
# on ANY 7 text-derived, activation-aligned features destroys probe AUROC —
# removing 7 random PROJECTIONS OF THE ACTIVATIONS drops it 0.903 -> 0.506. So a
# drop under template residualisation means nothing on its own; it only means
# something relative to this placebo. See confounds.residualisation_test.
PLACEBO_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\*", "has_mult"),
    (r"/", "has_div"),
    (r"\(", "has_paren"),
    (r"=", "has_equals"),
    (r"\d{2,}", "has_multidigit"),
    (r"-", "has_minus"),
    (r"\+", "has_plus"),
)
_COMPILED_PLACEBO = tuple(
    (re.compile(pat, re.MULTILINE), name) for pat, name in PLACEBO_PATTERNS
)


# ---------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------


def _block_correct(equation: str, target: int, nums: Iterable[int]) -> bool:
    """Verifier semantics for one <answer> block (same code path as compute_score)."""
    eq = equation.strip()
    if not validate_equation(eq, list(nums)):
        return False
    result = evaluate_equation(eq)
    return result is not None and abs(result - target) < 1e-5


def block_labels(response: str, target: int, nums: Iterable[int]) -> dict[str, Any]:
    """Per-rollout correctness under both rules, plus structural descriptors."""
    blocks = [m.group(1) for m in _ANSWER_RE.finditer(response)]
    nums = list(nums)
    first = bool(blocks) and _block_correct(blocks[0], target, nums)
    last = bool(blocks) and _block_correct(blocks[-1], target, nums)
    return {
        "first_block": int(first),
        "last_block": int(last),
        "n_blocks": len(blocks),
        "resp_chars": len(response),
        "template_score": template_score(response),
    }


def template_score(text: str) -> int:
    """Count of distinct template patterns present (0-7). Each counted once."""
    lower = text.lower()
    return sum(1 for pat, _ in _COMPILED_TEMPLATE if pat.search(lower))


def template_features(text: str) -> dict[str, int]:
    """Per-pattern indicator dict (for finer-grained confound analysis)."""
    lower = text.lower()
    return {name: int(bool(pat.search(lower))) for pat, name in _COMPILED_TEMPLATE}


def placebo_features(text: str) -> dict[str, int]:
    """Matched-count surface features with no confidence/structure semantics."""
    return {name: int(bool(pat.search(text))) for pat, name in _COMPILED_PLACEBO}


# ---------------------------------------------------------------------------
# eval JSON -> label table
# ---------------------------------------------------------------------------


def load_eval_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a sampler output file (JSONL: one prompt per line, N responses each)."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_label_table(
    eval_json: str | Path,
    *,
    max_responses_per_prompt: int | None = None,
) -> pd.DataFrame:
    """(prompt_idx, resp_idx) -> both label rules + structural descriptors.

    Recomputes correctness from the response text rather than trusting the
    `scores` field, because `scores` is last-block-only and we need both rules.
    A `scores_field` column is kept so any disagreement is visible rather than
    silently resolved.
    """
    rows = load_eval_rows(eval_json)
    out = []
    for p_idx, row in enumerate(rows):
        target = int(row["target"])
        nums = list(row["nums"])
        responses = row["response"]
        scores = row.get("scores", [None] * len(responses))
        if max_responses_per_prompt is not None:
            responses = responses[:max_responses_per_prompt]
            scores = scores[:max_responses_per_prompt]
        for r_idx, (resp, sc) in enumerate(zip(responses, scores)):
            rec = {"prompt_idx": p_idx, "resp_idx": r_idx}
            rec.update(block_labels(resp, target, nums))
            # Per-pattern indicators, not just the summed score: the H0 controls
            # that residualise or stratify on the confound need the components,
            # and the placebo set is what makes a residualisation result readable.
            rec.update(template_features(resp))
            rec.update(placebo_features(resp))
            rec["scores_field"] = float(sc) if sc is not None else float("nan")
            out.append(rec)
    df = pd.DataFrame(out)
    if df.empty:
        raise ValueError(f"{eval_json}: no rollouts found")
    return df


def orig_prompt_idx(
    row: dict[str, Any], local_idx: int, subset_map: dict[int, int] | None = None
) -> int:
    """The prompt's index in the ORIGINAL eval file, not this subset's ordering.

    `extension/evaluation/sample_local_jsonl.py` rebuilds every output row from a
    fixed field list -- prompt, target, nums, ground_truth, response, scores --
    so a TOP-LEVEL `_orig_prompt_idx` written into the prompt file does not
    survive sampling. `ground_truth` is the one dict copied through verbatim, so
    that is where the index has to travel and where this looks first.

    Raises rather than falling back to `local_idx`: a silent fallback makes the
    caller's caches use a 0..N-1 subset namespace while every cache from the
    original project uses 0..499, which breaks the by-prompt split's
    comparability across ladders and mis-filters the clean-406 list.

    Same rule as `harvest_ladder._orig_index`, which now delegates here so the
    two cannot drift.
    """
    gt = row.get("ground_truth")
    if isinstance(gt, dict) and "_orig_prompt_idx" in gt:
        return int(gt["_orig_prompt_idx"])
    if "_orig_prompt_idx" in row:
        return int(row["_orig_prompt_idx"])
    if subset_map is not None and local_idx in subset_map:
        # Rollouts sampled before the index fix carry no original index, but the
        # subset is a deterministic order-preserving filter, so position i in the
        # rollout file is position i in the subset. Recovered rather than
        # re-sampled. NOT a silent fallback to the local index: the map is built
        # from the same subset file the sampler consumed.
        return int(subset_map[local_idx])
    raise KeyError(
        f"rollout row {local_idx} carries no _orig_prompt_idx and no subset map "
        "was supplied. Re-run build_prompt_subset, or pass the subset file."
    )


# ---------------------------------------------------------------------------
# contamination filter (clean-406)
# ---------------------------------------------------------------------------


def clean_prompt_idx(
    path: str | Path | None = None,
) -> set[int] | None:
    """The 406 uncontaminated prompt indices of countdown_eval_500, or None.

    Returns None (rather than raising) when the manifest is absent, so callers
    can treat "no filter available" as an explicit, logged condition.
    """
    p = Path(path) if path else (paths.PAPER_DATA / "contaminated_prompt_idx.json")
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return set(int(i) for i in d["clean"])


# ---------------------------------------------------------------------------
# global split
# ---------------------------------------------------------------------------


def prompt_split(
    prompt_idx: Iterable[int],
    *,
    val_frac: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    """Deterministic "train"/"val" assignment keyed on prompt_idx alone.

    Stable across checkpoints, machines, and python versions (hashlib, not
    builtin hash). Same prompt always lands in the same fold, so a probe never
    trains and evaluates on rollouts of the same problem.
    """
    idx = np.asarray(list(prompt_idx), dtype=np.int64)
    out = np.empty(len(idx), dtype=object)
    for i, p in enumerate(idx):
        h = hashlib.sha256(f"{seed}:{int(p)}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / float(1 << 64)
        out[i] = "val" if u < val_frac else "train"
    return out


def attach_split(df: pd.DataFrame, *, val_frac: float = 0.3, seed: int = 0) -> pd.DataFrame:
    df = df.copy()
    df["split"] = prompt_split(df["prompt_idx"], val_frac=val_frac, seed=seed)
    return df
