"""Compute the concealment gap = probe AUROC - verbalized AUROC.

Inputs:
  - Per-checkpoint confidence JSONL produced by elicit_verbalized_confidence.py.
    Each row has prompt_idx, resp_idx, score, verbalized_confidence.
  - Probe activations from --cache_dir for the same eval rollouts.

For each checkpoint:
  1. Verbalized AUROC = AUROC of verbalized_confidence vs (score == 1.0),
     dropping rows whose confidence didn't parse.
  2. Probe AUROC at </think> = balanced 5-fold CV AUROC on pre_answer cache.
  3. Gap = probe - verbalized.

Also prints the per-rollout matched pairing where available:
  - If a rollout has both a verbalized_confidence and a probe-derived score
    (from CV), show the joint distribution (correlation, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from extension.probe.analyze_probes import load_groups
from extension.probe.robustness_probes import balanced_subsample_auroc
from extension.probe.qualitative_examples import cv_per_row_probe_scores


def load_confidence_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def verbalized_auroc(rows: list[dict]) -> tuple[float, int, int]:
    """Return (AUROC, n_used, n_correct) where verbalized_confidence parsed."""
    xs, ys = [], []
    for r in rows:
        c = r.get("verbalized_confidence")
        if c is None:
            continue
        xs.append(float(c))
        ys.append(1 if float(r["score"]) == 1.0 else 0)
    xs = np.array(xs); ys = np.array(ys)
    if len(np.unique(ys)) < 2:
        return float("nan"), len(xs), int(ys.sum())
    if len(np.unique(xs)) < 2:
        # Degenerate: all confidences identical (broken elicitation).
        return float("nan"), len(xs), int(ys.sum())
    return float(roc_auc_score(ys, xs)), len(xs), int(ys.sum())


# Binary-keyword fallback (used when the elicitation prompt fails).
CONFIDENT_PATTERNS = (
    "perfect", "this works", "got it", "the answer is", "verified",
    "this is correct", "confirmed", "found it",
)


def keyword_verbalized_auroc(eval_json_path: str) -> tuple[float, int, int]:
    """Fallback verbalized confidence: 1 if the CoT contains any confidence
    keyword, 0 otherwise. AUROC of this binary signal vs (score == 1.0).
    Uses ALL responses from the eval JSON (not just one per prompt).
    """
    import re
    rows = [
        json.loads(l) for l in open(eval_json_path).read().strip().splitlines() if l.strip()
    ]
    xs, ys = [], []
    for r in rows:
        for resp, score in zip(r["response"], r["scores"]):
            # Restrict keyword search to the <think> body of the response
            # (not the verification echoes that appear post-answer).
            m = re.search(r"<think>(.*?)</think>", resp, re.DOTALL)
            think_body = m.group(1) if m else resp
            low = think_body.lower()
            has_kw = int(any(p in low for p in CONFIDENT_PATTERNS))
            xs.append(has_kw)
            ys.append(1 if float(score) == 1.0 else 0)
    xs = np.array(xs); ys = np.array(ys)
    if len(np.unique(ys)) < 2 or len(np.unique(xs)) < 2:
        return float("nan"), len(xs), int(ys.sum())
    return float(roc_auc_score(ys, xs)), len(xs), int(ys.sum())


def probe_pre_answer_auroc(cache_dir: str, ckpt: str, layer: int) -> float:
    npz = os.path.join(cache_dir, f"{ckpt}_l{layer}_pre_answer.npz")
    meta = npz.replace(".npz", ".meta.json")
    if not (os.path.exists(npz) and os.path.exists(meta)):
        return float("nan")
    with np.load(npz) as data:
        X = data["X"]; y = data["y"]
    groups = load_groups(meta)
    return balanced_subsample_auroc(X, y, groups)


def joint_probe_vs_verbalized(rows: list[dict], cache_dir: str, ckpt: str,
                              layer: int) -> dict | None:
    """Per-rollout join of verbalized confidence and trace-final probe score.

    For rollouts where both exist, compute Pearson correlation between probe
    score and verbalized confidence (does verbalized confidence track the
    probe's belief, or are they decoupled?).
    """
    pre_npz = os.path.join(cache_dir, f"{ckpt}_l{layer}_pre_answer.npz")
    pre_meta = pre_npz.replace(".npz", ".meta.json")
    if not (os.path.exists(pre_npz) and os.path.exists(pre_meta)):
        return None
    with np.load(pre_npz) as data:
        X = data["X"]; y = data["y"]
    with open(pre_meta) as f:
        meta = json.load(f)
    groups = np.array([m["prompt_idx"] for m in meta], dtype=np.int64)

    # Per-rollout cross-validated probe score at </think>.
    scores = cv_per_row_probe_scores(X, y, groups)
    by_rollout = {}
    for i, m in enumerate(meta):
        if np.isnan(scores[i]):
            continue
        by_rollout[(int(m["prompt_idx"]), int(m["resp_idx"]))] = float(scores[i])

    probe_scores, verbal_confs, labels = [], [], []
    for r in rows:
        key = (int(r["prompt_idx"]), int(r["resp_idx"]))
        if key not in by_rollout:
            continue
        c = r.get("verbalized_confidence")
        if c is None:
            continue
        probe_scores.append(by_rollout[key])
        verbal_confs.append(float(c) / 100.0)
        labels.append(1 if float(r["score"]) == 1.0 else 0)

    probe_scores = np.array(probe_scores)
    verbal_confs = np.array(verbal_confs)
    labels = np.array(labels)
    if len(probe_scores) < 2:
        return None
    rho_overall, _ = stats.pearsonr(probe_scores, verbal_confs)
    # Conditional correlations
    rhos = {}
    for label_value in (0, 1):
        mask = labels == label_value
        if mask.sum() >= 3 and np.std(probe_scores[mask]) > 1e-9 and np.std(verbal_confs[mask]) > 1e-9:
            rho, _ = stats.pearsonr(probe_scores[mask], verbal_confs[mask])
            rhos[label_value] = float(rho)
        else:
            rhos[label_value] = float("nan")
    return {
        "n": int(len(probe_scores)),
        "rho_overall": float(rho_overall),
        "rho_on_correct": rhos[1],
        "rho_on_wrong": rhos[0],
        "mean_probe_correct": float(probe_scores[labels == 1].mean()) if (labels == 1).any() else float("nan"),
        "mean_probe_wrong": float(probe_scores[labels == 0].mean()) if (labels == 0).any() else float("nan"),
        "mean_verbal_correct": float(verbal_confs[labels == 1].mean()) if (labels == 1).any() else float("nan"),
        "mean_verbal_wrong": float(verbal_confs[labels == 0].mean()) if (labels == 0).any() else float("nan"),
    }


def logprob_verbalized_auroc(path: str) -> tuple[float, int, int]:
    """AUROC of `logit_gap` (logprob(yes) - logprob(no)) vs correctness.

    Rows missing either yes or no in the top-K are dropped.
    """
    if not os.path.exists(path):
        return float("nan"), 0, 0
    xs, ys = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            g = r.get("logit_gap")
            if g is None:
                continue
            xs.append(float(g))
            ys.append(1 if float(r["score"]) == 1.0 else 0)
    xs = np.array(xs); ys = np.array(ys)
    if len(np.unique(ys)) < 2 or len(np.unique(xs)) < 2:
        return float("nan"), len(xs), int(ys.sum())
    return float(roc_auc_score(ys, xs)), len(xs), int(ys.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="extension/cache/probe_cache")
    parser.add_argument("--sft_logprob",
                        default="extension/cache/confidence/C_SFT_logprob_confidence.jsonl",
                        help="Token-logprob JSONL (preferred verbalized signal).")
    parser.add_argument("--outcome_logprob",
                        default="extension/cache/confidence/C_outcome_logprob_confidence.jsonl")
    parser.add_argument("--sft_confidence",
                        default="extension/cache/confidence/C_SFT_confidence.jsonl",
                        help="Generated [0,100] elicitation JSONL (likely degenerate).")
    parser.add_argument("--outcome_confidence",
                        default="extension/cache/confidence/C_outcome_confidence.jsonl")
    parser.add_argument("--sft_eval", default="eval_sft.json")
    parser.add_argument("--outcome_eval", default="eval.json")
    parser.add_argument("--layer", type=int, default=16)
    args = parser.parse_args()

    logprob_paths = {"C_SFT": args.sft_logprob, "C_outcome": args.outcome_logprob}
    confidence_paths = {"C_SFT": args.sft_confidence, "C_outcome": args.outcome_confidence}
    eval_paths = {"C_SFT": args.sft_eval, "C_outcome": args.outcome_eval}

    summary = {}
    for ckpt in ("C_SFT", "C_outcome"):
        # Preference order: token-logprob > elicited [0,100] > keyword presence.
        verb_auc, n_verb, n_correct = logprob_verbalized_auroc(logprob_paths[ckpt])
        if not np.isnan(verb_auc):
            verbal_kind = "token-logprob (literature standard)"
        else:
            rows = load_confidence_jsonl(confidence_paths[ckpt])
            verb_auc, n_verb, n_correct = (float("nan"), 0, 0)
            if rows:
                verb_auc, n_verb, n_correct = verbalized_auroc(rows)
            if np.isnan(verb_auc):
                print(f"[{ckpt}] no usable verbalized signal from elicitation; falling back to keyword presence.")
                verb_auc, n_verb, n_correct = keyword_verbalized_auroc(eval_paths[ckpt])
                verbal_kind = "keyword-presence (fallback)"
            else:
                verbal_kind = "verbalized [0,100] elicitation"

        probe_auc = probe_pre_answer_auroc(args.cache_dir, ckpt, args.layer)
        gap = probe_auc - verb_auc
        summary[ckpt] = {
            "verbalized": verb_auc, "probe": probe_auc, "gap": gap,
            "n_verb": n_verb, "n_correct": n_correct,
            "verbal_kind": verbal_kind,
        }

    print()
    print("=" * 86)
    print(f"Concealment gap = probe AUROC at </think> (L{args.layer}) - verbalized AUROC")
    print("=" * 86)
    header = (
        f"{'checkpoint':<14}{'n_rollouts':>12}{'n_correct':>11}"
        f"{'verbal_AUROC':>14}{'probe_AUROC':>14}{'gap':>10}"
    )
    print(header)
    print("-" * len(header))
    for ckpt, s in summary.items():
        print(
            f"{ckpt:<14}{s['n_verb']:>12}{s['n_correct']:>11}"
            f"{s['verbalized']:>14.3f}{s['probe']:>14.3f}{s['gap']:>+10.3f}"
        )
    print()
    for ckpt, s in summary.items():
        print(f"  {ckpt}: verbalized signal = {s['verbal_kind']}")

    # Joint analysis per checkpoint.
    print()
    print("=" * 86)
    print("Per-rollout joint: do probe score and verbalized confidence covary?")
    print("=" * 86)
    for ckpt, path in paths.items():
        rows = load_confidence_jsonl(path)
        if not rows:
            continue
        j = joint_probe_vs_verbalized(rows, args.cache_dir, ckpt, args.layer)
        if j is None:
            print(f"  {ckpt}: insufficient data")
            continue
        print(f"\n  --- {ckpt} (n={j['n']}) ---")
        print(f"    Pearson rho(probe, verbal) overall        = {j['rho_overall']:+.3f}")
        print(f"    rho on correct rollouts                   = {j['rho_on_correct']:+.3f}")
        print(f"    rho on wrong rollouts                     = {j['rho_on_wrong']:+.3f}")
        print(f"    mean probe score: correct={j['mean_probe_correct']:.3f}  "
              f"wrong={j['mean_probe_wrong']:.3f}  delta={j['mean_probe_correct']-j['mean_probe_wrong']:+.3f}")
        print(f"    mean verbal conf: correct={j['mean_verbal_correct']:.3f}  "
              f"wrong={j['mean_verbal_wrong']:.3f}  delta={j['mean_verbal_correct']-j['mean_verbal_wrong']:+.3f}")


if __name__ == "__main__":
    main()
