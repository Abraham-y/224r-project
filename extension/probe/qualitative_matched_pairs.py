"""Matched-pair qualitative figure for the concealment finding.

For each checkpoint and each prompt that has BOTH a correct rollout and a
wrong rollout (each emitting at least one assertion token):

  - Train a probe via GroupKFold(5) so every assertion vector gets a
    held-out probe score.
  - For each matched (correct, wrong) pair within the same prompt,
    compute the mean assertion-token probe score on each side.
  - The matched-pair AUROC compares these per-prompt means with a Wilcoxon
    sign-test style aggregation; the visceral story is the table of
    per-prompt (mean probe_score on correct, mean probe_score on wrong).

What should be visible:
  C_SFT     -> correct mean strongly > wrong mean per prompt.
  C_outcome -> the two means are roughly the same; no per-prompt
               discrimination at assertion positions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from extension.probe.qualitative_examples import cv_per_row_probe_scores, truncate_for_display


def matched_pairs(cache_dir: str, layer: int, ckpt: str, eval_json: str,
                  n_show: int = 3) -> None:
    ass_npz = os.path.join(cache_dir, f"{ckpt}_l{layer}_assertion.npz")
    ass_meta = os.path.join(cache_dir, f"{ckpt}_l{layer}_assertion.meta.json")
    if not (os.path.exists(ass_npz) and os.path.exists(ass_meta)):
        print(f"[skip] missing cache for {ckpt}")
        return
    if not os.path.exists(eval_json):
        print(f"[skip] missing eval JSON for {ckpt}: {eval_json}")
        return

    with np.load(ass_npz) as data:
        X = data["X"]; y = data["y"]
    with open(ass_meta) as f:
        meta = json.load(f)
    groups = np.array([row["prompt_idx"] for row in meta], dtype=np.int64)
    scores = cv_per_row_probe_scores(X, y, groups)

    # Per-rollout mean assertion-token probe score (only rollouts with >=1
    # assertion token are represented).
    rollout_scores: dict[tuple[int, int], dict] = {}
    for i, m in enumerate(meta):
        if np.isnan(scores[i]):
            continue
        key = (int(m["prompt_idx"]), int(m["resp_idx"]))
        if key not in rollout_scores:
            rollout_scores[key] = {"scores": [], "label": int(y[i])}
        rollout_scores[key]["scores"].append(float(scores[i]))

    # Group rollouts by prompt_idx.
    by_prompt: dict[int, dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: {"correct": [], "wrong": []}
    )
    for (p_idx, r_idx), v in rollout_scores.items():
        bucket = "correct" if v["label"] == 1 else "wrong"
        by_prompt[p_idx][bucket].append((r_idx, float(np.mean(v["scores"]))))

    # Matched-pair table: per prompt, mean(correct rollout means) vs mean(wrong rollout means).
    pairs = []
    for p_idx, buckets in by_prompt.items():
        if not buckets["correct"] or not buckets["wrong"]:
            continue
        correct_mean = float(np.mean([s for _, s in buckets["correct"]]))
        wrong_mean = float(np.mean([s for _, s in buckets["wrong"]]))
        pairs.append({
            "prompt_idx": p_idx,
            "correct_mean": correct_mean,
            "wrong_mean": wrong_mean,
            "delta": correct_mean - wrong_mean,
            "n_correct": len(buckets["correct"]),
            "n_wrong": len(buckets["wrong"]),
        })

    if not pairs:
        print(f"\n[{ckpt}] no prompts with both a correct and a wrong rollout containing assertions.")
        return

    n_pairs = len(pairs)
    deltas = np.array([p["delta"] for p in pairs])
    n_pos_delta = int((deltas > 0).sum())
    n_neg_delta = int((deltas < 0).sum())
    mean_delta = float(deltas.mean())
    median_delta = float(np.median(deltas))

    print()
    print("=" * 92)
    print(f"[{ckpt}] Matched-pair assertion-probe means at L{layer}")
    print("=" * 92)
    print(f"  prompts with both a correct AND a wrong rollout containing assertions: {n_pairs}")
    print(f"  mean(delta = correct_mean - wrong_mean):     {mean_delta:+.3f}")
    print(f"  median(delta):                                {median_delta:+.3f}")
    print(f"  prompts where probe ranks correct > wrong:    {n_pos_delta}/{n_pairs} "
          f"({100*n_pos_delta/n_pairs:.0f}%)")
    print(f"  prompts where probe ranks wrong > correct:    {n_neg_delta}/{n_pairs} "
          f"({100*n_neg_delta/n_pairs:.0f}%)")
    print()
    print(f"  {'prompt':>7}  {'n_corr':>7} {'n_wrong':>8}  "
          f"{'correct_mean':>14}  {'wrong_mean':>12}  {'delta':>10}")
    print("  " + "-" * 80)
    for p in sorted(pairs, key=lambda x: -x["delta"]):
        print(
            f"  {p['prompt_idx']:>7}  {p['n_correct']:>7} {p['n_wrong']:>8}  "
            f"{p['correct_mean']:>14.3f}  {p['wrong_mean']:>12.3f}  "
            f"{p['delta']:>+10.3f}"
        )

    # --- Show the most extreme pair as a concrete example. -----------
    with open(eval_json) as f:
        rows = [json.loads(line) for line in f.read().strip().splitlines() if line.strip()]

    pairs_sorted_by_abs = sorted(pairs, key=lambda x: -abs(x["delta"]))[:n_show]
    for k, p in enumerate(pairs_sorted_by_abs):
        p_idx = p["prompt_idx"]
        # Pick the highest-scoring correct rollout and the lowest-scoring wrong one.
        buckets = by_prompt[p_idx]
        best_correct = max(buckets["correct"], key=lambda t: t[1])
        worst_wrong = min(buckets["wrong"], key=lambda t: t[1])
        target = rows[p_idx]["target"]
        nums = rows[p_idx]["nums"]
        print()
        print("-" * 92)
        print(f"[{ckpt} matched-pair example {k+1}/{n_show}] prompt={p_idx}  "
              f"target={target}  nums={nums}")
        print(f"  correct rollout (resp_idx={best_correct[0]}): "
              f"mean assertion-probe = {best_correct[1]:.3f}")
        print(f"  wrong rollout   (resp_idx={worst_wrong[0]}): "
              f"mean assertion-probe = {worst_wrong[1]:.3f}")
        print(f"  delta = {best_correct[1] - worst_wrong[1]:+.3f}")
        print()
        print("  --- correct rollout text (truncated) ---")
        print("  " + truncate_for_display(
            rows[p_idx]["response"][best_correct[0]], max_chars=900
        ).replace("\n", "\n  "))
        print()
        print("  --- wrong rollout text (truncated) ---")
        print("  " + truncate_for_display(
            rows[p_idx]["response"][worst_wrong[0]], max_chars=900
        ).replace("\n", "\n  "))
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--n_show", type=int, default=2)
    parser.add_argument("--sft_eval", default="eval_sft.json")
    parser.add_argument("--outcome_eval", default="eval.json")
    args = parser.parse_args()

    matched_pairs(args.cache_dir, args.layer, "C_SFT", args.sft_eval, args.n_show)
    matched_pairs(args.cache_dir, args.layer, "C_outcome", args.outcome_eval, args.n_show)


if __name__ == "__main__":
    main()
