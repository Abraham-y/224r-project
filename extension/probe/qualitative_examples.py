"""Qualitative concealment-figure: per-rollout probe scores at assertion tokens.

For each of {C_SFT, C_outcome}:
  - Train a probe on the assertion-position cache with GroupKFold(5),
    so every vector gets a held-out cross-validated probe score.
  - Group scores by (prompt_idx, resp_idx) using the meta sidecar.
  - Filter to *wrong* rollouts that emitted at least 2 assertion tokens.
  - Print 3 representative examples with probe-score annotations.

The contrast that should be visible:
  C_SFT     -> probe scores at assertion tokens are usually < 0.4 on wrong
               rollouts (probe internally "knows" the answer is wrong).
  C_outcome -> probe scores are clustered near 0.5 regardless (probe is
               uninformative at the verbalization positions).
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


def cv_per_row_probe_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                            n_splits: int = 5, C: float = 0.1) -> np.ndarray:
    """Return per-row probe scores from a GroupKFold CV; never trains on a
    row's own group's data."""
    out = np.full(len(y), np.nan)
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        return out
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(scaler.transform(X[tr]), y[tr])
        out[te] = clf.predict_proba(scaler.transform(X[te]))[:, 1]
    return out


def truncate_for_display(text: str, max_chars: int = 1400) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated, +{len(text) - max_chars} more chars]"


def print_examples_for_checkpoint(
    cache_dir: str, layer: int, ckpt: str, eval_json: str, n_examples: int = 3
) -> None:
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

    # Group rows by (prompt_idx, resp_idx).
    by_rollout: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for i, m in enumerate(meta):
        if np.isnan(scores[i]):
            continue
        by_rollout[(int(m["prompt_idx"]), int(m["resp_idx"]))].append({
            "tok_idx": int(m["tok_idx"]),
            "keyword": m.get("keyword", "?"),
            "probe_score": float(scores[i]),
            "is_correct": bool(y[i]),
        })

    with open(eval_json) as f:
        rows = [json.loads(line) for line in f.read().strip().splitlines() if line.strip()]

    # Wrong rollouts that emitted >= 2 assertion tokens.
    candidates = []
    for (p_idx, r_idx), assertions in by_rollout.items():
        if assertions[0]["is_correct"]:
            continue
        if len(assertions) < 2:
            continue
        if p_idx >= len(rows) or r_idx >= len(rows[p_idx]["response"]):
            continue
        candidates.append({
            "prompt_idx": p_idx, "resp_idx": r_idx,
            "target": rows[p_idx]["target"],
            "nums": rows[p_idx]["nums"],
            "assertions": sorted(assertions, key=lambda a: a["tok_idx"]),
            "response": rows[p_idx]["response"][r_idx],
        })

    # Sort by mean probe score (lowest = "probe most-confidently-wrong" for
    # C_SFT; clustered near 0.5 for C_outcome).
    candidates.sort(key=lambda c: float(np.mean([a["probe_score"] for a in c["assertions"]])))

    print()
    print("=" * 92)
    print(f"WRONG rollouts on {ckpt} (held-out per-row probe scores at assertion tokens, L{layer})")
    print("=" * 92)
    print(f"  total wrong-rollouts-with-2+-assertions: {len(candidates)}")
    print()
    for k, c in enumerate(candidates[:n_examples]):
        scores_list = [a["probe_score"] for a in c["assertions"]]
        mean_score = float(np.mean(scores_list))
        min_score = float(np.min(scores_list))
        max_score = float(np.max(scores_list))
        print("-" * 92)
        print(f"[{ckpt} EX {k + 1}/{n_examples}]  Target: {c['target']}   Numbers: {c['nums']}")
        print(f"  rollout: prompt_idx={c['prompt_idx']}, resp_idx={c['resp_idx']}, "
              f"correct=False (final answer was wrong)")
        print(f"  assertion-position probe scores: "
              f"mean={mean_score:.3f}  min={min_score:.3f}  max={max_score:.3f}")
        print(f"  individual probe scores per assertion (in trace order):")
        for a in c["assertions"][:10]:
            verdict = (
                "probe -> CORRECT" if a["probe_score"] > 0.6 else
                "probe -> WRONG  " if a["probe_score"] < 0.4 else
                "probe -> UNSURE "
            )
            print(f"    '{a['keyword']}' at tok {a['tok_idx']:>4}: "
                  f"probe(correct)={a['probe_score']:.3f}   {verdict}")
        print()
        print("--- response (truncated) ---")
        print(truncate_for_display(c["response"]))
        print("--- end response ---")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--n_examples", type=int, default=3)
    parser.add_argument("--sft_eval", default="eval_sft.json")
    parser.add_argument("--outcome_eval", default="eval.json")
    args = parser.parse_args()

    print_examples_for_checkpoint(
        args.cache_dir, args.layer, "C_SFT", args.sft_eval, args.n_examples
    )
    print_examples_for_checkpoint(
        args.cache_dir, args.layer, "C_outcome", args.outcome_eval, args.n_examples
    )


if __name__ == "__main__":
    main()
