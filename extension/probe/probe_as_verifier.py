"""Probe-best-of-K as a verifier: pass@1 gain from probe-based rejection.

For each prompt with K=16 existing rollouts:
  1. Train held-out probe scores via GroupKFold(5) by prompt_idx
     (probe trained on OTHER problems, applied to this prompt's K rollouts)
  2. Rank the K rollouts by probe score; "probe-best-of-K" pick = argmax.
  3. Score: is the picked rollout correct (label == 1)?
  4. Compare to:
       - pass@1 (random pick, i.e. first rollout's correctness)
       - pass@K (oracle: any of K correct)
       - majority vote (if available)
       - average of top-k by probe (k=2,3,5)

This is the inference-time analog of trained-reset-token approaches:
we use the probe as the "wrong-trajectory detector" externally.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def load_cell(cache_dir, ckpt, layer, kind):
    npz = os.path.join(cache_dir, f"{ckpt}_l{layer}_{kind}.npz")
    if not os.path.exists(npz):
        return None
    with np.load(npz) as d:
        X = d["X"]; y = d["y"]
    meta = json.load(open(npz.replace(".npz", ".meta.json")))
    groups = np.array([int(m["prompt_idx"]) for m in meta])
    resps = np.array([int(m["resp_idx"]) for m in meta])
    return X, y, groups, resps


def heldout_scores(X, y, groups, n_splits=5, C=0.1):
    scores = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        pipe = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(C=C, max_iter=2000))])
        pipe.fit(X[tr], y[tr])
        scores[te] = pipe.predict_proba(X[te])[:, 1]
    return scores


def per_problem_table(scores, labels, groups, resps):
    """Return list of dicts: one per prompt with sorted rollouts by probe."""
    by_p = defaultdict(list)
    for i in range(len(scores)):
        if np.isnan(scores[i]):
            continue
        by_p[int(groups[i])].append((float(scores[i]), int(labels[i]), int(resps[i])))
    out = []
    for p, lst in by_p.items():
        lst_sorted = sorted(lst, key=lambda x: -x[0])  # highest probe first
        out.append({
            "prompt_idx": p,
            "n_rollouts": len(lst_sorted),
            "any_correct": int(any(t[1] == 1 for t in lst_sorted)),
            "majority_correct": int(sum(t[1] == 1 for t in lst_sorted) > len(lst_sorted) // 2),
            "pass_at_1": int(lst_sorted[0][1]) if lst_sorted else 0,  # NOT random; this is probe-best-of-K
            "first_in_seq": None,  # filled below
            "labels_by_probe_rank": [t[1] for t in lst_sorted],
            "probe_scores_by_probe_rank": [t[0] for t in lst_sorted],
            "resp_idx_by_probe_rank": [t[2] for t in lst_sorted],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True,
                    help="extension/cache/probe_cache_n500_clean406 (0.5B) or "
                         "extension/cache/probe_cache_1.5b_clean406 (1.5B)")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--label", default="ckpt", help="label for printout")
    ap.add_argument("--eval_json", required=True,
                    help="for the first-in-sequence baseline (resp_idx=0)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval_json) if l.strip()]
    # Build the FIRST-rollout (in sample order) outcome per prompt -> the standard "random pick" baseline.
    first_label_by_p = {p_idx: int(float(row["scores"][0]) == 1.0) for p_idx, row in enumerate(rows)}
    all_labels_by_p = {p_idx: [int(float(s) == 1.0) for s in row["scores"]] for p_idx, row in enumerate(rows)}

    cell_sft = load_cell(args.cache_dir, "C_SFT", args.layer, "pre_answer")
    cell_out = load_cell(args.cache_dir, "C_outcome", args.layer, "pre_answer")
    if cell_sft is None or cell_out is None:
        raise SystemExit(f"missing cache in {args.cache_dir}")

    print(f"=== Probe-as-verifier @ trace-final, {args.label} ===")
    for name, cell in (("C_SFT", cell_sft), ("C_outcome", cell_out)):
        X, y, g, r = cell
        print(f"\n--- {name}: {len(y)} rollouts, {len(set(g.tolist()))} prompts, pos%={y.mean():.1%}")
        scores = heldout_scores(X, y, g)
        per_p = per_problem_table(scores, y, g, r)

        # Aggregate over prompts that have ALL K rollouts captured in the cache.
        # (Some rollouts may have failed to find a </think> token; for those prompts, skip.)
        K_observed = max(p["n_rollouts"] for p in per_p)
        # Use only prompts where the cache has K >= 8 rollouts.
        per_p_full = [p for p in per_p if p["n_rollouts"] >= 8]
        n_p = len(per_p_full)
        print(f"  prompts with >=8 cached rollouts: {n_p}  (using these for selection metrics)")

        # First-rollout baseline (= the existing pass@1)
        pass_at_1 = float(np.mean([first_label_by_p[p["prompt_idx"]] for p in per_p_full]))
        # Oracle pass@K
        pass_at_k = float(np.mean([p["any_correct"] for p in per_p_full]))
        # Random pick over the K cached rollouts (expectation = mean acc per prompt)
        rand_pick = float(np.mean([np.mean(p["labels_by_probe_rank"]) for p in per_p_full]))
        # Majority of K
        majority = float(np.mean([p["majority_correct"] for p in per_p_full]))
        # Probe-best-of-K (top-1 by probe)
        probe_top1 = float(np.mean([p["labels_by_probe_rank"][0] for p in per_p_full]))
        # Probe-best-of-top-k average
        def probe_topk_acc(k):
            vals = []
            for p in per_p_full:
                top = p["labels_by_probe_rank"][:k]
                vals.append(np.mean(top))
            return float(np.mean(vals))
        probe_top2 = probe_topk_acc(2)
        probe_top3 = probe_topk_acc(3)
        probe_top5 = probe_topk_acc(5)
        # Probe-worst-of-K (for sanity: should be much LOWER than probe-best if probe works)
        probe_bot1 = float(np.mean([p["labels_by_probe_rank"][-1] for p in per_p_full]))

        # Aggregate AUROC (over rollouts with valid probe scores)
        aurocs = []  # per-problem
        for p in per_p_full:
            labs = p["labels_by_probe_rank"]
            scs = p["probe_scores_by_probe_rank"]
            if len(set(labs)) >= 2:
                aurocs.append(roc_auc_score(labs, scs))
        avg_perp_auroc = float(np.mean(aurocs)) if aurocs else float("nan")

        print(f"  pass@1 (first-rollout, real)               : {pass_at_1:.3f}")
        print(f"  random pick over K cached (expected)       : {rand_pick:.3f}")
        print(f"  majority vote of K                         : {majority:.3f}")
        print(f"  probe-best-of-K (argmax)                    : {probe_top1:.3f}   [gain over pass@1: {probe_top1 - pass_at_1:+.3f}]")
        print(f"  probe-top-2 mean                            : {probe_top2:.3f}")
        print(f"  probe-top-3 mean                            : {probe_top3:.3f}")
        print(f"  probe-top-5 mean                            : {probe_top5:.3f}")
        print(f"  probe-worst-of-K (sanity, should be low)    : {probe_bot1:.3f}")
        print(f"  oracle pass@K                              : {pass_at_k:.3f}")
        print(f"  per-problem AUROC (mean over {len(aurocs)} probs)  : {avg_perp_auroc:.3f}")

        # Save summary
        summary = {
            "checkpoint": name, "layer": args.layer, "n_prompts": n_p,
            "pass@1": pass_at_1, "random_pick_avg": rand_pick,
            "majority_of_K": majority, "oracle_pass@K": pass_at_k,
            "probe_best_of_K": probe_top1, "probe_top2_mean": probe_top2,
            "probe_top3_mean": probe_top3, "probe_top5_mean": probe_top5,
            "probe_worst_of_K": probe_bot1,
            "gain_probe_vs_pass1": probe_top1 - pass_at_1,
            "gain_probe_vs_majority": probe_top1 - majority,
            "per_problem_auroc_mean": avg_perp_auroc,
        }
        out_path = (args.out or f"extension/outputs/n500/text_{args.label}/25_probe_as_verifier_{name}.json")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
