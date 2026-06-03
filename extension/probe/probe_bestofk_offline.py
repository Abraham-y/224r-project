"""Offline probe-guided best-of-K + top-M enrichment — the shortest demo that
plays to the probe's strength (selecting correct rollouts), and a free preview
of whether probe_topk RLOO would help.

On existing rollouts (one model, K per problem):
  1. Forward-pass each rollout, read the L<layer> hidden state at `</think>`.
  2. Train a probe with GroupKFold(5)-by-PROBLEM, so every rollout gets a
     HELD-OUT probe score (no leakage: a problem's rollouts are scored by a
     probe that never saw that problem).
  3. Per problem, pick the rollout with the highest held-out probe score
     (probe-best-of-K) and compare its accuracy to first / random / oracle.
  4. Top-M enrichment: are the top-M-by-probe rollouts in each group more
     likely correct than random-M? (This is the signal probe_topk RLOO would
     exploit during training.)

Self-contained. Runs on Modal; prints + saves a JSON to the volume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from extension.training.probe_reward_rloo import (
    _load_frozen_model, _think_close_hidden, _first_block_correct)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="asingh15/qwen-sft-countdown-defaultproj")
    ap.add_argument("--rollouts_json", default="eval_c_sft_n500.json")
    ap.add_argument("--contam_json", default="extension/data/contaminated_prompt_idx.json")
    ap.add_argument("--clean_only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--max_responses", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--out_dir", default="/vol/outputs/probe_bestofk")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    import evaluation.countdown as cd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    rows = [json.loads(l) for l in open(args.rollouts_json) if l.strip()]
    if args.clean_only and os.path.exists(args.contam_json):
        clean = set(json.load(open(args.contam_json))["clean"])
        rows = [r for i, r in enumerate(rows) if i in clean]
    print(f"[bestofk] {len(rows)} problems, model={args.model}", flush=True)

    tok, model = _load_frozen_model(args.model)

    # Per-rollout records (flat), grouped by problem.
    X, y, grp = [], [], []
    has_h = []                 # whether a hidden state was extractable
    per_problem: dict[int, list[int]] = {}
    gi = 0
    for pidx, row in enumerate(rows):
        gt = {"target": int(row["target"]), "numbers": list(row["nums"])}
        for resp in row["response"][: args.max_responses]:
            h = _think_close_hidden(tok, model, row["prompt"], resp, args.layer)
            X.append(h if h is not None else np.zeros(896, dtype=np.float32))
            has_h.append(h is not None)
            y.append(_first_block_correct(resp, gt, cd))
            grp.append(pidx)
            per_problem.setdefault(pidx, []).append(gi)
            gi += 1
        if (pidx + 1) % 100 == 0:
            print(f"[bestofk]   {pidx+1}/{len(rows)} problems", flush=True)

    X = np.asarray(X, np.float32); y = np.asarray(y, int)
    grp = np.asarray(grp); has_h = np.asarray(has_h)
    print(f"[bestofk] {len(y)} rollouts, {(~has_h).sum()} without </think>; "
          f"overall first-block acc={y.mean():.3f}", flush=True)

    # Held-out probe score per rollout (only rollouts with a hidden state).
    probe_score = np.full(len(y), -1.0)   # -1 => never selected as best
    valid = np.where(has_h)[0]
    Xv, yv, gv = X[valid], y[valid], grp[valid]
    preds = np.full(len(valid), np.nan)
    for tr, te in GroupKFold(5).split(Xv, yv, gv):
        if len(set(yv[tr].tolist())) < 2:
            continue
        pipe = Pipeline([("s", StandardScaler()),
                         ("l", LogisticRegression(C=0.1, max_iter=2000))]).fit(Xv[tr], yv[tr])
        preds[te] = pipe.predict_proba(Xv[te])[:, 1]
    m = ~np.isnan(preds)
    auroc = float(roc_auc_score(yv[m], preds[m])) if len(set(yv[m].tolist())) == 2 else float("nan")
    probe_score[valid] = np.where(np.isnan(preds), -1.0, preds)
    print(f"[bestofk] held-out probe AUROC (first-block, </think> L{args.layer}) = {auroc:.3f}", flush=True)

    # Per-problem selection metrics.
    rng = np.random.RandomState(0)
    first_acc, rand_acc, probe_acc, oracle_acc = [], [], [], []
    # Top-M enrichment accumulators.
    Ms = [1, 2, 4, 8]
    topM_correct = {M: [] for M in Ms}
    base_rate = []
    for pidx, idxs in per_problem.items():
        idxs = np.array(idxs)
        yy = y[idxs]; ps = probe_score[idxs]
        first_acc.append(yy[0])
        rand_acc.append(yy.mean())                       # expected random pick
        oracle_acc.append(1.0 if yy.max() == 1 else 0.0)  # pass@K
        probe_acc.append(yy[int(np.argmax(ps))])          # probe-best-of-K pick
        order = np.argsort(-ps)                            # high probe first
        base_rate.append(yy.mean())
        for M in Ms:
            if len(idxs) >= M:
                topM_correct[M].append(yy[order[:M]].mean())

    res = {
        "model": args.model, "rollouts_json": args.rollouts_json,
        "n_problems": len(per_problem), "probe_auroc_first_block": auroc,
        "pass_at_1_first_rollout": float(np.mean(first_acc)),
        "random_pick": float(np.mean(rand_acc)),
        "probe_best_of_K": float(np.mean(probe_acc)),
        "oracle_pass_at_K": float(np.mean(oracle_acc)),
        "lift_probe_over_random": float(np.mean(probe_acc) - np.mean(rand_acc)),
        "frac_of_oracle_captured": float((np.mean(probe_acc) - np.mean(rand_acc))
                                         / max(1e-9, np.mean(oracle_acc) - np.mean(rand_acc))),
        "topM_enrichment": {str(M): {"top_correct": float(np.mean(topM_correct[M])),
                                     "base_rate": float(np.mean(base_rate)),
                                     "ratio": float(np.mean(topM_correct[M]) / max(1e-9, np.mean(base_rate)))}
                            for M in Ms},
    }
    print("\n=== probe best-of-K selection (held-out) ===", flush=True)
    print(f"  random pick (pass@1 baseline) : {res['random_pick']:.3f}", flush=True)
    print(f"  probe best-of-K               : {res['probe_best_of_K']:.3f}  "
          f"(+{res['lift_probe_over_random']*100:.1f} pp)", flush=True)
    print(f"  oracle pass@K (ceiling)       : {res['oracle_pass_at_K']:.3f}", flush=True)
    print(f"  fraction of oracle captured   : {res['frac_of_oracle_captured']*100:.0f}%", flush=True)
    print("\n=== top-M-by-probe enrichment (probe_topk preview) ===", flush=True)
    for M in Ms:
        e = res["topM_enrichment"][str(M)]
        print(f"  top-{M}: {e['top_correct']:.3f} correct vs base {e['base_rate']:.3f} "
              f"-> {e['ratio']:.2f}x", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[bestofk] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
