"""Probe-answer-commit variants: pick-max + higher-threshold sweep with
smarter fallbacks.

Builds on probe_answer_commit.py. For each 0.5B C_outcome multi-answer
rollout in clean-406, we test several block-picking strategies:

  Baselines:
    BASE0:  verifier-scored (last <answer> block, per the actual verifier)
    BASE1:  always commit at FIRST block (oracle pick-first)
    BASE2:  random block (mean correctness across all blocks)
    UPPER:  any-block-correct (oracle upper bound; pass-at-K-blocks)
    UPPER2: oracle commit-at-first-correct-block

  Probe strategies (variants of the user's question):
    PROBE_COMMIT_T:   first block with probe >= T; fallback to LAST block
    PROBE_COMMIT_FB:  first block with probe >= T; fallback to argmax-probe
    PROBE_MAX:        argmax probe score across all blocks (no threshold)
    PROBE_TOP_K:      mean correctness of top-K blocks by probe (K=1,2,3,5)

All cached from existing Phase 2A hidden-state cache. No new Modal compute.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings("ignore")


CACHE = "extension/cache/probe_cache_n500_answers/C_outcome_l16_answers.npz"
META = CACHE.replace(".npz", ".meta.json")
CONTAM = "extension/data/contaminated_prompt_idx.json"
EVAL = "eval_c_outcome_n500.json"
OUT_TXT = "extension/outputs/n500/text/27_probe_answer_variants.txt"
OUT_FIG = "extension/outputs/n500/figures/fig15_probe_commit_variants.png"


def main():
    print("[variants] loading cache + meta", flush=True)
    with np.load(CACHE) as d:
        X = d["X"]
    meta = json.load(open(META))
    keep = np.array([m.get("block_correct") is not None for m in meta], dtype=bool)
    X = X[keep]
    meta = [meta[i] for i in range(len(meta)) if keep[i]]
    y = np.array([1 if m["block_correct"] else 0 for m in meta], dtype=np.int32)
    groups = np.array([int(m["prompt_idx"]) for m in meta])

    clean = set(int(i) for i in json.load(open(CONTAM))["clean"])
    cmask = np.array([int(g) in clean for g in groups])
    X, y, groups, meta = X[cmask], y[cmask], groups[cmask], [meta[i] for i in range(len(meta)) if cmask[i]]

    # held-out probe scores
    print("[variants] training held-out probe via GroupKFold(5)", flush=True)
    scores = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
        pipe.fit(X[tr], y[tr])
        scores[te] = pipe.predict_proba(X[te])[:, 1]

    by_rollout: dict[tuple[int, int], list[tuple[int, float, int]]] = defaultdict(list)
    for i, m in enumerate(meta):
        if np.isnan(scores[i]):
            continue
        by_rollout[(int(m["prompt_idx"]), int(m["resp_idx"]))].append(
            (int(m["answer_block_idx"]), float(scores[i]), int(y[i]))
        )
    for k in by_rollout:
        by_rollout[k].sort()  # by block_idx

    # Verifier label from eval JSON (so BASE0 matches the actual deployed scorer)
    eval_rows = [json.loads(l) for l in open(EVAL) if l.strip()]
    verifier_label = {
        (p_idx, r_idx): (1 if float(s) == 1.0 else 0)
        for p_idx, row in enumerate(eval_rows)
        for r_idx, s in enumerate(row.get("scores", []))
    }

    # Only multi-answer rollouts with valid probe scores
    rollouts = [(k, v) for k, v in by_rollout.items() if len(v) >= 2]
    print(f"[variants] {len(rollouts)} multi-answer rollouts with valid probe scores")

    # ---- Baselines ----
    n = len(rollouts)
    base_verifier = sum(verifier_label.get(k, 0) for k, _ in rollouts) / n
    base_first = sum(v[0][2] for _, v in rollouts) / n
    base_last = sum(v[-1][2] for _, v in rollouts) / n
    base_rand = sum(np.mean([t[2] for t in v]) for _, v in rollouts) / n
    upper_any = sum(int(any(t[2] == 1 for t in v)) for _, v in rollouts) / n
    upper_commit_at_first_correct = 0.0
    for _, v in rollouts:
        commit_at_first_correct = next((t[2] for t in v if t[2] == 1), v[-1][2])
        upper_commit_at_first_correct += commit_at_first_correct
    upper_commit_at_first_correct /= n

    print(f"\n=== Baselines & upper bounds (n={n} multi-answer rollouts) ===")
    print(f"  BASE0  verifier-scored (last block)                    : {base_verifier:.4f}")
    print(f"  BASE1  oracle pick-first                               : {base_first:.4f}")
    print(f"  BASE2  random block (mean over all blocks)             : {base_rand:.4f}")
    print(f"  ----  cached-last block correctness                    : {base_last:.4f}")
    print(f"  UPPER  any block correct (probe-free oracle pass@K_blk): {upper_any:.4f}")
    print(f"  UPPER2 oracle commit-at-first-correct (if any)         : {upper_commit_at_first_correct:.4f}")

    # ---- Probe strategies ----

    # 1. probe-max: argmax probe score across all blocks
    probe_max_acc = 0
    for _, v in rollouts:
        argmax_i = max(range(len(v)), key=lambda i: v[i][1])
        probe_max_acc += v[argmax_i][2]
    probe_max_acc /= n

    # 2. probe-top-K mean
    def probe_topk(k):
        s = 0
        for _, v in rollouts:
            sorted_v = sorted(v, key=lambda t: -t[1])
            top = sorted_v[:k]
            s += np.mean([t[2] for t in top])
        return s / n

    # 3. probe-commit-T with fallback to LAST (original)
    def probe_commit_fallback_last(T):
        s = 0; fb = 0
        for _, v in rollouts:
            picked = None
            for bi, sc, bc in v:
                if sc >= T:
                    picked = bc
                    break
            if picked is None:
                fb += 1
                picked = v[-1][2]
            s += picked
        return s / n, fb / n

    # 4. probe-commit-T with fallback to argmax-probe (NEW)
    def probe_commit_fallback_argmax(T):
        s = 0; fb = 0
        for _, v in rollouts:
            picked = None
            for bi, sc, bc in v:
                if sc >= T:
                    picked = bc
                    break
            if picked is None:
                fb += 1
                argmax_i = max(range(len(v)), key=lambda i: v[i][1])
                picked = v[argmax_i][2]
            s += picked
        return s / n, fb / n

    print(f"\n=== Probe strategies ===")
    print(f"  PROBE-MAX  (argmax over all blocks, no threshold)      : {probe_max_acc:.4f}   [gain vs BASE0: {probe_max_acc - base_verifier:+.4f}]")
    print(f"  PROBE-TOP-2 mean                                       : {probe_topk(2):.4f}")
    print(f"  PROBE-TOP-3 mean                                       : {probe_topk(3):.4f}")
    print(f"  PROBE-TOP-5 mean                                       : {probe_topk(5):.4f}")

    print(f"\n  Threshold sweep with two fallback strategies:")
    print(f"  {'T':>6} {'commit→last':>14} {'commit→argmax':>16} {'fallback%':>12}")
    rows = []
    for T in np.linspace(0.05, 0.99, 20):
        a1, f1 = probe_commit_fallback_last(T)
        a2, f2 = probe_commit_fallback_argmax(T)
        print(f"  {T:>6.2f} {a1:>14.4f} {a2:>16.4f} {100*f1:>11.1f}%")
        rows.append({"T": float(T), "fallback_last": float(a1), "fallback_argmax": float(a2), "fallback_rate": float(f1)})

    # Headline
    print(f"\n=== Headline ===")
    best_last = max(rows, key=lambda r: r["fallback_last"])
    best_argmax = max(rows, key=lambda r: r["fallback_argmax"])
    print(f"  best threshold (fallback=last)   : T={best_last['T']:.2f}, acc={best_last['fallback_last']:.4f} (gain {best_last['fallback_last']-base_verifier:+.4f})")
    print(f"  best threshold (fallback=argmax) : T={best_argmax['T']:.2f}, acc={best_argmax['fallback_argmax']:.4f} (gain {best_argmax['fallback_argmax']-base_verifier:+.4f})")
    print(f"  PROBE-MAX                         : acc={probe_max_acc:.4f}            (gain {probe_max_acc-base_verifier:+.4f})")
    print(f"")
    print(f"  Upper bound (oracle any-correct)  : {upper_any:.4f}")
    print(f"  Oracle commit-at-first-correct    : {upper_commit_at_first_correct:.4f}")

    # Save txt
    txt = []
    txt.append(f"Probe-answer-commit variants on 0.5B C_outcome multi-answer rollouts (n={n})")
    txt.append(f"")
    txt.append(f"=== Baselines ===")
    txt.append(f"  BASE0  verifier-scored (last block)                    : {base_verifier:.4f}")
    txt.append(f"  BASE1  oracle pick-first                               : {base_first:.4f}")
    txt.append(f"  BASE2  random block                                    : {base_rand:.4f}")
    txt.append(f"  UPPER  any block correct (probe-free upper bound)      : {upper_any:.4f}")
    txt.append(f"  UPPER2 oracle commit-at-first-correct                  : {upper_commit_at_first_correct:.4f}")
    txt.append(f"")
    txt.append(f"=== Probe strategies ===")
    txt.append(f"  PROBE-MAX                                              : {probe_max_acc:.4f}  (gain vs BASE0: {probe_max_acc-base_verifier:+.4f})")
    txt.append(f"  PROBE-TOP-2 mean                                       : {probe_topk(2):.4f}")
    txt.append(f"  PROBE-TOP-3 mean                                       : {probe_topk(3):.4f}")
    txt.append(f"  PROBE-COMMIT @ T={best_last['T']:.2f}, fb=LAST          : {best_last['fallback_last']:.4f}")
    txt.append(f"  PROBE-COMMIT @ T={best_argmax['T']:.2f}, fb=ARGMAX       : {best_argmax['fallback_argmax']:.4f}")
    txt.append(f"")
    txt.append(f"=== Threshold sweep ===")
    txt.append(f"  {'T':>6} {'fb=last':>14} {'fb=argmax':>16} {'fallback%':>12}")
    for r in rows:
        txt.append(f"  {r['T']:>6.2f} {r['fallback_last']:>14.4f} {r['fallback_argmax']:>16.4f} {100*r['fallback_rate']:>11.1f}%")
    os.makedirs(os.path.dirname(OUT_TXT) or ".", exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(txt) + "\n")
    print(f"\nwrote {OUT_TXT}")

    # Figure
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    Ts = [r["T"] for r in rows]
    last_accs = [r["fallback_last"] for r in rows]
    arg_accs = [r["fallback_argmax"] for r in rows]
    ax.plot(Ts, last_accs, marker="o", color="#3a6dba", lw=2,
            label=f"probe-commit (fallback=LAST block)")
    ax.plot(Ts, arg_accs, marker="s", color="#7d4ca2", lw=2,
            label=f"probe-commit (fallback=ARGMAX-probe block)")
    ax.axhline(base_verifier, color="#c45252", lw=1.6, ls="--",
               label=f"BASE0 verifier-scored: {base_verifier:.3f}")
    ax.axhline(probe_max_acc, color="#3a8b2f", lw=1.6, ls="-",
               label=f"PROBE-MAX (argmax over blocks): {probe_max_acc:.3f}")
    ax.axhline(base_first, color="#dba24c", lw=1.2, ls=":",
               label=f"BASE1 oracle pick-first: {base_first:.3f}")
    ax.axhline(upper_commit_at_first_correct, color="#888888", lw=1.2, ls=":",
               label=f"UPPER2 oracle commit-first-correct: {upper_commit_at_first_correct:.3f}")
    ax.axhline(upper_any, color="black", lw=1.0, ls=":",
               label=f"UPPER any block correct (pass@K_blocks): {upper_any:.3f}")
    ax.set_xlabel("probe threshold (commit at first block with probe ≥ T)")
    ax.set_ylabel("accuracy on 0.5B C_outcome multi-answer rollouts")
    ax.set_title(f"Probe-answer-commit variants — multi-answer C_outcome, clean-406, n={n}")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG) or ".", exist_ok=True)
    fig.savefig(OUT_FIG)
    plt.close(fig)
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
