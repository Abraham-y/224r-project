"""Probe-as-answer-selector: in multi-answer C_outcome rollouts, simulate
committing at the FIRST `<answer>` block whose probe score crosses threshold.

This tests the user's question: at 0.5B C_outcome we see 87% of rollouts emit
multiple `<answer>` blocks (mean 7.6), and 9% are T->F drift rollouts where
the model emits a correct equation early then drifts to a wrong final one
(which is what the verifier scores). The position-appropriate `<answer>`-
opening probe (writeup §2.4, held-out AUROC 0.920) can score each block as
it appears. If we commit at the first probe-confident block, can we recover
the T->F drift rollouts?

Procedure (purely local, ~5 min CPU):
  1. Train held-out probe on the existing answer-opening cache via
     GroupKFold(5) by prompt_idx (same as §2.4).
  2. For each (prompt_idx, resp_idx) rollout in clean-406:
        baseline: verifier-scored label (= score == 1.0)
        commit-first-above-threshold: scan blocks in order; if probe_t > T,
          return that block's correctness; else fall back to verifier-scored.
  3. Sweep threshold; report headline accuracy at the optimal threshold.

Run on 0.5B clean-406 only -- 1.5B doesn't ramble.
"""

from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

CACHE = "extension/cache/probe_cache_n500_answers/C_outcome_l16_answers.npz"
META = CACHE.replace(".npz", ".meta.json")
CONTAM = "extension/data/contaminated_prompt_idx.json"
EVAL = "eval_c_outcome_n500.json"
OUT_TXT = "extension/outputs/n500/text/26_probe_answer_commit.txt"
OUT_JSON = "extension/outputs/n500/text/26_probe_answer_commit.json"
OUT_FIG = "extension/outputs/n500/figures/fig14_probe_answer_commit.png"


def main():
    print("[commit] loading cache + meta", flush=True)
    with np.load(CACHE) as d:
        X = d["X"]
    meta = json.load(open(META))
    # Filter to rows with known block_correct labels
    keep = np.array([m.get("block_correct") is not None for m in meta], dtype=bool)
    X = X[keep]
    meta = [meta[i] for i in range(len(meta)) if keep[i]]
    y = np.array([1 if m["block_correct"] else 0 for m in meta], dtype=np.int32)
    groups = np.array([int(m["prompt_idx"]) for m in meta])
    print(f"[commit] {len(y)} <answer>-opening positions, {len(set(groups.tolist()))} prompts, pos%={y.mean():.1%}")

    # Filter to clean-406 prompts
    clean = set(int(i) for i in json.load(open(CONTAM))["clean"])
    cmask = np.array([int(g) in clean for g in groups])
    X, y, groups, meta = X[cmask], y[cmask], groups[cmask], [meta[i] for i in range(len(meta)) if cmask[i]]
    print(f"[commit] clean-406 only: {len(y)} positions across {len(set(groups.tolist()))} prompts")

    # Held-out probe scores via GroupKFold(5)
    print("[commit] training held-out probes...", flush=True)
    scores = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
        pipe.fit(X[tr], y[tr])
        scores[te] = pipe.predict_proba(X[te])[:, 1]

    # Load eval JSON to get verifier-scored label per (prompt_idx, resp_idx)
    print("[commit] loading verifier-scored labels from eval JSON", flush=True)
    eval_rows = [json.loads(l) for l in open(EVAL) if l.strip()]
    verifier_label = {}
    for p_idx, row in enumerate(eval_rows):
        for r_idx, s in enumerate(row.get("scores", [])):
            verifier_label[(p_idx, r_idx)] = 1 if float(s) == 1.0 else 0

    # Group probe scores + block_correct by (prompt_idx, resp_idx), ordered by answer_block_idx
    by_rollout: dict[tuple[int, int], list[tuple[int, float, int]]] = defaultdict(list)
    for i, m in enumerate(meta):
        if np.isnan(scores[i]):
            continue
        by_rollout[(int(m["prompt_idx"]), int(m["resp_idx"]))].append(
            (int(m["answer_block_idx"]), float(scores[i]), int(y[i]))
        )
    for k in by_rollout:
        by_rollout[k].sort()  # by block_idx

    # ---- Baselines ----
    # Verifier-scored: just the eval JSON's score
    # Last-block-correctness: the cached block_correct of the LAST seen block
    # First-block-correctness: the cached block_correct of the FIRST seen block
    n_rollouts = 0
    verifier_acc = 0
    last_acc = 0
    first_acc = 0
    # Strategy: commit at first block whose probe > threshold; if none, default to LAST block
    # We sweep threshold
    thresholds = np.linspace(0.05, 0.95, 19)
    commit_correct = {t: 0 for t in thresholds}
    fallback_uses = {t: 0 for t in thresholds}

    # Limit to rollouts with at least 1 cached block (so we have probe data)
    for (p, r), lst in by_rollout.items():
        if not lst: continue
        n_rollouts += 1
        verifier_acc += verifier_label.get((p, r), 0)
        first_acc += lst[0][2]  # first block correctness
        last_acc += lst[-1][2]  # last block correctness (caveat: this is "last cached" not "verifier's last" if some blocks failed caching)
        for t in thresholds:
            picked = None
            for bi, sc, bc in lst:
                if sc >= t:
                    picked = bc
                    break
            if picked is None:
                fallback_uses[t] += 1
                # Fallback: take the last block's correctness from cache
                picked = lst[-1][2]
            commit_correct[t] += picked

    print(f"\n[commit] analyzing {n_rollouts} multi-answer rollouts on 0.5B C_outcome clean-406")
    print(f"[commit] (excluded: rollouts where Phase 2A cache had no valid blocks)")
    print(f"\n=== Baselines (committed-correctness aggregated over multi-answer rollouts) ===")
    print(f"  verifier-scored (eval JSON, last <answer> rule): {verifier_acc / n_rollouts:.4f}")
    print(f"  first-block correctness (oracle pick-first):     {first_acc / n_rollouts:.4f}")
    print(f"  last-block correctness (cache's last):           {last_acc / n_rollouts:.4f}")

    print(f"\n=== Probe-commit (commit at first block with probe >= T, else fall back to last) ===")
    print(f"{'threshold':>10} {'accuracy':>10} {'gain vs verifier':>18} {'fallback rate':>15}")
    best_t = None; best_acc = -1
    rows_for_json = []
    for t in thresholds:
        acc = commit_correct[t] / n_rollouts
        fall = fallback_uses[t] / n_rollouts
        gain = acc - (verifier_acc / n_rollouts)
        marker = ""
        if acc > best_acc:
            best_acc = acc; best_t = float(t)
            marker = " ←"
        print(f"{t:>10.2f} {acc:>10.4f} {gain:>+18.4f} {fall:>15.2%}{marker}")
        rows_for_json.append({"threshold": float(t), "accuracy": float(acc), "gain_vs_verifier": float(gain), "fallback_rate": float(fall)})

    print(f"\nBest threshold = {best_t:.2f}  -> accuracy = {best_acc:.4f}")
    print(f"vs verifier-scored baseline = {verifier_acc / n_rollouts:.4f}")
    print(f"GAIN: {best_acc - verifier_acc / n_rollouts:+.4f}")

    # Save
    summary = {
        "n_rollouts_analyzed": n_rollouts,
        "verifier_scored_acc": float(verifier_acc / n_rollouts),
        "first_block_oracle_acc": float(first_acc / n_rollouts),
        "last_block_acc": float(last_acc / n_rollouts),
        "best_threshold": best_t,
        "best_acc": float(best_acc),
        "gain_vs_verifier": float(best_acc - verifier_acc / n_rollouts),
        "by_threshold": rows_for_json,
    }
    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    txt_lines = [
        f"Probe-as-answer-selector simulation on 0.5B C_outcome clean-406 multi-answer rollouts",
        f"  n_rollouts: {n_rollouts}",
        f"  verifier-scored baseline acc: {verifier_acc / n_rollouts:.4f}",
        f"  first-block-correctness (oracle pick-first): {first_acc / n_rollouts:.4f}",
        f"  last-block-correctness (cache): {last_acc / n_rollouts:.4f}",
        f"",
        f"  best probe-commit threshold: {best_t:.2f}",
        f"  best probe-commit acc: {best_acc:.4f}",
        f"  GAIN over verifier-scored baseline: {best_acc - verifier_acc / n_rollouts:+.4f}",
    ]
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(txt_lines) + "\n")

    # Figure: accuracy vs threshold
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        ts = [r["threshold"] for r in rows_for_json]
        accs = [r["accuracy"] for r in rows_for_json]
        ax.plot(ts, accs, marker="o", color="#3a6dba", lw=2, label="probe-commit (commit at first block ≥ threshold)")
        ax.axhline(verifier_acc / n_rollouts, color="#c45252", lw=1.6, ls="--", label=f"verifier-scored baseline (last <answer>): {verifier_acc/n_rollouts:.3f}")
        ax.axhline(first_acc / n_rollouts, color="#3a8b2f", lw=1.6, ls=":", label=f"oracle pick-first: {first_acc/n_rollouts:.3f}")
        ax.axhline(last_acc / n_rollouts, color="#888888", lw=1.0, ls=":", label=f"cache-last-block: {last_acc/n_rollouts:.3f}")
        ax.set_xlabel("probe threshold (commit at first block with probe ≥ T)")
        ax.set_ylabel("accuracy on multi-answer C_outcome rollouts")
        ax.set_title(f"Probe-as-answer-selector on 0.5B C_outcome (clean-406, n={n_rollouts} multi-answer rollouts)\n"
                     f"best threshold={best_t:.2f} → acc={best_acc:.3f} (vs verifier {verifier_acc/n_rollouts:.3f}; gain {best_acc - verifier_acc/n_rollouts:+.3f})")
        ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        os.makedirs(os.path.dirname(OUT_FIG) or ".", exist_ok=True)
        fig.savefig(OUT_FIG)
        plt.close(fig)
        print(f"\nwrote {OUT_FIG}")
    except Exception as e:
        print(f"figure write failed: {e}")

    print(f"wrote {OUT_TXT} and {OUT_JSON}")


if __name__ == "__main__":
    main()
