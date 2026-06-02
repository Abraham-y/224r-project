"""Probe-as-eval-proxy + failure-mode diagnostic.

(D) Probe-as-eval-proxy: use the probe to estimate dataset accuracy
    without the verifier.
    For each prompt, mean probe across K rollouts ≈ per-prompt acc.
    Average across prompts -> dataset-level acc estimate.
    Test: how close is the estimated dataset acc to the true acc?
    Useful when ground truth is expensive (e.g., LLM-judge tasks).

(E) Probe failure-mode diagnostic: where does the probe make mistakes?
    Classify each rollout into:
      TP: probe ≥ 0.5 AND first-block correct
      TN: probe < 0.5 AND first-block wrong
      FP: probe ≥ 0.5 AND first-block wrong  (overconfidence)
      FN: probe < 0.5 AND first-block correct (underconfidence)
    For each class, compute statistics:
      - rambling rate (mean blocks/rollout)
      - rollout length
      - probe magnitude (how confident vs how wrong)
    Helps characterize WHEN the probe fails.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

CACHE = "extension/cache/probe_cache_n500_clean406/C_outcome_l16_pre_answer.npz"
META = CACHE.replace(".npz", ".meta.json")
EVAL = "eval_c_outcome_n500.json"
OUT_TXT = "extension/outputs/n500/text/37_probe_eval_proxy_and_failure.txt"
OUT_FIG = "extension/outputs/n500/figures/fig24_probe_eval_proxy.png"

_ANSWER_OPEN_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def check_block(eq, target, nums):
    try:
        nums_in_eq = [int(n) for n in re.findall(r"\d+", eq.strip())]
        if sorted(nums_in_eq) != sorted([int(x) for x in nums]):
            return False
        if not re.match(r"^[\d+\-*/().\s]+$", eq.strip()):
            return False
        result = eval(eq.strip(), {"__builtins__": None}, {})
        return abs(result - int(target)) < 1e-5
    except Exception:
        return False


def main():
    with np.load(CACHE) as d:
        X = d["X"]
    meta = json.load(open(META))
    groups = np.array([int(m["prompt_idx"]) for m in meta])
    rows = [json.loads(l) for l in open(EVAL) if l.strip()]

    first_correct = {}
    n_blocks = {}
    rollout_len = {}
    for p_idx, row in enumerate(rows):
        target = int(row["target"]); nums = list(row["nums"])
        for r_idx, resp in enumerate(row["response"]):
            matches = list(_ANSWER_OPEN_RE.finditer(resp))
            first_correct[(p_idx, r_idx)] = bool(matches) and check_block(matches[0].group(1), target, nums)
            n_blocks[(p_idx, r_idx)] = len(matches)
            rollout_len[(p_idx, r_idx)] = len(resp)

    y_new = np.array([int(first_correct.get((int(m["prompt_idx"]), int(m["resp_idx"])), 0)) for m in meta], dtype=np.int32)
    scores = np.full(len(y_new), np.nan)
    for tr, te in GroupKFold(5).split(X, y_new, groups):
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
        pipe.fit(X[tr], y_new[tr])
        scores[te] = pipe.predict_proba(X[te])[:, 1]

    out = []

    # (D) Probe-as-eval-proxy
    print("\n=== (D) Probe-as-eval-proxy: dataset-level accuracy estimate ===")
    # True per-prompt accuracy
    by_p = defaultdict(list)
    for i, m in enumerate(meta):
        if np.isnan(scores[i]): continue
        p = int(m["prompt_idx"]); r = int(m["resp_idx"])
        by_p[p].append((scores[i], int(first_correct.get((p, r), 0))))
    prompts = sorted(by_p.keys())
    per_p_acc_true = np.array([np.mean([t[1] for t in by_p[p]]) for p in prompts])
    per_p_probe_mean = np.array([np.mean([t[0] for t in by_p[p]]) for p in prompts])

    true_acc = float(np.mean(per_p_acc_true))
    est_acc_probe_mean = float(np.mean(per_p_probe_mean))
    est_acc_probe_thresh = float(np.mean(per_p_probe_mean >= 0.5))
    print(f"  True dataset accuracy (verifier): {true_acc:.4f}")
    print(f"  Probe-estimated (mean probe across rollouts, no thresh): {est_acc_probe_mean:.4f}")
    print(f"  Probe-estimated (per-prompt mean probe >= 0.5 vote): {est_acc_probe_thresh:.4f}")
    print(f"  Error |est - true|: probe-mean {abs(est_acc_probe_mean - true_acc):.4f}, probe-vote {abs(est_acc_probe_thresh - true_acc):.4f}")
    # Per-rollout proxy
    proxy_acc_per_rollout = float(np.nanmean(scores))
    print(f"  Per-rollout probe-mean (pool all rollouts equally): {proxy_acc_per_rollout:.4f}")
    out.append("=== (D) Probe-as-eval-proxy ===")
    out.append(f"  True dataset acc: {true_acc:.4f}")
    out.append(f"  Probe-mean estimate (per-prompt mean): {est_acc_probe_mean:.4f} (diff {est_acc_probe_mean-true_acc:+.4f})")
    out.append(f"  Probe-vote estimate (per-prompt mean >= 0.5): {est_acc_probe_thresh:.4f} (diff {est_acc_probe_thresh-true_acc:+.4f})")

    # (E) Failure-mode diagnostic
    print("\n=== (E) Probe failure-mode diagnostic ===")
    valid_idx = ~np.isnan(scores)
    sc = scores[valid_idx]
    lab = y_new[valid_idx]
    metas = [meta[i] for i in range(len(meta)) if valid_idx[i]]
    nb = np.array([n_blocks[(int(m["prompt_idx"]), int(m["resp_idx"]))] for m in metas])
    rl = np.array([rollout_len[(int(m["prompt_idx"]), int(m["resp_idx"]))] for m in metas])

    classes = {
        "TP (probe>=0.5, correct)": (sc >= 0.5) & (lab == 1),
        "TN (probe<0.5, wrong)":    (sc <  0.5) & (lab == 0),
        "FP (probe>=0.5, wrong) - overconfidence": (sc >= 0.5) & (lab == 0),
        "FN (probe<0.5, correct) - underconfidence": (sc <  0.5) & (lab == 1),
    }
    print(f"  n_rollouts (with valid probe): {len(sc)}")
    print(f"  {'class':<46} {'n':>6}  {'mean_blocks':>11}  {'mean_len':>9}  {'mean_probe':>10}")
    for name, m in classes.items():
        n = int(m.sum())
        if n == 0: continue
        print(f"  {name:<46} {n:>6}  {nb[m].mean():>11.2f}  {rl[m].mean():>9.0f}  {sc[m].mean():>10.3f}")
        out.append(f"  {name}: n={n}, mean_blocks={nb[m].mean():.2f}, mean_len={rl[m].mean():.0f}, mean_probe={sc[m].mean():.3f}")

    # Overconfidence rate
    fp_rate = float(((sc >= 0.5) & (lab == 0)).sum() / max(1, (sc >= 0.5).sum()))
    fn_rate = float(((sc < 0.5) & (lab == 1)).sum() / max(1, (sc < 0.5).sum()))
    print(f"\n  Overconfidence rate (P(wrong | probe>=0.5)): {fp_rate:.3f}")
    print(f"  Underconfidence rate (P(correct | probe<0.5)): {fn_rate:.3f}")
    out.append(f"\n  Overconfidence rate (P(wrong | probe>=0.5)): {fp_rate:.3f}")
    out.append(f"  Underconfidence rate (P(correct | probe<0.5)): {fn_rate:.3f}")

    # Figure: 2-panel
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=160)
    ax = axes[0]
    # Scatter true vs estimated per-prompt
    ax.scatter(per_p_acc_true, per_p_probe_mean, alpha=0.4, s=15, color="#3a6dba")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="y=x")
    ax.set_xlabel("True per-prompt accuracy (verifier)")
    ax.set_ylabel("Mean probe across rollouts")
    ax.set_title(f"Probe-as-eval-proxy: per-prompt acc vs probe-mean\n(n=406; estimate diff = {est_acc_probe_mean-true_acc:+.3f})")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend()
    ax = axes[1]
    names = list(classes.keys()); names_short = ["TP", "TN", "FP (overconf)", "FN (underconf)"]
    counts = [int(m.sum()) for m in classes.values()]
    blocks = [nb[m].mean() if m.sum() else 0 for m in classes.values()]
    colors_c = ["#3a8b2f", "#888888", "#c45252", "#dba24c"]
    bars = ax.bar(names_short, counts, color=colors_c, alpha=0.85)
    for b, c, blk in zip(bars, counts, blocks):
        ax.text(b.get_x() + b.get_width() / 2, c + 50, f"n={c}\nblocks={blk:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("count")
    ax.set_title("Probe failure modes: rollout counts by (probe>=0.5) × (correct)")
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG)
    plt.close(fig)
    print(f"\n  wrote {OUT_FIG}")

    os.makedirs(os.path.dirname(OUT_TXT) or ".", exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
