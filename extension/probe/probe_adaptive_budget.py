"""Probe-guided adaptive test-time budget allocation.

Setup: total rollout budget B across N prompts. Compare strategies:

  (1) UNIFORM: K rollouts per prompt; best-of-K by probe.
  (2) PROBE-ADAPTIVE-WATERFILL: round 1, generate 1 rollout per prompt.
      Score with probe. Sort prompts by probe ascending (least confident
      first). In rounds 2..R, give extra rollouts to the lowest-probe
      prompts until budget exhausted.
  (3) PROBE-THRESHOLD-STOP: round 1, generate 1; if probe >= T_stop,
      commit; else generate more. Already in restart experiment but
      add as comparison.

Question: given a FIXED total budget, does probe-adaptive allocation
beat uniform allocation? In expectation, hard prompts should get
more rollouts and easy ones fewer, raising overall accuracy.

We simulate using the existing cached 16 rollouts per prompt as the
rollout pool. "Generating rollout i for prompt p" means revealing the
i-th cached rollout's probe score and label.

Uses corrected-label trace-final probe held-out scores on 0.5B
C_outcome clean-406.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from collections import defaultdict

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
OUT_TXT = "extension/outputs/n500/text/36_probe_adaptive_budget.txt"
OUT_FIG = "extension/outputs/n500/figures/fig23_probe_adaptive_budget.png"

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
    for p_idx, row in enumerate(rows):
        target = int(row["target"]); nums = list(row["nums"])
        for r_idx, resp in enumerate(row["response"]):
            m = _ANSWER_OPEN_RE.search(resp)
            first_correct[(p_idx, r_idx)] = bool(m) and check_block(m.group(1), target, nums)

    y_new = np.array([int(first_correct.get((int(m["prompt_idx"]), int(m["resp_idx"])), 0)) for m in meta], dtype=np.int32)
    scores = np.full(len(y_new), np.nan)
    for tr, te in GroupKFold(5).split(X, y_new, groups):
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
        pipe.fit(X[tr], y_new[tr])
        scores[te] = pipe.predict_proba(X[te])[:, 1]

    by_p = defaultdict(list)
    for i, m in enumerate(meta):
        if np.isnan(scores[i]): continue
        p = int(m["prompt_idx"]); r = int(m["resp_idx"])
        by_p[p].append((r, float(scores[i]), int(first_correct.get((p, r), 0))))
    for p in by_p: by_p[p].sort()
    prompts = [p for p in sorted(by_p.keys()) if len(by_p[p]) >= 16]
    N = len(prompts)
    print(f"[budget] {N} prompts with full 16 cached rollouts available")

    # ---- UNIFORM strategy: K per prompt, best-of-K by probe
    def uniform_strategy(K):
        total = K * N
        acc = 0
        for p in prompts:
            v = by_p[p][:K]
            best = max(v, key=lambda t: t[1])
            acc += best[2]
        return acc / N, total

    # ---- PROBE-ADAPTIVE-WATERFILL
    # Round 1: 1 rollout per prompt.
    # Round t>1: pick the prompt with lowest current-best probe; give it one more rollout.
    # Stop when total budget hit. Final answer = best-by-probe among assigned rollouts.
    def adaptive_strategy(total_budget):
        # Per-prompt: list of (r, sc, lab) assigned so far
        assigned = {p: [by_p[p][0]] for p in prompts}  # round 1
        used = N
        # Heap-ish: track current best-probe per prompt
        # Then give one more to the lowest-best-probe prompt
        while used < total_budget:
            # Find prompt with lowest current best probe and remaining rollouts
            cand_p = None; cand_best = 2.0
            for p in prompts:
                if len(assigned[p]) >= 16: continue
                cur_best = max(t[1] for t in assigned[p])
                if cur_best < cand_best:
                    cand_best = cur_best
                    cand_p = p
            if cand_p is None: break  # all maxed
            next_idx = len(assigned[cand_p])
            assigned[cand_p].append(by_p[cand_p][next_idx])
            used += 1
        # Compute accuracy = best-by-probe
        acc = 0
        for p in prompts:
            v = assigned[p]
            best = max(v, key=lambda t: t[1])
            acc += best[2]
        n_per_prompt = [len(assigned[p]) for p in prompts]
        return acc / N, used, n_per_prompt

    # ---- THRESHOLD-WATERFILL: give more to prompts whose current best probe is below T
    def threshold_strategy(total_budget, T):
        assigned = {p: [by_p[p][0]] for p in prompts}
        used = N
        progress = True
        while used < total_budget and progress:
            progress = False
            # All prompts whose best is below T, in order of current best (ascending)
            cands = [(p, max(t[1] for t in assigned[p])) for p in prompts if len(assigned[p]) < 16 and max(t[1] for t in assigned[p]) < T]
            if not cands: break
            cands.sort(key=lambda x: x[1])
            for p, _ in cands:
                if used >= total_budget: break
                next_idx = len(assigned[p])
                assigned[p].append(by_p[p][next_idx])
                used += 1
                progress = True
        acc = 0
        for p in prompts:
            v = assigned[p]
            best = max(v, key=lambda t: t[1])
            acc += best[2]
        n_per_prompt = [len(assigned[p]) for p in prompts]
        return acc / N, used, n_per_prompt

    # Run sweep over total budget
    budgets = [N, 2 * N, 4 * N, 6 * N, 8 * N, 12 * N, 16 * N]
    out = []
    out.append(f"Probe-adaptive test-time budget allocation, N={N} prompts")
    out.append(f"Total budget = K_avg * N")
    out.append("")
    out.append(f"{'K_avg':>5}  {'budget':>7}  {'uniform':>10}  {'adaptive':>10}  {'thresh@0.5':>11}  {'thresh@0.8':>11}  {'thresh@0.95':>11}")

    sweep_results = []
    for total in budgets:
        K_avg = total / N
        u_acc, _ = uniform_strategy(int(round(K_avg)))
        a_acc, a_used, a_nper = adaptive_strategy(total)
        t05_acc, _, _ = threshold_strategy(total, 0.5)
        t08_acc, _, _ = threshold_strategy(total, 0.8)
        t095_acc, _, _ = threshold_strategy(total, 0.95)
        print(f"K_avg={K_avg:>4.1f}  budget={total:>5}  uniform={u_acc:.4f}  adapt={a_acc:.4f}  T0.5={t05_acc:.4f}  T0.8={t08_acc:.4f}  T0.95={t095_acc:.4f}")
        out.append(f"  {K_avg:>5.1f}  {total:>7d}  {u_acc:>10.4f}  {a_acc:>10.4f}  {t05_acc:>11.4f}  {t08_acc:>11.4f}  {t095_acc:>11.4f}")
        sweep_results.append({"K_avg": K_avg, "budget": total,
                              "uniform": u_acc, "adaptive": a_acc,
                              "t05": t05_acc, "t08": t08_acc, "t095": t095_acc})

    # Distribution of n_per_prompt under adaptive at K_avg=4
    _, _, nper = adaptive_strategy(4 * N)
    print(f"\nAt K_avg=4, adaptive allocation: min={min(nper)}, max={max(nper)}, mean={np.mean(nper):.2f}, n_capped={sum(1 for x in nper if x == 16)}")
    out.append(f"\nAt K_avg=4, adaptive: min={min(nper)}, max={max(nper)}, mean={np.mean(nper):.2f}, n_capped={sum(1 for x in nper if x == 16)}")

    os.makedirs(os.path.dirname(OUT_TXT) or ".", exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nwrote {OUT_TXT}")

    # Figure
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=160)
    K_avgs = [r["K_avg"] for r in sweep_results]
    ax.plot(K_avgs, [r["uniform"] for r in sweep_results], marker="o", lw=2, color="#888", label="uniform K-per-prompt + best-of-K")
    ax.plot(K_avgs, [r["adaptive"] for r in sweep_results], marker="s", lw=2, color="#3a6dba", label="probe-adaptive waterfill (lowest-probe-first)")
    ax.plot(K_avgs, [r["t05"] for r in sweep_results], marker="^", lw=1.5, alpha=0.6, color="#dba24c", label="threshold T=0.5")
    ax.plot(K_avgs, [r["t08"] for r in sweep_results], marker="v", lw=1.5, alpha=0.6, color="#3a8b2f", label="threshold T=0.8")
    ax.plot(K_avgs, [r["t095"] for r in sweep_results], marker="D", lw=1.5, alpha=0.6, color="#c45252", label="threshold T=0.95")
    ax.set_xlabel("average rollouts per prompt (compute proxy)")
    ax.set_ylabel("accuracy (first-<answer>-block correctness)")
    ax.set_title(f"Probe-adaptive test-time budget allocation (0.5B C_outcome clean-406, N={N})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG)
    plt.close(fig)
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
