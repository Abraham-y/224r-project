"""Probe-guided budgeted restart sampling — applied result.

Simulation of the deployment strategy: generate one rollout at a time;
probe at </think> after each; if probe >= T, accept and return; else
re-sample. Stop when probe-confident or budget B exhausted.

Unlike probe-best-of-K (which always uses K=16 samples and picks best),
budgeted restart accepts the FIRST sample crossing threshold, and only
generates more if needed. Saves compute in expectation.

Reports accuracy and average-rollouts-used per prompt for a sweep of
(B, T) — produces an accuracy-vs-compute Pareto curve.

Uses corrected-label trace-final probe scores from existing pre_answer
cache. Local-only, no Modal.
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
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

CACHE = "extension/cache/probe_cache_n500_clean406/C_outcome_l16_pre_answer.npz"
META = CACHE.replace(".npz", ".meta.json")
EVAL = "eval_c_outcome_n500.json"
OUT_TXT = "extension/outputs/n500/text/32_probe_guided_restart.txt"
OUT_FIG = "extension/outputs/n500/figures/fig18_probe_guided_restart.png"

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
    print("[restart] loading pre_answer cache + meta", flush=True)
    with np.load(CACHE) as d:
        X = d["X"]
    meta = json.load(open(META))
    groups = np.array([int(m["prompt_idx"]) for m in meta])

    # Corrected labels: first-<answer>-block correctness for each rollout
    print("[restart] computing corrected labels (first <answer> block correctness)", flush=True)
    rows = [json.loads(l) for l in open(EVAL) if l.strip()]
    first_block_correct = {}  # (p, r) -> bool
    for p_idx, row in enumerate(rows):
        target = int(row["target"]); nums = list(row["nums"])
        for r_idx, resp in enumerate(row["response"]):
            m = _ANSWER_OPEN_RE.search(resp)
            if m is None:
                first_block_correct[(p_idx, r_idx)] = bool(row["scores"][r_idx] == 1.0)
            else:
                first_block_correct[(p_idx, r_idx)] = check_block(m.group(1), target, nums)

    y_new = np.array([int(first_block_correct.get((int(m["prompt_idx"]), int(m["resp_idx"])), 0)) for m in meta], dtype=np.int32)
    print(f"[restart] pos% after relabel: {y_new.mean():.3f}")

    # Held-out probe scores via GroupKFold(5)
    print("[restart] training held-out probe with corrected labels", flush=True)
    scores = np.full(len(y_new), np.nan)
    for tr, te in GroupKFold(5).split(X, y_new, groups):
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
        pipe.fit(X[tr], y_new[tr])
        scores[te] = pipe.predict_proba(X[te])[:, 1]

    # Sanity AUROC on balanced subsample
    pos = np.where(y_new == 1)[0]; neg = np.where(y_new == 0)[0]
    nb = min(len(pos), len(neg))
    rng = np.random.RandomState(0)
    idx = np.concatenate([rng.choice(pos, nb, replace=False), rng.choice(neg, nb, replace=False)])
    bal_auc = float(roc_auc_score(y_new[idx], scores[idx]))
    print(f"[restart] held-out balanced AUROC (corrected labels) = {bal_auc:.3f}")

    # Group by prompt: list of (resp_idx, probe_score, label)
    # Each prompt has up to K=16 rollouts (in original sample order = resp_idx order)
    by_prompt: dict[int, list] = defaultdict(list)
    for i, m in enumerate(meta):
        p = int(m["prompt_idx"]); r = int(m["resp_idx"])
        if np.isnan(scores[i]): continue
        # We use the FIRST-<answer>-CORRECTNESS as the "what would have been the verifier score
        # if we'd stopped here" reward signal — i.e. the model's first commit's truth value.
        label_first = int(first_block_correct.get((p, r), 0))
        by_prompt[p].append((r, float(scores[i]), label_first))
    for p in by_prompt:
        by_prompt[p].sort()
    prompts = sorted(by_prompt.keys())
    print(f"[restart] {len(prompts)} prompts; avg rollouts per prompt: {np.mean([len(by_prompt[p]) for p in prompts]):.1f}")

    # === Strategies ===
    # Compare:
    # (1) pass@1 = just take rollout 1's correctness
    # (2) best-of-K = take rollout with highest probe (all K samples)
    # (3) probe-restart(B, T) = iterate rollouts 1..B; if probe >= T, accept; else continue;
    #     after B, return best-by-probe seen so far
    # Report accuracy and avg-rollouts-used per prompt

    def run_strategy_pass1():
        acc = 0; nu = 0
        for p in prompts:
            v = by_prompt[p]
            if not v: continue
            acc += v[0][2]; nu += 1
        return acc / nu, 1.0

    def run_strategy_best_of_K(K):
        acc = 0; nu = 0; n_used_avg = 0
        for p in prompts:
            v = by_prompt[p][:K]
            if not v: continue
            best = max(v, key=lambda t: t[1])
            acc += best[2]; nu += 1; n_used_avg += len(v)
        return acc / nu, n_used_avg / nu

    def run_strategy_restart(B, T):
        acc = 0; nu = 0; n_used_avg = 0
        for p in prompts:
            v = by_prompt[p][:B]
            if not v: continue
            picked = None
            n_used = 0
            best_so_far = None
            for r, sc, lbl in v:
                n_used += 1
                if best_so_far is None or sc > best_so_far[1]:
                    best_so_far = (lbl, sc)
                if sc >= T:
                    picked = lbl
                    break
            if picked is None:
                picked = best_so_far[0]
            acc += picked
            nu += 1
            n_used_avg += n_used
        return acc / nu, n_used_avg / nu

    # Baselines
    p1_acc, p1_used = run_strategy_pass1()
    print(f"\n=== Probe-guided budgeted restart (corrected labels, first-<answer> correctness as truth) ===")
    print(f"  pass@1 (=resp 0)                          : acc={p1_acc:.4f}  n_used={p1_used:.2f}")

    Ks = [2, 4, 8, 16]
    print(f"\n  Best-of-K (always uses K rollouts):")
    for K in Ks:
        a, u = run_strategy_best_of_K(K)
        print(f"  K={K:>2}                                       : acc={a:.4f}  n_used={u:.2f}")

    # Restart sweep
    print(f"\n  Probe-guided restart (B=budget, T=threshold; returns first ≥T, else best-by-probe):")
    print(f"  {'B':>3}  {'T':>4}  {'acc':>7}  {'n_used':>7}")
    results = []
    for B in (2, 4, 8, 16):
        for T in (0.30, 0.50, 0.70, 0.85, 0.95):
            a, u = run_strategy_restart(B, T)
            print(f"  {B:>3}  {T:>4.2f}  {a:>7.4f}  {u:>7.2f}")
            results.append({"B": B, "T": float(T), "acc": float(a), "n_used": float(u)})

    # Summary: best result.
    # NOTE: this maximises over the (B, T) grid on the very prompts it reports,
    # so it is a fitted parameter presented as a test score. The held-out version
    # below picks (B, T) on prompts the reported ones are not in.
    best = max(results, key=lambda r: r["acc"])
    print(f"\nBest (IN-SAMPLE): B={best['B']}, T={best['T']:.2f}  "
          f"-> acc={best['acc']:.4f}, n_used={best['n_used']:.2f}")

    # ---- Held-out (B, T) selection -----------------------------------------
    fold_of = {p: i % 2 for i, p in enumerate(prompts)}
    grid = [(B, T) for B in (2, 4, 8, 16) for T in (0.30, 0.50, 0.70, 0.85, 0.95)]

    def restart_on(subset, B, T):
        acc = 0; nu = 0; used = 0
        for p in subset:
            v = by_prompt[p][:B]
            if not v:
                continue
            picked = None; best_so_far = None; n_used = 0
            for _r, sc, lbl in v:
                n_used += 1
                if best_so_far is None or sc > best_so_far[1]:
                    best_so_far = (lbl, sc)
                if sc >= T:
                    picked = lbl
                    break
            acc += best_so_far[0] if picked is None else picked
            nu += 1; used += n_used
        return (acc / nu, used / nu) if nu else (float("nan"), float("nan"))

    pooled_acc = pooled_used = pooled_n = 0
    picks = {}
    for held in (0, 1):
        sel = [p for p in prompts if fold_of[p] != held]
        ev = [p for p in prompts if fold_of[p] == held]
        if not sel or not ev:
            continue
        B_star, T_star = max(grid, key=lambda bt: restart_on(sel, *bt)[0])
        picks[held] = {"B": B_star, "T": T_star}
        a, u = restart_on(ev, B_star, T_star)
        pooled_acc += a * len(ev); pooled_used += u * len(ev); pooled_n += len(ev)
    ho_acc = pooled_acc / pooled_n if pooled_n else float("nan")
    ho_used = pooled_used / pooled_n if pooled_n else float("nan")
    print(f"Best (HELD-OUT, 2-fold by prompt): picks={picks}")
    print(f"  acc={ho_acc:.4f}, n_used={ho_used:.2f}  (n_prompts={pooled_n})  "
          f"<-- report THIS, not the in-sample max")

    # ORACLE majority reference: "are >50% of the K rollouts first-block correct?".
    # This consumes the answer key, so it is NOT a deployable selector and NOT
    # self-consistency (which votes on the answer string). Reference line only.
    by_prompt_majority = {}
    for p in prompts:
        v = by_prompt[p]
        labs = [t[2] for t in v]
        by_prompt_majority[p] = int(sum(labs) > len(labs) / 2)
    oracle_maj_acc = float(np.mean(list(by_prompt_majority.values())))
    print(f"\nORACLE majority-of-K (needs answer key; not a deployable baseline): acc = {oracle_maj_acc:.4f}")

    # Save
    out_lines = [
        f"Probe-guided budgeted restart on 0.5B C_outcome clean-406",
        f"  Truth: first-<answer>-block correctness per rollout",
        f"  Held-out balanced AUROC of </think> probe with corrected labels: {bal_auc:.3f}",
        "",
        f"BASELINES:",
        f"  pass@1: {p1_acc:.4f}",
        f"  ORACLE majority-of-K (needs answer key): {oracle_maj_acc:.4f}",
    ]
    for K in Ks:
        a, u = run_strategy_best_of_K(K)
        out_lines.append(f"  best-of-{K}: acc={a:.4f}, n_used={u:.2f}")
    out_lines += ["", "RESTART SWEEP:"]
    for r in results:
        out_lines.append(f"  B={r['B']:>2}, T={r['T']:.2f}: acc={r['acc']:.4f}, n_used={r['n_used']:.2f}")
    out_lines.append("")
    out_lines.append(f"BEST (IN-SAMPLE, (B,T) maximised on these same prompts -- optimistic):")
    out_lines.append(f"  B={best['B']}, T={best['T']:.2f}: acc={best['acc']:.4f}, n_used={best['n_used']:.2f}")
    out_lines.append(f"BEST (HELD-OUT, 2-fold by prompt) -- report this:")
    out_lines.append(f"  picks per fold: {picks}")
    out_lines.append(f"  acc={ho_acc:.4f}, n_used={ho_used:.2f} (n_prompts={pooled_n})")
    os.makedirs(os.path.dirname(OUT_TXT) or ".", exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\nwrote {OUT_TXT}")

    # Figure: accuracy vs n_used
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    # best-of-K points
    bok_pts = [(K, *run_strategy_best_of_K(K)) for K in Ks]
    ax.plot([p[2] for p in bok_pts], [p[1] for p in bok_pts], marker="s", color="#3a8b2f", lw=2, markersize=10,
            label="best-of-K (always uses K rollouts)")
    for K, a, u in bok_pts:
        ax.annotate(f"K={K}", (u, a), xytext=(5, 5), textcoords="offset points", fontsize=8, color="#3a8b2f")
    # Restart points
    color_map = {0.30: "#dba24c", 0.50: "#c45252", 0.70: "#7d4ca2", 0.85: "#3a6dba", 0.95: "#000000"}
    for T in (0.30, 0.50, 0.70, 0.85, 0.95):
        rows_T = [r for r in results if r["T"] == T]
        rows_T.sort(key=lambda r: r["B"])
        xs = [r["n_used"] for r in rows_T]
        ys = [r["acc"] for r in rows_T]
        ax.plot(xs, ys, marker="o", lw=1.5, alpha=0.7, color=color_map[T], label=f"restart T={T}")
    ax.scatter([p1_used], [p1_acc], color="red", marker="*", s=120, zorder=5, label=f"pass@1 = {p1_acc:.3f}")
    ax.scatter([16], [oracle_maj_acc], color="grey", marker="^", s=90, zorder=5, label=f"ORACLE majority of 16 = {oracle_maj_acc:.3f}")
    ax.set_xlabel("average rollouts used per prompt (compute proxy)")
    ax.set_ylabel("accuracy (first-<answer> correctness)")
    ax.set_title("Probe-guided budgeted restart: accuracy vs compute (0.5B C_outcome clean-406)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG) or ".", exist_ok=True)
    fig.savefig(OUT_FIG)
    plt.close(fig)
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
