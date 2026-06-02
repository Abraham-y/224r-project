"""Three additional creative uses of the probe.

(A) Probe-variance as problem difficulty signal.
    For each prompt, compute Var[probe across K rollouts]. Test correlation
    with per-prompt accuracy (verifier rate over K) and with per-prompt
    "is this problem hard" labels. If variance > X identifies "uncertain"
    problems and mean predicts difficulty, we have a free problem-difficulty
    estimator from internal hidden states.

(B) Cross-checkpoint applied probe transfer.
    Train probe on C_SFT activations + C_SFT labels. Deploy on C_outcome
    cached rollouts as the selector. Does the C_SFT-trained probe still
    give a best-of-K lift? Tests whether the probe is checkpoint-specific
    or whether the correctness representation is shared.

(C) Multi-position probe ensemble for selection.
    Train probes at pre_answer, assertion, neutral separately on
    C_outcome. For each rollout, get all three scores. Test combinations:
      (1) pre_answer alone (baseline; matches §18)
      (2) pre_answer + assertion (mean)
      (3) pre_answer * assertion (product / 'AND' confidence)
      (4) max(pre, ass)
    Does multi-position aggregation help, or is pre_answer already
    saturated?

Local-only, no Modal.
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
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

CACHE_DIR = "extension/cache/probe_cache_n500_clean406"
EVAL = "eval_c_outcome_n500.json"
EVAL_SFT = "eval_c_sft_n500.json"
OUT_TXT = "extension/outputs/n500/text/35_probe_creative_extensions.txt"
OUT_FIG = "extension/outputs/n500/figures/fig21_probe_variance_difficulty.png"
OUT_FIG2 = "extension/outputs/n500/figures/fig22_multi_position_ensemble.png"

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


def first_block_labels(eval_path):
    """Map (p_idx, r_idx) -> int(first-<answer>-block correct)."""
    rows = [json.loads(l) for l in open(eval_path) if l.strip()]
    labs = {}
    for p_idx, row in enumerate(rows):
        target = int(row["target"]); nums = list(row["nums"])
        for r_idx, resp in enumerate(row["response"]):
            m = _ANSWER_OPEN_RE.search(resp)
            if m is None: labs[(p_idx, r_idx)] = 0
            else: labs[(p_idx, r_idx)] = int(check_block(m.group(1), target, nums))
    return labs


def held_out_probe_scores(X, y, groups):
    scores = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
        pipe.fit(X[tr], y[tr])
        scores[te] = pipe.predict_proba(X[te])[:, 1]
    return scores


def load_cache(name):
    cache = f"{CACHE_DIR}/{name}.npz"
    meta = f"{CACHE_DIR}/{name}.meta.json"
    with np.load(cache) as d: X = d["X"]
    m = json.load(open(meta))
    g = np.array([int(x["prompt_idx"]) for x in m], dtype=np.int32)
    return X, m, g


def main():
    out = []
    print("[creative] loading C_outcome caches + corrected labels", flush=True)
    labs_outcome = first_block_labels(EVAL)
    labs_sft = first_block_labels(EVAL_SFT)

    # ===========================================================
    # (A) Probe-variance as problem difficulty signal
    # ===========================================================
    print("\n=== (A) Probe-variance as problem difficulty ===")
    X, meta, groups = load_cache("C_outcome_l16_pre_answer")
    y_new = np.array([labs_outcome.get((int(m["prompt_idx"]), int(m["resp_idx"])), 0) for m in meta], dtype=np.int32)
    sc = held_out_probe_scores(X, y_new, groups)

    by_p = defaultdict(list)
    for i, m in enumerate(meta):
        if np.isnan(sc[i]): continue
        p = int(m["prompt_idx"]); r = int(m["resp_idx"])
        by_p[p].append((sc[i], y_new[i]))

    rows_A = []
    for p, v in by_p.items():
        if len(v) < 3: continue
        probes = np.array([t[0] for t in v]); labs = np.array([t[1] for t in v])
        rows_A.append({
            "p": p,
            "n": len(v),
            "mean_probe": float(probes.mean()),
            "std_probe": float(probes.std()),
            "acc": float(labs.mean()),  # fraction of rollouts first-block correct
        })
    print(f"  n_prompts: {len(rows_A)}   K per prompt mean: {np.mean([r['n'] for r in rows_A]):.1f}")
    mean_probes = np.array([r["mean_probe"] for r in rows_A])
    std_probes = np.array([r["std_probe"] for r in rows_A])
    accs = np.array([r["acc"] for r in rows_A])
    r_mean, p_mean = pearsonr(mean_probes, accs)
    r_std, p_std = pearsonr(std_probes, accs)
    rs_mean, ps_mean = spearmanr(mean_probes, accs)
    rs_std, ps_std = spearmanr(std_probes, accs)
    print(f"  mean_probe ~ accuracy: Pearson r={r_mean:+.3f} (p={p_mean:.2e}), Spearman r={rs_mean:+.3f}")
    print(f"  std_probe  ~ accuracy: Pearson r={r_std:+.3f} (p={p_std:.2e}), Spearman r={rs_std:+.3f}")
    # Binned: split prompts into quartiles by std_probe; report acc per bin
    q = np.percentile(std_probes, [25, 50, 75])
    bins = np.digitize(std_probes, q)
    for b in range(4):
        m = bins == b
        if m.sum() == 0: continue
        print(f"  std_probe quartile {b}: n={m.sum():>3}  mean_acc={accs[m].mean():.3f}  mean_mean_probe={mean_probes[m].mean():.3f}")
    out.append("=== (A) Probe-variance as problem difficulty ===")
    out.append(f"  mean_probe ~ accuracy: Pearson r={r_mean:+.3f} (p={p_mean:.2e})")
    out.append(f"  std_probe  ~ accuracy: Pearson r={r_std:+.3f} (p={p_std:.2e})")

    # Figure: scatter mean_probe vs accuracy, colored by std_probe
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=160)
    ax = axes[0]
    sc1 = ax.scatter(mean_probes, accs, c=std_probes, s=25, alpha=0.7, cmap="viridis")
    fig.colorbar(sc1, ax=ax, label="std_probe (across K rollouts)")
    ax.set_xlabel("mean probe at </think> (across K rollouts)")
    ax.set_ylabel("fraction of rollouts first-block correct")
    ax.set_title(f"Per-prompt: mean probe vs accuracy (Pearson r={r_mean:.3f})")
    ax.grid(True, ls="--", alpha=0.3)
    ax = axes[1]
    ax.scatter(std_probes, accs, alpha=0.6, color="#3a6dba", s=20)
    ax.set_xlabel("std probe at </think> (across K rollouts)")
    ax.set_ylabel("fraction of rollouts first-block correct")
    ax.set_title(f"Per-prompt: std probe vs accuracy (Pearson r={r_std:.3f})")
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG)
    plt.close(fig)
    print(f"  wrote {OUT_FIG}")

    # ===========================================================
    # (B) Cross-checkpoint applied probe transfer
    # ===========================================================
    print("\n=== (B) Cross-checkpoint applied probe transfer ===")
    # Train probe on C_SFT activations + C_SFT first-block labels
    Xs, meta_s, groups_s = load_cache("C_SFT_l16_pre_answer")
    y_sft = np.array([labs_sft.get((int(m["prompt_idx"]), int(m["resp_idx"])), 0) for m in meta_s], dtype=np.int32)
    # Train on FULL C_SFT data
    pipe_sft = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
    pipe_sft.fit(Xs, y_sft)
    # Deploy on C_outcome activations
    scores_sft_on_outcome = pipe_sft.predict_proba(X)[:, 1]
    # AUROC of this transfer
    pos = np.where(y_new == 1)[0]; neg = np.where(y_new == 0)[0]
    nb = min(len(pos), len(neg))
    rng = np.random.RandomState(0)
    if nb > 0:
        idx = np.concatenate([rng.choice(pos, nb, replace=False), rng.choice(neg, nb, replace=False)])
        bal_auc_xfer = float(roc_auc_score(y_new[idx], scores_sft_on_outcome[idx]))
    else:
        bal_auc_xfer = float("nan")
    print(f"  C_SFT-trained probe held-out balanced AUROC on C_outcome (corrected labels): {bal_auc_xfer:.3f}")

    # Now: deploy as a SELECTOR
    by_p_xfer = defaultdict(list)
    for i, m in enumerate(meta):
        p = int(m["prompt_idx"]); r = int(m["resp_idx"])
        lab = int(labs_outcome.get((p, r), 0))
        by_p_xfer[p].append((r, float(scores_sft_on_outcome[i]), lab))
    for p in by_p_xfer: by_p_xfer[p].sort()
    prompts_xfer = sorted(by_p_xfer.keys())
    pass1 = float(np.mean([by_p_xfer[p][0][2] for p in prompts_xfer if by_p_xfer[p]]))
    bok_xfer = float(np.mean([max(by_p_xfer[p][:16], key=lambda t: t[1])[2] for p in prompts_xfer if by_p_xfer[p]]))
    print(f"  pass@1: {pass1:.4f}")
    print(f"  probe-best-of-16 (C_SFT probe, deployed on C_outcome): {bok_xfer:.4f}  (+{100*(bok_xfer-pass1):+.1f} pp)")
    print(f"  vs in-distribution C_outcome probe best-of-16 (from §18): 0.6700  (+12.1 pp)")
    out.append("\n=== (B) Cross-checkpoint applied probe transfer ===")
    out.append(f"  C_SFT-trained probe AUROC deployed on C_outcome: {bal_auc_xfer:.3f}")
    out.append(f"  best-of-16 with C_SFT probe: {bok_xfer:.4f}")

    # ===========================================================
    # (C) Multi-position probe ensemble for selection
    # ===========================================================
    print("\n=== (C) Multi-position probe ensemble ===")
    # Need: per-rollout probe scores at pre_answer, assertion, neutral.
    # pre_answer has 1 per rollout (the </think> token)
    # assertion has many per rollout (we'll aggregate by mean)
    # neutral has many per rollout (mean)
    X_pre, meta_pre, groups_pre = X, meta, groups  # already loaded
    sc_pre = sc  # already computed
    X_ass, meta_ass, groups_ass = load_cache("C_outcome_l16_assertion")
    y_ass = np.array([labs_outcome.get((int(m["prompt_idx"]), int(m["resp_idx"])), 0) for m in meta_ass], dtype=np.int32)
    sc_ass = held_out_probe_scores(X_ass, y_ass, groups_ass)
    X_neu, meta_neu, groups_neu = load_cache("C_outcome_l16_neutral")
    y_neu = np.array([labs_outcome.get((int(m["prompt_idx"]), int(m["resp_idx"])), 0) for m in meta_neu], dtype=np.int32)
    sc_neu = held_out_probe_scores(X_neu, y_neu, groups_neu)

    # Aggregate assertion + neutral by mean per (p, r)
    ass_by_pr = defaultdict(list); neu_by_pr = defaultdict(list)
    for i, m in enumerate(meta_ass):
        p = int(m["prompt_idx"]); r = int(m["resp_idx"])
        if not np.isnan(sc_ass[i]): ass_by_pr[(p, r)].append(sc_ass[i])
    for i, m in enumerate(meta_neu):
        p = int(m["prompt_idx"]); r = int(m["resp_idx"])
        if not np.isnan(sc_neu[i]): neu_by_pr[(p, r)].append(sc_neu[i])
    ass_score = {k: float(np.mean(v)) for k, v in ass_by_pr.items()}
    neu_score = {k: float(np.mean(v)) for k, v in neu_by_pr.items()}

    # Build per-rollout combined scores for those rollouts with all three
    multi = defaultdict(list)  # p -> list of (r, pre, ass, neu, lab)
    for i, m in enumerate(meta_pre):
        p = int(m["prompt_idx"]); r = int(m["resp_idx"])
        if np.isnan(sc_pre[i]): continue
        a = ass_score.get((p, r))
        n = neu_score.get((p, r))
        if a is None or n is None: continue
        lab = int(labs_outcome.get((p, r), 0))
        multi[p].append((r, float(sc_pre[i]), a, n, lab))
    for p in multi: multi[p].sort()
    prompts_m = sorted(multi.keys())
    n_m = len(prompts_m)
    print(f"  n_prompts with all three probes: {n_m}")

    def best_of_K_via(score_fn, prompts, K=16):
        acc = 0; nu = 0
        for p in prompts:
            v = multi[p][:K]
            if not v: continue
            best = max(v, key=lambda t: score_fn(t))
            acc += best[4]; nu += 1
        return acc / nu

    # Strategies
    strategies = {
        "pre_answer alone":        lambda t: t[1],
        "assertion alone":         lambda t: t[2],
        "neutral alone":           lambda t: t[3],
        "mean(pre, ass)":          lambda t: 0.5 * (t[1] + t[2]),
        "mean(pre, ass, neu)":     lambda t: (t[1] + t[2] + t[3]) / 3.0,
        "product(pre, ass)":       lambda t: t[1] * t[2],
        "max(pre, ass)":           lambda t: max(t[1], t[2]),
        "min(pre, ass)":           lambda t: min(t[1], t[2]),
        "0.7*pre + 0.3*ass":       lambda t: 0.7 * t[1] + 0.3 * t[2],
    }

    results_C = {}
    pass1_m = float(np.mean([multi[p][0][4] for p in prompts_m if multi[p]]))
    print(f"  pass@1 baseline: {pass1_m:.4f}")
    for name, fn in strategies.items():
        a = best_of_K_via(fn, prompts_m, 16)
        results_C[name] = a
        print(f"  best-of-16 by {name:<25}: {a:.4f}  (+{100*(a-pass1_m):+.1f} pp)")

    out.append("\n=== (C) Multi-position probe ensemble for selection ===")
    out.append(f"  n_prompts: {n_m}")
    out.append(f"  pass@1: {pass1_m:.4f}")
    for name, v in results_C.items():
        out.append(f"  best-of-16 by {name}: {v:.4f}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=160)
    names = list(results_C.keys())
    vals = [results_C[n] for n in names]
    colors = ["#3a6dba" if "pre_answer alone" in n else "#999999" if "alone" in n else "#3a8b2f" for n in names]
    bars = ax.barh(names, vals, color=colors, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(v + 0.003, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    ax.axvline(pass1_m, color="red", linestyle="--", lw=1.0, label=f"pass@1 = {pass1_m:.3f}")
    ax.set_xlabel("accuracy (first-<answer> correctness)")
    ax.set_title(f"Multi-position probe ensemble for best-of-16 selection (0.5B C_outcome, n={n_m})")
    ax.set_xlim(0.5, 0.75)
    ax.legend()
    ax.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG2)
    plt.close(fig)
    print(f"  wrote {OUT_FIG2}")

    # Save
    os.makedirs(os.path.dirname(OUT_TXT) or ".", exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
