"""Two probe-as-applied experiments — local, free.

(1) Probe-guided abstention: for each problem, take the model's first
    rollout's probe score at </think>. If probe >= T, commit to that
    rollout's first-<answer> correctness. Else abstain (no answer
    counted; doesn't enter accuracy). Plot selective accuracy on
    attempted problems vs coverage as T varies.

    Question: does the probe support a useful selective-prediction
    tradeoff? E.g., can we achieve high accuracy on a subset of
    problems we choose to attempt?

(2) Probe + majority vote ensemble: combine the K rollouts' probe
    scores with the K rollouts' votes (which equation is most common
    in the first <answer>). Test:
      - probe-best-of-K (already done)
      - majority-vote-of-K
      - weighted: per-rollout score = α * probe + (1-α) * vote_weight
      - intersection: take majority equation IFF its probe score > T

    Question: is the probe complementary to self-consistency (majority
    vote)? If their selection criteria are uncorrelated, hybrid
    should beat either individually.

Uses corrected-label trace-final probe on 0.5B C_outcome clean-406.
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
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

CACHE = "extension/cache/probe_cache_n500_clean406/C_outcome_l16_pre_answer.npz"
META = CACHE.replace(".npz", ".meta.json")
EVAL = "eval_c_outcome_n500.json"
OUT_TXT = "extension/outputs/n500/text/33_probe_abstention_hybrid.txt"
OUT_FIG = "extension/outputs/n500/figures/fig19_probe_abstention.png"
OUT_FIG2 = "extension/outputs/n500/figures/fig20_probe_majority_hybrid.png"

_ANSWER_OPEN_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def normalize_equation(eq: str) -> str:
    """Canonical form for majority voting (strip whitespace)."""
    return "".join(eq.strip().split())


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
    print("[applied] loading cache + corrected labels + held-out probe scores", flush=True)
    with np.load(CACHE) as d:
        X = d["X"]
    meta = json.load(open(META))
    groups = np.array([int(m["prompt_idx"]) for m in meta])
    rows = [json.loads(l) for l in open(EVAL) if l.strip()]

    # Corrected labels: first <answer> correctness
    first_correct = {}
    first_eq = {}
    for p_idx, row in enumerate(rows):
        target = int(row["target"]); nums = list(row["nums"])
        for r_idx, resp in enumerate(row["response"]):
            m = _ANSWER_OPEN_RE.search(resp)
            if m is None:
                first_correct[(p_idx, r_idx)] = False
                first_eq[(p_idx, r_idx)] = None
            else:
                first_correct[(p_idx, r_idx)] = check_block(m.group(1), target, nums)
                first_eq[(p_idx, r_idx)] = normalize_equation(m.group(1))

    y_new = np.array([int(first_correct.get((int(m["prompt_idx"]), int(m["resp_idx"])), 0)) for m in meta], dtype=np.int32)
    # Held-out probe scores
    scores = np.full(len(y_new), np.nan)
    for tr, te in GroupKFold(5).split(X, y_new, groups):
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
        pipe.fit(X[tr], y_new[tr])
        scores[te] = pipe.predict_proba(X[te])[:, 1]

    pos = np.where(y_new == 1)[0]; neg = np.where(y_new == 0)[0]
    nb = min(len(pos), len(neg))
    rng = np.random.RandomState(0)
    idx = np.concatenate([rng.choice(pos, nb, replace=False), rng.choice(neg, nb, replace=False)])
    bal_auc = float(roc_auc_score(y_new[idx], scores[idx]))
    print(f"[applied] held-out balanced AUROC (corrected labels): {bal_auc:.3f}")

    # Group by prompt
    by_p = defaultdict(list)
    for i, m in enumerate(meta):
        p = int(m["prompt_idx"]); r = int(m["resp_idx"])
        if np.isnan(scores[i]): continue
        lab = int(first_correct.get((p, r), 0))
        eq = first_eq.get((p, r))
        by_p[p].append((r, float(scores[i]), lab, eq))
    for p in by_p:
        by_p[p].sort()
    prompts = sorted(by_p.keys())
    n_prompts = len(prompts)

    # ===========================================================
    # (1) Probe-guided ABSTENTION
    # ===========================================================
    print("\n=== (1) Probe-guided abstention: selective accuracy vs coverage ===")
    print(f"  {n_prompts} prompts; using first cached rollout per prompt for the abstention decision.")

    # For each prompt, use the FIRST rollout (resp_idx=0)'s probe score and correctness
    first_probe = {}
    first_label = {}
    for p, v in by_p.items():
        first_probe[p] = v[0][1]
        first_label[p] = v[0][2]
    # Sweep threshold
    thresholds = np.linspace(0.0, 1.0, 51)
    abs_rows = []
    for T in thresholds:
        attempted = [p for p in prompts if first_probe[p] >= T]
        if not attempted:
            abs_rows.append({"T": float(T), "coverage": 0.0, "accuracy_on_attempted": float("nan"),
                              "n_attempted": 0})
            continue
        accuracy = float(np.mean([first_label[p] for p in attempted]))
        abs_rows.append({"T": float(T), "coverage": len(attempted) / n_prompts,
                         "accuracy_on_attempted": accuracy, "n_attempted": len(attempted)})
    print(f"\n{'T':>6}  {'coverage':>10}  {'n_attempted':>13}  {'acc_on_attempted':>17}")
    for r in abs_rows[::5]:
        cov = r["coverage"]
        acc = r["accuracy_on_attempted"]
        print(f"  {r['T']:>4.2f}  {cov:>10.3f}  {r['n_attempted']:>13d}  {acc:>17.3f}")
    base_acc = float(np.mean([first_label[p] for p in prompts]))
    print(f"\n  Baseline accuracy (no abstention, all {n_prompts}): {base_acc:.3f}")
    print(f"  Reference points:")
    for tgt_cov in [0.9, 0.7, 0.5, 0.3]:
        # Closest threshold giving coverage near tgt_cov
        close = min(abs_rows, key=lambda r: abs(r["coverage"] - tgt_cov))
        print(f"    coverage ≈ {tgt_cov:.2f}: T={close['T']:.2f}, n_attempted={close['n_attempted']}, "
              f"accuracy={close['accuracy_on_attempted']:.3f}")

    # Figure 1: accuracy vs coverage curve
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=160)
    cov = [r["coverage"] for r in abs_rows]
    acc = [r["accuracy_on_attempted"] for r in abs_rows]
    ax.plot(cov, acc, marker="o", color="#3a6dba", lw=2, markersize=4)
    ax.axhline(base_acc, color="grey", linestyle="--", lw=1.0, label=f"no-abstention accuracy = {base_acc:.3f}")
    ax.set_xlabel("coverage (fraction of problems attempted)")
    ax.set_ylabel("accuracy on attempted problems")
    ax.set_title(f"Probe-guided selective abstention on 0.5B C_outcome clean-406\n"
                 f"(decision: commit if probe at first </think> >= T, else abstain)\n"
                 f"trace-final probe AUROC = {bal_auc:.3f} on corrected labels")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1.02)
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG)
    plt.close(fig)
    print(f"\n  wrote {OUT_FIG}")

    # ===========================================================
    # (2) Probe + majority vote ENSEMBLE
    # ===========================================================
    print("\n=== (2) Probe + majority-vote ensemble ===")
    # For each prompt:
    #   probe-best: take rollout with highest probe; its label is the prediction
    #   majority-vote: take the most common first <answer> equation; check if that equation is correct
    #     (i.e., did the majority vote produce a correct answer)
    #   majority-vote-confidence: like majority-vote, BUT only if the majority block's mean probe > T
    K = 16
    probe_best = []
    majority_acc = []
    intersection_acc = []  # majority vote IFF its mean probe >= T
    # Agreement-gated SELECTIVE selector. Note there is no non-degenerate
    # always-commit combination of {probe-best, majority} gated on agreement:
    # when the two agree they return the same equation, so "agree -> that, else
    # -> probe-best" is identically probe-best, and "agree -> that, else ->
    # majority" is identically majority. (The previous `union_acc` was the
    # latter: both branches appended majority_label, so the reported "union"
    # row was just a duplicate of the majority row.) The one informative
    # version is selective: commit only when they agree, and report coverage.
    agree_labels = []   # labels on the subset where probe-best == majority
    intersection_T = 0.5
    for p in prompts:
        v = by_p[p][:K]
        if not v: continue
        # Probe-best
        best = max(v, key=lambda t: t[1])
        probe_best.append(best[2])
        # Majority vote: most common first-eq, get its correctness from any rollout that produced it
        eq_to_label = {}
        eq_to_probes = defaultdict(list)
        for r, sc, lab, eq in v:
            if eq is None: continue
            eq_to_label[eq] = lab
            eq_to_probes[eq].append(sc)
        if not eq_to_label:
            majority_acc.append(0); intersection_acc.append(0); continue
        counts = Counter()
        for r, sc, lab, eq in v:
            if eq is not None: counts[eq] += 1
        top_eq, top_count = counts.most_common(1)[0]
        majority_label = eq_to_label[top_eq]
        majority_acc.append(majority_label)
        # Intersection: only commit if majority's mean probe >= T
        mean_probe = float(np.mean(eq_to_probes[top_eq]))
        if mean_probe >= intersection_T:
            intersection_acc.append(majority_label)
        else:
            # Fallback to probe-best
            intersection_acc.append(best[2])
        # Agreement-gated selective: only counted when probe-best and majority
        # pick the same equation. Abstain otherwise (so this has coverage < 1).
        if first_eq[(p, best[0])] == top_eq:
            agree_labels.append(majority_label)

    pa = float(np.mean(probe_best))
    ma = float(np.mean(majority_acc))
    ia = float(np.mean(intersection_acc))
    agree_cov = len(agree_labels) / max(1, len(probe_best))
    agree_acc = float(np.mean(agree_labels)) if agree_labels else float("nan")
    print(f"  probe-best-of-{K}                   : acc={pa:.4f}")
    print(f"  majority-of-{K}                     : acc={ma:.4f}")
    print(f"  intersection (majority if mean_probe>=0.5, else probe-best): acc={ia:.4f}")
    print(f"  agree-gated SELECTIVE (commit iff probe-best ≡ majority): "
          f"acc={agree_acc:.4f} at coverage={agree_cov:.3f} (n={len(agree_labels)})")

    # Are probe-best vs majority-best the same? Agreement rate
    same = 0
    for p in prompts:
        v = by_p[p][:K]
        if not v: continue
        best = max(v, key=lambda t: t[1])
        counts = Counter([t[3] for t in v if t[3] is not None])
        if not counts: continue
        top_eq = counts.most_common(1)[0][0]
        if first_eq[(p, best[0])] == top_eq:
            same += 1
    print(f"  agreement rate (probe-best equation == majority equation): {same}/{n_prompts} = {100*same/n_prompts:.1f}%")

    # Compare on the disagreement subset
    print(f"\n  When probe-best and majority DISAGREE:")
    pb_lab = []; mb_lab = []
    for p in prompts:
        v = by_p[p][:K]
        if not v: continue
        best = max(v, key=lambda t: t[1])
        counts = Counter([t[3] for t in v if t[3] is not None])
        if not counts: continue
        top_eq, _ = counts.most_common(1)[0]
        if first_eq[(p, best[0])] != top_eq:
            # disagreement
            pb_lab.append(best[2])
            mb_lab.append(by_p[p][0][2])  # using majority's label proxy (TODO: get from eq_to_label)
            # better: get majority's label
            for r, sc, lab, eq in v:
                if eq == top_eq:
                    mb_lab[-1] = lab
                    break
    if pb_lab:
        print(f"    n_disagreement: {len(pb_lab)}")
        print(f"    probe-best wins: {sum(1 for a, b in zip(pb_lab, mb_lab) if a > b)}")
        print(f"    majority wins:   {sum(1 for a, b in zip(pb_lab, mb_lab) if b > a)}")
        print(f"    both correct:    {sum(1 for a, b in zip(pb_lab, mb_lab) if a == b == 1)}")
        print(f"    both wrong:      {sum(1 for a, b in zip(pb_lab, mb_lab) if a == b == 0)}")

    # Figure 2: bar chart of selector strategies
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=160)
    methods = ["probe-best-of-16", "majority-of-16", "intersection\n(majority if probe>0.5)",
               f"agree-gated\n(SELECTIVE, cov={agree_cov:.2f})"]
    values = [pa, ma, ia, agree_acc]
    colors = ["#3a6dba", "#888888", "#3a8b2f", "#dba24c"]
    bars = ax.bar(methods, values, color=colors, alpha=0.85)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("accuracy (first-<answer> correctness)")
    ax.set_title(f"Probe + majority vote ensemble (0.5B C_outcome clean-406, n={n_prompts})")
    ax.set_ylim(0, 1)
    ax.axhline(base_acc, color="red", linestyle="--", lw=1.0, label=f"pass@1 = {base_acc:.3f}")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG2)
    plt.close(fig)
    print(f"\n  wrote {OUT_FIG2}")

    # Save txt
    out_lines = [
        f"Two probe-applied experiments on 0.5B C_outcome clean-406",
        f"  Held-out balanced AUROC of </think> probe (corrected labels): {bal_auc:.3f}",
        "",
        "=== (1) Probe-guided abstention ===",
        f"  baseline accuracy (no abstention): {base_acc:.3f}",
    ]
    for tgt_cov in [0.9, 0.7, 0.5, 0.3]:
        close = min(abs_rows, key=lambda r: abs(r["coverage"] - tgt_cov))
        out_lines.append(f"  coverage≈{tgt_cov:.2f}: T={close['T']:.2f}, acc={close['accuracy_on_attempted']:.3f}")
    out_lines.append("")
    out_lines.append("=== (2) Probe + majority-vote ensemble ===")
    out_lines.append(f"  probe-best-of-{K}: acc={pa:.4f}")
    out_lines.append(f"  majority-of-{K}: acc={ma:.4f}")
    out_lines.append(f"  intersection (majority if probe>=0.5, else probe-best): acc={ia:.4f}")
    out_lines.append(f"  agree-gated SELECTIVE (commit iff probe-best == majority): "
                     f"acc={agree_acc:.4f} at coverage={agree_cov:.3f} (n={len(agree_labels)})")
    out_lines.append(f"    NOTE: this row is selective (coverage < 1); it is NOT comparable")
    out_lines.append(f"    to the always-commit rows above at face value.")
    out_lines.append(f"  agreement rate: {100*same/n_prompts:.1f}%")
    os.makedirs(os.path.dirname(OUT_TXT) or ".", exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
