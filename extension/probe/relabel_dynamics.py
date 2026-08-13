"""Option B dynamics gap trajectory with CORRECTED labels.

Re-runs §2.2 for each snapshot:
  C_SFT (from main probe cache, eval_c_sft_n500.json)
  step_30, step_60, step_90 (from probe_cache_dynamics_optB,
                             eval_c_outcome_step_*_n200.json)
  C_outcome final (from main probe cache)

For each snapshot, retrain pre_answer + assertion probes on the
correctly-labeled data and report (1) AUROC at each position,
(2) gap = pre - ass, (3) mean blocks per rollout (rambling rate).
Then compute Pearson r between rambling rate and gap.
"""

from __future__ import annotations

import json
import os
import re
import warnings

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

CACHE_MAIN = "extension/cache/probe_cache_n500_clean406"
CACHE_DYN = "extension/cache/probe_cache_dynamics_optB"
OUT = "extension/outputs/n500/text/43_relabel_dynamics.txt"
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
    rows = [json.loads(l) for l in open(eval_path) if l.strip()]
    labs = {}; nblocks = {}
    for p, row in enumerate(rows):
        target = int(row["target"]); nums = list(row["nums"])
        for r_i, resp in enumerate(row["response"]):
            ms = list(_ANSWER_OPEN_RE.finditer(resp))
            labs[(p, r_i)] = int(bool(ms) and check_block(ms[0].group(1), target, nums))
            nblocks[(p, r_i)] = len(ms)
    return labs, nblocks


def load_and_score(cache_path, eval_path):
    """Load pre_answer and assertion caches, label by first-<answer>
    correctness, train probes via GroupKFold(5), return AUROCs + gap +
    mean blocks/rollout.

    mean_blocks is computed over the SAME rollouts the AUROC is computed on
    (the ones present in the pre_answer cache), not over every rollout in the
    eval file. The caches for C_SFT/C_outcome are clean-406-filtered while the
    eval JSONs are the full n500, so averaging over the latter mixed two
    different populations into one scatter point.
    """
    labs, nblocks = first_block_labels(eval_path)
    aucs = {}
    cached_keys: set = set()
    for kind in ["pre_answer", "assertion"]:
        cache_file = f"{cache_path}_l16_{kind}"
        with np.load(f"{cache_file}.npz") as d: X = d["X"]
        m = json.load(open(f"{cache_file}.meta.json"))
        groups = np.array([int(x['prompt_idx']) for x in m])
        if kind == "pre_answer":
            cached_keys = {(int(x['prompt_idx']), int(x['resp_idx'])) for x in m}
        y = np.array([labs.get((int(x['prompt_idx']), int(x['resp_idx'])), 0) for x in m], dtype=np.int32)
        sc = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X, y, groups):
            pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
            pipe.fit(X[tr], y[tr])
            sc[te] = pipe.predict_proba(X[te])[:, 1]
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
        nb = min(len(pos), len(neg))
        if nb > 0:
            rng = np.random.RandomState(0)
            idx = np.concatenate([rng.choice(pos, nb, replace=False), rng.choice(neg, nb, replace=False)])
            aucs[kind] = float(roc_auc_score(y[idx], sc[idx]))
        else:
            aucs[kind] = float("nan")
    sel = [v for k, v in nblocks.items() if k in cached_keys] or list(nblocks.values())
    mean_blocks = float(np.mean(sel))
    return aucs["pre_answer"], aucs["assertion"], aucs["pre_answer"] - aucs["assertion"], mean_blocks


def main():
    out_lines = ["Option B dynamics with CORRECTED labels", ""]
    snapshots = [
        ("C_SFT",      f"{CACHE_MAIN}/C_SFT",            "eval_c_sft_n500.json"),
        ("step_30",    f"{CACHE_DYN}/C_outcome_step_30", "eval_c_outcome_step_30_n200.json"),
        ("step_60",    f"{CACHE_DYN}/C_outcome_step_60", "eval_c_outcome_step_60_n200.json"),
        ("step_90",    f"{CACHE_DYN}/C_outcome_step_90", "eval_c_outcome_step_90_n200.json"),
        ("C_outcome",  f"{CACHE_MAIN}/C_outcome",        "eval_c_outcome_n500.json"),
    ]
    results = []
    for label, cache_root, eval_path in snapshots:
        if not os.path.exists(f"{cache_root}_l16_pre_answer.npz"):
            print(f"  {label}: MISSING cache at {cache_root}")
            continue
        print(f"[dynamics] {label} ...")
        pre, ass, gap, mb = load_and_score(cache_root, eval_path)
        print(f"    pre={pre:.3f}  ass={ass:.3f}  gap={gap:+.3f}  mean_blocks={mb:.2f}")
        out_lines.append(f"  {label}: pre={pre:.3f}, ass={ass:.3f}, gap={gap:+.3f}, mean_blocks={mb:.2f}")
        results.append({"label": label, "pre": pre, "ass": ass, "gap": gap, "blocks": mb})

    # Correlation across snapshots.
    #
    # HEALTH WARNING, printed with the number because it is easy to over-read:
    #   * n is the number of SNAPSHOTS (5), not rollouts. A Pearson p-value on
    #     n=5 has essentially no power and an unstable estimate; r=0.89 at n=5
    #     is p<0.05 by a hair and would not survive one point moving.
    #   * The snapshots are not drawn from one population: C_SFT and C_outcome
    #     come from the n500 clean-406 caches, step_30/60/90 from separate
    #     n200 evals over DIFFERENT prompts. So this mixes two problem sets.
    #   * The points are also not independent -- they are successive checkpoints
    #     of one training run, so this is 5 correlated observations of a
    #     monotone-in-time trend, which is close to guaranteed to correlate with
    #     any other monotone-in-time quantity.
    # Treat it as descriptive; do not report the p-value as evidence.
    if len(results) >= 3:
        gaps = np.array([r["gap"] for r in results])
        blocks = np.array([r["blocks"] for r in results])
        r_p, p_p = pearsonr(blocks, gaps)
        # Spearman is the more honest statistic for 5 monotone points -- but
        # scipy's asymptotic p-value is meaningless at n=5 (it returns ~1e-24
        # for rho=1.0, which is nonsense: with 5 points there are only 5! = 120
        # orderings, so the smallest attainable two-sided p is 2/120 = 0.017).
        # Use the exact permutation p-value instead.
        from itertools import permutations
        from scipy.stats import spearmanr
        rho, _p_asymptotic = spearmanr(blocks, gaps)
        if len(blocks) <= 8:
            perms = list(permutations(range(len(blocks))))
            null = [spearmanr(blocks, gaps[list(perm)])[0] for perm in perms]
            p_rho = float(np.mean([abs(r) >= abs(rho) - 1e-12 for r in null]))
            p_kind = f"exact permutation, {len(perms)} orderings"
        else:
            p_rho = float(_p_asymptotic)
            p_kind = "asymptotic"
        warn = ("DESCRIPTIVE ONLY: n = number of snapshots, not rollouts; the "
                "snapshots come from two different eval sets (n500 clean-406 vs "
                "n200) and are successive checkpoints of one run, so they are "
                "neither independent nor identically sampled. Do not quote this "
                "p-value as evidence.")
        out_lines.append("")
        out_lines.append(f"Pearson  r(mean_blocks, gap) across snapshots: r={r_p:+.3f} (p={p_p:.2e}, n={len(results)})")
        out_lines.append(f"Spearman rho(mean_blocks, gap):                rho={rho:+.3f} (p={p_rho:.3f} [{p_kind}], n={len(results)})")
        out_lines.append(f"  !! {warn}")
        print(f"\n  Pearson  r(mean_blocks, gap) = {r_p:+.3f} (p={p_p:.2e}, n={len(results)})")
        print(f"  Spearman rho                 = {rho:+.3f} (p={p_rho:.3f} [{p_kind}])")
        print(f"  !! {warn}")

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
