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
    mean blocks/rollout."""
    labs, nblocks = first_block_labels(eval_path)
    aucs = {}
    for kind in ["pre_answer", "assertion"]:
        cache_file = f"{cache_path}_l16_{kind}"
        with np.load(f"{cache_file}.npz") as d: X = d["X"]
        m = json.load(open(f"{cache_file}.meta.json"))
        groups = np.array([int(x['prompt_idx']) for x in m])
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
    mean_blocks = float(np.mean(list(nblocks.values())))
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

    # Correlation
    if len(results) >= 3:
        gaps = np.array([r["gap"] for r in results])
        blocks = np.array([r["blocks"] for r in results])
        r_p, p_p = pearsonr(blocks, gaps)
        out_lines.append("")
        out_lines.append(f"Pearson r(mean_blocks, gap) across snapshots: r={r_p:+.3f} (p={p_p:.2e}, n={len(results)})")
        print(f"\n  Pearson r(mean_blocks, gap) = {r_p:+.3f} (p={p_p:.2e})")

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
