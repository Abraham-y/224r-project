"""Probe-direction cosine analysis with CORRECTED labels.

Re-runs §2.10. For each (ckpt, position) at L16, train probe with
corrected labels on FULL data (no held-out), extract weight vector
(after applying inverse-scaler: w / scaler.scale_), normalize.
Compute pairwise cosines:
  - within-ckpt cross-position (pre vs ass, pre vs neu, ass vs neu)
  - cross-ckpt within-position (C_SFT pre vs C_outcome pre, etc.)
"""

from __future__ import annotations

import json
import os
import re
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

CACHE_DIR = "extension/cache/probe_cache_n500_clean406"
OUT = "extension/outputs/n500/text/41_relabel_cosines.txt"
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
    labs = {}
    for p, row in enumerate(rows):
        target = int(row["target"]); nums = list(row["nums"])
        for r_i, resp in enumerate(row["response"]):
            m = _ANSWER_OPEN_RE.search(resp)
            labs[(p, r_i)] = int(bool(m) and check_block(m.group(1), target, nums))
    return labs


def load_and_label(name, eval_path, kind):
    p = f"{CACHE_DIR}/{name}.npz"
    with np.load(p) as d: X = d["X"]
    m = json.load(open(f"{CACHE_DIR}/{name}.meta.json"))
    labs_dict = first_block_labels(eval_path)
    if kind == "pre_answer":
        y = np.array([labs_dict[(int(x['prompt_idx']), int(x['resp_idx']))] for x in m], dtype=np.int32)
    else:
        # assertion / neutral: label by first-<answer>-block correctness of the rollout (same)
        # this aligns the "what is the probe predicting" definition consistently across positions
        y = np.array([labs_dict[(int(x['prompt_idx']), int(x['resp_idx']))] for x in m], dtype=np.int32)
    return X, y


def train_full(X, y):
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
    pipe.fit(X, y)
    w = pipe.named_steps["lr"].coef_[0]
    s = pipe.named_steps["sc"].scale_
    # The "real" probe direction in input space is w / s (since sc(x) = (x - mu) / s and
    # the LR computes w · sc(x) = (w / s) · x + const).
    direction = w / s
    return direction / np.linalg.norm(direction), pipe


def main():
    print("[cosines] training corrected-label probes...", flush=True)
    out_lines = ["Probe-direction cosines under CORRECTED labels", ""]
    cells = {}
    for ckpt, eval_path in [("C_SFT", "eval_c_sft_n500.json"), ("C_outcome", "eval_c_outcome_n500.json")]:
        for kind in ["pre_answer", "assertion", "neutral"]:
            X, y = load_and_label(f"{ckpt}_l16_{kind}", eval_path, kind)
            v, pipe = train_full(X, y)
            cells[(ckpt, kind)] = v
            print(f"  {ckpt}/{kind}: norm-unit direction shape {v.shape}, pos rate {y.mean():.3f}")

    out_lines.append("=== Within-checkpoint cross-position cosines ===")
    for ckpt in ["C_SFT", "C_outcome"]:
        out_lines.append(f"  {ckpt}:")
        for a, b in [("pre_answer","assertion"), ("pre_answer","neutral"), ("assertion","neutral")]:
            c = float(np.dot(cells[(ckpt, a)], cells[(ckpt, b)]))
            print(f"  {ckpt}: cos({a}, {b}) = {c:+.4f}")
            out_lines.append(f"    cos({a}, {b}) = {c:+.4f}")

    out_lines.append("\n=== Cross-checkpoint within-position cosines ===")
    for kind in ["pre_answer", "assertion", "neutral"]:
        c = float(np.dot(cells[("C_SFT", kind)], cells[("C_outcome", kind)]))
        print(f"  {kind}: cos(C_SFT, C_outcome) = {c:+.4f}")
        out_lines.append(f"  {kind}: cos(C_SFT, C_outcome) = {c:+.4f}")

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
