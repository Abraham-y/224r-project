"""Analyze the causal-steering JSONL output: accuracy per (alpha, direction)."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="causal_steering_results.jsonl")
    parser.add_argument("--out", default="extension/outputs/n500/text/23_causal_steering.txt")
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    print(f"loaded {len(rows)} rows")

    # Group by (alpha, direction)
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[(r["alpha"], r["direction"])].append(r)

    # Also compute baseline (alpha=0, direction=zero) accuracy
    lines = ["Causal steering at </think>, L16 residual stream (C_outcome, clean-406 subset)",
             f"  input: {args.input}",
             "",
             f"{'condition':<22} {'n':>5} {'patched':>8} {'acc_correct':>12} {'acc_format':>12} {'mean_score':>12}"]
    lines.append("-" * 76)
    # Sort: zero first, then probe + alpha asc, then rand + alpha asc
    keys = sorted(by_cond.keys(), key=lambda k: (0 if k[1] == "zero" else 1 if k[1] == "probe" else 2, k[0]))
    for k in keys:
        rs = by_cond[k]
        alpha, direction = k
        scores = [r["new_score"] for r in rs]
        acc_correct = float(np.mean([s == 1.0 for s in scores]))
        acc_format = float(np.mean([s >= 0.1 for s in scores]))
        mean = float(np.mean(scores))
        n_patched = sum(1 for r in rs if r["patch_applied"])
        lines.append(f"{direction+f' alpha={alpha:+.1f}':<22} {len(rs):>5} {n_patched:>8} {acc_correct:>12.3f} {acc_format:>12.3f} {mean:>12.3f}")

    # Pair-wise difference: probe vs random at each alpha
    lines.append("")
    lines.append("Probe direction vs random direction at matched magnitude:")
    lines.append("(probe alpha k) - (random alpha k) accuracy delta:")
    for alpha in sorted({k[0] for k in by_cond if k[0] != 0.0}):
        if (alpha, "probe") in by_cond and (alpha, "rand") in by_cond:
            p_acc = float(np.mean([r["new_score"] == 1.0 for r in by_cond[(alpha, "probe")]]))
            r_acc = float(np.mean([r["new_score"] == 1.0 for r in by_cond[(alpha, "rand")]]))
            lines.append(f"  alpha={alpha:+.1f}: probe acc {p_acc:.3f}  vs  random acc {r_acc:.3f}  delta = {p_acc - r_acc:+.3f}")

    # Match prefixes paired analysis: for each prefix, compare baseline vs probe vs rand
    lines.append("")
    lines.append("Paired analysis (same prefix, different conditions):")
    by_prefix = defaultdict(dict)
    for r in rows:
        by_prefix[(r["prompt_idx"], r["resp_idx"])][(r["alpha"], r["direction"])] = r
    n_prefix = len(by_prefix)
    lines.append(f"  n prefixes: {n_prefix}")
    # For each alpha, count: probe_correct & baseline_wrong, probe_wrong & baseline_correct
    for alpha in sorted({k[0] for k in by_cond if k[0] != 0.0}):
        gained = 0; lost = 0; tied_correct = 0; tied_wrong = 0
        prefixes_with_all = 0
        for prefix, conds in by_prefix.items():
            base = conds.get((0.0, "zero"))
            probe = conds.get((alpha, "probe"))
            if base is None or probe is None:
                continue
            prefixes_with_all += 1
            b_ok = base["new_score"] == 1.0
            p_ok = probe["new_score"] == 1.0
            if p_ok and not b_ok: gained += 1
            elif b_ok and not p_ok: lost += 1
            elif p_ok and b_ok: tied_correct += 1
            else: tied_wrong += 1
        lines.append(f"  alpha={alpha:+.1f}: probe gained {gained}, lost {lost}, both correct {tied_correct}, both wrong {tied_wrong}  (n={prefixes_with_all})")

    # Same for rand
    lines.append("")
    lines.append("Same analysis for random-direction control:")
    for alpha in sorted({k[0] for k in by_cond if k[0] != 0.0}):
        gained = 0; lost = 0; tied_correct = 0; tied_wrong = 0
        prefixes_with_all = 0
        for prefix, conds in by_prefix.items():
            base = conds.get((0.0, "zero"))
            rand = conds.get((alpha, "rand"))
            if base is None or rand is None:
                continue
            prefixes_with_all += 1
            b_ok = base["new_score"] == 1.0
            r_ok = rand["new_score"] == 1.0
            if r_ok and not b_ok: gained += 1
            elif b_ok and not r_ok: lost += 1
            elif r_ok and b_ok: tied_correct += 1
            else: tied_wrong += 1
        lines.append(f"  alpha={alpha:+.1f}: random gained {gained}, lost {lost}, both correct {tied_correct}, both wrong {tied_wrong}  (n={prefixes_with_all})")

    txt = "\n".join(lines)
    print(txt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
