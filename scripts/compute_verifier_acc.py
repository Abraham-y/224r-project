"""Compute verifier accuracy + mean blocks per checkpoint for the
multiplicative RLOO run.

Reads the eval_probe_mult_step{N}_n500.json files (and eval_probe_mult_final_n500.json)
from repo root after they've been downloaded from Modal volume, then prints a
clean table:

  step | pass@1 (first-block acc) | pass@1 (last-block acc) | mean blocks | mean chars

If the file is missing, that checkpoint shows --- (eval still in flight).

Compares to vanilla C_outcome baseline (eval_c_outcome_FIXEDSTOP_n500.json)
which is the right comparison given multiplicative was init'd from C_SFT and
trained 100 steps -- same scope as vanilla outcome RL from C_SFT to C_outcome.
"""
import json
import os
import re
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.countdown import validate_equation, evaluate_equation


def score_block(eq: str, gt) -> float:
    if not validate_equation(eq, gt["numbers"]):
        return 0.1
    try:
        r = evaluate_equation(eq)
        if r is None:
            return 0.1
        return 1.0 if abs(r - gt["target"]) < 1e-5 else 0.1
    except Exception:
        return 0.1


def metrics_for_file(path: str):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try: rows.append(json.loads(line))
                except: pass
    if not rows:
        return None
    acc_first, acc_last, blocks, lens = [], [], [], []
    for r in rows:
        gt = r.get("ground_truth") or {"target": r["target"], "numbers": r["nums"]}
        for resp in r.get("response", []):
            matches = list(re.finditer(r"<answer>(.*?)</answer>", resp, re.DOTALL))
            blocks.append(len(matches))
            lens.append(len(resp))
            if not matches:
                acc_first.append(0.0); acc_last.append(0.0); continue
            first_eq = matches[0].group(1).strip()
            last_eq = matches[-1].group(1).strip()
            acc_first.append(1.0 if score_block(first_eq, gt) == 1.0 else 0.0)
            acc_last.append(1.0 if score_block(last_eq, gt) == 1.0 else 0.0)
    return {
        "n": len(acc_first),
        "acc_first": float(np.mean(acc_first)) if acc_first else 0,
        "acc_last": float(np.mean(acc_last)) if acc_last else 0,
        "mean_blocks": float(np.mean(blocks)) if blocks else 0,
        "mean_chars": float(np.mean(lens)) if lens else 0,
    }


def main():
    print()
    print(f"{'Checkpoint':<28} {'n':>5} {'first_acc':>10} {'last_acc':>10} {'mean_blk':>9} {'mean_char':>10}")
    print("-" * 80)

    baselines = [
        ("C_SFT (FIXEDSTOP)",      "eval_c_sft_FIXEDSTOP_n500.json"),
        ("C_outcome (FIXEDSTOP)",  "eval_c_outcome_FIXEDSTOP_n500.json"),
    ]
    for label, fname in baselines:
        m = metrics_for_file(fname)
        if m is None:
            print(f"{label:<28} {'(missing)':>5}")
        else:
            print(f"{label:<28} {m['n']:>5} {m['acc_first']:>10.4f} {m['acc_last']:>10.4f} {m['mean_blocks']:>9.2f} {m['mean_chars']:>10.0f}")

    print()
    print("Multiplicative RLOO (probe_mult_csft_run1):")
    for step in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        fname = f"eval_probe_mult_step{step}_n500.json"
        m = metrics_for_file(fname)
        label = f"  step {step}"
        if m is None:
            print(f"{label:<28} {'(eval pending)':>15}")
        else:
            print(f"{label:<28} {m['n']:>5} {m['acc_first']:>10.4f} {m['acc_last']:>10.4f} {m['mean_blocks']:>9.2f} {m['mean_chars']:>10.0f}")

    fname = "eval_probe_mult_final_n500.json"
    m = metrics_for_file(fname)
    label = "  final (latest_ckpt)"
    if m is None:
        print(f"{label:<28} {'(eval pending)':>15}")
    else:
        print(f"{label:<28} {m['n']:>5} {m['acc_first']:>10.4f} {m['acc_last']:>10.4f} {m['mean_blocks']:>9.2f} {m['mean_chars']:>10.0f}")


if __name__ == "__main__":
    main()
