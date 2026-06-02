"""Downstream analysis on post-Goodhart probe-RL checkpoints.

Compares to vanilla C_outcome + firstanswer + C_SFT:
  - mean blocks per rollout (rambling rate)
  - first-block accuracy
  - last-block accuracy
  - mean response length
"""
from __future__ import annotations
import json, re
from collections import Counter
import numpy as np

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


def analyze(eval_path, label, clean_idx=None):
    if not __import__('os').path.exists(eval_path):
        print(f"  [{label}] file not found: {eval_path}")
        return None
    rows = [json.loads(l) for l in open(eval_path) if l.strip()]
    block_counts = []
    first_correct = 0; last_correct = 0
    n_zero = 0; n_one = 0; n_multi = 0
    lengths = []
    n_rollouts = 0
    for p_idx, row in enumerate(rows):
        if clean_idx is not None and p_idx not in clean_idx: continue
        target = int(row["target"]); nums = list(row["nums"])
        for r_idx, resp in enumerate(row["response"]):
            n_rollouts += 1
            lengths.append(len(resp))
            matches = list(_ANSWER_OPEN_RE.finditer(resp))
            nb = len(matches)
            block_counts.append(nb)
            if nb == 0: n_zero += 1
            elif nb == 1: n_one += 1
            else: n_multi += 1
            if matches:
                first_correct += int(check_block(matches[0].group(1), target, nums))
                last_correct += int(check_block(matches[-1].group(1), target, nums))
    block_counts = np.array(block_counts)
    return {
        "label": label,
        "n_rollouts": n_rollouts,
        "mean_blocks": float(block_counts.mean()),
        "pct_zero": 100 * n_zero / n_rollouts,
        "pct_one": 100 * n_one / n_rollouts,
        "pct_multi": 100 * n_multi / n_rollouts,
        "first_acc": first_correct / n_rollouts,
        "last_acc": last_correct / n_rollouts,
        "mean_len": float(np.mean(lengths)),
    }


def main():
    clean = set(json.load(open("extension/data/contaminated_prompt_idx.json"))["clean"])
    results = []
    for path, label in [
        ("eval_c_sft_n500.json", "C_SFT (init, no RL)"),
        ("eval_c_outcome_n500.json", "vanilla C_outcome (verifier RLOO)"),
        ("eval_c_firstanswer_n500.json", "firstanswer C_outcome' (first-block verifier RLOO)"),
        ("eval_runA_postRL_n500.json", "probe-RL runA (C_outcome init + temp1 probe)"),
        ("eval_runB_postRL_n500.json", "probe-RL runB (C_SFT init + temp1 probe)"),
    ]:
        r = analyze(path, label, clean_idx=clean)
        if r is None: continue
        results.append(r)
        print(f"\n=== {label} (n_rollouts={r['n_rollouts']}) ===")
        print(f"  mean blocks/rollout: {r['mean_blocks']:.3f}")
        print(f"  multi-answer rate: {r['pct_multi']:.1f}%   one-block: {r['pct_one']:.1f}%   zero-block: {r['pct_zero']:.1f}%")
        print(f"  first-block accuracy: {r['first_acc']:.4f}")
        print(f"  last-block accuracy: {r['last_acc']:.4f}")
        print(f"  mean response length: {r['mean_len']:.0f}")

    print("\n=== SUMMARY TABLE ===")
    print(f"{'checkpoint':<55} {'blocks':>8} {'multi%':>8} {'first_acc':>10} {'last_acc':>10} {'len':>8}")
    for r in results:
        print(f"  {r['label']:<53} {r['mean_blocks']:>8.2f} {r['pct_multi']:>8.1f} {r['first_acc']:>10.4f} {r['last_acc']:>10.4f} {r['mean_len']:>8.0f}")

    import os
    os.makedirs("extension/outputs/n500/text", exist_ok=True)
    with open("extension/outputs/n500/text/50_probe_rl_downstream.txt", "w") as f:
        f.write("PROBE-RL DOWNSTREAM ANALYSIS\n\n")
        for r in results:
            f.write(f"=== {r['label']} (n_rollouts={r['n_rollouts']}) ===\n")
            for k, v in r.items():
                if k == "label": continue
                if isinstance(v, float): f.write(f"  {k}: {v:.4f}\n")
                else: f.write(f"  {k}: {v}\n")
            f.write("\n")
        f.write("\nSUMMARY:\n")
        f.write(f"{'checkpoint':<55} {'blocks':>8} {'multi%':>8} {'first_acc':>10} {'last_acc':>10} {'len':>8}\n")
        for r in results:
            f.write(f"  {r['label']:<53} {r['mean_blocks']:>8.2f} {r['pct_multi']:>8.1f} {r['first_acc']:>10.4f} {r['last_acc']:>10.4f} {r['mean_len']:>8.0f}\n")
    print(f"\nwrote extension/outputs/n500/text/50_probe_rl_downstream.txt")


if __name__ == "__main__":
    main()
