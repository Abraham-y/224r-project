"""Pass@k comparison bar plot: SFT baseline vs RLOO on the Countdown test set.

Reads two eval JSONs produced by `countdown_eval.py` (each row has a `scores`
list of per-sample verifier scores), computes pass@k with the unbiased
estimator, and saves a grouped bar plot.

Usage:
    python evaluation/plot_passk_comparison.py \
        --sft_json eval_sft.json --rloo_json eval_rloo.json \
        --out rloo_pass_k.png
"""

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt


def load_scores(path):
    """Return a list of per-prompt score lists from a (JSONL or JSON-array) file."""
    with open(path) as f:
        text = f.read().strip()
    try:
        data = json.loads(text)
        rows = data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        # HuggingFace Dataset.to_json writes JSON-lines by default.
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [row["scores"] for row in rows]


def pass_at_k(n, c, k):
    """Unbiased pass@k estimator (Chen et al., 2021): n samples, c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def pass_at_k_curve(score_lists):
    """Average pass@k over prompts for k = 1..n_samples. A sample is correct iff score == 1.0."""
    correctness = np.array([[s == 1.0 for s in scores] for scores in score_lists])
    n_problems, n_samples = correctness.shape
    curve = []
    for k in range(1, n_samples + 1):
        per_problem = [pass_at_k(n_samples, int(correctness[i].sum()), k) for i in range(n_problems)]
        curve.append(np.mean(per_problem))
    return np.array(curve)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_json", required=True, help="Eval JSON for the SFT baseline.")
    parser.add_argument("--rloo_json", required=True, help="Eval JSON for the RLOO model.")
    parser.add_argument("--out", default="rloo_pass_k.png")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                        help="Which k values to show as bars.")
    args = parser.parse_args()

    sft_curve = pass_at_k_curve(load_scores(args.sft_json))
    rloo_curve = pass_at_k_curve(load_scores(args.rloo_json))

    # curve index k-1 holds pass@k; clip requested ks to what's available.
    ks = [k for k in args.ks if k <= len(sft_curve) and k <= len(rloo_curve)]
    sft_vals = np.array([sft_curve[k - 1] for k in ks]) * 100
    rloo_vals = np.array([rloo_curve[k - 1] for k in ks]) * 100

    print(f"SFT  pass@1={sft_curve[0] * 100:.1f}%  pass@{len(sft_curve)}={sft_curve[-1] * 100:.1f}%")
    print(f"RLOO pass@1={rloo_curve[0] * 100:.1f}%  pass@{len(rloo_curve)}={rloo_curve[-1] * 100:.1f}%")

    x = np.arange(len(ks))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    b1 = ax.bar(x - width / 2, sft_vals, width, label="SFT Baseline", color="#9bb8d3")
    b2 = ax.bar(x + width / 2, rloo_vals, width, label="RLOO Aligned", color="#2f6db5")

    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xlabel("k")
    ax.set_ylabel("Pass@k (%)")
    ax.set_title("Pass@k on Countdown Test Set: SFT vs RLOO")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
