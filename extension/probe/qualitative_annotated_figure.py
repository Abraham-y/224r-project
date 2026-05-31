"""Annotated qualitative figure: probe-score trajectory across paired rollouts.

For a chosen prompt that has BOTH a correct and a wrong C_outcome rollout
(each with at least one cached confidence-keyword assertion), plot two panels
side-by-side:

  - Left panel: the wrong rollout. Probe scores at each cached assertion token
    are plotted; vertical dashed lines mark every <answer>...</answer> block
    with its equation; the verifier-scored final <answer> is highlighted.
  - Right panel: the correct rollout. Same plot.

The visual story is the H3 signature: at the assertion tokens, the probe's
predictions are similar (or backwards) between correct and wrong rollouts of
the *same problem* — the position-resolved internal representation no longer
discriminates "this rollout will end correct" from "this rollout will end
wrong" even when the model verbalizes confidence.

Output: extension/outputs/figures/fig8_annotated_qualitative.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from extension.probe.qualitative_examples import cv_per_row_probe_scores
from extension.probe.analyze_probes import load_groups


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_CHARS_PER_TOKEN_APPROX = 3.5


def find_answer_blocks(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(1).strip()) for m in _ANSWER_RE.finditer(text)]


def char_to_approx_token(prompt_len: int, char_offset_in_full: int) -> int:
    return int(char_offset_in_full / _CHARS_PER_TOKEN_APPROX)


def pick_best_prompt(meta_path: str, eval_json: str):
    meta = json.load(open(meta_path))
    rows = [json.loads(l) for l in open(eval_json)]
    counts = Counter((m["prompt_idx"], m["resp_idx"]) for m in meta)
    best = None
    for p_idx, row in enumerate(rows):
        pairs = []
        for r_idx, (resp, score) in enumerate(zip(row["response"], row["scores"])):
            n_ass = counts.get((p_idx, r_idx), 0)
            if n_ass >= 1:
                pairs.append((r_idx, score, n_ass, resp.count("<answer>")))
        correct = [p for p in pairs if p[1] == 1.0]
        wrong = [p for p in pairs if p[1] != 1.0]
        if correct and wrong:
            wrong_best = max(wrong, key=lambda x: (x[2], x[3]))
            correct_best = max(correct, key=lambda x: (x[2], x[3]))
            score = wrong_best[2] + correct_best[2] + wrong_best[3] * 0.05
            if best is None or score > best[0]:
                best = (score, p_idx, correct_best[0], wrong_best[0], row)
    if best is None:
        return None
    _, p_idx, c_r_idx, w_r_idx, row = best
    return p_idx, c_r_idx, w_r_idx, row


def collect_points(meta, scores_cv, p_idx, r_idx):
    out = []
    for i, m in enumerate(meta):
        if (int(m["prompt_idx"]) == p_idx and int(m["resp_idx"]) == r_idx
                and not np.isnan(scores_cv[i])):
            out.append((int(m["tok_idx"]), float(scores_cv[i]), m.get("keyword", "?")))
    out.sort()
    return out


def plot_panel(ax, prompt_len: int, response: str, points: list[tuple[int, float, str]],
                label: str, score: float, target: int, nums: list[int]) -> None:
    """Plot a single rollout panel."""
    color_map = {
        "Perfect": "#c45252", "this works": "#dba24c", "got it": "#2f6db5",
        "the answer is": "#3a8b2f", "verified": "#7d4ca2",
    }

    blocks = find_answer_blocks(response)
    last_block_idx = len(blocks) - 1 if blocks else -1

    # x-axis range: approx token positions over the response.
    response_token_end = char_to_approx_token(prompt_len, prompt_len + len(response))
    response_token_start = char_to_approx_token(prompt_len, prompt_len)
    ax.set_xlim(response_token_start - 10, response_token_end + 10)

    # Background shading: light grey over the post-answer ramble region (after the FIRST <answer>).
    if blocks:
        first_block_tok = char_to_approx_token(prompt_len, prompt_len + blocks[0][0])
        ax.axvspan(first_block_tok, response_token_end + 10, color="#eeeeee", alpha=0.55, zorder=0,
                   label="post-first-answer ramble")

    # Plot <answer> block markers (thin, no labels — they cluster). Highlight only the SCORED one.
    for k, (s, _e, eq) in enumerate(blocks):
        tok_x = char_to_approx_token(prompt_len, prompt_len + s)
        if k == last_block_idx:
            color = "#3a8b2f" if score == 1.0 else "#c45252"
            ax.axvline(tok_x, color=color, linestyle="-", linewidth=2.0, alpha=0.9, zorder=2)
            ax.text(tok_x, 0.02, f"  scored: <answer>{eq}</answer>",
                    fontsize=9, va="bottom", ha="left", color=color, fontweight="bold",
                    transform=ax.get_xaxis_transform())
        else:
            ax.axvline(tok_x, color="#aaaaaa", linestyle="--", linewidth=0.5, alpha=0.55, zorder=1)
    # First answer only — single label
    if blocks:
        first_s, _e, first_eq = blocks[0]
        tok_x = char_to_approx_token(prompt_len, prompt_len + first_s)
        ax.text(tok_x, 0.10, f"  first <answer>{first_eq}</answer>",
                fontsize=8, va="bottom", ha="left", color="#444",
                transform=ax.get_xaxis_transform())

    # Plot probe-at-assertion-token points.
    if points:
        for x, yv, kw in points:
            ax.scatter(x, yv, s=120, color=color_map.get(kw, "#888"),
                       edgecolors="white", linewidths=1.5, zorder=4)
        # Annotate ONLY the values, stacked to avoid overlap.
        for i, (x, yv, kw) in enumerate(points):
            ax.annotate(f"probe={yv:.2f}", xy=(x, yv), xytext=(8, 12 if i % 2 == 0 else -18),
                        textcoords="offset points", ha="left",
                        fontsize=9, color=color_map.get(kw, "#888"), fontweight="bold")
        # Summary text in upper-left of the panel
        mean_probe = float(np.mean([p[1] for p in points]))
        ax.text(0.02, 0.18, f"mean probe at\n'this works' tokens: {mean_probe:.2f}",
                transform=ax.transAxes, fontsize=10, color="#dba24c", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#dba24c", linewidth=1))

    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)

    outcome = "CORRECT (score=1.0)" if score == 1.0 else f"WRONG (score={score})"
    ax.set_title(f"{label} rollout — {outcome}\n"
                 f"{len(blocks)} <answer> blocks, {len(points)} cached assertion-token probes",
                 fontsize=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="extension/cache/probe_cache")
    parser.add_argument("--eval_json", default="eval.json")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--out", default="extension/outputs/figures/fig8_annotated_qualitative.png")
    args = parser.parse_args()

    ass_npz = os.path.join(args.cache_dir, f"C_outcome_l{args.layer}_assertion.npz")
    ass_meta = ass_npz.replace(".npz", ".meta.json")
    with np.load(ass_npz) as data:
        X = data["X"]; y = data["y"]
    meta = json.load(open(ass_meta))
    groups = load_groups(ass_meta)
    scores_cv = cv_per_row_probe_scores(X, y, groups)

    picked = pick_best_prompt(ass_meta, args.eval_json)
    if picked is None:
        raise SystemExit("no prompt with both correct and wrong cached assertions found")
    p_idx, c_r_idx, w_r_idx, row = picked
    target = int(row["target"])
    nums = list(row["nums"])
    prompt_len = len(row["prompt"])

    c_resp = row["response"][c_r_idx]
    w_resp = row["response"][w_r_idx]
    c_score = row["scores"][c_r_idx]
    w_score = row["scores"][w_r_idx]

    c_points = collect_points(meta, scores_cv, p_idx, c_r_idx)
    w_points = collect_points(meta, scores_cv, p_idx, w_r_idx)

    print(f"prompt {p_idx}: target={target}, nums={nums}")
    print(f"  CORRECT rollout (resp_idx={c_r_idx}): {len(c_points)} assertions, "
          f"{c_resp.count('<answer>')} <answer> blocks")
    for tok_idx, sc, kw in c_points:
        print(f"    tok {tok_idx:>4}: probe(correct)={sc:.3f}  '{kw}'")
    print(f"  WRONG   rollout (resp_idx={w_r_idx}): {len(w_points)} assertions, "
          f"{w_resp.count('<answer>')} <answer> blocks")
    for tok_idx, sc, kw in w_points:
        print(f"    tok {tok_idx:>4}: probe(correct)={sc:.3f}  '{kw}'")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160, sharey=True)
    plot_panel(axes[0], prompt_len, w_resp, w_points, "WRONG", w_score, target, nums)
    plot_panel(axes[1], prompt_len, c_resp, c_points, "CORRECT", c_score, target, nums)
    axes[0].set_ylabel("probe(correct) at confidence-asserting token")
    axes[0].set_xlabel("approx. token position in (prompt + response)")
    axes[1].set_xlabel("approx. token position in (prompt + response)")

    fig.suptitle(
        f"C_outcome (L{args.layer}) probe scores at confidence-asserting tokens, "
        f"same problem (target={target}, nums={nums})\n"
        f"dashed lines mark <answer> blocks; the verifier-scored final answer is "
        f"highlighted (green=correct / red=wrong)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
