"""Quantify the accuracy + token cost of the rambling reward-hack, and how it
grows over RLOO training.

Outcome-only RLOO trains the 0.5B model to emit MANY `<answer>` blocks and let
the verifier's "last-block-scored" rule pick the winner (the rambling reward
hack). This script asks: is that rambling *functional* (does the model rescue
itself from wrong to right) or *net-negative* (does it talk itself out of
correct answers)? And how many tokens does the rambling tail waste?

It does this with NO new generation and NO GPU -- it re-scores the EXISTING
rollouts (committed at repo root) with the same verifier, scoring the FIRST
`<answer>` block vs the LAST (= the verifier default = what was actually
scored), and counts the tokens after the first `</answer>`.

Outputs (per checkpoint):
  - first-block pass@1, last-block pass@1, Δ = first - last
  - drift (correct->wrong, b) vs rescue (wrong->correct, c) counts
  - exact McNemar two-sided p on the discordant pairs (b, c)
  - mean `<answer>` blocks/rollout, % multi-answer
  - token-waste %: generated tokens after the first `</answer>`, / total

Two views:
  - HEADLINE: C_SFT vs C_outcome on the full clean-406 set.
  - DYNAMICS: C_SFT / step30 / step60 / step90 / C_outcome on the common
    clean-∩-first-200 problem set (the snapshots are n=200), so the
    over-training trend is apples-to-apples.

Run locally:  python extension/probe/reward_hack_cost.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import evaluation.countdown as cd
from evaluation.countdown import compute_score

# Silence compute_score's stochastic debug prints (random 1/64 chance per call).
cd.random.randint = lambda a, b: 2  # type: ignore

_ANSWER_PAIR_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_ANSWER_OPEN_RE = re.compile(r"<answer>")
_ANSWER_CLOSE = "</answer>"


def first_answer_solution(response: str) -> str | None:
    """Return a string containing ONLY the first `<answer>...</answer>` span,
    so compute_score (which extracts the last match) scores the first block."""
    m = _ANSWER_PAIR_RE.search(response)
    if not m:
        return None
    return f"<answer>{m.group(1)}</answer>"


def n_answer_blocks(response: str) -> int:
    return len(_ANSWER_OPEN_RE.findall(response))


def token_waste(response: str, tokenizer) -> tuple[int, int] | None:
    """(tokens_after_first_</answer>, total_response_tokens) or None if no
    complete answer / tokenizer disabled."""
    if tokenizer is None:
        return None
    close_idx = response.find(_ANSWER_CLOSE)
    if close_idx < 0:
        return None
    end_char = close_idx + len(_ANSWER_CLOSE)
    enc = tokenizer(response, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    total = len(offsets)
    after = sum(1 for (s, _e) in offsets if s >= end_char)
    return after, total


def process_file(path: str, allowed: set[int], tokenizer) -> dict:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    first_correct: list[int] = []
    last_correct: list[int] = []
    blocks: list[int] = []
    tok_after = 0
    tok_total = 0
    n_with_answer = 0

    for p_idx, row in enumerate(rows):
        if p_idx not in allowed:
            continue
        gt = {"target": int(row["target"]), "numbers": list(row["nums"])}
        for response in row["response"]:
            last = int(float(compute_score(response, gt)) == 1.0)
            fa = first_answer_solution(response)
            first = int(float(compute_score(fa, gt)) == 1.0) if fa else 0
            first_correct.append(first)
            last_correct.append(last)
            blocks.append(n_answer_blocks(response))
            tw = token_waste(response, tokenizer)
            if tw is not None:
                tok_after += tw[0]
                tok_total += tw[1]
                n_with_answer += 1

    fc = np.array(first_correct); lc = np.array(last_correct); bl = np.array(blocks)
    n = len(fc)
    b = int(((fc == 1) & (lc == 0)).sum())   # drift: correct first -> wrong last
    c = int(((fc == 0) & (lc == 1)).sum())   # rescue: wrong first -> correct last

    from scipy.stats import binomtest
    mcnemar_p = float(binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) > 0 else 1.0

    return {
        "n_rollouts": int(n),
        "first_acc": float(fc.mean()) if n else float("nan"),
        "last_acc": float(lc.mean()) if n else float("nan"),
        "delta_first_minus_last": float(fc.mean() - lc.mean()) if n else float("nan"),
        "drift_correct_to_wrong": b,
        "rescue_wrong_to_correct": c,
        "mcnemar_p": mcnemar_p,
        "mean_blocks": float(bl.mean()) if n else float("nan"),
        "pct_multi_answer": float((bl >= 2).mean()) if n else float("nan"),
        "token_waste_pct": (tok_after / tok_total) if tok_total else None,
        "n_with_answer": n_with_answer,
    }


def fmt_row(name: str, s: dict) -> str:
    tw = "n/a" if s["token_waste_pct"] is None else f"{s['token_waste_pct']*100:5.1f}%"
    return (f"{name:<14} n={s['n_rollouts']:>5} "
            f"first={s['first_acc']:.3f} last={s['last_acc']:.3f} "
            f"Δ={s['delta_first_minus_last']:+.3f} "
            f"drift={s['drift_correct_to_wrong']:>4} rescue={s['rescue_wrong_to_correct']:>4} "
            f"McNemar_p={s['mcnemar_p']:.2e} "
            f"blocks={s['mean_blocks']:5.2f} multi={s['pct_multi_answer']*100:4.0f}% "
            f"tok_waste={tw}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sft", default="eval_c_sft_n500.json")
    ap.add_argument("--outcome", default="eval_c_outcome_n500.json")
    ap.add_argument("--snap30", default="eval_c_outcome_step_30_n200.json")
    ap.add_argument("--snap60", default="eval_c_outcome_step_60_n200.json")
    ap.add_argument("--snap90", default="eval_c_outcome_step_90_n200.json")
    ap.add_argument("--contam_json", default="extension/data/contaminated_prompt_idx.json")
    ap.add_argument("--tokenizer", default="asingh15/qwen-sft-countdown-defaultproj")
    ap.add_argument("--no_tokens", action="store_true", help="Skip token-waste counting.")
    ap.add_argument("--out_dir", default="extension/outputs/n500")
    args = ap.parse_args()

    try:  # Windows consoles default to cp1252; our output has Δ / ∩.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    clean = set(json.load(open(args.contam_json))["clean"])
    clean_full = clean                              # all clean-406
    clean_200 = {i for i in clean if i < 200}       # common set for the snapshots

    tokenizer = None
    if not args.no_tokens:
        from transformers import AutoTokenizer
        print(f"[reward-hack] loading tokenizer {args.tokenizer} (CPU)...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    print("\n=== HEADLINE: full clean-406 ===", flush=True)
    head = {}
    for name, path in [("C_SFT", args.sft), ("C_outcome", args.outcome)]:
        head[name] = process_file(path, clean_full, tokenizer)
        print(fmt_row(name, head[name]), flush=True)

    d_first = head["C_outcome"]["first_acc"] - head["C_SFT"]["first_acc"]
    print(f"\n  C_outcome: committing to the FIRST answer would change pass@1 by "
          f"{head['C_outcome']['delta_first_minus_last']*100:+.1f} pp "
          f"(first {head['C_outcome']['first_acc']*100:.1f}% vs last "
          f"{head['C_outcome']['last_acc']*100:.1f}%), McNemar p={head['C_outcome']['mcnemar_p']:.2e}.",
          flush=True)
    if head["C_outcome"]["token_waste_pct"] is not None:
        print(f"  post-first-answer rambling tail = "
              f"{head['C_outcome']['token_waste_pct']*100:.1f}% of generated tokens.", flush=True)

    print("\n=== DYNAMICS: common clean-∩-first-200 across RLOO steps ===", flush=True)
    dyn_specs = [("C_SFT(step0)", args.sft), ("step30", args.snap30),
                 ("step60", args.snap60), ("step90", args.snap90),
                 ("C_outcome", args.outcome)]
    dyn = {}
    for name, path in dyn_specs:
        if not os.path.exists(path):
            print(f"  (skip {name}: {path} missing)", flush=True)
            continue
        dyn[name] = process_file(path, clean_200, tokenizer)
        print(fmt_row(name, dyn[name]), flush=True)

    # Save JSON + figure.
    os.makedirs(os.path.join(args.out_dir, "text"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
    out_json = os.path.join(args.out_dir, "text", "35_reward_hack_cost.json")
    with open(out_json, "w") as f:
        json.dump({"headline_clean406": head, "dynamics_clean200": dyn}, f, indent=2)
    print(f"\n[reward-hack] wrote {out_json}", flush=True)

    if dyn:
        _plot(dyn, os.path.join(args.out_dir, "figures", "fig25_reward_hack_cost.png"))


def _plot(dyn: dict, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [k for k in ("C_SFT(step0)", "step30", "step60", "step90", "C_outcome") if k in dyn]
    x = list(range(len(order)))
    first = [dyn[k]["first_acc"] for k in order]
    last = [dyn[k]["last_acc"] for k in order]
    gap = [dyn[k]["delta_first_minus_last"] for k in order]
    blocks = [dyn[k]["mean_blocks"] for k in order]
    waste = [(dyn[k]["token_waste_pct"] or 0) * 100 for k in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(x, first, "o-", color="tab:green", label="first-answer pass@1")
    ax1.plot(x, last, "s-", color="tab:red", label="last-answer pass@1 (scored)")
    ax1.fill_between(x, last, first, color="tab:orange", alpha=0.2,
                     label="reward-hack accuracy tax")
    ax1.set_xticks(x); ax1.set_xticklabels(order, rotation=20)
    ax1.set_ylabel("pass@1"); ax1.set_title("Accuracy: first vs last answer over RLOO")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    ax2b = ax2.twinx()
    l1, = ax2.plot(x, blocks, "o-", color="tab:blue", label="mean <answer> blocks")
    l2, = ax2b.plot(x, waste, "^--", color="tab:purple", label="% tokens wasted (rambling tail)")
    ax2.set_xticks(x); ax2.set_xticklabels(order, rotation=20)
    ax2.set_ylabel("mean <answer> blocks/rollout", color="tab:blue")
    ax2b.set_ylabel("% generated tokens after first answer", color="tab:purple")
    ax2.set_title("Rambling grows over RLOO")
    ax2.legend(handles=[l1, l2], fontsize=9, loc="upper left"); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[reward-hack] wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
