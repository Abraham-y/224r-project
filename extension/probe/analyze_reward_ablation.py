"""Analyze the reward-design ablation (outcome vs first-answer vs probe reward).

Reads sampled rollout JSONs produced by `sample_local` on the procedural set:
    <input_dir>/<arm>_step_<N>.json   (snapshots, n=200)
    <input_dir>/<arm>_latest.json     (final, n=500)
where <arm> in {outcome_v2, firstanswer_v2, probe_reward_v2}.

For each checkpoint, on the clean-406 subset present in that file, computes:
    - last-block pass@1   (the real verifier; what compute_score scores)
    - first-block pass@1  (the model's post-<think> commit)
    - mean <answer> blocks/rollout, % multi-answer  (rambling)
    - % generated tokens after the first </answer>   (token waste)

Then overlays each arm's TRUE accuracy trajectory against arm C's probe
`reward_mean` from W&B -> the Goodhart panel (probe reward up, true acc?).

CPU only. Run after downloading the JSONs:
    python extension/probe/analyze_reward_ablation.py --input_dir extension/outputs/ablation_eval
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_PROBE_DIR = os.path.join(_REPO_ROOT, "extension", "probe")
if _PROBE_DIR not in sys.path:
    sys.path.insert(0, _PROBE_DIR)

import evaluation.countdown as cd
from evaluation.countdown import compute_score
cd.random.randint = lambda a, b: 2  # silence stochastic debug prints

# Reuse the exact helpers from the reward-hack-cost experiment.
from reward_hack_cost import first_answer_solution, n_answer_blocks, token_waste

ARMS = ["outcome_v2", "firstanswer_v2", "probe_reward_v2"]
ARM_LABEL = {"outcome_v2": "outcome (A)", "firstanswer_v2": "first-answer (B)",
             "probe_reward_v2": "probe reward (C)"}
_FNAME_RE = re.compile(r"(?P<arm>[a-z_]+_v2)_(?P<tag>step_\d+|latest)\.json$")


def step_of(tag: str) -> int:
    return 100 if tag == "latest" else int(tag.split("_")[1])


def analyze_file(path: str, clean: set, tokenizer) -> dict:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    first, last, blocks = [], [], []
    tok_after = tok_total = n_ans = 0
    for p_idx, row in enumerate(rows):
        if p_idx not in clean:
            continue
        gt = {"target": int(row["target"]), "numbers": list(row["nums"])}
        for resp in row["response"]:
            last.append(int(float(compute_score(resp, gt)) == 1.0))
            fa = first_answer_solution(resp)
            first.append(int(float(compute_score(fa, gt)) == 1.0) if fa else 0)
            blocks.append(n_answer_blocks(resp))
            tw = token_waste(resp, tokenizer)
            if tw is not None:
                tok_after += tw[0]; tok_total += tw[1]; n_ans += 1
    n = len(last)
    bl = np.array(blocks) if blocks else np.array([0])
    return {
        "n_rollouts": n,
        "last_acc": float(np.mean(last)) if n else float("nan"),
        "first_acc": float(np.mean(first)) if n else float("nan"),
        "mean_blocks": float(bl.mean()),
        "pct_multi": float((bl >= 2).mean()),
        "token_waste_pct": (tok_after / tok_total) if tok_total else None,
    }


def fetch_probe_reward_mean() -> dict:
    """{arm: {step: reward_mean}} from W&B (best-effort)."""
    out = {}
    try:
        import wandb
        api = wandb.Api(); ent = api.default_entity
        for arm in ARMS:
            rs = [r for r in api.runs(f"{ent}/rloo_reward_ablation") if r.name == arm]
            if not rs:
                continue
            series = {}
            for h in rs[0].scan_history(keys=["_step", "sampling/reward_mean"]):
                v = h.get("sampling/reward_mean")
                if v is not None:
                    series[int(h["_step"])] = float(v)
            out[arm] = series
    except Exception as e:
        print(f"[ablation] W&B fetch failed ({type(e).__name__}); skipping reward overlay")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_dir", default="extension/outputs/ablation_eval")
    ap.add_argument("--contam_json", default="extension/data/contaminated_prompt_idx.json")
    ap.add_argument("--tokenizer", default="asingh15/qwen-sft-countdown-defaultproj")
    ap.add_argument("--no_tokens", action="store_true")
    ap.add_argument("--out_dir", default="extension/outputs/n500")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    clean = set(json.load(open(args.contam_json))["clean"])
    tokenizer = None
    if not args.no_tokens:
        from transformers import AutoTokenizer
        print(f"[ablation] loading tokenizer {args.tokenizer}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*_v2_*.json")))
    print(f"[ablation] {len(files)} files in {args.input_dir}", flush=True)
    # results[arm][step] = metrics
    results: dict[str, dict[int, dict]] = {a: {} for a in ARMS}
    for f in files:
        m = _FNAME_RE.search(os.path.basename(f))
        if not m:
            continue
        arm, tag = m.group("arm"), m.group("tag")
        if arm not in results:
            continue
        s = analyze_file(f, clean, tokenizer)
        results[arm][step_of(tag)] = s
        print(f"  {arm:16} step {step_of(tag):3} | last={s['last_acc']:.3f} "
              f"first={s['first_acc']:.3f} blocks={s['mean_blocks']:.2f} "
              f"multi={s['pct_multi']*100:.0f}% "
              f"tok_waste={'n/a' if s['token_waste_pct'] is None else format(s['token_waste_pct']*100,'.1f')+'%'} "
              f"(n={s['n_rollouts']})", flush=True)

    rmean = fetch_probe_reward_mean()

    # Final-checkpoint comparison (step 100 = latest).
    print("\n=== FINAL (latest) — true verifier accuracy across arms ===", flush=True)
    for arm in ARMS:
        s = results[arm].get(100)
        if s:
            pr = rmean.get(arm, {})
            last_rm = pr.get(max(pr)) if pr else None
            print(f"  {ARM_LABEL[arm]:18} last-blk={s['last_acc']:.3f} "
                  f"first-blk={s['first_acc']:.3f} blocks={s['mean_blocks']:.2f} "
                  f"tok_waste={'n/a' if s['token_waste_pct'] is None else format(s['token_waste_pct']*100,'.0f')+'%'}"
                  + (f"  | probe reward_mean(final)={last_rm:.3f}" if last_rm is not None else ""),
                  flush=True)

    os.makedirs(os.path.join(args.out_dir, "text"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
    out_json = os.path.join(args.out_dir, "text", "43_reward_ablation.json")
    with open(out_json, "w") as f:
        json.dump({"results": {a: {str(k): v for k, v in d.items()} for a, d in results.items()},
                   "probe_reward_mean": rmean}, f, indent=2)
    print(f"\n[ablation] wrote {out_json}", flush=True)
    _plot(results, rmean, os.path.join(args.out_dir, "figures", "fig26_reward_ablation.png"))


def _plot(results, rmean, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"outcome_v2": "tab:blue", "firstanswer_v2": "tab:green",
              "probe_reward_v2": "tab:red"}
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: true (last-block) verifier accuracy vs step
    for arm in ARMS:
        d = results[arm]
        if not d:
            continue
        xs = sorted(d)
        ax1.plot(xs, [d[s]["last_acc"] for s in xs], "o-", color=colors[arm],
                 label=ARM_LABEL[arm])
    ax1.set_xlabel("RLOO step"); ax1.set_ylabel("true verifier pass@1 (last-block)")
    ax1.set_title("True accuracy over training"); ax1.legend(); ax1.grid(alpha=0.3)

    # Panel 2: Goodhart — arm C probe reward_mean vs true accuracy
    arm = "probe_reward_v2"
    d = results[arm]
    if d:
        xs = sorted(d)
        ax2.plot(xs, [d[s]["last_acc"] for s in xs], "o-", color="tab:red",
                 label="true verifier pass@1")
        ax2.plot(xs, [d[s]["first_acc"] for s in xs], "s--", color="tab:orange",
                 label="first-block pass@1")
    pr = rmean.get(arm, {})
    if pr:
        prx = sorted(pr)
        ax2.plot(prx, [pr[s] for s in prx], color="black", lw=1.5, alpha=0.7,
                 label="probe reward_mean (optimized)")
    ax2.set_xlabel("RLOO step"); ax2.set_ylabel("score")
    ax2.set_title("Arm C: probe reward vs TRUE accuracy (Goodhart)")
    ax2.legend(); ax2.grid(alpha=0.3); ax2.set_ylim(0, 1.02)

    # Panel 3: rambling (blocks/rollout) vs step
    for arm in ARMS:
        d = results[arm]
        if not d:
            continue
        xs = sorted(d)
        ax3.plot(xs, [d[s]["mean_blocks"] for s in xs], "o-", color=colors[arm],
                 label=ARM_LABEL[arm])
    ax3.set_xlabel("RLOO step"); ax3.set_ylabel("mean <answer> blocks/rollout")
    ax3.set_title("Rambling over training"); ax3.legend(); ax3.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[ablation] wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
