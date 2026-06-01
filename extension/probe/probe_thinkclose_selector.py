"""Probe-as-answer-selector using the CORRECTED trace-final probe at every
`</think>` token. Re-runs the §17 selector experiment with the better probe.

Workflow:
  1. Load extension/cache/probe_cache_n500_all_thinkclose/C_outcome_l16_thinkclose.npz
     and meta (each row = one </think> in some rollout; meta has
     prompt_idx, resp_idx, think_close_idx, total_think_closes, tok_idx).
  2. Derive labels for each </think>: correctness of the <answer> block that
     IMMEDIATELY FOLLOWS this </think>.
  3. Train held-out probe via GroupKFold(5) by prompt with corrected labels.
  4. For each multi-thinkclose rollout, score every </think>'s probe, then
     run probe-commit strategies (commit at first </think> with probe >= T,
     etc.). The label of the corresponding <answer> block is the prediction.

Pure local re-analysis. Filters to clean-406 prompts only.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

CACHE = "extension/cache/probe_cache_n500_all_thinkclose/C_outcome_l16_thinkclose.npz"
META = CACHE.replace(".npz", ".meta.json")
CONTAM = "extension/data/contaminated_prompt_idx.json"
EVAL = "eval_c_outcome_n500.json"
PAC = "extension/outputs/n500/per_answer_correctness.jsonl"
OUT_TXT = "extension/outputs/n500/text/30_probe_thinkclose_selector.txt"
OUT_FIG = "extension/outputs/n500/figures/fig16_probe_thinkclose_selector.png"

_ANSWER_OPEN_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def check_block(eq, target, nums):
    eq = eq.strip()
    try:
        nums_in_eq = [int(n) for n in re.findall(r"\d+", eq)]
        if sorted(nums_in_eq) != sorted([int(x) for x in nums]):
            return False
        if not re.match(r"^[\d+\-*/().\s]+$", eq):
            return False
        result = eval(eq, {"__builtins__": None}, {})
        return abs(result - int(target)) < 1e-5
    except Exception:
        return False


def main():
    print("[selector] loading thinkclose cache + meta", flush=True)
    with np.load(CACHE) as d:
        X = d["X"]
    meta = json.load(open(META))
    print(f"[selector] {len(meta)} </think> positions cached")

    # Build rollout info: (p, r) -> list of (char_open, block_correct), scored, prompt_text, response
    print("[selector] parsing rollouts + per-block correctness", flush=True)
    rows = [json.loads(l) for l in open(EVAL) if l.strip()]
    rollout_info = {}
    for p_idx, row in enumerate(rows):
        prompt_text = row["prompt"]
        target = int(row["target"])
        nums = list(row["nums"])
        for r_idx, resp in enumerate(row["response"]):
            blocks = []
            for m in _ANSWER_OPEN_RE.finditer(resp):
                bc = check_block(m.group(1), target, nums)
                blocks.append((len(prompt_text) + m.start(), bc))
            rollout_info[(p_idx, r_idx)] = {
                "blocks": blocks,
                "scored_correct": bool(row["scores"][r_idx] == 1.0),
                "prompt_text": prompt_text,
                "response": resp,
            }

    # Get char positions of </think> tokens (use the tok_idx in meta + re-tokenize)
    print("[selector] mapping </think> tokens to next <answer> block", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("asingh15/qwen-sft-countdown-defaultproj", use_fast=True)

    offset_cache = {}
    y = np.zeros(len(meta), dtype=np.int32)
    next_block_idx = np.full(len(meta), -1, dtype=np.int32)
    for i, m in enumerate(meta):
        p = int(m["prompt_idx"]); r = int(m["resp_idx"]); tok_idx = int(m["tok_idx"])
        ri = rollout_info.get((p, r))
        if ri is None: continue
        if (p, r) not in offset_cache:
            full = ri["prompt_text"] + ri["response"]
            enc = tok(full, return_offsets_mapping=True, truncation=True, max_length=2048)
            offset_cache[(p, r)] = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
        offs = offset_cache[(p, r)]
        if tok_idx >= len(offs):
            y[i] = int(ri["scored_correct"]); continue
        cpos = offs[tok_idx][0]
        next_idx = None
        for k, (char_open, bc) in enumerate(ri["blocks"]):
            if char_open > cpos:
                next_idx = k
                break
        if next_idx is None:
            y[i] = int(ri["scored_correct"])
        else:
            y[i] = int(ri["blocks"][next_idx][1])
            next_block_idx[i] = next_idx

    pos_frac = float(y.mean())
    print(f"[selector] labeled {len(y)} </think> rows; pos%={pos_frac:.3f}", flush=True)

    # Filter to clean-406
    clean = set(int(i) for i in json.load(open(CONTAM))["clean"])
    cmask = np.array([int(m["prompt_idx"]) in clean for m in meta])
    X_c = X[cmask]; y_c = y[cmask]; meta_c = [meta[i] for i in range(len(meta)) if cmask[i]]
    next_block_idx_c = next_block_idx[cmask]
    groups_c = np.array([int(m["prompt_idx"]) for m in meta_c])
    print(f"[selector] clean-406: {len(y_c)} positions; pos%={y_c.mean():.3f}")

    # Held-out probe scores via GroupKFold(5)
    print("[selector] training held-out probe (corrected labels)", flush=True)
    scores = np.full(len(y_c), np.nan)
    for tr, te in GroupKFold(5).split(X_c, y_c, groups_c):
        pipe = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(C=0.1, max_iter=2000))])
        pipe.fit(X_c[tr], y_c[tr])
        scores[te] = pipe.predict_proba(X_c[te])[:, 1]
    # Balanced diagnostic AUROC
    pos = np.where(y_c == 1)[0]; neg = np.where(y_c == 0)[0]
    nb = min(len(pos), len(neg))
    rng = np.random.RandomState(0)
    idx = np.concatenate([rng.choice(pos, nb, replace=False), rng.choice(neg, nb, replace=False)])
    bal_auc = float(roc_auc_score(y_c[idx], scores[idx]))
    print(f"[selector] balanced held-out AUROC at </think> (corrected labels): {bal_auc:.3f}")

    # Group by (p, r): list of (think_close_idx, tok_idx, probe_score, next_block_idx, label)
    by_rollout = defaultdict(list)
    for i, m in enumerate(meta_c):
        if np.isnan(scores[i]) or next_block_idx_c[i] < 0:
            continue
        by_rollout[(int(m["prompt_idx"]), int(m["resp_idx"]))].append(
            (int(m["think_close_idx"]), int(m["tok_idx"]), float(scores[i]),
             int(next_block_idx_c[i]), int(y_c[i]))
        )
    for k in by_rollout:
        by_rollout[k].sort()

    # Only rollouts with >=2 </think>'s
    rollouts = [(k, v) for k, v in by_rollout.items() if len(v) >= 2]
    print(f"[selector] {len(rollouts)} rollouts with >=2 </think> positions")

    # === Strategies ===
    n = len(rollouts)
    base_verifier = 0
    for k, v in rollouts:
        ri = rollout_info[k]
        base_verifier += int(ri["scored_correct"])
    base_verifier /= n

    # Oracle pick-first-answer (= first </think>'s next block correctness)
    oracle_pick_first = sum(v[0][4] for _, v in rollouts) / n
    # Oracle pick-last (== verifier? approximately)
    oracle_pick_last = sum(v[-1][4] for _, v in rollouts) / n
    # Oracle any-correct (upper bound)
    upper_any = sum(int(any(t[4] == 1 for t in v)) for _, v in rollouts) / n

    # Probe-max
    probe_max = sum(max(v, key=lambda t: t[2])[4] for _, v in rollouts) / n

    # Threshold sweep with two fallbacks
    def commit_fb_last(T):
        s = 0
        for _, v in rollouts:
            picked = None
            for ti, tok, sc, nbi, lbl in v:
                if sc >= T:
                    picked = lbl; break
            if picked is None:
                picked = v[-1][4]
            s += picked
        return s / n
    def commit_fb_argmax(T):
        s = 0
        for _, v in rollouts:
            picked = None
            for ti, tok, sc, nbi, lbl in v:
                if sc >= T:
                    picked = lbl; break
            if picked is None:
                picked = max(v, key=lambda t: t[2])[4]
            s += picked
        return s / n

    print(f"\n=== Probe-as-answer-selector using TRACE-FINAL probe at every </think> ===")
    print(f"  n_rollouts (>=2 </think>): {n}")
    print(f"  balanced AUROC of the </think> probe (corrected labels) = {bal_auc:.3f}")
    print()
    print(f"  BASE   verifier-scored (last <answer>)          : {base_verifier:.4f}")
    print(f"  ORC    oracle pick-first-block                  : {oracle_pick_first:.4f}")
    print(f"  ORC    oracle pick-last-block                   : {oracle_pick_last:.4f}")
    print(f"  UPPER  oracle any block correct                 : {upper_any:.4f}")
    print(f"  PROBE  argmax over all </think>                 : {probe_max:.4f}  (gain {probe_max - base_verifier:+.4f})")
    print()
    print(f"  Threshold sweep:")
    print(f"  {'T':>6}  {'fb=last':>10}  {'fb=argmax':>12}")
    best = (0.0, -1, "")
    rows_for_txt = []
    for T in np.linspace(0.1, 0.95, 18):
        a1 = commit_fb_last(T); a2 = commit_fb_argmax(T)
        print(f"  {T:>6.2f}  {a1:>10.4f}  {a2:>12.4f}")
        rows_for_txt.append((T, a1, a2))
        if a1 > best[0]: best = (a1, T, "fb=last")
        if a2 > best[0]: best = (a2, T, "fb=argmax")

    print()
    print(f"  Best: {best[2]} @ T={best[1]:.2f}: acc={best[0]:.4f}  (gain {best[0] - base_verifier:+.4f})")
    print(f"  Probe-as-selector ceiling vs the oracle pass@K_blocks:")
    print(f"     base={base_verifier:.3f}, best probe={best[0]:.3f}, oracle={upper_any:.3f}")
    print(f"     headroom captured: {100*(best[0] - base_verifier) / max(0.001, upper_any - base_verifier):.0f}%")

    out_lines = [
        f"Probe-as-answer-selector with CORRECTED trace-final probe at every </think>",
        f"  cache: {CACHE}",
        f"  n_rollouts (multi-</think>): {n}",
        f"  balanced AUROC of </think> probe with corrected labels: {bal_auc:.3f}",
        "",
        f"BASE   verifier-scored last <answer>            : {base_verifier:.4f}",
        f"ORC    oracle pick-first                        : {oracle_pick_first:.4f}",
        f"ORC    oracle pick-last                         : {oracle_pick_last:.4f}",
        f"UPPER  oracle any block correct (pass@K_blocks) : {upper_any:.4f}",
        f"PROBE  argmax over all </think>                 : {probe_max:.4f}",
        "",
        f"Threshold sweep:",
        f"  {'T':>6}  {'fb=last':>10}  {'fb=argmax':>12}",
    ]
    for T, a1, a2 in rows_for_txt:
        out_lines.append(f"  {T:>6.2f}  {a1:>10.4f}  {a2:>12.4f}")
    out_lines.append("")
    out_lines.append(f"Best: {best[2]} @ T={best[1]:.2f}: acc={best[0]:.4f}")
    out_lines.append(f"Gain over verifier baseline: {best[0] - base_verifier:+.4f}")
    out_lines.append(f"Headroom captured (vs oracle): {100*(best[0] - base_verifier) / max(0.001, upper_any - base_verifier):.0f}%")
    os.makedirs(os.path.dirname(OUT_TXT) or ".", exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\nwrote {OUT_TXT}")


if __name__ == "__main__":
    main()
