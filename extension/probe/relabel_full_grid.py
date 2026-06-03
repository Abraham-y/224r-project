"""Re-label probe rows by NEXT-`<answer>` correctness across the full grid:
  {C_SFT, C_outcome} × {pre_answer, assertion} × {L12, L16, L20}.

For each (ckpt, kind, layer) cell, compare:
  - Original AUROC: labels = verifier's last-answer correctness
  - Corrected AUROC: labels = correctness of the <answer> block that
    immediately follows the cached token position (pre_answer = the first
    </think>'s following <answer>; assertion = the next <answer> after the
    keyword token).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# Defaults; can be overridden via CLI for the fixed-sampler replication.
CACHE_DIR = "extension/cache/probe_cache_n500_clean406"
EVAL = {
    "C_SFT": "eval_c_sft_n500.json",
    "C_outcome": "eval_c_outcome_n500.json",
}
OUT_TXT = "extension/outputs/n500/text/29_relabel_full_grid.txt"

_ANSWER_OPEN_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_CLOSE_RE = re.compile(r"</think>")


def check_block(eq: str, target: int, nums: list[int]) -> bool:
    """Verify equation uses each num once + evaluates to target."""
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
    global CACHE_DIR, EVAL, OUT_TXT
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default=CACHE_DIR)
    ap.add_argument("--eval_sft", default=EVAL["C_SFT"])
    ap.add_argument("--eval_outcome", default=EVAL["C_outcome"])
    ap.add_argument("--out_txt", default=OUT_TXT)
    args = ap.parse_args()
    CACHE_DIR = args.cache_dir
    EVAL = {"C_SFT": args.eval_sft, "C_outcome": args.eval_outcome}
    OUT_TXT = args.out_txt

    from transformers import AutoTokenizer
    print(f"[relabel-full] cache_dir = {CACHE_DIR}", flush=True)
    print(f"[relabel-full] eval files = {EVAL}", flush=True)
    print("[relabel-full] loading tokenizers", flush=True)
    tok_sft = AutoTokenizer.from_pretrained("asingh15/qwen-sft-countdown-defaultproj", use_fast=True)
    tok_out = tok_sft  # same tokenizer

    # Load rollouts + per-block correctness (computed inline for both ckpts)
    print("[relabel-full] parsing rollouts + per-block correctness for both ckpts", flush=True)
    rollout_info: dict = {}
    for ckpt, path in EVAL.items():
        rows = [json.loads(l) for l in open(path) if l.strip()]
        for p_idx, row in enumerate(rows):
            prompt_text = row["prompt"]
            target = int(row["target"])
            nums = list(row["nums"])
            for r_idx, resp in enumerate(row["response"]):
                scored_correct = bool(row["scores"][r_idx] == 1.0)
                blocks = []
                for m in _ANSWER_OPEN_RE.finditer(resp):
                    eq = m.group(1)
                    bc = check_block(eq, target, nums)
                    char_open = len(prompt_text) + m.start()
                    blocks.append((char_open, bc))
                rollout_info[(ckpt, p_idx, r_idx)] = {
                    "blocks": blocks,
                    "scored_correct": scored_correct,
                    "prompt_text": prompt_text,
                    "response": resp,
                }
        print(f"[relabel-full]   {ckpt}: parsed {sum(1 for k in rollout_info if k[0] == ckpt)} rollouts")

    def get_offsets(ckpt, p, r):
        ri = rollout_info[(ckpt, p, r)]
        full = ri["prompt_text"] + ri["response"]
        tok = tok_sft if ckpt == "C_SFT" else tok_out
        enc = tok(full, return_offsets_mapping=True, truncation=True, max_length=2048)
        return [(int(s), int(e)) for s, e in enc["offset_mapping"]]

    def relabel_one(ckpt: str, kind: str, layer: int) -> dict[str, Any]:
        npz = os.path.join(CACHE_DIR, f"{ckpt}_l{layer}_{kind}.npz")
        if not os.path.exists(npz):
            return {"ckpt": ckpt, "kind": kind, "layer": layer, "skip": True}
        with np.load(npz) as d:
            X = d["X"]; y_orig = d["y"]
        meta = json.load(open(npz.replace(".npz", ".meta.json")))
        n = len(meta)
        y_new = np.zeros(n, dtype=np.int32)
        n_diff = 0; n_no_next = 0

        offset_cache: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for i, m in enumerate(meta):
            p = int(m["prompt_idx"]); r = int(m["resp_idx"]); tok_idx = int(m["tok_idx"])
            key = (ckpt, p, r)
            if key not in rollout_info:
                y_new[i] = int(y_orig[i]); continue
            ri = rollout_info[key]
            ok = (p, r)
            if ok not in offset_cache:
                offset_cache[ok] = get_offsets(ckpt, p, r)
            offs = offset_cache[ok]
            if tok_idx >= len(offs):
                y_new[i] = int(y_orig[i]); continue
            char_pos = offs[tok_idx][0]
            next_block_correct = None
            for char_open, bc in ri["blocks"]:
                if char_open > char_pos:
                    next_block_correct = bc
                    break
            if next_block_correct is None:
                y_new[i] = int(ri["scored_correct"]); n_no_next += 1
            else:
                y_new[i] = int(next_block_correct)
            if y_new[i] != int(y_orig[i]):
                n_diff += 1

        groups = np.array([int(m["prompt_idx"]) for m in meta])

        def heldout_auroc(y_arr):
            rng = np.random.RandomState(0)
            pos = np.where(y_arr == 1)[0]; neg = np.where(y_arr == 0)[0]
            nb = min(len(pos), len(neg))
            if nb < 5: return float("nan"), 0
            idx = np.concatenate([rng.choice(pos, nb, replace=False),
                                   rng.choice(neg, nb, replace=False)])
            Xs, ys, gs = X[idx], y_arr[idx], groups[idx]
            preds = np.full(len(ys), np.nan)
            for tr, te in GroupKFold(5).split(Xs, ys, gs):
                pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
                pipe.fit(Xs[tr], ys[tr])
                preds[te] = pipe.predict_proba(Xs[te])[:, 1]
            return float(roc_auc_score(ys, preds)), nb

        auc_o, nb_o = heldout_auroc(y_orig)
        auc_n, nb_n = heldout_auroc(y_new)
        print(f"  {ckpt:10} {kind:11} L{layer}  orig={auc_o:.3f} ({nb_o}/cls)  corrected={auc_n:.3f} ({nb_n}/cls)  Δ={auc_n - auc_o:+.3f}  ({n_diff}/{n} relabeled)",
              flush=True)
        return {
            "ckpt": ckpt, "kind": kind, "layer": layer,
            "auc_orig": auc_o, "auc_new": auc_n, "delta": auc_n - auc_o,
            "n_balanced_orig": nb_o, "n_balanced_new": nb_n,
            "n_relabeled_diff": n_diff, "n_no_next": n_no_next, "n_total": n,
        }

    results = []
    for ckpt in ("C_SFT", "C_outcome"):
        for kind in ("pre_answer", "assertion"):
            for layer in (12, 16, 20):
                results.append(relabel_one(ckpt, kind, layer))

    # Build summary
    summary_lines = ["Full-grid relabel: assertion/pre_answer probes with next-<answer>-block labels",
                     f"  cache: {CACHE_DIR}",
                     ""]
    summary_lines.append(f"{'ckpt':<12} {'kind':<11} {'layer':<5} {'orig':>7} {'corr':>7} {'Δ':>7} {'n_relabeled':>12}")
    summary_lines.append("-" * 65)
    for r in results:
        if r.get("skip"): continue
        summary_lines.append(
            f"{r['ckpt']:<12} {r['kind']:<11} L{r['layer']:<4} "
            f"{r['auc_orig']:>7.3f} {r['auc_new']:>7.3f} {r['delta']:>+7.3f} "
            f"{r['n_relabeled_diff']:>5}/{r['n_total']}"
        )

    # Gap recomputation
    summary_lines.append("")
    summary_lines.append("=== Position gap (pre_answer − assertion) at L16 ===")
    by = {(r["ckpt"], r["kind"]): r for r in results if r.get("layer") == 16}
    for ckpt in ("C_SFT", "C_outcome"):
        if (ckpt, "pre_answer") in by and (ckpt, "assertion") in by:
            pa_o = by[(ckpt, "pre_answer")]["auc_orig"]
            pa_n = by[(ckpt, "pre_answer")]["auc_new"]
            as_o = by[(ckpt, "assertion")]["auc_orig"]
            as_n = by[(ckpt, "assertion")]["auc_new"]
            gap_o = pa_o - as_o
            gap_n = pa_n - as_n
            summary_lines.append(
                f"  {ckpt:<10}  orig gap = pre {pa_o:.3f} − ass {as_o:.3f} = {gap_o:+.3f}    "
                f"corrected = pre {pa_n:.3f} − ass {as_n:.3f} = {gap_n:+.3f}    "
                f"Δgap = {gap_n - gap_o:+.3f}"
            )

    txt = "\n".join(summary_lines)
    print("\n" + txt)
    os.makedirs(os.path.dirname(OUT_TXT) or ".", exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write(txt + "\n")
    # JSON
    with open(OUT_TXT.replace(".txt", ".json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_TXT}")


if __name__ == "__main__":
    main()
