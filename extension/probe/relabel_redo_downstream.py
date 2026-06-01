"""Recompute matched-pair + per-problem AUROC analyses with CORRECTED labels
(next-`<answer>`-block correctness instead of verifier-final-answer correctness).

For each (ckpt, layer) at the assertion and pre_answer positions:
  1. Load cached hidden states + meta.
  2. Compute corrected labels: each row gets the correctness of the
     `<answer>` block immediately following its cached token.
  3. Train held-out probe via GroupKFold(5) using CORRECTED labels.
  4. Compute held-out probe scores for all rows.
  5. Re-run downstream analyses:
       - Matched-pair within-prompt at assertion position
       - Per-problem AUROC at pre_answer

Pure local re-analysis, no Modal. Updates the matched-pair, per-problem
Spearman, and per-problem AUROC stats with the relabeled probe.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

_ANSWER_OPEN_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def check_block(eq: str, target: int, nums: list[int]) -> bool:
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


def parse_rollouts(eval_json: str) -> dict:
    """For each (prompt_idx, resp_idx) -> list of (char_open, block_correct)."""
    out = {}
    rows = [json.loads(l) for l in open(eval_json) if l.strip()]
    for p_idx, row in enumerate(rows):
        prompt_text = row["prompt"]
        target = int(row["target"])
        nums = list(row["nums"])
        for r_idx, resp in enumerate(row["response"]):
            scored = bool(row["scores"][r_idx] == 1.0)
            blocks = []
            for m in _ANSWER_OPEN_RE.finditer(resp):
                bc = check_block(m.group(1), target, nums)
                blocks.append((len(prompt_text) + m.start(), bc))
            out[(p_idx, r_idx)] = {
                "blocks": blocks, "scored_correct": scored,
                "prompt_text": prompt_text, "response": resp,
            }
    return out


def relabel_meta(meta: list[dict], rollout_info: dict, tokenizer) -> np.ndarray:
    """Compute next-<answer>-block correctness label per cached row."""
    n = len(meta)
    new_y = np.zeros(n, dtype=np.int32)
    offsets_cache: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i, m in enumerate(meta):
        p = int(m["prompt_idx"]); r = int(m["resp_idx"]); tok_idx = int(m["tok_idx"])
        ri = rollout_info.get((p, r))
        if ri is None: continue
        if (p, r) not in offsets_cache:
            full = ri["prompt_text"] + ri["response"]
            enc = tokenizer(full, return_offsets_mapping=True, truncation=True, max_length=2048)
            offsets_cache[(p, r)] = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
        offs = offsets_cache[(p, r)]
        if tok_idx >= len(offs):
            new_y[i] = int(ri["scored_correct"]); continue
        cpos = offs[tok_idx][0]
        nxt = None
        for char_open, bc in ri["blocks"]:
            if char_open > cpos:
                nxt = bc; break
        new_y[i] = int(nxt) if nxt is not None else int(ri["scored_correct"])
    return new_y


def heldout_scores(X, y, groups, n_splits=5):
    """One held-out probe score per row, no class balancing (we want all rows)."""
    scores = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        pipe = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(C=0.1, max_iter=2000))])
        pipe.fit(X[tr], y[tr])
        scores[te] = pipe.predict_proba(X[te])[:, 1]
    return scores


def balanced_auroc(X, y, groups, seed=0):
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    nb = min(len(pos), len(neg))
    if nb < 5: return float("nan")
    idx = np.concatenate([rng.choice(pos, nb, replace=False), rng.choice(neg, nb, replace=False)])
    Xs, ys, gs = X[idx], y[idx], groups[idx]
    preds = np.full(len(ys), np.nan)
    for tr, te in GroupKFold(5).split(Xs, ys, gs):
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
        pipe.fit(Xs[tr], ys[tr])
        preds[te] = pipe.predict_proba(Xs[te])[:, 1]
    return float(roc_auc_score(ys, preds))


def matched_pair_within_prompt(scores, labels, groups):
    """For each prompt with mixed-label rows, mean(probe|correct) - mean(probe|wrong)."""
    by_p = defaultdict(list)
    for i in range(len(scores)):
        if np.isnan(scores[i]): continue
        by_p[int(groups[i])].append((float(scores[i]), int(labels[i])))
    deltas = []
    for p, lst in by_p.items():
        c = [s for s, l in lst if l == 1]
        w = [s for s, l in lst if l == 0]
        if len(c) > 0 and len(w) > 0:
            deltas.append(np.mean(c) - np.mean(w))
    return np.array(deltas)


def per_problem_auroc(scores, labels, groups):
    by_p = defaultdict(list)
    for i in range(len(scores)):
        if np.isnan(scores[i]): continue
        by_p[int(groups[i])].append((float(scores[i]), int(labels[i])))
    aurocs = {}
    for p, lst in by_p.items():
        ls = [t[1] for t in lst]
        if len(set(ls)) < 2: continue
        aurocs[p] = roc_auc_score(ls, [t[0] for t in lst])
    return aurocs


def per_prompt_acc(eval_json, n_problems):
    rows = [json.loads(l) for l in open(eval_json) if l.strip()]
    acc = np.full(n_problems, np.nan)
    for p_idx, row in enumerate(rows):
        if p_idx >= n_problems: break
        scs = row.get("scores", [])[:16]
        if scs:
            acc[p_idx] = float(np.mean([float(s) == 1.0 for s in scs]))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--sft_eval", required=True)
    ap.add_argument("--outcome_eval", required=True)
    ap.add_argument("--tokenizer_sft", required=True)
    ap.add_argument("--tokenizer_outcome", required=True)
    ap.add_argument("--contam_json", default="extension/data/contaminated_prompt_idx.json")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--label", required=True, help="0.5B or 1.5B")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[{args.label}] parsing rollouts + per-block correctness", flush=True)
    sft_info = parse_rollouts(args.sft_eval)
    out_info = parse_rollouts(args.outcome_eval)
    print(f"[{args.label}]   SFT: {len(sft_info)} rollouts, C_outcome: {len(out_info)} rollouts")

    print(f"[{args.label}] loading tokenizer", flush=True)
    tok_sft = AutoTokenizer.from_pretrained(args.tokenizer_sft, use_fast=True)
    tok_out = AutoTokenizer.from_pretrained(args.tokenizer_outcome, use_fast=True)

    results = {}
    held_out_scores: dict = {}
    held_out_labels: dict = {}

    for ckpt, info, tok in (("C_SFT", sft_info, tok_sft), ("C_outcome", out_info, tok_out)):
        for kind in ("pre_answer", "assertion"):
            npz = os.path.join(args.cache_dir, f"{ckpt}_l{args.layer}_{kind}.npz")
            if not os.path.exists(npz):
                print(f"[{args.label}] missing {npz}")
                continue
            with np.load(npz) as d:
                X = d["X"]; y_orig = d["y"]
            meta = json.load(open(npz.replace(".npz", ".meta.json")))
            groups = np.array([int(m["prompt_idx"]) for m in meta])
            print(f"[{args.label}] relabeling {ckpt} {kind}: {len(meta)} rows", flush=True)
            y_new = relabel_meta(meta, info, tok)
            n_diff = int((y_new != y_orig).sum())
            print(f"[{args.label}]   {n_diff}/{len(meta)} labels changed; new pos%={y_new.mean():.3f}")

            auc_o = balanced_auroc(X, y_orig, groups, seed=0)
            auc_n = balanced_auroc(X, y_new, groups, seed=0)
            print(f"[{args.label}]   AUROC: orig {auc_o:.3f} -> corrected {auc_n:.3f}  (Δ {auc_n - auc_o:+.3f})", flush=True)

            print(f"[{args.label}]   computing held-out scores (corrected labels)", flush=True)
            scores_new = heldout_scores(X, y_new, groups)
            held_out_scores[(ckpt, kind)] = scores_new
            held_out_labels[(ckpt, kind)] = y_new

            results.setdefault(ckpt, {})[kind] = {
                "auc_orig": auc_o, "auc_new": auc_n,
                "n_relabeled": n_diff, "n_total": len(meta),
                "pos_frac_orig": float(y_orig.mean()), "pos_frac_new": float(y_new.mean()),
            }

    # === Matched-pair within-prompt (assertion) with CORRECTED labels ===
    print(f"\n[{args.label}] === Matched-pair (assertion-position, corrected labels) ===")
    mp_results = {}
    for ckpt in ("C_SFT", "C_outcome"):
        if (ckpt, "assertion") not in held_out_scores: continue
        scores = held_out_scores[(ckpt, "assertion")]
        labels = held_out_labels[(ckpt, "assertion")]
        npz = os.path.join(args.cache_dir, f"{ckpt}_l{args.layer}_assertion.npz")
        meta = json.load(open(npz.replace(".npz", ".meta.json")))
        groups = np.array([int(m["prompt_idx"]) for m in meta])
        deltas = matched_pair_within_prompt(scores, labels, groups)
        n = len(deltas)
        above = int((deltas > 0).sum())
        below = int((deltas < 0).sum())
        median = float(np.median(deltas)) if n else float("nan")
        mean = float(np.mean(deltas)) if n else float("nan")
        try:
            w_stat, w_p = wilcoxon(deltas, alternative="greater")
            w_p = float(w_p)
        except Exception:
            w_p = float("nan")
        print(f"  {ckpt}: n={n}  median Δ={median:+.4f}  mean Δ={mean:+.4f}  above={above}/{n} ({100*above/max(1,n):.0f}%)  Wilcoxon p (>0) = {w_p:.3e}")
        mp_results[ckpt] = {"n": n, "median_delta": median, "mean_delta": mean,
                             "above_diag": above, "below_diag": below,
                             "wilcoxon_p_greater": w_p, "deltas": deltas.tolist()}
    if "C_SFT" in mp_results and "C_outcome" in mp_results:
        s = np.array(mp_results["C_SFT"]["deltas"])
        o = np.array(mp_results["C_outcome"]["deltas"])
        try:
            u, p = mannwhitneyu(s, o, alternative="greater")
            print(f"  Mann-Whitney U (C_SFT deltas > C_outcome deltas, corrected): p = {float(p):.3e}")
            mp_results["mw_between_p_greater"] = float(p)
        except Exception:
            mp_results["mw_between_p_greater"] = float("nan")

    # === Per-problem AUROC + Spearman (pre_answer trace-final) ===
    print(f"\n[{args.label}] === Per-problem trace-final AUROC + Spearman (corrected labels) ===")
    pp_results = {}
    n_problems = 500  # cover the full procedural range
    acc_sft = per_prompt_acc(args.sft_eval, n_problems)
    acc_rloo = per_prompt_acc(args.outcome_eval, n_problems)
    accuracy_delta_full = acc_rloo - acc_sft

    clean = set(int(i) for i in json.load(open(args.contam_json))["clean"]) if os.path.exists(args.contam_json) else set(range(n_problems))

    for ckpt in ("C_SFT", "C_outcome"):
        if (ckpt, "pre_answer") not in held_out_scores: continue
        scores = held_out_scores[(ckpt, "pre_answer")]
        labels = held_out_labels[(ckpt, "pre_answer")]
        npz = os.path.join(args.cache_dir, f"{ckpt}_l{args.layer}_pre_answer.npz")
        meta = json.load(open(npz.replace(".npz", ".meta.json")))
        groups = np.array([int(m["prompt_idx"]) for m in meta])
        aurocs = per_problem_auroc(scores, labels, groups)
        # Apply clean filter
        aurocs = {p: v for p, v in aurocs.items() if p in clean}
        a = np.full(n_problems, np.nan)
        for p, v in aurocs.items():
            a[p] = v
        pp_results[ckpt] = a
        print(f"  {ckpt} per-problem AUROC: mean={np.nanmean(a):.3f}  med={np.nanmedian(a):.3f}  n={(~np.isnan(a)).sum()}")

    # Spearman
    if "C_SFT" in pp_results and "C_outcome" in pp_results:
        probe_drop = pp_results["C_SFT"] - pp_results["C_outcome"]
        valid = np.isfinite(probe_drop) & np.isfinite(accuracy_delta_full)
        for p in range(n_problems):
            if p not in clean:
                valid[p] = False
        n_v = int(valid.sum())
        r, p_val = spearmanr(probe_drop[valid], accuracy_delta_full[valid])
        # quadrants
        pd_v = probe_drop[valid]; ad_v = accuracy_delta_full[valid]
        decoup = int(((pd_v > 0) & (ad_v > 0)).sum())
        damage = int(((pd_v > 0) & (ad_v < 0)).sum())
        bothup = int(((pd_v < 0) & (ad_v > 0)).sum())
        noise = int(((pd_v < 0) & (ad_v < 0)).sum())
        on_axis = n_v - decoup - damage - bothup - noise
        print(f"\n  Spearman (probe_drop vs accuracy_delta, n={n_v}): r={float(r):+.3f}  p={float(p_val):.3e}")
        print(f"  Quadrants:")
        print(f"    decoupling (probe↓ acc↑): {decoup} ({100*decoup/n_v:.0f}%)")
        print(f"    damage     (probe↓ acc↓): {damage} ({100*damage/n_v:.0f}%)")
        print(f"    both up    (probe↑ acc↑): {bothup} ({100*bothup/n_v:.0f}%)")
        print(f"    noise      (probe↑ acc↓): {noise} ({100*noise/n_v:.0f}%)")
        print(f"    on-axis: {on_axis}")
        pp_results["spearman_r"] = float(r)
        pp_results["spearman_p"] = float(p_val)
        pp_results["n_valid"] = n_v
        pp_results["quadrants"] = {"decoupling": decoup, "damage": damage, "both_up": bothup, "noise": noise, "on_axis": on_axis}

    # Save
    out_path = os.path.join(args.out_dir, f"relabel_downstream_{args.label}.json")
    payload = {
        "label": args.label, "layer": args.layer,
        "cache_dir": args.cache_dir,
        "auroc_results": results,
        "matched_pair": {ckpt: {k: v for k, v in r.items() if k != "deltas"}
                          for ckpt, r in mp_results.items() if isinstance(r, dict)},
        "matched_pair_mw_p": mp_results.get("mw_between_p_greater"),
        "per_problem": {"spearman_r": pp_results.get("spearman_r"),
                         "spearman_p": pp_results.get("spearman_p"),
                         "n_valid": pp_results.get("n_valid"),
                         "quadrants": pp_results.get("quadrants")},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
