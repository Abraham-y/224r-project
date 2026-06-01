"""Test whether the trace-final-vs-assertion AUROC gap is explained by
distance-to-`</think>` (i.e., "time in reasoning") or by token kind.

For each cached row in {pre_answer, assertion, neutral} on C_outcome 0.5B
clean-406:
  1. Compute distance in tokens from this row's tok_idx to the rollout's
     `</think>` token (using pre_answer cache to look up `</think>` tok_idx
     per rollout).
  2. Bin by distance.
  3. Compute balanced GroupKFold(5) AUROC per (kind, distance_bin).

If AUROC depends only on distance-to-`</think>` regardless of position kind,
the gap is purely time-in-reasoning. If at matched distance the kind matters
(e.g., assertion-keyword > neutral), there IS a kind-of-position effect.

Also: most-common-token analysis on the neutral cache. We decode each cached
neutral token, group by token text, compute per-token AUROC for tokens with
>= 50 occurrences. Compares to per-keyword AUROC from §4.2.
"""

from __future__ import annotations

import json
import os
import warnings
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

CACHE_DIR = "extension/cache/probe_cache_n500_clean406"
LAYER = 16
CKPT = "C_outcome"
EVAL_JSON = "eval_c_outcome_n500.json"
OUT_TXT = "extension/outputs/n500/text/31_position_resolved_auroc.txt"
OUT_FIG = "extension/outputs/n500/figures/fig17_position_resolved_auroc.png"


def load_cache(kind):
    npz = os.path.join(CACHE_DIR, f"{CKPT}_l{LAYER}_{kind}.npz")
    with np.load(npz) as d:
        X = d["X"]; y = d["y"]
    meta = json.load(open(npz.replace(".npz", ".meta.json")))
    return X, y, meta


def balanced_auroc(X, y, groups, seed=0):
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    nb = min(len(pos), len(neg))
    if nb < 10: return float("nan"), nb
    idx = np.concatenate([rng.choice(pos, nb, replace=False), rng.choice(neg, nb, replace=False)])
    Xs, ys, gs = X[idx], y[idx], groups[idx]
    preds = np.full(len(ys), np.nan)
    try:
        for tr, te in GroupKFold(5).split(Xs, ys, gs):
            pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))])
            pipe.fit(Xs[tr], ys[tr])
            preds[te] = pipe.predict_proba(Xs[te])[:, 1]
        if len(set(ys.tolist())) >= 2:
            return float(roc_auc_score(ys, preds)), nb
    except Exception:
        pass
    return float("nan"), nb


def main():
    print("[pos-res] loading caches", flush=True)
    pre_X, pre_y, pre_meta = load_cache("pre_answer")
    ass_X, ass_y, ass_meta = load_cache("assertion")
    neu_X, neu_y, neu_meta = load_cache("neutral")

    # Build pre_answer index: (p, r) -> tok_idx of </think>
    pre_tok = {(int(m["prompt_idx"]), int(m["resp_idx"])): int(m["tok_idx"]) for m in pre_meta}
    print(f"[pos-res] {len(pre_tok)} rollouts have </think> tok_idx cached")

    # Compute distance-to-</think> per row in assertion / neutral
    def add_distance(meta, pre_tok):
        dists = []
        for m in meta:
            key = (int(m["prompt_idx"]), int(m["resp_idx"]))
            if key not in pre_tok:
                dists.append(None); continue
            d = pre_tok[key] - int(m["tok_idx"])
            # positive = before </think>, negative = after
            dists.append(d)
        return dists

    ass_dist = add_distance(ass_meta, pre_tok)
    neu_dist = add_distance(neu_meta, pre_tok)
    pre_dist = [0] * len(pre_meta)  # by definition

    # Distance distribution
    ass_d_clean = [d for d in ass_dist if d is not None and d >= 0]
    print(f"[pos-res] assertion distance-to-</think> distribution (positive = before </think>):")
    print(f"   p10={np.percentile(ass_d_clean, 10):.0f}, p25={np.percentile(ass_d_clean, 25):.0f}, "
          f"p50={np.percentile(ass_d_clean, 50):.0f}, p75={np.percentile(ass_d_clean, 75):.0f}, "
          f"p90={np.percentile(ass_d_clean, 90):.0f}")
    neu_d_clean = [d for d in neu_dist if d is not None and d >= 0]
    if neu_d_clean:
        print(f"[pos-res] neutral distance-to-</think> distribution:")
        print(f"   p10={np.percentile(neu_d_clean, 10):.0f}, p25={np.percentile(neu_d_clean, 25):.0f}, "
              f"p50={np.percentile(neu_d_clean, 50):.0f}, p75={np.percentile(neu_d_clean, 75):.0f}, "
              f"p90={np.percentile(neu_d_clean, 90):.0f}")

    # Bin by distance and compute AUROC per (kind, bin)
    bins = [(0, 1), (1, 50), (50, 150), (150, 400), (400, 800), (800, 2000)]
    bin_labels = [f"0 (= </think>)", "1-50 (just before)", "50-150", "150-400", "400-800", "800+"]

    print(f"\n[pos-res] AUROC by distance bin (corrected next-<answer> labels NOT yet applied here; "
          f"using original cache labels):")
    print(f"{'kind':>11}  {'dist bin':>20}  {'n':>6}  {'AUROC':>7}  {'n_bal/cls':>10}")
    rows = []
    for kind, X, y, meta, dists in (("pre_answer", pre_X, pre_y, pre_meta, pre_dist),
                                     ("assertion", ass_X, ass_y, ass_meta, ass_dist),
                                     ("neutral", neu_X, neu_y, neu_meta, neu_dist)):
        for (lo, hi), label in zip(bins, bin_labels):
            mask = np.array([d is not None and lo <= d < hi for d in dists])
            if mask.sum() < 50:
                continue
            X_sel = X[mask]; y_sel = y[mask]
            groups_sel = np.array([int(m["prompt_idx"]) for m in [meta[i] for i in range(len(meta)) if mask[i]]])
            auc, nb = balanced_auroc(X_sel, y_sel, groups_sel)
            print(f"{kind:>11}  {label:>20}  {int(mask.sum()):>6}  {auc:>7.3f}  {nb:>10}")
            rows.append({"kind": kind, "bin": label, "n": int(mask.sum()), "auroc": auc, "n_balanced": nb})

    # --- Token-frequency analysis on neutral cache (and per-keyword on assertion) ---
    print(f"\n[pos-res] Decoding neutral cache token strings...", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("asingh15/qwen-sft-countdown-defaultproj", use_fast=True)
    eval_rows = [json.loads(l) for l in open(EVAL_JSON) if l.strip()]
    rollout_text = {(p_idx, r_idx): row["prompt"] + row["response"][r_idx]
                    for p_idx, row in enumerate(eval_rows) for r_idx in range(len(row["response"]))}

    # Re-tokenize each rollout and get token strings at neutral tok_idx
    offset_cache = {}
    neutral_token_strings = []
    for m in neu_meta:
        key = (int(m["prompt_idx"]), int(m["resp_idx"]))
        if key not in offset_cache:
            full = rollout_text[key]
            enc = tok(full, return_offsets_mapping=True, truncation=True, max_length=2048)
            input_ids = enc["input_ids"]
            offset_cache[key] = (input_ids, [(int(s), int(e)) for s, e in enc["offset_mapping"]])
        input_ids, offsets = offset_cache[key]
        tid = int(m["tok_idx"])
        if tid < len(input_ids):
            neutral_token_strings.append(tok.decode([input_ids[tid]]).strip().lower())
        else:
            neutral_token_strings.append("")

    counts = Counter(neutral_token_strings)
    print(f"[pos-res] Most common neutral token strings:")
    for word, c in counts.most_common(15):
        print(f"   {word!r}: {c}")

    # Per-token AUROC for tokens with enough occurrences
    print(f"\n[pos-res] Per-neutral-token AUROC (tokens with >= 100 occurrences):")
    by_token = defaultdict(lambda: {"X": [], "y": [], "groups": [], "dists": []})
    for i, m in enumerate(neu_meta):
        word = neutral_token_strings[i]
        if word == "": continue
        by_token[word]["X"].append(neu_X[i])
        by_token[word]["y"].append(neu_y[i])
        by_token[word]["groups"].append(int(m["prompt_idx"]))
        by_token[word]["dists"].append(neu_dist[i])
    print(f"{'token':>15}  {'n':>5}  {'AUROC':>7}  {'mean_dist':>10}")
    per_token_rows = []
    for word, d in counts.most_common(20):
        if d < 100: continue
        Xs = np.array(by_token[word]["X"])
        ys = np.array(by_token[word]["y"])
        gs = np.array(by_token[word]["groups"])
        dists_w = [x for x in by_token[word]["dists"] if x is not None and x >= 0]
        if len(Xs) < 100: continue
        auc, nb = balanced_auroc(Xs, ys, gs)
        md = float(np.mean(dists_w)) if dists_w else float("nan")
        print(f"{word!r:>15}  {len(Xs):>5}  {auc:>7.3f}  {md:>10.0f}")
        per_token_rows.append({"token": word, "n": len(Xs), "auroc": auc, "mean_distance": md})

    # Save outputs
    out = {
        "by_distance_bin": rows,
        "by_neutral_token": per_token_rows,
        "neutral_token_counts": dict(counts.most_common(30)),
    }
    os.makedirs(os.path.dirname(OUT_TXT) or ".", exist_ok=True)
    with open(OUT_TXT.replace(".txt", ".json"), "w") as f:
        json.dump(out, f, indent=2)
    txt_lines = ["Position-resolved AUROC + neutral-token frequency analysis", "",
                 "=== AUROC by (kind, distance-to-</think>) bin ==="]
    for r in rows:
        txt_lines.append(f"  {r['kind']:<11}  {r['bin']:<22}  n={r['n']:<6}  AUROC={r['auroc']:.3f}")
    txt_lines += ["", "=== Per-neutral-token AUROC ==="]
    for r in per_token_rows:
        txt_lines.append(f"  {r['token']:<15}  n={r['n']:<5}  AUROC={r['auroc']:.3f}  mean_dist={r['mean_distance']:.0f}")
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(txt_lines) + "\n")
    print(f"\nwrote {OUT_TXT}")

    # Figure: AUROC vs distance bin per kind
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    kinds = ["pre_answer", "assertion", "neutral"]
    colors = {"pre_answer": "#3a8b2f", "assertion": "#c45252", "neutral": "#888888"}
    for kind in kinds:
        kind_rows = [r for r in rows if r["kind"] == kind]
        if not kind_rows: continue
        xs = list(range(len(kind_rows)))
        ys = [r["auroc"] for r in kind_rows]
        labels = [r["bin"] for r in kind_rows]
        ax.plot(xs, ys, marker="o", color=colors[kind], lw=2, markersize=10, label=kind)
        for x, y_val, lab in zip(xs, ys, labels):
            ax.annotate(f"{y_val:.2f}\n(n={[r for r in kind_rows if r['auroc'] == y_val][0]['n']})",
                        (x, y_val), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color=colors[kind])
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_xlabel("distance from cached token to </think> (in tokens)")
    ax.set_ylabel("balanced GroupKFold(5) AUROC")
    ax.set_title("Position-resolved AUROC: is the gap explained by distance-to-</think>?\n"
                 "(C_outcome 0.5B L16, clean-406)")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG) or ".", exist_ok=True)
    fig.savefig(OUT_FIG)
    plt.close(fig)
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
