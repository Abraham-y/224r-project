"""Per-layer probe AUROC sweep across all 25 hidden-state layers (0..24).

For each (ckpt, layer, kind), trains a balanced GroupKFold(5) probe and
reports diagonal AUROC + gap pre_answer - assertion.

LABELS: by default this uses CORRECTED first-<answer>-block correctness, the
same label definition as every applied-probe and relabel script. The `y` array
baked into the .npz at cache time is the verifier's ROLLOUT-FINAL score, which
is a different quantity -- using it here (as this script previously did) put the
per-layer figure on a different label definition from the headline AUROC table.
Pass --labels cached to get the old behaviour.

Output: a table + a 2-panel figure (one panel per ckpt).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _check_block(eq: str, target: int, nums: list) -> bool:
    eq = eq.strip()
    try:
        n = [int(x) for x in re.findall(r"\d+", eq)]
        if sorted(n) != sorted([int(x) for x in nums]):
            return False
        if not re.match(r"^[\d+\-*/().\s]+$", eq):
            return False
        return abs(eval(eq, {"__builtins__": None}, {}) - int(target)) < 1e-5
    except Exception:
        return False


def first_block_labels(eval_path: str) -> dict:
    """(prompt_idx, resp_idx) -> first-<answer>-block correctness."""
    labs = {}
    for p, row in enumerate(json.loads(l) for l in open(eval_path) if l.strip()):
        t = int(row["target"]); nums = list(row["nums"])
        for r_i, resp in enumerate(row["response"]):
            m = _ANSWER_RE.search(resp)
            labs[(p, r_i)] = int(bool(m) and _check_block(m.group(1), t, nums))
    return labs


def load_cell(cache_dir: str, ckpt: str, layer: int, kind: str, label_map=None):
    npz = os.path.join(cache_dir, f"{ckpt}_l{layer}_{kind}.npz")
    meta_path = npz.replace(".npz", ".meta.json")
    if not (os.path.exists(npz) and os.path.exists(meta_path)):
        return None
    with np.load(npz) as d:
        X = d["X"]; y_cached = d["y"]
    meta = json.load(open(meta_path))
    groups = np.array([m["prompt_idx"] for m in meta])
    if label_map is None:
        y = y_cached
    else:
        y = np.array([label_map.get((int(m["prompt_idx"]), int(m["resp_idx"])),
                                    int(y_cached[i]))
                      for i, m in enumerate(meta)], dtype=np.int32)
    return X, y, groups


def balanced_auroc(X, y, groups, seed=0):
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    n = min(len(pos), len(neg))
    if n < 5: return float("nan")
    idx = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
    Xs, ys, gs = X[idx], y[idx], groups[idx]
    preds = np.full(len(ys), np.nan)
    for tr, te in GroupKFold(5).split(Xs, ys, gs):
        sc = StandardScaler().fit(Xs[tr])
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(Xs[tr]), ys[tr])
        preds[te] = clf.predict_proba(sc.transform(Xs[te]))[:, 1]
    mask = ~np.isnan(preds)
    return float(roc_auc_score(ys[mask], preds[mask]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="extension/cache/probe_cache_n500_all_layers_clean406")
    parser.add_argument("--n_layers", type=int, default=25)
    parser.add_argument("--out_txt", default="extension/outputs/n500/text/22_per_layer_sweep.txt")
    parser.add_argument("--out_fig", default="extension/outputs/n500/figures/fig11_per_layer_sweep.png")
    parser.add_argument("--labels", choices=("corrected", "cached"), default="corrected",
                        help="'corrected' = first-<answer>-block correctness (matches the rest "
                             "of the paper). 'cached' = the verifier rollout-final label baked "
                             "into the .npz (the old, inconsistent behaviour).")
    parser.add_argument("--eval_sft", default="eval_c_sft_n500.json")
    parser.add_argument("--eval_outcome", default="eval_c_outcome_n500.json")
    args = parser.parse_args()

    layers = list(range(args.n_layers))
    ckpts = ("C_SFT", "C_outcome")
    kinds = ("pre_answer", "assertion", "neutral")

    label_maps = {c: None for c in ckpts}
    if args.labels == "corrected":
        label_maps = {"C_SFT": first_block_labels(args.eval_sft),
                      "C_outcome": first_block_labels(args.eval_outcome)}
    print(f"[per-layer] labels = {args.labels}")

    results: dict = {(c, k): {} for c in ckpts for k in kinds}
    for layer in layers:
        for c in ckpts:
            for k in kinds:
                cell = load_cell(args.cache_dir, c, layer, k, label_maps[c])
                if cell is None:
                    print(f"missing {c} L{layer} {k}")
                    continue
                auc = balanced_auroc(*cell, seed=0)
                results[(c, k)][layer] = auc
                print(f"{c:>10} L{layer:>2} {k:<11} {auc:.3f}")

    # Build table
    lines = ["Per-layer probe AUROC sweep (clean-406, balanced GroupKFold(5))", ""]
    lines.append(f"{'layer':>6}  {'SFT_pre':>8} {'OUT_pre':>8} {'SFT_ass':>8} {'OUT_ass':>8} {'SFT_neu':>8} {'OUT_neu':>8}  {'OUT_gap':>8}")
    for layer in layers:
        row = f"{layer:>6}  "
        for ck in ((c, k) for k in ("pre_answer","assertion","neutral") for c in ("C_SFT","C_outcome")):
            v = results[(ck[0], ck[1])].get(layer, float("nan"))
            row += f"{v:>8.3f} "
        gap = results[("C_outcome","pre_answer")].get(layer, float("nan")) - results[("C_outcome","assertion")].get(layer, float("nan"))
        row += f"  {gap:>8.3f}"
        lines.append(row)

    # Find max-gap layer
    gaps = {l: results[("C_outcome","pre_answer")].get(l, float("nan")) - results[("C_outcome","assertion")].get(l, float("nan")) for l in layers}
    valid_gaps = {l: g for l, g in gaps.items() if not np.isnan(g)}
    max_gap_layer = max(valid_gaps, key=valid_gaps.get)
    lines.append("")
    lines.append(f"Max C_outcome (pre-ass) gap is at L{max_gap_layer}: {valid_gaps[max_gap_layer]:.3f}")
    lines.append(f"Min C_outcome (pre-ass) gap is at L{min(valid_gaps, key=valid_gaps.get)}: {min(valid_gaps.values()):.3f}")

    txt = "\n".join(lines)
    print()
    print(txt)
    os.makedirs(os.path.dirname(args.out_txt) or ".", exist_ok=True)
    with open(args.out_txt, "w") as f:
        f.write(txt + "\n")

    # Figure: 2 panels (one per ckpt), 3 lines each (pre/ass/neu)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160, sharey=True)
    for ax, ckpt in zip(axes, ckpts):
        for kind, color in (("pre_answer", "#3a8b2f"), ("assertion", "#c45252"), ("neutral", "#888888")):
            xs = sorted(results[(ckpt, kind)].keys())
            ys = [results[(ckpt, kind)][x] for x in xs]
            ax.plot(xs, ys, marker="o", label=kind, color=color, linewidth=2)
        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.7, alpha=0.6)
        ax.set_xlabel("hidden-state layer (0=embedding, 24=final)", fontsize=11)
        ax.set_title(ckpt, fontsize=12)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)
        ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("balanced GroupKFold(5) probe AUROC", fontsize=11)
    fig.suptitle("Per-layer probe AUROC sweep, clean-406 (n=406 unseen problems)", fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_fig) or ".", exist_ok=True)
    fig.savefig(args.out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {args.out_fig}")


if __name__ == "__main__":
    main()
