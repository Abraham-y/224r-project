"""Train logistic-regression probes on cached hidden states and report AUROCs.

Walks a directory of `.npz` files produced by `cache_hidden_states.py`
(one file per checkpoint x layer x position) and trains a held-out-problem
5-fold cross-validated probe per file. Reports:

  - Probe AUROC (held-out problem)
  - Shuffled-label baseline AUROC (should be ~0.5)
  - Random-direction baseline AUROC (should be ~0.5)

Plus a summary table per (checkpoint, layer, position) and the three key
comparisons the milestone needs:

  H1/H2: pre_answer AUROC on C_SFT vs C_outcome (does outcome RL change
         the internal correctness representation?)
  H3:    assertion AUROC vs neutral AUROC on C_outcome (does the concealment
         signal concentrate at confidence-asserting token positions?)

Pure CPU. Runs on a laptop in seconds.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# .npz filenames look like e.g. "C_outcome_l16_assertion.npz" or
# "C_SFT_l12_pre_answer.npz". Use a regex that handles the underscore-rich
# position name.
_FILENAME_RE = re.compile(r"^(?P<ckpt>[^_]+(?:_[^_]+)?)_l(?P<layer>\d+)_(?P<kind>pre_answer|assertion|neutral)\.npz$")


def parse_filename(name: str) -> tuple[str, int, str] | None:
    """Returns (checkpoint, layer, kind) or None if the name doesn't match."""
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    return m.group("ckpt"), int(m.group("layer")), m.group("kind")


def load_groups(meta_path: str) -> np.ndarray:
    """Load prompt_idx per row from a meta.json sidecar; used as the CV grouping key."""
    with open(meta_path) as f:
        meta = json.load(f)
    return np.array([row["prompt_idx"] for row in meta], dtype=np.int64)


def probe_auc_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5,
                 C: float = 0.1, seed: int = 0) -> float:
    """Mean held-out-problem AUROC over a GroupKFold(n_splits) CV."""
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        return float("nan")
    gkf = GroupKFold(n_splits=n_splits)
    aucs = []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        if len(np.unique(y[test_idx])) < 2:
            continue  # both classes must be present in holdout for AUROC
        scaler = StandardScaler().fit(X[train_idx])
        Xt = scaler.transform(X[train_idx])
        Xv = scaler.transform(X[test_idx])
        clf = LogisticRegression(
            penalty="l2", C=C, max_iter=2000, random_state=seed, solver="lbfgs"
        )
        clf.fit(Xt, y[train_idx])
        proba = clf.predict_proba(Xv)[:, 1]
        aucs.append(roc_auc_score(y[test_idx], proba))
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def shuffled_label_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> float:
    rng = np.random.RandomState(seed)
    y_shuf = rng.permutation(y)
    return probe_auc_cv(X, y_shuf, groups, seed=seed)


def random_direction_auc(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Project X onto a random unit vector and treat the projection as a classifier score."""
    rng = np.random.RandomState(seed)
    w = rng.normal(size=X.shape[1])
    w /= np.linalg.norm(w)
    s = X @ w
    if len(np.unique(y)) < 2:
        return float("nan")
    auc = roc_auc_score(y, s)
    return float(max(auc, 1 - auc))  # direction-agnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True,
                        help="Directory containing the .npz / .meta.json files.")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--C", type=float, default=0.1)
    args = parser.parse_args()

    results: dict[tuple[str, int, str], dict[str, float]] = {}

    files = sorted(glob.glob(os.path.join(args.cache_dir, "*.npz")))
    print(f"[analyze] found {len(files)} .npz files in {args.cache_dir}")
    for npz_path in files:
        name = os.path.basename(npz_path)
        parsed = parse_filename(name)
        if parsed is None:
            print(f"[analyze]   skipping unparseable {name}")
            continue
        ckpt, layer, kind = parsed
        meta_path = npz_path.replace(".npz", ".meta.json")
        if not os.path.exists(meta_path):
            print(f"[analyze]   missing meta for {name}, skipping")
            continue

        with np.load(npz_path) as data:
            X = data["X"]
            y = data["y"]
        groups = load_groups(meta_path)
        if X.shape[0] != groups.shape[0]:
            print(f"[analyze]   meta/X length mismatch for {name}, skipping")
            continue

        probe_auc = probe_auc_cv(X, y, groups, n_splits=args.n_splits, C=args.C)
        shuf_auc = shuffled_label_auc(X, y, groups)
        rand_auc = random_direction_auc(X, y)

        results[(ckpt, layer, kind)] = {
            "n": int(X.shape[0]),
            "n_correct": int(y.sum()),
            "probe": probe_auc,
            "shuffled": shuf_auc,
            "random_dir": rand_auc,
        }

    # --- Per-cell report -------------------------------------------------
    print()
    print("=" * 92)
    print("Per-(checkpoint, layer, position) probe AUROC")
    print("=" * 92)
    header = f"{'checkpoint':<14}{'layer':>6}{'kind':>14}{'n':>6}{'pos%':>7}{'probe':>10}{'shuffled':>10}{'rand':>10}"
    print(header)
    print("-" * len(header))
    for (ckpt, layer, kind), r in sorted(results.items()):
        pos_pct = 100.0 * r["n_correct"] / max(r["n"], 1)
        print(
            f"{ckpt:<14}{layer:>6}{kind:>14}{r['n']:>6}{pos_pct:>6.0f}%"
            f"{r['probe']:>10.3f}{r['shuffled']:>10.3f}{r['random_dir']:>10.3f}"
        )

    # --- Headline comparisons --------------------------------------------
    print()
    print("=" * 92)
    print("Headline comparisons")
    print("=" * 92)

    def best_layer_auroc(ckpt: str, kind: str) -> tuple[int, float] | None:
        cand = [
            (layer, r["probe"])
            for (c, layer, k), r in results.items()
            if c == ckpt and k == kind and not np.isnan(r["probe"])
        ]
        if not cand:
            return None
        return max(cand, key=lambda x: x[1])

    # H1 / H2: pre_answer AUROC on C_SFT vs C_outcome
    sft_pre = best_layer_auroc("C_SFT", "pre_answer")
    out_pre = best_layer_auroc("C_outcome", "pre_answer")
    print("\n[H1/H2] pre_answer (</think> position) probe AUROC, best layer per checkpoint:")
    if sft_pre and out_pre:
        sft_l, sft_a = sft_pre
        out_l, out_a = out_pre
        print(f"  C_SFT     L{sft_l}: {sft_a:.3f}")
        print(f"  C_outcome L{out_l}: {out_a:.3f}")
        print(f"  delta = {out_a - sft_a:+.3f}  ({'↑ RL strengthens internal correctness signal'
                                                  if out_a > sft_a else
                                                  '↓ RL weakens or removes internal correctness signal'})")
    else:
        print("  insufficient data")

    # H3: assertion AUROC vs neutral AUROC on C_outcome
    out_ass = best_layer_auroc("C_outcome", "assertion")
    out_neu = best_layer_auroc("C_outcome", "neutral")
    print("\n[H3] On C_outcome, probe AUROC at confidence-asserting vs neutral token positions:")
    if out_ass and out_neu:
        a_l, a_a = out_ass
        n_l, n_a = out_neu
        print(f"  assertion L{a_l}: {a_a:.3f}")
        print(f"  neutral   L{n_l}: {n_a:.3f}")
        delta = a_a - n_a
        verdict = (
            "↑ H3 supported (gap concentrates at confidence-asserting tokens)"
            if delta > 0.02 else
            ("↓ H3 falsified (assertion < neutral)" if delta < -0.02 else
             "≈ H3 null (gap is spatially uniform)")
        )
        print(f"  delta = {delta:+.3f}   {verdict}")
    else:
        print("  insufficient data")

    # Same comparison on C_SFT for context
    sft_ass = best_layer_auroc("C_SFT", "assertion")
    sft_neu = best_layer_auroc("C_SFT", "neutral")
    print("\n[H3 reference] Same comparison on C_SFT (does the localization preexist RL?):")
    if sft_ass and sft_neu:
        print(f"  assertion L{sft_ass[0]}: {sft_ass[1]:.3f}")
        print(f"  neutral   L{sft_neu[0]}: {sft_neu[1]:.3f}")
        print(f"  delta = {sft_ass[1] - sft_neu[1]:+.3f}")


if __name__ == "__main__":
    main()
