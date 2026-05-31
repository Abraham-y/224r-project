"""Length-matched cross-checkpoint transfer control.

The §2.4 transfer matrix shows off-diagonal AUROCs collapsing to ~0.52-0.62
in both directions. The standard reading is representation drift. The
alternative reading is that the probes are exploiting checkpoint-specific
text-distribution features (e.g., response length) rather than a shared
correctness subspace.

This script filters each checkpoint's rollouts so the response-length
distribution matches the other checkpoint's, then re-runs the transfer
analysis on the length-matched data.

If off-diagonals stay near chance after length-matching -> drift reading
strengthens (the probes were not just length-classifiers).
If off-diagonals recover noticeably -> the original framing needs to be
qualified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from extension.probe.analyze_probes import load_groups
from extension.probe.robustness_probes import balanced_subsample_auroc as _bal_cv


def load_response_lengths(eval_json: str) -> dict[tuple[int, int], int]:
    """Returns {(prompt_idx, resp_idx): char_length}."""
    out: dict[tuple[int, int], int] = {}
    with open(eval_json) as f:
        text = f.read().strip()
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    for p_idx, row in enumerate(rows):
        for r_idx, resp in enumerate(row["response"]):
            out[(p_idx, r_idx)] = len(resp)
    return out


def load_cell(cache_dir: str, ckpt: str, layer: int, kind: str):
    npz = os.path.join(cache_dir, f"{ckpt}_l{layer}_{kind}.npz")
    meta_path = npz.replace(".npz", ".meta.json")
    if not (os.path.exists(npz) and os.path.exists(meta_path)):
        return None
    with np.load(npz) as data:
        X = data["X"]; y = data["y"]
    with open(meta_path) as f:
        meta = json.load(f)
    groups = load_groups(meta_path)
    return X, y, groups, meta


def per_row_lengths(meta: list[dict], eval_lengths: dict[tuple[int, int], int]) -> np.ndarray:
    return np.array(
        [eval_lengths.get((int(m["prompt_idx"]), int(m["resp_idx"])), -1)
         for m in meta],
        dtype=np.int64,
    )


def length_matched_mask(
    src_lengths: np.ndarray, ref_lengths: np.ndarray, tolerance: float = 0.3
) -> np.ndarray:
    """Mask over src rows whose length is within `tolerance` (relative) of the
    ref median, OR inside the ref IQR (whichever is more inclusive)."""
    ref_med = float(np.median(ref_lengths))
    ref_q1 = float(np.percentile(ref_lengths, 25))
    ref_q3 = float(np.percentile(ref_lengths, 75))
    rel_lo = ref_med * (1 - tolerance)
    rel_hi = ref_med * (1 + tolerance)
    lo = min(rel_lo, ref_q1)
    hi = max(rel_hi, ref_q3)
    return (src_lengths >= lo) & (src_lengths <= hi)


def train_probe_balanced(X: np.ndarray, y: np.ndarray, seed: int = 0):
    """Train one LR probe on a balanced subsample of all of (X, y)."""
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n = min(len(pos), len(neg))
    if n == 0:
        return None, None
    idx = np.concatenate(
        [rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)]
    )
    Xs, ys = X[idx], y[idx]
    scaler = StandardScaler().fit(Xs)
    clf = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs", random_state=seed)
    clf.fit(scaler.transform(Xs), ys)
    return scaler, clf


def auroc_balanced(scaler, clf, X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n = min(len(pos), len(neg))
    if n == 0 or scaler is None or clf is None:
        return float("nan")
    idx = np.concatenate(
        [rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)]
    )
    Xs, ys = X[idx], y[idx]
    if len(np.unique(ys)) < 2:
        return float("nan")
    proba = clf.predict_proba(scaler.transform(Xs))[:, 1]
    return float(roc_auc_score(ys, proba))


def run_kind(cache_dir: str, layer: int, kind: str, sft_eval: str, outcome_eval: str):
    sft = load_cell(cache_dir, "C_SFT", layer, kind)
    out = load_cell(cache_dir, "C_outcome", layer, kind)
    if sft is None or out is None:
        return None
    sft_X, sft_y, sft_g, sft_meta = sft
    out_X, out_y, out_g, out_meta = out

    sft_lens = per_row_lengths(sft_meta, load_response_lengths(sft_eval))
    out_lens = per_row_lengths(out_meta, load_response_lengths(outcome_eval))

    # Filter SFT rows to lengths matching outcome's distribution.
    sft_to_out_mask = length_matched_mask(sft_lens, out_lens)
    out_to_sft_mask = length_matched_mask(out_lens, sft_lens)

    print(f"\n--- {kind} (L{layer}) ---")
    print(f"  C_SFT     mean response length: {sft_lens.mean():>7.0f}   (n={len(sft_lens)})")
    print(f"  C_outcome mean response length: {out_lens.mean():>7.0f}   (n={len(out_lens)})")
    print(f"  C_SFT rows surviving length-match -> outcome window: "
          f"{int(sft_to_out_mask.sum())}/{len(sft_to_out_mask)} "
          f"({100 * sft_to_out_mask.mean():.0f}%)")
    print(f"  C_outcome rows surviving length-match -> SFT window: "
          f"{int(out_to_sft_mask.sum())}/{len(out_to_sft_mask)} "
          f"({100 * out_to_sft_mask.mean():.0f}%)")

    # Train length-matched probes.
    sft_scaler, sft_clf = train_probe_balanced(sft_X[sft_to_out_mask], sft_y[sft_to_out_mask])
    out_scaler, out_clf = train_probe_balanced(out_X[out_to_sft_mask], out_y[out_to_sft_mask])

    # Diagonal (held-out CV on length-matched subsample).
    sft_diag = _bal_cv(sft_X[sft_to_out_mask], sft_y[sft_to_out_mask], sft_g[sft_to_out_mask])
    out_diag = _bal_cv(out_X[out_to_sft_mask], out_y[out_to_sft_mask], out_g[out_to_sft_mask])

    # Off-diagonals: trained-on-other-checkpoint-length-matched, eval-on-own.
    sft_on_out = auroc_balanced(sft_scaler, sft_clf, out_X, out_y)
    out_on_sft = auroc_balanced(out_scaler, out_clf, sft_X, sft_y)

    print(f"  diagonals (held-out CV, balanced):     "
          f"C_SFT={sft_diag:.3f}   C_outcome={out_diag:.3f}")
    print(f"  off-diagonal C_SFT_probe -> C_outcome: {sft_on_out:.3f}")
    print(f"  off-diagonal C_outcome_probe -> C_SFT: {out_on_sft:.3f}")

    return {
        "kind": kind, "layer": layer,
        "sft_mean_len": float(sft_lens.mean()),
        "out_mean_len": float(out_lens.mean()),
        "sft_to_out_fraction": float(sft_to_out_mask.mean()),
        "out_to_sft_fraction": float(out_to_sft_mask.mean()),
        "sft_diag": sft_diag, "out_diag": out_diag,
        "sft_on_out": sft_on_out, "out_on_sft": out_on_sft,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="extension/cache/probe_cache")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--sft_eval", default="eval_sft.json")
    parser.add_argument("--outcome_eval", default="eval.json")
    args = parser.parse_args()

    print("=" * 86)
    print("LENGTH-MATCHED cross-checkpoint transfer control")
    print("=" * 86)
    print("  Filtering each checkpoint's rollouts to lengths inside the other's")
    print("  IQR before training the cross-evaluated probe. If off-diagonals stay")
    print("  near chance, the original drift framing holds.")

    for kind in ("pre_answer", "assertion"):
        run_kind(args.cache_dir, args.layer, kind, args.sft_eval, args.outcome_eval)


if __name__ == "__main__":
    main()
