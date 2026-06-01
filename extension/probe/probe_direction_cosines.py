"""Cosine similarity between probe weight vectors across (ckpt, kind, layer).

The cross-position-transfer AUROCs (Phase 1, §2.3 in writeup) measure how
well a probe trained on position A predicts at position B. But low transfer
AUROC could mean (i) the weight vectors point in different directions, or
(ii) the weight magnitudes / activation scales differ enough to confuse
the LR sigmoid. This script computes the direct cosine similarity between
probe directions to disambiguate.

Trains one probe per (ckpt, kind, layer) on the FULL balanced subsample of
that cell (no held-out CV -- we want the most stable direction estimate).
Reports the cosine matrix:
  - within-ckpt cross-position: C_outcome pre vs C_outcome ass vs C_outcome neu, etc.
  - within-position cross-ckpt: C_SFT pre vs C_outcome pre, etc.
  - same-cell same-layer cosine = 1 (sanity)

Also reports the L2-norm of each probe vector (large norm = StandardScaler
shrunk the activations a lot at that position).
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

CACHE = "extension/cache/probe_cache_n500_clean406"
KINDS = ("pre_answer", "assertion", "neutral")
CKPTS = ("C_SFT", "C_outcome")


def load_cell(ckpt: str, layer: int, kind: str):
    npz = os.path.join(CACHE, f"{ckpt}_l{layer}_{kind}.npz")
    if not os.path.exists(npz):
        return None
    with np.load(npz) as d:
        return d["X"], d["y"]


def train_direction(X, y, seed=0):
    """Balanced subsample, fit LR, return (probe_weight_in_input_space, norm)."""
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    n = min(len(pos), len(neg))
    if n < 5: return None
    idx = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
    Xs, ys = X[idx], y[idx]
    scaler = StandardScaler().fit(Xs)
    Xz = scaler.transform(Xs)
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(Xz, ys)
    # Weight vector lives in STANDARDIZED space. To get the direction in
    # the original input space (so cosines across cells are comparable),
    # multiply componentwise by 1/scaler.scale_ (since z = (x - mean) / scale,
    # w·z = (w/scale)·(x - mean) ).
    w_input = clf.coef_[0] / scaler.scale_
    return w_input, float(np.linalg.norm(w_input))


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    # Train one probe per (ckpt, layer, kind)
    probes: dict[tuple[str, int, str], tuple[np.ndarray, float]] = {}
    for layer in (12, 16, 20):
        for ckpt in CKPTS:
            for kind in KINDS:
                c = load_cell(ckpt, layer, kind)
                if c is None: continue
                res = train_direction(*c, seed=0)
                if res is None: continue
                probes[(ckpt, layer, kind)] = res

    lines = []
    lines.append("Cosine similarity between probe direction vectors (input space)")
    lines.append(f"  cache: {CACHE}; trained on balanced subsample per cell")
    lines.append("")

    # Within-ckpt cross-position at each layer
    lines.append("=" * 70)
    lines.append("WITHIN-CHECKPOINT cross-position cosines")
    lines.append("=" * 70)
    for ckpt in CKPTS:
        lines.append(f"\n{ckpt}:")
        for layer in (12, 16, 20):
            avail = [k for k in KINDS if (ckpt, layer, k) in probes]
            if len(avail) < 2: continue
            lines.append(f"  L{layer}:")
            for a in avail:
                for b in avail:
                    if a >= b: continue
                    wa, _ = probes[(ckpt, layer, a)]
                    wb, _ = probes[(ckpt, layer, b)]
                    lines.append(f"    cos(<{a}> , <{b}>) = {cos(wa, wb):+.3f}")

    # Cross-checkpoint within-position
    lines.append("\n" + "=" * 70)
    lines.append("CROSS-CHECKPOINT within-position cosines (C_SFT vs C_outcome)")
    lines.append("=" * 70)
    for layer in (12, 16, 20):
        lines.append(f"\nL{layer}:")
        for kind in KINDS:
            if (("C_SFT", layer, kind) in probes) and (("C_outcome", layer, kind) in probes):
                w_sft, _ = probes[("C_SFT", layer, kind)]
                w_out, _ = probes[("C_outcome", layer, kind)]
                lines.append(f"  cos(C_SFT-{kind}, C_outcome-{kind}) = {cos(w_sft, w_out):+.3f}")

    # Probe-vector norms (input space)
    lines.append("\n" + "=" * 70)
    lines.append("Probe vector norms (input space)")
    lines.append("=" * 70)
    lines.append(f"  {'cell':<32} {'norm':>10}")
    for k in sorted(probes.keys()):
        _, n = probes[k]
        ckpt, layer, kind = k
        lines.append(f"  {ckpt+' L'+str(layer)+' '+kind:<32} {n:>10.2f}")

    # Now the headline summary
    lines.append("\n" + "=" * 70)
    lines.append("HEADLINE")
    lines.append("=" * 70)
    if all((c, 16, k) in probes for c in CKPTS for k in ("pre_answer", "assertion")):
        cos_sft_pa = cos(probes[("C_SFT", 16, "pre_answer")][0], probes[("C_SFT", 16, "assertion")][0])
        cos_out_pa = cos(probes[("C_outcome", 16, "pre_answer")][0], probes[("C_outcome", 16, "assertion")][0])
        cos_pre_cross = cos(probes[("C_SFT", 16, "pre_answer")][0], probes[("C_outcome", 16, "pre_answer")][0])
        cos_ass_cross = cos(probes[("C_SFT", 16, "assertion")][0], probes[("C_outcome", 16, "assertion")][0])
        lines.append(f"cos(C_SFT pre, C_SFT ass) at L16 = {cos_sft_pa:+.3f}")
        lines.append(f"cos(C_outcome pre, C_outcome ass) at L16 = {cos_out_pa:+.3f}")
        lines.append(f"  ratio: change in within-ckpt cross-position cosine = {cos_out_pa - cos_sft_pa:+.3f}")
        lines.append("")
        lines.append(f"cos(C_SFT pre, C_outcome pre) at L16 = {cos_pre_cross:+.3f}")
        lines.append(f"cos(C_SFT ass, C_outcome ass) at L16 = {cos_ass_cross:+.3f}")

    txt = "\n".join(lines)
    print(txt)
    os.makedirs("extension/outputs/n500/text", exist_ok=True)
    with open("extension/outputs/n500/text/21_probe_cosines.txt", "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
