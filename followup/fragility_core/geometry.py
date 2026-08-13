"""Representation geometry: drift, separation, CKA, distributedness.

Metric definitions follow followup/CLAUDE.md exactly.

  dir_drift(t)   cosine between the t=0 probe direction and the checkpoint-t
                 retrained linear probe direction. Reported in two spaces
                 because "after whitening" is basis-dependent and both readings
                 are cheap:
                   dir_drift_input     cos in raw activation space
                   dir_drift_whitened  cos after scaling each coordinate by
                                       checkpoint-t's per-dim activation std
                 The whitened version is the headline (it is the space the
                 logistic probe actually separates in); both are always stored.

  sep(t)         standardised separation of the projection onto the FIXED t=0
                 probe direction: (mean_correct - mean_wrong) / pooled_std.
                 Unlike AUROC this is signed and unbounded, so it distinguishes
                 "information gone" from "information inverted".

  CKA            linear centered-kernel alignment between checkpoint-t and
                 checkpoint-0 activations on the SAME rows. Only defined when
                 rows correspond, i.e. in the fixed-text arm. Callers must not
                 compute it across arms where rollouts were resampled; the
                 function raises on a row-count mismatch instead of guessing.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# directions
# ---------------------------------------------------------------------------


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = unit(a), unit(b)
    if a.size != b.size:
        raise ValueError(f"dimension mismatch: {a.size} vs {b.size}")
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return float("nan")
    return float(np.clip(a @ b, -1.0, 1.0))


def dir_drift(
    v0: np.ndarray,
    vt: np.ndarray,
    *,
    act_std: np.ndarray | None = None,
) -> dict[str, float]:
    """Cosine between two probe directions, in input space and whitened space.

    `act_std` is checkpoint-t's per-dimension activation std. A direction w acts
    on whitened coordinates as w * sigma (score = (w*sigma) . ((x-mu)/sigma)), so
    scaling by sigma is what maps input-space directions into the whitened basis.
    """
    out = {"dir_drift_input": cosine(v0, vt), "dir_drift_whitened": float("nan")}
    if act_std is not None:
        s = np.asarray(act_std, dtype=np.float64).ravel()
        if s.size != np.asarray(v0).size:
            raise ValueError("act_std dimension does not match probe direction")
        # A dead dimension (zero variance) must get ZERO weight in the whitened
        # cosine. Imputing sigma=1 gave it FULL weight — backwards, and it would
        # bite on any sparse basis (e.g. SAE features).
        s = np.where(s > 0, s, 0.0)
        out["dir_drift_whitened"] = cosine(np.asarray(v0).ravel() * s, np.asarray(vt).ravel() * s)
    return out


def participation_ratio(w: np.ndarray) -> float:
    """Effective number of coordinates a direction spreads over.

    PR = (sum w_i^2)^2 / sum w_i^4. Ranges from 1 (a single coordinate carries
    everything) to d (perfectly distributed). A pre-hoc H3 feature: the
    hypothesis is that highly concentrated probes are easier to evade.
    """
    w = np.asarray(w, dtype=np.float64).ravel()
    s2 = float(np.sum(w**2))
    s4 = float(np.sum(w**4))
    if s4 == 0:
        return float("nan")
    return (s2**2) / s4


# ---------------------------------------------------------------------------
# separation along a fixed direction
# ---------------------------------------------------------------------------


def projections(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Signed projection of each row onto the unit-normalised direction v."""
    return np.asarray(X, dtype=np.float64) @ unit(v)


def sep(X: np.ndarray, y: np.ndarray, v: np.ndarray) -> dict[str, float]:
    """sep(t) plus the pieces it is built from (so a null can be diagnosed)."""
    p = projections(X, v)
    y = np.asarray(y).astype(int)
    pc, pw = p[y == 1], p[y == 0]
    if len(pc) < 2 or len(pw) < 2:
        return {
            "sep": float("nan"),
            "proj_mean_correct": float("nan"),
            "proj_mean_wrong": float("nan"),
            "proj_pooled_std": float("nan"),
        }
    n1, n0 = len(pc), len(pw)
    pooled = np.sqrt(((n1 - 1) * pc.var(ddof=1) + (n0 - 1) * pw.var(ddof=1)) / (n1 + n0 - 2))
    return {
        "sep": float((pc.mean() - pw.mean()) / pooled) if pooled > 0 else float("nan"),
        "proj_mean_correct": float(pc.mean()),
        "proj_mean_wrong": float(pw.mean()),
        "proj_pooled_std": float(pooled),
    }


def projection_variance_fraction(X: np.ndarray, v: np.ndarray) -> float:
    """Share of total activation variance lying along v. Is the probe axis loud?"""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    total = float(np.sum(Xc**2))
    if total == 0:
        return float("nan")
    along = float(np.sum((Xc @ unit(v)) ** 2))
    return along / total


def activation_stats(X: np.ndarray) -> dict[str, float]:
    """Global scale descriptors, to separate "probe axis moved" from "everything grew"."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    return {
        "act_norm_mean": float(np.mean(np.linalg.norm(X, axis=1))),
        "act_std_mean": float(np.mean(X.std(axis=0))),
        "act_total_var": float(np.sum(Xc**2) / max(len(X) - 1, 1)),
        "act_effective_rank": effective_rank(Xc),
    }


def per_dim_std(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64).std(axis=0)


def effective_rank(Xc: np.ndarray) -> float:
    """exp(entropy of the normalised eigenspectrum) of the covariance."""
    Xc = np.asarray(Xc, dtype=np.float64)
    if Xc.shape[0] < 2:
        return float("nan")
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = s**2
    tot = ev.sum()
    if tot <= 0:
        return float("nan")
    p = ev / tot
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


# ---------------------------------------------------------------------------
# representational similarity
# ---------------------------------------------------------------------------


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear centered-kernel alignment. Requires ROW-CORRESPONDING inputs.

    1.0 = identical representational geometry up to rotation/scale; 0.0 = none
    shared. Raises on a row mismatch rather than silently aligning by truncation,
    because a wrong pairing produces a plausible-looking but meaningless number.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"CKA needs row correspondence: got {X.shape[0]} and {Y.shape[0]} rows. "
            "CKA is only defined in the fixed-text arm."
        )
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = float(np.linalg.norm(Y.T @ X, ord="fro") ** 2)
    den = float(
        np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    )
    if den == 0:
        return float("nan")
    return num / den


def subspace_principal_angles(A: np.ndarray, B: np.ndarray, k: int = 10) -> np.ndarray:
    """Cosines of principal angles between the top-k PCs of two activation sets.

    Complements CKA: CKA is a single scalar over the whole geometry, this shows
    whether the leading subspaces coincide. Does NOT require row correspondence.
    """
    def top_pcs(M: np.ndarray, k: int) -> np.ndarray:
        M = np.asarray(M, dtype=np.float64)
        M = M - M.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(M, full_matrices=False)
        return vt[:k].T

    Qa = np.linalg.qr(top_pcs(A, k))[0]
    Qb = np.linalg.qr(top_pcs(B, k))[0]
    return np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0.0, 1.0)
