"""Two-view geometry on unit bearing vectors — essential matrix, pose, triangulation.

Conventions
-----------
- ``f1, f2`` are **unit bearing vectors** (rays), shape ``(N, 3)``, one correspondence per
  row: ``f1[i]`` in camera 1, ``f2[i]`` in camera 2, looking at the same 3D point.
- Relative pose ``(R, t)`` maps a point from camera 1 to camera 2: ``X2 = R @ X1 + t``.
  ``t`` is recovered only up to scale, so it is returned **unit-length**; its sign is fixed
  by cheirality (the point must lie in front of both cameras).
- "In front" for a wide-FOV camera means **positive depth along the bearing vector**
  (``λ > 0``), not ``z > 0`` — a ray past 90° is still a valid observation.

The calibrated epipolar constraint is ``f2ᵀ E f1 = 0`` with ``E = [t]_× R`` (rank 2). It
holds for bearing vectors of any central camera, which is why nothing here is pinhole-specific.

Derivations, proofs, and numerical-stability notes: ``docs/research/mvg_two_view_geometry.md``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

_W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _as_rays(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"rays must have shape (N, 3), got {f.shape}")
    n = np.linalg.norm(f, axis=1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("zero-length bearing vector")
    return f / n


def epipolar_residual(E: np.ndarray, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """Algebraic epipolar residual ``f2ᵀ E f1`` per correspondence, shape ``(N,)``.

    Zero (to numerical precision) for a perfect correspondence + exact ``E``.
    """
    f1 = _as_rays(f1)
    f2 = _as_rays(f2)
    return np.einsum("ij,jk,ik->i", f2, np.asarray(E, float), f1)


def essential_from_rays(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """Essential matrix from ≥8 ray correspondences (eight-point on bearing vectors).

    Solves ``f2ᵀ E f1 = 0`` in the least-squares (smallest-singular-vector) sense, then
    projects onto the essential manifold (singular values forced to ``(1, 1, 0)``).

    Note: this is the plain eight-point on rays. For noisy 360°-FOV data a spherical
    pre-conditioning (Robust 360-8PA) tightens the estimate — a planned refinement (C2);
    pixel-domain Hartley normalization does **not** apply to bearing vectors.
    """
    f1 = _as_rays(f1)
    f2 = _as_rays(f2)
    if f1.shape[0] < 8:
        raise ValueError(f"need ≥8 correspondences, got {f1.shape[0]}")
    # Each row: coefficients of vec(E) (row-major) in f2ᵀ E f1 = 0  →  kron(f2, f1).
    A = f2[:, [0, 0, 0, 1, 1, 1, 2, 2, 2]] * np.tile(f1, 3)
    _, _, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)
    # Project onto the essential manifold: equal non-zero singular values, rank 2.
    U, _, Vt2 = np.linalg.svd(E)
    return U @ np.diag([1.0, 1.0, 0.0]) @ Vt2


def decompose_essential(E: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """The four ``(R, t)`` candidates consistent with an essential matrix.

    ``t`` is unit-length (translation is scale-free). Cheirality (``recover_pose``) selects
    the one physical solution among the four.
    """
    U, _, Vt = np.linalg.svd(np.asarray(E, float))
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    R1 = U @ _W @ Vt
    R2 = U @ _W.T @ Vt
    t = U[:, 2]
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def triangulate_rays(f1: np.ndarray, f2: np.ndarray, R: np.ndarray, t: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Midpoint triangulation of ray pairs under pose ``(R, t)``.

    Returns ``(X, depth1, depth2)``: 3D points in **camera-1** frame ``(N, 3)``, and the
    signed depth of each point along ``f1`` and along ``f2``. Positive depths ⇒ the point is
    in front of both cameras.
    """
    f1 = _as_rays(f1)
    f2 = _as_rays(f2)
    R = np.asarray(R, float)
    t = np.asarray(t, float).reshape(3)
    # Both rays expressed in camera-1 frame.
    c2 = -R.T @ t                       # camera-2 centre in camera-1 frame
    d2 = f2 @ R                         # = (R.T @ f2.T).T, ray-2 directions in camera-1 frame
    w0 = -c2                            # o1 - o2, with o1 = 0
    b = np.einsum("ij,ij->i", f1, d2)   # f1·d2  (a = f1·f1 = 1, c = d2·d2 = 1)
    d = f1 @ w0                          # f1·w0
    e = d2 @ w0                          # d2·w0
    denom = 1.0 - b * b                  # a*c - b^2
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    lam1 = (b * e - d) / denom           # (b*e - c*d)/denom, c = 1
    lam2 = (e - b * d) / denom           # (a*e - b*d)/denom, a = 1
    P1 = lam1[:, None] * f1
    P2 = c2[None, :] + lam2[:, None] * d2
    X = 0.5 * (P1 + P2)
    return X, lam1, lam2


def recover_pose(f1: np.ndarray, f2: np.ndarray, E: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relative pose ``(R, t)`` and triangulated points from ray correspondences.

    Computes ``E`` (if not supplied), enumerates the four decompositions, and picks the one
    with the most points in front of **both** cameras (ray cheirality). Returns
    ``(R, t, X)`` with ``t`` unit-length and ``X`` the triangulated points (camera-1 frame).
    """
    f1 = _as_rays(f1)
    f2 = _as_rays(f2)
    if E is None:
        E = essential_from_rays(f1, f2)
    best = None
    for R, t in decompose_essential(E):
        _, d1, d2 = triangulate_rays(f1, f2, R, t)
        n_front = int(np.sum((d1 > 0) & (d2 > 0)))
        if best is None or n_front > best[0]:
            best = (n_front, R, t)
    _, R, t = best
    X, _, _ = triangulate_rays(f1, f2, R, t)
    return R, t, X


def relative_pose(f1: np.ndarray, f2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience: ``(R, t)`` from ray correspondences (eight-point + cheirality)."""
    R, t, _ = recover_pose(f1, f2)
    return R, t
