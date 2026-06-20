"""Angular reprojection error and two-view bundle refinement (Tier-1 C5).

The eight-point + cheirality estimate (C1/C2) is *algebraic*; a nonlinear refinement that
minimizes the **geometric** reprojection error tightens it. For a wide-FOV camera that error must
be measured as an **angle on the sphere**, not a pixel distance: pixel error is anisotropic and
ill-defined past 90°, whereas the angle between the observed ray and the predicted ray is the
honest, model-free residual everywhere. This refines relative pose **and** structure together by
least-squares on the tangent-plane residual. Implements unit **C5** of the Tier-1 spec.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import least_squares

from ..core.lie import so3_exp
from .two_view import _as_rays


def _tangent_basis(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the tangent plane at each unit ray ``f`` (N,3)."""
    helper = np.tile(np.array([0.0, 1.0, 0.0]), (f.shape[0], 1))
    near = np.abs(f @ np.array([0.0, 1.0, 0.0])) > 0.9
    helper[near] = np.array([1.0, 0.0, 0.0])
    e_u = np.cross(helper, f)
    e_u /= np.linalg.norm(e_u, axis=1, keepdims=True)
    e_v = np.cross(f, e_u)
    return e_u, e_v


def angular_reprojection_error(f1: np.ndarray, f2: np.ndarray,
                               R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Per-point reprojection error in **degrees**: max of the two view angles.

    View 1 predicts direction ``X``; view 2 predicts ``R X + t``. Each is compared (as an angle)
    to the observed ray.
    """
    f1, f2 = _as_rays(f1), _as_rays(f2)
    R = np.asarray(R, float)
    t = np.asarray(t, float).reshape(3)
    d1 = X / np.linalg.norm(X, axis=1, keepdims=True)
    d2 = X @ R.T + t
    d2 /= np.linalg.norm(d2, axis=1, keepdims=True)
    a1 = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", d1, f1), -1, 1)))
    a2 = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", d2, f2), -1, 1)))
    return np.maximum(a1, a2)


def refine_two_view(f1: np.ndarray, f2: np.ndarray,
                    R0: np.ndarray, t0: np.ndarray, X0: np.ndarray, *,
                    max_nfev: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nonlinear refinement of ``(R, t, X)`` minimizing the tangent-plane angular residual.

    **Manifold-correct.** The rotation update is a *local perturbation* retracted through the
    exponential map, ``R = R₀ · exp([δω]_×)`` with ``δω`` optimized from ``0`` — so it never goes
    near the ``‖r‖ = π`` axis-angle singularity, unlike optimizing an absolute axis-angle vector.
    Camera 1 is fixed at the identity (reference frame) and ``t`` is kept unit-length, pinning the
    7-DOF similarity gauge (the angular error is scale-invariant). Returns the refined ``(R, t, X)``.
    """
    f1, f2 = _as_rays(f1), _as_rays(f2)
    n = f1.shape[0]
    eu1, ev1 = _tangent_basis(f1)
    eu2, ev2 = _tangent_basis(f2)
    R0 = np.asarray(R0, float)
    t0 = np.asarray(t0, float).reshape(3)
    t0 = t0 / np.linalg.norm(t0)
    X0 = np.asarray(X0, float)

    def unpack(p):
        R = R0 @ so3_exp(p[:3])                          # retract: base ∘ exp(δω)
        t = t0 + p[3:6]
        t = t / np.linalg.norm(t)
        X = X0 + p[6:].reshape(n, 3)
        return R, t, X

    def residual(p):
        R, t, X = unpack(p)
        d1 = X / np.linalg.norm(X, axis=1, keepdims=True)
        d2 = X @ R.T + t
        d2 /= np.linalg.norm(d2, axis=1, keepdims=True)
        r1 = np.stack([np.einsum("ij,ij->i", d1, eu1), np.einsum("ij,ij->i", d1, ev1)], 1)
        r2 = np.stack([np.einsum("ij,ij->i", d2, eu2), np.einsum("ij,ij->i", d2, ev2)], 1)
        return np.concatenate([r1.ravel(), r2.ravel()])

    p0 = np.zeros(6 + 3 * n)                              # perturbation starts at 0
    sol = least_squares(residual, p0, method="lm", max_nfev=max_nfev)
    return unpack(sol.x)
