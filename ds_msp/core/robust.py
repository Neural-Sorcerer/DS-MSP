r"""Robust M-estimation kernels + auto-scale + graduated non-convexity (pure NumPy).

These are the *outlier-robustness* primitives the in-house manifold solver
(:mod:`ds_msp.core.optimize`) uses. The cost convention matches the solver and is
the standard re-weighted least squares::

    f(θ) = Σ_i  w_i · ρ_c(s_i),      s_i = ‖r_i‖²   (squared block residual)

with per-block IRLS weight ``ω(s) = 2·ρ'(s)`` (dimensionless, ``ω ≡ 1`` for L2),
so the reweighted normal equations are ``(Jᵀ W̃ J + λD) δ = −Jᵀ W̃ r`` with
``W̃ = diag(w_i · ω(s_i))`` repeated over the block's rows.

A *block* is the group of residual components that share one robustness decision —
2 for an image point ``(u, v)``, but the caller declares it (two-view stacks 4 per
3-D point: two views × two tangent components).

The differentiability-only pieces (the Triggs ``ω'`` curvature term, learnable
Barron α) are intentionally omitted — this is a forward-only solver.
"""

from __future__ import annotations

import numpy as np

VALID_KERNELS = ("none", "huber", "pseudo_huber", "cauchy", "geman_mcclure", "barron")

_BARRON_EPS = 1e-5


def _barron_bd(alpha: float) -> tuple[float, float]:
    """Singularity-guarded ``(b, d)`` from Barron, CVPR 2019, Appendix B."""
    b = abs(alpha - 2.0) + _BARRON_EPS
    d = alpha + _BARRON_EPS if alpha >= 0 else alpha - _BARRON_EPS
    return b, d

#: Fisher-consistency constant for the MAD of 2-D residual NORMS. ‖r_i‖ is
#: Rayleigh(σ) under isotropic N(0, σ²I₂) noise, whose MAD is 0.448453·σ, so the
#: consistent multiplier is 1/0.448453. The familiar 1.4826 is the *Gaussian*
#: constant and under-estimates σ by ~33% on 2-vector residual norms.
KAPPA_RAYLEIGH = 2.2298876546890014

#: 95%-efficiency tuning: kernel scale ``c = TUNING[kernel] · σ̂``.
KERNEL_TUNING = {
    "none": 1.0, "huber": 1.345, "pseudo_huber": 1.345,
    "cauchy": 2.385, "geman_mcclure": 3.0, "barron": 2.385,
}


def robust_cost(s: np.ndarray, kernel: str, scale: float, alpha: float = -2.0) -> np.ndarray:
    """Per-block robust cost ``ρ(s)``; ``s`` is the squared block residual (px²).

    ``alpha`` is the Barron shape (only used by ``kernel='barron'``): ``α=2`` → L2,
    ``α=1`` → pseudo-Huber, ``α=0`` → Cauchy, ``α=−2`` → Geman-McClure, ``α→−∞`` → Welsch.
    """
    if kernel == "none":
        return 0.5 * s
    c2 = scale * scale
    if kernel == "huber":
        root = np.sqrt(np.maximum(s, 1e-24))
        return np.where(s <= c2, 0.5 * s, scale * root - 0.5 * c2)
    if kernel == "pseudo_huber":                       # C∞ Huber
        return c2 * (np.sqrt(1.0 + s / c2) - 1.0)
    if kernel == "cauchy":
        return 0.5 * c2 * np.log1p(s / c2)
    if kernel == "geman_mcclure":
        return 0.5 * s / (1.0 + s / c2)
    if kernel == "barron":
        b, d = _barron_bd(alpha)
        return (c2 * b / d) * (np.power(s / (c2 * b) + 1.0, 0.5 * d) - 1.0)
    raise ValueError(f"kernel must be one of {VALID_KERNELS!r}, got {kernel!r}")


def robust_weight(s: np.ndarray, kernel: str, scale: float, alpha: float = -2.0) -> np.ndarray:
    """IRLS weight ``ω(s) = 2·ρ'(s)`` — dimensionless, ``ω ≡ 1`` for L2."""
    if kernel == "none":
        return np.ones_like(s)
    c2 = scale * scale
    if kernel == "huber":
        root = np.sqrt(np.maximum(s, 1e-24))
        return np.where(s <= c2, 1.0, scale / root)
    if kernel == "pseudo_huber":
        return 1.0 / np.sqrt(1.0 + s / c2)
    if kernel == "cauchy":
        return 1.0 / (1.0 + s / c2)
    if kernel == "geman_mcclure":
        inv = 1.0 / (1.0 + s / c2)
        return inv * inv
    if kernel == "barron":
        b, d = _barron_bd(alpha)
        return np.power(s / (c2 * b) + 1.0, 0.5 * d - 1.0)
    raise ValueError(f"kernel must be one of {VALID_KERNELS!r}, got {kernel!r}")


def mad_scale(block_norms: np.ndarray, floor: float = 0.3,
              kappa: float = KAPPA_RAYLEIGH) -> float:
    r"""Robust inlier-σ estimate ``σ̂ = max(κ · MAD(‖r_i‖), floor)``.

    50%-breakdown, Fisher-consistent for the per-component inlier noise std of a
    2-D residual. Re-estimated each IRLS iteration this is Huber's "Proposal 2"
    alternation. ``block_norms`` are the per-block residual norms ``‖r_i‖``.
    """
    med = np.median(block_norms)
    mad = np.median(np.abs(block_norms - med))
    return float(max(kappa * mad, floor))


def auto_kernel_scale(block_norms: np.ndarray, kernel: str,
                      floor: float = 0.3) -> float:
    """Effective auto kernel scale ``c = TUNING[kernel] · σ̂(r)``."""
    return KERNEL_TUNING.get(kernel, 2.385) * mad_scale(block_norms, floor=floor)


def gnc_scale(iteration: int, gnc_iters: int,
              scale_start: float, scale_end: float) -> float:
    """Graduated-non-convexity schedule: geometric decay of the kernel scale from
    ``scale_start`` to ``scale_end`` over ``gnc_iters`` iterations, then held.

    A wide kernel early makes spurious minima (e.g. the ~180°-flipped basin)
    vanish — their gradient pulls the iterate to the global basin; by the time the
    kernel sharpens to the calibrated scale the iterate is already in it.
    """
    if gnc_iters <= 0 or scale_start <= 0:
        return scale_end
    t = min(iteration / gnc_iters, 1.0)
    return float(scale_end * (scale_start / scale_end) ** (1.0 - t))


#: Floor on the ``(I − H_ii)`` eigenvalues — caps the studentization inflation of an
#: extreme-leverage block at ``1/eps`` (default 20x), a standard robust-IRLS safeguard.
STUDENT_EPS = 0.05


def studentized_sq(J: np.ndarray, r: np.ndarray, *, block: int = 2,
                   weights: np.ndarray | None = None, eps: float = STUDENT_EPS) -> np.ndarray:
    r"""Bounded-influence (Mallows-type) studentized squared residual per block.

    A high-leverage point can pull the fit toward itself so its *own* residual stays small
    — a residual-only kernel never sees it ("self-masking" leverage outlier). The hat block
    ``H_ii = w_i·J_i (JᵀWJ)⁻¹ J_iᵀ`` measures that influence; deflating by
    ``M_i = I − H_ii + εI`` and forming ``s̃_i = r_iᵀ M_i⁻¹ r_i`` inflates the residual of
    leverage points so the kernel down-weights them. Returns ``(n_blocks,)`` squared
    residuals to feed :func:`robust_weight` / :func:`robust_cost` in place of ``‖r_i‖²``.

    ``J`` is ``(n_rows, p)``, ``r`` is ``(n_rows,)``, rows grouped in consecutive ``block``s
    (2 for an image point). ``weights`` are per-block user weights (the kernel's ω is
    deliberately excluded so a down-weighted outlier cannot launder its own leverage).
    """
    n_rows, p = J.shape
    n = n_rows // block
    Jp = J.reshape(n, block, p)
    rp = r.reshape(n, block)
    if weights is not None:
        w = np.asarray(weights, float).reshape(n, 1, 1)
        H = np.einsum("nki,nkj->ij", Jp * w, Jp)
    else:
        H = np.einsum("nki,nkj->ij", Jp, Jp)
    tr = np.trace(H)
    Hinv = np.linalg.inv(H + (1e-10 * tr / max(p, 1)) * np.eye(p))
    Hii = np.einsum("nik,kl,njl->nij", Jp, Hinv, Jp)          # (n, block, block)
    if weights is not None:
        Hii = Hii * np.asarray(weights, float).reshape(n, 1, 1)
    eye = np.eye(block)
    M = eye - Hii + eps * eye
    M = 0.5 * (M + M.transpose(0, 2, 1))
    sol = np.linalg.solve(M, rp[:, :, None])[:, :, 0]        # M_i⁻¹ r_i
    return np.einsum("nk,nk->n", rp, sol)
