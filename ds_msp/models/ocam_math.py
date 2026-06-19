"""
Pure OCamCalib / Scaramuzza omnidirectional polynomial-camera math.

Parameters (10): center (cx, cy), affine stretch A = [[c, d], [e, 1]], and the
degree-4 "world" polynomial w(rho) = a0 + a1*rho + a2*rho^2 + a3*rho^3 + a4*rho^4.

Unprojection (image -> ray, direct):
    [x; y] = A^{-1} ([u; v] - [cx; cy]),   rho = sqrt(x^2 + y^2)
    ray = normalize([x, y, -w(rho)])        (so the optical centre maps to +Z)

Projection (ray -> image, solve a polynomial):
    norm = sqrt(X^2 + Y^2),  m = Z / norm
    find rho > 0 with   w(rho) + m*rho = 0   (Newton)
    [u; v] = A [rho*X/norm; rho*Y/norm] + [cx; cy]

Analytic Jacobians use the implicit function theorem for d(rho).
Self-contained: numpy only (enforced by the independence gate).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# Polynomial argument is normalized by R_REF so coefficients are O(1) — this is
# the standard OCamCalib conditioning trick and keeps both the optimizer and the
# finite-difference gradient check well-behaved.
R_REF = 100.0


def _polyw(rho, a0, a1, a2, a3, a4):
    rn = rho / R_REF
    return a0 + rn*(a1 + rn*(a2 + rn*(a3 + rn*a4)))


def _dpolyw(rho, a1, a2, a3, a4):
    """d w(rho) / d rho (note the 1/R_REF from the chain rule)."""
    rn = rho / R_REF
    return (a1 + rn*(2*a2 + rn*(3*a3 + rn*4*a4))) / R_REF


def _solve_rho(m, a0, a1, a2, a3, a4):
    """Solve w(rho) + m*rho = 0 for rho > 0 by Newton; returns (rho, converged)."""
    denom = a1 / R_REF + m
    rho = np.where(np.abs(denom) > 1e-9, -a0 / np.where(denom == 0, 1e-9, denom), np.abs(a0))
    rho = np.abs(rho)
    for _ in range(30):
        F = _polyw(rho, a0, a1, a2, a3, a4) + m * rho
        Fr = _dpolyw(rho, a1, a2, a3, a4) + m
        rho = rho - F / np.where(np.abs(Fr) < 1e-12, 1e-12, Fr)
        rho = np.maximum(rho, 0.0)
    F = _polyw(rho, a0, a1, a2, a3, a4) + m * rho
    converged = np.abs(F) < 1e-6
    return rho, converged


def ocam_project(points_3d, cx, cy, c, d, e, a0, a1, a2, a3, a4
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame points. Returns ``(u, v, valid)``."""
    X, Y, Z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]
    norm = np.sqrt(X*X + Y*Y)
    onaxis = norm < 1e-12
    normsafe = np.where(onaxis, 1.0, norm)
    m = Z / normsafe
    rho, conv = _solve_rho(m, a0, a1, a2, a3, a4)
    s = X / normsafe
    t = Y / normsafe
    ix = rho * s
    iy = rho * t
    u = c * ix + d * iy + cx
    v = e * ix + iy + cy
    # on-axis -> centre
    u = np.where(onaxis, cx, u)
    v = np.where(onaxis, cy, v)
    valid = conv | onaxis
    return u, v, valid


def ocam_unproject(points_2d, cx, cy, c, d, e, a0, a1, a2, a3, a4
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject pixels to unit rays (direct polynomial). Returns ``(rays, valid)``."""
    du = points_2d[..., 0] - cx
    dv = points_2d[..., 1] - cy
    det = c * 1.0 - d * e
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    x = (du - d * dv) / det
    y = (-e * du + c * dv) / det
    rho = np.sqrt(x*x + y*y)
    w = _polyw(rho, a0, a1, a2, a3, a4)
    ray = np.stack([x, y, -w], axis=-1)
    ray = ray / np.maximum(np.linalg.norm(ray, axis=-1, keepdims=True), 1e-12)
    valid = np.isfinite(ray).all(axis=-1)
    ray[~valid] = 0.0
    return ray, valid


def ocam_project_jacobian(points_3d, cx, cy, c, d, e, a0, a1, a2, a3, a4
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analytic Jacobian. Returns ``(u, v, J_point(N,2,3), J_param(N,2,10), valid)``.

    Parameter order: ``(cx, cy, c, d, e, a0, a1, a2, a3, a4)``.
    """
    X, Y, Z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]
    norm = np.sqrt(X*X + Y*Y)
    onaxis = norm < 1e-12
    normsafe = np.where(onaxis, 1.0, norm)
    m = Z / normsafe
    rho, conv = _solve_rho(m, a0, a1, a2, a3, a4)
    s = X / normsafe
    t = Y / normsafe
    ix = rho * s
    iy = rho * t
    u = c * ix + d * iy + cx
    v = e * ix + iy + cy

    Fr = _dpolyw(rho, a1, a2, a3, a4) + m            # dF/drho

    # --- J_param (cx,cy,c,d,e,a0,a1,a2,a3,a4) ---
    shape = points_3d.shape[:-1]
    J_param = np.zeros(shape + (2, 10), dtype=np.float64)
    J_param[..., 0, 0] = 1.0                          # du/dcx
    J_param[..., 1, 1] = 1.0                          # dv/dcy
    J_param[..., 0, 2] = ix                           # du/dc
    J_param[..., 0, 3] = iy                           # du/dd
    J_param[..., 1, 4] = ix                           # dv/de
    # d rho / d a_k = -rho^k / Fr
    cu = c * s + d * t                                # du/d(rho)
    cv = e * s + t                                    # dv/d(rho)
    rn = rho / R_REF
    powers = [np.ones_like(rho), rn, rn**2, rn**3, rn**4]  # dF/da_k = (rho/R)^k
    for k in range(5):
        drho = -powers[k] / np.where(np.abs(Fr) < 1e-12, 1e-12, Fr)
        J_param[..., 0, 5 + k] = cu * drho
        J_param[..., 1, 5 + k] = cv * drho

    # --- J_point (X, Y, Z) ---
    n3 = normsafe**3
    ds_dX = Y*Y / n3
    ds_dY = -X*Y / n3
    dt_dX = -X*Y / n3
    dt_dY = X*X / n3
    dm_dX = -Z*X / n3
    dm_dY = -Z*Y / n3
    dm_dZ = 1.0 / normsafe
    # rho depends on point through m: drho/dpoint = -rho/Fr * dm/dpoint
    Frs = np.where(np.abs(Fr) < 1e-12, 1e-12, Fr)
    drho_dX = -rho / Frs * dm_dX
    drho_dY = -rho / Frs * dm_dY
    drho_dZ = -rho / Frs * dm_dZ

    dix_dX = drho_dX * s + rho * ds_dX
    dix_dY = drho_dY * s + rho * ds_dY
    dix_dZ = drho_dZ * s
    diy_dX = drho_dX * t + rho * dt_dX
    diy_dY = drho_dY * t + rho * dt_dY
    diy_dZ = drho_dZ * t

    J_point = np.zeros(shape + (2, 3), dtype=np.float64)
    J_point[..., 0, 0] = c * dix_dX + d * diy_dX
    J_point[..., 0, 1] = c * dix_dY + d * diy_dY
    J_point[..., 0, 2] = c * dix_dZ + d * diy_dZ
    J_point[..., 1, 0] = e * dix_dX + diy_dX
    J_point[..., 1, 1] = e * dix_dY + diy_dY
    J_point[..., 1, 2] = e * dix_dZ + diy_dZ

    valid = conv | onaxis
    return u, v, J_point, J_param, valid
