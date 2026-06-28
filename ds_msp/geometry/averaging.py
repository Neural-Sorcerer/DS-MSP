"""Robust averaging of noisy relative poses — Markley SVD quaternion rotation
averaging (``getAverageRotation``, geometrytools.cpp:868) and component-wise median
translation (``initInterTransform``, McCalib.cpp:843).

Quaternion layout follows MC-Calib: ``[x, y, z, w]`` with the scalar at index 3, so the
antipodal sign fix tests ``q[3] < 0``.
"""

from __future__ import annotations

from typing import List

import numpy as np


def _mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion ``[x, y, z, w]`` (scalar last)."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion ``[x, y, z, w]`` -> rotation matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def average_rotation(Rs: List[np.ndarray]) -> np.ndarray:
    """Markley SVD quaternion averaging of rotation matrices (geometrytools.cpp:881)."""
    Rs = list(Rs)
    if len(Rs) == 1:
        return np.asarray(Rs[0], float).copy()
    A = np.zeros((4, 4))
    for R in Rs:
        q = _mat_to_quat_xyzw(R)
        if q[3] < 0:                       # antipodal fix on the scalar (cpp:897)
            q = -q
        A += np.outer(q, q)
    A /= len(Rs)
    # A is symmetric PSD: leading eigenvector == leading left-singular vector (cpp uses SVD U[:,0]).
    _, V = np.linalg.eigh(A)
    return _quat_xyzw_to_mat(V[:, -1])


def average_translation(ts: np.ndarray) -> np.ndarray:
    """Component-wise median translation (McCalib.cpp:843 / geometrytools.cpp:697)."""
    return np.median(np.asarray(ts, float).reshape(-1, 3), axis=0)


def average_transform(Ts: List[np.ndarray]) -> np.ndarray:
    """Fuse a stack of noisy 4x4 transforms into one (``initInterTransform`` analogue).

    Rotation by Markley averaging, translation by component-wise median.
    """
    Ts = [np.asarray(T, float) for T in Ts]
    R = average_rotation([T[:3, :3] for T in Ts])
    t = average_translation(np.array([T[:3, 3] for T in Ts]))
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = t
    return out


def _rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle (rad) between two rotations."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def robust_average_transform(Ts: List[np.ndarray], *, iters: int = 3) -> np.ndarray:
    """Outlier-robust fusion of noisy relative transforms.

    A handful of per-frame relative poses can be grossly wrong (a frame whose object pose
    was estimated from an outlier-corrupted view), and plain :func:`average_transform`
    weights every sample equally, so the Markley rotation average is dragged into a wrong
    basin — the dominant failure mode of rig-extrinsics init under gross outliers. This
    estimator instead **selects an inlier consensus**: start from the (robust) median
    estimate, score each sample by its rotation+translation deviation, keep the samples
    within a MAD-based gate, and re-fuse — iterated a few times. Falls back to the plain
    average when too few samples survive. The translation scale (metric, from the boards)
    sets the gate, so SE(3) — not Sim(3) — is all that is needed.
    """
    Ts = [np.asarray(T, float) for T in Ts]
    if len(Ts) <= 2:
        return average_transform(Ts)
    Rs = [T[:3, :3] for T in Ts]
    ts = np.array([T[:3, 3] for T in Ts])
    keep = np.ones(len(Ts), bool)
    est = average_transform(Ts)
    for _ in range(iters):
        Re, te = est[:3, :3], est[:3, 3]
        ang = np.array([_rot_angle(Re, R) for R in Rs])          # rotation residual (rad)
        dist = np.linalg.norm(ts - te, axis=1)                   # translation residual
        # MAD-based gates (1.4826·MAD ≈ σ); floor avoids over-tight gates on clean data.
        a_thr = max(np.median(ang) + 3.0 * 1.4826 * np.median(np.abs(ang - np.median(ang))),
                    np.deg2rad(2.0))
        d_med = float(np.median(dist))
        d_thr = max(d_med + 3.0 * 1.4826 * float(np.median(np.abs(dist - d_med))),
                    0.02 * max(d_med, 1e-6) + 1e-3)
        new_keep = (ang <= a_thr) & (dist <= d_thr)
        if new_keep.sum() < max(3, int(0.3 * len(Ts))):
            break                                                # too aggressive — stop
        est = average_transform([Ts[i] for i in range(len(Ts)) if new_keep[i]])
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return est


def mean_transform(Ts: List[np.ndarray]) -> np.ndarray:
    """Markley rotation averaging but *arithmetic-mean* translation — the convention
    ``CameraGroupObs::computeObjectsPose`` uses (CameraGroupObs.cpp:95), distinct from
    :func:`average_transform`'s median."""
    Ts = [np.asarray(T, float) for T in Ts]
    R = average_rotation([T[:3, :3] for T in Ts])
    t = np.mean(np.array([T[:3, 3] for T in Ts]), axis=0)
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = t
    return out
