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
