"""
Relative pose between two cameras from shared board observations.

If two synchronized cameras both observe the same calibration board, each frame
gives the board pose in *each* camera — and their composition is the fixed rigid
transform between the cameras. Averaging that estimate over many frames yields the
stereo extrinsics. This is the standard way to bootstrap a stereo calibration
(Kalibr/OpenCV do the same before any joint refinement).

Pure NumPy + OpenCV's Rodrigues (already a calibration dependency).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

Pose = Tuple[np.ndarray, np.ndarray]  # (rvec(3,), tvec(3,)) = board-to-camera


def _geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angle (degrees) of the rotation taking Ra to Rb."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def estimate_relative_pose(poses_from: Sequence[Pose],
                           poses_to: Sequence[Pose]) -> Dict:
    """Estimate the rigid transform ``T_to_from`` between two cameras.

    Given board poses observed by each camera on the **same frames**
    (``poses_from[i]`` and ``poses_to[i]`` are the board seen at frame ``i`` by the
    "from" and "to" cameras), returns the transform that maps a point in the *from*
    camera's frame into the *to* camera's frame — e.g. pass cam0 then cam1 to get
    Kalibr's ``T_cn_cnm1`` (= ``T_cam1_cam0``).

    Each frame yields one estimate ``T_to_board ∘ (T_from_board)^-1``; the rotations
    are averaged on SO(3) (chordal / SVD projection) and the translation by the
    component-wise median (robust to per-frame PnP noise).

    Returns ``{T, R, t, n, rot_rms_deg, t_std_mm}`` where ``T`` is 4x4 and
    ``rot_rms_deg`` / ``t_std_mm`` report how consistent the per-frame estimates are.
    """
    if len(poses_from) != len(poses_to) or not poses_from:
        raise ValueError("poses_from and poses_to must be non-empty and the same length")

    rots: List[np.ndarray] = []
    trans: List[np.ndarray] = []
    for (rvf, tvf), (rvt, tvt) in zip(poses_from, poses_to):
        Rf, _ = cv2.Rodrigues(np.asarray(rvf, dtype=np.float64))
        Rt, _ = cv2.Rodrigues(np.asarray(rvt, dtype=np.float64))
        R = Rt @ Rf.T
        t = np.asarray(tvt, dtype=np.float64).ravel() - R @ np.asarray(tvf, dtype=np.float64).ravel()
        rots.append(R)
        trans.append(t)
    trans = np.asarray(trans)

    # rotation: average the matrices, then project back onto SO(3) via SVD
    U, _, Vt = np.linalg.svd(np.mean(rots, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy()
        U[:, -1] *= -1.0
        R = U @ Vt
    t = np.median(trans, axis=0)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    rot_rms = float(np.sqrt(np.mean([_geodesic_deg(R, Ri) ** 2 for Ri in rots])))
    return {
        "T": T, "R": R, "t": t, "n": len(rots),
        "rot_rms_deg": rot_rms,
        "t_std_mm": np.std(trans, axis=0) * 1000.0,
    }


def relative_pose_error(T_a: np.ndarray, T_b: np.ndarray) -> Dict:
    """Compare two rigid transforms: rotation angle (deg) and translation error (mm)."""
    T_a, T_b = np.asarray(T_a, float), np.asarray(T_b, float)
    return {
        "rot_deg": _geodesic_deg(T_a[:3, :3], T_b[:3, :3]),
        "trans_mm": float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]) * 1000.0),
    }
