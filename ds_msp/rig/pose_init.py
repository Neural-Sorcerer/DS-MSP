"""Robust pose initialization — model-aware RANSAC PnP (``ransacP3PDistortion`` +
``BoardObs::estimatePose``, geometrytools.cpp:710 / BoardObs.cpp:121) and object-in-rig
pose averaging (``CameraGroupObs::computeObjectsPose``, CameraGroupObs.cpp:42).

The DS-MSP twist: instead of MC-Calib's per-distortion-type ``undistortPoints`` branch,
unproject pixels with the camera's *own* model (every model exposes ``unproject``) and
PnP on the normalized plane — the same trick ``calib.bundle._seed_poses`` already uses.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..core.contracts import CameraModel
from .averaging import average_rotation


def _focal(model: CameraModel) -> float:
    K = model.K
    return 0.5 * (abs(K[0, 0]) + abs(K[1, 1]))


def estimate_pose_ransac(
    model: CameraModel,
    object_pts: np.ndarray,
    image_pts: np.ndarray,
    *,
    thresh_px: float = 3.0,
    max_iters: int = 1000,
    confidence: float = 0.99,
    min_inliers: int = 4,
    seed: int = 0,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """RANSAC P3P of ``object_pts`` (N,3) against ``image_pts`` (N,2) pixels.

    Returns ``(T_cam_obj (4,4) | None, inliers (N,) bool)``. ``None`` when fewer than
    ``min_inliers`` points survive — MC-Calib invalidates a BoardObs below 4
    (BoardObs.cpp:149).
    """
    object_pts = np.asarray(object_pts, float)
    image_pts = np.asarray(image_pts, float)
    n_all = len(object_pts)
    rays, ok = model.unproject(image_pts)
    ok = ok & (rays[:, 2] > 1e-6)
    idx = np.where(ok)[0]
    if len(idx) < min_inliers:
        return None, np.zeros(n_all, bool)

    Xv = object_pts[idx]
    pnv = (rays[idx, :2] / rays[idx, 2:3]).astype(np.float64)
    thresh = thresh_px / _focal(model)            # pixel tol -> normalized-plane tol
    rng = np.random.default_rng(seed)
    n = len(Xv)

    best_inl, best_rvec, best_tvec = None, None, None
    it, iters = 0, max_iters
    K_eye = np.eye(3)
    while it < iters and it < max_iters:
        it += 1
        if n == 4:
            sample = np.arange(4)
        else:
            sample = rng.choice(n, 4, replace=False)
        try:
            okp, rvec, tvec = cv2.solvePnP(Xv[sample], pnv[sample], K_eye, None,
                                           flags=cv2.SOLVEPNP_P3P)
        except cv2.error:
            continue
        if not okp:
            continue
        proj, _ = cv2.projectPoints(Xv, rvec, tvec, K_eye, None)
        err = np.linalg.norm(proj.reshape(-1, 2) - pnv, axis=1)
        inl = err < thresh
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best_rvec, best_tvec = inl, rvec, tvec
            frac = float(np.clip(inl.mean(), 1e-6, 1.0))
            if frac >= 1.0:
                break
            fr3 = frac ** 3                        # adaptive iteration count, exponent 3 (cpp:300)
            # guard: tiny frac makes (1 - fr3) round to 1.0 -> log 0 -> div-by-zero
            iters = (max_iters if fr3 < 1e-9
                     else min(max_iters, int(np.log(1 - confidence) / np.log(1 - fr3)) + 1))

    if best_inl is None or best_inl.sum() < min_inliers:
        return None, np.zeros(n_all, bool)

    rvec, tvec = best_rvec, best_tvec
    if best_inl.sum() >= 6:                        # DLT refine needs >=6; else keep hypothesis
        okf, rv, tv = cv2.solvePnP(Xv[best_inl], pnv[best_inl], K_eye, None,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        if okf:
            rvec, tvec = rv, tv
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    inliers = np.zeros(n_all, bool)
    inliers[idx[best_inl]] = True
    return T, inliers


def average_object_pose_in_group(
    T_c_o_per_cam: List[Tuple[int, np.ndarray]],
    T_c_g: dict,
    ref_cam_id: int,
) -> np.ndarray:
    """Recover an object's pose in the group frame (``T_g_o``) from one or more cameras.

    Mirrors ``CameraGroupObs::computeObjectsPose``: if the reference camera sees the
    object, use it directly; otherwise average across the non-ref cameras with Markley
    rotation averaging and **arithmetic-mean** translation (CameraGroupObs.cpp:95).
    """
    # T_g_o = inv(T_c_g) @ T_c_o   (object->cam lifted into the group frame)
    lifted = []
    for cam_id, T_c_o in T_c_o_per_cam:
        T_g_o = np.linalg.inv(T_c_g[cam_id]) @ T_c_o
        if cam_id == ref_cam_id:
            return T_g_o
        lifted.append(T_g_o)
    R = average_rotation([T[:3, :3] for T in lifted])
    t = np.mean(np.array([T[:3, 3] for T in lifted]), axis=0)
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = t
    return out
