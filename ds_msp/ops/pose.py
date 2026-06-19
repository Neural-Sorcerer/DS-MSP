"""
Pose estimation that works on ANY camera model.

Depends only on the CameraModel contract (project/unproject) + OpenCV — never on
a concrete model. Unprojects to bearing rays, keeps the front-facing valid ones,
and solves PnP in the normalized plane.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from ..core.contracts import CameraModel


def solve_pnp(model: CameraModel, object_points: np.ndarray, image_points: np.ndarray,
              method: int = cv2.SOLVEPNP_ITERATIVE
              ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """Estimate pose from 3D-2D correspondences for any fisheye/omni model.

    Parameters
    ----------
    model : CameraModel
        Any model implementing the contract.
    object_points : (N, 3) world points.
    image_points : (N, 2) distorted pixels.

    Returns ``(success, rvec, tvec)`` with squeezed vectors, or ``(False, None, None)``.
    """
    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)

    rays, valid = model.unproject(image_points)
    usable = valid & (rays[:, 2] > 1e-6)
    if not usable.all():
        object_points = object_points[usable]
        rays = rays[usable]
        if len(object_points) < 4:
            return False, None, None

    pts_norm = rays[:, :2] / rays[:, 2:3]
    success, rvec, tvec = cv2.solvePnP(
        object_points, pts_norm.astype(np.float64),
        np.eye(3, dtype=np.float64), np.zeros(5, dtype=np.float64), flags=method)
    if not success:
        return False, None, None
    return True, rvec.squeeze(), tvec.squeeze()
