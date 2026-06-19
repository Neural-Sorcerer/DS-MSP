"""
Generic pinhole helpers shared across models and services.

Model-independent, numpy-only — lives in ``core`` so any layer (models, ops,
cv, ldc) may use it without creating a cross-dependency.
"""

from __future__ import annotations

import numpy as np


def balanced_pinhole_K(fx: float, fy: float, width: int, height: int,
                       balance: float = 0.5) -> np.ndarray:
    """
    Build a pinhole intrinsic matrix for the undistorted (rectified) image.

    The new focal length is a fraction of the original, controlled by `balance`:
        balance 0.0 -> 0.4x  (widest FOV, more of the scene kept)
        balance 0.5 -> 0.6x  (default)
        balance 1.0 -> 0.8x  (narrowest FOV, least peripheral stretch)
    The principal point is placed at the image center.
    """
    focal_scale = 0.4 + balance * 0.4
    f_new = ((fx + fy) / 2.0) * focal_scale
    return np.array([
        [f_new, 0.0, width / 2.0],
        [0.0, f_new, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
