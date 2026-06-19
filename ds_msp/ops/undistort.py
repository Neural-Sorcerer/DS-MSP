"""
Image / point undistortion that works on ANY camera model.

The stateful map cache lives HERE (in the service), not on the model — keeping
models as pure value objects. Depends only on the contract + core pinhole helper.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from ..core.contracts import CameraModel
from ..core.pinhole import balanced_pinhole_K


class Undistorter:
    """Undistort images/points from any model into a pinhole view.

    Example
    -------
    >>> und = Undistorter(model, 1920, 1080)
    >>> img_rect, K_new = und.undistort_image(img)        # any CameraModel
    """

    def __init__(self, model: CameraModel, width: int, height: int) -> None:
        self.model = model
        self.width = int(width)
        self.height = int(height)
        self._mapx = None
        self._mapy = None
        self._K_new = None

    def new_K(self, balance: float = 0.5) -> np.ndarray:
        K = self.model.K
        return balanced_pinhole_K(K[0, 0], K[1, 1], self.width, self.height, balance)

    def maps(self, K_new: Optional[np.ndarray] = None
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if K_new is None:
            K_new = self.new_K()
        if self._mapx is not None and self._K_new is not None \
                and np.array_equal(K_new, self._K_new):
            return self._mapx, self._mapy, self._K_new

        fx_n, fy_n = K_new[0, 0], K_new[1, 1]
        cx_n, cy_n = K_new[0, 2], K_new[1, 2]
        xg, yg = np.meshgrid(np.arange(self.width, dtype=np.float64),
                             np.arange(self.height, dtype=np.float64), indexing="xy")
        rays = np.stack([(xg - cx_n) / fx_n, (yg - cy_n) / fy_n, np.ones_like(xg)], axis=-1)
        uv, valid = self.model.project(rays)
        mapx = uv[..., 0].astype(np.float32)
        mapy = uv[..., 1].astype(np.float32)
        mapx[~valid] = -1
        mapy[~valid] = -1
        self._mapx, self._mapy, self._K_new = mapx, mapy, K_new
        return mapx, mapy, K_new

    def undistort_image(self, img: np.ndarray, K_new: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
        mapx, mapy, K_new = self.maps(K_new)
        out = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return out, K_new

    def undistort_points(self, points: np.ndarray, K_new: Optional[np.ndarray] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Distorted pixels -> rectified pinhole pixels (in the ``K_new`` frame)."""
        if K_new is None:
            K_new = self.new_K()
        rays, valid = self.model.unproject(np.asarray(points, dtype=np.float64))
        rays_n = rays / (rays[:, 2:3] + 1e-12)
        u = K_new[0, 0] * rays_n[:, 0] + K_new[0, 2]
        v = K_new[1, 1] * rays_n[:, 1] + K_new[1, 2]
        return np.stack([u, v], axis=-1), valid

    def distort_points(self, points: np.ndarray, K_new: Optional[np.ndarray] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse of :meth:`undistort_points`: rectified pinhole pixels (in the
        ``K_new`` frame) -> distorted pixels in the original image."""
        if K_new is None:
            K_new = self.new_K()
        pts = np.asarray(points, dtype=np.float64)
        mx = (pts[:, 0] - K_new[0, 2]) / K_new[0, 0]
        my = (pts[:, 1] - K_new[1, 2]) / K_new[1, 1]
        rays = np.stack([mx, my, np.ones_like(mx)], axis=-1)
        rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
        return self.model.project(rays)
