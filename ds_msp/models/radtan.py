"""Pinhole + radial-tangential (Brown) model — OpenCV cv2.projectPoints compatible."""

from __future__ import annotations

from typing import ClassVar, Tuple

import numpy as np

from .radtan_math import radtan_project, radtan_project_jacobian, radtan_unproject


class RadTanModel:
    """Pinhole with radial-tangential distortion. Satisfies ``CameraModel``.

    ``distortion`` returns ``[k1, k2, p1, p2, k3]`` (OpenCV order) for direct use
    with ``cv2.projectPoints`` / ``cv2.undistortPoints``. Narrow-FOV: only models
    rays with ``z > 0``.
    """

    name: ClassVar[str] = "radtan"
    param_names: ClassVar[Tuple[str, ...]] = (
        "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")

    def __init__(self, fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0) -> None:
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.k1, self.k2, self.k3 = float(k1), float(k2), float(k3)
        self.p1, self.p2 = float(p1), float(p2)

    @classmethod
    def sample(cls) -> "RadTanModel":
        return cls(600.0, 602.0, 320.0, 240.0, -0.12, 0.05, 0.001, -0.0015, 0.008)

    @property
    def params(self) -> np.ndarray:
        return np.array([self.fx, self.fy, self.cx, self.cy,
                         self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def distortion(self) -> np.ndarray:
        """OpenCV distCoeffs order [k1, k2, p1, p2, k3]."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

    def _coeffs(self):
        return (self.k1, self.k2, self.p1, self.p2, self.k3)

    def project(self, P):
        u, v, valid = radtan_project(np.asarray(P, dtype=np.float64),
                                     self.fx, self.fy, self.cx, self.cy, *self._coeffs())
        return np.stack([u, v], axis=-1), valid

    def unproject(self, uv):
        return radtan_unproject(np.asarray(uv, dtype=np.float64),
                                self.fx, self.fy, self.cx, self.cy, *self._coeffs())

    def project_jacobian(self, P):
        u, v, J_point, J_param, valid = radtan_project_jacobian(
            np.asarray(P, dtype=np.float64),
            self.fx, self.fy, self.cx, self.cy, *self._coeffs())
        return np.stack([u, v], axis=-1), J_point, J_param, valid

    @classmethod
    def from_params(cls, p):
        return cls(*np.asarray(p, dtype=np.float64).ravel())

    @classmethod
    def param_bounds(cls):
        lb = np.array([1.0, 1.0, -1e5, -1e5, -2.0, -2.0, -1.0, -1.0, -2.0], dtype=np.float64)
        ub = np.array([1e5, 1e5, 1e5, 1e5, 2.0, 2.0, 1.0, 1.0, 2.0], dtype=np.float64)
        return lb, ub

    def initialize_from_correspondences(self, K_seed, rays, pixels) -> None:
        self.fx, self.fy = float(K_seed[0, 0]), float(K_seed[1, 1])
        self.cx, self.cy = float(K_seed[0, 2]), float(K_seed[1, 2])
        self.k1 = self.k2 = self.k3 = self.p1 = self.p2 = 0.0

    def to_dict(self) -> dict:
        d = {"model": self.name}
        d.update({k: float(v) for k, v in zip(self.param_names, self.params)})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RadTanModel":
        return cls(**{k: d[k] for k in cls.param_names})

    def __repr__(self) -> str:
        return ("RadTanModel(fx={:.3f}, fy={:.3f}, cx={:.3f}, cy={:.3f}, "
                "k1={:.5f}, k2={:.5f}, p1={:.5f}, p2={:.5f}, k3={:.5f})").format(
                    self.fx, self.fy, self.cx, self.cy,
                    self.k1, self.k2, self.p1, self.p2, self.k3)
