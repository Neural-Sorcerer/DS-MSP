"""Kannala-Brandt (equidistant fisheye) model — OpenCV cv2.fisheye compatible."""

from __future__ import annotations

from typing import ClassVar, Tuple

import numpy as np

from .kb_math import kb_project, kb_project_jacobian, kb_unproject


class KannalaBrandtModel:
    """Kannala-Brandt / equidistant fisheye. Satisfies ``CameraModel``.

    ``K`` and ``distortion`` ([k1,k2,k3,k4]) plug directly into ``cv2.fisheye``.
    """

    name: ClassVar[str] = "kb"
    param_names: ClassVar[Tuple[str, ...]] = (
        "fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")

    def __init__(self, fx, fy, cx, cy, k1=0.0, k2=0.0, k3=0.0, k4=0.0) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.k3 = float(k3)
        self.k4 = float(k4)

    @classmethod
    def sample(cls) -> "KannalaBrandtModel":
        return cls(320.0, 321.0, 320.0, 240.0, 0.05, 0.01, -0.002, 0.0008)

    @property
    def params(self) -> np.ndarray:
        return np.array([self.fx, self.fy, self.cx, self.cy,
                         self.k1, self.k2, self.k3, self.k4], dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def distortion(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.k3, self.k4], dtype=np.float64)

    def project(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u, v, valid = kb_project(np.asarray(P, dtype=np.float64),
                                 self.fx, self.fy, self.cx, self.cy,
                                 self.k1, self.k2, self.k3, self.k4)
        return np.stack([u, v], axis=-1), valid

    def unproject(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return kb_unproject(np.asarray(uv, dtype=np.float64),
                            self.fx, self.fy, self.cx, self.cy,
                            self.k1, self.k2, self.k3, self.k4)

    def project_jacobian(self, P):
        u, v, J_point, J_param, valid = kb_project_jacobian(
            np.asarray(P, dtype=np.float64),
            self.fx, self.fy, self.cx, self.cy,
            self.k1, self.k2, self.k3, self.k4)
        return np.stack([u, v], axis=-1), J_point, J_param, valid

    @classmethod
    def from_params(cls, p: np.ndarray) -> "KannalaBrandtModel":
        return cls(*np.asarray(p, dtype=np.float64).ravel())

    @classmethod
    def param_bounds(cls) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.array([1.0, 1.0, -1e5, -1e5, -1.0, -1.0, -1.0, -1.0], dtype=np.float64)
        ub = np.array([1e5, 1e5, 1e5, 1e5, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        return lb, ub

    def initialize_from_correspondences(self, K_seed, rays, pixels) -> None:
        self.fx, self.fy = float(K_seed[0, 0]), float(K_seed[1, 1])
        self.cx, self.cy = float(K_seed[0, 2]), float(K_seed[1, 2])
        rays = np.asarray(rays, dtype=np.float64)
        theta = np.arctan2(np.sqrt(rays[:, 0]**2 + rays[:, 1]**2), rays[:, 2])
        mx = (pixels[:, 0] - self.cx) / self.fx
        my = (pixels[:, 1] - self.cy) / self.fy
        ru = np.sqrt(mx*mx + my*my)
        # ru = theta + k1 th^3 + k2 th^5 + k3 th^7 + k4 th^9 -> linear in k.
        A = np.stack([theta**3, theta**5, theta**7, theta**9], axis=1)
        b = ru - theta
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        self.k1, self.k2, self.k3, self.k4 = (float(c) for c in coeffs)

    def to_dict(self) -> dict:
        d = {"model": self.name}
        d.update({k: float(v) for k, v in zip(self.param_names, self.params)})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "KannalaBrandtModel":
        return cls(**{k: d[k] for k in cls.param_names})

    def __repr__(self) -> str:
        return ("KannalaBrandtModel(fx={:.3f}, fy={:.3f}, cx={:.3f}, cy={:.3f}, "
                "k=[{:.5f}, {:.5f}, {:.5f}, {:.5f}])").format(
                    self.fx, self.fy, self.cx, self.cy,
                    self.k1, self.k2, self.k3, self.k4)
