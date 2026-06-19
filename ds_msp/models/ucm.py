"""Unified Camera Model implementing the CameraModel contract."""

from __future__ import annotations

from typing import ClassVar, Tuple

import numpy as np

from .ucm_math import ucm_project, ucm_project_jacobian, ucm_unproject


class UCMModel:
    """Unified Camera Model (single sphere). Satisfies ``CameraModel``."""

    name: ClassVar[str] = "ucm"
    param_names: ClassVar[Tuple[str, ...]] = ("fx", "fy", "cx", "cy", "alpha")

    def __init__(self, fx: float, fy: float, cx: float, cy: float, alpha: float) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.alpha = float(alpha)

    @classmethod
    def sample(cls) -> "UCMModel":
        return cls(711.57, 711.24, 949.18, 518.81, 0.62)

    @property
    def params(self) -> np.ndarray:
        return np.array([self.fx, self.fy, self.cx, self.cy, self.alpha], dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def distortion(self) -> np.ndarray:
        return np.array([self.alpha], dtype=np.float64)

    def project(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u, v, valid = ucm_project(np.asarray(P, dtype=np.float64),
                                  self.fx, self.fy, self.cx, self.cy, self.alpha)
        return np.stack([u, v], axis=-1), valid

    def unproject(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return ucm_unproject(np.asarray(uv, dtype=np.float64),
                             self.fx, self.fy, self.cx, self.cy, self.alpha)

    def project_jacobian(self, P):
        u, v, J_point, J_param, valid = ucm_project_jacobian(
            np.asarray(P, dtype=np.float64),
            self.fx, self.fy, self.cx, self.cy, self.alpha)
        return np.stack([u, v], axis=-1), J_point, J_param, valid

    @classmethod
    def from_params(cls, p: np.ndarray) -> "UCMModel":
        return cls(*np.asarray(p, dtype=np.float64).ravel())

    @classmethod
    def param_bounds(cls) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.array([1.0, 1.0, -1e5, -1e5, 1e-6], dtype=np.float64)
        ub = np.array([1e5, 1e5, 1e5, 1e5, 1.0 - 1e-6], dtype=np.float64)
        return lb, ub

    def initialize_from_correspondences(self, K_seed, rays, pixels) -> None:
        self.fx, self.fy = float(K_seed[0, 0]), float(K_seed[1, 1])
        self.cx, self.cy = float(K_seed[0, 2]), float(K_seed[1, 2])
        rays = np.asarray(rays, dtype=np.float64)
        x, y, z = rays[:, 0], rays[:, 1], rays[:, 2]
        mx = (pixels[:, 0] - self.cx) / self.fx
        my = (pixels[:, 1] - self.cy) / self.fy
        # unit rays (d = 1): alpha = (x - mx*z) / (mx*(1 - z)), solved linearly.
        A = np.concatenate([mx * (1.0 - z), my * (1.0 - z)])
        b = np.concatenate([x - mx * z, y - my * z])
        denom = float(A @ A)
        self.alpha = float(np.clip((A @ b) / denom, 1e-6, 1.0 - 1e-6)) if denom > 1e-12 else 0.5

    def to_dict(self) -> dict:
        d = {"model": self.name}
        d.update({k: float(v) for k, v in zip(self.param_names, self.params)})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "UCMModel":
        return cls(**{k: d[k] for k in cls.param_names})

    def __repr__(self) -> str:
        return "UCMModel(fx={:.3f}, fy={:.3f}, cx={:.3f}, cy={:.3f}, alpha={:.4f})".format(
            self.fx, self.fy, self.cx, self.cy, self.alpha)
