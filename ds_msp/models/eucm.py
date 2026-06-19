"""Enhanced Unified Camera Model implementing the CameraModel contract."""

from __future__ import annotations

from typing import ClassVar, Tuple

import numpy as np

from .eucm_math import eucm_project, eucm_project_jacobian, eucm_unproject


class EUCMModel:
    """Enhanced UCM (Khomutenko et al. 2016). Satisfies ``CameraModel``."""

    name: ClassVar[str] = "eucm"
    param_names: ClassVar[Tuple[str, ...]] = ("fx", "fy", "cx", "cy", "alpha", "beta")

    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 alpha: float, beta: float) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.alpha = float(alpha)
        self.beta = float(beta)

    @classmethod
    def sample(cls) -> "EUCMModel":
        return cls(711.57, 711.24, 949.18, 518.81, 0.6, 1.1)

    @property
    def params(self) -> np.ndarray:
        return np.array([self.fx, self.fy, self.cx, self.cy, self.alpha, self.beta],
                        dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def distortion(self) -> np.ndarray:
        return np.array([self.alpha, self.beta], dtype=np.float64)

    def project(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u, v, valid = eucm_project(np.asarray(P, dtype=np.float64),
                                   self.fx, self.fy, self.cx, self.cy, self.alpha, self.beta)
        return np.stack([u, v], axis=-1), valid

    def unproject(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return eucm_unproject(np.asarray(uv, dtype=np.float64),
                              self.fx, self.fy, self.cx, self.cy, self.alpha, self.beta)

    def project_jacobian(self, P):
        u, v, J_point, J_param, valid = eucm_project_jacobian(
            np.asarray(P, dtype=np.float64),
            self.fx, self.fy, self.cx, self.cy, self.alpha, self.beta)
        return np.stack([u, v], axis=-1), J_point, J_param, valid

    @classmethod
    def from_params(cls, p: np.ndarray) -> "EUCMModel":
        return cls(*np.asarray(p, dtype=np.float64).ravel())

    @classmethod
    def param_bounds(cls) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.array([1.0, 1.0, -1e5, -1e5, 1e-6, 1e-3], dtype=np.float64)
        ub = np.array([1e5, 1e5, 1e5, 1e5, 1.0 - 1e-6, 10.0], dtype=np.float64)
        return lb, ub

    def initialize_from_correspondences(self, K_seed, rays, pixels) -> None:
        self.fx, self.fy = float(K_seed[0, 0]), float(K_seed[1, 1])
        self.cx, self.cy = float(K_seed[0, 2]), float(K_seed[1, 2])
        self.beta = 1.0
        rays = np.asarray(rays, dtype=np.float64)
        x, y, z = rays[:, 0], rays[:, 1], rays[:, 2]
        mx = (pixels[:, 0] - self.cx) / self.fx
        my = (pixels[:, 1] - self.cy) / self.fy
        # beta=1, unit rays (d=1): alpha = (x - mx*z)/(mx*(1 - z)), linear LS.
        A = np.concatenate([mx * (1.0 - z), my * (1.0 - z)])
        b = np.concatenate([x - mx * z, y - my * z])
        denom = float(A @ A)
        self.alpha = float(np.clip((A @ b) / denom, 1e-6, 1.0 - 1e-6)) if denom > 1e-12 else 0.5

    def to_dict(self) -> dict:
        d = {"model": self.name}
        d.update({k: float(v) for k, v in zip(self.param_names, self.params)})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EUCMModel":
        return cls(**{k: d[k] for k in cls.param_names})

    def __repr__(self) -> str:
        return ("EUCMModel(fx={:.3f}, fy={:.3f}, cx={:.3f}, cy={:.3f}, "
                "alpha={:.4f}, beta={:.4f})").format(
                    self.fx, self.fy, self.cx, self.cy, self.alpha, self.beta)
