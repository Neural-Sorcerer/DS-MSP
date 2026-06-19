"""
Double Sphere model implementing the CameraModel contract.

Thin value-object wrapper over the pure math in ``ds_math``. Depends only on
``ds_math`` (numpy) and ``core.contracts`` — no OpenCV, no services.
"""

from __future__ import annotations

from typing import ClassVar, Tuple

import numpy as np

from .ds_math import ds_project, ds_project_jacobian, ds_unproject


class DoubleSphereModel:
    """Double Sphere camera (Usenko et al. 2018). Satisfies ``CameraModel``."""

    name: ClassVar[str] = "ds"
    param_names: ClassVar[Tuple[str, ...]] = ("fx", "fy", "cx", "cy", "xi", "alpha")

    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 xi: float, alpha: float) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.xi = float(xi)
        self.alpha = float(alpha)

    @classmethod
    def sample(cls) -> "DoubleSphereModel":
        """Realistic instance for contract testing (the bundled calibration)."""
        return cls(711.57, 711.24, 949.18, 518.81, 0.183, 0.809)

    # -- parameter access -------------------------------------------------
    @property
    def params(self) -> np.ndarray:
        return np.array([self.fx, self.fy, self.cx, self.cy, self.xi, self.alpha],
                        dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def distortion(self) -> np.ndarray:
        return np.array([self.xi, self.alpha], dtype=np.float64)

    # -- core math --------------------------------------------------------
    def project(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u, v, valid = ds_project(np.asarray(P, dtype=np.float64),
                                 self.fx, self.fy, self.cx, self.cy, self.xi, self.alpha)
        return np.stack([u, v], axis=-1), valid

    def unproject(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return ds_unproject(np.asarray(uv, dtype=np.float64),
                            self.fx, self.fy, self.cx, self.cy, self.xi, self.alpha)

    def project_jacobian(
        self, P: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u, v, J_point, J_param, valid = ds_project_jacobian(
            np.asarray(P, dtype=np.float64),
            self.fx, self.fy, self.cx, self.cy, self.xi, self.alpha)
        return np.stack([u, v], axis=-1), J_point, J_param, valid

    # -- construction / bounds -------------------------------------------
    @classmethod
    def from_params(cls, p: np.ndarray) -> "DoubleSphereModel":
        return cls(*np.asarray(p, dtype=np.float64).ravel())

    @classmethod
    def param_bounds(cls) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.array([1.0, 1.0, -1e5, -1e5, -1.0, 1e-6], dtype=np.float64)
        ub = np.array([1e5, 1e5, 1e5, 1e5, 1.0, 1.0 - 1e-6], dtype=np.float64)
        return lb, ub

    # -- conversion hook --------------------------------------------------
    def initialize_from_correspondences(
        self, K_seed: np.ndarray, rays: np.ndarray, pixels: np.ndarray
    ) -> None:
        # Inherit pinhole intrinsics from the source.
        self.fx, self.fy = float(K_seed[0, 0]), float(K_seed[1, 1])
        self.cx, self.cy = float(K_seed[0, 2]), float(K_seed[1, 2])
        # Seed distortion: xi = 0 reduces DS to UCM; for unit rays (d1 = 1),
        #   mx = x / (alpha*(1 - z) + z)  =>  alpha = (x - mx*z) / (mx*(1 - z)).
        # Solve linearly over both axes.
        rays = np.asarray(rays, dtype=np.float64)
        x, y, z = rays[:, 0], rays[:, 1], rays[:, 2]
        mx = (pixels[:, 0] - self.cx) / self.fx
        my = (pixels[:, 1] - self.cy) / self.fy
        A = np.concatenate([mx * (1.0 - z), my * (1.0 - z)])
        b = np.concatenate([x - mx * z, y - my * z])
        denom = float(A @ A)
        self.xi = 0.0
        self.alpha = float(np.clip((A @ b) / denom, 1e-6, 1.0 - 1e-6)) if denom > 1e-12 else 0.5

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        d = {"model": self.name}
        d.update({k: float(v) for k, v in zip(self.param_names, self.params)})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DoubleSphereModel":
        return cls(**{k: d[k] for k in cls.param_names})

    def __repr__(self) -> str:
        return ("DoubleSphereModel(fx={:.3f}, fy={:.3f}, cx={:.3f}, cy={:.3f}, "
                "xi={:.4f}, alpha={:.4f})").format(
                    self.fx, self.fy, self.cx, self.cy, self.xi, self.alpha)
