"""OCamCalib / Scaramuzza omnidirectional polynomial model."""

from __future__ import annotations

from typing import ClassVar, Tuple

import numpy as np

from .ocam_math import R_REF, ocam_project, ocam_project_jacobian, ocam_unproject


class OCamModel:
    """Scaramuzza polynomial omni-camera. Satisfies ``CameraModel``.

    Parameters: centre (cx, cy), affine (c, d, e), world polynomial (a0..a4).
    Has no native fx/fy; ``K`` exposes a pinhole-equivalent focal ~ |a0|.
    """

    name: ClassVar[str] = "ocam"
    param_names: ClassVar[Tuple[str, ...]] = (
        "cx", "cy", "c", "d", "e", "a0", "a1", "a2", "a3", "a4")

    def __init__(self, cx, cy, c=1.0, d=0.0, e=0.0,
                 a0=-200.0, a1=0.0, a2=0.0, a3=0.0, a4=0.0) -> None:
        self.cx, self.cy = float(cx), float(cy)
        self.c, self.d, self.e = float(c), float(d), float(e)
        self.a0, self.a1, self.a2, self.a3, self.a4 = map(float, (a0, a1, a2, a3, a4))

    @classmethod
    def sample(cls) -> "OCamModel":
        # Polynomial argument is normalized by R_REF=100, so a2..a4 are O(1).
        return cls(320.0, 240.0, 1.0, 0.0, 0.0, -220.0, 0.0, 6.0, -1.5, 0.25)

    def _p(self):
        return (self.cx, self.cy, self.c, self.d, self.e,
                self.a0, self.a1, self.a2, self.a3, self.a4)

    @property
    def params(self) -> np.ndarray:
        return np.array(self._p(), dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        f = abs(self.a0)
        return np.array([[f, 0.0, self.cx], [0.0, f, self.cy], [0.0, 0.0, 1.0]],
                        dtype=np.float64)

    @property
    def distortion(self) -> np.ndarray:
        return np.array([self.c, self.d, self.e,
                         self.a0, self.a1, self.a2, self.a3, self.a4], dtype=np.float64)

    def project(self, P):
        u, v, valid = ocam_project(np.asarray(P, dtype=np.float64), *self._p())
        return np.stack([u, v], axis=-1), valid

    def unproject(self, uv):
        return ocam_unproject(np.asarray(uv, dtype=np.float64), *self._p())

    def project_jacobian(self, P):
        u, v, J_point, J_param, valid = ocam_project_jacobian(
            np.asarray(P, dtype=np.float64), *self._p())
        return np.stack([u, v], axis=-1), J_point, J_param, valid

    @classmethod
    def from_params(cls, p):
        return cls(*np.asarray(p, dtype=np.float64).ravel())

    @classmethod
    def param_bounds(cls):
        lb = np.array([-1e5, -1e5, 0.1, -1.0, -1.0, -1e5, -1e3, -1e3, -1e3, -1e3],
                      dtype=np.float64)
        ub = np.array([1e5, 1e5, 10.0, 1.0, 1.0, -1.0, 1e3, 1e3, 1e3, 1e3],
                      dtype=np.float64)
        return lb, ub

    def initialize_from_correspondences(self, K_seed, rays, pixels) -> None:
        self.cx, self.cy = float(K_seed[0, 2]), float(K_seed[1, 2])
        self.c, self.d, self.e = 1.0, 0.0, 0.0
        rays = np.asarray(rays, dtype=np.float64)
        x = pixels[:, 0] - self.cx
        y = pixels[:, 1] - self.cy
        rho = np.sqrt(x*x + y*y)
        rxy = np.maximum(np.sqrt(rays[:, 0]**2 + rays[:, 1]**2), 1e-9)
        # ray = [x, y, -w]/n  =>  w(rho) = -rho * rz / |rxy|
        w = -rho * rays[:, 2] / rxy
        rn = rho / R_REF
        A = np.stack([np.ones_like(rn), rn, rn**2, rn**3, rn**4], axis=1)
        coeffs, *_ = np.linalg.lstsq(A, w, rcond=None)
        self.a0, self.a1, self.a2, self.a3, self.a4 = (float(v) for v in coeffs)

    def to_dict(self) -> dict:
        d = {"model": self.name}
        d.update({k: float(v) for k, v in zip(self.param_names, self.params)})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "OCamModel":
        return cls(**{k: d[k] for k in cls.param_names})

    def __repr__(self) -> str:
        return ("OCamModel(cx={:.2f}, cy={:.2f}, affine=[{:.3f},{:.3f},{:.3f}], "
                "a=[{:.4g},{:.4g},{:.4g},{:.4g},{:.4g}])").format(
                    self.cx, self.cy, self.c, self.d, self.e,
                    self.a0, self.a1, self.a2, self.a3, self.a4)
