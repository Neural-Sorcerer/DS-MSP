"""
Test-support utilities shipped with the package (à la ``numpy.testing``).

Provides:
- :class:`FakeModel`: a trivial perfect-pinhole camera that satisfies the
  :class:`~ds_msp.core.contracts.CameraModel` Protocol. Services and the
  converter are tested against it, proving they depend only on the contract and
  run in the **absence of any real (fisheye) camera model**.
- Finite-difference Jacobian helpers and sampling utilities reused by the
  model-agnostic contract test suite, and available to downstream users who want
  to validate their own model implementations.

Pure NumPy — no OpenCV/SciPy — so it imports cleanly anywhere.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from .core.contracts import CameraModel


class FakeModel:
    """Reference pinhole model: ``u = fx·x/z + cx``, ``v = fy·y/z + cy``.

    Implements the full :class:`CameraModel` contract with closed-form analytic
    Jacobians. Used as a dependency-free stand-in so the rest of the library can
    be developed and tested before any fisheye model exists.
    """

    name = "fake_pinhole"
    param_names = ("fx", "fy", "cx", "cy")

    def __init__(self, fx: float, fy: float, cx: float, cy: float) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

    # -- factory used by the contract suite -------------------------------
    @classmethod
    def sample(cls) -> "FakeModel":
        """A realistic instance for contract testing."""
        return cls(fx=600.0, fy=605.0, cx=320.0, cy=240.0)

    # -- parameter access -------------------------------------------------
    @property
    def params(self) -> np.ndarray:
        return np.array([self.fx, self.fy, self.cx, self.cy], dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def distortion(self) -> np.ndarray:
        return np.empty(0, dtype=np.float64)

    # -- core math --------------------------------------------------------
    def project(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        P = np.asarray(P, dtype=np.float64)
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        valid = z > 1e-9
        zsafe = np.where(valid, z, 1.0)
        u = self.fx * x / zsafe + self.cx
        v = self.fy * y / zsafe + self.cy
        uv = np.stack([u, v], axis=-1)
        uv = np.where(valid[..., None], uv, 0.0)
        return uv, valid

    def unproject(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        uv = np.asarray(uv, dtype=np.float64)
        mx = (uv[..., 0] - self.cx) / self.fx
        my = (uv[..., 1] - self.cy) / self.fy
        rays = np.stack([mx, my, np.ones_like(mx)], axis=-1)
        rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
        valid = np.ones(uv.shape[:-1], dtype=bool)
        return rays, valid

    # -- analytic Jacobian ------------------------------------------------
    def project_jacobian(
        self, P: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        P = np.asarray(P, dtype=np.float64)
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        valid = z > 1e-9
        zsafe = np.where(valid, z, 1.0)
        inv = 1.0 / zsafe
        u = self.fx * x * inv + self.cx
        v = self.fy * y * inv + self.cy
        uv = np.stack([u, v], axis=-1)

        J_point = np.zeros(P.shape[:-1] + (2, 3), dtype=np.float64)
        J_point[..., 0, 0] = self.fx * inv
        J_point[..., 0, 2] = -self.fx * x * inv * inv
        J_point[..., 1, 1] = self.fy * inv
        J_point[..., 1, 2] = -self.fy * y * inv * inv

        J_param = np.zeros(P.shape[:-1] + (2, 4), dtype=np.float64)
        J_param[..., 0, 0] = x * inv   # du/dfx
        J_param[..., 0, 2] = 1.0       # du/dcx
        J_param[..., 1, 1] = y * inv   # dv/dfy
        J_param[..., 1, 3] = 1.0       # dv/dcy
        return uv, J_point, J_param, valid

    # -- construction / bounds -------------------------------------------
    @classmethod
    def from_params(cls, p: np.ndarray) -> "FakeModel":
        p = np.asarray(p, dtype=np.float64).ravel()
        return cls(*p)

    @classmethod
    def param_bounds(cls) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.array([1.0, 1.0, -1e5, -1e5], dtype=np.float64)
        ub = np.array([1e5, 1e5, 1e5, 1e5], dtype=np.float64)
        return lb, ub

    # -- conversion hook --------------------------------------------------
    def initialize_from_correspondences(
        self, K_seed: np.ndarray, rays: np.ndarray, pixels: np.ndarray
    ) -> None:
        # Linear least-squares seed: u = fx*(x/z) + cx, per axis.
        xn = rays[:, 0] / rays[:, 2]
        yn = rays[:, 1] / rays[:, 2]
        A_x = np.stack([xn, np.ones_like(xn)], axis=1)
        A_y = np.stack([yn, np.ones_like(yn)], axis=1)
        (self.fx, self.cx) = np.linalg.lstsq(A_x, pixels[:, 0], rcond=None)[0]
        (self.fy, self.cy) = np.linalg.lstsq(A_y, pixels[:, 1], rcond=None)[0]

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        d = {"model": self.name}
        d.update({k: float(v) for k, v in zip(self.param_names, self.params)})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FakeModel":
        return cls(**{k: d[k] for k in cls.param_names})


# ---------------------------------------------------------------------------
# Reusable contract helpers (used by the model-agnostic test suite and by
# downstream users validating their own CameraModel implementations).
# ---------------------------------------------------------------------------

def sample_forward_points(n: int = 48, seed: int = 0,
                          z_range: Tuple[float, float] = (1.0, 5.0),
                          xy: float = 0.8) -> np.ndarray:
    """Deterministic camera-frame points in front of the camera (``z > 0``)."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(z_range[0], z_range[1], n)
    x = rng.uniform(-xy, xy, n) * z
    y = rng.uniform(-xy, xy, n) * z
    return np.stack([x, y, z], axis=-1)


def finite_difference_point_jacobian(model: CameraModel, P: np.ndarray,
                                     eps: float = 1e-6) -> np.ndarray:
    """Numerical ``d(u,v)/d(x,y,z)`` via central differences, shape ``(N,2,3)``."""
    P = np.asarray(P, dtype=np.float64)
    J = np.zeros(P.shape[:-1] + (2, 3), dtype=np.float64)
    for k in range(3):
        dp = np.zeros(3)
        dp[k] = eps
        up, _ = model.project(P + dp)
        um, _ = model.project(P - dp)
        J[..., k] = (up - um) / (2 * eps)
    return J


def finite_difference_param_jacobian(model: CameraModel, P: np.ndarray,
                                     eps: float = 1e-6) -> np.ndarray:
    """Numerical ``d(u,v)/d(params)`` via central differences, shape ``(N,2,P)``.

    Uses a per-parameter RELATIVE step (scaled by ``|p_k|``) so it stays accurate
    for parameters spanning many magnitudes (e.g. an OCam coefficient ~1e-9
    alongside a focal ~1e3), where a fixed absolute step would be meaningless.
    """
    p = np.asarray(model.params, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    J = np.zeros(P.shape[:-1] + (2, p.size), dtype=np.float64)
    cls = type(model)
    for k in range(p.size):
        h = eps * max(abs(p[k]), 1.0)
        pp = p.copy()
        pp[k] += h
        pm = p.copy()
        pm[k] -= h
        up, _ = cls.from_params(pp).project(P)
        um, _ = cls.from_params(pm).project(P)
        J[..., k] = (up - um) / (2 * h)
    return J


#: Sample factories the contract suite always runs against. Later phases append
#: ``(UCMModel.sample)``, ``(KannalaBrandtModel.sample)``, etc.
REFERENCE_MODELS: List[Tuple[str, Callable[[], CameraModel]]] = [
    ("fake_pinhole", FakeModel.sample),
]
