"""
Core contracts for the multi-model camera library.

This module is the single source of truth for the **interface** every camera
model implements and the **data conventions** every module exchanges. It is
deliberately dependency-free (numpy + typing only) so it sits at the bottom of
the dependency graph: ``core`` knows nothing about concrete models, services, or
I/O. Everything else depends on this; this depends on nothing internal.

Data conventions (the "wire format" between modules)
----------------------------------------------------
====================  ============  =========  ==========================================
Name                  Shape         dtype      Meaning / convention
====================  ============  =========  ==========================================
Points3D              (N, 3)        float64    camera-frame points, meters, +Z forward
Pixels                (N, 2)        float64    (u, v), pixel centers, origin top-left
Rays                  (N, 3)        float64    unit-norm bearing vectors
Valid                 (N,)          bool       per-point projectability / feasibility mask
Params                (P,)          float64    model parameters in ``param_names`` order
J_point               (N, 2, 3)     float64    d(u, v) / d(x, y, z)
J_param               (N, 2, P)     float64    d(u, v) / d(params)
====================  ============  =========  ==========================================

Additional rules:
- Leading batch dimensions are preserved: ``(..., 3)`` / ``(..., 2)`` are accepted.
- Invalid rows are zeroed, never NaN, so masks compose without poisoning arithmetic.
- Parameter vectors are always produced/consumed in ``param_names`` order.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, Tuple, runtime_checkable

import numpy as np

# Type aliases. All are ``np.ndarray`` at runtime; the names document intent and
# the shapes/dtypes promised by the table above.
Points3D = np.ndarray
Pixels = np.ndarray
Rays = np.ndarray
Valid = np.ndarray
Params = np.ndarray

__all__ = [
    "CameraModel",
    "Points3D",
    "Pixels",
    "Rays",
    "Valid",
    "Params",
]


@runtime_checkable
class CameraModel(Protocol):
    """Structural interface every camera model satisfies.

    Models satisfy this by **shape** (duck typing) — they need not inherit from
    it — which keeps concrete models decoupled from ``core`` at runtime while
    static checkers still verify conformance. Services and the converter accept
    ``CameraModel`` and therefore work with any model interchangeably.
    """

    #: Short model identifier, e.g. ``"ds"`` or ``"kb"``.
    name: ClassVar[str]
    #: Parameter names in canonical order, e.g. ``("fx","fy","cx","cy","xi","alpha")``.
    param_names: ClassVar[Tuple[str, ...]]

    # -- parameter access -------------------------------------------------
    @property
    def params(self) -> Params:
        """Flat parameter vector ``(P,)`` in ``param_names`` order."""

    @property
    def K(self) -> np.ndarray:
        """3x3 pinhole intrinsic matrix ``[[fx,0,cx],[0,fy,cy],[0,0,1]]``."""

    @property
    def distortion(self) -> np.ndarray:
        """Model-specific distortion tail (the params after fx,fy,cx,cy)."""

    # -- core math (closed form, vectorized) ------------------------------
    def project(self, P: Points3D) -> Tuple[Pixels, Valid]:
        """Project camera-frame points ``(...,3)`` to pixels ``(...,2)`` + valid mask."""

    def unproject(self, uv: Pixels) -> Tuple[Rays, Valid]:
        """Unproject pixels ``(...,2)`` to unit rays ``(...,3)`` + valid mask."""

    # -- analytic Jacobians (no autodiff) ---------------------------------
    def project_jacobian(
        self, P: Points3D
    ) -> Tuple[Pixels, np.ndarray, np.ndarray, Valid]:
        """Return ``(uv, J_point, J_param, valid)`` with analytic derivatives.

        ``J_point`` is ``(N,2,3)`` = d(u,v)/d(x,y,z);
        ``J_param`` is ``(N,2,P)`` = d(u,v)/d(params).
        """

    # -- construction / bounds -------------------------------------------
    @classmethod
    def from_params(cls, p: Params) -> "CameraModel":
        """Build a model from a flat parameter vector in ``param_names`` order."""

    @classmethod
    def param_bounds(cls) -> Tuple[np.ndarray, np.ndarray]:
        """Lower/upper optimizer bounds ``(lb, ub)`` aligned with ``param_names``."""

    # -- conversion hook --------------------------------------------------
    def initialize_from_correspondences(
        self, K_seed: np.ndarray, rays: Rays, pixels: Pixels
    ) -> None:
        """Seed parameters (closed-form/linear) from ray<->pixel correspondences."""

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        """Serialize to a plain dict (``name`` + named parameters)."""

    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel":
        """Reconstruct from :meth:`to_dict` output."""
