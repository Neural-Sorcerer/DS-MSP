"""Core contracts and shared primitives (dependency-free foundation layer)."""

from .contracts import (
    CameraModel,
    Params,
    Pixels,
    Points3D,
    Rays,
    Valid,
)
from .pinhole import balanced_pinhole_K

__all__ = ["CameraModel", "Points3D", "Pixels", "Rays", "Valid", "Params",
           "balanced_pinhole_K"]
