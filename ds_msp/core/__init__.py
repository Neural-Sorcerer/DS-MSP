"""Core contracts and shared primitives (dependency-free foundation layer)."""

from .contracts import (
    CameraModel,
    Params,
    Pixels,
    Points3D,
    Rays,
    Valid,
)

__all__ = ["CameraModel", "Points3D", "Pixels", "Rays", "Valid", "Params"]
