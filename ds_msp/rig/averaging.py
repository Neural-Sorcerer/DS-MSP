"""Backward-compatibility shim.

Robust pose-averaging primitives were promoted to :mod:`ds_msp.geometry.averaging`
(neutral, reusable beyond rig). This module re-exports them for existing
``ds_msp.rig.averaging`` importers.
"""

from __future__ import annotations

from ..geometry.averaging import (
    average_rotation,
    average_transform,
    average_translation,
    mean_transform,
    robust_average_transform,
)

__all__ = [
    "average_rotation",
    "average_translation",
    "average_transform",
    "robust_average_transform",
    "mean_transform",
]
