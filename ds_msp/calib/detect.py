"""Backward-compatibility shim.

AprilGrid detection was promoted to :mod:`ds_msp.detect.detect`. This module re-exports
its API (plus the ``_detect_union`` / ``_recover_missing`` helpers used by tests).
"""

from __future__ import annotations

from ..detect.detect import (  # noqa: F401
    _detect_union,
    _recover_missing,
    detect_aprilgrid,
)

__all__ = ["detect_aprilgrid"]
