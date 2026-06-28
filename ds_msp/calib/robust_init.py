"""Backward-compatibility shim.

The robust resection / PnP primitives were promoted to the neutral
:mod:`ds_msp.geometry.resection` layer so that both single-camera (``calib``) and
multi-camera (``rig``) calibration share one implementation. This module re-exports them
for existing ``ds_msp.calib.robust_init`` importers.
"""

from __future__ import annotations

from ..geometry.resection import (
    decompose_P,
    dlt_projection,
    intrinsics_seed,
    ransac_pnp_normalized,
    ransac_resection,
)

__all__ = [
    "dlt_projection",
    "decompose_P",
    "ransac_resection",
    "intrinsics_seed",
    "ransac_pnp_normalized",
]
