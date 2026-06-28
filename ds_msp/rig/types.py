"""Backward-compatibility shim.

The rig data containers were promoted to the neutral :mod:`ds_msp.data` layer so that
``calib`` and ``io`` no longer have to import *up* into ``rig`` for shared types. This
module re-exports them so existing ``ds_msp.rig.types`` importers keep working.
"""

from __future__ import annotations

from ..data.observations import BoardObs, Object3D, ObjectObs, RigState

__all__ = ["BoardObs", "Object3D", "ObjectObs", "RigState"]
