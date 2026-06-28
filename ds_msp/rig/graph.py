"""Backward-compatibility shim.

Covisibility-graph utilities were promoted to :mod:`ds_msp.geometry.graph` (neutral,
reusable beyond rig). This module re-exports them for existing ``ds_msp.rig.graph``
importers.
"""

from __future__ import annotations

from ..geometry.graph import (
    Edge,
    connected_components,
    covis_weights,
    shortest_path,
)

__all__ = ["Edge", "covis_weights", "connected_components", "shortest_path"]
