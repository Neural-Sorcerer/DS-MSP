"""Backward-compatibility shim.

ChArUco detection was promoted to the :mod:`ds_msp.detect.charuco` layer so cv2 is
confined to the detection adapters and the geometry/solver path stays NumPy-native. This
module re-exports the public API (plus ``_frame_id_from_name`` used by ``rig.reconstruct``).
"""

from __future__ import annotations

from ..detect.charuco import (  # noqa: F401
    BoardSpec,
    _frame_id_from_name,
    board_object_points,
    detect_folder,
    detect_image,
    detect_rig,
    make_detectors,
    single_board_object,
)

__all__ = [
    "BoardSpec",
    "make_detectors",
    "board_object_points",
    "single_board_object",
    "detect_image",
    "detect_folder",
    "detect_rig",
]
