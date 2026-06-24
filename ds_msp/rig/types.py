"""Data structures for rig calibration — Python analogues of MC-Calib's
``BoardObs`` / ``Object3DObs`` / ``Object3D`` / ``CameraGroup``.

Pose convention (matches ``core.lie`` / ``calib.bundle``): every pose is a 4x4
homogeneous matrix ``T`` mapping a point in the *child* frame into the *parent*
frame, ``X_parent = T @ [X; 1]``. Frame names used throughout:

* ``T_co_b`` — board -> object  ("board-in-object"); lives in :class:`Object3D`.
* ``T_g_o``  — object -> group-reference  ("object-in-group"); per (object, frame).
* ``T_c_g``  — group-reference -> camera  ("camera-in-group", the extrinsic solved).

A reprojection composes board -> object -> group -> camera -> project, i.e.
``X_cam = T_c_g @ T_g_o @ T_co_b @ X_board``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.contracts import CameraModel


@dataclass
class BoardObs:
    """One planar board seen by one camera in one frame (cf. ``BoardObs.cpp``)."""

    cam_id: int
    frame_id: int
    board_id: int
    corner_ids: np.ndarray      # (K,) int  — board-local corner ids that were detected
    pts_2d: np.ndarray          # (K, 2)     — detected pixels
    T_c_b: Optional[np.ndarray] = None   # (4,4) board->camera from robust PnP
    valid: bool = True          # False if PnP inliers < 4 (BoardObs.cpp:149)


@dataclass
class Object3D:
    """Several planar boards fused into one rigid 3D point cloud (cf. ``Object3D.cpp``)."""

    object_id: int
    board_ids: List[int]
    ref_board_id: int                                   # min(board_ids) (McCalib.cpp:898)
    T_co_b: Dict[int, np.ndarray]                       # board_id -> (4,4) board->object
    pts_3d: np.ndarray                                  # (P, 3) all corners in object frame
    pts_obj_2_board: np.ndarray                         # (P, 2) [board_id, corner_id]
    pts_board_2_obj: Dict[Tuple[int, int], int]         # (board_id, corner_id) -> row

    def row_of(self, board_id: int, corner_id: int) -> int:
        return self.pts_board_2_obj[(int(board_id), int(corner_id))]


@dataclass
class ObjectObs:
    """One fused object seen by one camera in one frame (cf. ``Object3DObs.cpp``)."""

    cam_id: int
    frame_id: int
    object_id: int
    point_rows: np.ndarray      # (K,) int  — rows into Object3D.pts_3d
    pts_2d: np.ndarray          # (K, 2)
    T_c_o: Optional[np.ndarray] = None   # (4,4) object->camera from robust PnP


@dataclass
class RigState:
    """The optimization variable mutated by the staged global BA (``rig.ba``)."""

    cameras: Dict[int, CameraModel]                     # per-camera intrinsics
    T_c_g: Dict[int, np.ndarray]                        # camera-in-group; ref cam = identity
    ref_cam_id: int
    object_poses: Dict[Tuple[int, int], np.ndarray]     # (object_id, frame_id) -> T_g_o
    objects: Dict[int, Object3D]                        # holds T_co_b board poses
    img_size: Dict[int, Tuple[int, int]] = field(default_factory=dict)  # cam_id -> (w, h)
