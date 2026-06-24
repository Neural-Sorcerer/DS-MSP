"""Reader for MC-Calib's Blender benchmark data (OpenCV YAML via ``cv2.FileStorage``).

Loads the detected keypoints, the fused 3D object, the synthetic ground truth, and
MC-Calib's own calibration result — everything needed to drive ``rig.calibrate_rig`` on
identical 2D observations and compare extrinsics against both references.

File formats (per scenario directory ``<scn>/Results/``):

* ``detected_keypoints_data.yml`` — per ``camera_<i>``: ``frame_idxs`` / ``board_idxs``
  (flat, one entry per board-observation) and ``pts_2d`` / ``charuco_idxs`` (parallel
  sequences of flattened ``[u,v,...]`` and corner-id arrays).
* ``calibrated_objects_data.yml`` — ``object_<j>.points``: a ``(5, N)`` matrix whose rows
  are ``[x, y, z, board_id, corner_id]`` in the object frame.
* ``calibrated_cameras_data.yml`` — per camera: ``camera_matrix``, ``distortion_vector``,
  ``camera_pose_matrix`` (group-ref -> camera), ``img_width/height``.
* ``<scn>/GroundTruth.yml`` — ``K_<i>`` and ``P_<i>`` (4x4 pose) per camera.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..models.radtan import RadTanModel
from ..rig.types import Object3D, ObjectObs


def _seq(node) -> list:
    return [node.at(i) for i in range(node.size())]


def _flat(node) -> np.ndarray:
    """Read an inline YAML sequence ``[ ... ]`` as a flat float array."""
    return np.array([node.at(i).real() for i in range(node.size())], float)


@dataclass
class CameraGT:
    K: np.ndarray
    dist: Optional[np.ndarray]
    pose: np.ndarray            # 4x4 (group-ref -> camera convention, as stored)


@dataclass
class Scenario:
    name: str
    object: Object3D
    object_obs: List[ObjectObs]                 # one per (camera, frame)
    cam_ids: List[int]
    img_size: Dict[int, Tuple[int, int]]
    gt: Dict[int, CameraGT]
    mccalib: Dict[int, CameraGT]
    mccalib_rms: Dict[int, float]


def _load_object(path: str) -> Object3D:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    M = fs.getNode("object_0").getNode("points").mat()          # (5, N)
    fs.release()
    xyz = M[:3].T.astype(float)
    board_ids = M[3].astype(int)
    corner_ids = M[4].astype(int)
    b2o = {(int(b), int(c)): i for i, (b, c) in enumerate(zip(board_ids, corner_ids))}
    boards = sorted(set(int(b) for b in board_ids))
    return Object3D(
        object_id=0, board_ids=boards, ref_board_id=min(boards),
        T_co_b={b: np.eye(4) for b in boards},                  # baked into pts_3d
        pts_3d=xyz, pts_obj_2_board=np.c_[board_ids, corner_ids],
        pts_board_2_obj=b2o,
    )


def _load_detections(path: str, obj: Object3D):
    """Return ``{cam_id: {frame_id: (point_rows, pts_2d)}}`` and per-cam image size."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    nb = int(fs.getNode("nb_camera").real())
    per_cam: Dict[int, Dict[int, Tuple[list, list]]] = {}
    img_size: Dict[int, Tuple[int, int]] = {}
    for ci in range(nb):
        cn = fs.getNode(f"camera_{ci}")
        img_size[ci] = (int(cn.getNode("img_width").real()),
                        int(cn.getNode("img_height").real()))
        frame_idxs = _flat(cn.getNode("frame_idxs")).astype(int)
        board_idxs = _flat(cn.getNode("board_idxs")).astype(int)
        pts_seq = _seq(cn.getNode("pts_2d"))
        cid_seq = _seq(cn.getNode("charuco_idxs"))
        frames: Dict[int, Tuple[list, list]] = {}
        for k in range(len(frame_idxs)):
            f = int(frame_idxs[k]); b = int(board_idxs[k])
            pts = _flat(pts_seq[k]).reshape(-1, 2)
            cids = _flat(cid_seq[k]).astype(int).ravel()
            rows, uvs = frames.setdefault(f, ([], []))
            for cid, uv in zip(cids, pts):
                key = (b, int(cid))
                if key in obj.pts_board_2_obj:
                    rows.append(obj.pts_board_2_obj[key])
                    uvs.append(uv)
        per_cam[ci] = frames
    fs.release()
    return per_cam, img_size


def _load_cameras(path: str) -> Tuple[Dict[int, CameraGT], Dict[int, float]]:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    nb = int(fs.getNode("nb_camera").real())
    cams: Dict[int, CameraGT] = {}
    for ci in range(nb):
        cn = fs.getNode(f"camera_{ci}")
        K = cn.getNode("camera_matrix").mat()
        dist = cn.getNode("distortion_vector").mat()
        pose = cn.getNode("camera_pose_matrix").mat()
        cams[ci] = CameraGT(K=K, dist=dist.ravel() if dist is not None else None, pose=pose)
    fs.release()
    return cams, {}


def _load_groundtruth(path: str) -> Dict[int, CameraGT]:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    nb = int(fs.getNode("nb_camera").real())
    gt: Dict[int, CameraGT] = {}
    for ci in range(1, nb + 1):
        K = fs.getNode(f"K_{ci}").mat()
        P = fs.getNode(f"P_{ci}").mat()
        gt[ci - 1] = CameraGT(K=K, dist=None, pose=P)
    fs.release()
    return gt


def load_scenario(scn_dir: str) -> Scenario:
    """Load a ``Blender_Images/Scenario_*`` directory into a :class:`Scenario`."""
    name = os.path.basename(scn_dir.rstrip("/"))
    results = os.path.join(scn_dir, "Results")
    obj = _load_object(os.path.join(results, "calibrated_objects_data.yml"))
    per_cam, img_size = _load_detections(
        os.path.join(results, "detected_keypoints_data.yml"), obj)
    mccalib, _ = _load_cameras(os.path.join(results, "calibrated_cameras_data.yml"))
    gt_path = os.path.join(scn_dir, "GroundTruth.yml")
    gt = _load_groundtruth(gt_path) if os.path.exists(gt_path) else {}

    object_obs: List[ObjectObs] = []
    for cam_id, frames in per_cam.items():
        for frame_id, (rows, uvs) in frames.items():
            if not rows:
                continue
            object_obs.append(ObjectObs(
                cam_id=cam_id, frame_id=frame_id, object_id=0,
                point_rows=np.array(rows, int), pts_2d=np.array(uvs, float),
            ))
    return Scenario(
        name=name, object=obj, object_obs=object_obs,
        cam_ids=sorted(per_cam), img_size=img_size,
        gt=gt, mccalib=mccalib, mccalib_rms={},
    )


def radtan_from_cameragt(cam: CameraGT) -> RadTanModel:
    """Build a DS-MSP ``RadTanModel`` from an MC-Calib camera (distortion_type 0)."""
    K = cam.K
    d = cam.dist if cam.dist is not None else np.zeros(5)
    d = np.asarray(d, float).ravel()
    k1, k2, p1, p2, k3 = (list(d) + [0, 0, 0, 0, 0])[:5]
    return RadTanModel(K[0, 0], K[1, 1], K[0, 2], K[1, 2], k1, k2, p1, p2, k3)
