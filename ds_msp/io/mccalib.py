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


def _T_to_rodrigues(T: np.ndarray):
    """4x4 transform -> (rvec(3,), tvec(3,)) like MC-Calib's getPoseVec."""
    rvec = cv2.Rodrigues(np.asarray(T[:3, :3], float))[0].ravel()
    return rvec, np.asarray(T[:3, 3], float).ravel()


def save_mccalib_cameras(rig, path: str, *, cam_groups: Optional[Dict[int, int]] = None,
                         cam_order=None) -> None:
    """Write ``calibrated_cameras_data.yml`` in MC-Calib's exact OpenCV-YAML schema.

    Per ``Calibration::saveCamerasParams`` (McCalib.cpp:386): a top-level ``nb_camera`` and,
    for each camera, a ``camera_<i>`` map with ``camera_matrix`` (3x3), ``distortion_vector``
    (1xN), ``camera_model`` (string), ``camera_group``, ``img_width``, ``img_height`` and
    ``camera_pose_matrix`` — the **camera->world** pose, i.e. ``inv(T_c_g)`` (MC-Calib writes
    ``getCameraPoseMat().inv()``; ``RigState.T_c_g`` is the world->camera projection extrinsic).
    """
    from ..models.registry import mccalib_name
    order = list(cam_order) if cam_order is not None else sorted(rig.cameras)
    groups = cam_groups or {c: 0 for c in order}
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("nb_camera", int(len(order)))
    for c in order:
        model = rig.cameras[c]
        w, h = rig.img_size.get(c, (0, 0))
        cam2world = np.linalg.inv(np.asarray(rig.T_c_g[c], float))
        fs.startWriteStruct(f"camera_{c}", cv2.FileNode_MAP)
        fs.write("camera_matrix", np.asarray(model.K, float))
        fs.write("distortion_vector",
                 np.asarray(model.distortion, float).reshape(1, -1))
        fs.write("camera_model", mccalib_name(model.name))
        fs.write("camera_group", int(groups.get(c, 0)))
        fs.write("img_width", int(w))
        fs.write("img_height", int(h))
        fs.write("camera_pose_matrix", cam2world)
        fs.endWriteStruct()
    fs.release()


def save_mccalib_objects(obj: Object3D, path: str) -> None:
    """Write ``calibrated_objects_data.yml`` (McCalib.cpp:427): per ``object_<j>`` a
    ``points`` matrix of shape ``(5, N)`` whose rows are ``[x, y, z, board_id, corner_id]``."""
    rows = obj.pts_obj_2_board                                   # (N,2) = [board_id, corner_id]
    pts = np.vstack([obj.pts_3d.T, rows[:, 0], rows[:, 1]]).astype(np.float32)  # (5,N)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.startWriteStruct(f"object_{obj.object_id}", cv2.FileNode_MAP)
    fs.write("points", pts)
    fs.endWriteStruct()
    fs.release()


def save_mccalib_object_poses(rig, path: str, *, object_id: int = 0) -> None:
    """Write ``calibrated_objects_pose_data.yml`` (McCalib.cpp:469): per ``object_<j>`` a
    ``poses`` matrix ``(6, M)`` of ``[rx, ry, rz, tx, ty, tz]`` over the frames the object is
    seen, ``T_g_o`` (object->group)."""
    keys = sorted(k for k in rig.object_poses if k[0] == object_id)
    pose_mat = np.zeros((6, len(keys)), float)
    for a, key in enumerate(keys):
        rvec, tvec = _T_to_rodrigues(rig.object_poses[key])
        pose_mat[:3, a] = rvec
        pose_mat[3:, a] = tvec
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.startWriteStruct(f"object_{object_id}", cv2.FileNode_MAP)
    fs.write("poses", pose_mat)
    fs.endWriteStruct()
    fs.release()


def _obs_reprojection(rig, o):
    """Detected vs reprojected pixels for one ObjectObs: ``(uv_det, uv_rep, valid)``.

    Reproject the object's 3D points through ``T_c_o = T_c_g[cam] @ T_g_o`` and the camera's
    model — the same composition MC-Calib uses (``getCameraPoseMat * getPoseInGroupMat``)."""
    cam = o.cam_id
    key = (o.object_id, o.frame_id)
    if cam not in rig.cameras or key not in rig.object_poses or cam not in rig.T_c_g:
        return None
    obj = next(iter(rig.objects.values()))
    X = obj.pts_3d[o.point_rows]
    T_c_o = np.asarray(rig.T_c_g[cam], float) @ np.asarray(rig.object_poses[key], float)
    Xc = (T_c_o[:3, :3] @ X.T).T + T_c_o[:3, 3]
    uv_rep, valid = rig.cameras[cam].project(Xc)
    return np.asarray(o.pts_2d, float), uv_rep, valid


def save_mccalib_reprojection_error(rig, object_obs, path: str, *, cam_group: int = 0) -> None:
    """Write ``reprojection_error_data.yml`` in MC-Calib's schema (McCalib.cpp:2278):
    ``nb_camera_group`` then per ``camera_group_<g>`` a ``frame_<idx>`` map holding, per
    ``camera_<id>``, ``nb_pts`` and an ``error_list`` (1xN per-point pixel distances), plus a
    ``camera_list`` per frame and a ``frame_list`` for the group."""
    by_frame: Dict[int, List] = {}
    for o in object_obs:
        by_frame.setdefault(o.frame_id, []).append(o)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("nb_camera_group", 1)
    fs.startWriteStruct(f"camera_group_{cam_group}", cv2.FileNode_MAP)
    frame_list = []
    for fr in sorted(by_frame):
        cam_list = []
        fs.startWriteStruct(f"frame_{fr}", cv2.FileNode_MAP)
        for o in by_frame[fr]:
            rep = _obs_reprojection(rig, o)
            if rep is None:
                continue
            uv_det, uv_rep, valid = rep
            err = np.linalg.norm(uv_rep[valid] - uv_det[valid], axis=1)
            cam_list.append(o.cam_id)
            fs.startWriteStruct(f"camera_{o.cam_id}", cv2.FileNode_MAP)
            fs.write("nb_pts", int(valid.sum()))
            fs.write("error_list", err.reshape(1, -1).astype(np.float64))
            fs.endWriteStruct()
        fs.write("camera_list", np.asarray(cam_list, np.int32).reshape(-1, 1))
        fs.endWriteStruct()
        frame_list.append(fr)
    fs.write("frame_list", np.asarray(frame_list, np.int32).reshape(-1, 1))
    fs.endWriteStruct()
    fs.release()


def save_reprojection_images(rig, object_obs, image_root: str, save_dir: str, *,
                             cam_prefix: str = "Cam_", ext: str = "png") -> int:
    """Draw detected (green) vs reprojected (red) corners per frame and save under
    ``<save_dir>/Reprojection/<cam:03d>/<frame:06d>.jpg`` — the MC-Calib layout
    (McCalib.cpp:1923). Images are looked up as ``<image_root>/<cam_prefix><cam+1:03d>/
    <frame+1:05d>.<ext>``. Returns the number of images written (0 if no images found)."""
    root = os.path.join(save_dir, "Reprojection")
    written = 0
    by_cf: Dict[Tuple[int, int], List] = {}
    for o in object_obs:
        by_cf.setdefault((o.cam_id, o.frame_id), []).append(o)
    for (cam, fr), obs_list in by_cf.items():
        img_path = None
        for cand in (f"{fr + 1:05d}.{ext}", f"{fr:05d}.{ext}", f"{fr + 1:06d}.{ext}"):
            p = os.path.join(image_root, f"{cam_prefix}{cam + 1:03d}", cand)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            continue
        image = cv2.imread(img_path)
        if image is None:
            continue
        for o in obs_list:
            rep = _obs_reprojection(rig, o)
            if rep is None:
                continue
            uv_det, uv_rep, valid = rep
            for i in np.where(valid)[0]:
                cv2.circle(image, (int(round(uv_rep[i, 0])), int(round(uv_rep[i, 1]))), 4,
                           (0, 0, 255), cv2.FILLED, 8)
                cv2.circle(image, (int(round(uv_det[i, 0])), int(round(uv_det[i, 1]))), 4,
                           (0, 255, 0), cv2.FILLED, 8)
        out_dir = os.path.join(root, f"{cam:03d}")
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"{fr:06d}.jpg"), image)
        written += 1
    return written


def save_mccalib_results(rig, save_dir: str, *, object3d: Optional[Object3D] = None,
                         object_obs=None, cam_groups: Optional[Dict[int, int]] = None,
                         camera_params_file_name: str = "") -> Dict[str, str]:
    """Write the full MC-Calib result set into ``save_dir`` and return the paths written.

    Always writes ``calibrated_cameras_data.yml`` (or ``camera_params_file_name`` if given)
    and ``calibrated_objects_pose_data.yml``; writes ``calibrated_objects_data.yml`` when an
    ``Object3D`` is provided (or taken from ``rig.objects``).
    """
    os.makedirs(save_dir, exist_ok=True)
    cam_name = camera_params_file_name or "calibrated_cameras_data.yml"
    paths = {"cameras": os.path.join(save_dir, cam_name),
             "object_poses": os.path.join(save_dir, "calibrated_objects_pose_data.yml")}
    save_mccalib_cameras(rig, paths["cameras"], cam_groups=cam_groups)
    save_mccalib_object_poses(rig, paths["object_poses"])
    obj = object3d if object3d is not None else next(iter(getattr(rig, "objects", {}).values()), None)
    if obj is not None:
        paths["objects"] = os.path.join(save_dir, "calibrated_objects_data.yml")
        save_mccalib_objects(obj, paths["objects"])
    if object_obs is not None:
        paths["reprojection_error"] = os.path.join(save_dir, "reprojection_error_data.yml")
        save_mccalib_reprojection_error(rig, object_obs, paths["reprojection_error"])
    return paths


def radtan_from_cameragt(cam: CameraGT) -> RadTanModel:
    """Build a DS-MSP ``RadTanModel`` from an MC-Calib camera (distortion_type 0)."""
    K = cam.K
    d = cam.dist if cam.dist is not None else np.zeros(5)
    d = np.asarray(d, float).ravel()
    k1, k2, p1, p2, k3 = (list(d) + [0, 0, 0, 0, 0])[:5]
    return RadTanModel(K[0, 0], K[1, 1], K[0, 2], K[1, 2], k1, k2, p1, p2, k3)
