"""Single-file, MC-Calib-compatible entry point — ``calibrate_from_config(calib_param.yml)``.

MC-Calib is driven by exactly one config file: you edit ``calib_param.yml`` and run
``./calibrate calib_param.yml``. This module reads that *same* file (every key MC-Calib
reads, parsed with ``cv2.FileStorage`` so ``%YAML:1.0`` works) and runs the whole DS-MSP[rig]
pipeline from it — raw images or pre-detected keypoints in, MC-Calib-format results out —
so the tool is as easy to use as MC-Calib, with one extension: a *per-camera camera model*.

Config keys (MC-Calib's, plus the ``camera_models`` extension):

* board geometry — ``number_x_square``, ``number_y_square``, ``length_square``,
  ``length_marker``, ``square_size``, ``number_board`` (+ ``*_per_board`` overrides);
* models — ``distortion_model`` (0 Brown→``radtan``, 1 Kannala→``kb``),
  ``distortion_per_camera`` (per-camera 0/1), and the DS-MSP extension ``camera_models``
  (per-camera model name from {radtan,ucm,eucm,ds,kb,ocam}, highest precedence);
* input — ``number_camera``, ``root_path`` + ``cam_prefix`` (raw images) or
  ``keypoints_path`` (pre-detected; ``"None"`` ⇒ detect from images);
* intrinsics — ``fix_intrinsic`` (0/1), ``cam_params_path`` (initial intrinsics);
* output — ``save_path``, ``camera_params_file_name``, ``save_detection``,
  ``save_reprojection``.

Relative paths resolve against the config file's directory; pass ``overrides`` to retarget
them (e.g. point ``root_path`` at the real dataset). Multi-board *fused-object* geometry is
MC-Calib's board-group reconstruction — for ``number_board > 1`` the object model is loaded
from ``object_path`` / the keypoints directory; single-board objects are built from config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..calib.charuco import BoardSpec, detect_rig, single_board_object
from ..io.mccalib import (Scenario, _load_cameras, _load_detections, _load_groundtruth,
                          _load_object, radtan_from_cameragt)
from ..models.registry import canonical_name
from ..rig.types import ObjectObs
from .run import calibrate_scenario

_BROWN, _KANNALA = 0, 1
_DIST_TO_MODEL = {_BROWN: "radtan", _KANNALA: "kb"}


def _node(fs, name, default=None):
    n = fs.getNode(name)
    return n if not n.empty() else None


def _scalar(fs, name, default):
    n = fs.getNode(name)
    if n is None or n.empty():
        return default
    if n.isString():
        return n.string()
    return n.real()


def _int_seq(fs, name) -> List[int]:
    n = fs.getNode(name)
    if n is None or n.empty() or not n.isSeq():
        return []
    return [int(n.at(i).real()) for i in range(n.size())]


def _str_seq(fs, name) -> List[str]:
    n = fs.getNode(name)
    if n is None or n.empty() or not n.isSeq():
        return []
    return [n.at(i).string() for i in range(n.size())]


@dataclass
class RigConfig:
    """Parsed MC-Calib ``calib_param.yml`` (paths already resolved to absolute)."""
    boards: List[BoardSpec]
    number_camera: int
    camera_models: List[str]                    # one per camera, canonical names
    root_path: Optional[str]
    cam_prefix: str
    keypoints_path: Optional[str]
    fix_intrinsic: bool
    cam_params_path: Optional[str]
    save_path: Optional[str]
    camera_params_file_name: str
    save_detection: bool
    save_reprojection: bool
    ransac_threshold: float
    number_iterations: int
    he_approach: int
    object_path: Optional[str] = None
    raw: Dict = field(default_factory=dict)

    @property
    def number_board(self) -> int:
        return len(self.boards)


def _resolve(base_dir: str, path: Optional[str]) -> Optional[str]:
    if path is None or str(path).strip() in ("", "None"):
        return None
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir, path))


def _board_specs(fs) -> List[BoardSpec]:
    nb = int(_scalar(fs, "number_board", 1))
    nx = int(_scalar(fs, "number_x_square", 5))
    ny = int(_scalar(fs, "number_y_square", 5))
    ls = float(_scalar(fs, "length_square", 0.04))
    lm = float(_scalar(fs, "length_marker", 0.03))
    sq = float(_scalar(fs, "square_size", 1.0))
    nx_pb, ny_pb = _int_seq(fs, "number_x_square_per_board"), _int_seq(fs, "number_y_square_per_board")
    sq_pb = _int_seq(fs, "square_size_per_board")            # rarely set; falls back to sq
    specs = []
    for b in range(nb):
        specs.append(BoardSpec(
            n_x=nx_pb[b] if b < len(nx_pb) else nx,
            n_y=ny_pb[b] if b < len(ny_pb) else ny,
            length_square=ls, length_marker=lm,
            square_size=float(sq_pb[b]) if b < len(sq_pb) else sq))
    return specs


def _camera_models(fs, n_cam: int) -> List[str]:
    """Per-camera model: ``camera_models`` (extension) > ``distortion_per_camera`` > global
    ``distortion_model``. Names are canonicalized through the model registry."""
    names = _str_seq(fs, "camera_models")
    if names:
        if len(names) == 1:
            names = names * n_cam
        return [canonical_name(s) for s in names]
    per = _int_seq(fs, "distortion_per_camera")
    glob = int(_scalar(fs, "distortion_model", _BROWN))
    return [_DIST_TO_MODEL.get(per[c] if c < len(per) else glob, "radtan") for c in range(n_cam)]


def load_config(config_path: str, overrides: Optional[Dict] = None) -> RigConfig:
    """Parse a ``calib_param.yml`` into a :class:`RigConfig`. ``overrides`` replaces raw
    config values *before* path resolution (e.g. ``{"root_path": "/abs/Images"}``)."""
    base = os.path.dirname(os.path.abspath(config_path))
    fs = cv2.FileStorage(config_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(config_path)
    n_cam = int(_scalar(fs, "number_camera", 1))
    ov = overrides or {}

    def path_key(name, default=None):
        if name in ov:
            return _resolve(base, ov[name])
        return _resolve(base, _scalar(fs, name, default))

    cfg = RigConfig(
        boards=_board_specs(fs),
        number_camera=n_cam,
        camera_models=ov.get("camera_models") or _camera_models(fs, n_cam),
        root_path=path_key("root_path", "None"),
        cam_prefix=str(_scalar(fs, "cam_prefix", "Cam_")),
        keypoints_path=path_key("keypoints_path", "None"),
        fix_intrinsic=bool(int(ov.get("fix_intrinsic", _scalar(fs, "fix_intrinsic", 0)))),
        cam_params_path=path_key("cam_params_path", "None"),
        save_path=path_key("save_path", "None"),
        camera_params_file_name=str(_scalar(fs, "camera_params_file_name", "")),
        save_detection=bool(int(_scalar(fs, "save_detection", 0))),
        save_reprojection=bool(int(_scalar(fs, "save_reprojection", 0))),
        ransac_threshold=float(_scalar(fs, "ransac_threshold", 10.0)),
        number_iterations=int(_scalar(fs, "number_iterations", 1000)),
        he_approach=int(_scalar(fs, "he_approach", 0)),
        object_path=_resolve(base, ov.get("object_path")) if ov.get("object_path") else None,
        raw={"path": config_path},
    )
    fs.release()
    return cfg


def _find_object(cfg: RigConfig):
    """Single board → build from config; multi-board → load the fused object model."""
    if cfg.number_board == 1:
        return single_board_object(cfg.boards[0])
    for cand in (cfg.object_path,
                 os.path.join(os.path.dirname(cfg.keypoints_path or ""), "calibrated_objects_data.yml")
                 if cfg.keypoints_path else None,
                 os.path.join(cfg.save_path or "", "calibrated_objects_data.yml")
                 if cfg.save_path else None):
        if cand and os.path.exists(cand):
            return _load_object(cand)
    raise FileNotFoundError(
        f"number_board={cfg.number_board} needs a fused object model; set 'object_path' or "
        f"provide calibrated_objects_data.yml next to the keypoints/save path")


def _obs_from_keypoints(cfg: RigConfig, obj):
    per_cam, img_size = _load_detections(cfg.keypoints_path, obj)
    obs: List[ObjectObs] = []
    for cam_id, frames in per_cam.items():
        for frame_id, (rows, uvs) in frames.items():
            if rows:
                obs.append(ObjectObs(cam_id=cam_id, frame_id=frame_id, object_id=0,
                                     point_rows=np.array(rows, int),
                                     pts_2d=np.array(uvs, float)))
    return obs, img_size


def _gt_and_mccalib(cfg: RigConfig):
    """Best-effort GroundTruth.yml / calibrated_cameras for metrics; empty if absent."""
    anchor = cfg.root_path or cfg.keypoints_path or cfg.save_path or ""
    scn_dir = anchor
    for _ in range(3):                              # walk up to the scenario root
        scn_dir = os.path.dirname(scn_dir.rstrip("/"))
        if os.path.exists(os.path.join(scn_dir, "GroundTruth.yml")):
            break
    gt_path = os.path.join(scn_dir, "GroundTruth.yml")
    gt = _load_groundtruth(gt_path) if os.path.exists(gt_path) else {}
    mc_path = os.path.join(scn_dir, "Results", "calibrated_cameras_data.yml")
    mccalib = _load_cameras(mc_path)[0] if os.path.exists(mc_path) else {}
    return gt, mccalib


def calibrate_from_config(config_path: str, overrides: Optional[Dict] = None) -> Dict:
    """Run the full rig calibration from one MC-Calib ``calib_param.yml``.

    Detects ChArUco corners from ``root_path`` (or loads ``keypoints_path``), builds the
    per-camera model map, optimizes extrinsics-only (``fix_intrinsic=1``) or
    intrinsics+extrinsics, and writes MC-Calib-format results to ``save_path``. Returns the
    :func:`~ds_msp.rig.run.calibrate_scenario` result dict plus the parsed ``config``.
    """
    cfg = load_config(config_path, overrides)
    obj = _find_object(cfg)

    if cfg.keypoints_path:
        object_obs, img_size = _obs_from_keypoints(cfg, obj)
    elif cfg.root_path:
        cam_ids = list(range(cfg.number_camera))
        object_obs, img_size = detect_rig(cfg.root_path, cam_ids, cfg.boards, obj,
                                          cam_prefix=cfg.cam_prefix, min_corners=8)
    else:
        raise ValueError("config has neither keypoints_path nor root_path")

    cam_ids = sorted({o.cam_id for o in object_obs})
    spec = {c: cfg.camera_models[c] if c < len(cfg.camera_models) else cfg.camera_models[-1]
            for c in cam_ids}

    init_cameras = None
    if cfg.fix_intrinsic:
        if not cfg.cam_params_path or not os.path.exists(cfg.cam_params_path):
            raise FileNotFoundError("fix_intrinsic=1 needs cam_params_path with initial intrinsics")
        cams = _load_cameras(cfg.cam_params_path)[0]
        init_cameras = {c: radtan_from_cameragt(cams[c]) for c in cam_ids if c in cams}

    gt, mccalib = _gt_and_mccalib(cfg)
    scn = Scenario(name=os.path.basename(os.path.dirname(config_path)) or "rig", object=obj,
                   object_obs=object_obs, cam_ids=cam_ids, img_size=img_size,
                   gt=gt, mccalib=mccalib, mccalib_rms={})
    image_root = cfg.root_path if (cfg.save_reprojection and cfg.root_path) else None
    res = calibrate_scenario(scn, spec, fix_intrinsics=cfg.fix_intrinsic,
                             init_cameras=init_cameras, save_dir=cfg.save_path,
                             camera_params_file_name=cfg.camera_params_file_name,
                             image_root=image_root, cam_prefix=cfg.cam_prefix)
    res["config"] = cfg
    return res
