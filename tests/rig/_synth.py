"""Synthetic multi-camera rig generator for rig-calibration tests.

Places N cameras at known extrinsics looking at a fused multi-board object, samples
random object poses over K frames, and projects with a RadTan model (+ optional pixel
noise). Ground-truth extrinsics are returned so every stage can be checked.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ds_msp.models.radtan import RadTanModel
from ds_msp.rig.types import Object3D, ObjectObs


def _grid_board(nx=4, ny=4, pitch=0.1) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(nx) * pitch, np.arange(ny) * pitch)
    return np.c_[xs.ravel(), ys.ravel(), np.zeros(nx * ny)]


def make_object(board_poses: Dict[int, np.ndarray], nx=4, ny=4, pitch=0.1) -> Object3D:
    """Fuse boards (given their board->object 4x4 poses) into one Object3D."""
    pts, rows, b2o = [], [], {}
    P_b = _grid_board(nx, ny, pitch)
    for bid, T in board_poses.items():
        P_o = (T @ np.c_[P_b, np.ones(len(P_b))].T).T[:, :3]
        for k, p in enumerate(P_o):
            b2o[(bid, k)] = len(pts)
            rows.append((bid, k))
            pts.append(p)
    return Object3D(object_id=0, board_ids=sorted(board_poses),
                    ref_board_id=min(board_poses),
                    T_co_b=dict(board_poses), pts_3d=np.array(pts),
                    pts_obj_2_board=np.array(rows, int), pts_board_2_obj=b2o)


def make_rig(n_cam=3, n_frame=40, noise_px=0.0, seed=0, w=1280, h=960,
             multi_board=True, model_factory=None
             ) -> Tuple[Object3D, List[ObjectObs], Dict, Dict, Dict]:
    """Return ``(object, object_obs, img_size, gt_extrinsics, gt_models)``.

    ``gt_extrinsics[c]`` is the ground-truth ``T_c_g`` (group-ref -> camera, cam 0 = id).
    ``gt_models[c]`` is the ground-truth camera model used to project for camera ``c``.

    ``model_factory(cam_id, rng) -> CameraModel`` lets the caller represent cameras with
    any model (DS/UCM/EUCM/KB/...) to exercise model-agnosticism. Defaults to RadTan.
    """
    rng = np.random.default_rng(seed)
    f = 800.0
    if model_factory is None:
        def model_factory(cam_id, rng):
            return RadTanModel(f, f, w / 2, h / 2, -0.05, 0.01, 0.0, 0.0, 0.0)
    gt_models = {c: model_factory(c, rng) for c in range(n_cam)}

    from ds_msp.core.lie import so3_exp as _exp
    boards = {0: np.eye(4)}
    if multi_board:
        # Genuinely 3D target (like a calibration cube): tilt the extra boards and offset
        # them in depth so every camera — even an obliquely angled one — sees a non-planar
        # point cloud. A near-coplanar target would leave each camera's focal ambiguous.
        T1 = np.eye(4); T1[:3, :3] = _exp([0.0, 0.6, 0.0]); T1[:3, 3] = [0.45, 0.0, 0.25]
        T2 = np.eye(4); T2[:3, :3] = _exp([-0.6, 0.0, 0.0]); T2[:3, 3] = [0.0, 0.45, 0.25]
        boards[1] = T1
        boards[2] = T2
    obj = make_object(boards)

    # cameras: ref at origin, others mildly spread on an arc, all keeping the object
    # well inside the frame (so every camera is well-conditioned for intrinsics).
    gt_ext: Dict[int, np.ndarray] = {0: np.eye(4)}
    for c in range(1, n_cam):
        ang = np.deg2rad(8.0 * c)
        R = np.array([[np.cos(ang), 0, np.sin(ang)],
                      [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
        t = np.array([0.15 * c, 0.0, 0.0])
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        gt_ext[c] = T

    object_obs: List[ObjectObs] = []
    for fr in range(n_frame):
        # random object pose in front of the rig
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        ang = rng.uniform(-0.55, 0.55)
        from ds_msp.core.lie import so3_exp
        Rg = so3_exp(axis * ang)
        # sweep the target over a good fraction of the image (so each model's distortion
        # is observable -> focal well constrained) while keeping most views full.
        tg = np.array([rng.uniform(-0.35, 0.35), rng.uniform(-0.3, 0.3),
                       rng.uniform(1.8, 2.6)])
        T_g_o = np.eye(4); T_g_o[:3, :3] = Rg; T_g_o[:3, 3] = tg
        Xg = (T_g_o[:3, :3] @ obj.pts_3d.T).T + T_g_o[:3, 3]
        for c in range(n_cam):
            Xc = (gt_ext[c][:3, :3] @ Xg.T).T + gt_ext[c][:3, 3]
            uv, valid = gt_models[c].project(Xc)
            inb = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            rows = np.where(inb)[0]
            if len(rows) < 6:
                continue
            pts = uv[rows] + (rng.normal(scale=noise_px, size=(len(rows), 2))
                              if noise_px else 0.0)
            object_obs.append(ObjectObs(cam_id=c, frame_id=fr, object_id=0,
                                        point_rows=rows, pts_2d=pts))
    img_size = {c: (w, h) for c in range(n_cam)}
    return obj, object_obs, img_size, gt_ext, gt_models
