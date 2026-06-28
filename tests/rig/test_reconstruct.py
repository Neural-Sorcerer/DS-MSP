"""Multi-board fused-object reconstruction from raw detections (MC-Calib's
``calibrate3DObjects``, McCalib.cpp:765-956).

A 3-board rigid object viewed by a 2-camera rig: synthesize per-board ChArUco detections
(no object given), reconstruct the fused object, and check the recovered geometry matches
ground truth and that observations map onto it. This closes the parity gap where
``number_board > 1`` previously required a pre-built ``calibrated_objects_data.yml``.
"""
import numpy as np

from ds_msp.calib.charuco import BoardSpec, board_object_points
from ds_msp.core.lie import so3_exp
from ds_msp.models.radtan import RadTanModel
from ds_msp.rig.reconstruct import object_obs_from_board_obs, reconstruct_object
from ds_msp.rig.types import BoardObs

W, H, F = 1280, 960, 800.0


def _T(rvec, t):
    T = np.eye(4)
    T[:3, :3] = so3_exp(np.asarray(rvec, float))
    T[:3, 3] = t
    return T


def _make_board_obs(noise_px=0.1, seed=0):
    specs = [BoardSpec(5, 5, 0.04, 0.03, 0.1) for _ in range(3)]
    # ground-truth board->object poses; board 0 = object frame (its min id is the reference)
    T_co_b = {0: np.eye(4),
              1: _T([0.0, 0.6, 0.0], [0.45, 0.0, 0.25]),
              2: _T([-0.6, 0.0, 0.0], [0.0, 0.45, 0.25])}
    bp = {b: board_object_points(specs[b]) for b in range(3)}
    gt_pts = {}
    for b in range(3):
        Ph = np.c_[bp[b], np.ones(len(bp[b]))]
        gt_pts[b] = (T_co_b[b] @ Ph.T).T[:, :3]

    model = RadTanModel(F, F, W / 2, H / 2, -0.05, 0.01, 0.0, 0.0, 0.0)
    cams = {0: np.eye(4), 1: _T([0.0, 0.14, 0.0], [0.15, 0.0, 0.0])}
    rng = np.random.default_rng(seed)
    board_obs = []
    for fr in range(40):
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        Tg = _T(axis * rng.uniform(-0.5, 0.5),
                [rng.uniform(-0.25, 0.25), rng.uniform(-0.2, 0.2), rng.uniform(1.9, 2.5)])
        for c, Tcg in cams.items():
            for b in range(3):
                Xg = (Tg[:3, :3] @ gt_pts[b].T).T + Tg[:3, 3]
                Xc = (Tcg[:3, :3] @ Xg.T).T + Tcg[:3, 3]
                uv, val = model.project(Xc)
                inb = val & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                rows = np.where(inb)[0]
                if len(rows) < 6:
                    continue
                pts = uv[rows] + rng.normal(scale=noise_px, size=(len(rows), 2))
                board_obs.append(BoardObs(cam_id=c, frame_id=fr, board_id=b,
                                          corner_ids=rows, pts_2d=pts))
    return specs, board_obs, gt_pts, {0: (W, H), 1: (W, H)}


def test_reconstruct_fuses_all_boards_with_correct_geometry():
    specs, board_obs, gt_pts, img_size = _make_board_obs()
    obj = reconstruct_object(board_obs, specs, img_size)

    assert set(obj.board_ids) == {0, 1, 2}, "all three co-observed boards must fuse"
    # per-point error vs GT: object frame is board 0 in both, so no extra alignment needed
    err = []
    for r, (b, c) in enumerate(obj.pts_obj_2_board):
        err.append(np.linalg.norm(obj.pts_3d[r] - gt_pts[int(b)][int(c)]))
    err = np.array(err)
    assert np.median(err) < 3e-3, f"median geometry error {np.median(err) * 1e3:.2f} mm"
    assert err.max() < 1e-2, f"max geometry error {err.max() * 1e3:.2f} mm"


def test_object_obs_pool_boards_per_image():
    specs, board_obs, _gt, img_size = _make_board_obs()
    obj = reconstruct_object(board_obs, specs, img_size)
    obs = object_obs_from_board_obs(board_obs, obj)
    # one ObjectObs per (camera, frame) — all boards pooled into one pose constraint
    keys = {(o.cam_id, o.frame_id) for o in obs}
    assert len(obs) == len(keys), "boards in one image must pool into a single ObjectObs"
    # rows index the fused cloud, and pooling yields more corners than any single board
    assert all(o.point_rows.max() < len(obj.pts_3d) for o in obs)
    assert max(len(o.point_rows) for o in obs) > specs[0].n_corners


def test_reconstruct_with_model_aware_init_models():
    """``init_models`` resects each board with the camera's native model (MC-Calib's
    cam_params_path init) instead of the Brown bootstrap — required for wide-FOV lenses, and
    here it recovers the same correct fused geometry on the synthetic rig."""
    specs, board_obs, gt_pts, img_size = _make_board_obs()
    # the true generating model for every camera (a known prior, like cam_params_path)
    init_models = {0: RadTanModel(F, F, W / 2, H / 2, -0.05, 0.01, 0.0, 0.0, 0.0),
                   1: RadTanModel(F, F, W / 2, H / 2, -0.05, 0.01, 0.0, 0.0, 0.0)}
    obj = reconstruct_object(board_obs, specs, img_size, init_models=init_models)
    assert set(obj.board_ids) == {0, 1, 2}
    err = np.array([np.linalg.norm(obj.pts_3d[r] - gt_pts[int(b)][int(c)])
                    for r, (b, c) in enumerate(obj.pts_obj_2_board)])
    assert np.median(err) < 3e-3, f"median geometry error {np.median(err) * 1e3:.2f} mm"
    assert err.max() < 1e-2
