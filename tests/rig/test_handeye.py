"""Disjoint camera groups linked by hand-eye on the calibration-object motion.

MC-Calib forms a *camera group* per connected component of the camera co-visibility graph.
When the rig splits into several groups that never co-observe the object directly, the
inter-group transform is recovered from the object's motion seen by a reference camera in
each group (``handeye_bootstrap`` / ``link_groups``, geometrytools.cpp:621). This checks
that path recovers a known inter-group transform.
"""
import numpy as np

from ds_msp.core.lie import so3_exp
from ds_msp.rig.handeye import handeye_bootstrap, link_groups
from ds_msp.rig.types import ObjectObs


def _T(rvec, t):
    T = np.eye(4)
    T[:3, :3] = so3_exp(np.asarray(rvec, float))
    T[:3, 3] = t
    return T


def _object_traj(n, seed=0):
    """A diverse object-pose trajectory (rotation diversity is what hand-eye needs)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        out.append(_T(axis * rng.uniform(0.2, 1.0),
                      [rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(1.5, 2.5)]))
    return out


def test_handeye_bootstrap_recovers_known_transform():
    # group A ref = cam0 (at world origin); group B ref = cam2 with a known pose.
    T_c0_w = np.eye(4)
    T_c2_w = _T([0.05, 0.9, -0.1], [0.7, -0.2, 0.15])      # B-ref relative to world
    traj = _object_traj(30)
    poses_a = [T_c0_w @ Tw for Tw in traj]                 # object->cam0
    poses_b = [T_c2_w @ Tw for Tw in traj]                 # object->cam2
    # handeye_bootstrap estimates T_b_a (group-a-ref -> group-b-ref) = T_c2_w @ inv(T_c0_w)
    T_est = handeye_bootstrap(poses_a, poses_b, seed=0)
    T_gt = T_c2_w @ np.linalg.inv(T_c0_w)
    ang = np.degrees(np.arccos(np.clip(
        (np.trace(T_est[:3, :3].T @ T_gt[:3, :3]) - 1) / 2, -1, 1)))
    assert ang < 1.0, f"rotation off by {ang:.3f} deg"
    assert np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]) < 0.02


def test_link_groups_rebases_disjoint_rig():
    """Two groups {0,1} and {2,3} with known extrinsics, never auto-merged: link_groups
    recovers every camera's pose relative to the global reference (cam0)."""
    T = {0: np.eye(4),
         1: _T([0.0, 0.15, 0.0], [0.15, 0.0, 0.0]),       # cam1 rel world (≈cam0)
         2: _T([0.05, 0.9, -0.1], [0.7, -0.2, 0.15]),     # cam2 rel world (group B ref)
         3: _T([0.0, 1.05, 0.0], [0.85, 0.0, 0.1])}       # cam3 rel world
    groups = [[0, 1], [2, 3]]
    # intra-group extrinsics T_c_gref (group A ref = 0, group B ref = 2)
    extr = {0: np.eye(4), 1: T[1] @ np.linalg.inv(T[0]),
            2: np.eye(4), 3: T[3] @ np.linalg.inv(T[2])}
    traj = _object_traj(30, seed=1)
    obs = []
    for f, Tw in enumerate(traj):
        for c in (0, 1, 2, 3):
            o = ObjectObs(cam_id=c, frame_id=f, object_id=0,
                          point_rows=np.zeros(0, int), pts_2d=np.zeros((0, 2)))
            o.T_c_o = T[c] @ Tw
            obs.append(o)
    out = link_groups(groups, extr, obs)
    for c in (2, 3):
        gt = T[c] @ np.linalg.inv(T[0])                    # cam c relative to global ref cam0
        ang = np.degrees(np.arccos(np.clip(
            (np.trace(out[c][:3, :3].T @ gt[:3, :3]) - 1) / 2, -1, 1)))
        assert ang < 1.5, f"cam{c} rotation off by {ang:.3f} deg"
        assert np.linalg.norm(out[c][:3, 3] - gt[:3, 3]) < 0.05, f"cam{c} translation off"
