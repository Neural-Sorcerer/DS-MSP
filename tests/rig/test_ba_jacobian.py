"""The critical test: the analytic BA Jacobian must match finite differences.

Guards the board->object->camera->project chain (implementation doc §9.1). A regression
here is the classic source of a BA that "converges" to the wrong answer.
"""

import numpy as np

from ds_msp.rig import ba
from ds_msp.rig.rig_calibrate import _front_end_opencv
from ds_msp.rig.types import RigState
from ._synth import make_rig


def _build_small_rig():
    obj, obs, img_size, gt_ext, model = make_rig(n_cam=2, n_frame=4, seed=3)
    from collections import defaultdict
    by_cam = defaultdict(list)
    for o in obs:
        by_cam[o.cam_id].append(o)
    cams = _front_end_opencv(obj, by_cam, img_size)
    # build a RigState from GT extrinsics + per-frame object poses (rough init via T_c_o)
    object_poses = {}
    for o in obs:
        key = (o.object_id, o.frame_id)
        if key not in object_poses and o.cam_id == 0 and o.T_c_o is not None:
            object_poses[key] = o.T_c_o
    for o in obs:                                   # fill any frames cam0 missed
        key = (o.object_id, o.frame_id)
        if key not in object_poses and o.T_c_o is not None:
            object_poses[key] = np.linalg.inv(gt_ext[o.cam_id]) @ o.T_c_o
    return RigState(cameras=cams, T_c_g=dict(gt_ext), ref_cam_id=0,
                    object_poses=object_poses, objects={0: obj}, img_size=img_size), obs


def _check(fix_intrinsics, fix_extrinsics=False):
    rig, obs = _build_small_rig()
    state0, residual, jacobian, retract, K = ba.build_problem(
        rig, obs, fix_intrinsics=fix_intrinsics, fix_extrinsics=fix_extrinsics)
    J = jacobian(state0)
    r0 = residual(state0)
    eps = 1e-6
    rng = np.random.default_rng(1)
    # check a random subset of tangent directions
    for j in rng.choice(K, size=min(K, 25), replace=False):
        d = np.zeros(K); d[j] = eps
        rp = residual(retract(state0, d))
        rm = residual(retract(state0, -d))
        fd = (rp - rm) / (2 * eps)
        assert np.allclose(J[:, j], fd, atol=1e-3, rtol=1e-3), \
            f"Jacobian column {j} mismatch (fix_intrinsics={fix_intrinsics}, " \
            f"fix_extrinsics={fix_extrinsics})"


def test_jacobian_poses_only():
    _check(fix_intrinsics=True)


def test_jacobian_with_intrinsics():
    _check(fix_intrinsics=False)


def test_jacobian_object_poses_only():
    # the per-object intermediate stage: cameras + intrinsics fixed, only object poses
    _check(fix_intrinsics=True, fix_extrinsics=True)
