"""Stereo relative-pose estimation recovers a known rigid transform."""

import cv2
import numpy as np
import pytest

from ds_msp.calib import estimate_relative_pose, relative_pose_error


def _T(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T


def test_recovers_known_transform_exactly():
    rng = np.random.default_rng(0)
    # ground-truth rig transform T_cam1_cam0
    R_rel, _ = cv2.Rodrigues(np.array([0.01, 0.05, -0.02]))
    t_rel = np.array([-0.101, -0.002, -0.001])

    poses0, poses1 = [], []
    for _ in range(20):
        rvec0 = rng.uniform(-0.5, 0.5, 3)
        tvec0 = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(1.0, 2.0)])
        R0, _ = cv2.Rodrigues(rvec0)
        # board seen by cam1: T_cam1_board = T_cam1_cam0 @ T_cam0_board
        R1 = R_rel @ R0
        t1 = R_rel @ tvec0 + t_rel
        poses0.append((rvec0, tvec0))
        poses1.append((cv2.Rodrigues(R1)[0].ravel(), t1))

    rig = estimate_relative_pose(poses0, poses1)
    err = relative_pose_error(rig["T"], _T(cv2.Rodrigues(R_rel)[0].ravel(), t_rel))
    assert err["rot_deg"] < 1e-6, err
    assert err["trans_mm"] < 1e-6, err
    assert rig["n"] == 20


def test_relative_pose_error_basic():
    A = _T([0, 0, 0], [0, 0, 0])
    B = _T([0, 0, np.deg2rad(2.0)], [0.01, 0, 0])
    e = relative_pose_error(A, B)
    assert abs(e["rot_deg"] - 2.0) < 1e-6
    assert abs(e["trans_mm"] - 10.0) < 1e-6


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        estimate_relative_pose([(np.zeros(3), np.zeros(3))], [])

# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("FR-CALIB-003")
