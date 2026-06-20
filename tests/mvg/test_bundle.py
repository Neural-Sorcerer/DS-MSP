"""Angular two-view bundle refinement (C5): tightens the algebraic estimate on noisy rays."""

import numpy as np

from ds_msp.mvg import recover_pose
from ds_msp.mvg.bundle import angular_reprojection_error, refine_two_view


def _rot(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _rot_err_deg(R, Rh):
    return np.degrees(np.arccos(np.clip((np.trace(R.T @ Rh) - 1) / 2, -1, 1)))


def _scene(n=50, seed=0, noise=2e-3):
    rng = np.random.default_rng(seed)
    R = _rot(rng.standard_normal(3), 0.6)
    t = rng.standard_normal(3)
    t = t / np.linalg.norm(t)
    X1 = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-2, 2, n), rng.uniform(2, 8, n)])
    X2 = (R @ X1.T).T + t
    f1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
    f2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
    f1 = f1 + noise * rng.standard_normal(f1.shape)
    f2 = f2 + noise * rng.standard_normal(f2.shape)
    f1 /= np.linalg.norm(f1, axis=1, keepdims=True)
    f2 /= np.linalg.norm(f2, axis=1, keepdims=True)
    return f1, f2, R, t


def test_angular_error_zero_for_exact_geometry():
    f1, f2, R, t = _scene(noise=0.0)
    R0, t0, X = recover_pose(f1, f2)
    assert angular_reprojection_error(f1, f2, R0, t0, X).max() < 1e-4


def test_refinement_reduces_angular_reprojection_error():
    f1, f2, R, t = _scene(seed=1, noise=3e-3)
    R0, t0, X0 = recover_pose(f1, f2)
    before = angular_reprojection_error(f1, f2, R0, t0, X0).mean()
    Rr, tr, Xr = refine_two_view(f1, f2, R0, t0, X0)
    after = angular_reprojection_error(f1, f2, Rr, tr, Xr).mean()
    assert after < before                                  # nonlinear refinement tightens the fit


def test_refinement_improves_pose_accuracy_on_average():
    pose_lin, pose_ref = [], []
    for s in range(8):
        f1, f2, R, t = _scene(seed=s, noise=3e-3)
        R0, t0, X0 = recover_pose(f1, f2)
        Rr, tr, Xr = refine_two_view(f1, f2, R0, t0, X0)
        pose_lin.append(_rot_err_deg(R, R0))
        pose_ref.append(_rot_err_deg(R, Rr))
    assert np.mean(pose_ref) <= np.mean(pose_lin) + 1e-9   # refinement never worse on average
