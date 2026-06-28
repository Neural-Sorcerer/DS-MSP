"""Unit tests for the from-scratch robust intrinsic seed + RANSAC PnP (no OpenCV).

These are the primitives that replaced the non-robust ``cv2.calibrateCamera`` /
``cv2.solvePnP`` seeding; the rig's outlier robustness rests on them, so they get direct
coverage: exact recovery on clean data, and inlier-correct recovery under gross outliers.
"""
import numpy as np
import pytest

from ds_msp.core.lie import so3_exp
from ds_msp.calib.robust_init import (dlt_projection, decompose_P, ransac_resection,
                                      intrinsics_seed, ransac_pnp_normalized)

W, H = 1280, 960
K_TRUE = np.array([[820.0, 0, 645.0], [0, 810.0, 470.0], [0, 0, 1.0]])


def _make_view(K, R, t, n=40, noise=0.0, outlier_frac=0.0, rng=None, w=W, h=H):
    rng = rng or np.random.default_rng(0)
    X = rng.uniform([-0.4, -0.4, -0.2], [0.4, 0.4, 0.2], size=(n, 3))
    Xc = X @ R.T + t
    uv = (Xc[:, :2] / Xc[:, 2:3]) @ np.diag([K[0, 0], K[1, 1]]) + [K[0, 2], K[1, 2]]
    if noise:
        uv = uv + rng.normal(scale=noise, size=uv.shape)
    if outlier_frac:
        bad = rng.random(n) < outlier_frac
        uv[bad] += rng.uniform(-40, 40, size=(int(bad.sum()), 2))
    return X, uv


def test_dlt_decompose_exact_on_clean_data():
    R, t = so3_exp([0.1, -0.2, 0.05]), np.array([0.1, -0.05, 2.2])
    X, uv = _make_view(K_TRUE, R, t, noise=0.0)
    K, Rr, tr = decompose_P(dlt_projection(X, uv))
    assert np.abs(K - K_TRUE).max() < 1e-6
    assert np.abs(Rr - R).max() < 1e-7
    assert np.abs(tr - t).max() < 1e-6


def test_ransac_resection_robust_to_gross_outliers():
    R, t = so3_exp([0.05, 0.15, -0.1]), np.array([-0.1, 0.08, 2.0])
    X, uv = _make_view(K_TRUE, R, t, noise=0.3, outlier_frac=0.20,
                       rng=np.random.default_rng(3))
    P, inl = ransac_resection(X, uv, thresh_px=3.0)
    assert P is not None and inl.sum() >= 0.7 * len(X)
    K, _, _ = decompose_P(P)
    assert 100 * abs(K[0, 0] - K_TRUE[0, 0]) / K_TRUE[0, 0] < 2.0
    assert 100 * abs(K[1, 1] - K_TRUE[1, 1]) / K_TRUE[1, 1] < 2.0


def test_intrinsics_seed_median_under_outliers():
    rng = np.random.default_rng(7)
    views = [_make_view(K_TRUE, so3_exp(rng.normal(scale=0.3, size=3)),
                        np.array([rng.uniform(-.2, .2), rng.uniform(-.2, .2),
                                  rng.uniform(1.8, 2.6)]),
                        noise=0.3, outlier_frac=0.15, rng=rng) for _ in range(24)]
    op = [v[0] for v in views]
    ip = [v[1] for v in views]
    K, _poses = intrinsics_seed(op, ip, W, H)
    assert 100 * abs(K[0, 0] - K_TRUE[0, 0]) / K_TRUE[0, 0] < 1.0
    assert 100 * abs(K[1, 1] - K_TRUE[1, 1]) / K_TRUE[1, 1] < 1.0


def test_ransac_pnp_normalized_robust_to_outliers():
    R, t = so3_exp([0.1, -0.2, 0.05]), np.array([0.1, -0.05, 2.2])
    rng = np.random.default_rng(5)
    X, pn = _make_view(np.eye(3), R, t, noise=0.0, rng=rng)   # K=I -> normalized coords
    bad = rng.random(len(pn)) < 0.20
    pn = pn.copy()
    pn[bad] += rng.uniform(-0.05, 0.05, size=(int(bad.sum()), 2))
    T, inl = ransac_pnp_normalized(X, pn, focal=800.0, thresh_px=3.0)
    assert T is not None
    assert np.abs(T[:3, :3] - R).max() < 1e-2
    assert np.abs(T[:3, 3] - t).max() < 1e-2


def test_resection_returns_none_when_underdetermined():
    X = np.random.default_rng(0).uniform(-1, 1, size=(5, 3))
    uv = np.random.default_rng(1).uniform(0, W, size=(5, 2))
    P, inl = ransac_resection(X, uv)
    assert P is None and not inl.any()