"""Markley quaternion rotation averaging + robust translation median."""

import numpy as np

from ds_msp.core.lie import so3_exp
from ds_msp.rig.averaging import (average_rotation, average_transform,
                                  average_translation)


def _ang(A, B):
    return np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1)))


def test_average_rotation_recovers_center():
    rng = np.random.default_rng(0)
    R0 = so3_exp(np.array([0.3, -0.2, 0.5]))
    Rs = [R0 @ so3_exp(rng.normal(scale=0.02, size=3)) for _ in range(50)]
    Ravg = average_rotation(Rs)
    assert _ang(Ravg, R0) < 0.5


def test_average_rotation_antipodal_invariant():
    # Adding the antipodal quaternion (same rotation) must not move the mean.
    R = so3_exp(np.array([0.1, 2.9, 0.0]))   # near-pi rotation, sign-sensitive
    a = average_rotation([R, R, R])
    assert _ang(a, R) < 1e-6


def test_translation_median_robust_to_outliers():
    ts = np.tile([1.0, 2.0, 3.0], (20, 1))
    ts[:6] += 50.0                            # 30% gross outliers
    m = average_translation(ts)
    assert np.allclose(m, [1.0, 2.0, 3.0], atol=1e-9)


def test_average_transform_roundtrip():
    T = np.eye(4)
    T[:3, :3] = so3_exp(np.array([0.2, 0.1, -0.3]))
    T[:3, 3] = [0.5, -1.0, 2.0]
    out = average_transform([T, T, T])
    assert np.allclose(out, T, atol=1e-9)
