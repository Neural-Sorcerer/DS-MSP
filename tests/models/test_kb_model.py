"""KB unit tests: exact agreement with cv2.fisheye (external oracle) + interop."""

import numpy as np
import cv2

from ds_msp.models.kb import KannalaBrandtModel
from ds_msp.testing import sample_forward_points


def test_kb_matches_opencv_fisheye_project():
    m = KannalaBrandtModel.sample()
    P = sample_forward_points()
    uv, _ = m.project(P)
    obj = P.reshape(-1, 1, 3).astype(np.float64)
    cv_uv, _ = cv2.fisheye.projectPoints(
        obj, np.zeros(3), np.zeros(3), m.K, m.distortion.reshape(4, 1))
    assert np.allclose(uv, cv_uv.reshape(-1, 2), atol=1e-9)


def test_kb_K_and_distortion_are_opencv_ready():
    m = KannalaBrandtModel.sample()
    assert m.K.shape == (3, 3)
    assert m.distortion.shape == (4,)
    # Should be directly usable by cv2.fisheye.undistortPoints without error.
    pts = np.array([[[330.0, 250.0]]], dtype=np.float64)
    out = cv2.fisheye.undistortPoints(pts, m.K, m.distortion)
    assert out.shape == (1, 1, 2)


def test_kb_roundtrip():
    m = KannalaBrandtModel.sample()
    P = sample_forward_points()
    uv, v1 = m.project(P)
    rays, v2 = m.unproject(uv)
    ok = v1 & v2
    d = P[ok] / np.linalg.norm(P[ok], axis=1, keepdims=True)
    cos = np.sum(rays[ok] * d, axis=1)
    assert (cos > 1 - 1e-5).all()
