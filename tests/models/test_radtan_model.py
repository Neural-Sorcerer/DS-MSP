"""RadTan unit tests: exact agreement with cv2.projectPoints (external oracle)."""

import numpy as np
import cv2

from ds_msp.models.radtan import RadTanModel
from ds_msp.testing import sample_forward_points


def test_radtan_matches_opencv_projectpoints():
    m = RadTanModel.sample()
    P = sample_forward_points()
    uv, _ = m.project(P)
    cv_uv, _ = cv2.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                 m.K, m.distortion)
    assert np.allclose(uv, cv_uv.reshape(-1, 2), atol=1e-9)


def test_radtan_distortion_is_opencv_order():
    m = RadTanModel(600, 600, 320, 240, k1=0.1, k2=0.2, p1=0.01, p2=0.02, k3=0.3)
    assert np.allclose(m.distortion, [0.1, 0.2, 0.01, 0.02, 0.3])  # k1,k2,p1,p2,k3


def test_radtan_rejects_behind_camera():
    m = RadTanModel.sample()
    P = np.array([[0.1, 0.0, -1.0]])  # behind the camera
    _, valid = m.project(P)
    assert not valid[0]
