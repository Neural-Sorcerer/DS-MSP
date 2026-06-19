"""EUCM unit tests: EUCM with beta=1 reduces to UCM; round-trip holds."""

import numpy as np

from ds_msp.models.eucm import EUCMModel
from ds_msp.models.ucm import UCMModel
from ds_msp.testing import sample_forward_points


def test_eucm_beta1_equals_ucm():
    alpha = 0.62
    eucm = EUCMModel(700.0, 700.0, 640.0, 360.0, alpha, beta=1.0)
    ucm = UCMModel(700.0, 700.0, 640.0, 360.0, alpha)
    P = sample_forward_points()
    uv_e, _ = eucm.project(P)
    uv_u, _ = ucm.project(P)
    assert np.allclose(uv_e, uv_u, atol=1e-9)


def test_eucm_roundtrip():
    m = EUCMModel.sample()
    P = sample_forward_points()
    uv, v1 = m.project(P)
    rays, v2 = m.unproject(uv)
    ok = v1 & v2
    d = P[ok] / np.linalg.norm(P[ok], axis=1, keepdims=True)
    cos = np.sum(rays[ok] * d, axis=1)
    assert (cos > 1 - 1e-6).all()
