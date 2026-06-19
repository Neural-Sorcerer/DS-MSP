"""UCM unit tests: UCM must equal Double Sphere with xi = 0."""

import numpy as np

from ds_msp.models.ucm import UCMModel
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.testing import sample_forward_points


def test_ucm_equals_ds_with_zero_xi():
    alpha = 0.62
    ucm = UCMModel(700.0, 700.0, 640.0, 360.0, alpha)
    ds = DoubleSphereModel(700.0, 700.0, 640.0, 360.0, 0.0, alpha)
    P = sample_forward_points()
    uv_u, vu = ucm.project(P)
    uv_d, vd = ds.project(P)
    assert np.allclose(uv_u, uv_d, atol=1e-9)
    assert np.array_equal(vu, vd)
    # unprojection agreement
    rays_u, _ = ucm.unproject(uv_u)
    rays_d, _ = ds.unproject(uv_d)
    assert np.allclose(rays_u, rays_d, atol=1e-9)


def test_ucm_roundtrip():
    m = UCMModel.sample()
    P = sample_forward_points()
    uv, v1 = m.project(P)
    rays, v2 = m.unproject(uv)
    ok = v1 & v2
    d = P[ok] / np.linalg.norm(P[ok], axis=1, keepdims=True)
    cos = np.sum(rays[ok] * d, axis=1)
    assert (cos > 1 - 1e-6).all()
