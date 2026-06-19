"""OCamCalib model: round-trip + a DS->OCam conversion (OCam represents DS well)."""

import numpy as np

from ds_msp.models.ocam import OCamModel
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.adapt import convert
from ds_msp.testing import sample_forward_points


def test_ocam_roundtrip():
    m = OCamModel.sample()
    P = sample_forward_points()
    uv, v1 = m.project(P)
    rays, v2 = m.unproject(uv)
    ok = v1 & v2
    d = P[ok] / np.linalg.norm(P[ok], axis=1, keepdims=True)
    cos = np.sum(rays[ok] * d, axis=1)
    assert (cos > 1 - 1e-5).all()


def test_ds_to_ocam_conversion():
    ds = DoubleSphereModel.sample()
    ocam, report = convert(ds, OCamModel, width=1920, height=1080, n_samples=600)
    assert report["converged"]
    assert report["rms_px"] < 1.0, report
