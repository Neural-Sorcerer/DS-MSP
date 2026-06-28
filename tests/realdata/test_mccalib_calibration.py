"""Real-data validation of robust ``calibrate()`` (FR-CALIB-001, NFR-NUM-004).

The synthetic suites pin the robustness *contract*; this is the real-data half of the
release gate. It calibrates a genuine fisheye camera — 58 views of a 7x7 ChArUco board,
~1.9k real corner detections at 2592x1800 — from a **generic** init (focal = W/pi, centre
= image centre; no reference intrinsics fed in) and asserts that robust auto-init:

  * reaches sub-pixel median reprojection and **beats MC-Calib's own result** on the same
    corners (MC-Calib fisheye: 0.481 px), and
  * recovers the intrinsics matrix to within a tight tolerance of MC-Calib's reference K,
  * with the closed-form-invertible DS+ model clearly out-resolving plain Double Sphere,
    which is under-parameterised for this lens.

Dataset-gated: the calibration data is local-only (gitignored) so this test SKIPS when the
data is absent (PR CI), and is required-green in the pre-release job that closes the gate.
Point it at the data with ``DSMSP_MCCALIB_CAM0=/path/to/calibration_corners.json``.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from ds_msp.calib import calibrate
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.dsplus import DSPlusModel

# MC-Calib's published result on this lens (calibration.xml reprojection_error / K).
MCCALIB_MEDIAN_PX = 0.481
MCCALIB_K = (995.52, 995.50, 1276.84, 875.73)   # fx, fy, cx, cy


def _find_cam0_corners() -> Path | None:
    """Locate cam0 ChArUco corners JSON (env override, else known local layout)."""
    env = os.environ.get("DSMSP_MCCALIB_CAM0")
    if env:
        p = Path(env)
        return p if p.exists() else None
    root = Path(__file__).resolve().parents[2]
    cand = root / "2026_06_26_MC-Calib" / "calibration_images_0" / "calibration_corners.json"
    return cand if cand.exists() else None


def _load_cam0(path: Path):
    """Build (X_world, keypoints, visibility) per view from MC-Calib cam0 corners.

    Corner id ``k`` maps to the board point ``((k % ncx)*sq, (k // ncx)*sq, 0)`` — the
    row-major interior-corner convention shared by ``ds_msp.detect.charuco``.
    """
    d = json.loads(path.read_text())
    sq = d["board_params"]["square_size"]
    ncx = d["board_params"]["cols"] - 1
    res = d["camera_params"]["resolution"]
    Xs, kps, vis = [], [], []
    for s in d["samples"]:
        kp = np.asarray(s["keypoints"], float)
        ids = kp[:, 2].astype(int)
        X = np.column_stack([(ids % ncx) * sq, (ids // ncx) * sq, np.zeros(len(ids))])
        Xs.append(X.astype(float))
        kps.append(kp[:, :2].copy())
        vis.append(np.ones(len(ids), bool))
    return Xs, kps, vis, int(res["width"]), int(res["height"])


pytestmark = [pytest.mark.realdata, pytest.mark.req("FR-CALIB-001", "NFR-NUM-004")]


@pytest.fixture(scope="module")
def cam0():
    path = _find_cam0_corners()
    if path is None:
        pytest.skip("MC-Calib cam0 corners not present (set DSMSP_MCCALIB_CAM0)")
    Xs, kps, vis, W, H = _load_cam0(path)
    assert len(Xs) >= 30, f"expected a real multi-view set, got {len(Xs)} views"
    return Xs, kps, vis, W, H


def test_dsplus_beats_mccalib_from_generic_init(cam0):
    """DS+ auto-inits from a generic guess and out-resolves MC-Calib's own fit."""
    Xs, kps, vis, W, H = cam0
    f0, cx, cy = W / np.pi, W / 2.0, H / 2.0          # generic, no reference intrinsics
    init = DSPlusModel(f0, f0, cx, cy, 0.5, 0.0, 0.0, 0.0, 0.0)
    r = calibrate(init, Xs, kps, vis, max_nfev=200)

    assert r["success"], r
    # Beats MC-Calib's published median on the very same corners, and is sub-pixel in mean/p95.
    assert r["median_px"] < MCCALIB_MEDIAN_PX, r
    assert r["mean_px"] < 0.45, r
    assert r["p95_px"] < 1.0, r

    # Recovers MC-Calib's reference intrinsics from the generic init (no K was supplied).
    K = r["model"].K
    got = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
    ref = np.array(MCCALIB_K)
    assert np.allclose(got[:2], ref[:2], rtol=0.01), (got, ref)       # focal within 1%
    assert np.allclose(got[2:], ref[2:], atol=5.0), (got, ref)        # centre within 5 px


def test_dsplus_outresolves_double_sphere(cam0):
    """DS+ (closed-form invertible, 9 params) clearly out-resolves plain DS on this lens."""
    Xs, kps, vis, W, H = cam0
    f0, cx, cy = W / np.pi, W / 2.0, H / 2.0
    ds = calibrate(DoubleSphereModel(f0, f0, cx, cy, 0.0, 0.5), Xs, kps, vis, max_nfev=200)
    dsp = calibrate(DSPlusModel(f0, f0, cx, cy, 0.5, 0.0, 0.0, 0.0, 0.0),
                    Xs, kps, vis, max_nfev=200)
    assert dsp["median_px"] < 0.5 * ds["median_px"], (dsp["median_px"], ds["median_px"])
