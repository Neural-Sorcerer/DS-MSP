"""Real-data validation of robust ``calibrate()`` + ``convert()`` (FR-CALIB-001,
FR-ADAPT-001, NFR-NUM-004).

The synthetic suites pin the robustness *contract*; this is the real-data half of the
release gate. It calibrates **three** genuine fisheye cameras from MC-Calib (each ~60 views
of a 7x7 ChArUco board, ~2k real corner detections at 2592x1800) entirely from a **generic**
init (focal = W/pi, centre = image centre; no reference intrinsics supplied) and asserts:

  * **Per camera, robust auto-init is sub-pixel** and recovers MC-Calib's reference
    intrinsics K to a tight tolerance (``test_intrinsics_recovered_per_camera``), and
  * **The intrinsics are model-consistent**: for each camera we independently calibrate a
    second model (EUCM+) *and* image-free-``convert()`` the from-scratch DS+ into EUCM+; the
    converted model must reproduce DS+ and agree with the from-scratch EUCM+ over the
    calibrated FOV (``test_convert_agrees_with_from_scratch_per_camera``). This proves the
    recovered intrinsics are physical, not a single model's artefact.

Corners are detected on the fly with ``ds_msp.detect.charuco`` (verified to reproduce
MC-Calib's own published cam0 corner set). ``convert()`` is restricted to the FOV the corners
actually span: outside it the two model families legitimately diverge (extrapolation), so an
unrestricted convert is *expected* to disagree and would be a meaningless comparison.

Dataset-gated: the imagery is local-only (gitignored) so this test SKIPS when absent (PR CI),
and is required-green in the pre-release job that closes the gate. Point it at the data with
``DSMSP_MCCALIB_DIR=/path/to/2026_06_26_MC-Calib``.
"""

import glob
import os
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pytest

from ds_msp.adapt import convert, sample_image_grid
from ds_msp.calib import calibrate
from ds_msp.detect.charuco import BoardSpec, detect_image, make_detectors
from ds_msp.models.dsplus import DSPlusModel
from ds_msp.models.eucmplus import EUCMPlusModel

CAMERAS = (0, 2, 4)
# MC-Calib's board for this rig: 7x7 ChArUco, DICT_6X6_1000, 0.1 m squares (legacy pattern).
_SPEC = BoardSpec(n_x=7, n_y=7, length_square=0.1, length_marker=0.075, square_size=0.1)
_NCX = _SPEC.n_x - 1
_SQ = _SPEC.square_size


def _dataset_root() -> Path | None:
    env = os.environ.get("DSMSP_MCCALIB_DIR")
    root = Path(env) if env else Path(__file__).resolve().parents[2] / "2026_06_26_MC-Calib"
    return root if (root / "calibration_images_0").is_dir() else None


def _K4(model) -> np.ndarray:
    K = model.K
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])


def _reference_K(cam_dir: Path):
    """MC-Calib's published (fx, fy, cx, cy), reprojection error, and image size."""
    r = ET.parse(cam_dir / "calibration.xml").getroot()
    cm = r.find("camera_matrix")
    K = np.array([float(cm.find(t).text) for t in ("fx", "fy", "ppx", "ppy")])
    W = int(r.find("image_size/width").text)
    H = int(r.find("image_size/height").text)
    return K, float(r.find("reprojection_error").text), W, H


def _detect_cam(cam_dir: Path):
    """Detect board-0 ChArUco corners across a camera's images -> calibration inputs.

    Corner id ``k`` maps to board point ``((k % ncx)*sq, (k // ncx)*sq, 0)`` — the row-major
    interior-corner convention of ``ds_msp.detect.charuco`` (and MC-Calib's renderer).
    """
    detectors = make_detectors([_SPEC], legacy=True)
    Xs, kps, vis, all_px = [], [], [], []
    for f in sorted(glob.glob(str(cam_dir / "*.png"))):
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if gray is None:                       # a few frames in this set are truncated PNGs
            continue
        for board_id, ids, pts in detect_image(detectors, gray, min_corners=6):
            if board_id != 0:
                continue
            X = np.column_stack([(ids % _NCX) * _SQ, (ids // _NCX) * _SQ, np.zeros(len(ids))])
            Xs.append(X.astype(float))
            kps.append(pts.astype(float))
            vis.append(np.ones(len(ids), bool))
            all_px.append(pts)
    return Xs, kps, vis, np.vstack(all_px)


# Calibrating three cameras x two models is a few seconds each; cache per camera so both
# test functions (and every parametrization) reuse one fit.
_CACHE: dict[int, dict] = {}


def _fit_camera(cam: int) -> dict:
    if cam in _CACHE:
        return _CACHE[cam]
    root = _dataset_root()
    cam_dir = root / f"calibration_images_{cam}"
    refK, ref_reproj, W, H = _reference_K(cam_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # benign invalid-divide on non-projectable rays
        Xs, kps, vis, all_px = _detect_cam(cam_dir)
        assert len(Xs) >= 30, f"cam{cam}: only {len(Xs)} usable views detected"
        f0 = W / np.pi
        dsp = calibrate(DSPlusModel(f0, f0, W / 2, H / 2, 0.5, 0, 0, 0, 0),
                        Xs, kps, vis, max_nfev=200)
        eup = calibrate(EUCMPlusModel(f0, f0, W / 2, H / 2, 0.5, 1.0, 0, 0, 0),
                        Xs, kps, vis, max_nfev=200)
        # FOV (full angle) actually spanned by the detected corners, via the DS+ fit.
        rays, valid = dsp["model"].unproject(all_px)
        half = float(np.degrees(np.arccos(np.clip(rays[valid, 2], -1, 1))).max())
        eup_conv, conv_report = convert(dsp["model"], EUCMPlusModel,
                                        width=W, height=H, n_samples=900, max_fov_deg=2 * half)
    out = dict(dsp=dsp, eup=eup, eup_conv=eup_conv, conv_report=conv_report,
               refK=refK, ref_reproj=ref_reproj, W=W, H=H, half_fov=half,
               n_views=len(Xs), n_corners=sum(len(x) for x in Xs))
    _CACHE[cam] = out
    return out


@pytest.fixture(scope="session")
def fitted():
    if _dataset_root() is None:
        pytest.skip("MC-Calib imagery not present (set DSMSP_MCCALIB_DIR)")
    return _fit_camera


pytestmark = [pytest.mark.realdata,
              pytest.mark.req("FR-CALIB-001", "FR-ADAPT-001", "NFR-NUM-004")]


@pytest.mark.parametrize("cam", CAMERAS)
def test_intrinsics_recovered_per_camera(fitted, cam):
    """Each real fisheye calibrates sub-pixel from a generic init and recovers MC-Calib's K."""
    r = fitted(cam)
    dsp, refK = r["dsp"], r["refK"]
    assert dsp["success"], (cam, dsp)
    assert dsp["median_px"] < 0.40, (cam, dsp["median_px"])
    assert dsp["mean_px"] < 0.45, (cam, dsp["mean_px"])
    assert dsp["p95_px"] < 1.0, (cam, dsp["p95_px"])
    got = _K4(dsp["model"])
    assert np.allclose(got[:2], refK[:2], rtol=0.01), (cam, got, refK)   # focal within 1%
    assert np.allclose(got[2:], refK[2:], atol=8.0), (cam, got, refK)    # centre within 8 px


@pytest.mark.parametrize("cam", CAMERAS)
def test_convert_agrees_with_from_scratch_per_camera(fitted, cam):
    """The from-scratch DS+, image-free-converted into EUCM+, reproduces DS+ and matches the
    independently from-scratch-calibrated EUCM+ over the calibrated FOV — so the recovered
    intrinsics are model-consistent, not a single model's artefact."""
    r = fitted(cam)
    eup, eup_conv, report = r["eup"], r["eup_conv"], r["conv_report"]

    # 1. EUCM+ is itself a valid sub-pixel fit to the same corners (two-model agreement).
    assert eup["median_px"] < 0.40, (cam, eup["median_px"])

    # 2. convert() reproduces the DS+ source over the calibrated FOV (image-free).
    assert report["rms_px"] < 0.8, (cam, report["rms_px"])

    # 3. converted EUCM+ agrees with the from-scratch EUCM+: focal within 1%, and pixel-space
    #    agreement is sub-few-px over the FOV the corners actually constrain.
    conv_K, scratch_K = _K4(eup_conv), _K4(eup["model"])
    assert np.allclose(conv_K[:2], scratch_K[:2], rtol=0.01), (cam, conv_K, scratch_K)

    px = sample_image_grid(r["W"], r["H"], 1500)
    rays, valid = eup["model"].unproject(px)
    ang = np.degrees(np.arccos(np.clip(rays[:, 2], -1, 1)))
    keep = valid & (rays[:, 2] > 1e-6) & (ang <= r["half_fov"])
    uv_scratch, _ = eup["model"].project(rays[keep])
    uv_conv, _ = eup_conv.project(rays[keep])
    agreement = float(np.sqrt(np.mean(np.sum((uv_scratch - uv_conv) ** 2, axis=1))))
    assert agreement < 4.0, (cam, agreement)
