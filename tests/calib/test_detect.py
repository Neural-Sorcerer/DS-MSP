"""Multi-scale union + board-guided recovery in the AprilGrid detector.

These exercise the detection *logic* with a fake detector — no real images and no
``aprilgrid`` backend required (the heavy parts are duck-typed).
"""

import numpy as np
import pytest

from ds_msp.calib import AprilGridTarget
from ds_msp.calib.detect import _detect_union, _recover_missing, detect_aprilgrid


class _Det:
    """Minimal stand-in for an aprilgrid detection (has .tag_id and .corners)."""

    def __init__(self, tag_id, corners):
        self.tag_id = tag_id
        self.corners = np.asarray(corners, dtype=np.float32)


def test_detect_union_unions_scales_and_maps_corners_back():
    native = {0: np.array([[1., 1], [2, 1], [2, 2], [1, 2]]),
              1: np.array([[3., 3], [4, 3], [4, 4], [3, 4]])}

    class FakeDetector:
        """tag 0 visible at every scale; tag 1 only once the image is upscaled.

        Returns corners in the *given image's* pixels, following OpenCV's resize
        pixel-centre convention (native N appears at (N+0.5)*s - 0.5), so the union's
        inverse mapping must recover native coordinates exactly.
        """
        def detect(self, img):
            s = img.shape[0] / 10.0                      # native image is 10x10
            ids = [0] if s < 1.5 else [0, 1]
            return [_Det(t, (native[t] + 0.5) * s - 0.5) for t in ids]

    gray = np.zeros((10, 10), dtype=np.uint8)
    found = _detect_union(FakeDetector(), gray, scales=(1, 2))

    assert set(found) == {0, 1}                          # scale 2 recovered tag 1
    assert np.allclose(found[0], native[0])              # corners mapped back to native
    assert np.allclose(found[1], native[1])


def test_detect_union_single_scale_is_old_behaviour():
    class FakeDetector:
        def detect(self, img):
            s = img.shape[0] / 10.0
            return [_Det(0, np.zeros((4, 2)))] if s < 1.5 else [_Det(1, np.zeros((4, 2)))]

    found = _detect_union(FakeDetector(), np.zeros((10, 10), np.uint8), scales=(1,))
    assert set(found) == {0}                             # no upscaled pass → only tag 0


def test_recover_requires_a_target():
    with pytest.raises(ValueError, match="recover=True needs a `target`"):
        detect_aprilgrid([], recover=True, target=None)


def test_recover_missing_adds_only_id_verified_tags():
    """Predict missing tags from a homography of seeds, re-detect in ROI, accept by id."""
    target = AprilGridTarget(tag_rows=3, tag_cols=3, tag_size=0.088, tag_spacing=0.3)
    # Ground-truth board->image map (a plain similarity is a valid homography).
    def to_img(xy):
        return xy * 500.0 + 50.0

    seeds = {t: to_img(target.object_points(t)[:, :2]).astype(np.float32)
             for t in (0, 2, 6, 8)}                       # 4 corner tags seed the homography
    gray = np.zeros((320, 320), dtype=np.uint8)

    class ROIDetector:
        """Returns tag 4 for any ROI it's handed; everything else stays missing."""
        def detect(self, img):
            return [_Det(4, np.array([[1., 1], [3, 1], [3, 3], [1, 3]]))]

    out = _recover_missing(ROIDetector(), gray, dict(seeds), target,
                           neighbors=6, roi_pad=0.6, roi_target_px=160.0)

    assert set(out) == set(seeds) | {4}                  # only the id-verified tag is added
    assert np.isfinite(out[4]).all()
    # recovered corners land inside the image near the predicted (board-centre) location
    cx, cy = out[4].mean(0)
    assert 0 <= cx < 320 and 0 <= cy < 320
