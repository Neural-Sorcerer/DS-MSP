"""AprilGrid board geometry + correspondence assembly (pure numpy, no detector)."""

import numpy as np
import pytest

from ds_msp.calib import AprilGridTarget


def test_tag0_corner_order_and_size():
    t = AprilGridTarget(tag_rows=6, tag_cols=6, tag_size=0.088, tag_spacing=0.3)
    p = t.object_points(0)
    assert p.shape == (4, 3)
    # bottom-left tag sits at the origin; corners are BL, BR, TR, TL, CCW.
    assert np.allclose(p[0], [0.0, 0.0, 0.0])
    assert np.allclose(p[1], [0.088, 0.0, 0.0])
    assert np.allclose(p[2], [0.088, 0.088, 0.0])
    assert np.allclose(p[3], [0.0, 0.088, 0.0])
    assert np.allclose(np.linalg.norm(p[1] - p[0]), 0.088)  # side == tag_size


def test_tag_pitch_includes_spacing():
    t = AprilGridTarget(tag_size=0.1, tag_spacing=0.3)
    # tag 1 is the next column: origin shifted by size*(1+spacing) = 0.13 in x.
    assert np.allclose(t.object_points(1)[0], [0.13, 0.0, 0.0])
    # tag 6 is the next row (6 cols): shifted by 0.13 in y.
    assert np.allclose(t.object_points(6)[0], [0.0, 0.13, 0.0])


def test_object_points_range():
    t = AprilGridTarget()
    assert t.n_tags == 36
    with pytest.raises(ValueError):
        t.object_points(36)


def test_build_correspondences_shapes_and_filter():
    t = AprilGridTarget()
    full = {tid: np.zeros((4, 2)) for tid in range(36)}      # 144 corners
    sparse = {0: np.zeros((4, 2))}                            # 4 corners -> dropped
    Xs, UVs, VIs = t.build_correspondences([full, sparse], min_corners=8)
    assert len(Xs) == 1                                       # sparse frame filtered out
    assert Xs[0].shape == (144, 3)
    assert UVs[0].shape == (144, 2)
    assert VIs[0].shape == (144,) and VIs[0].all()
    # 3D points come from the board geometry, in tag-id order.
    assert np.allclose(Xs[0][:4], t.object_points(0))
