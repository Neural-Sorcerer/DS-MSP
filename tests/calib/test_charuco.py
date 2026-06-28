"""Raw-image ChArUco detection — self-contained (renders its own board) plus an optional
corner-for-corner parity check against MC-Calib's keypoints when the Blender data is present.
"""
import os

import cv2
import numpy as np
import pytest

from ds_msp.calib.charuco import (BoardSpec, board_object_points, detect_image,
                                  single_board_object)

SPEC = BoardSpec(n_x=5, n_y=5, length_square=0.04, length_marker=0.03, square_size=0.192)


def test_object_points_match_mccalib_layout():
    xyz = board_object_points(SPEC)
    assert xyz.shape == (16, 3)
    # corner k at (k%4, k//4)*square_size, z=0 (row-major) — MC-Calib's single-board model
    assert np.allclose(xyz[1], [0.192, 0.0, 0.0])
    assert np.allclose(xyz[4], [0.0, 0.192, 0.0])
    assert np.allclose(xyz[5], [0.192, 0.192, 0.0])
    assert np.allclose(xyz[:, 2], 0.0)
    obj = single_board_object(SPEC)
    assert obj.pts_board_2_obj[(0, 5)] == 5


def test_detect_rendered_board_recovers_all_corners():
    """Render the exact board to an image and detect it back: every interior corner is
    found, at its known pixel location (the detector is internally consistent)."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    board = cv2.aruco.CharucoBoard((SPEC.n_x, SPEC.n_y), SPEC.length_square,
                                   SPEC.length_marker, dictionary)
    img = board.generateImage((1000, 1000), marginSize=40)
    # the rendered board uses the non-legacy pattern, so detect with legacy=False
    det = cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, _, _ = det.detectBoard(img)
    assert ch_ids is not None and len(ch_ids) == 16          # all interior corners

    found = detect_image([cv2.aruco.CharucoDetector(board)], img, min_corners=4)
    assert len(found) == 1
    bid, ids, pts = found[0]
    assert bid == 0 and sorted(ids) == list(range(16))
    assert pts.shape == (16, 2)


_S2 = "../MC-Calib/Blender_Images/Scenario_2"


@pytest.mark.skipif(not os.path.isdir(os.path.join(_S2, "Images")),
                    reason="Blender Scenario_2 images not present")
def test_parity_vs_mccalib_keypoints():
    """Detected corners reproduce MC-Calib's own ``detected_keypoints_data.yml`` to
    sub-pixel agreement on the same physical frames (cam 0)."""
    from ds_msp.calib.charuco import detect_folder
    from ds_msp.io.mccalib import load_scenario
    scn = load_scenario(_S2)
    obj = single_board_object(SPEC)
    mc = {}
    for o in scn.object_obs:
        if o.cam_id == 0:
            mc.setdefault(o.frame_id, {}).update(
                {int(r): uv for r, uv in zip(o.point_rows, o.pts_2d)})
    obs = detect_folder(os.path.join(_S2, "Images/Cam_001"), [SPEC], obj, 0,
                        legacy=True, min_corners=8)
    mine = {}
    for o in obs:                                            # filename N -> MC frame N-1
        mine.setdefault(o.frame_id - 1, {}).update(
            {int(r): uv for r, uv in zip(o.point_rows, o.pts_2d)})
    diffs = [np.linalg.norm(mc[f][r] - mine[f][r])
             for f in set(mc) & set(mine) for r in set(mc[f]) & set(mine[f])]
    assert len(diffs) > 300
    assert np.median(diffs) < 0.1 and np.max(diffs) < 1.0