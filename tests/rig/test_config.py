"""MC-Calib calib_param.yml parsing + the single-file ``calibrate_from_config`` entry."""
import os
import textwrap

import numpy as np
import pytest

from ds_msp.rig.config import calibrate_from_config, load_config


def _write_cfg(tmp_path, body):
    p = tmp_path / "calib_param.yml"
    p.write_text("%YAML:1.0\n" + textwrap.dedent(body))
    return str(p)


def test_parse_models_precedence(tmp_path):
    # camera_models (extension) overrides distortion_model
    cfg = load_config(_write_cfg(tmp_path, """
        number_camera: 3
        number_board: 1
        number_x_square: 5
        number_y_square: 5
        square_size: 0.192
        distortion_model: 0
        camera_models: [ ds, ucm, eucm ]
        keypoints_path: "None"
        root_path: "None"
    """))
    assert cfg.camera_models == ["ds", "ucm", "eucm"]
    assert cfg.number_board == 1 and cfg.boards[0].n_x == 5


def test_distortion_model_mapping(tmp_path):
    # 0 -> radtan (Brown), 1 -> kb (Kannala); per-camera overrides the global
    cfg = load_config(_write_cfg(tmp_path, """
        number_camera: 3
        number_board: 1
        distortion_model: 1
        distortion_per_camera: [ 0, 1, 0 ]
        keypoints_path: "None"
    """))
    assert cfg.camera_models == ["radtan", "kb", "radtan"]


_S2 = "../MC-Calib/Blender_Images/Scenario_2"


@pytest.mark.skipif(not os.path.isdir(os.path.join(_S2, "Images")),
                    reason="Blender Scenario_2 images not present")
def test_calibrate_from_config_raw_images(tmp_path):
    """One config file → detect from raw images → MC-Calib output, extrinsics within 1%."""
    cfgp = "../MC-Calib/configs/Blender_Images/calib_param_synth_Scenario2.yml"
    ov = {"root_path": os.path.abspath(os.path.join(_S2, "Images")),
          "keypoints_path": "None", "save_path": str(tmp_path),
          "camera_models": ["radtan", "ucm", "eucm", "ds", "radtan"]}
    res = calibrate_from_config(cfgp, ov)
    assert res["models"] == {0: "radtan", 1: "ucm", 2: "eucm", 3: "ds", 4: "radtan"}
    assert res["metrics"]["worst_baseline_pct_vs_gt"] < 1.0
    for fn in ("calibrated_cameras_data.yml", "calibrated_objects_data.yml",
               "calibrated_objects_pose_data.yml"):
        assert (tmp_path / fn).exists()
