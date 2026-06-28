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


# --- intrinsic initialization from cam_params_path (init_K seeding + model conversion) ---

def test_init_model_native_is_exact_and_conversion_reproduces_lens():
    """``_init_model`` returns the file's native params for the matching model, and converts the
    *same lens* into another model so it reprojects the source over the FOV (sub-pixel for a
    moderate fisheye — the "two models, one camera" identity used by the fixed-intrinsic path)."""
    from ds_msp.io.mccalib import CameraGT
    from ds_msp.models.kb import KannalaBrandtModel
    from ds_msp.rig.config import _init_model, _source_model

    K = np.array([[420.0, 0, 320.0], [0, 421.0, 240.0], [0, 0, 1.0]])
    kb = CameraGT(K=K, dist=np.array([-0.02, 0.004, -0.001, 0.0003]), pose=np.eye(4))
    wh = (640, 480)

    native = _init_model(kb, "kb", wh)                       # native: exact stored params
    assert native.name == "kb"
    assert np.allclose(native.params, KannalaBrandtModel(K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                                                         *kb.dist).params)

    src = _source_model(kb)
    uu, vv = np.meshgrid(np.linspace(0.1 * wh[0], 0.9 * wh[0], 30),
                         np.linspace(0.1 * wh[1], 0.9 * wh[1], 30))
    pix = np.column_stack([uu.ravel(), vv.ravel()])
    rays, val = src.unproject(pix)
    val = np.asarray(val).ravel().astype(bool)
    rays, pix = np.asarray(rays, float)[val], pix[val]
    for name in ("ucm", "eucm", "ds"):
        m = _init_model(kb, name, wh)
        assert m.name == name
        uv, ok = m.project(rays)
        ok = np.asarray(ok).ravel().astype(bool)
        rms = float(np.sqrt(np.mean(np.sum((uv[ok] - pix[ok]) ** 2, axis=1))))
        assert rms < 1.0, f"{name} conversion RMS {rms:.3f}px"


def test_init_K_seeds_front_end_and_bypasses_heterogeneous_consensus():
    """A provided ``init_K`` seeds each camera's focal and bypasses the cross-camera focal
    consensus reset, so a rig whose cameras have very different focals (a mixed-resolution /
    mixed-lens rig) is not collapsed onto the median focal."""
    from ds_msp.models.radtan import RadTanModel
    from ds_msp.rig.rig_calibrate import make_bundle_front_end, paraxial_focal
    from ._synth import make_rig

    w, h = 1280, 960

    def mf(cam_id, rng):
        f = 820.0 if cam_id == 0 else 380.0                  # >25% apart -> consensus would reset
        return RadTanModel(f, f, w / 2, h / 2, -0.05, 0.01, 0.0, 0.0, 0.0)

    obj, obs, img_size, _gt_ext, gt_models = make_rig(n_cam=2, n_frame=40, w=w, h=h,
                                                      model_factory=mf, seed=3)
    from collections import defaultdict
    obs_by_cam = defaultdict(list)
    for o in obs:
        obs_by_cam[o.cam_id].append(o)
    init_K = {c: gt_models[c].K for c in (0, 1)}
    cams = make_bundle_front_end({0: "radtan", 1: "radtan"}, init_K=init_K)(obj, obs_by_cam, img_size)
    # the low-focal camera keeps its own focal, not the ~600 median it would be reset to.
    assert abs(paraxial_focal(cams[1])[0] - 380.0) / 380.0 < 0.1
    assert abs(paraxial_focal(cams[0])[0] - 820.0) / 820.0 < 0.1
