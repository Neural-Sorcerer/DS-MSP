"""End-to-end MC-Calib-style runner: per-camera models, both optimize modes, exact output.

Builds a synthetic Scenario (so the test is self-contained, no external dataset) and exercises
``rig.run.calibrate_scenario`` the way ``scripts/calibrate_rig.py`` does on real data.
"""
import os

import cv2
import numpy as np
import pytest

from ds_msp.io.mccalib import CameraGT, Scenario
from ds_msp.models.radtan import RadTanModel
from ds_msp.rig.run import calibrate_scenario, random_model_assignment
from ._synth import make_rig

W, H = 1280, 960


def _scenario(seed=0, n_cam=3):
    def fac(cam_id, rng):
        fx = 800.0 * rng.uniform(0.98, 1.02)
        return RadTanModel(fx, fx, W / 2, H / 2, -0.05, 0.01, 0.0, 0.0, 0.0)
    obj, obs, img, gt_ext, gtm = make_rig(n_cam=n_cam, n_frame=45, noise_px=0.3, seed=seed,
                                          w=W, h=H, model_factory=fac)
    # GT/MC-Calib store camera->world; rig T_c_g is world->camera -> invert for the reference.
    gt = {c: CameraGT(K=gtm[c].K, dist=None, pose=np.linalg.inv(gt_ext[c]))
          for c in range(n_cam)}
    return Scenario(name="synth", object=obj, object_obs=obs, cam_ids=sorted(img),
                    img_size=img, gt=gt, mccalib=gt, mccalib_rms={}), gtm


def test_calibrate_scenario_per_camera_writes_mccalib_output(tmp_path):
    scn, _ = _scenario()
    spec = {0: "radtan", 1: "double_sphere", 2: "ucm"}
    res = calibrate_scenario(scn, spec, save_dir=str(tmp_path))
    assert res["models"] == {0: "radtan", 1: "ds", 2: "ucm"}
    # within 1% of GT baselines
    assert res["metrics"]["worst_baseline_pct_vs_gt"] < 1.0
    # the full MC-Calib output set exists and parses
    for fn in ("calibrated_cameras_data.yml", "calibrated_objects_data.yml",
               "calibrated_objects_pose_data.yml", "reprojection_error_data.yml"):
        assert (tmp_path / fn).exists()
    fs = cv2.FileStorage(str(tmp_path / "reprojection_error_data.yml"), cv2.FILE_STORAGE_READ)
    assert int(fs.getNode("nb_camera_group").real()) == 1
    fs.release()


def test_fix_intrinsics_extrinsics_only(tmp_path):
    scn, gtm = _scenario(seed=1)
    init = {c: gtm[c] for c in scn.cam_ids}                 # known intrinsics held fixed
    res = calibrate_scenario(scn, "radtan", fix_intrinsics=True, init_cameras=init,
                             save_dir=str(tmp_path))
    # intrinsics unchanged (held fixed), extrinsics recovered within 1%
    for c in scn.cam_ids:
        assert np.allclose(res["rig"].cameras[c].K, gtm[c].K, atol=1e-9)
    assert res["metrics"]["worst_baseline_pct_vs_gt"] < 1.0


def test_random_assignment_runs_within_1pct(tmp_path):
    scn, _ = _scenario(seed=2)
    spec = random_model_assignment(scn.cam_ids, kind="pinhole", seed=3)
    res = calibrate_scenario(scn, spec, save_dir=str(tmp_path))
    assert res["metrics"]["worst_baseline_pct_vs_gt"] < 1.0
