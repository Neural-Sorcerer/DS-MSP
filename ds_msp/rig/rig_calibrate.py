"""Top-level N-camera rig calibration orchestrator — the ``runCalibrationWorkflow``
analogue (apps/calibrate/src/calibrate.cpp:9), trimmed to DS-MSP's scope.

Stages: per-camera intrinsics + object poses (front-end) -> camera-group covisibility
+ extrinsics init -> object-in-group pose averaging -> staged global BA (poses, then
full joint incl. intrinsics). Non-overlapping groups are linked by hand-eye when
present (``rig.handeye``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..calib.bundle import calibrate as _calibrate_single
from ..core.contracts import CameraModel
from ..models.radtan import RadTanModel
from . import ba
from .extrinsics import init_camera_groups
from .pose_init import average_object_pose_in_group, estimate_pose_ransac
from .types import Object3D, ObjectObs, RigState


def _T_from_rt(rvec, tvec) -> np.ndarray:
    """4x4 object->camera transform from an OpenCV (rvec, tvec) pair."""
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float))[0]
    T[:3, 3] = np.asarray(tvec, float).ravel()
    return T


def _set_poses(obs: List[ObjectObs], rvecs, tvecs) -> None:
    for o, rv, tv in zip(obs, rvecs, tvecs):
        o.T_c_o = _T_from_rt(rv, tv)


def _front_end_opencv(obj: Object3D, obs_by_cam: Dict[int, List[ObjectObs]],
                      img_size: Dict[int, Tuple[int, int]]):
    """Per-camera intrinsics + per-frame object poses via ``cv2.calibrateCamera``
    (Brown/RadTan, distortion_type 0). Fills each ObjectObs' ``T_c_o``.

    A camera that only ever sees a single near-planar board (and at low tilt) suffers
    the focal/distance ambiguity, so ``calibrateCamera`` can return an implausible focal.
    Such cameras are detected (focal far outside the image-plausible range) and re-seeded
    from the consensus of the well-constrained cameras; the global BA then refines their
    intrinsics through the rigid-rig constraint. The seed only needs to land the camera's
    pose in the right basin.
    """
    raw: Dict[int, dict] = {}
    for cam_id, obs in obs_by_cam.items():
        objpts = [obj.pts_3d[o.point_rows].astype(np.float32) for o in obs]
        imgpts = [o.pts_2d.astype(np.float32) for o in obs]
        w, h = img_size[cam_id]
        K0 = np.array([[float(w), 0.0, w / 2.0],
                       [0.0, float(w), h / 2.0],
                       [0.0, 0.0, 1.0]])
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpts, imgpts, img_size[cam_id], K0, None,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        diag = float(np.hypot(w, h))
        plausible = 0.2 * diag < K[0, 0] < 4.0 * diag and 0.2 * diag < K[1, 1] < 4.0 * diag
        raw[cam_id] = dict(K=K, dist=dist.ravel(), rvecs=rvecs, tvecs=tvecs,
                           objpts=objpts, imgpts=imgpts, ok=plausible)

    good = [r["K"] for r in raw.values() if r["ok"]]
    consensus = np.median(np.stack(good), axis=0) if good else None

    cameras: Dict[int, CameraModel] = {}
    for cam_id, obs in obs_by_cam.items():
        r = raw[cam_id]
        if r["ok"] or consensus is None:
            K, d, rvecs, tvecs = r["K"], r["dist"], r["rvecs"], r["tvecs"]
        else:
            # degenerate camera: adopt the consensus intrinsics and re-solve its poses
            K = consensus.copy()
            d = np.zeros(5)
            rvecs, tvecs = [], []
            for op, ip in zip(r["objpts"], r["imgpts"]):
                ok, rv, tv = cv2.solvePnP(op, ip, K, d)
                rvecs.append(rv if ok else np.zeros(3))
                tvecs.append(tv if ok else np.array([0.0, 0.0, 1.0]))
        cameras[cam_id] = RadTanModel(K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                                      d[0], d[1], d[2], d[3], d[4])
        _set_poses(obs, rvecs, tvecs)
    return cameras


# Neutral seed values per intrinsic parameter name — a generic, GT-free starting point
# for from-scratch single-camera calibration of any model.
_NEUTRAL = {"alpha": 0.5, "xi": 0.0, "beta": 1.0, "k1": 0.0, "k2": 0.0, "k3": 0.0,
            "k4": 0.0, "p1": 0.0, "p2": 0.0}


def _seed_from_K(model_cls, K: np.ndarray) -> CameraModel:
    """Build a seed instance of ``model_cls`` from a pinhole ``K`` (focal + principal
    point) with neutral distortion — the from-scratch starting point for any model."""
    vals = {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2], **_NEUTRAL}
    vec = np.array([vals[n] for n in model_cls.param_names], float)
    return model_cls.from_params(vec)


def make_bundle_front_end(model_cls, *, loss: str = "cauchy", f_scale: float = 1.0,
                          max_nfev: int = 150):
    """Build a model-agnostic front-end that calibrates each camera with ``model_cls``.

    Seeding is data-driven and model-independent: a pinhole+Brown pre-calibration
    (``cv2.calibrateCamera``) gives a robust focal / principal-point seed (the same
    initialization MC-Calib uses), then the DS-MSP single-camera bundle adjuster
    (``calib.bundle.calibrate``) refines the *target model's* full parameter vector from
    that seed. Avoiding a blind focal sweep matters for models whose distortion can absorb
    a focal-seed error (e.g. KB), where an RMS-tie would otherwise pick a wrong-focal
    basin. The rig pipeline downstream operates entirely in ``model_cls``. Returns a
    callable with the ``front_end`` signature used by :func:`calibrate_rig`.
    """
    def front_end(obj, obs_by_cam, img_size):
        raw = {}
        for cam_id, obs in obs_by_cam.items():
            w, h = img_size[cam_id]
            ge6 = [o for o in obs if len(o.point_rows) >= 6]   # views usable for intrinsics
            objpts = [obj.pts_3d[o.point_rows].astype(np.float32) for o in ge6]
            imgpts = [o.pts_2d.astype(np.float32) for o in ge6]
            K0 = np.array([[float(w), 0, w / 2.0], [0, float(w), h / 2.0], [0, 0, 1.0]])
            _, Kp, _, _, _ = cv2.calibrateCamera(objpts, imgpts, (w, h), K0, None,
                                                 flags=cv2.CALIB_USE_INTRINSIC_GUESS)
            seed = _seed_from_K(model_cls, Kp)
            # keep only frames that retain >=6 valid rays under the seed model, so
            # calibrate()'s DLT pose seeding (needs 6) never chokes on a partial view.
            keep = []
            for o in ge6:
                rays, vr = seed.unproject(o.pts_2d)
                if int((vr & (rays[:, 2] > 1e-6)).sum()) >= 6:
                    keep.append(o)
            X = [obj.pts_3d[o.point_rows] for o in keep]
            uv = [o.pts_2d for o in keep]
            vis = [np.ones(len(o.point_rows), bool) for o in keep]
            res = _calibrate_single(seed, X, uv, vis, loss=loss, f_scale=f_scale,
                                    max_nfev=max_nfev)
            raw[cam_id] = dict(model=res["model"], keep=keep, poses=res["poses"], obs=obs)

        # Consensus guard: a camera that views the target near-planar (e.g. an obliquely
        # angled camera seeing one board) hits the focal ambiguity and calibrates to a wrong
        # / anamorphic focal. Detect such cameras by deviation from the per-rig median focal,
        # reset their intrinsics to the consensus, and let the global BA refine them through
        # the rigid-rig constraint (the object pose is pinned by the well-constrained cameras).
        fxs = np.array([raw[c]["model"].params[0] for c in raw])
        fys = np.array([raw[c]["model"].params[1] for c in raw])
        med_fx, med_fy = np.median(fxs), np.median(fys)

        cameras = {}
        for cam_id, r in raw.items():
            model = r["model"]
            fx, fy = model.params[0], model.params[1]
            degenerate = (abs(fx - med_fx) > 0.25 * med_fx
                          or abs(fy - med_fy) > 0.25 * med_fy)
            if degenerate and len(raw) >= 2:
                w, h = img_size[cam_id]
                Kc = np.array([[med_fx, 0, w / 2.0], [0, med_fy, h / 2.0], [0, 0, 1.0]])
                model = _seed_from_K(model_cls, Kc)
                cameras[cam_id] = model
                for o in r["obs"]:
                    o.T_c_o = _gated_pnp(model, obj.pts_3d[o.point_rows], o.pts_2d)
            else:
                cameras[cam_id] = model
                clean = {id(o): pose for o, pose in zip(r["keep"], r["poses"])}
                for o in r["obs"]:
                    if id(o) in clean:
                        o.T_c_o = _T_from_rt(*clean[id(o)])
                    else:
                        o.T_c_o = _gated_pnp(model, obj.pts_3d[o.point_rows], o.pts_2d)
        return cameras
    return front_end


def _gated_pnp(model, X, uv, max_rms_px: float = 2.0):
    """Robust PnP whose result is accepted only if it reprojects well — a bad pose from a
    hard partial view is dropped (returns ``None``) rather than poisoning the graph."""
    T, inl = estimate_pose_ransac(model, X, uv)
    if T is None:
        return None
    Xc = (T[:3, :3] @ X.T).T + T[:3, 3]
    proj, valid = model.project(Xc)
    if valid.sum() < 4:
        return None
    d = proj[valid] - uv[valid]
    rms = float(np.sqrt((d * d).sum() / valid.sum()))
    return T if rms < max_rms_px else None


def calibrate_rig(obj: Object3D, object_obs: List[ObjectObs],
                  img_size: Dict[int, Tuple[int, int]],
                  *, fix_intrinsics: bool = False, verbose: bool = False,
                  front_end: Optional[Callable] = None) -> RigState:
    """Calibrate a multi-camera rig from fused-object observations.

    Returns a :class:`RigState` with per-camera intrinsics, ``T_c_g`` extrinsics
    (reference camera = identity), and per-frame object poses.
    """
    obs_by_cam: Dict[int, List[ObjectObs]] = defaultdict(list)
    for o in object_obs:
        obs_by_cam[o.cam_id].append(o)
    cam_ids = sorted(obs_by_cam)

    # 1. per-camera intrinsics + object poses (T_c_o)
    fe = front_end or _front_end_opencv
    cameras = fe(obj, obs_by_cam, img_size)
    if verbose:
        print(f"[front-end] calibrated {len(cameras)} cameras")

    # 2. camera-group covisibility -> extrinsics init (T_c_g; ref cam = identity)
    groups, extr = init_camera_groups(object_obs, cam_ids)
    if verbose:
        print(f"[groups] {len(groups)} group(s): {groups}")
    if len(groups) > 1:
        # Non-overlapping groups: link with hand-eye, then re-base every camera to the
        # global reference (group 0's reference camera).
        from .handeye import link_groups
        extr = link_groups(groups, extr, object_obs)
    ref_cam = groups[0][0]

    # 3. per-frame object-in-group poses (average over the cameras that see it)
    by_fo: Dict[Tuple[int, int], List[Tuple[int, np.ndarray]]] = defaultdict(list)
    for o in object_obs:
        if o.T_c_o is not None and o.cam_id in extr:
            by_fo[(o.object_id, o.frame_id)].append((o.cam_id, o.T_c_o))
    object_poses: Dict[Tuple[int, int], np.ndarray] = {}
    for key, lst in by_fo.items():
        object_poses[key] = average_object_pose_in_group(lst, extr, ref_cam)

    rig = RigState(cameras=cameras, T_c_g=extr, ref_cam_id=ref_cam,
                   object_poses=object_poses, objects={0: obj}, img_size=img_size)

    # 4. staged global BA: poses-only, then full joint (incl. intrinsics)
    rig = ba.refine(rig, object_obs, fix_intrinsics=True, verbose=verbose)
    if not fix_intrinsics:
        rig = ba.refine(rig, object_obs, fix_intrinsics=False, verbose=verbose)
    return rig
