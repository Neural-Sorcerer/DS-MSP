"""
Generic bundle-adjustment calibration for ANY camera model.

Given checkerboard correspondences and an initial model (for intrinsics seed +
the model type), jointly refines intrinsics and per-image extrinsics by
Levenberg-Marquardt using the model's **analytic** projection Jacobian (no
autodiff). Works for DS/UCM/EUCM/KB/RadTan or any ``CameraModel``.

The model-agnostic BA loop lives in :mod:`ds_msp.geometry.calibrate_core`; this module
adds pose seeding and result formatting. The model type comes from the injected
``init_model`` (no concrete-model import).
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from ..core.contracts import CameraModel
from ..geometry.calibrate_core import bundle_adjust

#: Map the historical SciPy ``loss`` names to the in-house IRLS kernels.
_LOSS_TO_KERNEL = {
    "linear": "none", "huber": "huber", "cauchy": "cauchy",
    "soft_l1": "pseudo_huber",
}


def _seed_poses(init_model, X_world_list, keypoints_list, visibility_list):
    rvecs, tvecs = [], []
    for Xw, uv, vis in zip(X_world_list, keypoints_list, visibility_list):
        Xv = Xw[vis].astype(np.float64)
        uvv = uv[vis].astype(np.float64)
        rays, vr = init_model.unproject(uvv)
        use = vr & (rays[:, 2] > 1e-6)
        if use.sum() >= 4:
            pn = rays[use, :2] / rays[use, 2:3]
            ok, rv, tv = cv2.solvePnP(Xv[use], pn, np.eye(3), None)
            if ok:
                rvecs.append(rv.ravel())
                tvecs.append(tv.ravel())
                continue
        rvecs.append(np.zeros(3))
        tvecs.append(np.array([0.0, 0.0, 1.5]))
    return rvecs, tvecs


def calibrate(init_model: CameraModel,
              X_world_list: List[np.ndarray],
              keypoints_list: List[np.ndarray],
              visibility_list: List[np.ndarray],
              *, max_nfev: int = 200, verbose: int = 0,
              loss: str = "linear", f_scale: float = 1.0) -> Dict:
    """Calibrate any model from checkerboard correspondences.

    Parameters
    ----------
    loss : str
        Robust loss for the least-squares solve (SciPy ``least_squares`` kernels):
        ``"linear"`` (plain L2), ``"huber"``, ``"soft_l1"``, ``"cauchy"``. A robust
        kernel keeps *every* corner but **down-weights** large residuals instead of
        letting one mis-localized corner drag the L2 fit — the right tool when a few
        peripheral corners are mis-detected. Prefer this over hard outlier dropping.
    f_scale : float
        Residual scale (px) at which down-weighting kicks in; residuals below it stay
        ~quadratic. ~1 px is sensible for sub-pixel-targeted corner detection.

    Returns a dict ``{model, poses, rms_px, success}`` where ``poses`` is a list of
    ``(rvec, tvec)`` per image and ``rms_px`` is the **true** reprojection RMS over
    valid observations (independent of ``loss``, so it stays comparable across kernels).
    """
    cls = type(init_model)
    n_img = len(X_world_list)
    masks = [np.asarray(v, bool) for v in visibility_list]

    rvecs, tvecs = _seed_poses(init_model, X_world_list, keypoints_list, visibility_list)
    R0 = np.stack([cv2.Rodrigues(np.asarray(r, float))[0] for r in rvecs])   # (n,3,3)
    t0 = np.stack([np.asarray(t, float) for t in tvecs])                     # (n,3)

    kernel = _LOSS_TO_KERNEL.get(loss, loss)
    params, Rb, t, out = bundle_adjust(
        cls, init_model.params, R0, t0,
        X_world_list, keypoints_list, visibility_list,
        kernel=kernel, scale=(f_scale if kernel != "none" else 1.0), max_iter=max_nfev)
    model = cls.from_params(params)

    # Return absolute (rvec, tvec) per image so downstream code (cv2.Rodrigues) is unaffected.
    poses = [(cv2.Rodrigues(Rb[i])[0].ravel(), np.asarray(t[i], float)) for i in range(n_img)]

    # True reprojection RMS over valid observations — computed directly so it means the same thing
    # under any robust kernel (the kernel reshapes the cost; the pixel error of the fit is reported).
    sq, n = 0.0, 0
    for (rvec, tv), Xw, uv, vis in zip(poses, X_world_list, keypoints_list, masks):
        R, _ = cv2.Rodrigues(rvec)
        uvp, valid = model.project((R @ Xw.T).T + tv)
        mm = vis & valid
        d = uvp[mm] - uv[mm]
        sq += float((d * d).sum())
        n += int(mm.sum())
    rms = float(np.sqrt(sq / n)) if n else float("nan")
    return {"model": model, "poses": poses, "rms_px": rms, "success": bool(out.success)}
