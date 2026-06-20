"""
Generic bundle-adjustment calibration for ANY camera model.

Given checkerboard correspondences and an initial model (for intrinsics seed +
the model type), jointly refines intrinsics and per-image extrinsics by
Levenberg-Marquardt using the model's **analytic** projection Jacobian (no
autodiff). Works for DS/UCM/EUCM/KB/RadTan or any ``CameraModel``.

Decoupled: imports only the contract + SciPy/OpenCV; the model type comes from
the injected ``init_model`` (no concrete-model import).
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
from scipy.optimize import least_squares

from ..core.contracts import CameraModel
from ..core.lie import so3_exp, so3_right_jacobian


def _skew_batch(V: np.ndarray) -> np.ndarray:
    """Stack of skew-symmetric matrices ``[Vₙ]_×``, shape ``(N, 3, 3)``."""
    K = np.zeros((V.shape[0], 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -V[:, 2], V[:, 1]
    K[:, 1, 0], K[:, 1, 2] = V[:, 2], -V[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -V[:, 1], V[:, 0]
    return K


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
    P = len(cls.param_names)
    n_img = len(X_world_list)
    sizes = [len(X) for X in X_world_list]

    rvecs, tvecs = _seed_poses(init_model, X_world_list, keypoints_list, visibility_list)
    # Manifold-correct extrinsics: keep each seed rotation as a base matrix and optimize a *local*
    # perturbation δω (retracted via R_base·exp([δω]_×)) instead of an absolute axis-angle vector.
    # δω starts at 0 and stays small for a good seed, so it never nears the ‖r‖=π singularity.
    R_bases = [cv2.Rodrigues(np.asarray(r, float))[0] for r in rvecs]
    x0 = np.concatenate([init_model.params]
                        + [np.concatenate([np.zeros(3), t]) for t in tvecs])

    lb_i, ub_i = cls.param_bounds()
    ext_lb = np.array([-np.pi, -np.pi, -np.pi, -10.0, -10.0, 1e-3])
    ext_ub = np.array([np.pi, np.pi, np.pi, 10.0, 10.0, 50.0])
    lb = np.concatenate([lb_i] + [ext_lb] * n_img)
    ub = np.concatenate([ub_i] + [ext_ub] * n_img)
    x0 = np.clip(x0, lb, ub)

    def residual(p):
        m = cls.from_params(p[:P])
        out = []
        off = P
        for i, (Xw, uv, vis) in enumerate(zip(X_world_list, keypoints_list, visibility_list)):
            dw, t = p[off:off + 3], p[off + 3:off + 6]
            off += 6
            R = R_bases[i] @ so3_exp(dw)
            Xc = (R @ Xw.T).T + t
            uvp, valid = m.project(Xc)
            diff = np.zeros_like(uv, dtype=np.float64)
            mask = vis & valid
            diff[mask] = uvp[mask] - uv[mask]
            out.append(diff.ravel())
        return np.concatenate(out)

    def jac(p):
        m = cls.from_params(p[:P])
        J = np.zeros((2 * sum(sizes), P + 6 * n_img), dtype=np.float64)
        row = 0
        off = P
        for i, (Xw, uv, vis) in enumerate(zip(X_world_list, keypoints_list, visibility_list)):
            dw = p[off:off + 3]
            t = p[off + 3:off + 6]
            off += 6
            R = R_bases[i] @ so3_exp(dw)
            Xc = (R @ Xw.T).T + t
            _, J_point, J_param, valid = m.project_jacobian(Xc)
            mask = (vis & valid)[:, None, None].astype(np.float64)
            # ∂Xc/∂δω = -R [Xw]_× J_r(δω)  (right-perturbation Jacobian of the retraction)
            dXc_dw = -np.einsum('ij,njk,kl->nil', R, _skew_batch(Xw), so3_right_jacobian(dw))
            J_rvec = np.einsum('nij,njc->nic', J_point, dXc_dw)
            J_ext = np.concatenate([J_rvec, J_point], axis=-1) * mask
            J_par = J_param * mask
            N = sizes[i]
            J[row:row + 2 * N, 0:P] = J_par.reshape(2 * N, P)
            ec = P + 6 * i
            J[row:row + 2 * N, ec:ec + 6] = J_ext.reshape(2 * N, 6)
            row += 2 * N
        return J

    res = least_squares(residual, x0, jac=jac, bounds=(lb, ub),
                        method="trf", x_scale="jac", max_nfev=max_nfev, verbose=verbose,
                        loss=loss, f_scale=f_scale)

    model = cls.from_params(res.x[:P])
    # Convert each optimized perturbation back to an absolute (rvec, tvec) so the returned poses
    # match the original interface (downstream code does cv2.Rodrigues(rvec)).
    poses = []
    for i in range(n_img):
        dw = res.x[P + 6 * i:P + 6 * i + 3]
        t = res.x[P + 6 * i + 3:P + 6 * i + 6]
        R = R_bases[i] @ so3_exp(dw)
        poses.append((cv2.Rodrigues(R)[0].ravel(), np.asarray(t, float)))

    # True reprojection RMS over valid observations. Computed directly (not from
    # res.cost) so it means the same thing under any robust ``loss``: a robust
    # kernel reshapes the cost, but the pixel error of the fit is what we report.
    sq, n = 0.0, 0
    for (rvec, t), Xw, uv, vis in zip(poses, X_world_list, keypoints_list, visibility_list):
        R, _ = cv2.Rodrigues(rvec)
        uvp, valid = model.project((R @ Xw.T).T + t)
        m = vis & valid
        d = uvp[m] - uv[m]
        sq += float((d * d).sum())
        n += int(m.sum())
    rms = float(np.sqrt(sq / n)) if n else float("nan")
    return {"model": model, "poses": poses, "rms_px": rms, "success": bool(res.success)}
