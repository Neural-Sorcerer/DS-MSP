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

from ..core.contracts import CameraModel
from ..core.lie import so3_exp
from ..core.optimize import lm_solve

#: Map the historical SciPy ``loss`` names to the in-house IRLS kernels.
_LOSS_TO_KERNEL = {
    "linear": "none", "huber": "huber", "cauchy": "cauchy",
    "soft_l1": "pseudo_huber",
}


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
    total = 2 * sum(sizes)
    masks = [np.asarray(v, bool) for v in visibility_list]
    lb_i, ub_i = cls.param_bounds()

    rvecs, tvecs = _seed_poses(init_model, X_world_list, keypoints_list, visibility_list)
    # Manifold state: each seed rotation is kept as a *base matrix* re-based every accepted step by
    # the solver (R ← R·exp([δω]_×), δω reset to 0). Because δω is always linearized at 0 the
    # retraction Jacobian J_r(0) = I drops out — the extrinsics Jacobian is the cheap -R[Xw]_×, and
    # δω never drifts toward the ‖r‖=π singularity. This is exactly what a black-box scipy solve
    # (fixed base for the whole solve) cannot do, and why this path is both faster and stabler.
    R0 = np.stack([cv2.Rodrigues(np.asarray(r, float))[0] for r in rvecs])   # (n,3,3)
    t0 = np.stack([np.asarray(t, float) for t in tvecs])                     # (n,3)
    state0 = (np.clip(init_model.params, lb_i, ub_i).copy(), R0, t0)

    def residual(state):
        params, Rb, t = state
        m = cls.from_params(params)
        out = np.zeros((total,))
        row = 0
        for i, (Xw, uv) in enumerate(zip(X_world_list, keypoints_list)):
            N = sizes[i]
            Xc = (Rb[i] @ Xw.T).T + t[i]
            uvp, valid = m.project(Xc)
            mask = masks[i] & valid
            diff = np.zeros_like(uv, dtype=np.float64)
            diff[mask] = uvp[mask] - uv[mask]
            out[row:row + 2 * N] = diff.ravel()
            row += 2 * N
        return out

    def jacobian(state):
        params, Rb, t = state
        m = cls.from_params(params)
        J = np.zeros((total, P + 6 * n_img))
        row = 0
        for i, (Xw, uv) in enumerate(zip(X_world_list, keypoints_list)):
            N = sizes[i]
            Xc = (Rb[i] @ Xw.T).T + t[i]
            _, J_point, J_param, valid = m.project_jacobian(Xc)
            mask = (masks[i] & valid)[:, None, None].astype(np.float64)
            # δω linearized at 0 ⇒ J_r = I, so ∂Xc/∂δω = -R[Xw]_×; ∂Xc/∂δt = I.
            dXc_dw = -np.einsum('ij,njk->nik', Rb[i], _skew_batch(Xw))
            J_rvec = np.einsum('nij,njc->nic', J_point, dXc_dw)
            J_ext = np.concatenate([J_rvec, J_point], axis=-1) * mask
            J[row:row + 2 * N, 0:P] = (J_param * mask).reshape(2 * N, P)
            ec = P + 6 * i
            J[row:row + 2 * N, ec:ec + 6] = J_ext.reshape(2 * N, 6)
            row += 2 * N
        return J

    def retract(state, delta):
        params, Rb, t = state
        params = np.clip(params + delta[:P], lb_i, ub_i)         # keep intrinsics valid
        Rb, t = Rb.copy(), t.copy()
        for i in range(n_img):
            d = delta[P + 6 * i:P + 6 * i + 6]
            Rb[i] = Rb[i] @ so3_exp(d[:3])
            t[i] = t[i] + d[3:]
        return (params, Rb, t)

    kernel = _LOSS_TO_KERNEL.get(loss, loss)
    out = lm_solve(state0, residual, jacobian, retract, block=2, max_iter=max_nfev,
                   robust_kernel=kernel, robust_scale=(f_scale if kernel != "none" else 1.0))
    params, Rb, t = out.state
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
