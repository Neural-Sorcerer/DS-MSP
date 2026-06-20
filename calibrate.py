"""
Part 1.2: Double Sphere Camera Calibration

This module defines:
- Parameter vector layout (intrinsics + DS params + per-image extrinsics)
- Residual function for non-linear least squares calibration
- Synthetic test harness to validate the pipeline
- Real COCO-based calibration entry point

"""

from typing import List, Tuple

import json
import numpy as np
import cv2
import os
from scipy.optimize import least_squares

from ds_msp.model import ds_project, ds_project_jacobian, DoubleSphereCamera
from ds_msp.utils import (
    pack_params,
    unpack_params,
    load_coco_calibration,
)


# ================================================================
# Residual function for least squares
# ================================================================


def calibration_residuals(
    params: np.ndarray,
    X_world_list: List[np.ndarray],
    keypoints_list: List[np.ndarray],
    visibility_list: List[np.ndarray],
) -> np.ndarray:
    """
    Compute reprojection residuals for all images and points.

    CRITICAL: residual vector length MUST be constant across all evaluations.

    For each image i and each point j:
        - if point is visible (vis_ij == True) AND projectable by DS (valid_ds_ij == True):
              residual = uv_proj_ij - uv_obs_ij
        - else:
              residual = [0, 0]  (no contribution)

    That way, residual dimension = num_images * N_points * 2 is fixed.
    """
    num_images = len(X_world_list)
    assert (
        num_images == len(keypoints_list) == len(visibility_list)
    ), "Inconsistent list lengths for images / keypoints / visibility"

    # Unpack global + per-image params
    fx, fy, cx, cy, xi, alpha, r_list, t_list = unpack_params(
        params, num_images=num_images
    )

    residuals_all = []

    for i in range(num_images):
        Xw = X_world_list[i]  # (N, 3)
        uv_obs = keypoints_list[i]  # (N, 2)
        vis = visibility_list[i]  # (N,)

        assert (
            Xw.shape[0] == uv_obs.shape[0] == vis.shape[0]
        ), f"Mismatch in N for image {i}"

        # --- World to camera: X_cam = R_i * X_world + t_i ---
        r_i = r_list[i]
        t_i = t_list[i]

        R_i, _ = cv2.Rodrigues(r_i.astype(np.float64))
        Xc = (R_i @ Xw.T).T + t_i.reshape(1, 3)

        # --- Project using Double Sphere model ---
        u_proj, v_proj, valid_ds = ds_project(Xc, fx, fy, cx, cy, xi, alpha)
        uv_proj = np.stack([u_proj, v_proj], axis=-1)

        # Combine annotation visibility + DS validity
        valid = vis & valid_ds

        # Fixed-size residuals: start with zeros
        diff = np.zeros_like(uv_obs, dtype=np.float64)  # (N, 2)

        # Only where valid do we use true reprojection error
        diff[valid] = uv_proj[valid] - uv_obs[valid]

        # Append flattened residuals for this image (2N,)
        residuals_all.append(diff.reshape(-1))

    # Final residual vector: shape = (num_images * N * 2,)
    return np.concatenate(residuals_all, axis=0)


def calibration_jacobian(
    params: np.ndarray,
    X_world_list: List[np.ndarray],
    keypoints_list: List[np.ndarray],
    visibility_list: List[np.ndarray],
) -> np.ndarray:
    """
    Analytic Jacobian of `calibration_residuals` w.r.t. the full parameter vector.

    The residual for image i / point j depends only on the 6 shared intrinsics
    and that image's 6 extrinsic parameters, so the Jacobian is block-sparse.
    For each block we chain the analytic Double Sphere projection Jacobian with
    the world->camera transform:

        d(residual) / d(intrinsics) = J_intr
        d(residual) / d(rvec)       = J_point @ d(R(rvec) Xw) / d(rvec)
        d(residual) / d(tvec)       = J_point          (since d(R Xw + t)/dt = I)

    Rows for invisible / non-projectable points are left at zero, matching the
    fixed-size residual convention. Supplying this to least_squares replaces
    finite differencing, cutting function evaluations and improving conditioning.
    """
    num_images = len(X_world_list)
    fx, fy, cx, cy, xi, alpha, r_list, t_list = unpack_params(
        params, num_images=num_images
    )

    sizes = [Xw.shape[0] for Xw in X_world_list]
    m = int(2 * sum(sizes))
    n = 6 + 6 * num_images
    J = np.zeros((m, n), dtype=np.float64)

    row = 0
    for i in range(num_images):
        Xw = X_world_list[i]
        vis = visibility_list[i]
        N = sizes[i]

        r_i = r_list[i].astype(np.float64)
        t_i = t_list[i]
        R_i, jacR = cv2.Rodrigues(r_i)
        Xc = (R_i @ Xw.T).T + t_i.reshape(1, 3)

        _, _, J_point, J_intr, valid_ds = ds_project_jacobian(
            Xc, fx, fy, cx, cy, xi, alpha
        )
        valid = vis & valid_ds

        # d(R(rvec) Xw)/d(rvec): dR[a,b,c] = dR_ab/drvec_c (see layout note above).
        dR = jacR.T.reshape(3, 3, 3)
        dXc_dr = np.einsum('abc,nb->nac', dR, Xw)            # (N, 3, 3)
        J_rvec = np.einsum('nij,njc->nic', J_point, dXc_dr)  # (N, 2, 3)
        J_tvec = J_point                                     # (N, 2, 3)

        # Zero out rows that contribute no residual.
        mask = valid[:, None, None].astype(np.float64)
        J_intr = J_intr * mask
        J_ext = np.concatenate([J_rvec, J_tvec], axis=-1) * mask  # (N, 2, 6)

        J[row:row + 2 * N, 0:6] = J_intr.reshape(2 * N, 6)
        ecol = 6 + 6 * i
        J[row:row + 2 * N, ecol:ecol + 6] = J_ext.reshape(2 * N, 6)
        row += 2 * N

    return J


# ================================================================
# Bounds for optimization
# ================================================================


def build_bounds(
    num_images: int, img_width: float, img_height: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build lower/upper bounds for all parameters:
      - intrinsics: fx, fy, cx, cy, xi, alpha
      - per-image extrinsics: r_i (3), t_i (3) for i in [0, num_images)

    Bounds are deliberately wide but rule out crazy configurations.
    """
    # ---- Intrinsic + DS bounds ----
    # Focal lengths: between 500 and 4000 pixels
    fx_lb, fy_lb = 500.0, 500.0
    fx_ub, fy_ub = 4000.0, 4000.0

    # Principal point: anywhere on the image plane
    cx_lb, cy_lb = 0.0, 0.0
    cx_ub, cy_ub = img_width, img_height

    # DS params (well-posed domain of the Double Sphere model):
    #   xi in [-1, 1]  -- matches the basalt/kalibr reference clamp. Real fisheye
    #     lenses sit roughly in [-0.2, 0.6]; xi > 1 drives the model into a
    #     non-injective ("folding") regime where unprojection is no longer the
    #     inverse of projection, and xi < 0 is required for some real lenses, so
    #     the old [0.1, 3.0] range was wrong on both ends.
    #   alpha in (0, 1)
    xi_lb, xi_ub = -1.0, 1.0
    alpha_lb, alpha_ub = 0.01, 0.99

    intr_lb = np.array([fx_lb, fy_lb, cx_lb, cy_lb, xi_lb, alpha_lb], dtype=np.float64)
    intr_ub = np.array([fx_ub, fy_ub, cx_ub, cy_ub, xi_ub, alpha_ub], dtype=np.float64)

    # ---- Extrinsics bounds ----
    # Rotation as Rodrigues: each component in [-pi, pi]
    rot_lb = -np.pi * np.ones(3, dtype=np.float64)
    rot_ub = np.pi * np.ones(3, dtype=np.float64)

    # Translation: x,y in [-5, 5] meters, z in [0.2, 20]
    # (board is always in front of camera)
    tx_lb, ty_lb, tz_lb = -5.0, -5.0, 0.2
    tx_ub, ty_ub, tz_ub = 5.0, 5.0, 20.0

    trans_lb = np.array([tx_lb, ty_lb, tz_lb], dtype=np.float64)
    trans_ub = np.array([tx_ub, ty_ub, tz_ub], dtype=np.float64)

    # Pack all bounds
    lb_list = [intr_lb]
    ub_list = [intr_ub]

    for _ in range(num_images):
        lb_list.append(rot_lb)
        lb_list.append(trans_lb)
        ub_list.append(rot_ub)
        ub_list.append(trans_ub)

    lb = np.concatenate(lb_list, axis=0)
    ub = np.concatenate(ub_list, axis=0)

    return lb, ub


# ================================================================
# Real-data calibration main
# ================================================================


def main_real():
    """
    Part 1.2 – Real calibration pipeline.
    Runs Double Sphere calibration using COCO keypoints,
    then saves results into `results/` directory:

        results/calibration_params.json
        results/poses.json
        results/metrics.json

    Validation scripts (validate.py) will read these files.
    """
    # --------------------------------------------------------------
    # 0. Paths + checkerboard settings
    # --------------------------------------------------------------
    # Local path resolution for portability
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "anns.json")
    ph = 5
    pw = 6
    pLength = 0.20  # 20 cm spacing

    # --------------------------------------------------------------
    # 1. Load COCO data
    # --------------------------------------------------------------
    (X_world_list, keypoints_list, visibility_list, image_info_list) = (
        load_coco_calibration(json_path, ph, pw, pLength)
    )

    num_images = len(X_world_list)
    if num_images == 0:
        raise RuntimeError("No checkerboard annotations found.")

    img_width = float(image_info_list[0]["width"])
    img_height = float(image_info_list[0]["height"])

    # --------------------------------------------------------------
    # 2. Initial guess (Intrinsics)
    # --------------------------------------------------------------
    f0 = 0.8 * max(img_width, img_height)
    fx0 = fy0 = f0
    cx0 = img_width / 2
    cy0 = img_height / 2
    xi0 = 0.5
    alpha0 = 0.5

    # --------------------------------------------------------------
    # 3. Robust Per-Image Extrinsics Initialization via PnP
    # --------------------------------------------------------------
    r0_list = []
    t0_list = []
    cam0 = DoubleSphereCamera(fx0, fy0, cx0, cy0, xi0, alpha0, int(img_width), int(img_height))

    for i in range(num_images):
        Xw = X_world_list[i]
        uv_obs = keypoints_list[i]
        vis = visibility_list[i]
        
        # Filter valid points
        Xw_valid = Xw[vis].astype(np.float64)
        uv_valid = uv_obs[vis].astype(np.float64)
        
        # Unproject to normalized plane
        rays, valid_unproj = cam0.unproject(uv_valid)
        
        Xw_pnp = Xw_valid[valid_unproj]
        rays_pnp = rays[valid_unproj]
        
        if len(rays_pnp) < 4:
            # Fallback
            rvec0 = np.zeros(3)
            tvec0 = np.array([0.0, 0.0, 1.5])
        else:
            # Solve PnP on normalized plane coordinates (x/z, y/z)
            pts_norm = rays_pnp[:, :2] / np.maximum(rays_pnp[:, 2:3], 1e-10)
            ret, rvec0, tvec0 = cv2.solvePnP(Xw_pnp, pts_norm, np.eye(3), None)
            if not ret:
                rvec0 = np.zeros(3)
                tvec0 = np.array([0.0, 0.0, 1.5])
                
        r0_list.append(rvec0.squeeze())
        t0_list.append(tvec0.squeeze())

    x0 = pack_params(fx0, fy0, cx0, cy0, xi0, alpha0, r0_list, t0_list)
    lb, ub = build_bounds(num_images, img_width, img_height)

    # --------------------------------------------------------------
    # 3. Run Levenberg–Marquardt (TRF) optimization
    # --------------------------------------------------------------
    result = least_squares(
        calibration_residuals,
        x0,
        jac=calibration_jacobian,
        args=(X_world_list, keypoints_list, visibility_list),
        method="trf",
        loss="linear",
        verbose=2,
        max_nfev=200,
        x_scale="jac",
        bounds=(lb, ub),
    )

    print("\n[REAL] Optimization success:", result.success)
    print("[REAL] Final cost:", result.cost)

    # --------------------------------------------------------------
    # 4. Unpack calibrated parameters
    # --------------------------------------------------------------
    fx, fy, cx, cy, xi, alpha, r_list, t_list = unpack_params(result.x, num_images)

    # RMS reprojection error (global)
    M = result.fun.shape[0]
    rms_px = np.sqrt(2 * result.cost / M)
    print(f"[REAL] RMS reprojection error: {rms_px:.4f} px")

    # --------------------------------------------------------------
    # 5. Save outputs to results/
    # --------------------------------------------------------------
    os.makedirs("results", exist_ok=True)

    # --- intrinsics + DS params ---
    calib_json = {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "xi": float(xi),
        "alpha": float(alpha),
        "image_width": int(img_width),
        "image_height": int(img_height),
        "checkerboard_rows": ph,
        "checkerboard_cols": pw,
        "pLength": pLength,
    }
    with open("results/calibration_params.json", "w") as f:
        json.dump(calib_json, f, indent=4)

    # --- extrinsics per image ---
    poses = {}
    for i, img in enumerate(image_info_list):
        poses[str(img["file_name"])] = {
            "r": r_list[i].tolist(),
            "t": t_list[i].tolist(),
        }
    with open("results/poses.json", "w") as f:
        json.dump(poses, f, indent=4)

    # --- metrics (global only; per-image in validate.py) ---
    metrics = {
        "global_rms_px": float(rms_px),
        "num_images": num_images,
    }
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nSaved:")
    print("  results/calibration_params.json")
    print("  results/poses.json")
    print("  results/metrics.json")
    print("\nRun validate.py to produce visualizations.")


if __name__ == "__main__":
    main_real()
