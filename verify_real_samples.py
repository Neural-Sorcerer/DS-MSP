"""
End-to-end verification of the DS-MSP changes on REAL images + REAL calibrated
parameters. Exercises every code path touched in this review and writes labeled
visual outputs to `verification_output/` for manual confirmation.

Paths verified per image:
  1. Image undistortion via DoubleSphereCamera.undistort_image  (analytic)
  2. Image undistortion via ds_msp.cv.undistortImage            (decoupled wrapper) -> must match (1)
  3. Image undistortion via the TI LDC displacement mesh        (hardware-style)   -> compared to (1)
  4. PnP pose (solve_pnp) + reprojection overlay in the distorted image (FOV-fixed project)
  5. 3D axes drawn from the estimated pose (draw_axes)
  6. Keypoint undistortion: DS analytic vs LDC fixed-point, on the undistorted image
"""

import os
import json
import cv2
import numpy as np

from ds_msp import DoubleSphereCamera
import ds_msp.cv as dscv
from ds_msp.ldc import TI_LDC_MeshGenerator, TI_LDC_PointUndistorter
from ds_msp.utils import build_checkerboard_points

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "verification_output")
os.makedirs(OUT, exist_ok=True)

BALANCE = 0.5
DSAMP = 4  # LDC downsample factor (step = 16)


def label(img, text, org=(20, 45), color=(0, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
    return img


def ldc_full_res_map(mesh_float, step, W, H):
    """Upsample the downsampled LDC displacement mesh to a full-res cv2.remap map."""
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    mx = xs / step
    my = ys / step
    mx0 = np.clip(np.floor(mx).astype(int), 0, mesh_float.shape[1] - 2)
    my0 = np.clip(np.floor(my).astype(int), 0, mesh_float.shape[0] - 2)
    fx = mx - mx0
    fy = my - my0
    Q00 = mesh_float[my0, mx0]
    Q10 = mesh_float[my0, mx0 + 1]
    Q01 = mesh_float[my0 + 1, mx0]
    Q11 = mesh_float[my0 + 1, mx0 + 1]
    delta = (
        ((1 - fx) * (1 - fy))[..., None] * Q00
        + (fx * (1 - fy))[..., None] * Q10
        + ((1 - fx) * fy)[..., None] * Q01
        + (fx * fy)[..., None] * Q11
    )
    map_x = (xs + delta[..., 0]).astype(np.float32)
    map_y = (ys + delta[..., 1]).astype(np.float32)
    return map_x, map_y


def main():
    calib = json.load(open(os.path.join(ROOT, "results", "calibration_params.json")))
    cfg = json.load(open(os.path.join(ROOT, "test_config.json")))
    fx, fy = calib["fx"], calib["fy"]
    cx, cy = calib["cx"], calib["cy"]
    xi, alpha = calib["xi"], calib["alpha"]
    W, H = calib["image_width"], calib["image_height"]
    ph, pw, pL = calib["checkerboard_rows"], calib["checkerboard_cols"], calib["pLength"]
    board = build_checkerboard_points(ph, pw, pL)

    print(f"Camera: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} xi={xi:.4f} alpha={alpha:.4f}")

    cam = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha, W, H)
    K = cam.K
    D = cam.D

    # LDC mesh (shared K_new with the analytic path by construction)
    ldc = TI_LDC_MeshGenerator(cam)
    mesh_res = ldc.generate_mesh_and_intrinsics(W, H, downsample_factor=DSAMP, balance=BALANCE)
    K_new = mesh_res["K_new"]
    mesh_float = mesh_res["mesh_lut_float"]
    ldc_pt = TI_LDC_PointUndistorter(mesh_float, K_new, DSAMP, W, H)

    summary = {}

    for entry in cfg["test_images"]:
        name = os.path.splitext(os.path.basename(entry["file"]))[0]
        path = os.path.join(ROOT, entry["file"])
        if not os.path.exists(path):
            path = os.path.join(ROOT, "assets", os.path.basename(entry["file"]))
        img = cv2.imread(path)
        if img is None:
            print(f"  SKIP {name}: image not found")
            continue
        kpts = np.array(entry["keypoints_2d"], dtype=np.float64)
        print(f"\n=== {name} ===")
        s = {}

        # ---- (1) analytic undistort + (2) cv wrapper ----
        und_a, _ = cam.undistort_image(img, K_new)
        und_cv = dscv.undistortImage(img, K, D, Knew=K_new, new_size=(W, H))
        diff = float(np.abs(und_a.astype(int) - und_cv.astype(int)).max())
        s["undistort_model_vs_cvwrapper_max_pixel_diff"] = diff
        print(f"  undistort: model vs cv-wrapper max pixel diff = {diff}")

        # ---- (3) LDC mesh image undistort ----
        mx, my = ldc_full_res_map(mesh_float, 2 ** DSAMP, W, H)
        und_ldc = cv2.remap(img, mx, my, cv2.INTER_LINEAR)

        # ---- (4) PnP + reprojection (distorted domain) ----
        ok, rvec, tvec = cam.solve_pnp(board, kpts)
        R, _ = cv2.Rodrigues(rvec)
        Xc = (R @ board.T).T + tvec
        uv_re, valid = cam.project(Xc)
        err = np.linalg.norm(uv_re[valid] - kpts[valid], axis=1)
        rms = float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan")
        s["pnp_success"] = bool(ok)
        s["reprojection_rms_px"] = rms
        s["t_world"] = [round(float(v), 4) for v in tvec]
        print(f"  PnP ok={ok}  reprojection RMS={rms:.3f}px  t={s['t_world']}")

        repro = img.copy()
        for (uo, vo), (ur, vr) in zip(kpts, uv_re):
            cv2.circle(repro, (int(uo), int(vo)), 6, (255, 0, 0), -1)   # observed blue
            cv2.circle(repro, (int(ur), int(vr)), 4, (0, 0, 255), -1)   # reproj   red
            cv2.line(repro, (int(uo), int(vo)), (int(ur), int(vr)), (0, 255, 0), 1)
        label(repro, f"{name}: reproj  obs=blue pred=red  RMS={rms:.2f}px")

        # ---- (5) draw axes from pose ----
        axes = cam.draw_axes(img.copy(), rvec, tvec, axis_length=0.4)
        label(axes, f"{name}: pose axes (R=x,G=y,B=z)")

        # ---- (6) keypoint undistortion: DS analytic vs LDC fixed-point ----
        kp_ds, vds = cam.undistort_points(kpts, K_new)
        kp_ldc, vldc = ldc_pt.undistort_points(kpts)
        both = vds & vldc
        kp_gap = float(np.linalg.norm(kp_ds[both] - kp_ldc[both], axis=1).max()) if both.any() else float("nan")
        s["kp_undistort_ds_vs_ldc_max_gap_px"] = kp_gap
        print(f"  keypoint undistort: DS-analytic vs LDC max gap = {kp_gap:.3f}px")

        und_pts = und_a.copy()
        for (u, v) in kp_ds[vds]:
            cv2.circle(und_pts, (int(u), int(v)), 6, (0, 0, 255), -1)        # DS analytic red
        for (u, v) in kp_ldc[vldc]:
            cv2.circle(und_pts, (int(u), int(v)), 3, (0, 255, 255), -1)      # LDC yellow
        label(und_pts, f"{name}: undistorted kpts  DS=red  LDC=yellow")

        # ---- montages ----
        def stack(a, b, lab_a, lab_b):
            a, b = a.copy(), b.copy()
            label(a, lab_a); label(b, lab_b)
            return np.hstack([a, b])

        cv2.imwrite(os.path.join(OUT, f"{name}_1_undistort_model_vs_ldc.jpg"),
                    stack(und_a, und_ldc, f"{name}: analytic undistort", f"{name}: TI-LDC mesh undistort"))
        cv2.imwrite(os.path.join(OUT, f"{name}_2_reprojection.jpg"), repro)
        cv2.imwrite(os.path.join(OUT, f"{name}_3_pose_axes.jpg"), axes)
        cv2.imwrite(os.path.join(OUT, f"{name}_4_undistorted_keypoints.jpg"), und_pts)
        summary[name] = s

    json.dump(summary, open(os.path.join(OUT, "verification_summary.json"), "w"), indent=2)
    print(f"\nSaved outputs + verification_summary.json to: {OUT}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
