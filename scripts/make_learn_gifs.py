#!/usr/bin/env python3
"""
Generate the visual assets used by the learning curriculum (docs/learn/), all from the
real TUM-VI data so the docs *show* what the code does.

    python scripts/make_learn_gifs.py        # writes assets/learn/*.gif

Produces:
  - aprilgrid_detection.gif        AprilGrid corners detected on real cam0 calib frames
  - calibration_reprojection.gif   detected (green) vs reprojected (red) corners after
                                   calibration — the model predicting where corners land
  - stereo_pair.gif                synchronized cam0 | cam1 views with detections (the
                                   input to stereo extrinsic calibration)

Needs `pip install -e ".[calib]"`, Pillow, and the TUM-VI download.
"""
from __future__ import annotations

import glob
import os

import cv2
import numpy as np
from PIL import Image

from ds_msp.calib import AprilGridTarget, calibrate, detect_aprilgrid
from ds_msp.models import KannalaBrandtModel

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ = os.path.join(HERE, "datasets", "tumvi", "dataset-calib-cam1_512_16", "mav0")
OUT = os.path.join(HERE, "assets", "learn")
TARGET = AprilGridTarget(tag_rows=6, tag_cols=6, tag_size=0.088, tag_spacing=0.3)

GREEN, RED, CYAN = (0, 230, 0), (0, 0, 255), (255, 200, 0)


def load_gray_u8(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im.dtype != np.uint8:
        im = (im / 256).astype(np.uint8)
    return np.ascontiguousarray(im)


def to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def caption(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    return img


def save_gif(frames_bgr, name, duration=350, scale=1.0):
    pil = []
    for f in frames_bgr:
        if scale != 1.0:
            f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        pil.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    path = os.path.join(OUT, name)
    pil[0].save(path, save_all=True, append_images=pil[1:], loop=0, duration=duration,
                optimize=True)
    print(f"  wrote assets/learn/{name}  ({os.path.getsize(path) / 1e6:.2f} MB,"
          f" {len(pil)} frames)")


def draw_detections(img_bgr, dets, color=GREEN, dot=3):
    for tag_id, corners in dets.items():
        c = np.asarray(corners, np.int32)
        cv2.polylines(img_bgr, [c.reshape(-1, 1, 2)], True, color, 1, cv2.LINE_AA)
        for p in corners:
            cv2.circle(img_bgr, (int(p[0]), int(p[1])), dot, color, -1, cv2.LINE_AA)
        ctr = corners.mean(0).astype(int)
        cv2.putText(img_bgr, str(int(tag_id)), tuple(ctr - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, color, 1, cv2.LINE_AA)
    return img_bgr


def gif_detection(cam0_paths):
    frames = []
    for path, dets in zip(cam0_paths, detect_aprilgrid(cam0_paths, min_tags=0)):
        img = draw_detections(to_bgr(load_gray_u8(path)), dets)
        caption(img, f"AprilGrid detected: {len(dets)} tags  ({len(dets) * 4} corners)")
        frames.append(img)
    save_gif(frames, "aprilgrid_detection.gif")


def gif_reprojection(cam0_paths):
    # detect keeping index alignment, then keep frames with enough of the board
    det_all = detect_aprilgrid(cam0_paths, min_tags=0)
    keep = [i for i, d in enumerate(det_all) if len(d) >= 8]
    paths = [cam0_paths[i] for i in keep]
    dets = [det_all[i] for i in keep]
    Xs, UVs, VIs = TARGET.build_correspondences(dets, min_corners=8)
    out = calibrate(KannalaBrandtModel(190, 190, 256, 256), Xs, UVs, VIs,
                    max_nfev=120, loss="cauchy", f_scale=0.5)
    model, poses = out["model"], out["poses"]

    frames = []
    for path, X, uv, (rvec, tvec) in zip(paths, Xs, UVs, poses):
        R, _ = cv2.Rodrigues(rvec)
        proj, _ = model.project((R @ X.T).T + tvec)
        err = float(np.linalg.norm(proj - uv, axis=1).mean())
        img = to_bgr(load_gray_u8(path))
        for (ud, vd), (up, vp) in zip(uv, proj):
            cv2.circle(img, (int(ud), int(vd)), 4, GREEN, 1, cv2.LINE_AA)   # detected
            cv2.circle(img, (int(up), int(vp)), 2, RED, -1, cv2.LINE_AA)    # reprojected
        caption(img, f"green=detected  red=reprojected   mean error {err:.3f} px")
        frames.append(img)
    # show a spread of ~10 of the calibrated frames
    step = max(1, len(frames) // 10)
    save_gif(frames[::step][:10], "calibration_reprojection.gif", duration=500)


def gif_stereo(names):
    frames = []
    for n in names:
        p0 = os.path.join(SEQ, "cam0", "data", n)
        p1 = os.path.join(SEQ, "cam1", "data", n)
        i0 = draw_detections(to_bgr(load_gray_u8(p0)),
                             detect_aprilgrid([p0], min_tags=0)[0], CYAN, dot=2)
        i1 = draw_detections(to_bgr(load_gray_u8(p1)),
                             detect_aprilgrid([p1], min_tags=0)[0], CYAN, dot=2)
        caption(i0, "cam0")
        caption(i1, "cam1  (same instant)")
        gap = np.full((i0.shape[0], 6, 3), 255, np.uint8)
        frames.append(np.hstack([i0, gap, i1]))
    save_gif(frames, "stereo_pair.gif", duration=450, scale=0.85)


def main() -> None:
    if not os.path.isdir(os.path.join(SEQ, "cam0", "data")):
        raise SystemExit("Fetch TUM-VI first: bash scripts/download_datasets.sh tumvi")
    os.makedirs(OUT, exist_ok=True)
    all_c0 = sorted(glob.glob(os.path.join(SEQ, "cam0", "data", "*.png")))
    names = [os.path.basename(p) for p in all_c0]

    # coarse scan -> pick ~10 frames where the grid is well seen, spread out
    scan = list(range(0, len(all_c0), 6))
    seen = [(i, len(d)) for i, d in
            zip(scan, detect_aprilgrid([all_c0[i] for i in scan], min_tags=0))]
    good = [i for i, n in seen if n >= 18]
    pick = good[:: max(1, len(good) // 10)][:10]
    print(f"Generating learn GIFs from {len(pick)} well-detected frames ...")

    gif_detection([all_c0[i] for i in pick])
    gif_stereo([names[i] for i in pick])
    gif_reprojection(all_c0[::6])


if __name__ == "__main__":
    main()
