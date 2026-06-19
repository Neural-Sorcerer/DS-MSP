#!/usr/bin/env python3
"""
Chapter 1 demo — a real fisheye camera, end to end, in ~30 lines of library calls.

What this teaches (run it, read the printed numbers):
  1. A *published* fisheye calibration is just a few numbers in a YAML file.
     We load TUM-VI's reference camchain straight into a camera model.
  2. project() and unproject() are inverses. We measure the round-trip error on
     a real image grid — it should be ~1e-12 px (machine precision), which is how
     you *prove* an unprojection is correct rather than hoping it is.
  3. Undistortion = "what would a pinhole camera have seen?". We rectify a real
     fisheye frame and save before/after so you can see the bent lines straighten.

Prerequisites:
  - `uv pip install -e .` (or `pip install -e .`) in a venv.
  - TUM-VI room1 downloaded: `bash scripts/download_datasets.sh tumvi`.

Run:
  python examples/01_realdata_fisheye_tumvi.py
"""
from __future__ import annotations

import glob
import os

import cv2
import numpy as np

from ds_msp import Undistorter
from ds_msp.io.kalibr import load_kalibr_with_resolution

# --- locate the downloaded TUM-VI room1 sequence -----------------------------
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ = os.path.join(HERE, "datasets", "tumvi", "dataset-room1_512_16")
CAMCHAIN = os.path.join(SEQ, "dso", "camchain.yaml")
OUT = os.path.join(HERE, "results", "learn")


def main() -> None:
    if not os.path.exists(CAMCHAIN):
        raise SystemExit(
            "TUM-VI room1 not found. Fetch it first:\n"
            "    bash scripts/download_datasets.sh tumvi"
        )
    os.makedirs(OUT, exist_ok=True)

    # 1) A calibration is just numbers. Load the PUBLISHED reference into a model.
    #    TUM-VI ships `pinhole + equidistant` = Kannala-Brandt (OpenCV fisheye).
    cam, (W, H) = load_kalibr_with_resolution(CAMCHAIN, cam="cam0")
    print(f"[1] loaded published TUM-VI cam0 calibration -> {cam.name} model "
          f"({W}x{H})")
    print(f"    {cam}")

    # 2) project() and unproject() must be inverses. Measure it on a real grid.
    #    Take a lattice of pixels, unproject to rays, project back, compare.
    us, vs = np.meshgrid(np.linspace(20, W - 20, 40), np.linspace(20, H - 20, 40))
    pix = np.stack([us.ravel(), vs.ravel()], axis=-1).astype(np.float64)
    rays, ok = cam.unproject(pix)               # 2D pixels -> 3D unit bearing rays
    back, ok2 = cam.project(rays)               # 3D rays   -> 2D pixels again
    m = ok & ok2
    err = np.linalg.norm(back[m] - pix[m], axis=-1)
    print(f"[2] project(unproject(x)) round-trip on {m.sum()} pixels: "
          f"mean={err.mean():.2e}px  max={err.max():.2e}px   (≈machine precision)")

    # 3) Undistort a real frame: what a pinhole camera would have seen.
    img_path = sorted(glob.glob(os.path.join(SEQ, "mav0", "cam0", "data", "*.png")))[0]
    img = cv2.imread(img_path)
    und = Undistorter(cam, W, H)
    rect, K_new = und.undistort_image(img, K_new=und.new_K(balance=0.0))
    cv2.imwrite(os.path.join(OUT, "01_fisheye_raw.png"), img)
    cv2.imwrite(os.path.join(OUT, "01_fisheye_rectified.png"), rect)
    print(f"[3] undistorted a real frame -> {OUT}/01_fisheye_rectified.png")
    print(f"    rectified pinhole intrinsics: fx={K_new[0,0]:.1f} cx={K_new[0,2]:.1f}")

    print("\nDone. Open the two images in results/learn/ and watch the lens "
          "curvature straighten out.")


if __name__ == "__main__":
    main()
