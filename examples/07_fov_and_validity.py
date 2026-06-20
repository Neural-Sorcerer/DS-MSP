#!/usr/bin/env python3
"""
Chapter 3 demo — the >180° validity cone, and the FOV-vs-coverage trade of undistortion.

A fisheye sees more than a hemisphere, but a pinhole image cannot hold it. This script puts
numbers on both facts, using the original Double Sphere calibration:

  1. THE VALIDITY LIMIT. The Double Sphere model can only project rays inside a half-space,
     `z > -w2·d1` (see ds_msp/models/ds_math.py). We compute the largest incidence angle the
     model accepts — the edge of the "valid cone" — analytically and confirm it numerically.
  2. THE UNDISTORTION TRADE. Rectifying to a pinhole view trades field of view for how much
     of the frame is filled. We sweep the `balance` knob and print both numbers, so the
     trade-off you *see* in the images is one you can also read off.

Prerequisites:
  - `pip install -e .`
  - the bundled `test_image.jpg` + `test_config.json` (in the repo)

Run:
  python examples/07_fov_and_validity.py
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from ds_msp import DoubleSphereCamera

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(HERE, "test_config.json")
IMG = os.path.join(HERE, "assets", "test_image.jpg")


def main() -> None:
    intr = json.load(open(CFG))["intrinsics"]
    W, H = intr["width"], intr["height"]
    cam = DoubleSphereCamera(intr["fx"], intr["fy"], intr["cx"], intr["cy"],
                             intr["xi"], intr["alpha"], width=W, height=H)
    xi, alpha = cam.xi, cam.alpha

    # 1. the validity cone --------------------------------------------------------
    # For a unit ray d1=1, the half-space test z > -w2*d1 becomes cos(theta) > -w2.
    w1 = (1 - alpha) / alpha if alpha > 0.5 else alpha / (1 - alpha)
    w2 = (w1 + xi) / np.sqrt(2 * w1 * xi + xi * xi + 1.0)
    theta_max_analytic = np.degrees(np.arccos(-w2))

    # confirm it numerically: sweep rays from 0 to 180deg and ask which the model accepts
    thetas = np.linspace(0, np.pi, 4000)
    rays = np.stack([np.sin(thetas), np.zeros_like(thetas), np.cos(thetas)], axis=1)
    _, valid = cam.project(rays)
    theta_max_numeric = np.degrees(thetas[valid].max())

    print(f"Double Sphere model  (xi={xi:.3f}, alpha={alpha:.3f})")
    print(f"  w2 (half-space coefficient) = {w2:.4f}")
    print(f"  max valid incidence angle   = {theta_max_analytic:.1f} deg   "
          f"(numeric check: {theta_max_numeric:.1f} deg)")
    print(f"  => the model accepts a field of view up to {2 * theta_max_analytic:.0f} deg "
          f"— well beyond a 180 deg hemisphere.")
    print("  A naive `z > 0` test would cap this at exactly 180 deg, silently dropping every")
    print("  ray past 90 deg — the classic fisheye bug this model avoids.")

    # 2. the undistortion trade ---------------------------------------------------
    img = cv2.imread(IMG)
    if img is None:
        raise SystemExit(f"missing {IMG}")
    print("\nUndistortion: field of view vs how much of the frame is filled")
    print("  balance   rectified hFOV   frame filled (non-black)")
    for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
        K_new = cam.compute_K_new(balance=b)
        rect, _ = cam.undistort_image(img, K_new)
        hfov = np.degrees(2 * np.arctan((W / 2) / K_new[0, 0]))
        filled = float((cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY) > 0).mean()) * 100
        print(f"   {b:4.2f}      {hfov:6.1f} deg        {filled:5.1f} %")
    print("\n  balance 0 keeps the widest view but leaves black border; balance 1 crops to a")
    print("  filled image but throws field of view away. Neither can hold the full >180 deg —")
    print("  a pinhole plane is infinite at 90 deg, so the periphery has nowhere to go.")


if __name__ == "__main__":
    main()
