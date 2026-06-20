#!/usr/bin/env python3
"""
Generate the README demo GIF: a real fisheye frame next to its rectified pinhole view,
sweeping the `balance` knob (0 = widest FOV / most border, 1 = tightest crop).

    python scripts/make_demo_gif.py        # writes assets/undistort_demo.gif

Uses the bundled test image + calibration; needs only the library + Pillow.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
from PIL import Image

from ds_msp import DoubleSphereCamera

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(HERE, "assets", "test_image.jpg")
CFG = os.path.join(HERE, "test_config.json")
OUT = os.path.join(HERE, "assets", "undistort_demo.gif")

PANEL_W = 420            # width of each panel in the GIF (keeps the file small)


def _label(img, text):
    """Draw a caption bar at the top of a BGR image."""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    return out


def _fit(img, w):
    h = round(img.shape[0] * w / img.shape[1])
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def main() -> None:
    img = cv2.imread(IMG)
    if img is None:
        raise SystemExit(f"missing {IMG}")
    intr = json.load(open(CFG))["intrinsics"]
    H, W = img.shape[:2]
    cam = DoubleSphereCamera(intr["fx"], intr["fy"], intr["cx"], intr["cy"],
                             intr["xi"], intr["alpha"], width=W, height=H)

    raw = _label(_fit(img, PANEL_W), "raw fisheye")
    gap = np.full((raw.shape[0], 10, 3), 255, np.uint8)

    balances = list(np.linspace(0.0, 1.0, 11))
    balances = balances + balances[::-1][1:-1]   # ping-pong for a seamless loop

    frames = []
    for b in balances:
        K_new = cam.compute_K_new(balance=float(b))
        rect, _ = cam.undistort_image(img, K_new)
        right = _label(_fit(rect, PANEL_W), f"rectified  (balance = {b:.1f})")
        combo = np.hstack([raw, gap, right])
        frames.append(Image.fromarray(cv2.cvtColor(combo, cv2.COLOR_BGR2RGB)))

    frames[0].save(OUT, save_all=True, append_images=frames[1:], loop=0, duration=180,
                   optimize=True)
    print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.2f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
