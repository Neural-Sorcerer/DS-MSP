"""
AprilGrid detection adapter — pixels in, tag corners out.

The one place in the library that depends on an AprilTag backend. It wraps the
pure-Python ``aprilgrid`` detector (``pip install ds_msp[calib]``) and OpenCV for
image loading + subpixel corner refinement, and hands back per-image
``{tag_id: (4, 2)}`` dictionaries that ``AprilGridTarget.build_correspondences``
turns into calibration inputs. Isolating the heavy/optional dependency here keeps
the rest of ``ds_msp`` installable and importable without it.

**Why a dedicated AprilGrid backend and not OpenCV's aruco or apriltag3?**
Kalibr-style boards (TUM-VI, EuRoC, …) print each tag with a **2-cell black
border**; stock AprilTag-3 / aruco assume a **1-cell** border, locate the tag quad
but then sample the code bits at the wrong places and decode nothing. The
``aprilgrid`` package defaults to the 2-cell border, which is why it succeeds where
the others silently return zero detections.

Imports: numpy + OpenCV always; ``aprilgrid`` lazily (only when you detect).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import cv2
import numpy as np

_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)


def _load_gray_u8(path: str) -> np.ndarray:
    """Load an image as contiguous 8-bit grayscale (TUM-VI ships 16-bit PNGs)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = (img.astype(np.float64) / 256.0).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def _make_detector(family: str):
    try:
        from aprilgrid import Detector  # optional dependency
    except ImportError as exc:  # pragma: no cover - import-guard
        raise ImportError(
            "AprilGrid detection needs the 'aprilgrid' package. Install it with:\n"
            "    pip install ds_msp[calib]      (or: pip install aprilgrid)\n"
            "It defaults to Kalibr's 2-cell tag border, which AprilTag-3/aruco do not."
        ) from exc
    return Detector(family)


def detect_aprilgrid(
    image_paths: Sequence[str], *,
    family: str = "t36h11", min_tags: int = 6, refine: bool = True,
    subpix_window: int = 5,
) -> List[Dict[int, np.ndarray]]:
    """Detect AprilGrid tags in a list of images.

    Parameters
    ----------
    image_paths : sequence of str
        Paths to the calibration frames.
    family : str
        AprilTag family of the board (TUM-VI / Kalibr default is ``"t36h11"``).
    min_tags : int
        Skip frames where fewer than this many tags are found (a near-empty frame
        contributes little and risks outliers).
    refine : bool
        Apply ``cv2.cornerSubPix`` to each corner. The raw detector localizes to
        ~pixel; subpixel refinement is what brings calibration RMS from ~0.6 px to
        ~0.2 px (Kalibr does the same).

    Returns
    -------
    list of dict
        One ``{tag_id: (4, 2) float64}`` per *kept* frame, corners in the detector's
        order (bottom-left, bottom-right, top-right, top-left) — matching
        ``AprilGridTarget.object_points``.
    """
    detector = _make_detector(family)
    w = (subpix_window, subpix_window)
    out: List[Dict[int, np.ndarray]] = []
    for path in image_paths:
        gray = _load_gray_u8(path)
        dets = detector.detect(gray)
        if len(dets) < min_tags:
            continue
        frame: Dict[int, np.ndarray] = {}
        for d in dets:
            corners = np.ascontiguousarray(
                np.asarray(d.corners, dtype=np.float32).reshape(4, 1, 2))
            if refine:
                cv2.cornerSubPix(gray, corners, w, (-1, -1), _SUBPIX_CRITERIA)
            frame[int(d.tag_id)] = corners.reshape(4, 2).astype(np.float64)
        out.append(frame)
    return out
