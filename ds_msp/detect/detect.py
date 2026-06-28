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

**Detecting tags at the fisheye periphery (the hard part).** A single ``detect()``
pass on the raw image misses tags that sit away from the centre: a fisheye
*shrinks* peripheral tags, and the detector's minimum-cluster-size gate drops them
before decoding (on TUM-VI's calib set, a fully-visible corner board can fall to
~4/36 tags). Production calibrators (Basalt's AprilTag-3 backend, TartanCalib) beat
this two ways, both implemented here:

  1. **Multi-scale detection** (``scales``): also run the detector on up-scaled
     copies so shrunken peripheral tags clear the size gate, then map corners back
     and keep the union. This alone recovers most missed tags (≈36 %→94 % on the
     TUM-VI calib set).
  2. **Board-guided recovery** (``recover=True`` + a ``target``): for each tag the
     first pass missed, fit a *local* homography from nearby detected tags, predict
     where the tag must be, and re-detect inside that small up-scaled ROI — the
     AprilGrid analogue of OpenCV ``refineDetectedMarkers`` / TartanCalib's
     ``cornerpredictor``. Only tags whose id re-detects are accepted, so it adds no
     false corners.

Imports: numpy + OpenCV always; ``aprilgrid`` lazily (only when you detect).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

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


def _subpix(img: np.ndarray, corners: np.ndarray, max_win: int) -> np.ndarray:
    """Subpixel-refine 4 corners, with a window scaled to the tag's size in ``img``.

    A fixed window that suits big central tags spans several corners on a small tag
    and drags it onto the wrong gradient, so clamp it to ~¼ the shortest edge.
    """
    c = np.ascontiguousarray(corners.reshape(4, 1, 2).astype(np.float32))
    side = float(np.linalg.norm(np.diff(c[[0, 1, 2, 3, 0], 0], axis=0), axis=1).min())
    win = int(max(2, min(max_win, round(side / 4.0))))
    cv2.cornerSubPix(img, c, (win, win), (-1, -1), _SUBPIX_CRITERIA)
    return c.reshape(4, 2)


def _detect_union(detector, gray: np.ndarray, scales: Sequence[float],
                  *, refine: bool = False, subpix_window: int = 5
                  ) -> Dict[int, np.ndarray]:
    """Detect at each scale, map corners back to native pixels, union by tag id.

    The first scale that finds a tag wins. Each tag is subpixel-refined **at the
    scale it was found** (where an up-scaled peripheral tag is large enough for a
    real window) and only then mapped back to native — refining the tiny tag at
    native resolution instead lets ``cornerSubPix`` wander onto neighbouring corners
    and wrecks exactly the wide-FOV corners multi-scale just gained.
    """
    found: Dict[int, np.ndarray] = {}
    for s in scales:
        g = (gray if s == 1 else
             cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC))
        for d in detector.detect(g):
            tid = int(d.tag_id)
            if tid in found:
                continue
            c = np.asarray(d.corners, np.float32).reshape(4, 2)
            if refine:
                c = _subpix(g, c, subpix_window)
            # Map back to native with OpenCV's pixel-centre convention. Plain `c/s`
            # over-shifts every corner by 0.5*(1-1/s) — a constant bias that lands
            # straight on the estimated principal point.
            found[tid] = (c + 0.5) / float(s) - 0.5
    return found


def _recover_missing(detector, gray: np.ndarray, found: Dict[int, np.ndarray],
                     target, *, neighbors: int, roi_pad: float, roi_target_px: float,
                     refine: bool = False, subpix_window: int = 5,
                     max_passes: int = 2) -> Dict[int, np.ndarray]:
    """Board-guided recovery of tags the multi-scale pass missed.

    For each missing tag: fit a homography from the ``neighbors`` nearest *detected*
    tags (local → distortion-tolerant, like ChArUco), predict the tag's corners,
    crop+upscale that ROI, re-detect, and accept only if the predicted id appears.
    """
    H, W = gray.shape[:2]
    board = {t: target.object_points(t)[:, :2].astype(np.float32)
             for t in range(target.n_tags)}
    centers = {t: board[t].mean(0) for t in range(target.n_tags)}
    out = dict(found)

    for _ in range(max_passes):
        det_ids = list(out.keys())
        if len(det_ids) < 3:                      # too few seeds → bad predictions
            break
        added = 0
        for tid in range(target.n_tags):
            if tid in out:
                continue
            nn = sorted(det_ids, key=lambda d: float(np.hypot(*(centers[d] - centers[tid]))))
            nn = nn[:neighbors] if len(nn) >= 4 else det_ids
            src = np.concatenate([board[d] for d in nn])
            dst = np.concatenate([out[d] for d in nn]).astype(np.float32)
            Hmat, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            if Hmat is None:
                continue
            pred = cv2.perspectiveTransform(board[tid][None], Hmat).reshape(4, 2)
            tag_px = float(max(np.ptp(pred[:, 0]), np.ptp(pred[:, 1])))
            if not np.isfinite(tag_px) or tag_px <= 1.0:
                continue
            pad = roi_pad * tag_px
            x0 = int(max(0, pred[:, 0].min() - pad))
            x1 = int(min(W, pred[:, 0].max() + pad))
            y0 = int(max(0, pred[:, 1].min() - pad))
            y1 = int(min(H, pred[:, 1].max() + pad))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            s = max(1.0, roi_target_px / tag_px)
            roi = cv2.resize(gray[y0:y1, x0:x1], None, fx=s, fy=s,
                             interpolation=cv2.INTER_CUBIC)
            for d in detector.detect(roi):
                if int(d.tag_id) == tid:
                    c = np.asarray(d.corners, np.float32).reshape(4, 2)
                    if refine:
                        c = _subpix(roi, c, subpix_window)   # refine in the up-scaled ROI
                    c = (c + 0.5) / s - 0.5                   # undo resize (pixel-centre)
                    c[:, 0] += x0
                    c[:, 1] += y0
                    out[tid] = c
                    added += 1
                    break
        if added == 0:
            break
    return out


def detect_aprilgrid(
    image_paths: Sequence[str], *,
    family: str = "t36h11", min_tags: int = 6, refine: bool = True,
    subpix_window: int = 5, scales: Sequence[float] = (1, 2, 3),
    target: Optional[object] = None, recover: bool = False,
    recover_neighbors: int = 6, recover_roi_pad: float = 0.6,
    recover_roi_px: float = 160.0,
) -> List[Dict[int, np.ndarray]]:
    """Detect AprilGrid tags in a list of images, robust to the fisheye periphery.

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
        Apply ``cv2.cornerSubPix`` to each corner **at the scale it was detected**
        (window auto-sized to the tag), then map back. The raw detector localizes to
        ~pixel; subpixel refinement is what brings calibration RMS from ~0.6 px to
        ~0.2 px (Kalibr does the same). Refining up-scaled peripheral tags this way is
        what makes multi-scale corners calibration-grade.
    subpix_window : int
        Half-window (pixels) for ``cv2.cornerSubPix``.
    scales : sequence of float
        Image scales to detect at, unioned by tag id (default ``(1, 2, 3)``). Extra
        scales recover peripheral tags the fisheye shrinks below the detector's size
        gate. Pass ``(1,)`` for the old single-pass behaviour.
    target : AprilGridTarget, optional
        Board geometry; required when ``recover=True``.
    recover : bool
        Enable board-guided recovery of still-missing tags (needs ``target``).
        Predicts each missing tag from a local homography of nearby detected tags
        and re-detects in an up-scaled ROI; only id-verified tags are added.
    recover_neighbors, recover_roi_pad, recover_roi_px :
        Recovery knobs — nearest detected tags used for the local homography, ROI
        padding (× tag size), and the pixel size each ROI tag is up-scaled to.

    Returns
    -------
    list of dict
        One ``{tag_id: (4, 2) float64}`` per *kept* frame, corners in the detector's
        order (bottom-left, bottom-right, top-right, top-left) — matching
        ``AprilGridTarget.object_points``.
    """
    if recover and target is None:
        raise ValueError("recover=True needs a `target` (AprilGridTarget) for the board geometry.")
    detector = _make_detector(family)
    out: List[Dict[int, np.ndarray]] = []
    for path in image_paths:
        gray = _load_gray_u8(path)
        found = _detect_union(detector, gray, scales,
                              refine=refine, subpix_window=subpix_window)
        if recover and target is not None:
            found = _recover_missing(detector, gray, found, target,
                                     neighbors=recover_neighbors, roi_pad=recover_roi_pad,
                                     roi_target_px=recover_roi_px,
                                     refine=refine, subpix_window=subpix_window)
        if len(found) < min_tags:
            continue
        out.append({int(tid): c.astype(np.float64) for tid, c in found.items()})
    return out
