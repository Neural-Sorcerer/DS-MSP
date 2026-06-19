"""
Conversion quality evaluation (pure numpy + the CameraModel contract).

Reprojection Error (RE) = || project_target(unproject_source(u)) - u ||, plus
FOV coverage so lossy conversions (e.g. fisheye -> pinhole) are visible, never
silent.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .sampling import sample_image_grid


def reprojection_report(source, target, width: int, height: int,
                        n_samples: int = 2000,
                        max_fov_deg: Optional[float] = None,
                        gt_params: Optional[np.ndarray] = None) -> dict:
    """Measure how well ``target`` reproduces ``source`` over the image.

    Returns a dict with rms/max/median pixel error, sample counts, FOV coverage,
    and (if ``gt_params`` given) parameter error. ``max_fov_deg`` restricts the
    evaluated region to match a narrower target (e.g. pinhole/RadTan), so the
    report reflects the region the target is meant to cover rather than being
    dominated by unrepresentable peripheral rays.
    """
    pixels = sample_image_grid(width, height, n_samples)
    rays, valid = source.unproject(pixels)
    keep = valid & (rays[:, 2] > 1e-6)
    if max_fov_deg is not None:
        ang_all = np.degrees(np.arccos(np.clip(rays[:, 2], -1.0, 1.0)))
        keep &= ang_all <= (max_fov_deg / 2.0)
    pixels_k, rays_k = pixels[keep], rays[keep]

    uv, vt = target.project(rays_k)
    ok = vt
    err = np.linalg.norm(uv[ok] - pixels_k[ok], axis=1)

    ang = np.degrees(np.arccos(np.clip(rays_k[:, 2], -1.0, 1.0)))
    report = {
        "rms_px": float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan"),
        "max_px": float(err.max()) if err.size else float("nan"),
        "median_px": float(np.median(err)) if err.size else float("nan"),
        "n_sampled": int(len(pixels)),
        "n_forward": int(keep.sum()),
        "n_target_valid": int(ok.sum()),
        "fov_covered_deg": float(2.0 * ang.max()) if ang.size else float("nan"),
    }
    if gt_params is not None:
        report["param_error"] = float(np.linalg.norm(np.asarray(gt_params) - target.params))
    return report
