"""
Camera model conversion ("adapter").

Converts an already-calibrated source model to a target model **without images or
recalibration**: sample pixels -> unproject with the source -> linear seed the
target -> refine with Levenberg-Marquardt using each model's **analytic** param
Jacobian (no autodiff). Mirrors the fisheye-calib-adapter pipeline in pure Python.

Decoupled by dependency injection: ``convert`` takes the source instance and the
target *class*, so this module imports no concrete model — only the contract and
SciPy. Works with any model satisfying ``CameraModel``.
"""

from __future__ import annotations

from typing import Optional, Tuple, Type

import numpy as np
from scipy.optimize import least_squares

from ..core.contracts import CameraModel
from .evaluate import reprojection_report
from .sampling import sample_image_grid


def convert(source: CameraModel, target_cls: Type[CameraModel], *,
            width: int, height: int, n_samples: int = 500,
            max_fov_deg: Optional[float] = None,
            verbose: bool = False) -> Tuple[CameraModel, dict]:
    """Fit ``target_cls`` parameters to reproduce ``source`` over the image.

    Parameters
    ----------
    source : CameraModel
        The calibrated source model.
    target_cls : type[CameraModel]
        The model class to convert into.
    width, height : int
        Image size used for sampling and reporting.
    n_samples : int
        Approximate number of grid samples used for the fit.
    max_fov_deg : float, optional
        Restrict the fitted FOV (full angle). Useful when the target is narrower
        than the source (e.g. converting a >180 deg fisheye into a pinhole-like
        model) so the fit is not dragged by unrepresentable rays.

    Returns
    -------
    (target, report) : the fitted model and a quality report (see
    ``reprojection_report``).
    """
    # 1. sample pixels -> source bearing rays (forward hemisphere only)
    pixels = sample_image_grid(width, height, n_samples)
    rays, valid = source.unproject(pixels)
    keep = valid & (rays[:, 2] > 1e-6)
    if max_fov_deg is not None:
        ang = np.degrees(np.arccos(np.clip(rays[:, 2], -1.0, 1.0)))
        keep &= ang <= (max_fov_deg / 2.0)
    rays, pixels = rays[keep], pixels[keep]
    if len(rays) < len(target_cls.param_names):
        raise ValueError(
            f"Too few forward correspondences ({len(rays)}) to fit "
            f"{target_cls.name}; increase n_samples or max_fov_deg."
        )

    # 2. linear seed: inherit intrinsics from the source, SVD-seed distortion
    target = target_cls.from_params(np.zeros(len(target_cls.param_names)))
    target.initialize_from_correspondences(source.K, rays, pixels)

    # 3. nonlinear refine: minimize project_target(rays) - pixels, analytic Jac.
    lb, ub = target_cls.param_bounds()
    x0 = np.clip(target.params, lb, ub)

    def residual(p):
        uv, _ = target_cls.from_params(p).project(rays)
        return (uv - pixels).ravel()

    def jac(p):
        _, _, j_param, _ = target_cls.from_params(p).project_jacobian(rays)
        return j_param.reshape(-1, p.size)

    res = least_squares(residual, x0, jac=jac, bounds=(lb, ub),
                        method="trf", x_scale="jac",
                        verbose=2 if verbose else 0)
    target = target_cls.from_params(res.x)

    # 4. evaluate over the image
    report = reprojection_report(source, target, width, height,
                                 gt_params=None)
    report["converged"] = bool(res.success)
    report["source_model"] = source.name
    report["target_model"] = target_cls.name
    return target, report
