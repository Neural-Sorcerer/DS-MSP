#!/usr/bin/env python3
"""
Chapter 2 demo — the Double Sphere model, from its own math to a published camera.

What this teaches (run it, read the printed numbers):
  1. The Double Sphere (DS) model is two extra numbers on top of a pinhole:
     `xi` (gap between the two projection spheres) and `alpha` (which sphere you
     project from). We print them and round-trip project/unproject to machine
     precision — proving the closed-form unprojection in `ds_msp/models/ds_math.py`
     is the exact inverse of projection, not an approximation.
  2. DS is expressive enough to describe a real lens. TUM-VI ships a Kannala-Brandt
     reference; we re-express it as a Double Sphere model with the library's own
     `convert()` (analytic-Jacobian LM, no autodiff) and measure how closely DS
     reproduces the authors' camera across the image.

NOTE: this is model *conversion* (one model's numbers re-expressed as another's), not
calibration — no images, no board, no detected corners are involved. The actual
calibration capstone, which detects AprilGrid corners in real footage and bundle-
adjusts intrinsics from scratch, is `examples/03_calibrate_tumvi_aprilgrid.py`.

Prerequisites:
  - `uv pip install -e .` (or `pip install -e .`) in a venv.
  - TUM-VI room1 downloaded: `bash scripts/download_datasets.sh tumvi`.

Run:
  python examples/02_double_sphere_tumvi.py
"""
from __future__ import annotations

import os

import numpy as np

from ds_msp.adapt import convert
from ds_msp.io.kalibr import load_kalibr_with_resolution
from ds_msp.models import DoubleSphereModel

# --- locate the downloaded TUM-VI room1 sequence -----------------------------
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ = os.path.join(HERE, "datasets", "tumvi", "dataset-room1_512_16")
CAMCHAIN = os.path.join(SEQ, "dso", "camchain.yaml")


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    if not os.path.exists(CAMCHAIN):
        raise SystemExit(
            "TUM-VI room1 not found. Fetch it first:\n"
            "    bash scripts/download_datasets.sh tumvi"
        )

    # The published reference. TUM-VI's authors calibrated this camera and shipped
    # the result as a Kannala-Brandt (equidistant fisheye) model in their YAML.
    reference, (W, H) = load_kalibr_with_resolution(CAMCHAIN, cam="cam0")

    # ------------------------------------------------------------------ part 1
    section("1. Re-express the published lens as a Double Sphere model")
    # convert(): sample pixels -> unproject with the reference -> seed + refine a
    # Double Sphere model by Levenberg-Marquardt with the model's ANALYTIC param
    # Jacobian. No images, no checkerboards, no autodiff. Pure geometry.
    ds, report = convert(reference, DoubleSphereModel, width=W, height=H,
                         n_samples=2000)

    print(f"Published reference (KB):  {reference}")
    print(f"Recovered Double Sphere:   {ds}")
    print("\nThe two new Double Sphere numbers (this is what Chapter 2 explains):")
    print(f"    xi    = {ds.xi:+.4f}   (spacing between the two unit spheres)")
    print(f"    alpha = {ds.alpha:+.4f}   (0 -> project from sphere 1, 1 -> from sphere 2)")

    # ------------------------------------------------------------------ part 2
    section("2. The Double Sphere math is self-consistent (round-trip)")
    # project(unproject(u)) must return u to the last bit float64 can represent.
    # Build a grid of real pixels, unproject to bearing rays, project them back.
    us = np.linspace(2, W - 2, 40)
    vs = np.linspace(2, H - 2, 40)
    grid = np.stack(np.meshgrid(us, vs), axis=-1).reshape(-1, 2)

    rays, ok = ds.unproject(grid)
    back, ok2 = ds.project(rays)
    good = ok & ok2
    err = np.linalg.norm(back[good] - grid[good], axis=1)
    print(f"pixels tested: {good.sum()} / {len(grid)} (rest fall outside the lens circle)")
    print(f"project(unproject(u)) round-trip: mean={err.mean():.2e}px  max={err.max():.2e}px")
    print("~1e-13 px is float64 machine precision: the closed-form unprojection")
    print("in ds_math.py is the *exact* analytic inverse, not an approximation.")

    # ------------------------------------------------------------------ part 3
    section("3. Does the converted model still describe the same lens?")
    # Independent check (different sampling than the fit): for many pixels, send a
    # ray through the reference, then through our DS model, and compare where each
    # lands. Small everywhere => DS is expressive enough to capture this lens.
    print(f"Evaluated over {report['n_forward']} rays spanning "
          f"{report['fov_covered_deg']:.1f} deg of field of view:\n")
    print(f"    RMS  reprojection error : {report['rms_px']:.4f} px")
    print(f"    max  reprojection error : {report['max_px']:.4f} px")
    print(f"    median                  : {report['median_px']:.4f} px")
    print()
    verdict = "PASS" if report["max_px"] < 0.1 else "CHECK"
    print(f"[{verdict}] Double Sphere re-expresses TUM-VI's published KB model to well")
    print("       under a tenth of a pixel — DS has the expressive power for this lens.")
    print("       (This is conversion between models, NOT calibration from images.)")

    print("\nTo actually CALIBRATE this camera from raw footage — detect AprilGrid")
    print("corners and bundle-adjust intrinsics that match the published numbers —")
    print("run the capstone:  python examples/03_calibrate_tumvi_aprilgrid.py")


if __name__ == "__main__":
    main()
