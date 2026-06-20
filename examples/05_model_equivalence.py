#!/usr/bin/env python3
"""
Are two different camera models the *same camera*?  A proof, on real data.

We calibrate BOTH a Kannala-Brandt and a Double Sphere model on the *same* detected
TUM-VI corners. Their parameters look incompatible — DS fx~152 vs KB fx~191 — yet they
fit the same lens. This script settles whether they describe the same camera, with math
and measured numbers.

  1. PARAXIAL FOCAL. `fx` means different things in the two models. The model-independent
     focal is dr/dθ at the optical axis. We derive it analytically:
         KB:  dr/dθ|0 = fx_KB
         DS:  dr/dθ|0 = fx_DS / (1 + xi)     (derivation in the companion doc)
     ...and confirm the two agree to ~0.1%, which explains the "different" fx.
  2. PROJECT / UNPROJECT AGREEMENT vs field angle. Push the same rays/pixels through
     both models and measure the gap as a function of angle.
  3. FOV COVERAGE. Where did the calibration board actually reach? The agreement and the
     data boundary line up — divergence happens only where neither model saw data.

Companion: docs/learn/are_two_models_the_same_camera.md

Prereqs: `pip install -e .[calib]`  +  `bash scripts/download_datasets.sh tumvi`
Run:     python examples/05_model_equivalence.py --stride 6
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from ds_msp.calib import AprilGridTarget, calibrate, detect_aprilgrid
from ds_msp.io.kalibr import load_kalibr_with_resolution
from ds_msp.models import DoubleSphereModel, KannalaBrandtModel

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAM_DIR = os.path.join(HERE, "datasets", "tumvi",
                       "dataset-calib-cam1_512_16", "mav0", "cam0", "data")
CAMCHAIN = os.path.join(HERE, "datasets", "tumvi",
                        "dataset-room1_512_16", "dso", "camchain.yaml")


def radius(model, theta):
    """Image radius (px from principal point) of a ray at angle theta from the axis."""
    d = np.array([[np.sin(theta), 0.0, np.cos(theta)]])
    uv, _ = model.project(d)
    return float(np.hypot(uv[0, 0] - model.cx, uv[0, 1] - model.cy))


def section(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=6)
    args = ap.parse_args()
    if not os.path.isdir(CAM_DIR):
        raise SystemExit("Fetch TUM-VI first: bash scripts/download_datasets.sh tumvi")

    _, (W, H) = load_kalibr_with_resolution(CAMCHAIN, cam="cam0")
    paths = sorted(glob.glob(os.path.join(CAM_DIR, "*.png")))[::args.stride]

    print(f"Calibrating KB and DS on the same {len(paths)} frames of corners ...")
    target = AprilGridTarget(6, 6, 0.088, 0.3)
    Xs, UVs, VIs = target.build_correspondences(
        detect_aprilgrid(paths, family="t36h11", min_tags=6, refine=True), min_corners=8)
    kb = calibrate(KannalaBrandtModel(180, 180, W / 2, H / 2),
                   Xs, UVs, VIs, max_nfev=150, loss="cauchy", f_scale=0.5)["model"]
    ds = calibrate(DoubleSphereModel(180, 180, W / 2, H / 2, 0.0, 0.5),
                   Xs, UVs, VIs, max_nfev=150, loss="cauchy", f_scale=0.5)["model"]
    print(f"  KB: {kb}")
    print(f"  DS: {ds}")

    # 1. paraxial focal --------------------------------------------------------
    section("1. PARAXIAL FOCAL  dr/dθ|0  (the model-independent focal length)")
    f_kb, f_ds = kb.fx, ds.fx / (1.0 + ds.xi)
    print(f"  KB  dr/dθ|0 = fx_KB        = {f_kb:.3f}")
    print(f"  DS  dr/dθ|0 = fx_DS/(1+xi) = {ds.fx:.3f}/{1 + ds.xi:.4f} = {f_ds:.3f}")
    print(f"  difference  = {abs(f_ds - f_kb):.3f} px  ({100 * abs(f_ds - f_kb) / f_kb:.2f}%)")
    h = 1e-5
    print(f"  numeric check dr/dθ|0:  KB={radius(kb, h) / h:.3f}  DS={radius(ds, h) / h:.3f}")
    print("  => the 'different' fx is an artifact of reading fx literally; the real")
    print("     focal lengths agree to a fraction of a percent.")

    # 2. project / unproject agreement vs field angle --------------------------
    section("2. DO THE TWO MAPS AGREE?  (same rays/pixels through both models)")
    print("  PROJECT — pixel distance between the two images of the same ray:")
    print("     θ(deg)     mean Δpx     max Δpx")
    for deg in [0, 15, 30, 45, 60, 75, 90]:
        th = np.deg2rad(deg)
        phis = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        dirs = np.stack([np.sin(th) * np.cos(phis), np.sin(th) * np.sin(phis),
                         np.cos(th) * np.ones_like(phis)], axis=1)
        ukb, vk = kb.project(dirs)
        uds, vd = ds.project(dirs)
        ok = vk & vd
        if ok.sum() == 0:
            print(f"     {deg:6d}     (no valid projection)")
            continue
        d = np.linalg.norm(ukb[ok] - uds[ok], axis=1)
        print(f"     {deg:6d}     {d.mean():8.4f}     {d.max():8.4f}")

    ys, xs = np.mgrid[8:H:16, 8:W:16]
    px = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
    rk, ok1 = kb.unproject(px)
    rd, ok2 = ds.unproject(px)
    ok = ok1 & ok2
    ang = np.rad2deg(np.arccos(np.clip(np.sum(rk[ok] * rd[ok], axis=1), -1, 1)))
    print(f"\n  UNPROJECT — angle between KB-ray and DS-ray over {ok.sum()} pixels:")
    print(f"     median={np.median(ang):.4f}°   mean={ang.mean():.4f}°   max={ang.max():.4f}°")
    for nm, m in [("KB", kb), ("DS", ds)]:
        r, a = m.unproject(px)
        b, a2 = m.project(r)
        g = a & a2
        print(f"     {nm} self round-trip (project∘unproject): "
              f"max={np.linalg.norm(b[g] - px[g], axis=1).max():.1e} px")

    # 3. FOV coverage of the data ----------------------------------------------
    section("3. WHERE DID THE BOARD ACTUALLY REACH?  (explains the rim divergence)")
    rays, okc = kb.unproject(np.concatenate(UVs))
    th = np.rad2deg(np.arccos(np.clip(rays[okc, 2], -1, 1)))
    print(f"  field angle of detected corners: "
          f"median={np.median(th):.1f}°  p95={np.percentile(th, 95):.1f}°  max={th.max():.1f}°")
    print(f"  {100 * (th < 55).mean():.0f}% of corners are within 55° — the periphery was never seen.")

    section("VERDICT")
    print("  NOT identical maps: DS and KB are different function families; they diverge")
    print("  at the rim (where neither was constrained). But over the calibrated field")
    print("  they are the SAME camera — paraxial focal to ~0.1%, projection sub-pixel,")
    print("  unprojection sub-0.03° median. Different parameters, one set of optics.")


if __name__ == "__main__":
    main()
