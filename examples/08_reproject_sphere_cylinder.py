#!/usr/bin/env python3
"""
Deep-dive demo — one fisheye, three central representations: sphere, cylinder, pinhole.

A *central* camera (all rays through one point) is fully described by a bijection between
ray directions (points on the unit sphere) and pixels. The flat pinhole image is just one
way to store that bijection — and a poor one past 90°, where `tan` blows up. This script
re-stores the SAME real fisheye three ways and proves the pixel-to-pixel maps that link
them are exact inverses.

  1. THREE REAL IMAGES. Resample the bundled fisheye into
       - an equirectangular (unit-sphere) panorama:  row ∝ elevation,        col ∝ azimuth
       - a cylindrical panorama:                      row ∝ tan(elevation),   col ∝ azimuth
       - a pinhole (gnomonic) rectification:          row ∝ tan, col ∝ tan    (both gnomonic)
     Every sample comes from `DoubleSphereCamera.project`, so the pictures are the library's
     own geometry, not a look-alike.

  2. THE NUMBER YOU CAN VERIFY. Round-trip a dense pixel grid
       sphere → ray → cylinder → ray → sphere      and    sphere → ray → pinhole → ray → sphere
     The max residual is ~1e-12 px — the maps are inverses, not approximations.

  3. WHERE THE CYLINDER BREAKS. Sphere row is linear in elevation, so it reaches the pole;
     cylinder row is tan(elevation), so the SAME image height holds far less elevation and
     the caps (straight up / down) sit at infinity. We print both limits.

Prerequisites:  `pip install -e .`  + the bundled `test_image.jpg` / `test_config.json`.
Run:            python examples/08_reproject_sphere_cylinder.py
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
OUT = os.path.join(HERE, "assets", "learn")

# --- panorama intrinsics, shared by the sphere and the cylinder --------------------
# One focal (pixels per radian) and one centre for BOTH panoramas, so the only thing
# that differs between them is the VERTICAL law (elevation vs tan-elevation). That shared
# axis is the whole point: azimuth is stored identically, the cylinder just warps height.
W_PANO, H_PANO = 1200, 700
F_PANO = 320.0                       # px / rad
CX_PANO, CY_PANO = W_PANO / 2.0, H_PANO / 2.0


# ===================================================================================
# The pixel <-> ray maps. These are the math the doc describes, in code.
# Ray convention matches the library: x right, y down, z forward; project([x,y,z]) -> (u,v).
# A ray's spherical angles:  azimuth  lambda = atan2(x, z),  elevation  psi = atan2(-y, hypot(x,z)).
# ===================================================================================
def sphere_pix_to_ray(u, v):
    """Equirectangular: column is linear in azimuth, row is linear in elevation."""
    lam = (u - CX_PANO) / F_PANO
    psi = (CY_PANO - v) / F_PANO
    cps = np.cos(psi)
    return np.stack([cps * np.sin(lam), -np.sin(psi), cps * np.cos(lam)], axis=-1)


def cylinder_pix_to_ray(u, v):
    """Cylinder: column is linear in azimuth (SAME as sphere); row is linear in tan(elevation)."""
    lam = (u - CX_PANO) / F_PANO
    h = (CY_PANO - v) / F_PANO                 # h == tan(elevation)
    return np.stack([np.sin(lam), -h, np.cos(lam)], axis=-1)


def ray_to_sphere_pix(d):
    """Inverse of sphere_pix_to_ray: ray -> equirectangular pixel."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    lam = np.arctan2(x, z)
    psi = np.arctan2(-y, np.hypot(x, z))
    return np.stack([CX_PANO + F_PANO * lam, CY_PANO - F_PANO * psi], axis=-1)


def ray_to_cylinder_pix(d):
    """Inverse of cylinder_pix_to_ray: ray -> cylindrical pixel (height = tan(elevation))."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    lam = np.arctan2(x, z)
    h = -y / np.hypot(x, z)                    # = tan(elevation)
    return np.stack([CX_PANO + F_PANO * lam, CY_PANO - F_PANO * h], axis=-1)


def reproject(cam, img, pix_to_ray, w, h):
    """Resample `img` (a fisheye) into a w*h view whose pixels follow `pix_to_ray`."""
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    rays = pix_to_ray(u, v)
    pts, valid = cam.project(rays.reshape(-1, 3))
    mapx = pts[:, 0].reshape(h, w).astype(np.float32)
    mapy = pts[:, 1].reshape(h, w).astype(np.float32)
    valid = valid.reshape(h, w)
    mapx[~valid] = -1
    mapy[~valid] = -1
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def main() -> None:
    intr = json.load(open(CFG))
    if "intrinsics" in intr:
        intr = intr["intrinsics"]
    W, H = intr["width"], intr["height"]
    cam = DoubleSphereCamera(intr["fx"], intr["fy"], intr["cx"], intr["cy"],
                             intr["xi"], intr["alpha"], width=W, height=H)
    img = cv2.imread(IMG)
    if img is None:
        raise SystemExit(f"missing {IMG}")
    os.makedirs(OUT, exist_ok=True)

    print("Re-storing ONE fisheye three ways (sphere / cylinder / pinhole)")
    print(f"  source fisheye   : {W}x{H}  DS(xi={cam.xi:.3f}, alpha={cam.alpha:.3f})")
    print(f"  panorama model   : f={F_PANO:.0f} px/rad, {W_PANO}x{H_PANO}, "
          f"shared by sphere & cylinder\n")

    # 1. three real images --------------------------------------------------------
    sphere = reproject(cam, img, sphere_pix_to_ray, W_PANO, H_PANO)
    cylinder = reproject(cam, img, cylinder_pix_to_ray, W_PANO, H_PANO)
    pinhole, K_new = cam.undistort_image(img, cam.compute_K_new(balance=0.0))
    cv2.imwrite(os.path.join(OUT, "reproj_sphere.png"), sphere)
    cv2.imwrite(os.path.join(OUT, "reproj_cylinder.png"), cylinder)
    cv2.imwrite(os.path.join(OUT, "reproj_pinhole.png"), pinhole)
    print("  wrote assets/learn/reproj_{sphere,cylinder,pinhole}.png")

    # 2. the number you can verify ------------------------------------------------
    # Round-trip a dense interior grid (avoid the very edge where a map can leave the frame).
    uu, vv = np.meshgrid(np.linspace(120, W_PANO - 120, 200),
                         np.linspace(120, H_PANO - 120, 120))
    grid = np.stack([uu, vv], axis=-1)

    # sphere -> ray -> cylinder pixel -> ray -> sphere pixel
    r1 = sphere_pix_to_ray(grid[..., 0], grid[..., 1])
    cyl_px = ray_to_cylinder_pix(r1)
    r2 = cylinder_pix_to_ray(cyl_px[..., 0], cyl_px[..., 1])
    back = ray_to_sphere_pix(r2)
    err_cyl = np.abs(back - grid).max()

    # sphere -> ray -> pinhole pixel -> ray -> sphere pixel (pinhole only valid for z>0)
    fp, cxp, cyp = K_new[0, 0], K_new[0, 2], K_new[1, 2]
    front = r1[..., 2] > 1e-6
    rp = r1[front]
    up = cxp + fp * (rp[..., 0] / rp[..., 2])
    vp = cyp + fp * (rp[..., 1] / rp[..., 2])
    xn = (up - cxp) / fp
    yn = (vp - cyp) / fp
    r_back = np.stack([xn, yn, np.ones_like(xn)], axis=-1)
    back_p = ray_to_sphere_pix(r_back)
    err_pin = np.abs(back_p - grid[front]).max()

    print("\nCross-maps are exact inverses (max round-trip residual over a 200x120 grid):")
    print(f"  sphere -> cylinder -> sphere : {err_cyl:.2e} px")
    print(f"  sphere -> pinhole  -> sphere : {err_pin:.2e} px   (front hemisphere only)")

    # 3. where the cylinder breaks ------------------------------------------------
    psi_sphere = (CY_PANO - 0) / F_PANO                     # top row -> elevation
    psi_cyl = np.arctan((CY_PANO - 0) / F_PANO)             # top row -> atan(tan-height)
    print("\nSame image height, different elevation reach (top row of the panorama):")
    print(f"  sphere row 0   reaches elevation {np.degrees(psi_sphere):5.1f} deg")
    print(f"  cylinder row 0 reaches elevation {np.degrees(psi_cyl):5.1f} deg "
          f"(tan compresses it; the poles at +-90 deg sit at infinity)")
    print("\n  => sphere is the complete central model; the cylinder is a horizontal-band")
    print("     convenience that silently drops the polar cone. Same azimuth, warped height.")


if __name__ == "__main__":
    main()
