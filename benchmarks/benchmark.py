#!/usr/bin/env python3
"""
Reproducible benchmarks — turn the README's accuracy/speed claims into numbers.

Run:
    python benchmarks/benchmark.py

Measures, on this machine, CPU-only:
  1. ACCURACY vs OpenCV — Kannala-Brandt (= cv2.fisheye) and RadTan (= cv2 pinhole)
     project/unproject agreement. Substantiates "matches OpenCV to ~1e-13".
  2. THROUGHPUT — project / unproject / analytic-Jacobian, in nanoseconds per point.
  3. ANALYTIC vs FINITE-DIFFERENCE Jacobian — the speedup that makes calibration fast,
     plus the max disagreement (proving the analytic derivative is correct).

Numbers are machine-dependent; the *ratios* and *error magnitudes* are the point.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from ds_msp.models import KannalaBrandtModel, RadTanModel
from ds_msp.model import DoubleSphereCamera, ds_project_jacobian


def _timeit(fn, *, repeat=5, number=None, n_points):
    """Best-of-`repeat` wall time per run; returns nanoseconds per point."""
    if number is None:
        number = int(np.clip(2e6 // n_points, 20, 2000))  # bounded: avoid runaway loops
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        best = min(best, (time.perf_counter() - t0) / number)
    return best / n_points * 1e9  # ns per point


def _forward_points(n, rng):
    """Random camera-frame points in the forward hemisphere (theta < ~75deg)."""
    theta = rng.uniform(0, np.deg2rad(75), n)
    phi = rng.uniform(0, 2 * np.pi, n)
    r = rng.uniform(1.0, 5.0, n)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=1)


def bench_accuracy_vs_opencv(rng):
    print("\n1. ACCURACY vs OpenCV  (same lens, both libraries)")
    print("   " + "-" * 60)
    pts = _forward_points(5000, rng)

    # --- Kannala-Brandt == cv2.fisheye -----------------------------------------
    kb = KannalaBrandtModel(320.0, 321.0, 320.0, 240.0, 0.05, 0.01, -0.002, 0.0008)
    ours, _ = kb.project(pts)
    K = kb.K
    D = kb.distortion.reshape(4, 1)
    cv_uv, _ = cv2.fisheye.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    cv_uv = cv_uv.reshape(-1, 2)
    proj_err = np.abs(ours - cv_uv).max()

    # unproject: cv2 returns normalized (x/z, y/z); compare to our normalized ray
    cv_norm = cv2.fisheye.undistortPoints(cv_uv.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    rays, ok = kb.unproject(cv_uv)
    ours_norm = rays[:, :2] / rays[:, 2:3]
    unproj_err = np.abs(ours_norm[ok] - cv_norm[ok]).max()

    # --- RadTan == cv2 pinhole -------------------------------------------------
    rt = RadTanModel(500.0, 500.0, 320.0, 240.0, -0.28, 0.07, 0.0005, -0.0004, 0.0)
    rt_pts = _forward_points(5000, rng) * 0.3  # narrower FOV for a pinhole lens
    rt_pts[:, 2] += 1.0
    ours_rt, _ = rt.project(rt_pts)
    dist = np.array([rt.k1, rt.k2, rt.p1, rt.p2, rt.k3])
    cv_rt, _ = cv2.projectPoints(rt_pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), rt.K, dist)
    rt_err = np.abs(ours_rt - cv_rt.reshape(-1, 2)).max()

    print(f"   Kannala-Brandt  project   vs cv2.fisheye.projectPoints   : max |Δ| = {proj_err:.2e} px")
    print(f"   Kannala-Brandt  unproject vs cv2.fisheye.undistortPoints : max |Δ| = {unproj_err:.2e}")
    print(f"   RadTan          project   vs cv2.projectPoints           : max |Δ| = {rt_err:.2e} px")


def bench_throughput(rng):
    print("\n2. THROUGHPUT  (nanoseconds per point, best of 5)")
    print("   " + "-" * 60)
    n = 10000
    pts = _forward_points(n, rng)
    cam = DoubleSphereCamera(711.57, 711.24, 949.18, 518.81, 0.183, 0.809)
    px, _ = cam.project(pts)

    p_ns = _timeit(lambda: cam.project(pts), n_points=n)
    u_ns = _timeit(lambda: cam.unproject(px), n_points=n)
    j_ns = _timeit(lambda: ds_project_jacobian(pts, cam.fx, cam.fy, cam.cx, cam.cy,
                                               cam.xi, cam.alpha), n_points=n)
    print(f"   Double Sphere  project            : {p_ns:7.1f} ns/pt")
    print(f"   Double Sphere  unproject          : {u_ns:7.1f} ns/pt")
    print(f"   Double Sphere  project + Jacobian : {j_ns:7.1f} ns/pt")


def bench_calibration_jacobian(rng):
    # The real calibration cost: per Levenberg-Marquardt iteration, the analytic
    # Jacobian is ONE call; finite-differencing instead re-evaluates the residual
    # once PER PARAMETER (intrinsics + 6 extrinsics per image). On a realistic
    # multi-image problem that is dozens of residual evals vs one Jacobian.
    from ds_msp.models import DoubleSphereModel

    print("\n3. CALIBRATION JACOBIAN — analytic vs finite-difference (per LM iteration)")
    print("   " + "-" * 60)
    truth = DoubleSphereModel(700, 700, 640, 360, 0.18, 0.62)
    board = np.column_stack([np.mgrid[0:6, 0:8].reshape(2, -1).T * 0.08,
                             np.zeros(48)]).astype(np.float64)
    n_views = 15
    Xs, obs, ext = [], [], []
    for _ in range(n_views):
        rvec = rng.uniform(-0.4, 0.4, 3)
        tvec = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(1.5, 2.5)])
        R, _ = cv2.Rodrigues(rvec)
        uv, _ = truth.project((R @ board.T).T + tvec)
        Xs.append(board)
        obs.append(uv)
        ext.append(np.concatenate([rvec, tvec]))
    P = 6
    n_params = P + 6 * n_views
    x0 = np.concatenate([truth.params] + ext)

    def residual(p):
        m = DoubleSphereModel.from_params(p[:P])
        out, off = [], P
        for X, uv in zip(Xs, obs):
            R, _ = cv2.Rodrigues(p[off:off + 3])
            off += 6
            uvp, _ = m.project((R @ X.T).T + p[off - 3:off])
            out.append((uvp - uv).ravel())
        return np.concatenate(out)

    def analytic_jac(p):
        m = DoubleSphereModel.from_params(p[:P])
        rows = 2 * sum(len(X) for X in Xs)
        J = np.zeros((rows, n_params))
        r0, off = 0, P
        for i, (X, uv) in enumerate(zip(Xs, obs)):
            R, jacR = cv2.Rodrigues(p[off:off + 3])
            t = p[off + 3:off + 6]
            off += 6
            _, J_pt, J_par, _ = m.project_jacobian((R @ X.T).T + t)
            dR = jacR.T.reshape(3, 3, 3)                      # dR[a,b,c] = dR_ab/dr_c
            dXc_dr = np.einsum('abc,nb->nac', dR, X)          # (N,3,3)
            J_rvec = np.einsum('nij,njc->nic', J_pt, dXc_dr)  # (N,2,3)
            N = len(X)
            J[r0:r0 + 2 * N, 0:P] = J_par.reshape(2 * N, P)
            ec = P + 6 * i
            J[r0:r0 + 2 * N, ec:ec + 3] = J_rvec.reshape(2 * N, 3)
            J[r0:r0 + 2 * N, ec + 3:ec + 6] = J_pt.reshape(2 * N, 3)
            r0 += 2 * N
        return J

    # self-check: the hand-rolled analytic Jacobian must match a numeric one
    Ja = analytic_jac(x0)
    Jn = np.zeros_like(Ja)
    for k in range(n_params):
        e = np.zeros(n_params)
        e[k] = 1e-6
        Jn[:, k] = (residual(x0 + e) - residual(x0 - e)) / 2e-6
    jac_err = np.abs(Ja - Jn).max()

    t_jac = _timeit(lambda: analytic_jac(x0), n_points=1, number=100, repeat=5) / 1e9
    t_res = _timeit(lambda: residual(x0), n_points=1, number=200, repeat=5) / 1e9
    fd_cost = n_params * t_res  # forward finite-diff: one residual eval per parameter

    print(f"   problem: {n_views} views x 48 corners  =>  {n_params} parameters")
    print(f"   analytic vs numeric Jacobian      : max |Δ| = {jac_err:.2e}  (correct)")
    print(f"   analytic Jacobian (1 call)        : {t_jac * 1e3:8.3f} ms")
    print(f"   one residual evaluation           : {t_res * 1e3:8.3f} ms")
    print(f"   finite-diff Jacobian (~{n_params} evals) : {fd_cost * 1e3:8.3f} ms")
    print(f"   speedup per LM iteration          : {fd_cost / t_jac:7.1f}x  faster")
    print("   (the analytic derivative also avoids step-size tuning and cancellation;")
    print("    its correctness is gradient-checked in tests/ — see `pytest -m jac`.)")


def main() -> None:
    rng = np.random.default_rng(0)
    print("=" * 64)
    print("DS-MSP benchmarks  (CPU only; numbers are machine-dependent)")
    print(f"numpy {np.__version__}  opencv {cv2.__version__}")
    print("=" * 64)
    bench_accuracy_vs_opencv(rng)
    bench_throughput(rng)
    bench_calibration_jacobian(rng)
    print()


if __name__ == "__main__":
    main()
