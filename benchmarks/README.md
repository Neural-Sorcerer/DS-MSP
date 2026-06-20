# Benchmarks

Reproducible, CPU-only benchmarks that turn the README's accuracy and speed claims into
numbers you can run yourself.

```bash
python benchmarks/benchmark.py
```

## What it measures

1. **Accuracy vs OpenCV** — for the models that have OpenCV equivalents, project/unproject the
   same points through both libraries and report the max disagreement:
   - Kannala-Brandt (`= cv2.fisheye`) project & unproject,
   - RadTan (`= cv2.projectPoints`) project.

   These substantiate the *"matches OpenCV to ~1e-13"* claim.

2. **Throughput** — nanoseconds per point for Double Sphere `project`, `unproject`, and
   `project + analytic Jacobian`.

3. **Calibration Jacobian: analytic vs finite-difference** — on a realistic 15-view
   bundle-adjustment problem (96 parameters), the cost of one analytic Jacobian call vs
   finite-differencing every parameter (what you pay per Levenberg–Marquardt iteration
   without analytic derivatives). The script also **self-checks** the analytic Jacobian
   against a numeric one (max |Δ| ≈ 1e-7), so the speedup is over a *correct* derivative.

## Example output

Numbers are **machine-dependent** (this run: Apple M-series, NumPy 2.4, OpenCV 4.13) — the
*error magnitudes* and *ratios* are the point, not the absolute timings.

```
1. ACCURACY vs OpenCV
   Kannala-Brandt  project   vs cv2.fisheye.projectPoints   : max |Δ| = 2.6e-13 px
   Kannala-Brandt  unproject vs cv2.fisheye.undistortPoints : max |Δ| = 5.8e-15
   RadTan          project   vs cv2.projectPoints           : max |Δ| = 1.3e-13 px

2. THROUGHPUT (ns per point)
   Double Sphere  project            :  ~8 ns/pt
   Double Sphere  unproject          : ~20 ns/pt
   Double Sphere  project + Jacobian : ~43 ns/pt

3. CALIBRATION JACOBIAN (15 views, 96 parameters)
   analytic vs numeric Jacobian : max |Δ| = 9e-08   (correct)
   analytic Jacobian (1 call)   :  ~0.6 ms
   finite-diff Jacobian (~96)   : ~16   ms
   speedup per LM iteration     : ~28x faster
```

The Jacobian speedup **grows with the number of images** (more views → more parameters →
more finite-difference residual evaluations per iteration), so larger calibrations benefit
even more. The derivative's correctness is independently gradient-checked in the test suite
(`pytest -m jac`).
