# Roadmap

DS-MSP has two intertwined goals, and the roadmap serves both:

1. **A clean, correct, well-tested wide-FOV camera library** — Double Sphere plus a
   uniform multi-model layer (UCM, EUCM, Kannala-Brandt, RadTan, OCamCalib), model
   conversion, analytic Jacobians, and Kalibr I/O.
2. **A learning resource** — a [guided, runnable curriculum](learn/README.md) that teaches
   the geometry behind SLAM / AR / robot perception on real public data.

The principle that keeps these from fighting each other: **the library stays terse and
production-grade; the teaching lives in `docs/learn/` and `examples/`.** New features land
with both a clean implementation *and* a chapter that explains them.

## Now (foundation — done)
- Double Sphere model with the correct >180° projection-validity condition.
- Multi-model library + parameter conversion between models.
- Generic Levenberg–Marquardt calibration with analytic Jacobians.
- Kalibr camchain I/O; OpenCV-compatible wrappers; TI LDC hardware export.
- Reproducible setup: `pip install -e .`, 166 passing tests, dataset fetcher.
- Learning track: **Chapters 1–2** — camera models on real TUM-VI data, and the
  Double Sphere model reproducing TUM-VI's published calibration to 0.025 px.
- **Capstone**: calibrate a real fisheye end-to-end from AprilGrid footage
  (`detect → correspond → bundle-adjust`) and match the published intrinsics to
  ~1% / 0.18 px RMS. AprilGrid detection adds `ds_msp/calib/{targets,detect}.py`
  and the `[calib]` extra.

## Next (learning curriculum)
Build out [`docs/learn/`](learn/README.md) in order, each chapter anchored to existing
code and a runnable real-data script:
- **Ch.2** Double Sphere from first principles → `ds_msp/models/ds_math.py` ✅
- **Ch.3** Projection validity & the >180° cone (why `z>0` is the classic bug)
- **Ch.4** Analytic Jacobians vs autodiff — derive, then gradient-check
- **Ch.5** Calibration by Levenberg–Marquardt from corner detections — the theory behind
  the **[capstone](learn/capstone_calibrating_a_real_camera.md)** (already runnable)
- **Ch.6** Model conversion without re-shooting images
- **Ch.7** Reproducing a *published* calibration — the capstone already does this for
  TUM-VI; chapter writes up the method and extends to EuRoC

## Later (capability — geometry → systems)
Extensions that turn "camera models" into "perception systems", each laptop-runnable on
the public datasets already wired up:
- **Multi-camera & camera–IMU calibration**, validated against the datasets' published
  extrinsics (`T_cn_cnm1`, `T_cam_imu`).
- **Monocular visual odometry** on EuRoC / TUM-VI, reusing the bundle adjuster; reported
  with standard ATE/RPE against ground truth.
- **A C++ core** (pybind11) for the hot kernels, plus one Ceres/Eigen calibration.
- **Inference-only modern-3D demos**: learned features (SuperPoint/LightGlue) in the VO
  loop; monocular metric depth metrically corrected by the fisheye calibration.

## Design rules (so the two goals stay aligned)
- Every new capability ships with a test that **proves a number**, not a screenshot.
- Teaching verbosity goes in `docs/`/`examples/`, never in `ds_msp/`.
- Demos run on **free, small (<10 GB) public data** on a laptop — no special hardware.
- Prefer **validating against a published reference** over self-asserted correctness.
