# 🏆 Capstone — calibrate a real fisheye camera, and match the published numbers

> **Run alongside this:** `pip install -e .[calib]` then
> `python examples/03_calibrate_tumvi_aprilgrid.py` (after the [setup](README.md#setup-once)).

Everything else in this curriculum is practice for one claim: **give me raw footage of a
camera looking at a calibration board, and I will recover its lens parameters from scratch
— and they will agree with the numbers the experts published.** No `convert()`, no loading
someone's answer. Detect the corners, bundle-adjust the geometry, compare to truth.

We use TUM-VI's `cam0` calibration sequence — 436 frames of someone waving a 6×6 AprilGrid
in front of a 195° fisheye — and check our result against the Kannala-Brandt calibration
the dataset authors published.

## The pipeline (all library code)

```
raw frames ─▶ detect AprilGrid ─▶ 3D↔2D correspondences ─▶ bundle adjust ─▶ compare
            detect_aprilgrid()    AprilGridTarget          calibrate()      vs published
```

1. **Detect** the board corners — `ds_msp.calib.detect_aprilgrid` (the one piece that needs
   a tag backend; see the war-story below).
2. **Correspond** — `ds_msp.calib.AprilGridTarget` knows where every tag corner sits in 3D
   board metres, so a detected `tag_id` becomes a `(X,Y,Z) ↔ (u,v)` pair.
3. **Calibrate** — `ds_msp.calib.calibrate` jointly refines the 6/8 intrinsics and a pose
   per frame by Levenberg-Marquardt, using each model's **analytic** projection Jacobian
   (Chapter 4's payoff: no finite differences, no autodiff).
4. **Compare** — line our `fx, fy, cx, cy, k1..k4` up against the published reference.

We calibrate a **Kannala-Brandt** model precisely because TUM-VI's published reference is
KB — so the comparison is number-for-number, not hand-waving.

## The result

```
            fx        fy        cx        cy        k1        k2        k3        k4
published  190.978   190.973   254.932   256.897   0.00348   0.00072  -0.00205   0.00020
mine       192.271   192.242   254.934   256.752   0.00953  -0.01642   0.01186  -0.00308
|Δ|          1.292     1.269     0.002     0.146

median reprojection 0.115 px, inlier RMS 0.247 px — over all 5180 corners we
detected ourselves, none discarded.
```

- **Principal point to ~0.1 px.** `cx` lands within 0.002 px; `cy` within 0.15 px.
- **Focal length to ~0.7%.** From a single camera and a subset of frames, against a
  reference fit with the full Basalt/Kalibr pipeline (both cameras + IMU). The higher-order
  `k`'s differ more — they're weakly constrained and trade off against each other, which is
  why we judge the camera by reprojection error, not by staring at `k4`.
- **0.115 px median reprojection** is Kalibr-grade. That number is the proof the
  calibration is real. (We report median + inlier RMS rather than a single RMS: under a
  robust loss the plain all-corner RMS is inflated by the few outliers the loss correctly
  *ignored*, so it would understate the fit.)

The library's flagship **Double Sphere** model fits the very same corners just as tightly
(≈0.12 px median). Its focal lands elsewhere (≈152) — not a bug: `fx` is model-relative
(the true paraxial focal is `fx_DS/(1+ξ) ≈ 193`, matching KB to 0.1%), and on a *planar*
target DS additionally has a focal↔(`xi`,`alpha`) gauge freedom. A full proof that the DS
and KB calibrations are the *same camera* — and where they stop being — is in
**[are two models the same camera?](are_two_models_the_same_camera.md)**. Judge a model by
reprojection error, not by its raw focal.

## War-story: why the detector returned **zero** tags (and the real fix)

This is the part worth internalizing, because it's the kind of bug that wastes a day.

Our first attempts with OpenCV's `aruco` AprilTag detector **and** `pupil-apriltags` (the
standard AprilTag-3 binding) returned **0 tags on every frame** — even on frames where the
grid is large and sharp. The tags were obviously *there*. So what gives?

The detectors weren't broken: a *synthetic* `tag36h11` decoded fine. The difference is the
board. **Kalibr-style AprilGrids print each tag with a 2-cell-wide black border; stock
AprilTag-3 / aruco assume a 1-cell border.** The detector finds the tag's outer quad either
way, but then samples the 6×6 code bits at the wrong grid positions → every tag fails its
checksum → silently dropped. Zero detections, no error message.

The fix is a detector that matches the board: the pure-Python
[`aprilgrid`](https://github.com/powei-lin/aprilgrid) package, which *defaults to Kalibr's
2-cell border*. One line changes, and detection jumps from 0 to ~15–33 tags per frame.

The lesson isn't "use library X." It's: **when a detector returns nothing on data you can
see is valid, suspect a layout/convention mismatch, not your eyes** — and confirm it the way
we did, by checking that the *same* detector succeeds on a synthetic tag and fails on a
2-border one. (`blackTagBorder = 2` is the default in Kalibr's own
`GridCalibrationTargetAprilgrid` — this is exactly how the official benchmark detected the
same board.)

Two more details turn a mediocre fit into a Kalibr-grade one, both things Kalibr also does:

- **Subpixel refinement.** The raw detector localizes corners to ~1 px; `cv2.cornerSubPix`
  sharpens them to the underlying intensity edge (median reprojection ~0.6 px → ~0.12 px).
- **A robust loss, not a hard cut.** A few peripheral corners are mis-localized (curved-lens
  tags where `cornerSubPix` grabs the wrong edge) and would drag a plain L2 fit. We don't
  *drop* them — we calibrate with a **Cauchy loss** that keeps every corner but down-weights
  large residuals continuously (`calibrate(..., loss="cauchy", f_scale=0.5)`). It beats hard
  rejection on focal accuracy *and* discards no data. This has its own
  **[learning-by-doing page](robust_losses_and_evaluation.md)** with the IRLS math and a
  runnable hard-reject-vs-robust comparison (`examples/04`) — including why you must score a
  robust fit by **median / inlier RMS**, never by RMS over all corners.

## How it's built (the codebase's layering, applied)

The capstone adds two small modules that mirror the rest of the library — terse core,
heavy dependencies isolated at the edge:

| Module | Depends on | Role |
|---|---|---|
| [`ds_msp/calib/targets.py`](../../ds_msp/calib/targets.py) | numpy only | `AprilGridTarget`: board geometry + correspondence assembly. Pure, unit-tested without any image. |
| [`ds_msp/calib/detect.py`](../../ds_msp/calib/detect.py) | OpenCV + `aprilgrid` (optional) | the *only* place that touches a tag backend; lazily imported so `import ds_msp` never needs it. |
| [`ds_msp/calib/bundle.py`](../../ds_msp/calib/bundle.py) | scipy + the model contract | the model-agnostic LM optimizer; calibrates *any* `CameraModel`, with `loss=`/`f_scale=` for robust kernels. |

`detect_aprilgrid` is exposed through a lazy `__getattr__`, and `aprilgrid` lives in the
`[calib]` optional extra. Install the core lean; opt into the detector only when you
calibrate from real images. That's the "layer it, don't blend it" rule the whole repo
follows.

## Try it yourself
1. Run with `--stride 2` (more frames). Does `fx` tighten toward 191? More wide-angle
   coverage constrains focal length better.
2. Calibrate `cam1` instead of `cam0` and compare to that camera's published row in the
   camchain — the two stereo lenses differ slightly.
3. Turn off subpixel refinement (`refine=False` in `detect_aprilgrid`) and watch the median
   climb. Then switch the Cauchy loss back to plain `loss="linear"` and watch `fx` drift away
   from 191. Each guard earns its place by a number — `examples/04` makes that A/B explicit.

**Back to the path:** the theory chapters ([Ch.3](03_projection_validity.md) validity,
[Ch.4](04_jacobians.md) Jacobians, [Ch.5](05_calibration.md) the LM math) explain *why*
each step here works. *(coming soon — see [`../ROADMAP.md`](../ROADMAP.md))*
