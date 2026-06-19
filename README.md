# Double Sphere Camera Model (DS-MSP)

**Production-ready fisheye camera implementation for 3D vision tasks.**

This repository provides a robust, OpenCV-compatible wrapper for the **Double Sphere (DS) Camera Model**. It is designed to be easy to understand, test, and integrate, while offering deep technical insights into the model's capabilities and limitations.

> **It is also a small multi-model camera library.** Calibrate in Double Sphere and
> **convert the parameters to UCM, EUCM, Kannala-Brandt (OpenCV fisheye), RadTan
> (OpenCV pinhole), or OCamCalib** — then run every feature on any model. See
> **[§6.6](#66-multi-model-support--model-conversion)** and the full guide in
> [`docs/MULTI_MODEL.md`](docs/MULTI_MODEL.md). This capability is modeled on
> **[Fisheye-Calib-Adapter](https://github.com/eowjd0512/fisheye-calib-adapter)**
> (see [Credits](#10-credits)).

> 🎓 **New to wide-FOV camera geometry?** This repo doubles as a guided, runnable
> course. Start with **[`docs/learn/`](docs/learn/README.md)** — each chapter pairs a
> short explainer with a script that runs on real public data and prints a number you
> can verify. See the **[Roadmap](docs/ROADMAP.md)** for where it's heading.

---

## 📚 Table of Contents

1.  [Introduction](#1-introduction)
2.  [Installation](#2-installation)
3.  [Quick Start (Demo)](#3-quick-start-demo)
4.  [Tutorial 1: Calibration](#4-tutorial-1-calibration)
5.  [Tutorial 2: Validation](#5-tutorial-2-validation)
6.  [Tutorial 3: Core API Usage](#6-tutorial-3-core-api-usage)
7.  [Technical Deep Dive: FOV & Undistortion](#7-technical-deep-dive-fov--undistortion)
8.  [Geometric Accuracy Verification](#8-geometric-accuracy-verification)
9.  [FAQ](#9-faq)

---

## 1. Introduction

Fisheye lenses capture a very wide field of view (often > 180°), much like a human eye or a security camera. Standard "pinhole" camera models assume straight lines stay straight, which fails for these curved lenses.

The **Double Sphere** model is a mathematical way to accurately describe how these lenses bend light. This codebase provides:
-   **Core model**: `ds_msp/model.py` — `DoubleSphereCamera` + stateless `ds_project` / `ds_unproject` / `ds_project_jacobian`.
-   **OpenCV-style wrapper**: `ds_msp/cv.py` — drop-in functions mirroring `cv2.fisheye` (`projectPoints`, `undistortImage`, `solvePnP`, …).
-   **Hardware LDC export**: `ds_msp/ldc.py` — TI Jacinto J7/TDA4 displacement-mesh LUT generator.
-   **Calibration**: `calibrate.py` (Levenberg–Marquardt with **analytic Jacobians**).
-   **Validation**: `validate.py` (Visual verification and metrics).
-   **Verification**: `tests/` and `verify_real_samples.py` (unit + real-image end-to-end checks).

> **Correctness note.** The projection validity test is the Double Sphere
> half-space condition `z > -w2·d1` (Usenko et al. 2018, Eq. 43–45), so the model
> correctly handles the full **> 180° FOV** — not the naive `z > 0` check, which
> would silently clip the field of view below 180°. See [§7.2](#72-projection-validity-the-correct-condition).

---

## 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Munna-Manoj/DS-MSP.git
    cd DS-MSP
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy opencv-python scipy matplotlib
    ```

3.  **Verify installation:**
    ```bash
    python -c "import ds_msp; print('DS-MSP package loaded successfully')"
    ```

---

## 3. Quick Start (Demo)

Want to see it in action immediately? We provide a pre-configured test image and config file.

Run the validation script in "single config" mode:

```bash
python validate.py --config test_config.json
```

**What happens:**
1.  Loads test images (`test_image.jpg`, `test_image_96.jpg`) and intrinsics from `test_config.json`.
2.  Estimates the camera pose for each image using our robust **Double Sphere PnP** solver.
3.  Reprojects the 3D checkerboard points onto the images.
4.  Saves the results to `results/visualizations/validate_test_image.png` and `validate_test_image_96.png`.

**Expected Output:**
```text
Validating images from config: test_config.json

Processing: test_image.jpg
Pose Estimation Success.
RMS Reprojection Error: 0.4344 px

Processing: test_image_96.jpg
Pose Estimation Success.
RMS Reprojection Error: 0.8481 px
```

---

## 4. Tutorial 1: Calibration

This tutorial shows how to calibrate the camera using your own data.

**Input:** `anns.json` (COCO-style annotations of checkerboard corners)
**Output:** `results/calibration_params.json` (Intrinsics & Distortion)

### Step 1: Prepare Data
Ensure `anns.json` is in the root directory. It should contain:
-   `"images"`: List of image metadata (width, height, file_name).
-   `"annotations"`: Keypoints for the checkerboard corners.

### Step 2: Run Calibration
Run the calibration script:
```bash
python calibrate.py
```

**What happens inside:**
1.  Loads 2D keypoints and generates corresponding 3D world points for the checkerboard.
2.  Initializes intrinsics (focal length, principal point) and seeds each image's
    pose with a **PnP** solve on the unprojected rays ($\xi_0 = 0.5$, $\alpha_0 = 0.5$).
3.  Runs **Levenberg–Marquardt** (`scipy.least_squares`, TRF) with an **analytic
    Jacobian** (`ds_project_jacobian`) to minimize reprojection error. The exact
    Jacobian replaces finite differencing — on the bundled dataset this is ~**170×
    faster** while reaching the identical solution.
4.  Saves the optimized parameters to `results/`.

> **Parameter domain (important).** The optimizer constrains the distortion
> parameters to the *well-posed* Double Sphere range: $\alpha \in [0, 1]$ and
> $\xi \in [-1, 1]$ (matching the basalt/Kalibr reference). Outside this range the
> model becomes non-injective (projection "folds back", so unprojection can no
> longer invert it) — real fisheye lenses sit roughly in $\xi \in [-0.2, 0.6]$.

### Step 3: Check Results
Look at the output in your terminal:
```text
[REAL] Optimization success: True
[REAL] Final cost: 1362.23
[REAL] RMS reprojection error: 0.6367 px
```
An RMS error < 1.0 pixel is generally considered good. On the bundled data the
calibration converges to `fx≈711.6, fy≈711.2, cx≈949.2, cy≈518.8, xi≈0.183, alpha≈0.809`.

---

## 5. Tutorial 2: Validation

After calibration, you should validate the results visually on the full dataset.

**Input:** `results/calibration_params.json`, `results/poses.json`
**Output:** `results/visualizations/*.png`

### Step 1: Run Validation
```bash
python validate.py
```

### Step 2: Inspect Visualizations
Go to `results/visualizations/` and open the generated images (e.g., `reproj_000.png`).
-   **Blue Dots**: Original observed keypoints.
-   **Red Dots**: Reprojected points using the calibrated model.
-   **Green Lines**: Error vectors (should be very short).

---

## 6. Tutorial 3: Core API Usage

This section explains how to use `ds_msp/model.py` and `ds_msp/cv.py` in your own code.

### 6.1. Creating the Camera
The model needs **only the 6 intrinsics** for projection / unprojection / PnP.
`width` and `height` are optional — they are required *only* by the image-level
helpers (`compute_K_new`, `get_undistortion_maps`), which raise a clear error if
you call them without dimensions.

```python
from ds_msp import DoubleSphereCamera
import numpy as np

# (a) Math-only: no meaningless image dimensions required
cam = DoubleSphereCamera(fx=711.57, fy=711.24, cx=949.18, cy=518.81, xi=0.183, alpha=0.809)

# (b) With dimensions, for image undistortion
cam = DoubleSphereCamera(711.57, 711.24, 949.18, 518.81, 0.183, 0.809, width=1920, height=1080)

# (c) From a calibration result
cam = DoubleSphereCamera.from_json('results/calibration_params.json')

print(cam)                 # readable repr
K, D = cam.K, cam.D        # 3x3 intrinsic matrix and [xi, alpha] as properties
```

> `alpha` is validated to be in `[0, 1]` at construction. For the well-posed
> domain keep `xi` in `[-1, 1]` (see [§4](#4-tutorial-1-calibration)).

### 6.2. Project and Unproject
-   **Project (3D → 2D)**: Where does a 3D point appear on the image? (returns a
    `valid` mask using the correct half-space condition — see [§7.2](#72-projection-validity-the-correct-condition).)
-   **Unproject (2D → 3D)**: What unit ray corresponds to a pixel?

```python
points_3d = np.array([[0, 0, 1], [1, 1, 2]], dtype=np.float64)  # camera frame (N, 3)

points_2d, valid = cam.project(points_3d)     # (N, 2) pixels + (N,) validity
rays, valid = cam.unproject(points_2d)        # (N, 3) unit rays + (N,) validity
```

For hot loops (e.g. calibration) the same math is available as allocation-free
standalone functions, plus the **analytic Jacobian**:

```python
from ds_msp import ds_project, ds_unproject
from ds_msp.model import ds_project_jacobian

u, v, valid = ds_project(points_3d, cam.fx, cam.fy, cam.cx, cam.cy, cam.xi, cam.alpha)

# Exact derivatives: J_point = d(u,v)/d(x,y,z),  J_intr = d(u,v)/d(fx,fy,cx,cy,xi,alpha)
u, v, J_point, J_intr, valid = ds_project_jacobian(points_3d, cam.fx, cam.fy, cam.cx, cam.cy, cam.xi, cam.alpha)
```

### 6.3. Undistorting Images (OpenCV Style)
Use `ds_msp.cv` to undistort images, just like `cv2.fisheye`.

```python
import ds_msp.cv as ds_cv
import cv2

img = cv2.imread('assets/test_image.jpg')
K, D = cam.K, cam.D

# 1. Estimate new camera matrix (controls zoom/crop)
#    balance=0.0 -> widest FOV (0.4x focal, more scene, black borders)
#    balance=1.0 -> tightest crop (0.8x focal, less FOV, fewer borders)
K_new = ds_cv.estimateNewCameraMatrixForUndistortRectify(K, D, (1920, 1080), balance=0.0)

# 2. Undistort
img_undist = ds_cv.undistortImage(img, K, D, Knew=K_new)
cv2.imwrite('undistorted.jpg', img_undist)
```

The object API is equivalent: `img_undist, K_new = cam.undistort_image(img)`.

### 6.4. Robust PnP (Pose Estimation)
Standard PnP solvers assume a pinhole model and fail on raw fisheye points. The
DS solver unprojects to rays first, keeps the front-facing valid rays
(`z > 0`, the only ones a normalized-plane PnP can represent), then solves.

```python
# Object API (recommended): points_3d (N,3), points_2d (N,2) distorted pixels
success, rvec, tvec = cam.solve_pnp(points_3d, points_2d)

# OpenCV-style equivalent
success, rvec, tvec = ds_cv.solvePnP(points_3d, points_2d, cam.K, cam.D)
```

### 6.5. Hardware LDC Export (TI Jacinto J7 / TDA4)
Generate a downsampled displacement-mesh LUT for the on-chip Lens Distortion
Correction engine, plus the matching pinhole intrinsics:

```python
from ds_msp.ldc import TI_LDC_MeshGenerator

gen = TI_LDC_MeshGenerator(cam)                       # cam built with width/height
res = gen.generate_mesh_and_intrinsics(1920, 1080, downsample_factor=4, balance=0.5)
mesh_lut = res["mesh_lut"]     # int16, Q3 (1/8 px) displacements for the hardware
K_new    = res["K_new"]        # pinhole intrinsics of the rectified image
```

> **Best practice for keypoints on an LDC-undistorted image:** use the **LDC image**
> for the picture, but undistort *keypoints* with `cam.undistort_points(pts, K_new)`
> (closed-form) at the **same `balance`**. The mesh-based point inverse
> (`TI_LDC_PointUndistorter`) is exact at the center but diverges toward the
> periphery (its fixed-point iteration is not contractive there). Because the same
> `K_new` is shared, the analytic points and the LDC image stay consistent to ~0.1 px.

### 6.6. Multi-Model Support & Model Conversion

Beyond Double Sphere, DS-MSP ships a uniform multi-model library (UCM, EUCM,
Kannala-Brandt = OpenCV fisheye, RadTan = OpenCV pinhole, OCamCalib) behind one
`CameraModel` interface, and a **converter** to translate calibrated parameters
between models without images or recalibration — inspired by
[Fisheye-Calib-Adapter](https://github.com/eowjd0512/fisheye-calib-adapter).

```python
from ds_msp import DoubleSphereModel, KannalaBrandtModel, convert, Undistorter, solve_pnp
import cv2, json

ds = DoubleSphereModel.from_dict(json.load(open("results/calibration_params.json")))

# Convert DS -> OpenCV fisheye (KB), no images needed
kb, report = convert(ds, KannalaBrandtModel, width=1920, height=1080)
print(report["rms_px"])                                  # ~0.0002 px

# Every feature works on the converted model
solve_pnp(kb, object_points, image_points)
img_rect, K_new = Undistorter(kb, 1920, 1080).undistort_image(img)
cv2.fisheye.undistortImage(img, kb.K, kb.distortion, Knew=K_new)   # direct OpenCV interop
```

All models share the **same API** — `project` / `unproject` (2D↔3D), image
undistortion, point `undistort_points` / `distort_points`, and `solve_pnp` — so
swapping models is a one-line change. All use **analytic Jacobians** (no autodiff);
KB and RadTan match OpenCV to ~1e-13. You can also **calibrate any model**
(`ds_msp.calib.calibrate`) and read/write **Kalibr YAML** (`ds_msp.io`).

See **[`docs/MULTI_MODEL.md`](docs/MULTI_MODEL.md)** for the full guide: how each
model's geometry works, the **camera-geometry cookbook** (project/unproject,
undistort, point distort/undistort, PnP — identical on every model), the
conversion-accuracy table, and Kalibr field orderings.

---

## 7. Technical Deep Dive: FOV & Undistortion

**Generated by:** `visualize.py`

A common question is: *"Why are pixels missing from my undistorted image, even when I try to keep the whole image?"*

### 7.1. Visual Comparison: Undistortion Modes
We verified the wrapper on real data (`assets/test_image.jpg` and `assets/test_image_96.jpg`).

#### Sample 11 (`test_image.jpg`)
| Distorted | Undistorted (Crop) | Undistorted (Whole) | Undistorted (Zoom) |
| :---: | :---: | :---: | :---: |
| ![Distorted](assets/result_distorted_11.jpg) | ![Crop](assets/result_undistort_crop_11.jpg) | ![Whole](assets/result_undistort_whole_11.jpg) | ![Zoom](assets/result_undistort_zoom_11.jpg) |

#### Sample 96 (`test_image_96.jpg`)
| Distorted | Undistorted (Crop) | Undistorted (Whole) | Undistorted (Zoom) |
| :---: | :---: | :---: | :---: |
| ![Distorted](assets/result_distorted_96.jpg) | ![Crop](assets/result_undistort_crop_96.jpg) | ![Whole](assets/result_undistort_whole_96.jpg) | ![Zoom](assets/result_undistort_zoom_96.jpg) |

**Key Observations:**
1.  **Crop (`balance=1.0`)**: Keeps only the center valid pixels. No black borders, but loses FOV.
2.  **Whole (`balance=0.0`)**: Keeps all pixels that map to the image plane. Introduces black borders.
3.  **Zoom (Reduced Focal Length)**: Captures even more of the wide-angle content, but shrinks the center.

### 7.2. Projection Validity (The Correct Condition)
The Double Sphere model defines a projection function $\pi(\mathbf{x})$ that is **not**
valid for all 3D points. The exact projectability test (Usenko et al. 2018, Eq. 43–45) is the
half-space condition implemented in `ds_project`:

$$z > -w_2 \, d_1, \qquad d_1 = \sqrt{x^2 + y^2 + z^2}$$

$$w_1 = \begin{cases} \dfrac{\alpha}{1-\alpha} & \alpha \le 0.5 \\[1.1em] \dfrac{1-\alpha}{\alpha} & \alpha > 0.5 \end{cases} \qquad w_2 = \frac{w_1 + \xi}{\sqrt{2 w_1 \xi + \xi^2 + 1}}$$

This admits points with **$z \le 0$** (rays beyond $90°$), which is exactly why the model
supports a $> 180°$ field of view. A naive `z > 0` test — a common implementation mistake —
would reject those rays and silently cap the FOV below $180°$; this library does **not**
make that mistake.

**Forward / inverse equations** (for reference):

$$d_2 = \sqrt{x^2 + y^2 + (\xi d_1 + z)^2}, \quad
\begin{bmatrix} u \\ v \end{bmatrix} =
\begin{bmatrix} f_x \, x / (\alpha d_2 + (1-\alpha)(\xi d_1 + z)) + c_x \\
f_y \, y / (\alpha d_2 + (1-\alpha)(\xi d_1 + z)) + c_y \end{bmatrix}$$

Unprojection is closed-form; with $m_x=(u-c_x)/f_x$, $m_y=(v-c_y)/f_y$, $r^2=m_x^2+m_y^2$ it is
valid for all $r^2$ when $\alpha \le 0.5$, and for $r^2 \le 1/(2\alpha-1)$ when $\alpha > 0.5$.

**Valid parameter domain:** $\alpha \in [0, 1]$, $\xi \in [-1, 1]$.

### 7.3. Visualization (Augmented FOV Zones)
We generated an augmented visualization that overlays the **FOV Zones** directly onto the real image (Sample 96).
- **Green Zone (Frontal FOV)**: $\theta < 90^\circ$. Safe for standard pinhole projection.
- **Yellow Zone (Side/Back FOV)**: $90^\circ \le \theta < \theta_{limit}$. Valid in DS model, but mathematically impossible to project to a single pinhole image ($Z \le 0$).
- **Red Zone (Invalid Cone)**: $\theta \ge \theta_{limit}$. Mathematically impossible in DS model.
- **White Stars**: Real data keypoints. Notice how they all fall safely within the Green/Yellow valid regions.

![FOV Zones Augmented](assets/fov_zones_augmented.jpg)

**Reference**: [Double sphere model projection-failed region](https://jseobyun.tistory.com/457?category=1170976)

### 7.4. The Practical Limit (Infinite Size)
Even pixels in the **Yellow Zone** ($Z \le 0$) cannot be undistorted to a pinhole image.
- A pinhole camera can only see things **in front of it** ($Z > 0$).
- Rays at 90° project to infinity ($x/z \to \infty$).
- To capture these pixels, the undistorted image would need to be **infinitely wide**.

![Coverage Visualization](assets/coverage_vis.jpg)
*Bright: Preserved pixels. Dark: Lost pixels (due to mathematical or practical limits).*

---

## 8. Geometric Accuracy Verification

**Generated by:** `tests/verify_k_inverse.py` and `tests/verify_3d_reconstruction.py`

We performed rigorous checks to ensure the undistorted images are geometrically accurate for 3D measurement. You can run the full verification suite with:
```bash
bash verify_all.sh
```

### 8.1. Inverse Projection ($K^{-1}$) Analysis
We verified that any pixel $(u, v)$ in the undistorted image can be unprojected to a 3D ray using $\mathbf{d} = K_{new}^{-1} [u, v, 1]^T$.
- **Mean Error**: < 0.00003 pixels (across all modes).
- **Status**: ✅ Verified.

### 8.2. 3D Reconstruction Verification
We reconstructed the absolute 3D positions of the checkerboard corners from the undistorted images.
- **Mean Position Error**: `1.168 mm`
- **Reconstructed Square Size**: `20.01 cm` (Target: 20.00 cm)
- **Status**: ✅ Verified.

**Conclusion:** The undistorted images produced by this wrapper are **geometrically accurate pinhole projections** suitable for precise 3D computer vision tasks.

### 8.3. End-to-End Check on Real Images
`verify_real_samples.py` runs every code path on the real images with the
calibrated parameters and writes labeled visuals to `verification_output/`:

```bash
python verify_real_samples.py
```

| Check (real data) | `test_image` | `test_image_96` |
| :--- | :---: | :---: |
| PnP + reprojection RMS | **0.43 px** | **0.85 px** |
| Undistort: object API vs `cv.py` wrapper | identical | identical |
| Keypoint undistort: DS-analytic vs LDC mesh | 0.11 px | 9.96 px* |

\*The larger gap on `test_image_96` is the expected peripheral divergence of the
LDC fixed-point point-inverter (board near the image edge) — see the LDC best-practice note in [§6.5](#65-hardware-ldc-export-ti-jacinto-j7--tda4).

---

## 9. FAQ

### Q: My undistorted image has black borders?
**A:** This is normal for fisheye undistortion. The "pinhole" view cannot capture the full >180° FOV. You can adjust the `balance` parameter in `estimateNewCameraMatrixForUndistortRectify` to zoom in (crop borders) or zoom out (keep more content but more borders).

### Q: PnP fails or gives bad results?
**A:** Use `cam.solve_pnp(...)` (or `ds_msp.cv.solvePnP`). Standard `cv2.solvePnP` assumes a pinhole model and will fail on raw fisheye images. Also check that your 3D points are defined correctly (z=0 for planar targets). Note that PnP needs at least 4 points in front of the camera ($z > 0$ after unprojection).

### Q: What ranges are valid for `xi` and `alpha`?
**A:** `alpha ∈ [0, 1]` (enforced at construction) and `xi ∈ [-1, 1]`. Values of `xi > 1` push the model into a non-injective regime where unprojection can no longer invert projection; the calibrator constrains to this domain automatically. Real fisheye lenses typically land in `xi ∈ [-0.2, 0.6]`.

### Q: Do I have to pass `width` and `height`?
**A:** No. They are optional and only needed for image-level operations (undistortion maps / `compute_K_new`). For projection, unprojection, Jacobians, and PnP, construct with just the 6 intrinsics.

### Q: How do I use this with ROS?
**A:** You can easily wrap `ds_msp.cv.undistortImage` in a ROS node. Subscribe to `image_raw`, undistort, and publish to `image_rect`.

---

## 10. Credits

This project builds on excellent open-source work and research. Where we borrowed
ideas, algorithms, or formats, credit is due to the original authors:

### Model conversion (the multi-model adapter)
*   **Fisheye-Calib-Adapter** — **Sangjun Lee**, *"Fisheye-Calib-Adapter: An Easy
    Tool for Fisheye Camera Model Conversion"*, arXiv:**2407.12405** (2024) ·
    [github.com/eowjd0512/fisheye-calib-adapter](https://github.com/eowjd0512/fisheye-calib-adapter).
    Our model-conversion design (sample → unproject with the source → linear-seed →
    nonlinear refine on pixel reprojection error, with per-model analytic
    Jacobians) and the set of supported models follow this work. The original is
    C++/Ceres; DS-MSP re-implements the approach in pure Python/SciPy.

### Camera models
*   **Double Sphere** — **V. Usenko, N. Demmel, D. Cremers**, *"The Double Sphere
    Camera Model"*, 3DV 2018, arXiv:**1807.08957**. Reference implementation:
    **[basalt-headers](https://gitlab.com/VladyslavUsenko/basalt-headers)** (the
    half-space projection-validity condition and analytic Jacobians follow it).
*   **Kannala-Brandt** (equidistant fisheye) — **J. Kannala, S. Brandt**, 2006;
    cross-checked against **OpenCV** `cv2.fisheye`.
*   **Radial-Tangential (Brown-Conrady)** — **D. C. Brown**, 1966; cross-checked
    against **OpenCV** `cv2.projectPoints` / `calib3d`.
*   **OCamCalib** (omnidirectional polynomial) — **D. Scaramuzza et al.**
*   **Extended UCM (EUCM)** — **B. Khomutenko, G. Garcia, P. Martinet**, 2016.
*   **Unified Camera Model (UCM)** — **C. Geyer & K. Daniilidis** / **C. Mei & P.
    Rives**.

### Calibration ecosystem & tooling
*   **Kalibr** — **P. Furgale et al.**, [github.com/ethz-asl/kalibr](https://github.com/ethz-asl/kalibr)
    (DS & EUCM models contributed by V. Usenko). We follow Kalibr's `camchain`
    YAML format and per-model field orderings for interop.
*   **[dscamera](https://github.com/matsuren/dscamera)** — Python DS utilities.
*   **[Double Sphere Model Explanation](https://jseobyun.tistory.com/455)** &
    **[Projection-Failed Region Analysis](https://jseobyun.tistory.com/457?category=1170976)**
    — clear write-ups of the model and its "cone of invalidity".

### This codebase
*   **Muhammadjon Boboev** — original Python implementation of the Double Sphere
    intrinsics calibration this project grew from.

---

**Happy coding! 🚀**
