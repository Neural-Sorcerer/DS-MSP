# Datasets — what to download, where it goes, and how to start

This folder holds the public datasets the examples and roadmap run on. It is **git-ignored**
(only this README is tracked), so nothing here bloats the repo — you fetch the data yourself
with one command and it lands in the exact paths the scripts expect.

Everything is free, and the whole set fits in **< 10 GB**.

---

## 1. TL;DR — start the fisheye learning track in 4 commands

The entire `docs/learn/` curriculum (Chapters 1–2, the calibration capstone, and both
deep-dives — `examples/01`–`05`) needs **only TUM-VI (~2.5 GB)**:

```bash
# from the repo root
uv venv --python 3.12 && source .venv/bin/activate   # (or python -m venv .venv)
uv pip install -e ".[calib]"                          # library + AprilGrid detector
bash scripts/download_datasets.sh tumvi               # ~2.5 GB, resumable
python examples/01_realdata_fisheye_tumvi.py          # first hands-on lesson
```

Then walk the curriculum in order — see [`docs/learn/README.md`](../docs/learn/README.md).
You do **not** need EuRoC or TUM RGB-D until the later (VO / depth) tiers.

---

## 2. Where files land

The downloader writes everything under `datasets/`, in the standard EuRoC/ASL layout the
loaders expect:

```
datasets/
├── tumvi/
│   ├── dataset-room1_512_16/        # fisheye stereo seq (VO, depth, Ch.1–2)
│   │   ├── mav0/{cam0,cam1,imu0,mocap0}/   # images, IMU, mocap ground truth
│   │   └── dso/camchain.yaml        # <- published reference calibration
│   ├── dataset-calib-cam1_512_16/   # AprilGrid footage (calibration capstone)
│   │   └── mav0/{cam0,cam1,...}/
│   └── dataset-calib-imu1_512_16/   # cam+IMU calibration footage
├── euroc/
│   └── V1_01_easy/ (+ V1_02, V1_03)
│       └── mav0/{cam0,cam1,imu0,state_groundtruth_estimate0}/
│           └── cam0/sensor.yaml     # <- published radtan calibration
└── tumrgbd/
    └── rgbd_dataset_freiburg1_xyz/  # rgb/, depth/, groundtruth.txt
```

Re-fetch anytime: `bash scripts/download_datasets.sh [all|tumvi|tumrgbd|euroc]` (resumable).

---

## 3. What you need for each part of the roadmap

Find your goal, grab **only** the sub-part it needs.

| Goal / lesson | Exact data sub-part | Path | Get it | Run |
|---|---|---|---|---|
| **Ch.1** fisheye & camera models | TUM-VI room1 (cam0 + `camchain.yaml`) | `tumvi/dataset-room1_512_16/` | `… tumvi` | `examples/01_realdata_fisheye_tumvi.py` |
| **Ch.2** Double Sphere model | TUM-VI room1 (`camchain.yaml`) | `tumvi/dataset-room1_512_16/dso/` | `… tumvi` | `examples/02_double_sphere_tumvi.py` |
| **🏆 Capstone** calibrate a real camera | TUM-VI calib-cam1 (**cam0** AprilGrid frames) | `tumvi/dataset-calib-cam1_512_16/mav0/cam0/data/` | `… tumvi` | `examples/03_calibrate_tumvi_aprilgrid.py` |
| **Deep-dive** robust loss | same calib-cam1 corners | `tumvi/dataset-calib-cam1_512_16/` | `… tumvi` | `examples/04_robust_vs_rejection.py` |
| **Deep-dive** same camera? | same calib-cam1 corners | `tumvi/dataset-calib-cam1_512_16/` | `… tumvi` | `examples/05_model_equivalence.py` |
| **Tier 1** multi-camera extrinsics | TUM-VI calib-cam1 (**cam0 + cam1**) → vs `T_cn_cnm1` | `tumvi/dataset-calib-cam1_512_16/mav0/{cam0,cam1}/` | `… tumvi` | *(roadmap — Tier 1)* |
| **Tier 1** camera–IMU calibration | TUM-VI calib-imu1 (cam + **imu0**) → vs `T_cam_imu` | `tumvi/dataset-calib-imu1_512_16/mav0/` | `… tumvi` | *(roadmap — Tier 1)* |
| **Tier 1** fisheye stereo → depth | TUM-VI room1 (**cam0 + cam1** pairs) | `tumvi/dataset-room1_512_16/mav0/{cam0,cam1}/` | `… tumvi` | *(roadmap — Tier 1)* |
| **Tier 2** monocular VO + ATE/RPE | EuRoC V1_01 (**cam0** + GT) *or* TUM-VI room1 (cam0 + mocap0) | `euroc/V1_01_easy/mav0/` | `… euroc`† | *(roadmap — Tier 2)* |
| **Tier 4** metric fisheye depth | TUM RGB-D fr1 (**rgb + depth + GT**) | `tumrgbd/rgbd_dataset_freiburg1_xyz/` | `… tumrgbd` | *(roadmap — Tier 4)* |
| **Tier 4** SuperPoint VO | EuRoC V1_01 (cam0) | `euroc/V1_01_easy/mav0/cam0/` | `… euroc`† | *(roadmap — Tier 4)* |

`… X` is shorthand for `bash scripts/download_datasets.sh X`. †EuRoC needs a one-time link
(see §4.2).

---

## 4. Dataset details (sub-parts, structure, the reference file to validate against)

### 4.1 TUM-VI — fisheye stereo + IMU + mocap GT *(centerpiece, ~2.5 GB)*
Three sub-sequences, all 512×512 fisheye in EuRoC/ASL layout. `bash scripts/download_datasets.sh tumvi`
fetches all three.

- **`dataset-room1_512_16/`** — the motion sequence: stereo fisheye (`mav0/cam0`, `mav0/cam1`),
  200 Hz IMU (`mav0/imu0`), and **mocap ground-truth trajectory** (`mav0/mocap0`) for VO ATE/RPE.
- **`dataset-calib-cam1_512_16/`** — someone waving a **6×6 AprilGrid** (tag family `t36h11`,
  tag size 88 mm, spacing 0.3) in front of the camera. **436 frames** in
  `mav0/cam0/data/` — this is what the calibration capstone detects and calibrates from.
- **`dataset-calib-imu1_512_16/`** — calibration motion with synchronized cam + IMU, for
  camera–IMU extrinsic calibration.
- **Reference calibration (your validation target):** `dso/camchain.yaml` — a **Kalibr
  camchain** that `ds_msp.io.kalibr` parses directly. It is `pinhole + equidistant` =
  **Kannala-Brandt** (`KannalaBrandtModel`, OpenCV-fisheye compatible), *not* Double Sphere.
  It carries everything Tier 1 checks against:
  - per-cam **intrinsics** (cam0 ≈ `fx 190.98, fy 190.97, cx 254.93, cy 256.90` + 4 KB coeffs)
    → compare to your calibrated intrinsics (the capstone does this).
  - **`T_cn_cnm1`** → stereo extrinsics (baseline ≈ 0.101 m), for multi-camera calibration.
  - **`T_cam_imu`** → camera–IMU extrinsics, for cam-IMU calibration.
  > Want a **Double Sphere** reference number specifically? The reference is KB, so either
  > calibrate DS directly on the images (the capstone fits both KB and DS), or `convert()` the
  > published KB to DS (Chapter 2). Why DS `fx` looks different from KB is proven in
  > [`docs/learn/are_two_models_the_same_camera.md`](../docs/learn/are_two_models_the_same_camera.md).
- Source: `https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16/`.

### 4.2 EuRoC MAV — Vicon Room 1 *(stereo + IMU + GT + radtan, ~4.9 GB)*
Three sequences: `V1_01_easy`, `V1_02_medium`, `V1_03_difficult` (global-shutter stereo
752×480 + IMU + Vicon GT). One easy sequence is enough for Tier 2; the harder two are free
robustness cases.
- **Reference calibration (Tier-1 radtan target):** each `mav0/cam0/sensor.yaml` —
  `pinhole + radial-tangential`, cam0 `intrinsics [458.654, 457.296, 367.215, 248.375]` + 4
  radtan coeffs → validate the `RadTanModel`. `mav0/cam1/sensor.yaml` + the `T_BS` blocks give
  stereo extrinsics.
- **Ground truth:** `mav0/state_groundtruth_estimate0/data.csv` — the trajectory for ATE/RPE.
- **Download:** EuRoC moved to the ETH Research Collection, which serves a **browser/JS bundle**
  (no scriptable URL). Open the EuRoC page, right-click **"Vicon Room 1 Datasets"** → *Copy
  link*, then:
  ```bash
  EUROC_VR1='<pasted-url>' bash scripts/download_datasets.sh euroc
  ```
  (Only the ASL `.zip` sequences are kept; the ROS1 `.bag` files are discarded to stay under
  budget.)

### 4.3 TUM RGB-D — `freiburg1_xyz` *(RGB + depth + GT, ~0.46 GB)*
`tumrgbd/rgbd_dataset_freiburg1_xyz/`: 798 RGB + 798 aligned depth frames + `groundtruth.txt`.
- **Use:** Tier 4 — validate metric monocular-depth predictions against real depth.
- **Get it:** `bash scripts/download_datasets.sh tumrgbd`.
- Source: `https://cvg.cit.tum.de/rgbd/dataset/freiburg1/`.

---

## 5. Disk budget & optional extras

Current footprint with all three groups: **~7.8 GB** (TUM-VI 2.5 + EuRoC 4.9 + RGB-D 0.46) —
under the 10 GB target, with headroom.

Two roadmap items are **optional and intentionally not downloaded**:
- **EuRoC radtan *self*-calibration** would need EuRoC's separate "Calibration Datasets" bundle
  (~4.2 GB) — it **breaks the 10 GB budget** and is redundant: the TUM-VI capstone already shows
  "calibrate from detected corners → match published", and EuRoC's radtan reference in
  `sensor.yaml` is directly usable for VO without re-calibrating.
- **A Mip-NeRF 360 scene (~0.5–1.5 GB)** for the *optional* fisheye→3D-Gaussian-Splatting stretch
  (Tier 4). It fits under budget if you ever pursue that demo; not needed otherwise.

Everything else in the roadmap (Tiers 0–3, 5, and the rest of Tier 4) is fully covered by the
data above.

---

## 6. Start here (hands-on)

1. **Set up + fetch** the fisheye data (§1).
2. **Read + run Chapter 1**, then look at the printed numbers, not just the pretty pictures:
   [`docs/learn/01_fisheye_and_camera_models.md`](../docs/learn/01_fisheye_and_camera_models.md).
3. **Work through the curriculum** ([`docs/learn/README.md`](../docs/learn/README.md)):
   Ch.1 → Ch.2 → 🏆 capstone → the two deep-dives. Each pairs a short explainer with a script
   that prints a number you can verify.
4. **Change one thing and predict the result** before re-running (a different `--stride`, `cam1`
   instead of `cam0`, `loss="linear"` instead of Cauchy). Breaking it on purpose and watching the
   number move is the fastest way to learn the geometry.

> Everything runs on a laptop CPU in seconds. No GPU, no capture hardware — the datasets already
> contain the synchronized fisheye-stereo + IMU + ground-truth + published calibrations you'd
> otherwise need a rig to produce.
