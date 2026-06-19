# Datasets

Free, small (<10 GB total) public datasets for the `CAREER_ROADMAP.md` work. This folder is
**git-ignored** — re-fetch anytime with `bash scripts/download_datasets.sh [all|tumvi|tumrgbd|euroc]`
(resumable). Each dataset below is mapped to the roadmap tier it serves and the exact
**reference file to validate against**.

## TUM-VI — fisheye stereo + IMU + mocap GT  *(centerpiece)*
`tumvi/dataset-room1_512_16/` · `tumvi/dataset-calib-cam1_512_16/` · `tumvi/dataset-calib-imu1_512_16/`

- **room1**: 2821 stereo pairs (512×512 fisheye), 28k IMU samples (200 Hz), 16.5k mocap GT
  poses. `mav0/{cam0,cam1,imu0,mocap0}/` in EuRoC/ASL layout.
- **Reference calibration** (the validation target): `dso/camchain.yaml` — **Kalibr camchain**
  format, which `ds_msp.io.kalibr` already parses. It is `pinhole + equidistant`
  = **Kannala-Brandt** (your `KannalaBrandtModel`, OpenCV-fisheye compatible), *not* Double
  Sphere. It contains everything Tier 1 needs:
  - **intrinsics** per cam (e.g. cam0 `fx≈190.98, fy≈190.97, cx≈254.93, cy≈256.90` + 4 KB
    distortion coeffs) → validate your KB calibration against these.
  - **`T_cn_cnm1`** → stereo extrinsics (baseline ≈ 0.101 m) for **multi-camera calibration**.
  - **`T_cam_imu`** → **camera–IMU** extrinsics, for cam-IMU calibration.
  > To get a **Double Sphere** reference specifically, either run the repo's own DS calibrator
  > on these images, or `convert()` the published KB params to DS — both are good Tier-1 demos.
- **Roadmap use:** Tier 1 (intrinsics / multi-cam / cam-IMU calibration vs. reference),
  Tier 2 (VO/VIO — `mocap0` is the ground-truth trajectory for ATE/RPE).
- Source: TUM-VI, `https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16/`.

## TUM RGB-D (freiburg1_xyz) — RGB + depth + GT pose
`tumrgbd/rgbd_dataset_freiburg1_xyz/` — 798 RGB + 798 aligned depth frames + `groundtruth.txt`.
- **Roadmap use:** Tier 4 — validate metric monocular-depth predictions against real depth.
- Source: `https://cvg.cit.tum.de/rgbd/dataset/freiburg1/`.

## EuRoC MAV — Vicon Room 1  ✅ (ASL format)
`euroc/V1_01_easy/` · `euroc/V1_02_medium/` · `euroc/V1_03_difficult/` — stereo
(global-shutter, 752×480) + IMU + Vicon GT. V1_01 = 2912 stereo pairs, 29k IMU, 28.7k GT.
- **Reference calibration** (Tier-1 target): each `mav0/cam0/sensor.yaml` —
  `pinhole + radial-tangential`, cam0 `intrinsics [458.654, 457.296, 367.215, 248.375]` + 4
  radtan coeffs → validate your **RadTan** model against these. `mav0/cam1/sensor.yaml` +
  the `T_BS` blocks give stereo extrinsics; `mav0/state_groundtruth_estimate0/data.csv` is
  the GT trajectory for ATE/RPE.
- **Roadmap use:** Tier 2 — the universal VO benchmark (report ATE on V1_01_easy).
- Only the ASL `.zip` sequences were kept; the ROS1 `.bag` files (and the 5.6 GB outer
  bundle) were discarded to stay under budget. Re-fetch via `scripts/download_datasets.sh`
  with `EUROC_VR1=<link>` if needed.

---
**Note:** running the repo on this data needs its deps in your Python env:
`pip install numpy opencv-python scipy pyyaml` (ideally in a venv; system Python 3.13 here has none).
