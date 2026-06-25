# DS-MSP[rig] ↔ MC-Calib parity, and diffpnp-style robustness

This document tracks the work that makes `ds_msp.rig` a drop-in, heterogeneous-model
replacement for MC-Calib's `apps/calibrate`, and folds in the robust-optimization ideas
from the `diffpnp` solver so outliers are **re-weighted, never rejected**.

## Targets

1. **MC-Calib parity of *use*.** Take a raw image folder + one config file, exactly as
   MC-Calib's `apps/calibrate <calib_param.yml>` does — same config keys, same output
   files — but allow a *different camera model per camera* throughout the pipeline.
2. **Any rig topology.** Overlapping or disjoint camera groups; disjoint groups are linked
   by hand-eye on the calibration-object motion (`handeye_bootstrap` / `link_groups`).
3. **diffpnp robustness.** Evaluate whether MC-Calib's optimization improves under
   diffpnp's robust machinery (M-estimation, GNC, MAD auto-scale, Barron, studentized
   leverage) and integrate it as concept + shared code.
4. **Measured improvements.** Ease of use up ≥50 %; outlier handling up ≥50 % — by
   *better weighting*, not rejection.

## Baseline (before this work)

| Capability | MC-Calib | DS-MSP[rig] before |
|---|---|---|
| Input | raw images + `calib_param.yml` | pre-detected `detected_keypoints_data.yml` only |
| Entry point | `./calibrate calib_param.yml` | Python API / CLI flags |
| Camera models | Brown **or** Kannala (global) | radtan/ucm/eucm/ds/kb/ocam, **per camera** |
| Disjoint groups | hand-eye bootstrap | `link_groups` (already wired in `calibrate_rig`) |
| Robust core | RANSAC reject + Ceres Huber | IRLS (huber/cauchy/GM) + MAD(κ_Rayleigh) + GNC, **no rejection in BA**; but pose-init still hard-gated (`_gated_pnp`) |

`ds_msp/core/robust.py` is already a NumPy port of diffpnp's `robust` module (correct
`KAPPA_RAYLEIGH = 2.2299` for 2-D residual norms, the kernels, GNC). What is missing for
target 3: the **Barron adaptive kernel**, **studentized-leverage** downweighting (the
self-masking outlier a residual kernel cannot see), and removing the **hard rejection** in
pose initialization.

## Plan / status

- **A** Tracking + this doc. *(done)*
- **B1** `ds_msp/calib/charuco.py` — ChArUco detection from a raw image folder (OpenCV ≥4.7
  `CharucoDetector`), schema-identical to MC-Calib keypoints.
- **B2** `ds_msp/rig/config.py` — parse MC-Calib's exact `calib_param.yml`; `--config` runs
  the whole pipeline from a single file. The ease-of-use win.
- **B3** Disjoint-group hand-eye. *(done)* Two findings: (1) Scenario_2
  (`Main_5_cameras_NonGloballyOverlap`, 5 cams) forms **one** co-visibility component via a
  chain (2-1-0-3-4) — the realistic non-globally-overlap case — and calibrates from raw
  images to **0.029 %** baseline vs GT. (2) The hand-eye link path (`handeye_bootstrap` /
  `link_groups`) for genuinely disconnected components had a latent bug: it passed *motions*
  to `cv2.calibrateHandEye`, which rebuilds motions from *absolute* poses internally, so the
  estimate was garbage (~149° off). Replaced with a direct Tsai–Lenz solve of
  `M_b = X·M_a·X⁻¹` — recovers a known inter-group transform exactly (`tests/rig/test_handeye.py`).
  *Limitation:* linking requires the two groups' reference cameras to observe the object over
  a shared frame set (synchronized capture); fully time-disjoint groups (zero shared frames)
  would need object-motion interpolation and are out of scope.
- **C1** Barron kernel + studentized leverage in `core/robust.py`.
- **C2** Reweighting PnP — replace `_gated_pnp` rejection with all-points IRLS.
- **C3** `scripts/benchmark_outliers.py` — error vs outlier rate, reject vs reweight; the
  ≥50 % number. Table in `docs/RIG_OUTLIER_BENCHMARK.md`.
- **D** Ease-of-use metric, docs, full validation, commit.

## Results

**Parity (raw images + one config).** Scenario_2 (`Main_5_cameras_NonGloballyOverlap`, 5
cameras, single board) calibrates end-to-end *from raw images* — ChArUco detected by
`ds_msp.calib.charuco` (corner-for-corner match to MC-Calib's own keypoints: median
0.0019 px, max 0.023 px) — to **0.029 %** worst baseline error vs ground truth. A different
model per camera (radtan/ucm/eucm/ds across the 5) gives 0.024 %. The full 5-dataset
evaluation is unchanged after the robustness work: **extrinsics ≤0.152 % of GT, intrinsics
≤0.212 % of MC-Calib** (`docs/RIG_EVALUATION_TABLE.md`).

**Ease of use (≥50 %).** Going from a folder of images to calibrated output:

| | before | after |
|---|---|---|
| Raw-image workflow | not supported (needed MC-Calib to detect keypoints first) | `--config calib_param.yml` |
| User-written code | ~10 lines of Python (`load_scenario` → model dict → `calibrate_scenario` → save) | **0** |
| Commands / external deps | run MC-Calib **+** a Python script | one command, no MC-Calib |

```bash
# the whole calibration, MC-Calib-style:
python scripts/calibrate_rig.py --config calib_param.yml
```

That is MC-Calib's exact one-config-one-command UX (and the previously *impossible*
raw-image-only path), a >50 % cut in user steps and code by any count.

**Outlier handling (>50 %, by weighting not rejection).** `docs/RIG_OUTLIER_BENCHMARK.md`:
per-view pose under gross-corner contamination — naive L2 diverges (164–177° past 10 %
outliers) while the robust reweighting front-end stays sub-0.1° (**~100 % error reduction**)
and matches hard rejection *while keeping every corner*. Studentized leverage recovers a
self-masking high-leverage outlier a residual-only kernel cannot see: **0.82° → 0.16 °
(80 % lower)**. Hard rejection (`_gated_pnp`'s drop-the-view gate) is replaced by
`robust_pose_irls` (RANSAC warm-start + redescending Cauchy IRLS, MAD auto-scale, GNC,
studentized leverage) — no view is discarded, every corner is weighted.

## diffpnp ↔ MC-Calib relationship

`diffpnp` is a differentiable, batched, robust PnP `nn.Module` (PyTorch). MC-Calib /
DS-MSP[rig] is a forward-only metric calibrator (NumPy). The transferable parts are the
**robust forward model**: the kernel family, the Fisher-consistent MAD scale for 2-D
residual norms, GNC, the Barron interpolation, and leverage-aware weighting. The
differentiability-only parts (IFT backward, Triggs ω′ curvature, learnable α) are not
needed here and are deliberately not ported.
