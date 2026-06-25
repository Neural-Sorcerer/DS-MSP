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

## Feature-parity closure (multi-board reconstruction, hand-eye approach)

The remaining real MC-Calib feature gap — **multi-board fused-object reconstruction** — is
now closed. Previously `number_board > 1` required a pre-built `calibrated_objects_data.yml`;
DS-MSP loaded it rather than building it. The reconstruction MC-Calib performs in
`calibrate3DObjects` (McCalib.cpp:765-956) is now implemented in `ds_msp/rig/reconstruct.py`:

1. detect each board's ChArUco corners per image (object-free `BoardObs`);
2. bootstrap each camera's intrinsics (planar `calibrateCamera`, focal-plausibility guard +
   consensus fallback) and resect every board by PnP → `T_c_b`;
3. average the inter-board relative poses over every image that sees a board pair and fuse
   the boards into one rigid `Object3D` per covisibility component (`object3d.build_objects`,
   `init3DObjects` analogue);
4. pool all boards seen in one image into a single `ObjectObs` (the most-constrained pose).

`config.calibrate_from_config` now reconstructs the object when no pre-built one is supplied,
so a multi-board rig calibrates **straight from a raw image folder**, exactly like MC-Calib.

A latent multi-camera bug was fixed in `build_objects`: board-pair transforms must be keyed
by `(cam_id, frame_id)`, not `frame_id` alone — two boards seen by *different* cameras in the
same frame have `T_c_b` in different camera frames, so `inv(T_c_b2)·T_c_b1` would be garbage.
`he_approach` (0 = bootstrap Tsai, 1 = traditional single Tsai) is now consumed end-to-end
(`config → calibrate_scenario → calibrate_rig → link_groups`).

**Validation (real data, Scenario_1 — a 3-board object, 2 cameras).** Reconstructing the
fused object from the detected keypoints alone (no pre-built object) recovers MC-Calib's
object geometry to **0.058 mm RMS over a 2.83 m extent** (2·10⁻⁵ relative). The full
config-driven pipeline through that reconstructed object gives **0.013 % inter-camera
baseline error and 0.009° rotation error vs ground truth** (0.086 px reprojection RMS) —
matching the pre-built-object path. Synthetic regression in `tests/rig/test_reconstruct.py`.

### Parity matrix (final)

| MC-Calib capability | DS-MSP[rig] | Notes |
|---|---|---|
| Raw images + single `calib_param.yml` | ✅ | `--config`, every MC-Calib key + per-camera `camera_models` |
| Single-board object | ✅ | built from config |
| **Multi-board fused-object reconstruction** | ✅ **(new)** | `rig/reconstruct.py`; 0.058 mm vs MC-Calib object |
| Per-camera heterogeneous models | ✅ (exceeds) | radtan/ucm/eucm/ds/kb/ocam per camera |
| Overlapping camera groups | ✅ | covisibility graph + shortest-path compose |
| Disjoint groups via hand-eye | ✅ | Tsai–Lenz; `he_approach` 0/1 selectable |
| Fully time-disjoint groups (no shared frames) | ⚖️ *parity* | MC-Calib's `initNonOverlapPair` also needs ≥3 *common* frames — neither tool handles zero-overlap; not a gap |
| OCam in bundle adjustment | ✅ (exceeds) | DS-MSP has full project/unproject/Jacobian; MC-Calib's `OcamCalib` has no templated `project()` and is in *no* Ceres ladder |
| Huber/robust BA, SPARSE_SCHUR | ✅ (exceeds) | Schur + IRLS (Cauchy/GNC/Barron/studentized), no rejection |
| Outputs: cameras, object, poses, reproj-error, keypoints, overlay images | ✅ | MC-Calib's exact YAML schema |
| `ransac_threshold`, `number_iterations` config knobs | ⚖️ superseded | parsed; DS-MSP auto-tunes the RANSAC scale (κ-Rayleigh MAD) and BA iteration budget, so these tuning knobs are intentionally not hard-wired — see the improvement report |
| Board-structure refinement in BA (`refineObject`) | ◻️ future | inter-board `T_co_b` currently frozen post-reconstruction; see 30/30/30 report (accuracy) |

## diffpnp ↔ MC-Calib relationship

`diffpnp` is a differentiable, batched, robust PnP `nn.Module` (PyTorch). MC-Calib /
DS-MSP[rig] is a forward-only metric calibrator (NumPy). The transferable parts are the
**robust forward model**: the kernel family, the Fisher-consistent MAD scale for 2-D
residual norms, GNC, the Barron interpolation, and leverage-aware weighting. The
differentiability-only parts (IFT backward, Triggs ω′ curvature, learnable α) are not
needed here and are deliberately not ported.
