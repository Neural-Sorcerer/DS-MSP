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
- **B3** Disjoint-group hand-eye evaluation (Scenario_2 is `…NonGloballyOverlap`).
- **C1** Barron kernel + studentized leverage in `core/robust.py`.
- **C2** Reweighting PnP — replace `_gated_pnp` rejection with all-points IRLS.
- **C3** `scripts/benchmark_outliers.py` — error vs outlier rate, reject vs reweight; the
  ≥50 % number. Table in `docs/RIG_OUTLIER_BENCHMARK.md`.
- **D** Ease-of-use metric, docs, full validation, commit.

## diffpnp ↔ MC-Calib relationship

`diffpnp` is a differentiable, batched, robust PnP `nn.Module` (PyTorch). MC-Calib /
DS-MSP[rig] is a forward-only metric calibrator (NumPy). The transferable parts are the
**robust forward model**: the kernel family, the Fisher-consistent MAD scale for 2-D
residual norms, GNC, the Barron interpolation, and leverage-aware weighting. The
differentiability-only parts (IFT backward, Triggs ω′ curvature, learnable α) are not
needed here and are deliberately not ported.
