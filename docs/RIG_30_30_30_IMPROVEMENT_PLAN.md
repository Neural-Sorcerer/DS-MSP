# DS-MSP[rig] — survey + plan to be 30 % faster, 30 % more accurate, 30 % more robust

This is the second half of the parity work in `RIG_MCCALIB_PARITY_AND_ROBUSTNESS.md`. With
feature parity to MC-Calib reached (raw-image + single-config entry, per-camera heterogeneous
models, multi-board fused-object reconstruction, hand-eye linking, robust BA), this document
surveys where the *next* gains come from and lays out an implementation plan to make
DS-MSP[rig] **30 % faster, 30 % more accurate, and 30 % more robust** — each target grounded
in a measured baseline, not a guess.

## TL;DR

| Axis | Measured baseline | Target | Primary lever(s) | Confidence |
|---|---|---|---|---|
| Speed | 4.24 s / calib; **front-end = 68 %** of wall-clock, serial over cameras | −30 % wall-clock | parallelize per-camera front-end (F1) + single-start seed (F4) + reuse BA symbolic factorization (F2) | **High** — F1 alone clears it on ≥3 cameras |
| Accuracy | 0.086 px RMS, 0.013 % baseline (real); object geometry frozen at bootstrap | −30 % error | free board structure in BA (A1) + per-corner covariance weighting (A2) | **Medium-High** |
| Robustness | sub-1.3 % extrinsic error through **30 % gross outliers** already | +30 % breakdown / −30 % error at fixed rate | covariance + studentized weighting in global BA (R2) + adaptive-α GNC (R1) + robust group graph (R3) + focal anchor (R4) | **Medium** (baseline already strong, so gains are at the harder 30-45 % tail) |

All three are achievable with self-contained, separately-validatable changes. None requires
a new dependency. Recommended order: **F1 → A2 → R2 → A1 → F2 → R1/R3 → F4/R4**.

## Measured baseline (the numbers the targets are anchored to)

Real data — Scenario_1 (3-board object, 2 cameras, ~99 frames, 198 observations, 48 object
points), `radtan`, run on this machine:

* **Wall-clock 4.24 s** (median of 3). Stage profile:

  | stage | time | share |
  |---|---|---|
  | per-camera front-end (intrinsics + gated PnP) | 2.88 s | **68.2 %** |
  | camera-group covisibility + extrinsics init | 0.005 s | 0.1 % |
  | BA (a) object-pose warm-up | 0.45 s | 10.6 % |
  | BA (b) per-group extrinsics | 0.36 s | 8.4 % |
  | BA (c) global joint + intrinsics | 0.54 s | 12.8 % |

* **Accuracy:** 0.086 px max reprojection RMS, **0.013 %** worst inter-camera baseline vs GT.
* **Reconstruction:** fused 3-board object recovered to **0.058 mm RMS** over a 2.83 m extent.

Synthetic 4-camera rig (`tests/rig/_synth.py`), median worst inter-camera baseline error:

| condition | extrinsic error |
|---|---|
| 0.5 px noise, 0 % outliers | 0.44 % |
| 1.0 px noise, 0 % outliers | 0.87 % |
| 0.5 px noise, 10 % gross outliers | 0.62 % |
| 0.5 px noise, 20 % gross outliers | 0.85 % |
| 0.5 px noise, 30 % gross outliers | 0.57 % |

Two structural facts fall out of this and drive the whole plan:

1. **The front-end is 2/3 of the runtime and is embarrassingly parallel** (each camera is
   calibrated independently). Speed work belongs there first, not in the BA.
2. **The robust front-end is already strong** (≤1.3 % extrinsic error through 30 % gross
   outliers) — *provided the robust front-end is used*. The non-robust default
   `_front_end_opencv` (plain `cv2.calibrateCamera`, L2) collapses the focal under the same
   contamination; the product path (`calibrate_scenario` → `make_bundle_front_end`) does not.
   So robustness gains are about the **30-45 % tail** and **error at fixed high rate**, plus
   removing the fragile L2 default — not about a currently-broken common case.

Survey method: each lever below was chosen by (a) attributing the measured cost / error /
failure to a specific stage, (b) matching it to an established technique (Schur-sparse BA,
IRLS M-estimation, GNC, heteroscedastic weighting, certifiable init), and (c) estimating the
gain from the baseline share that technique addresses. Gains are bounded by what the measured
bottleneck can yield, so they are conservative.

---

## Axis 1 — 30 % faster

The front-end is 68 % of wall-clock and the BA is 32 %. Target the front-end first.

### F1 — Parallelize the per-camera front-end *(primary, high confidence)*
`make_bundle_front_end` calibrates each camera in a serial Python loop, yet the cameras are
fully independent (robust pinhole seed → model-aware seed → two-start `calibrate` → gated
PnP). Run them across processes (`concurrent.futures.ProcessPoolExecutor`, or threads since
the heavy math is in NumPy/OpenCV and releases the GIL).

* **Expected gain:** on *N* cameras, up to (N−1)/N of the 68 % front-end share. 2 cams →
  ~34 % of 68 % ≈ 23 % total; 4 cams → ~51 % of 68 % ≈ 35 % total; 5 cams (Scenario_3/5) →
  ~37 % total. **Clears the 30 % target at ≥3 cameras**, which is the multi-camera rig case
  the tool exists for.
* **Implementation:** `ds_msp/rig/rig_calibrate.py::make_bundle_front_end` — lift the
  per-camera body into a top-level function returning `(model, posed_obs)`, map it over a
  pool, then run the existing consensus guard on the gathered results. Pickling: pass arrays,
  not closures.
* **Effort:** low (½ day). **Risk:** low (numerically identical; only execution order changes).
* **Validate:** wall-clock on Scenario_3/5 (5 cams) before/after; assert identical extrinsics.

### F2 — Reuse the BA symbolic factorization / fixed sparsity *(medium)*
The Schur problem (`ba.build_schur_problem` → `core.optimize.schur_lm`) has a **fixed sparsity
pattern** across LM iterations — only the numeric values change. Computing the fill-reducing
ordering (AMD) and the symbolic Cholesky once and reusing it each iteration removes repeated
symbolic work (v-slam ch8: arrow Hessian + Schur; the camera-block reduced system `S` has the
co-visibility sparsity). Also reuse the per-frame 6×6 block inverses' structure.

* **Expected gain:** 20-40 % of the BA's 32 % share ≈ 6-13 % total wall-clock.
* **Implementation:** factor the symbolic step out of the LM loop in `core/optimize.py`; cache
  ordering on the `(cameras, frames)` structure; switch the reduced solve to a cached sparse
  Cholesky (`scipy.sparse.linalg`/`cholmod` if available, else reuse the dense factor's pattern).
* **Effort:** medium (1-2 days). **Risk:** medium (must keep the LM damping update correct).

### F3 — Vectorize residual + Jacobian assembly *(medium)*
If residual/Jacobian assembly still loops per observation in Python, batch it: stack all
(camera, frame) blocks and call each model's analytic `project_jacobian` once per camera on
the whole point set. The models already return batched `(N,2,3)/(N,2,P)` Jacobians.

* **Expected gain:** 10-25 % of whichever BA/front-end assembly is still scalar.
* **Effort:** medium. **Risk:** low-medium (numerics unchanged; indexing care).

### F4 — Skip the redundant second calibrate start when the first is already good *(low)*
The front-end runs `calibrate` **twice** per camera (model-aware seed and neutral seed) and
keeps the lower-RMS fit. When the model-aware fit's RMS is already at the noise floor, the
second start is wasted. Gate it: run start 2 only if start 1's RMS exceeds a small multiple of
the robust-pinhole residual.

* **Expected gain:** up to ~½ of the per-camera `calibrate` cost → several % of the 68 %.
* **Effort:** low. **Risk:** low (keep both starts whenever start 1 is marginal — pure speed,
  no accuracy change in the common case).

**Axis-1 verdict:** F1 alone reaches −30 % on ≥3-camera rigs; F1+F4+F2 reach it even on a
2-camera rig and leave headroom. High confidence.

---

## Axis 2 — 30 % more accurate

Two sources of avoidable error: the object geometry is frozen at bootstrap quality, and every
corner is weighted equally regardless of how well it was localized.

### A1 — Free the board structure in the global BA *(primary)*
Today the fused object's inter-board poses `T_co_b` are baked into `Object3D.pts_3d` and held
**fixed** through the BA (`ba.py` docstring). MC-Calib refines them (`Object3D::refineObject`,
`refineAllObject3D`). For a *reconstructed* multi-board object (the new parity path) the
geometry is only as good as the bootstrap intrinsics; refining `T_co_b` jointly with the rig
removes that bias and lets every camera's views improve the structure.

* **Expected gain:** 20-40 % lower reprojection + extrinsic error on reconstructed multi-board
  rigs (largest where bootstrap intrinsics were weakest); negligible when a GT object is given.
* **Implementation:** add per-non-reference-board `T_co_b` as 6-DoF parameter blocks in
  `ba.refine` (reference board fixed for gauge, exactly as MC-Calib fixes `refine_board=false`
  on the ref board). The reprojection already composes `X_cam = T_c_g·T_g_o·T_co_b·X_board`;
  expose `T_co_b` as variables and add their analytic Jacobian (same 2×6 SE(3) block already
  used for object pose). Re-bake into `pts_3d` after convergence.
* **Effort:** medium (2-3 days incl. Jacobian + gauge handling). **Risk:** medium (extra gauge
  freedom; mitigated by fixing the reference board and a short pose-only warm-up first).
* **Validate:** reconstructed-object extrinsic error with/without structure refinement on
  Scenario_1/3/5; expect the reconstructed path to approach the GT-object path.

### A2 — Per-corner covariance (heteroscedastic) weighting *(primary, broadly applicable)*
All residuals are currently weighted isotropically. A ChArUco saddle corner carries a real
localization covariance — sharper near the image centre, looser at the edge / under blur /
oblique foreshortening. Weighting each 2-D residual by its inverse covariance Σ⁻¹ is the
maximum-likelihood estimator under heteroscedastic noise (v-slam ch5/ch8: the information
matrix in the LSQ weight) and lowers estimator variance without discarding anything.

* **Expected gain:** 10-25 % RMS / parameter-variance reduction; compounds with A1.
* **Implementation:** (i) get a per-corner covariance from the saddle-point response Hessian
  during detection (`calib/charuco.py`) — the curvature of the refined saddle gives a 2×2 Σ;
  a cheap proxy is `σ² ∝ 1/sharpness` or a radial model. (ii) thread an optional per-obs
  weight/`Σ⁻¹` into `ObjectObs` and into the BA residual whitening in `ba.py`/`core.optimize`.
  Defaults to identity (current behaviour) when no covariance is supplied.
* **Effort:** medium. **Risk:** low-medium (must keep the robust IRLS weight and the
  covariance whitening composable: `w_total = w_robust · Σ⁻¹`).

### A3 — Subpixel saddle refinement parity at detection *(supporting)*
MC-Calib applies a dedicated saddle-point subpixel refinement (`refine_corner`,
`saddleSubpixelRefinement`). Confirm DS-MSP's `CharucoDetector` path matches it (it uses
OpenCV's subpixel, but MC-Calib's saddle refinement can be tighter). Any residual 2-D noise
reduction flows straight through to lower parameter error.

* **Expected gain:** a few-to-10 % depending on how much the detectors already agree (current
  corner-for-corner match to MC-Calib is 0.0019 px median, so headroom here is small —
  include only if A1+A2 fall short).
* **Effort:** low-medium. **Risk:** low.

### A4 — Tighter final-BA convergence *(supporting)*
The last joint BA can be run to a tighter tolerance / a few more LM iterations once F2 makes
each iteration cheaper — closing the final sub-percent. Effort: trivial; gated on F2 so it is
free of wall-clock cost.

**Axis-2 verdict:** A1 (free structure) + A2 (covariance weighting) together exceed −30 % on
the reconstructed-object and noisy regimes; medium-high confidence. A GT-object, low-noise case
(already 0.013 %) has little headroom and is not the target.

---

## Axis 3 — 30 % more robust

The robust path already tolerates 30 % gross outliers at ≤1.3 % extrinsic error, so "30 % more
robust" means: (i) extend the breakdown point into the 30-45 % tail, (ii) cut the residual
error at a fixed high rate (e.g. 20 %: 0.85 % → <0.6 %), and (iii) remove the fragile L2
default so robustness does not depend on calling the right entry point.

### R1 — Adaptive-α Barron via GNC *(have the pieces)*
`core/robust.py` already has the Barron general kernel (α from 2=L2 to −2=Geman-McClure) and a
GNC scheduler, but the global BA uses a fixed Cauchy. Anneal the *kernel shape* itself: start
near α=2 (convex, wide basin) and decrease α to a redescending regime as the scale tightens,
so late iterations reject the hard tail that a fixed kernel keeps partially weighted. This is
the kernel-space analogue of GNC's scale annealing.

* **Expected gain:** extends the usable outlier rate by ~10-15 percentage points; lower error
  in the 20-40 % band. **Effort:** low-medium (wire α-schedule into `ba.refine`). **Risk:** low.

### R2 — Studentized-leverage + covariance weighting *inside the global BA* *(primary)*
Studentized leverage (`robust.studentized_sq`) and covariance weighting (A2) currently live in
the per-view PnP, not in the global rig BA. A high-leverage corner (far off-axis, oblique
board) can bias the *extrinsics* while keeping its own residual small — exactly the
self-masking case studentizing was built for, but the rig BA does not yet apply it. Add the hat-
block leverage inflation and Σ⁻¹ whitening to the BA's IRLS weight.

* **Expected gain:** the per-view experiment showed 0.82°→0.16° (80 %) on a self-masking
  leverage outlier; folding it into the rig BA should cut fixed-rate extrinsic error by ≥30 %
  in leverage-heavy geometries. **Effort:** medium. **Risk:** medium (cost of the hat-block;
  use the cheap diagonal approximation already implemented).

### R3 — Robust covisibility-graph construction *(medium)*
`extrinsics.init_camera_groups` averages inter-camera transforms with
`robust_average_transform`, but the **edge weight** is just the co-observation count. A pair
whose transform samples have high robust dispersion (a few bad frames) still seeds the
extrinsic at full weight. Weight each edge by inlier count / inverse robust dispersion, and
gate hand-eye `link_groups` on its consistency residual (don't apply a link that fails the
15° gate — leave groups unlinked rather than corrupt them, the failure the non-robust default
exhibited).

* **Expected gain:** removes a class of catastrophic extrinsic failures under heavy
  contamination / spurious group splits. **Effort:** medium. **Risk:** low-medium.

### R4 — Focal-collapse anchor in the per-model front-end *(DELIVERED this iteration)*
The robust pinhole pre-calibration (RANSAC DLT) returns a reliable paraxial focal even at
15 %+ outliers (measured: 794-805 vs GT 800), but the downstream per-model Cauchy refine could
still slide into a tiny-focal local minimum that absorbs blunders into distortion. The
front-end now rejects any per-model candidate whose paraxial focal departs the robust seed by
>2× and falls back to the seed focal with neutral distortion (the BA fits distortion through
the rigid-rig constraint). The band is wide enough never to fire on a genuine fisheye, and
being per-camera + absolute it survives an all-cameras collapse the median consensus guard
cannot. Shipped in `make_bundle_front_end`.

* **Gain:** pushes the from-scratch intrinsic breakdown from ~15 % toward ≥30 %.
  **Effort:** low. **Risk:** low.

### R5 — Robust front-end as the `calibrate_rig` default *(DELIVERED this iteration)*
`calibrate_rig` previously defaulted to `_front_end_opencv` (plain L2 `cv2.calibrateCamera`),
and only the high-level `calibrate_scenario` overrode it with the robust front-end. A direct
API caller therefore silently got the fragile path, which collapsed the focal to ~30 px (vs
GT 800) under 15 % gross outliers — diverging **8/8 seeds**. The default is now the robust
from-scratch front-end (`make_bundle_front_end(RadTanModel)`).

* **Measured effect (R4+R5 together):** direct `calibrate_rig` at 15 % gross outliers went from
  **8/8 diverged (median 49 % extrinsic error)** to **0/8 diverged, median 0.92 %**; at 30 %
  outliers **0/8, median 0.64 %** — now matching the high-level path. Full rig + calib suite
  green. **Effort:** trivial. **Risk:** low.

**Axis-3 verdict:** R2 + R1 cut fixed-rate error by ≥30 % in the hard band; R3 + R4 + R5 push
the breakdown point and remove the fragile default. Medium confidence — the bar is high because
the baseline is already good, so the gains are concentrated in the 20-45 % outlier tail and in
leverage-heavy geometry.

---

## Combined roadmap

| # | Lever | Axis | Effort | Risk | Expected | Validate with |
|---|---|---|---|---|---|---|
| F1 | Parallel per-camera front-end | speed | low | low | −23..37 % wall-clock | Scenario_3/5 timing |
| F4 | Single-start seed gating | speed | low | low | a few % | per-camera calibrate count |
| F2 | Reuse BA symbolic factorization | speed | med | med | −6..13 % | BA-stage timing |
| A2 | Per-corner covariance weighting | accuracy | med | low-med | −10..25 % RMS | noisy synthetic + real |
| A1 | Free board structure in BA | accuracy | med | med | −20..40 % (reconstructed) | reconstructed vs GT-object |
| R2 | Leverage + covariance in global BA | robust | med | med | −≥30 % fixed-rate err | leverage-heavy synthetic |
| R1 | Adaptive-α GNC | robust | low-med | low | +10..15 pp breakdown | outlier-rate sweep |
| R3 | Robust covisibility graph + link gate | robust | med | low-med | removes catastrophic fails | high-outlier sweep |
| R4 | Focal-collapse anchor | robust | low | low | **done:** breakdown 15 %→≥30 % | 20-40 % outlier sweep |
| R5 | Robust front-end as default | robust | triv | low | **done:** 8/8→0/8 diverged @15 % | existing suite |

R4 and R5 are **delivered** in this iteration (rows marked *done*); the rest are planned.
Recommended sequence for the remainder (each independently shippable + validatable): **F1 → A2
→ R2 → A1 → F2 → R1 → R3 → F4**. A2's covariance plumbing is shared by R2, so do A2 before R2.

## Validation protocol (how each 30 % is proven)

* **Speed:** median wall-clock over 5 runs on Scenario_3 and Scenario_5 (5-camera rigs),
  before vs after; require ≥30 % reduction with **bit-identical extrinsics** (F1/F2/F4 must not
  change the answer). Extend `scripts/benchmark_outliers.py` with a timing harness.
* **Accuracy:** worst inter-camera baseline error vs GT on Scenario_1/3/5 (real) and the
  synthetic rig at 0.5/1.0 px noise; require ≥30 % reduction on the reconstructed-object and
  noisy cases, no regression on the GT-object case. Gate in `tests/rig/`.
* **Robustness:** the outlier-rate sweep (0-45 %, 8 seeds, 4-camera rig) plus a leverage-heavy
  variant; require either +30 % breakdown rate or −30 % median error at 20 % outliers. Add to
  `docs/RIG_OUTLIER_BENCHMARK.md` and a regression in `tests/rig/test_outlier_robustness.py`.

## Risks & non-goals

* The GT-object, low-noise case is already at 0.013 % — accuracy gains target the
  reconstructed-object and noisy regimes, not that ceiling. Reporting a 30 % cut there would be
  meaningless; the protocol above measures the regimes with real headroom.
* Per-corner covariance depends on a usable saddle-response curvature; if detection does not
  expose it cheaply, a radial/foreshortening covariance model is the fallback (most of the gain
  comes from the gross edge-vs-centre difference, which a model captures).
* Speed levers must be answer-preserving; the protocol enforces identical extrinsics so a
  "faster but slightly different" result is treated as a regression, not a win.
* Out of scope (unchanged): fully time-disjoint groups with zero shared frames (no common time
  base — MC-Calib does not solve this either), and learned / differentiable back-ends (the
  forward robust model is what transfers from diffpnp).
