# DS-MSP — Multi-Camera Rig Calibration Plan (borrowing from MC-Calib)

**What this is:** a gap analysis of DS-MSP against MC-Calib's robust multi-camera rig
calibrator, and a concrete, file-by-file plan for what to add where so DS-MSP can calibrate
an **N-camera rig** (overlapping *and* non-overlapping FOV) — not just single cameras and
two-view stereo.

DS-MSP today is an excellent **single-camera, multi-model** library plus a **two-view** MVG
stack. It already has the hard parts MC-Calib lacks (clean any-model intrinsics from scratch,
analytic Jacobians, cross-conversion). What it lacks is everything that turns *many cameras +
many boards over many frames* into one consistent rig. That machinery is exactly MC-Calib's
core, and it is **model-agnostic** — so it grafts cleanly onto DS-MSP's existing model layer.

---

## 1. Capability gap (DS-MSP vs MC-Calib)

| Capability | DS-MSP today | MC-Calib | Gap to close |
|---|---|---|---|
| Single-cam intrinsics, any model, from scratch | ✅ `calib/bundle.py` + `core/optimize.py` | partial (per-model) | — (DS-MSP ahead) |
| Cross-model conversion | ✅ `adapt/convert.py` | ❌ (not imported) | — (DS-MSP ahead) |
| Multi-board → one rigid "3D object" | ❌ single planar AprilGrid only | ✅ `Object3D` fusion | **add** |
| Co-visibility graph + connected components + shortest-path chaining | ❌ | ✅ boards/cameras/groups | **add** |
| Robust rotation/translation averaging | ❌ (stereo uses SVD-mean + median, non-robust) | ✅ quaternion (Markley) avg + translation median | **add** |
| RANSAC PnP extrinsics with inlier pruning | partial (RANSAC only for 2-view relative pose) | ✅ `ransacP3PDistortion` + validity flag | **add** |
| Hand-eye for non-overlapping cameras | ❌ | ✅ Tsai-bootstrap / Horaud | **add** |
| Global BA over {intrinsics, extrinsics, object poses, board poses} | ❌ (single-cam Schur only) | ✅ `UniversalReprojectionError` staged BA | **extend** |
| N>2 cameras | ❌ (stereo = exactly 2) | ✅ | **add** |

**Bottom line:** DS-MSP's `calib/stereo.py:estimate_relative_pose` is the *only* cross-camera
primitive — two cameras, same board same frame, non-robust SVD-mean rotation + median
translation, no joint refine. Closing the gap means a new `ds_msp/rig/` package plus a
multi-camera extension to the existing optimizer.

---

## 2. The 8 borrowable concepts (ranked by leverage)

From MC-Calib, in priority order for DS-MSP:

1. **Multi-board "3D object" fusion** — rigidly stitch several planar targets into one
   disconnected point cloud via shortest-path-composed averaged inter-board transforms, so
   any subset of boards yields a full target pose. *Single biggest co-visibility multiplier.*
   (MC-Calib `init3DObjects`, `Object3D.cpp`.)
2. **Co-visibility graphs + connected components + shortest-path chaining** at three levels
   (boards, cameras, groups), edge weight = `1 / co-observation-count`, to relate elements
   that never directly co-observe. (`initInterCamerasGraph`, `initCameraGroup`.)
3. **Quaternion (Markley/SVD) rotation averaging + translation median** to fuse many noisy
   per-frame relative poses into one stable transform. (`getAverageRotation`,
   `initInterTransform`.)
4. **Reference-anchored staged global BA** — hold ref-camera and ref-board poses constant;
   warm-start poses-only and intrinsics-fixed passes before the full joint solve, all under
   Huber loss. (`CameraGroup::refine*` family.)
5. **RANSAC PnP with model-aware undistortion + post-fit inlier pruning + validity flag.**
   (`ransacP3PDistortion`, `BoardObs::estimatePose`.)
6. **Bootstrap/cluster hand-eye for non-overlapping rigs** — k-means on motion translations
   to pick diverse motions, repeated Tsai solves, 15°-consistency gating, median/averaged
   aggregation. (`handeyeBootstraptTranslationCalibration`.)
7. **Object-in-rig pose averaging when the reference camera doesn't see the target** — average
   across the non-ref cameras that do. (`CameraGroupObs::computeObjectsPose`.)
8. **Iterate merge ↔ refine** — alternate combinatorial re-grouping with continuous
   refinement to escape bad initial groupings.

---

## 3. Implementation plan — what to add where

### 3.1 New package: `ds_msp/rig/`

```
ds_msp/rig/
  target.py        # multi-board target = list of planar boards in a shared frame
  object3d.py      # fused 3D object: board-in-object poses + concatenated point cloud
  graph.py         # covisibility graph, connected components, shortest-path chaining
  averaging.py     # quaternion (Markley SVD) rotation avg + translation median
  pose_init.py     # robust RANSAC PnP per (board|object, camera, frame) + inlier prune
  handeye.py       # Tsai-bootstrap hand-eye for non-overlapping camera groups
  rig_calibrate.py # the N-camera orchestrator (the McCalib.cpp analogue)
```

**Reuse what already exists — do not rebuild:**
- `ds_msp/models/*` — every model already has `project`/`unproject`/`project_jacobian`.
  The rig code is model-agnostic; it calls these.
- `ds_msp/core/optimize.py:schur_lm` / `lm_solve` — the manifold LM/BA engine. Extend it
  (§3.3) rather than writing a new optimizer.
- `ds_msp/core/lie.py` (`so3_exp`, manifold re-basing) — reuse for all pose updates.
- `ds_msp/mvg/ransac.py` (`ransac_relative_pose`, angular Sampson) — the RANSAC harness to
  adapt for PnP.
- `ds_msp/calib/detect.py` + `targets.py` (AprilGrid) — the detector; generalize `targets.py`
  to multiple boards (§3.2).
- `ds_msp/calib/stereo.py:estimate_relative_pose` — keep as the 2-camera fast path; the new
  rig path generalizes it.

### 3.2 Multi-board target & 3D-object fusion — `rig/target.py`, `rig/object3d.py`

DS-MSP's `calib/targets.py:AprilGridTarget` is a single planar board (`z=0`). Generalize:

- **`rig/target.py`** — a `MultiBoardTarget` holding a list of boards, each with its own tag
  id range so a detection maps unambiguously to (board_id, corner). Mirror MC-Calib's
  `BoardObs` remapping (`Object3DObs::insertNewBoardObs`).
- **`rig/object3d.py`** — `build_object3d(board_observations)`:
  1. Build the inter-board covisibility graph (boards seen together in any frame), edge
     weight `1/count` (`graph.py`).
  2. Per connected component, pick the min-id board as reference; each other board's
     pose-in-object = shortest-path composition of **averaged** inter-board transforms
     (`averaging.py`).
  3. Transform each board's corner points into the object frame and concatenate into one
     `pts_3d` cloud, keeping `pts_board↔obj` index maps.
  - This is the direct analogue of MC-Calib `init3DObjects` / `Object3D`. One PnP then
    recovers the whole object pose from any visible subset of boards.

### 3.3 Multi-camera global BA — extend `core/optimize.py`

Today `schur_lm` solves *one camera's intrinsics + that camera's per-image poses* (arrowhead
with a single intrinsic block + per-frame pose blocks). The rig needs a **wider arrowhead**:

Parameter blocks (mirror MC-Calib's `UniversalReprojectionError`, which composes
board→object→camera→project):
1. **Per-camera extrinsic** `T_cam_in_group` (6-DoF), ref camera fixed.
2. **Per-frame object pose** `T_object_in_group` (6-DoF).
3. **Per-board-in-object pose** `T_board_in_object` (6-DoF), ref board fixed.
4. **Per-camera intrinsics** (model param vector), optionally fixed (`fix_intrinsics` flag).

Residual = reproject object point through (3)∘(2)∘(1)∘`model.project`, Huber-robust, using
each model's analytic `project_jacobian` plus pose Jacobians from `core/lie.py`. Implement as
a new `core/optimize.py:rig_schur_lm` (or a `bundle`-style assembler in `rig/rig_calibrate.py`
that feeds the existing sparse solver). Keep the **staged** structure MC-Calib uses:
extrinsics+poses first (intrinsics fixed), then the full joint solve when `fix_intrinsics=False`.

### 3.4 Robust averaging — `rig/averaging.py`

Replace `calib/stereo.py`'s SVD-mean rotation. Implement:
- `average_rotation_quaternion(Rs)` — Markley SVD method: stack quaternions, fix antipodal
  signs, take the leading eigenvector of `Σ qqᵀ`. (MC-Calib `getAverageRotation`.)
- `average_translation(ts)` — per-component median (outlier-resistant). (`initInterTransform`.)

### 3.5 Robust PnP extrinsics — `rig/pose_init.py`

`estimate_pose_ransac(model, object_pts, image_pts, thresh_px)`:
- unproject pixels with the camera's model → bearing rays → PnP on the normalized plane
  (reuse the `bundle._seed_poses` pattern: `model.unproject` + `cv2.solvePnP`), wrapped in a
  RANSAC loop adapted from `mvg/ransac.py`;
- prune to inliers, recompute pose, flag invalid below a min-inlier count (MC-Calib
  `BoardObs::estimatePose` invalidates below 4). Feed only inliers downstream.

### 3.6 Hand-eye for non-overlapping cameras — `rig/handeye.py`

For camera groups that share no object view, port
`handeyeBootstraptTranslationCalibration`: collect paired absolute object poses per group,
k-means cluster translations to pick diverse motions, run repeated `cv2.calibrateHandEye`
(Tsai), gate solutions by a 15° rotational-consistency check, aggregate by median translation
+ averaged rotation. (OpenCV's `calibrateHandEye` is available in Python — minimal new math.)

### 3.7 Orchestrator — `rig/rig_calibrate.py`

The `McCalib.cpp` analogue, in order:
1. detect (per camera, per frame) → board observations (`calib/detect.py`).
2. per-camera intrinsics from scratch (`calib/bundle.py:calibrate`) **or** seed via
   cross-convert (`adapt/convert.py`) — see the unified vision doc.
3. build 3D objects (`rig/object3d.py`).
4. per-(object,camera,frame) robust pose (`rig/pose_init.py`).
5. inter-camera relative-pose hypotheses → averaging → covisibility graph → connected
   components = camera groups (`rig/graph.py`, `rig/averaging.py`).
6. link non-overlapping groups via hand-eye (`rig/handeye.py`).
7. staged global BA (`core/optimize.py:rig_schur_lm`).
8. report per-camera reprojection RMS + write extrinsics (extend `io/kalibr.py`, which
   already *reads* `T_cn_cnm1` — now also *write* what we estimate).

---

## 4. Phased roadmap

- **Phase 1 — N-camera, overlapping FOV, single board.** `rig/{graph,averaging,pose_init}.py`
  + multi-camera BA (§3.3) with one planar board. Generalizes `calib/stereo.py` to N cameras.
  *Smallest useful increment; unlocks N>2 immediately.*
- **Phase 2 — multi-board 3D objects.** `rig/{target,object3d}.py`. Boosts co-visibility,
  enables larger rigs and partial views.
- **Phase 3 — non-overlapping FOV.** `rig/handeye.py` + group merging + merge↔refine loop.
- **Phase 4 — cross-convert init bridge.** Wire `adapt/convert.py` as the intrinsics
  initializer for hard models (unified vision doc, §"calibrate any camera").

Each phase lands with tests against synthetic rigs (you can reuse MC-Calib's Blender
`Scenario_2`–`Scenario_5`, which are 4–5 camera rigs with ground-truth poses — read them via a
small loader) and a `docs/learn/` chapter, per DS-MSP's library+curriculum principle.

---

## 5. Why this grafts cleanly

The rig math never touches projection internals — it composes poses and calls
`model.project` / `model.unproject` / `model.project_jacobian`. DS-MSP's model layer already
exposes all three for all six models with analytic derivatives. So the entire borrowed
pipeline is **model-agnostic by construction**, exactly as it is in MC-Calib (whose
`UniversalReprojectionError` dispatches on model name but shares one rig path). Adding the rig
does not perturb the existing single-camera or adapter code.

See **`UNIFIED_CALIBRATION_VISION.md`** for how the rig (this doc), the from-scratch
any-model intrinsics, and cross-conversion combine into a single "calibrate any camera,
cross-convert for optimal intrinsics, robust extrinsics" tool.
