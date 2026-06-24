# Unified Calibration: Calibrate Any Camera → Cross-Convert for Optimal Intrinsics → Robust Extrinsics

**Exploration & implementation report.** Spans two codebases:
`DS-MSP` (Python, multi-model intrinsics + cross-conversion) and
`MC-Calib` (C++, robust multi-camera rig). The idea: instead of "calibrate a fisheye in one
fixed model," build **one pipeline that calibrates a rig of any chosen camera models from
scratch, uses model cross-conversion to reach globally-optimal intrinsics, and recovers robust
extrinsics** — overlapping or not.

This is a vision + design report, not a committed spec. It names exactly which existing pieces
become which pipeline stage, and where the genuinely new work is.

---

## 1. Thesis

Three capabilities are usually built as three separate tools. They are more powerful fused:

1. **Calibrate-from-scratch, any model** — detect a target, initialize, nonlinear-refine the
   intrinsics of *whatever* projection model the user picks (DS, UCM, EUCM, KB, RadTan, OCam).
2. **Cross-convert for global optimality** — model families have very different
   conditioning. Some are near-convex to initialize (UCM, EUCM); some are hard (OCam's
   high-order polynomial, RadTan on wide lenses). Calibrate in the *well-conditioned* model,
   then **convert** the result into the user's chosen model — this lands the hard model's
   optimizer in the right basin, i.e. closer to the global optimum than a cold start.
3. **Robust extrinsics** — fuse many noisy per-frame relative poses across an N-camera rig
   into one consistent, outlier-resistant geometry, including cameras that never share a view.

The punchline: **(2) makes (1) reliable for any model, and (3) reuses (1)'s intrinsics across
a whole rig.** Each pillar already exists — just in different repos and not wired together.

---

## 2. The three pillars: where each lives today

| Pillar | DS-MSP (Python) | MC-Calib (C++) | fisheye-calib-adapter (C++) |
|---|---|---|---|
| **Intrinsics from scratch, any model** | ✅ `calib/bundle.py` + `core/optimize.py` (manifold LM, analytic Jacobians, all 6 models) | partial — per-model `optimize()`; DS/UCM/EUCM in progress; pinhole/ocam gaps | ❌ (conversion only, not calibration) |
| **Cross-convert intrinsics** | ✅ `adapt/convert.py` (any→any, ray-sample LSQ, multi-start) + `adapt/autoselect.py` | ❌ models imported, adapter **not** | ✅ `src/adapter.cpp:adapt` (Ceres, single-start) |
| **Robust N-camera extrinsics** | ❌ only 2-cam stereo, non-robust | ✅ 3D-object fusion, covisibility graphs, quaternion avg, hand-eye, staged global BA | ❌ |

**Observation:** no single repo has all three. DS-MSP owns pillars 1–2; MC-Calib owns pillar
3 (and is *acquiring* pillar 1 via its `model/` port from FCA). The unified tool is whichever
repo we choose to complete with the missing pillar — see §6.

---

## 3. The unified pipeline

```
            ┌───────────────────────────────────────────────────────────────┐
            │  Stage 0   Detect target(s) per camera per frame              │
            │            DS-MSP calib/detect.py (AprilGrid) | MC-Calib ChArUco│
            └───────────────────────────────────────────────────────────────┘
                                        │
            ┌───────────────────────────────────────────────────────────────┐
   per-cam  │  Stage 1   Intrinsics from scratch, CHOSEN model               │
            │   1a. detect → init → manifold-LM refine (calib/bundle.py)     │
            │   1b. IF chosen model ill-conditioned to init:                 │
            │       calibrate in a well-conditioned model, then →Stage 2     │
            └───────────────────────────────────────────────────────────────┘
                                        │
            ┌───────────────────────────────────────────────────────────────┐
            │  Stage 2   Cross-convert for GLOBAL-OPTIMAL intrinsics         │
            │   convert(easy_model → chosen_model)  (adapt/convert.py)       │
            │   → seed Stage-1 refine of chosen model on REAL correspondences│
            │   → optional multi-representation consensus (see §4)           │
            └───────────────────────────────────────────────────────────────┘
                                        │
            ┌───────────────────────────────────────────────────────────────┐
   per-rig  │  Stage 3   Robust extrinsics                                   │
            │   3a. multi-board → 3D object fusion                           │
            │   3b. robust RANSAC PnP per (object,camera,frame)              │
            │   3c. covisibility graph → camera groups                       │
            │   3d. quaternion avg + translation median                     │
            │   3e. hand-eye link for non-overlapping groups                │
            └───────────────────────────────────────────────────────────────┘
                                        │
            ┌───────────────────────────────────────────────────────────────┐
            │  Stage 4   Joint global BA                                     │
            │   {intrinsics, extrinsics, object poses, board-in-object poses}│
            │   staged, reference-anchored, Huber, sparse Schur             │
            └───────────────────────────────────────────────────────────────┘
```

Stages 0, 1 and the building blocks of 3–4 already exist (DS-MSP single-cam; MC-Calib rig).
The genuinely new glue is **Stage 2 as a first-class stage** and the **N-camera extension of
Stage 4** in whichever repo hosts the tool.

---

## 4. "Cross-convert for global optimality of intrinsics" — what it concretely means

This is the novel core of your idea. Two designs, increasing ambition.

### Design A — conversion as initialization bridge (low-risk, high-value)
A model's optimizer only finds the global optimum if its *initial guess* is in the right
basin. Conversion gives a physically-consistent full parameter vector for free:

1. Calibrate in a **well-conditioned source** model on the real images. UCM/EUCM/DS have a
   1–2-DOF closed-form seed (linear LS for `alpha`/`beta`) that lands in-basin from almost
   anything; KB has a clean linear `(θ, r_u)` fit. (DS-MSP `models/*.initialize_from_correspondences`.)
2. `convert(source, ChosenModelClass)` (`adapt/convert.py`) — sample pixel grid → unproject
   under source → re-fit chosen model. This yields a globally-consistent guess for the *hard*
   model (OCam's 5-term polynomial, RadTan's cross-coupled tangential terms).
3. Re-run the chosen model's refine on the **real correspondences**, seeded from (2). Now the
   hard optimizer starts in the right basin instead of from a cold default.

DS-MSP's `convert` is structurally the same LM+analytic-Jacobian solver as `calibrate`, so the
seed is directly compatible. Its **multi-start** (`adapt/convert.py:_shape_seeds`) is the
concrete global-optimum mechanism: disperse only the shape params (intrinsics held at their
closed-form optimum), keep the lowest-cost basin.

### Design B — multi-representation consensus (research-grade)
Calibrate the *same camera* in several models simultaneously and enforce that they agree on
the underlying ray field:
- Each model `m` has params `θ_m`. Define a consensus loss over a shared dense pixel grid:
  for every pixel `u`, all models should unproject to the same bearing `r(u)`.
  `L = Σ_m Σ_u ‖unproject_m(u; θ_m) − r̄(u)‖²` with `r̄` the (robust) consensus ray.
- Refine all `θ_m` jointly against (i) the real reprojection residuals **and** (ii) the
  consensus term. The chosen model inherits "global optimality" in the sense that its
  parameters are pulled toward the ray field that *every* representable model agrees on,
  damping its own ill-conditioning. `adapt/autoselect.py:convert_best` already encodes the
  representability theory (a model with fewer shape DOF carries a positive global-optimum
  residual) — Design B operationalizes it as a regularizer.

**Recommendation:** ship Design A first (it's a wiring of existing parts and immediately
improves OCam/RadTan reliability); treat Design B as a research chapter with a clear
ablation (does consensus reduce reprojection variance on held-out frames?).

### Representability caveats (must surface to the user)
Conversions *among* the wide-FOV sphere/poly family (DS↔UCM↔EUCM↔KB↔OCam) are near-exact
(DS-MSP measured EUCM 0.014 px, KB 0.0002 px, OCam 0.54 px). Conversions **into RadTan/pinhole
from a >180° lens are lossy** (need `max_fov_deg`; diverge at the edge otherwise), and **into
UCM** (1 shape DOF) leaves a structural residual. The pipeline must report
`fov_covered_deg` + RMS per conversion and refuse silently-lossy targets (use
`adapt/evaluate.py:reprojection_report`).

---

## 5. Robust extrinsics — reuse, don't reinvent

Stage 3–4 are MC-Calib's existing strengths (see `MC-Calib/docs/` and the rig plan). The
borrowable set, already mapped: 3D-object fusion, covisibility graphs with `1/co-obs` edge
weights + shortest-path chaining, quaternion (Markley) rotation averaging + translation
median, RANSAC PnP with inlier pruning, Tsai-bootstrap hand-eye for non-overlapping cameras,
reference-anchored staged Schur BA, merge↔refine iteration. The key property: **all of it is
model-agnostic** — it composes poses and calls `project`/`unproject`. So it works for *any*
chosen camera model the moment that model exposes projection — which is exactly Stage 1's
output. Pillars 1 and 3 connect through the model interface, nothing else.

---

## 6. Which codebase hosts the unified tool?

Two viable hosts; they imply different work.

### Option 1 — Complete **MC-Calib** (C++) — *recommended for production*
MC-Calib is closest: it already has the entire robust rig (pillar 3) **and** is acquiring the
multi-model layer (pillar 1) via the `model/` port from FCA. Missing: a finished pillar 1
(DS/UCM/EUCM PnP-init fix, pinhole/ocam gaps — see
`MC-Calib/docs/CAMERA_MODEL_IMPLEMENTATION_GUIDE.md`) and pillar 2 (import FCA's `adapt()` —
the C++ conversion already exists in the sibling repo, just not pulled in).
- **Pros:** production C++/Ceres performance; the hard rig already works; the FCA adapter is
  C++ and importable.
- **Cons:** finishing all 6 models in C++ (analytic Jacobians, BA template + 4-ladder wiring
  per model) is more friction than DS-MSP's polymorphic Python; Design B is harder to
  prototype.
- **Net:** lowest total work to a *production* "any-model rig calibrator," because pillar 3 (the
  expensive part) is done and pillar 2's code exists next door.

### Option 2 — Complete **DS-MSP** (Python) — *recommended for research / fast iteration*
DS-MSP owns pillars 1–2 cleanly (all 6 models polymorphic, cross-convert validated,
multi-start). Missing: pillar 3 (the rig — see `DS-MSP/docs/RIG_CALIBRATION_PLAN.md`).
- **Pros:** model layer + conversion are already excellent and uniform; Python makes Design B
  (consensus) and ablations fast; analytic Jacobians already in place; reuses `schur_lm`.
- **Cons:** must build the rig machinery from scratch (graphs, averaging, hand-eye, multi-cam
  BA) — well-scoped but real; Python BA at large rig scale is slower than Ceres.
- **Net:** best for *exploring* the unified idea (especially Stage 2 Design B) before
  committing to C++.

### Recommendation
**Prototype the unified pipeline in DS-MSP (Option 2), then port the proven Stage-2 conversion
strategy and any new init logic into MC-Calib (Option 1) for production.** Rationale: Stage 2
(cross-convert for optimality) is the unproven, research-y core — iterate it in Python where
the model layer + adapter already live and Design B is cheap to test. Pillar 3 is *already
production-grade in MC-Calib*, so the eventual production tool is MC-Calib with (a) its models
finished and (b) FCA's adapter imported. The two repos stay complementary: DS-MSP = the
algorithm lab and source of the conversion/init strategy; MC-Calib = the production rig that
consumes it. This also matches the current trajectory (models are actively being added to
MC-Calib; conversion is actively studied in DS-MSP — see `DS-MSP/reports/wide_angle_model_conversion_study`).

---

## 7. Phased plan (cross-repo)

1. **P1 — Stage 2 in DS-MSP (Design A).** Wire `adapt/convert.py` as an init bridge for
   `calib/bundle.py`: `calibrate(model=hard, init="convert-from", source=easy)`. Benchmark
   OCam/RadTan reliability vs cold start on the existing TUM-VI / Blender data. *Pure wiring of
   existing parts.*
2. **P2 — DS-MSP rig, Phase 1–2** (`RIG_CALIBRATION_PLAN.md`): N-camera overlapping +
   multi-board 3D objects + multi-cam BA. Validate against MC-Calib's Blender `Scenario_2–5`
   (4–5 cam rigs with ground truth).
3. **P3 — Finish MC-Calib models** (`CAMERA_MODEL_IMPLEMENTATION_GUIDE.md` §4): DS/UCM/EUCM
   PnP-init via `solvePose`; pinhole `optimize/initialize`; OcamCalib BA wiring; fix the
   `parse()` path bug; per-model end-to-end tests.
4. **P4 — Import the adapter into MC-Calib** (pillar 2): pull FCA's `adapt()` into MC-Calib so
   Stage 2 runs in the production tool; expose `calibrate --model X --init-via Y`.
5. **P5 — Stage 2 Design B (research).** Multi-representation consensus in DS-MSP; ablate;
   write the `docs/learn/` chapter; port if it wins.
6. **P6 — Non-overlapping rigs** end-to-end in the production host (hand-eye + merge↔refine).

---

## 8. Open questions

- **Stage-2 ordering:** convert *before* or *after* the rig BA? Converting per-camera before
  Stage 3 gives all cameras the chosen model up front; converting after lets the rig BA settle
  geometry first. Likely: convert for *init*, then let the joint BA refine the chosen model's
  params directly (so conversion never has the final say).
- **Consensus weighting (Design B):** how to weight the consensus term vs real reprojection so
  it regularizes without biasing away from the data. Needs the held-out-frame ablation.
- **Target unification:** DS-MSP uses AprilGrid, MC-Calib uses ChArUco. The unified tool needs
  one detector abstraction (or both), since the 3D-object fusion logic is target-agnostic but
  the detector is not.
- **Degenerate/lossy conversions:** define the hard refusal policy (FOV coverage + RMS
  thresholds) so the tool never silently returns a model that can't represent the lens.

---

## 9. One-line summary

DS-MSP already calibrates any model and converts between them; MC-Calib already builds robust
N-camera geometry. The unified tool is **"calibrate in the model that initializes well,
cross-convert into the model you want, then solve the whole rig jointly"** — built by adding
Stage-2-as-a-stage and the missing pillar to whichever repo we pick (prototype in DS-MSP,
productionize in MC-Calib). Companion docs:
`MC-Calib/docs/CAMERA_MODEL_IMPLEMENTATION_GUIDE.md` and
`DS-MSP/docs/RIG_CALIBRATION_PLAN.md`.
