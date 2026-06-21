# Learning & documentation audit — 2026-06-21

> **Refresh — 2026-06-21 (later same day).** Since the original audit: (1) the README/ROADMAP
> **navigation fixes shipped** (Part I/II split, Tier-1 made visible) — the "Navigation fixes"
> section below is now **done**; (2) **stereo extrinsics is complete** — `stereo_extrinsics_calibration.md`
> shipped through the full doc pipeline (example `06` + chapter + a math-asserted figure),
> taking it 🟡→✅; (3) a **fourth coverage dimension — Figure — is now tracked** (the pipeline gained
> a `doc-illustrator` stage that owns figures). Adding it surfaces a new gap: three deep-dives carry
> no figure of their own. The Tier-1 MVG / stereo-depth / manifold stack remains undocumented — that
> is still the headline gap.

**Question:** are the learning docs (`docs/learn/`) up to date with the features that have
shipped, or lagging?

**Verdict:** **lagging, in two layers.** The last commit to touch `docs/learn/` was
`620b066` (AprilGrid deep-dive). Since then **11 feature commits** have landed — the entire
Tier-1 multi-view-geometry / stereo / manifold-optimization stack — with **zero** learning
material and **zero** runnable examples. The teaching layer's own promise ("every chapter
pairs an explainer with a script that prints a number you can verify") is currently unmet
for everything built after the capstone.

---

## Method & scope

Compared four things, feature by feature:
1. **Code shipped** — `ds_msp/` modules (public API + module docstrings).
2. **Runnable example** — `examples/NN_*.py` (the curriculum's "prints a verifiable number" contract).
3. **Learning doc** — a chapter or deep-dive in `docs/learn/`.
4. **Figure** — a real-data, reproducible visual *embedded in that doc* (the `doc-illustrator`
   stage's deliverable). A figure that lives in `assets/` but isn't embedded in the feature's own
   doc doesn't count for that feature.

A feature is **fully documented** only when code + example + doc all exist; the **figure** column
is tracked separately because a chapter without a figure is still a valid chapter (some concepts
need no visual), but a missing-yet-warranted figure is a real gap the pipeline can now close.
Research notes in `docs/research/` (the Tier-1 spec, the diffpnp survey) are **not** counted —
they're design records, not the teaching layer.

---

## Coverage matrix

Legend: ✅ present · ❌ missing · — n/a. "Figure" = embedded in the feature's own doc.

| Feature (code shipped) | Module | Example | Learn doc | Figure | Status |
|---|---|---|---|---|---|
| Fisheye / camera models | `models/`, `model.py` | `01` | Ch.1 | ✅ `undistort_demo.gif` | ✅ complete |
| Double Sphere model | `models/ds_math.py` | `02` | Ch.2 | ✅ pipeline + projection | ✅ complete |
| Projection validity / >180° cone | `models/double_sphere.py` | `07` | Ch.3 | ✅ fov/coverage | ✅ complete |
| Calibration capstone (AprilGrid → BA) | `calib/` | `03` | capstone + deep-dive | ✅ aprilgrid + reproj | ✅ complete |
| AprilGrid detection deep-dive | `calib/detect.py` | `03` | `robust_aprilgrid_detection.md` | ❌ none in this doc | 🟡 doc ok; **no own figure** |
| Robust loss / IRLS | `core/robust.py` | `04` | `robust_losses_and_evaluation.md` | ❌ none | 🟡 doc ok; **no figure** |
| Model equivalence | `adapt/` | `05` | `are_two_models_the_same_camera.md` | ❌ none | 🟡 doc ok; **no figure** |
| **Stereo extrinsics** | `calib/stereo.py` | `06` | `stereo_extrinsics_calibration.md` | ✅ `stereo_extrinsics_invariance.gif` | ✅ **complete (new)** |
| Chart reprojection (sphere/cyl) | `ops/reproject.py` | `08` | deep-dive | ✅ morph + corners | 🟡 doc predates the **library** module + cubemap/tangent |
| **C1 · two-view geometry on rays** | `mvg/two_view.py` | — | — | — | ❌ none |
| **C2 · robust relative pose (RANSAC)** | `mvg/ransac.py` | — | — | — | ❌ none |
| **C4 · sphere-sweep stereo (depth)** | `stereo/sphere_sweep.py` | — | — | — | ❌ none |
| **C5 · angular reprojection BA** | `mvg/bundle.py` | — | — | — | ❌ none |
| **C6 · spherical rectification** | `stereo/rectify.py` | — | — | — | ❌ none |
| **`estimate_relative_pose` end-to-end** | `mvg/two_view.py` | — | — | — | ❌ none |
| **Phase 1 · manifold pose opt (Lie)** | `core/lie.py` | — | — | — | ❌ none |
| **Phase 2 · in-house manifold LM** | `core/optimize.py` | — | — | — | ❌ none |
| **Schur-complement sparse BA** | `calib/bundle.py`, `core/` | — | — | — | ❌ none |

---

## The deeper finding

It is not just "chapters are missing." **The examples directory stops at `08`.** Every
Tier-1 capability ships with passing unit tests but **no runnable, real-data script** — so
even a reader willing to skip prose has nothing to *run*. For a portfolio whose explicit
thesis is "in 3D vision you measure that your math is right, you don't hope," the most
advanced work currently has no measured-number artifact a visitor can see.

This is also where the **portfolio value is highest**: two-view pose, stereo depth, and
manifold (Lie) optimization are exactly the SLAM/SfM geometry that 3D-vision roles screen
for. Right now a recruiter reading the repo sees a calibration library and would never learn
that stack exists — the README "path" diagram and chapter table end at the capstone.

---

## Gap categories

**Gap A — old promises unkept.** The README chapter table still lists four chapters as
*"coming soon"*: Ch.4 (analytic Jacobians), Ch.5 (LM calibration), Ch.6 (conversion),
Ch.7 (reproducing a published calibration). All four have code anchors that already exist;
only the write-ups are missing.

**Gap B — Tier-1 is undocumented** (*visibility half now fixed*). C1/C2 (two-view + RANSAC),
C4 (sphere-sweep), C5 (angular BA), C6 (rectification), the `mvg` `estimate_relative_pose`, and
the manifold-optimization refactor (Lie / in-house LM / sparse BA) still have **no chapters and
no examples** (the examples directory stops at `08`). The *invisibility* sub-problem is resolved —
the README/ROADMAP now split into Part I / Part II and surface the Tier-1 arc — but the teaching
content itself is the open work. This is the **highest-leverage gap**: two-view pose, stereo depth,
and Lie optimization are exactly the SLAM/SfM geometry 3D-vision roles screen for.

**Gap D — figure coverage** *(new, from the 4th column)*. Three otherwise-complete deep-dives carry
**no figure of their own**: `robust_losses_and_evaluation.md` (a loss-curve / IRLS-weight visual is
the obvious candidate), `are_two_models_the_same_camera.md` (an overlay of the two models' projections
agreeing), and `robust_aprilgrid_detection.md` (the AprilGrid GIF lives in the capstone, not here).
None blocks the chapter, but each is a cheap, high-value `doc-illustrator` job. Apply the device rule —
request an animation only where the *transition is the lesson*, else a static figure / small-multiple.

**Gap C — partial/stale.** `spherical_and_cylindrical_reprojection.md` (the C3 deep-dive)
predates the C3 *library* (`ops/reproject.py`) and covers only sphere+cylinder; the shipped
module also does cubemap + tangent images. The doc should be re-pointed at the library and
extended.

---

## Proposed curriculum restructure

The current track is one linear path that dead-ends at the capstone. Tier-1 is a *second
arc* — "from one calibrated camera to 3D structure." Proposal: keep the existing track as
**Part I — Calibration**, add **Part II — Geometry & 3D**, and surface both in the README.

### Part I — Calibration (mostly done; backfill Gap A)
- Ch.1–3, capstone, deep-dives — ✅ already shipped.
- Ch.4 Analytic Jacobians vs autodiff *(backfill)* → anchor `model.py`, gradient-check.
- Ch.5 Calibration by Levenberg–Marquardt *(backfill)* → anchor `calib/`, `calibrate.py`.
- Ch.6 Model conversion *(backfill)* → anchor `adapt/`.
- Ch.7 Reproducing a published calibration *(backfill)* → anchor `io/kalibr.py`.

### Part II — Geometry & 3D (new; closes Gap B) — each needs a chapter **and** an `examples/NN`
- **Ch.8 Rays, not pixels: two-view geometry on bearing vectors** (C1/C2) → essential
  matrix on rays, pose recovery, ray triangulation, RANSAC. Verifiable number: recovered
  pose vs ground-truth angle on a synthetic/real pair. Example `09`.
- **Ch.9 Optimizing on the manifold: SO(3)/SE(3), Lie, and the in-house LM solver**
  (Phase 1+2) → why flat parameterization is a correctness bug, re-basing LM. Ties to the
  existing `ds-msp-lie-vs-flat-finding` and `ds-msp-inhouse-lm-solver` notes. Example `10`.
- **Ch.10 Bundle adjustment that scales: angular residual + Schur complement** (C5 + sparse
  BA) → angular reprojection error, sparse normal equations. Example `11`.
- **Ch.11 Depth without rectifying: sphere-sweep stereo** (C4) → dense depth on raw fisheye.
  Verifiable number: recovered depth vs known synthetic scene. Example `12`.
- **Ch.12 Spherical epipolar rectification** (C6) → the pedagogically-clean complement to
  Ch.11; depth agrees with sphere-sweep to <1%. Example `13`.
- **Ch.13 Charts as front-ends** (C3, upgrade Gap C) → re-point the existing deep-dive at
  `ops/reproject.py`, add cubemap + tangent images.

### Navigation fixes — ✅ DONE (2026-06-21)
- ✅ README split into Part I / Part II with Tier-1 rows (commit `138f0d8`).
- ✅ README + ROADMAP extended past the capstone into Part II.
- ✅ Stereo extrinsics now has a prose home — `stereo_extrinsics_calibration.md`, linked from the
  learn README deep-dives block. (`06` is no longer an orphaned example.)

---

## Recommended sequence

0. ~~**Navigation first**~~ — ✅ **done** (Part I/II split shipped).
0b. ~~**Stereo extrinsics**~~ — ✅ **done** (chapter + figure shipped through the full pipeline).
1. **Ch.8 + example `09`** (two-view geometry) — **next.** Highest portfolio leverage, and the
   entry point to Part II that everything else builds on. Anchor `mvg/two_view.py` + `mvg/ransac.py`.
2. **Ch.9** (manifold optimization) — the most distinctive "I built the hard thing myself"
   story (in-house LM replacing scipy); memory already records the findings to draw from.
3. Then Ch.11/12 (stereo depth), Ch.10 (scalable BA), Ch.13 (charts upgrade, closes Gap C).
4. Backfill Gap A (Ch.4–7) opportunistically — lower novelty, but they're promised and the
   code already exists.
5. **Gap D figures** (cheap, parallelizable) — backfill the three figure-less deep-dives whenever
   convenient; each is a single `doc-illustrator` run.
