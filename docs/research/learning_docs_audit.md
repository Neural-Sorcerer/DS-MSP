# Learning & documentation audit — 2026-06-21

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

Compared three things, feature by feature:
1. **Code shipped** — `ds_msp/` modules (public API + module docstrings).
2. **Runnable example** — `examples/NN_*.py` (the curriculum's "prints a verifiable number" contract).
3. **Learning doc** — a chapter or deep-dive in `docs/learn/`.

A feature is **documented** only when all three exist. Research notes in `docs/research/`
(the Tier-1 spec, the diffpnp survey) are **not** counted — they're design records, not the
teaching layer.

---

## Coverage matrix

| Feature (code shipped) | Module | Example | Learn doc | Status |
|---|---|---|---|---|
| Fisheye / camera models | `models/`, `model.py` | `01` | Ch.1 | ✅ complete |
| Double Sphere model | `models/ds_math.py` | `02` | Ch.2 | ✅ complete |
| Projection validity / >180° cone | `models/double_sphere.py` | `07` | Ch.3 | ✅ complete |
| Calibration capstone (AprilGrid → BA) | `calib/` | `03` | capstone + deep-dive | ✅ complete |
| Robust loss / IRLS | `core/robust.py` | `04` | deep-dive | ✅ complete |
| Model equivalence | `adapt/` | `05` | deep-dive | ✅ complete |
| Stereo extrinsics | `calib/stereo.py` | `06` | — (roadmap only) | 🟡 example, no chapter |
| Chart reprojection (sphere/cyl) | `ops/reproject.py` | `08` | deep-dive | 🟡 concept covered; doc predates the **library** module + cubemap/tangent |
| **C1 · two-view geometry on rays** | `mvg/two_view.py` | — | — | ❌ none |
| **C2 · robust relative pose (RANSAC)** | `mvg/ransac.py` | — | — | ❌ none |
| **C4 · sphere-sweep stereo (depth)** | `stereo/sphere_sweep.py` | — | — | ❌ none |
| **C5 · angular reprojection BA** | `mvg/bundle.py` | — | — | ❌ none |
| **C6 · spherical rectification** | `stereo/rectify.py` | — | — | ❌ none |
| **`estimate_relative_pose` end-to-end** | `mvg/` | — | — | ❌ none |
| **Phase 1 · manifold pose opt (Lie)** | `core/lie.py` | — | — | ❌ none |
| **Phase 2 · in-house manifold LM** | `core/optimize.py` | — | — | ❌ none |
| **Schur-complement sparse BA** | `calib/bundle.py`, `core/` | — | — | ❌ none |

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

**Gap B — Tier-1 is invisible and undocumented.** C1–C6 + the manifold-optimization refactor
(Lie / in-house LM / sparse BA) have no chapters, no examples, and **no mention anywhere in
the curriculum's navigation** (README path diagram, chapter table, mermaid graph all stop at
the capstone).

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

### Navigation fixes (cheap, do regardless)
- README: split the chapter table into Part I / Part II; add the Tier-1 rows.
- README + ROADMAP: extend the mermaid "path" graph past the capstone into Part II.
- Add a one-line "stereo extrinsics" chapter or fold `06` into Ch.10's neighborhood (it
  currently has an example but no prose home).

---

## Recommended sequence

1. **Navigation first** (hours) — make Tier-1 *visible* in the README/ROADMAP even before the
   prose exists. Cheapest credibility win; stops the repo from understating itself.
2. **Ch.8 + example `09`** (two-view geometry) — highest portfolio leverage, and the entry
   point to Part II that everything else builds on.
3. **Ch.9** (manifold optimization) — the most distinctive "I built the hard thing myself"
   story (in-house LM replacing scipy); memory already records the findings to draw from.
4. Then Ch.11/12 (stereo depth), Ch.10 (scalable BA), Ch.13 (charts upgrade).
5. Backfill Gap A (Ch.4–7) opportunistically — lower novelty, but they're promised and the
   code already exists.
