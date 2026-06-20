# Findings record — image-domain representations for 3D tasks (stereo / SfM / reconstruction)

> **Provenance.** Multi-source deep-research run (2026-06-20): 6 search angles, 27 primary
> sources fetched, 127 candidate claims extracted, top 25 put through **3-vote adversarial
> verification** (a claim needs 2/3 *refutes* to be killed). Result: **24 confirmed, 1 killed**.
> This file is the faithful record. The engineering translation lives in
> [`tier1_implementation_spec.md`](tier1_implementation_spec.md); the roadmap integration in
> [`../ROADMAP.md`](../ROADMAP.md).
>
> Vote notation: `3-0` = all three verifiers confirmed; `0-3` = all three refuted (killed).

---

## Confirmed findings

### F1 — The pinhole chart is the gnomonic tangent plane, and it dies at 180° · `3-0`
The rectilinear/perspective chart is the **gnomonic** projection: a ray at incidence angle
`θ` maps to image radius `r ∝ tan θ`. It keeps straight 3D lines straight, which is exactly
why classical stereo/SfM get **straight epipolar lines**, a **3×3 homography** rectification,
and **disparity linear in inverse depth**. But `tan θ → ∞` at 90°, so a single pinhole image
**cannot represent FOV ≥ 180°** — the reason wide-FOV content needs spherical/ERP/cylindrical/
cubemap charts. (*"the conventional pinhole camera model is invalid for field of views of 180°
or more even when the calibration model can accommodate a wider FoV."*)
**Sources:** Schwalbe/Abraham (ISPRS, S0924271605000146); Meuleman et al. CVPR 2021; 360SD-Net.

### F2 — Omnidirectional imagery uses a standard catalog of charts · `3-0`
360 content uses dedicated formats, each with characteristic distortion / FOV / sampling
trade-offs: **ERP** (longitude–latitude onto a plane; severe pole distortion growing with
latitude, worst at ±90°), **cubemap (CMP)** (six 90°-FOV faces; less polar distortion),
**tangent projection (TP)** (locally distortion-free perspective patches; needs stitching),
and **polyhedron/icosahedron** (recursive subdivision → near-uniform sampling, higher
complexity).
**Sources:** arXiv:2401.09252 (ACM Computing Surveys), arXiv:2509.04444.

### F3 — ERP distortion breaks standard CNNs → multi-projection fusion · `3-0`
ERP's nonlinear sphere→plane mapping breaks the translation-equivariance assumption of standard
CNN filters (degraded features near the poles). This motivates **projection-driven methods**
that re-project ERP into **cubemap or tangent-plane** views and **fuse** multi-projection
features (SphereNet/Coors ECCV'18, OmniFusion, UniFuse).
**Sources:** arXiv:2509.04444, arXiv:2112.14331.

### F4 — Tangent images = gnomonic patches that let you reuse perspective models · `3-0`
**Tangent images** use **gnomonic** projection to convert ERP/spherical content into
low-distortion perspective patches placed at **cubemap (6)** or **regular-icosahedron (20+)**
vertices, enabling off-the-shelf perspective depth/flow models on 360 input. Per-patch outputs
**must be recombined** for global consistency (360MonoDepth: deformable multi-scale alignment +
gradient-domain blending). Originates with Eder et al. CVPR 2020.
**Sources:** arXiv:2112.14331, arXiv:2111.15669 (360MonoDepth), Eder CVPR 2020.

### F5 — Epipolar lines become *curves*; no homography can straighten them · `3-0`
In fisheye/omnidirectional projection the epipolar lines become **curved epipolar curves**
(conics for central catadioptric / unified models) because the projection is nonlinear, so a
3×3 homography (line-preserving) **cannot** rectify them. Dedicated fisheye epipolar
rectification maps the curves to straight lines for standard 1D stereo over the full wide FOV.
**Sources:** Abraham & Förstner 2005 (ISPRS S0924271605000146); Meuleman et al. CVPR 2021.

### F6 — The essential-matrix constraint holds on **unit bearing vectors** · `3-0`
For corresponding **unit bearing/ray vectors** between two spherical/omnidirectional images,
the same constraint holds: `q₂ᵀ E q₁ = 0`, `E = [t]_× R`, rank 2. So the **eight-point
algorithm** recovers motion and structure directly on rays — **no pinhole required.** Caveat:
pixel-domain **Hartley normalization must be redesigned for spherical bearing vectors**
(*Robust 360-8PA*, ICRA 2021, arXiv:2104.10900, ≈20% pose-accuracy improvement). Corroborated
by openMVG (essential matrix from bearing vectors on spherical panoramas).
**Sources:** Fujiki et al. MIRAGE 2007 (Epipolar Geometry via Rectification of Spherical Images).

### F7 — Spherical disparity is **angular (arc-length), not linear in inverse depth** · `3-0`
Disparity in spherical stereo is proportional to **arc length** along the epipolar great circle
and is **not** linearly proportional to inverse depth (unlike planar disparity). Traditional
spherical stereo rectifies and matches along the great circle, requiring **exhaustive
correspondence search** rather than the linear inverse-depth sampling of planar plane-sweep.
**Sources:** Meuleman et al. CVPR 2021; VCL3D Spherical View Synthesis (arXiv:1909.08112).

### F8 — Classical stereo can be reformulated on the spherical domain · `3-0`
The whole classical stack (epipolar great circles, spherical rectification, **sphere-sweep**
cost volumes, arc-length disparity) has a genuine spherical reformulation for 360 depth, and
extends to MVS/SfM (S-OmniMVS ACM MM'23, MCPDepth, GEER).
**Sources:** arXiv:2401.09252 (ACM Computing Surveys).

### F9 — Sphere-sweeping runs directly on raw fisheye, **no rectification** · `3-0`
Sphere-sweep stereo can run **directly on multiview fisheye** without spherical/ERP
rectification, avoiding the severe distortion and **position-dependent disparity
inconsistency** ERP introduces. This is the preferred modern approach for ≥180° fisheye stereo
where plane-sweep is invalid. (*"a given disparity does not correspond to the same distance
depending on its position in the image."*) Code: github.com/KAIST-VCLAB/sphere-stereo. Built on
by CasOmniMVS (2024), OmniStereo (CVPR 2025).
**Sources:** Meuleman et al. CVPR 2021.

### F10 — Top-bottom rigs give vertical-meridian epipolar lines in ERP · `3-0`
Omnidirectional stereo rigs are arranged **top-bottom (vertical baseline)** so epipolar great
circles pass through the poles and project to **constant-longitude vertical meridians** in ERP —
correspondences lie on the same vertical line, angular disparity `d = θ_b − θ_t`. Standard
left-right rigs do **not** preserve a straight epipolar-line property (horizontal 3D lines map
to ERP **sinusoids**).
**Sources:** 360SD-Net (BMVC); arXiv:2401.09252; GEER.

### F11 — Dense 360 reconstruction via optical-flow ERP rectification · `3-0`
Two spherical images with a **vertically displaced** orientation can be reconstructed densely by
using the **dense optical-flow field** (not sparse features) as the matching primitive: a single
non-linear minimization jointly aligns the 2D ERP flow, refining the 5-DOF epipolar geometry,
and reads off 3D structure from the converged flow magnitude.
**Sources:** Pathak et al. IEEE IST 2016 (document 7738212).

---

## Killed claim (do not build on this)

### ✗ "The unit sphere is the universal/canonical chart for all calibrated central cameras" · `0-3`
**Refuted by all three verifiers.** Sources say a sphere is *suitable* / *a* representation, not
the canonical one. **Design implication:** keep DS-MSP **chart-agnostic** — everything keys off
`project` / `unproject`; do **not** privilege one spherical chart as canonical.
**Source flagged against:** Fujiki et al. (ResearchGate 221055025).

---

## Caveats (from the verification pass)

- Source quality is strong: nearly all findings rest on peer-reviewed primaries (CVPR, ICRA,
  BMVC, ACM Computing Surveys, ISPRS) and several are independently corroborated.
- A few primary PDFs were gated (403); those quotes were confirmed via the search index +
  convergent secondary corroboration, not a byte-level match — substance multiply confirmed.
- Minor wording to discount: "360 *requires* dedicated formats" (sources say "suitable for",
  though planar charts genuinely can't cover ≥180° without singularity); ERP is not bijective at
  the poles (that degeneracy *is* the pole distortion); CNNs are translation-**equivariant** not
  invariant (the literature itself says "invariant").
- Time-sensitivity: deep-learning-on-spherical-charts (2020–2025) moves fast, so *method
  rankings* may shift — but the **geometric facts are stable** (epipolar curves, arc-length
  disparity, gnomonic limits, essential matrix on rays).
- The prioritized backlog is a **synthesized engineering recommendation** grounded in the
  verified findings, not a single sourced claim; the research did **not** inspect the live
  DS-MSP codebase.

## Open questions to resolve at design time

1. **Disparity/cost-volume parameterization** for sphere-sweep so one index = consistent depth
   across the image (sphere/depth candidates vs angular-disparity sampling), given F7.
2. **Spherical normalization** for the eight-point — adopt Robust 360-8PA preconditioning;
   wrap vs reimplement OpenGV minimal solvers (5-pt, generalized).
3. **Resampling robustness** — valid-mask, seam wraparound (±180° longitude, poles), and
   antialiasing strategy that keeps chart round-trips near the existing ~1e-13 px.
4. **Interop target first** — COLMAP camera-model export, openMVG spherical SfM, OpenMVS, or
   Basalt/OpenVSLAM fisheye, and the intrinsics/convention conversions each needs.

## Key sources & libraries (for implementation)

| Topic | Source / library |
|---|---|
| Sphere-sweep stereo on raw fisheye | Meuleman et al. CVPR 2021 · github.com/KAIST-VCLAB/sphere-stereo |
| Tangent images | Eder et al. CVPR 2020 · 360MonoDepth (arXiv:2111.15669) |
| Essential matrix on bearing vectors | Fujiki MIRAGE 2007 · **OpenGV** (github.com/laurentkneip/opengv) · **PoseLib** |
| Spherical 8-point normalization | Robust 360-8PA, ICRA 2021 (arXiv:2104.10900) |
| Top-bottom 360 stereo / angular disparity | 360SD-Net (BMVC) |
| Omnidirectional MVS (learned) | OmniMVS (ICCV 2019) · S-OmniMVS (ACM MM 2023) |
| Surveys | arXiv:2401.09252 (ACM Computing Surveys), arXiv:2509.04444 |
| Pipelines / camera models | COLMAP (colmap.github.io/cameras.html), openMVG, OpenMVS, Basalt, OpenVSLAM |
