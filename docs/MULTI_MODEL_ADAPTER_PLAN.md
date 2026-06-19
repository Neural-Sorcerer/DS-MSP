# DS-MSP → Multi-Model Camera Library + Adapter — Analysis & Plan

**Goal:** evolve DS-MSP from a Double-Sphere-only library into a small multi-model
camera library where a user can calibrate in Double Sphere (DS) and **convert the
result to any other supported model** (UCM, EUCM, KB, RadTan/pinhole, …), with
**every feature** (project, unproject, undistort, PnP, LDC export, draw axes,
calibrate) working uniformly on any model.

**Constraints (from the request):**
- Stay **pure Python** (NumPy/SciPy/OpenCV). No C++.
- Math should be **analytical** (closed-form project/unproject + hand-derived
  Jacobians), **not** autodiff.
- Keep it intuitive and drop-in.

**Reference:** `eowjd0512/fisheye-calib-adapter` (FCA), arXiv:2407.12405. FCA is
C++/Ceres; we replicate its *capability* in Python with SciPy.

---

## 1. What FCA actually does (the capability we want)

FCA converts an **already-calibrated** parameter set from a source model to a
target model **without images and without recalibration**:

1. **Sample** ~500 pixels on a regular image grid.
2. **Unproject** each pixel with the *source* model → 3D bearing rays; keep only
   `z > 0` (forward hemisphere).
3. **Initialize** the target model: inherit `fx, fy, cx, cy` from the source;
   seed the distortion params with a **linear least-squares (SVD)** solve.
4. **Refine** the target params with a nonlinear solver (Ceres, **analytic
   Jacobians**, DENSE_QR), minimizing **pixel reprojection error**
   `e = project_target(ray) − u_source`.
5. **Report** Reprojection Error (RE), Parameter Error vs a GT model (PE), and
   image PSNR/SSIM.

Supported models: **UCM(5), EUCM(6), DS(6), KB(8), OCamCalib, RadTan(9)**. No FOV
model. Conversions among the EUCM/KB/DS family are near-exact (RE ~1e-5…1e-10 px);
anything **into RadTan (pinhole) degrades** because a narrow-FOV model cannot
represent a wide fisheye (DS↔RadTan is the worst pair).

**Architectural core:** an abstract `Base` camera class with a uniform interface
(`project`, `unproject`, `initialize`, `optimize`, `parse`, `save`, `evaluate`),
and an `Adapter` that orchestrates sample→unproject→initialize→optimize→evaluate.

---

## 2. Gap analysis — DS-MSP today vs target

| Capability | DS-MSP today | Target | Work |
|---|---|---|---|
| Closed-form project/unproject | ✅ DS only | All models | Add per-model math |
| Analytic projection Jacobian | ✅ DS only (`ds_project_jacobian`) | All models | Add per-model Jacobian |
| Common model interface | ❌ none | `CameraModel` ABC | **New** |
| Undistortion (image/points) | ✅ but DS-coupled in `cv.py` | Model-agnostic | Re-route via interface |
| PnP | ✅ DS method | Model-agnostic | Move to service, take `CameraModel` |
| LDC export | ✅ DS-coupled | Model-agnostic | Take `CameraModel` |
| Calibration | ✅ DS-coupled residual+Jac | Any model | Parameterize by model |
| **Model conversion (adapter)** | ❌ none | DS→{UCM,EUCM,KB,RadTan,…} | **New (the headline feature)** |
| Multi-model I/O (Kalibr YAML) | ❌ DS JSON only | Per-model YAML/JSON | **New** |
| Conversion evaluation (RE/PE/PSNR/SSIM) | ❌ | Report | **New** |

**Encouraging fact (verified):** undistortion, PnP, LDC, and `draw_axes` already
call only `project`/`unproject`. They become polymorphic the moment an interface
exists — almost no logic change. Only `cv.py` and `calibrate.py` hardcode DS math.

---

## 3. Target architecture

```
ds_msp/
  core/
    base.py          # CameraModel ABC: the uniform interface (project/unproject/jacobian/params/IO)
    pinhole.py       # shared K helpers, ray <-> normalized-plane utilities
  models/
    double_sphere.py # DoubleSphereModel  (wraps existing ds_project/ds_unproject/ds_project_jacobian)
    ucm.py           # UCMModel
    eucm.py          # EUCMModel
    kb.py            # KannalaBrandtModel (OpenCV-fisheye compatible)
    radtan.py        # RadTanModel / pinhole+Brown (OpenCV compatible)
    ocam.py          # (optional, phase 3) Scaramuzza polynomial
  ops/
    undistort.py     # Undistorter(model, w, h): maps + caching (stateful, OFF the model)
    pose.py          # solve_pnp(model, obj, img)
    viz.py           # draw_axes(model, ...), draw_reprojection(...)
  adapt/
    sampling.py      # image-grid and ray-sphere samplers
    convert.py       # convert(source_model, target_cls, ...) -> (target_model, report)
    evaluate.py      # RE / PE / PSNR / SSIM between two models
  io/
    kalibr.py        # load/save Kalibr-format YAML per model
    json_io.py       # existing JSON formats
  cv.py              # OpenCV-style shims, now model-generic (D length picks the model)
  ldc.py             # TI LDC, now takes any CameraModel
calibrate.py         # generic: calibrate ANY model via its project + jacobian
```

**Design rules**
- A **model is a pure value object**: only its parameter vector + closed-form
  math + analytic Jacobian. No image dims, no caching, no PnP, no drawing.
- **Stateful / cross-cutting concerns are services** in `ops/` that *take* a
  `CameraModel`. (This is the SRP split flagged earlier; it is now a prerequisite,
  not optional.)
- **Backward compatibility:** keep `DoubleSphereCamera` as a thin subclass/alias of
  `DoubleSphereModel` with the current method names (`undistort_image`,
  `solve_pnp`, `draw_axes`) delegating to the new `ops/` services, so existing
  user code and tests keep working.

### Dependency graph (acyclic)
```
core/base  ←  models/*  ←  adapt/convert
                      ↖
                        ops/* (undistort, pose, viz, ldc)  ←  cv.py
core/base  ←  calibrate (generic)
io/*  →  models/*
```

---

## 4. The `CameraModel` interface (concrete sketch)

```python
class CameraModel(ABC):
    name: str
    param_names: tuple[str, ...]          # e.g. ("fx","fy","cx","cy","xi","alpha")

    # --- parameter access ---
    @property
    def params(self) -> np.ndarray: ...           # flat vector in param_names order
    @classmethod
    def from_params(cls, p: np.ndarray) -> "CameraModel": ...
    @property
    def K(self) -> np.ndarray: ...                # 3x3 pinhole block
    @property
    def distortion(self) -> np.ndarray: ...       # model-specific tail

    # --- core math (closed form, vectorized) ---
    @abstractmethod
    def project(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...      # (uv, valid)
    @abstractmethod
    def unproject(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...   # (rays, valid)

    # --- analytic Jacobians (NO autodiff) ---
    @abstractmethod
    def project_jacobian(self, P):                                             # -> (uv, J_point, J_param, valid)
        ...   # J_point = d(uv)/dP (N,2,3);  J_param = d(uv)/d(params) (N,2,|p|)

    # --- conversion hooks ---
    @abstractmethod
    def initialize_from_correspondences(self, K_seed, rays, pixels) -> None: ...
        # linear SVD seed of the distortion params (intrinsics inherited)
    @classmethod
    def param_bounds(cls): ...                    # (lb, ub) for the optimizer

    # --- IO ---
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel": ...
```

Every existing standalone DS function maps directly: `project`→`ds_project`,
`unproject`→`ds_unproject`, `project_jacobian`→`ds_project_jacobian`. We replicate
exactly this trio for each new model.

---

## 5. The Adapter (conversion) — pure-Python, analytic

```python
def convert(source: CameraModel, target_cls, *, width, height,
            n_samples=500, sampler="grid", max_fov_deg=None) -> tuple[CameraModel, dict]:
    # 1. sample pixels (grid) or rays (sphere)
    pixels = sample_image_grid(width, height, n_samples)        # adapt/sampling.py
    # 2. source unproject -> bearing rays; keep forward + valid
    rays, valid = source.unproject(pixels)
    keep = valid & (rays[:, 2] > 1e-6)
    if max_fov_deg: keep &= angle_from_axis(rays) <= radians(max_fov_deg)
    rays, pixels = rays[keep], pixels[keep]
    # 3. seed target: inherit fx,fy,cx,cy from source; linear SVD for distortion
    target = target_cls.from_seed(source.K)
    target.initialize_from_correspondences(source.K, rays, pixels)
    # 4. refine: minimize project_target(rays) - pixels, analytic Jacobian wrt params
    def residual(p):
        m = target_cls.from_params(p)
        uv, _ = m.project(rays)
        return (uv - pixels).ravel()
    def jac(p):
        m = target_cls.from_params(p)
        _, _, J_param, _ = m.project_jacobian(rays)   # rays fixed -> only param Jacobian
        return J_param.reshape(-1, p.size)
    res = least_squares(residual, target.params, jac=jac,
                        bounds=target_cls.param_bounds(), method="trf", x_scale="jac")
    target = target_cls.from_params(res.x)
    report = evaluate(source, target, width, height)              # RE / PE / coverage
    return target, report
```

This reuses the **exact pattern already in `calibrate.py`** (SciPy LM + analytic
Jacobian). The only per-model additions are `project`, `project_jacobian`, and a
linear `initialize_from_correspondences`.

**Why the Jacobian is easy here:** in calibration we differentiate
`project(R·X + t)` w.r.t. both pose and intrinsics. In *conversion* the rays are
fixed inputs, so we only need `∂project/∂params` — which is exactly the
`J_param`/`J_intr` block each model already computes for calibration. No new
derivative type is required.

---

## 6. Per-model implementation breakdown (the math work)

For each model we need three vectorized closed-form functions:
`*_project(P, params)`, `*_unproject(uv, params)`, `*_project_jacobian(P, params)`,
plus a linear `initialize`. Difficulty and notes:

| Model | Params | Project | Unproject | Jacobian | Linear init | Effort |
|---|---|---|---|---|---|---|
| **DS** | fx,fy,cx,cy,ξ,α (6) | done | done | done | n/a (source) | ✅ done |
| **UCM** | …,α (5) | `den=α·d+(1-α)·z`, `d=‖P‖` | via `ξ=α/(1-α)`, closed form | trivial (1 distortion param) | 1-D SVD for α | **S** |
| **EUCM** | …,α,β (6) | UCM form with `d=√(β(x²+y²)+z²)` | closed form | 2 params, straightforward | 2-D SVD | **S** |
| **KB** | …,k1..k4 (8) | `θ=atan2(r,z)`, `d(θ)=θ+Σkᵢθ^(2i+1)`; `u=fx·d(θ)·x/r+cx` | Newton solve `d(θ)=ru` (≤10 it) | analytic; unproject Jac via inverse-function theorem | **linear** `k1..k4` from `(θ,ru)` pairs | **M** |
| **RadTan/pinhole** | …,k1,k2,k3,p1,p2 (9) | normalized → Brown distortion → ×K | Newton (≤10 it) | analytic Brown Jac | linear radial seed | **M** |
| **OCamCalib** | affine + polys | poly radial map | order-4 poly | analytic poly Jac | linear poly fit (SVD) | **L** (optional) |
| **FOV** (Devernay) | …,w (5) | `rd=atan(2·ru·tan(w/2))/w` | closed form | analytic | 1-D | **S** (optional; FCA omits it) |

- **S = small** (½–1 day each), **M = medium** (1–2 days, includes Newton inverse
  + its Jacobian via inverse-function theorem), **L = large** (3–4 days).
- KB is the highest-value add: it is OpenCV's `cv2.fisheye` model, so converting
  DS→KB makes the calibration usable by the entire OpenCV ecosystem.
- All Jacobians are hand-derived and **gradient-checked vs finite differences**
  (same harness already used for `ds_project_jacobian`, max-err ~1e-7).

---

## 7. Making existing features model-agnostic (file-by-file)

| File | Change | Risk |
|---|---|---|
| `ds_msp/model.py` | Split: `DoubleSphereModel(CameraModel)` keeps the math; move `undistort_*`, `solve_pnp`, `draw_axes`, map cache to `ops/`. Keep `DoubleSphereCamera` as a back-compat facade. | Med (API surface) |
| `ds_msp/cv.py` | Replace hardcoded `xi,alpha = D[0],D[1]` + `ds_project` with a model resolved from `D` length (2→DS/UCM-ambiguous, so pass model name or a `CameraModel`). Cleanest: add `cv.project_points(model, ...)` generic, keep DS-typed shims. | Med |
| `ds_msp/ldc.py` | `TI_LDC_MeshGenerator(model)` typed as `CameraModel`; the `double_sphere_params` dict becomes `model.to_dict()`. Logic unchanged (uses `project`). | Low |
| `calibrate.py` | Parameterize residual/Jacobian by a `CameraModel` class instead of `ds_project`/`ds_project_jacobian`. Pack/unpack uses `model.param_names`. Enables calibrating **any** model, not just DS. | Med |
| `ds_msp/__init__.py` | Export models, `convert`, `CameraModel`, `ops` services. | Low |
| `tests/` | Add round-trip + Jacobian gradient-check per model, conversion accuracy tests vs known Kalibr pairs. | Low |

---

## 8. Valid-domain / FOV mismatch (the correctness subtlety)

Different models cover different FOV and have different validity regions:
- **DS/UCM/EUCM/KB** can represent ≥180°. **RadTan/pinhole cannot** (rays at 90°
  go to infinity).
- Conversion **into a narrower model** must restrict the sampled FOV and **report
  coverage + residual**, never silently fit garbage. FCA's `z>0` filter is the
  minimum; we add an explicit `max_fov_deg` and a coverage metric (fraction of the
  source's valid FOV the target reproduces within tolerance).
- The converter must use each model's **own** validity mask when projecting during
  the fit, and exclude invalid samples from the residual (same fixed-size-residual
  discipline already in `calibrate.py`).

Deliverable: `convert()` returns `{rms_px, max_px, fov_covered_deg, n_used, warnings}`
so the user *sees* when DS→RadTan is lossy (it will be).

---

## 9. I/O & evaluation

- **`io/kalibr.py`** — read/write Kalibr `camchain` YAML (`camera_model`,
  `intrinsics`, `distortion_coeffs`) for DS, EUCM, KB, RadTan, UCM, so DS-MSP
  interoperates with the standard calibration ecosystem (this is most of FCA's
  practical value).
- **`adapt/evaluate.py`** — RE (px), PE vs a GT model, and optional PSNR/SSIM by
  the "recover image" method (unproject with source, reproject with target,
  remap). Mirrors FCA's evaluation so numbers are comparable.

---

## 10. Phased roadmap

| Phase | Deliverable | Depends on | Effort |
|---|---|---|---|
| **0** | `CameraModel` ABC + `core/`; `DoubleSphereModel` implements it; back-compat `DoubleSphereCamera` facade; move undistort/PnP/viz to `ops/`. Tests stay green. | — | ~2 d |
| **1** | `convert()` + `sampling.py` + `evaluate.py`, validated **DS→DS** (identity, must return RE≈0) and DS→UCM/EUCM. | 0 | ~2 d |
| **2** | UCM, EUCM models (math + Jacobian + linear init + tests). | 0 | ~2 d |
| **3** | KB model (OpenCV-fisheye compatible) + Kalibr YAML I/O. Highest external value. | 0,1 | ~2-3 d |
| **4** | RadTan/pinhole model + `cv.py` generalization + OpenCV cross-checks. | 0,1 | ~2-3 d |
| **5** | Generic `calibrate.py` (calibrate any model), evaluation report, docs/README. | 2-4 | ~2 d |
| **6 (opt)** | OCamCalib and/or FOV models. | 0 | ~3-5 d |

**MVP that satisfies the request:** Phases 0–3 (DS calibration → convert to
UCM/EUCM/KB, all features polymorphic) ≈ **8–9 days**. Full parity with FCA incl.
RadTan + OCam + Kalibr I/O ≈ **15–18 days**.

---

## 11. Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Back-compat break for existing `DoubleSphereCamera` users/tests | High | Keep a facade subclass delegating to `ops/`; run the full existing suite each phase. |
| Hand-derived Jacobian bugs | Med | Gradient-check every Jacobian vs FD (existing harness); CI gate. |
| Newton-based unproject (KB/RadTan) non-convergence at extreme FOV | Med | Cap iters, return invalid mask, restrict sample FOV, report coverage. |
| Lossy conversions (→RadTan) misread as exact | Med | Always return RE/coverage; warn when RE>threshold; document per-pair limits (DS↔RadTan worst). |
| Scope creep (6+ models) | Med | Ship MVP (DS/UCM/EUCM/KB) first; OCam/FOV optional. |
| `cv.py` ambiguity (D-length can't always identify the model) | Low | Generic API takes a `CameraModel`; keep typed shims for cv2 parity. |

---

## 12. End-state API (what the user will write)

```python
from ds_msp import DoubleSphereModel, convert
from ds_msp.models import KannalaBrandtModel, EUCMModel
from ds_msp.ops import solve_pnp, Undistorter

# calibrate in DS (existing pipeline) -> a DoubleSphereModel
ds = DoubleSphereModel.from_json("results/calibration_params.json")

# convert to OpenCV-fisheye (KB) with one call
kb, report = convert(ds, KannalaBrandtModel, width=1920, height=1080)
print(report)   # {'rms_px': 8e-3, 'max_px': 0.02, 'fov_covered_deg': 192, ...}

# EVERY feature now works on EITHER model, unchanged:
und = Undistorter(kb, 1920, 1080); img_rect, K_new = und.undistort_image(img)
ok, rvec, tvec = solve_pnp(kb, object_pts, image_pts)

# interop: hand KB to OpenCV directly
import cv2; cv2.fisheye.undistortImage(img, kb.K, kb.distortion, Knew=K_new)

# or export Kalibr YAML for the rest of the ecosystem
kb.to_kalibr_yaml("camchain-kb.yaml")
```

---

## 13. Recommendation

1. **Do Phase 0 first regardless** — the `CameraModel` interface + SRP split is the
   foundation, and it also resolves the god-object issue already on the books.
2. **Target the MVP (Phases 0–3): DS + UCM + EUCM + KB + `convert()` + Kalibr I/O.**
   That delivers the requested capability (calibrate in DS, convert to other
   models, all features everywhere) and the single highest-value interop (KB =
   OpenCV fisheye) in ~8–9 days.
3. **Add RadTan + OCamCalib + FOV later** for full FCA parity, clearly labeling the
   lossy fisheye→pinhole conversions.
4. Keep the **analytic-Jacobian + SciPy-LM** approach throughout (no autodiff),
   gradient-checking each new Jacobian — consistent with what is already shipped
   for DS.
```
