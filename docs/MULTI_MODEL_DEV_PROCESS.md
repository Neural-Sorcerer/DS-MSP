# DS-MSP Multi-Model — Development Process (TDD, Decoupling, Branching)

> **Design note (internal).** An engineering process document — *how* the multi-model work was
> built — not a tutorial. To *use* the library see [`MULTI_MODEL.md`](MULTI_MODEL.md); to *learn*
> the geometry see [`learn/`](learn/README.md).

Companion to `MULTI_MODEL_ADAPTER_PLAN.md`. This document answers *how* we build
it: the decoupling architecture, the interface/data contracts, the contract-test
discipline, the TDD loop, and the branching strategy — so every module is
**independently developed, independently testable, and standalone** wherever
possible, coupling only through explicit contracts.

---

## 1. The decoupling principle: depend on contracts, not classes

The whole design rests on **two ideas**:

### (a) Two-layer pattern per model: pure math + thin class
Every model is split so the heavy logic needs *nothing* from the package:

```
models/ds_math.py     # PURE functions on ndarrays. Imports: numpy only.
                      #   ds_project(P, fx,fy,cx,cy,xi,alpha) -> (u,v,valid)
                      #   ds_unproject(uv, ...) -> (rays, valid)
                      #   ds_project_jacobian(P, ...) -> (u,v,J_point,J_param,valid)
models/double_sphere.py  # Thin class wrapping ds_math, implements the Protocol.
                         # Imports: numpy, core.contracts, models.ds_math
```

The `*_math.py` layer is **runtime-standalone**: a user (or a test) can
`from ds_msp.models.ds_math import ds_project` and call it on raw arrays with **no
camera object at all**. This is the strongest form of "works in the absence of the
camera module."

### (b) Services depend on a Protocol, never on a concrete model
`ops/` (undistort, pose, viz, ldc) and `adapt/` (convert) import **only**
`core.contracts.CameraModel` (a structural `typing.Protocol`). They never import
`DoubleSphereModel` or any sibling. They are tested against a **`FakeModel`** stub.
That makes them decoupled from every concrete model and from each other.

### Allowed import directions (enforced, not just intended)
```
numpy ──────────────────────────────────────────────► (everything)
core.contracts        (no internal imports except numpy/typing)
core.pinhole          → numpy
models/*_math         → numpy                              (PURE, standalone)
models/<Model>        → core.contracts, models.<model>_math
ops/*                 → core.contracts            (NOT models, NOT each other)
adapt/*               → core.contracts, (optionally) a model registry for factory
io/*                  → core.contracts, models.* (only the registry)
calibrate             → core.contracts
cv.py, ldc.py         → core.contracts
```
**Rule of thumb:** arrows only point *down* this list. `core` knows nothing about
`models`; `models` know nothing about `ops`; `ops` know nothing about each other.
A cycle or an up-arrow is a build break (see §6 import-linter gate).

> Consequence: you can develop, test, and ship `ops/undistort.py` **before any
> real model exists**, using only the contract + a stub. That is the independence
> the request asks for.

---

## 2. Interface & data contracts (integration compatibility)

Compatibility = everyone agrees on **signatures, dtypes, shapes, units, and
conventions** *before* code is written. These live in `core/contracts.py` and are
frozen first (the first branch).

### 2.1 Canonical data types (the wire format between modules)
| Name | Shape | dtype | Meaning / convention |
|---|---|---|---|
| `Points3D` | `(N, 3)` | float64 | camera-frame points, **meters**, +Z forward |
| `Pixels` | `(N, 2)` | float64 | `(u, v)`, pixel **centers**, origin top-left |
| `Rays` | `(N, 3)` | float64 | **unit-norm** bearing vectors |
| `Valid` | `(N,)` | bool | per-point projectability/feasibility mask |
| `Params` | `(P,)` | float64 | model params in `param_names` order |
| `J_point` | `(N, 2, 3)` | float64 | `∂(u,v)/∂(x,y,z)` |
| `J_param` | `(N, 2, P)` | float64 | `∂(u,v)/∂params` |

Conventions fixed once: pixel-center sampling, +Z forward, rays unit-norm, params
always returned/accepted in the declared order, invalid rows zeroed (not NaN).
Batch shapes broadcast: `(...,3)`/`(...,2)` accepted, leading dims preserved.

### 2.2 The Protocol (structural; no inheritance required)
```python
# core/contracts.py  -- imports: numpy, typing only
from typing import Protocol, runtime_checkable, ClassVar
import numpy as np

@runtime_checkable
class CameraModel(Protocol):
    name: ClassVar[str]
    param_names: ClassVar[tuple[str, ...]]

    @property
    def params(self) -> np.ndarray: ...
    @property
    def K(self) -> np.ndarray: ...
    @property
    def distortion(self) -> np.ndarray: ...

    def project(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    def unproject(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    def project_jacobian(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...

    @classmethod
    def from_params(cls, p: np.ndarray) -> "CameraModel": ...
    @classmethod
    def param_bounds(cls) -> tuple[np.ndarray, np.ndarray]: ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel": ...
    def initialize_from_correspondences(self, K_seed: np.ndarray,
                                        rays: np.ndarray, pixels: np.ndarray) -> None: ...
```
A model satisfies the contract by **shape** (duck typing) — no base-class import
needed, so models stay decoupled from `core` at runtime while `mypy` still checks
them statically. `@runtime_checkable` lets services `assert isinstance(m, CameraModel)`.

### 2.3 Static enforcement
`mypy --strict` on the package. Every service signature is typed with
`CameraModel`, every model is checked to satisfy the Protocol. Signature/dtype
drift becomes a type error, not a runtime surprise.

---

## 3. The contract test suite (the heart of "integration compatibility")

One **parametrized, model-agnostic** test module that *every* model must pass.
Adding a model = registering it; the suite proves it is a drop-in.

```python
# tests/contract/test_camera_model_contract.py
import numpy as np, pytest
from ds_msp.core.contracts import CameraModel

# Each model package registers a factory producing a realistic instance.
MODEL_FACTORIES = collect_registered_models()   # ds, ucm, eucm, kb, radtan, ...

@pytest.fixture(params=MODEL_FACTORIES, ids=lambda f: f.name)
def model(request): return request.param()

def test_satisfies_protocol(model):
    assert isinstance(model, CameraModel)

def test_project_shapes_and_dtypes(model):
    P = sample_forward_points()                       # (N,3) f64
    uv, valid = model.project(P)
    assert uv.shape == (len(P), 2) and uv.dtype == np.float64
    assert valid.shape == (len(P),) and valid.dtype == bool

def test_roundtrip_project_unproject(model):
    P = sample_forward_points()
    uv, v1 = model.project(P)
    rays, v2 = model.unproject(uv)
    ok = v1 & v2
    cos = cosine(rays[ok], normalize(P[ok]))
    assert (cos > 1 - 1e-6).all()                     # bearing recovered

def test_jacobian_matches_finite_difference(model):
    P = sample_forward_points()
    _, _, Jp, Jpar, _ = model.project_jacobian(P)
    assert np.abs(Jp   - fd_point(model, P)).max()  < 1e-5
    assert np.abs(Jpar - fd_param(model, P)).max()  < 1e-5

def test_param_vector_roundtrip(model):
    m2 = type(model).from_params(model.params)
    assert np.allclose(m2.params, model.params)

def test_serialization_roundtrip(model):
    m2 = type(model).from_dict(model.to_dict())
    assert np.allclose(m2.params, model.params)

def test_rays_are_unit_norm(model): ...
def test_invalid_rows_are_zeroed_not_nan(model): ...
```

This single suite guarantees **signature + data-type + behavioral compatibility**
across all models, so the converter and services can rely on them
interchangeably. The Jacobian FD-check is the same harness already used for DS
(max-err ~1e-7).

---

## 4. Independence guarantees & how they are verified

| Sense of "independent" | Mechanism | Verified by |
|---|---|---|
| **Runtime, no camera object** | `*_math.py` are pure array functions | unit tests call them with raw arrays only |
| **Import independence** (no concrete-sibling imports) | layered import rules (§1) | `import-linter` contract in CI |
| **Service tested without real models** | services take a Protocol; a `FakeModel` stub exists | `ops/`+`adapt/` tests use `FakeModel`, never `DoubleSphereModel` |
| **Module imports in isolation** | no top-level side effects, lazy optional deps | `test_imports.py` imports each submodule alone in a fresh interpreter |
| **Optional deps don't crash core** | OpenCV/SciPy imported only where used | math/contract tests run with numpy only |

`FakeModel` (in `tests/support/fake_model.py`) is a trivial perfect-pinhole model
implementing the Protocol in ~20 lines. Because `convert()`, `Undistorter`,
`solve_pnp`, and the LDC generator are all tested against it, we *prove* they work
"in the absence of the camera module" (no DS, no fisheye math required).

---

## 5. TDD loop per module (red → green → refactor)

Each module follows the same cycle; **tests and signatures land before logic**.

1. **Contract first.** Add/extend the type in `core/contracts.py` and the data
   conventions (§2). Commit (no behavior yet).
2. **Write failing tests.** Unit tests for the module + (for a model) register it
   into the contract suite. Run → **red**.
3. **Stub the signatures.** Implement the public functions/methods as typed stubs
   that `raise NotImplementedError`. `mypy` passes, tests still **red**.
4. **Implement to green.** Fill in the closed-form math + analytic Jacobian until
   unit + contract tests pass. Gradient-check gate must be green.
5. **Refactor.** Remove duplication, tighten docstrings; tests stay green.
6. **Integration check.** Run the *full* existing suite (DS regression) — nothing
   downstream breaks.

### Worked example — adding the UCM model (Phase 2 branch)
```
# commit 1 (contract): nothing new needed if Protocol already covers it
# commit 2 (tests, RED):
#   tests/models/test_ucm_math.py         -> project/unproject/roundtrip on known values
#   register UCMModel in MODEL_FACTORIES  -> contract suite now runs against UCM (fails)
# commit 3 (stubs): ucm_math.ucm_project/unproject/jacobian raise NotImplementedError
# commit 4 (GREEN): implement closed forms + Jacobian; gradient-check passes
# commit 5 (refactor): factor shared UCM/EUCM denominator helper
# DONE when: ucm unit tests + contract suite + DS regression all green; mypy clean
```

### Definition of Done (per module)
- [ ] Public signatures typed; `mypy --strict` clean.
- [ ] Unit tests + (models) contract suite green; coverage ≥ 90% for the module.
- [ ] Analytic Jacobian gradient-checked < 1e-5 vs FD.
- [ ] Imports obey the layer rules (import-linter green).
- [ ] Module imports & its tests run **standalone** (no sibling concrete imports).
- [ ] Full existing DS suite still green (no regression).
- [ ] Docstrings + a README/usage snippet that is executed in CI.

---

## 6. Branching & PR strategy

**Yes — one branch per module/feature**, small and independently reviewable, off a
long-lived integration branch.

```
main
 └── feat/multi-model            (integration branch; MVP merges here, then to main)
      ├── feat/mm-00-contracts    (core/contracts.py, data types, FakeModel, contract suite skeleton)
      ├── feat/mm-01-ds-impl      (DoubleSphereModel implements Protocol; back-compat facade)   ── needs 00
      ├── feat/mm-02-ops-split    (ops/undistort, ops/pose, ops/viz; tested with FakeModel)     ── needs 00
      ├── feat/mm-03-convert      (adapt/sampling, adapt/convert, adapt/evaluate; FakeModel)    ── needs 00
      ├── feat/mm-04-ucm          (models/ucm)                                                  ── needs 00
      ├── feat/mm-05-eucm         (models/eucm)                                                 ── needs 00
      ├── feat/mm-06-kb           (models/kb + io/kalibr)                                        ── needs 00,03
      └── feat/mm-07-radtan       (models/radtan + cv.py generalize)                            ── needs 00,03
```

**Rules**
- Branch names: `feat/mm-<NN>-<slug>`. Each maps to one Definition-of-Done unit.
- **Stacking:** `00-contracts` is the trunk for the rest; model/service branches
  start from it (they don't depend on each other), so they develop **in parallel**.
  `convert` and `kb` stack where a real dependency exists.
- **Small PRs:** tests-first commit, then stubs, then implementation — visible in
  history. Reviewer can see red→green.
- **CI gate (every PR):** ruff (lint) · mypy --strict · pytest (module + contract +
  full DS regression) · gradient-check · import-linter · coverage threshold.
- **Merge cadence:** each branch → `feat/multi-model` when green. When Phases 0–3
  (MVP) are all merged and the integration branch is green, **merge
  `feat/multi-model` → `main`** as one reviewed slice, then push (as we did for the
  last release).
- **No direct commits to `main`** for this work; `main` always stays releasable.

---

## 7. Per-branch test plan (what each branch proves)

| Branch | Unit tests | Contract/integration | Standalone proof |
|---|---|---|---|
| 00-contracts | data-type validators, `FakeModel` round-trip | contract suite runs on `FakeModel` | importable with numpy only |
| 01-ds-impl | DS already-tested math reused | DS passes the contract suite | `ds_math` usable w/o class |
| 02-ops-split | undistort maps, pnp, viz on `FakeModel` | services accept any Protocol | run with **no real model** |
| 03-convert | sampler counts/coverage; `convert(Fake→Fake)` RE≈0 | identity + Fake→Fake | converter w/o fisheye math |
| 04/05 ucm/eucm | known-value project/unproject; Jac FD | contract suite green | `*_math` standalone |
| 06 kb | OpenCV `cv2.fisheye` cross-check; Newton convergence | DS→KB RE small; Kalibr YAML round-trip | `kb_math` standalone |
| 07 radtan | OpenCV `projectPoints` cross-check | DS→RadTan reports coverage/limits | `radtan_math` standalone |

Cross-checks against OpenCV (`cv2.fisheye` for KB, `cv2.projectPoints` for RadTan)
give an **external oracle** beyond our own FD checks.

---

## 8. Tooling / CI summary

- **pytest** + `pytest-cov` (per-module + contract suites; coverage gate 90%).
- **mypy --strict** (signature/type/Protocol conformance = integration safety).
- **ruff** (lint/format).
- **import-linter** (`importlinter` contracts encode §1 layer rules → decoupling is
  CI-enforced, not aspirational).
- **gradient-check** marker (`pytest -m jac`) run on every model PR.
- One GitHub Actions workflow, matrix over Python versions; same gates locally via
  a `make check` / `tox` target so a branch is verifiable before pushing.

---

## 9. How this answers the requirements

- **TDD, well planned, tested** → §3, §5 (tests + signatures precede logic; contract
  suite; DoD with coverage + gradient-check gates).
- **Each feature a new branch** → §6 (per-module branches off an integration branch,
  parallel where independent, stacked where dependent).
- **Write test first, define signature, then connect** → §5 TDD loop steps 1–4.
- **Connection/integration compatibility (signature, dtype, arguments)** → §2 data
  contracts + Protocol + `mypy --strict`; §3 contract suite enforces behavior.
- **Well decoupled, runs as independent/standalone module even without the camera
  module** → §1 layering + pure `*_math` layer + Protocol-only services + §4
  `FakeModel` and import-isolation tests + §8 import-linter.
```
