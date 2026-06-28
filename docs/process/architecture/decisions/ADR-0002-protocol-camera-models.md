# ADR-0002 — One `CameraModel` protocol for all interchangeable models

- **Status:** Accepted (retrofit, recorded 2026-06-28)
- **Deciders:** maintainer
- **Relates to:** ARC-CORE, ARC-MODELS, FR-MODEL-001..005, NFR-ARCH-003
- **Supersedes:** —

## Context

The library supports eight camera models with very different parameterizations (Double Sphere, UCM,
EUCM, Kannala-Brandt, RadTan, OCam, and the DS⁺ / EUCM⁺ extensions). Every downstream service —
calibration, conversion, undistort/reproject, PnP, stereo, rig, VO — must work with *any* of them.
If services branched on a model enum, each new model would require edits across the whole codebase.

## Decision

Define a single structural **`CameraModel` protocol** (`ds_msp/core/contracts.py`) that every model
implements:

- `project(points_3d) -> (pixels, valid)`
- `unproject(pixels) -> unit bearing rays`
- `project_jacobian(...)` — analytic point and parameter Jacobians
- parameter serialization / deserialization (round-trip)

Higher layers depend on the **protocol**, never on a concrete model class. Models live in
`ds_msp/models` with their numerics isolated in pure-NumPy `*_math` modules and registered in a
registry for name-based construction.

## Consequences

**Positive**
- Any model is a drop-in for any other; new models need no edits to existing services.
- The protocol is the one stable seam the platform is built around — the basis for composability
  without autodiff ([ADR-0003](ADR-0003-analytic-jacobians.md)).

**Negative / costs**
- Every model must implement the full contract, including analytic Jacobians — a real authoring cost,
  paid down by the playbook and the contract test suite.

## Verification

- `tests/contract/test_camera_model_contract.py` — shapes/dtypes, unit-norm bearings, serialization
  round-trip, and protocol satisfaction across all models.
- `tests/contract/test_gradcheck.py` — analytic-Jacobian correctness for every model.

Covered requirements: **FR-MODEL-001, FR-MODEL-002, FR-MODEL-004, FR-MODEL-005, NFR-ARCH-003**
(and FR-MODEL-003 via ADR-0003).

## Alternatives considered

- *Enum + `if model_type == ...` branching.* Rejected — O(models × services) edits; no isolation.
- *Inheritance from a base class.* Rejected in favour of a structural Protocol — looser coupling, no
  forced MRO, easier to satisfy from a plain dataclass.
