# ADR-0003 — Hand-derived analytic Jacobians, finite-difference-checked (no autodiff)

- **Status:** Accepted (retrofit, recorded 2026-06-28)
- **Deciders:** maintainer
- **Relates to:** ARC-CORE, ARC-MODELS, FR-MODEL-003, NFR-NUM-001, NFR-REPRO-001
- **Supersedes:** —

## Context

Calibration and bundle adjustment are nonlinear least-squares problems solved by Levenberg-Marquardt
on a manifold (SO(3)/SE(3) re-basing, Schur complement for per-view poses). They need Jacobians of
the projection w.r.t. both 3D points and model parameters. The two ways to get them are automatic
differentiation (a framework dependency, runtime graph overhead) or hand-derived analytic
expressions.

## Decision

Use **hand-derived analytic Jacobians** throughout (`project_jacobian` on each model; the SO(3) right
Jacobian behind the solver), and treat their correctness as a *standing contract* checked against
finite differences to a strict tolerance.

The gradient check uses Richardson-extrapolated central differences
`(4·D(h/2) − D(h))/3` so finite-difference truncation error is negligible (empirically ~1e-11);
analytic-vs-FD relative error must be ≤ **1e-6**. A failure means the analytic derivative is wrong,
not the differencing.

## Consequences

**Positive**
- No autodiff framework dependency; the solver path stays pure NumPy and portable
  ([ADR-0004](ADR-0004-cv2-scipy-free-foundation.md)).
- Fast and numerically stable; full control over the manifold retraction.
- The "differentiability guarantee" is a test, runnable on every model PR (`-m jac`).

**Negative / costs**
- Each new model/residual must derive and implement its Jacobian — the main authoring cost, and the
  reason the gradient-check contract is mandatory before a model is considered done.

## Verification

- `tests/contract/test_gradcheck.py` (`@pytest.mark.jac`) — all eight models + the SO(3) retraction at
  rel-tol 1e-6; deterministic via fixed seeds.
- A cheap `allclose` smoke tier in `tests/contract/test_camera_model_contract.py` runs always-on.

Covered requirements: **FR-MODEL-003, NFR-NUM-001, NFR-REPRO-001**.

## Alternatives considered

- *Automatic differentiation (e.g. a tensor framework / JAX).* Rejected — a heavy dependency and
  runtime graph for derivatives we can write once and verify exactly; conflicts with ADR-0004.
- *Finite differences at runtime.* Rejected — too slow and too noisy inside an iterative solver.
