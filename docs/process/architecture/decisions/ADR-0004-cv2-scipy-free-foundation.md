# ADR-0004 — The math foundation is cv2/scipy-free

- **Status:** Accepted (retrofit, recorded 2026-06-28)
- **Deciders:** maintainer
- **Relates to:** ARC-CORE, ARC-DATA, ARC-GEOMETRY, ARC-MODELS, NFR-ARCH-002, NFR-PORT-001
- **Supersedes:** —

## Context

OpenCV (`cv2`) and SciPy are convenient but heavy and platform-sensitive dependencies. If they leak
into the numerical core, the solver path becomes hard to port (embedded targets, minimal
environments) and harder to reason about numerically. Yet board detection genuinely needs OpenCV, and
some IO/image utilities benefit from it.

## Decision

Keep the **math foundation — `core`, `data`, `geometry`, `models` — free of `cv2` and `scipy`.** Both
are confined to the adapter/service layers where they are actually warranted:

- `cv2` → `detect` (board detection), parts of `io`, and image-space services.
- `scipy` → only where a service needs it, never in the foundation.

The foundation uses NumPy (+ stdlib) only.

## Consequences

**Positive**
- The numerical core (models, Lie algebra, optimizer, robust kernels, BA driver) is portable and
  light; it can run where OpenCV/SciPy cannot.
- Clear seam: detection/IO concerns cannot bleed into geometry.

**Negative / costs**
- Occasionally re-implementing a small primitive in NumPy instead of calling SciPy — a deliberate,
  bounded cost.

## Verification

- `tests/contract/test_independence.py::test_math_foundation_is_cv2_and_scipy_free` — asserts no
  `cv2`/`scipy` import anywhere under `core`/`data`/`geometry`/`models`.
- The Python 3.10–3.12 CI matrix exercises the portable path on each version.

Covered requirements: **NFR-ARCH-002, NFR-PORT-001**.

## Alternatives considered

- *Allow `cv2`/`scipy` anywhere.* Rejected — destroys portability of the solver path and the
  detection/geometry seam.
- *Drop OpenCV entirely.* Rejected — board detection legitimately needs it; the right fix is
  confinement, not removal.
