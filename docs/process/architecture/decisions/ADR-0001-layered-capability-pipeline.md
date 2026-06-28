# ADR-0001 — Two-tier layered architecture: capabilities compose into pipelines

- **Status:** Accepted (retrofit, recorded 2026-06-28)
- **Deciders:** maintainer
- **Relates to:** ARC (all), NFR-ARCH-001
- **Supersedes:** —

## Context

DS-MSP must grow across many 3D tasks — intrinsics, extrinsics, conversion, stereo, two-view
geometry, visual odometry — without each new area re-implementing Lie algebra, optimization, or
bundle adjustment, and without the package collapsing into a tangle of cyclic imports. An early
"every service is independent" model proved wrong: some services (a multi-camera rig) genuinely build
on others (single-camera calibration), and pretending otherwise either duplicated code or created
hidden cycles.

## Decision

Adopt a strictly layered, acyclic dependency graph with a **two-tier service split**:

- A shared **math foundation** — `core` → `data` → `geometry`, plus pure-NumPy `models` — holds one
  implementation of each primitive (Lie groups, LM/Schur optimizer, robust kernels, resection, BA).
- **Capabilities** (`ops`, `adapt`, `calib`, `mvg`, `stereo`) are single-purpose and **mutually
  independent**.
- **Pipelines** (`rig`, `vo`) **compose** capabilities downward (`rig → calib`, `vo → mvg`) and stay
  independent of one another. Capabilities never import a pipeline.

A lower layer may never import a higher one; the two tiers give an acyclic graph shaped like a tensor
library (reusable building blocks orchestrated by higher-level drivers).

## Consequences

**Positive**
- One place to fix/improve each primitive; all layers inherit the fix.
- New capability or pipeline added without editing siblings (see playbooks).
- No import cycles possible within the rules; reasoning and testing stay local.

**Negative / costs**
- A genuinely cross-capability need must be pushed *down* into `geometry`/`core`, not solved by a
  sideways import — occasionally more up-front design.
- The capability/pipeline distinction must be kept current as services are added.

## Verification

Enforced, not merely intended:
- import-linter contracts in `pyproject.toml` (`[tool.importlinter]`).
- `tests/contract/test_independence.py` (AST mirror; capability independence + pipeline acyclicity).

Covered requirement: **NFR-ARCH-001**.

## Alternatives considered

- *Flat "all services independent".* Rejected — forces duplication or hidden cycles for legitimately
  layered work (rig-on-calib).
- *Single monolithic package.* Rejected — no enforceable boundaries; primitives drift and duplicate.
