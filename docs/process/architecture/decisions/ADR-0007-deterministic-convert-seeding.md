# ADR-0007 — Deterministic shape-parameter sweep in model conversion

- **Status:** Accepted (recorded 2026-06-28)
- **Deciders:** maintainer
- **Relates to:** ARC-ADAPT, FR-ADAPT-001, NFR-NUM-007
- **Supersedes:** —

## Context

`ds_msp.adapt.convert()` fits a target model's parameters to reproduce a source model's
projection over the image (image-free conversion), refined as a multi-start least-squares from a
linear seed plus `n_restarts` **random** dispersed shape seeds, keeping the lowest-cost fit.

Real-data study across multiple camera models exposed a defect: with the default `n_restarts`, the
random restarts are a basin-of-attraction *lottery*. For an EUCM+ target whose `beta` lies far from
the linear seed's `beta=1`, all default restarts could settle in a nearby wrong basin — so the
conversion failed to reach the global optimum and even **self-conversion** (a model into its own
class) landed several pixels away instead of at machine precision. Whether it succeeded depended on
the random seed, making conversion non-reproducible and, for EUCM+ targets, unreliable.

## Decision

Add an **always-on deterministic shape-parameter sweep** to the multi-start seed set: for each
finitely-bounded shape parameter, generate seeds that move *that one parameter* to a few fixed
fractions of its bound range (others held at the linear seed). The random `n_restarts` seeds are
retained on top for joint multi-parameter basins; `n_restarts` keeps its meaning (count of *random*
restarts) for backward compatibility.

The deterministic sweep guarantees that a shape optimum far from the linear seed is always reached,
independent of the random seed — conversion becomes **reproducible** and self-conversion becomes
**exact** for every model.

## Consequences

**Positive**
- Self-conversion is exact (machine precision) for every model; EUCM+ becomes a reliable target.
- Conversion is deterministic — no random-restart lottery; results are reproducible across seeds.
- Pure-numpy/scipy; no new dependency; no change to the public `convert()` signature.

**Negative / costs**
- A few extra least-squares starts per conversion (proportional to the target's finitely-bounded
  shape-parameter count). `convert()` is an occasional adapter call, so the cost is immaterial.

## Verification

- `tests/adapt/test_convert_robustness.py` (NFR-NUM-007): self-conversion exact for all models;
  DS+ a faithful universal target; EUCM+ self-convert deterministic across seeds.
- `tests/realdata/test_mccalib_calibration.py`: real-data self-convert exactness and DS+-as-target
  fidelity per FOV band.

## Alternatives considered

- *Raise the default `n_restarts`.* Rejected — still a lottery (more tickets, not a guarantee),
  ~3× slower by default, and non-reproducible.
- *Model-specific seeding (hardcode EUCM+ `beta` grid).* Rejected — leaks model knowledge into the
  decoupled converter; the bounded-parameter sweep is generic over any `CameraModel`.
