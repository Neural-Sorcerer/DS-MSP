# ADR-0006 — Synthetic-then-real-data release gate

- **Status:** Accepted (recorded 2026-06-28)
- **Deciders:** maintainer
- **Relates to:** ARC-CALIB, ARC-RIG, NFR-NUM-004, QVP (QA & V&V plan)
- **Supersedes:** —

## Context

Calibration code can pass synthetic tests yet fail on real lenses (real noise, real board detection,
real pose distributions). "Verified on synthetic, validated on real data before going public" is a
core project principle and must be enforced mechanically, not left to discipline — while keeping
per-PR CI fast and not requiring large datasets on every contributor's machine.

## Decision

Adopt a two-stage verification lifecycle with a machine-enforced release gate:

1. **Verify (synthetic)** — every PR runs the full deterministic test suite (unit, contract,
   gradient-check, integration on synthetic scenarios). Fast, dataset-free.
2. **Validate (real data)** — requirements marked **`release_gated`** in
   [`../srs/requirements.csv`](../srs/requirements.csv) must additionally have a `realdata` test
   (`@pytest.mark.realdata`) that runs against a real dataset.

`realdata` tests are **dataset-gated**: skipped in ordinary PR CI (keeping PRs fast) and required-green
in a pre-release / nightly job that the release depends on. `tools/check_traceability.py --release`
fails if any release-gated requirement lacks a `realdata` test. **No release-gated requirement may
ship without both a synthetic and a real-data test linked and green.**

## Consequences

**Positive**
- The project principle becomes an enforced gate: no public release without real-data validation.
- PRs stay fast; the heavy validation runs where datasets are available.

**Negative / costs**
- Maintaining real datasets and the pre-release job; release-gated work has a higher bar (by design).

## Verification

- `tools/check_traceability.py --release` — release-gated reqs must have `realdata` coverage.
- Current release-gated requirements: **FR-CALIB-001, FR-RIG-001, NFR-NUM-004** (the last is the
  real-data calibration-parity validation; see the QA & V&V plan).

## Alternatives considered

- *Run real-data tests on every PR.* Rejected — slow and requires large datasets everywhere.
- *Rely on reviewer discipline for real-data checks.* Rejected — not enforceable; the whole point is
  a machine gate.
