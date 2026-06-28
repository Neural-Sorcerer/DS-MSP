# Test Levels `[TLV]`

> The test taxonomy DS-MSP uses, mapped to the pytest markers that select each level. Referenced by
> the [QA & V&V plan](QA_VV_PLAN.md) and the [Definition of Done](DEFINITION_OF_DONE.md). Markers are
> registered in [`pyproject.toml`](../../../pyproject.toml).

| Level | Marker | What it covers | Speed | When it runs |
|-------|--------|----------------|-------|--------------|
| **Unit** | *(none)* | A single function/class in isolation (model math, lie ops, kernels) | fast | every PR |
| **Contract** | `contract` | Model-agnostic guarantees every model must meet (shapes, dtypes, unit bearings, serialize round-trip, protocol satisfaction) | fast | every PR |
| **Gradient-check** | `jac` | Analytic Jacobian vs Richardson-extrapolated finite differences ≤1e-6 | fast | every model PR |
| **Integration** | `integration` | Multiple components end-to-end on **synthetic** scenarios (calibrate → poses → residuals) | medium | every PR |
| **Statistical** | `slow` | Long-running statistical properties (e.g. robust-kernel behaviour over many trials) | slow | every PR (allowed to be slower) |
| **Real-data validation** | `realdata` | Validation against real datasets (e.g. calibration parity vs published intrinsics) | slow, dataset-gated | **pre-release / nightly**, skipped in ordinary PR CI |

## Selection

- Full suite (PR default): `pytest -q`
- A single level: `pytest -m jac`, `pytest -m contract`, `pytest -m "integration"`, …
- Real-data only (where the dataset is present): `pytest -m realdata`

## Requirement linkage

Every suite carries a `@pytest.mark.req("FR-…", "NFR-…")` marker tying it to the requirement(s) it
verifies (discovered by `tools/check_traceability.py`). A new test that verifies a requirement **must**
carry the marker, or traceability is incomplete; a new requirement whose `verify_method` points at a
`tests/` path **must** have a linked test, or the requirement is an orphan and CI fails.

## Determinism

All tests are deterministic via fixed seeds (NFR-REPRO-001). A test that needs randomness seeds an
explicit generator; no test depends on wall-clock time or unseeded RNG.
