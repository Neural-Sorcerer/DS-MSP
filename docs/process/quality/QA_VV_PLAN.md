# Quality Assurance & Verification/Validation Plan `[QVP]`

> Standards-informed after ISO/IEC/IEEE 29119-2 (test process) and 12207 (V&V activities). Defines
> *how* DS-MSP is verified (does it meet the spec?) and validated (does it work on real data?), the
> entry/exit criteria for each stage, and the machine-enforced release gate.

## 1. Verification vs validation

- **Verification** — the software meets its specification. Done with the deterministic, dataset-free
  test levels ([test-levels.md](test-levels.md)): unit, contract, gradient-check, integration,
  statistical. Runs on every PR.
- **Validation** — the software produces correct results on **real** data (real lenses, real noise,
  real board detection and pose distributions). Done with `realdata` tests/scripts against real
  datasets. Required before a release for release-gated requirements (ADR-0006).

## 2. Quality gates (per PR, all must pass)

The `lint + types + layering` and `tests` CI jobs ([CICD_PIPELINE.md](../management/CICD_PIPELINE.md))
enforce, on every pull request:

1. **Lint** — `ruff check .` clean.
2. **Layering** — `lint-imports` (import-linter contracts) clean; mirrored by
   `tests/contract/test_independence.py`.
3. **Types** — `mypy` clean on the typed core surface.
4. **Tests** — `pytest` green on the Python 3.10 / 3.11 / 3.12 matrix, with coverage reported.
5. **Governance** — `tools/check_traceability.py --check` and `tools/check_tree_hygiene.py` clean
   (no orphan requirements, no dangling REQ↔test links, ADR index in sync, no tracked local-only
   content).

## 3. Entry / exit criteria

**Verification (synthetic) — entry:** a change with the relevant tests added/updated and its
requirement marker(s) in place. **Exit:** all §2 gates green.

**Validation (real data) — entry:** verification passed; the change touches a release-gated
requirement (FR-CALIB-001, FR-RIG-001, NFR-NUM-004) or its area. **Exit:** the linked `realdata`
test(s) green on the real dataset, within the tolerance stated in the requirement.

**Release — entry:** every release-gated requirement has *both* a synthetic and a `realdata` test
linked and green (`check_traceability.py --release`). **Exit:** release-please cuts the tag and the
PyPI OIDC publish succeeds (ADR-0006, CON-07).

## 4. Coverage expectations

- New public behaviour ships with tests at the appropriate level (unit + contract for a model;
  integration for a pipeline; `realdata` for release-gated accuracy claims).
- Coverage is reported per PR (`pytest --cov`); the bar is *meaningful* coverage of new code paths,
  not a single global percentage. Numerical claims (accuracy, tolerances) are backed by an asserting
  test, never by prose alone.

## 5. The release gate

The policy: no release-gated requirement may ship without a green synthetic **and** a green real-data
test. This is enforced two ways, one active and one planned:

- **Active:** `tools/check_traceability.py --release` fails if a release-gated requirement lacks
  `realdata` coverage (run it before cutting a release).
- **Planned (RSK-07):** a pre-release / nightly validation job that runs the `realdata` suite against
  real datasets and must be green before release-please publishes. This job is **not yet wired** — until
  it is, the real-data validation is run manually as part of the release checklist.

See ADR-0006 and [CHANGE_RELEASE_MGMT.md](../management/CHANGE_RELEASE_MGMT.md).

## 6. Internal verification protocol (high-risk changes)

Beyond the public gates above, high-risk changes (new camera model, solver/optimizer changes,
anything affecting calibration accuracy) additionally go through a **local deep-verification
protocol** before entering the public PR flow: extended adversarial review of derivations, additional
synthetic stress scenarios, and broader real-data validation than the release gate requires. This
protocol is a local development practice; its artifacts are kept out of the tracked tree (CON-06,
NFR-PRIV-001) and are not a prerequisite for external contributors — the public gates in §2–§5 are the
contract every change is held to.

## 7. Defect handling

Failures and regressions are tracked per [ISSUE_DEFECT_PROCESS.md](../management/ISSUE_DEFECT_PROCESS.md)
(IEEE-1044-style taxonomy). A fixed defect ships with a regression test that fails before the fix and
passes after.
