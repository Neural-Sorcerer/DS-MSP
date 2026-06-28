# CI/CD Pipeline `[CICD]`

> The automated pipeline that enforces the quality gates and ships releases (ISO/IEC/IEEE 12207
> §6.3). Maps each GitHub Actions workflow to what it guards, and records how to keep this document in
> sync with the workflows.

## Workflows

### `ci.yml` — on every pull request and push to `main`

| Job | Steps | Guards |
|-----|-------|--------|
| `lint + types + layering` | `ruff check .`; `lint-imports`; `mypy ds_msp/core` | code style; the layered architecture (NFR-ARCH-001/002); typed core |
| `tests (py3.10/3.11/3.12)` | `pytest -q --cov=ds_msp` on the version matrix | all synthetic test levels; portability (NFR-PORT-001) |
| **`governance`** | `python tools/check_traceability.py --check`; `python tools/check_tree_hygiene.py` | requirement↔test↔ADR traceability; no tracked local-only/leak content (NFR-PRIV-001) |

The `governance` job uses only the standard library (no extra install), so it is fast and cannot break
on dependency drift.

### `release.yml` — on push to `main`

`release-please` maintains a release PR (version bump + `CHANGELOG.md` from Conventional Commits).
Merging it cuts the tag + GitHub Release and triggers the PyPI publish via **Trusted Publishing
(OIDC)** — no stored token (CON-07, ADR-0006). See [`RELEASING.md`](../../../RELEASING.md).

### `deploy-pages.yml`

Builds and publishes the documentation site.

## The release gate

A release of any **release-gated** requirement (FR-CALIB-001, FR-RIG-001, NFR-NUM-004) requires the
pre-release / nightly validation job — which runs the `realdata` tests against real datasets — to be
green, and `check_traceability.py --release` to pass. `realdata` tests are dataset-gated and skipped in
ordinary PR CI to keep PRs fast (ADR-0006).

## Lifecycle mapping (ISO/IEC/IEEE 12207)

```
requirement (§6.4.1)  →  design / ADR (§6.4.3, 42010)  →  branch (§6.3.2)
   →  implement (§6.4.4: ruff / mypy / import-linter + independence)
   →  verify-synthetic (§6.4.6: pytest, -m jac, contract, integration)
   →  validate-real-data (§6.4.9: realdata tests/scripts)
   →  review (§6.3.7: CODEOWNERS)  →  release (§6.4.10: release-please + OIDC)
```

## Keeping this document current

This file is **descriptive of the workflows**, which are authoritative. When a workflow changes:

1. Update the matching row/section here in the same PR.
2. If a new gate is added, add it to the [Definition of Done](../quality/DEFINITION_OF_DONE.md) and the
   PR template.
3. If a gate enforces a requirement, ensure that requirement's `verify_method` points at the workflow
   or test, so traceability stays complete.
