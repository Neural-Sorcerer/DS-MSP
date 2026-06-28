<!-- Thank you for contributing to DS-MSP. The Definition of Done is the bar for merge:
     docs/process/quality/DEFINITION_OF_DONE.md -->

## What & why

<!-- Brief description of the change and the motivation. -->

## Linked requirements / issues

- Requirement(s): <!-- e.g. FR-MODEL-006, NFR-NUM-001 — add a row to docs/process/srs/requirements.csv if new -->
- Closes: <!-- #issue -->
- ADR (if architecturally significant): <!-- ADR-NNNN -->

## Type of change

- [ ] `fix:` bug fix (patch)
- [ ] `feat:` new feature (minor)
- [ ] breaking change (`feat!:` / `BREAKING CHANGE:` — major; updates an IFC-* interface)
- [ ] `docs:` / `chore:` / `refactor:` / `test:` (no release on its own)

## Definition of Done

- [ ] Linked FR/NFR ID(s) above (or new row added to `requirements.csv`)
- [ ] Tests added/updated at the right level; new tests carry `@pytest.mark.req(...)`
- [ ] **Synthetic verification** green: `pytest` on 3.10/3.11/3.12, `ruff`, `mypy`, `lint-imports`
- [ ] **Governance** green: `python tools/check_traceability.py --check` and `python tools/check_tree_hygiene.py`
- [ ] Docs updated (and `interfaces.md` if a public interface changed)
- [ ] No internal-process content, secrets, or absolute local paths in tracked files
- [ ] Conventional Commit message

## Release-gated change? (FR-CALIB-001, FR-RIG-001, NFR-NUM-004, or their areas)

- [ ] N/A
- [ ] **Real-data validation** added (`@pytest.mark.realdata`) and green on a real dataset within the stated tolerance
- [ ] `python tools/check_traceability.py --release` passes

<!-- High-risk changes (new model, solver/optimizer, calibration accuracy) are expected to have gone
     through additional local deep-verification before this PR; the boxes above are the public contract. -->
