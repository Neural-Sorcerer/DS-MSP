# Definition of Done `[DOD]`

> A change is **done** only when every box below is checked. This is the contract the PR template
> encodes and reviewers enforce. "Done" means merged-ready, not "code written".

## Every change

- [ ] **Linked requirement** — the change references the FR/NFR ID(s) it implements or affects (or
      adds a new row to [`requirements.csv`](../srs/requirements.csv) with area, `arc_ref`,
      `code_module`, `verify_method`, `status`).
- [ ] **Tests** — added/updated at the right level ([test-levels.md](test-levels.md)); each new test
      verifying a requirement carries `@pytest.mark.req(...)`.
- [ ] **Synthetic verification green** — full suite passes locally and in CI on 3.10/3.11/3.12.
- [ ] **Lint / types / layering green** — `ruff check .`, `mypy` (typed surface), `lint-imports` /
      `test_independence.py` all clean. A new cross-layer edge has an ADR + updated contracts.
- [ ] **Governance green** — `check_traceability.py --check` and `check_tree_hygiene.py` clean.
- [ ] **Docs updated** — public API/behaviour changes update the relevant docs; a new interface updates
      [`interfaces.md`](../srs/interfaces.md).
- [ ] **No leakage** — no internal R&D / process content, secrets, or absolute local paths in tracked
      files (CON-06, NFR-PRIV-001).
- [ ] **Conventional Commit** — message follows Conventional Commits (drives versioning/changelog).

## Additionally, for a new camera model

- [ ] Implements the full `CameraModel` contract; passes `test_camera_model_contract.py`.
- [ ] Analytic Jacobians pass `test_gradcheck.py` at ≤1e-6 (`-m jac`).
- [ ] Registered in the model registry; round-trip (de)serialization tested.
- [ ] Followed [playbooks/add-a-camera-model.md](../playbooks/add-a-camera-model.md).

## Additionally, for a release-gated change (FR-CALIB-001, FR-RIG-001, NFR-NUM-004)

- [ ] A `realdata` test validates the behaviour on a real dataset within the stated tolerance.
- [ ] `check_traceability.py --release` passes (synthetic **and** real-data coverage present).
- [ ] The pre-release validation job is green (ADR-0006).

## Additionally, for a defect fix

- [ ] A regression test reproduces the defect (fails before, passes after).
- [ ] The tracking issue is updated per [ISSUE_DEFECT_PROCESS.md](../management/ISSUE_DEFECT_PROCESS.md).
