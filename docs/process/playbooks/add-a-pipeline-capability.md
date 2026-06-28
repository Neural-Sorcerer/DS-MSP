# Playbook — Add a capability or pipeline `[PBK]`

> Recipe for adding a new **capability** (single-purpose building block) or **pipeline** (orchestrator
> that composes capabilities), honouring the two-tier acyclic model
> ([ADR-0001](../architecture/decisions/ADR-0001-layered-capability-pipeline.md)). This is the path
> for new 3D functionality (a new geometry service, a new end-to-end pipeline).

## Decide: capability or pipeline?

- **Capability** (`ops`, `adapt`, `calib`, `mvg`, `stereo`) — single purpose; depends only on the math
  foundation; **imports no other service**.
- **Pipeline** (`rig`, `vo`) — orchestrates capabilities downward; **imports capabilities, never
  another pipeline**, and no capability imports it.

If your work needs two capabilities together, it is a *pipeline*. If it would make two capabilities
import each other, push the shared part **down** into `geometry`/`core` instead.

## Steps

1. **Requirement / ADR.** Add the FR row(s) to [`requirements.csv`](../srs/requirements.csv) with the
   right `arc_ref` (new `ARC-*` row in [`components.csv`](../architecture/components.csv) with its
   `depends_on`). A new layer edge or a new tier member warrants an ADR.
2. **Reuse the foundation.** Build on `core` (lie, optimize, robust), `geometry` (resection,
   averaging, graph, BA driver), `data` containers, and the `CameraModel` protocol. Do **not**
   re-implement a primitive that already exists — that is the whole point of the architecture.
3. **Implement** under `ds_msp/<name>/`. Respect the import rules: a capability imports no sibling
   service; a pipeline imports only capabilities (downward).
4. **Enforce the new edges.** Update the import-linter contracts in
   [`pyproject.toml`](../../../pyproject.toml) and the lists in `tests/contract/test_independence.py`
   (`CAPABILITIES` / `PIPELINES`) so the layering stays machine-checked. CI fails if you add an illegal
   edge — by design.
5. **Tests.** Unit + `integration` tests on synthetic scenarios; mark with `@pytest.mark.req(...)`. If
   the capability makes an accuracy claim that must hold on real data, add a `realdata` test and mark
   the requirement `release_gated` (ADR-0006).
6. **Docs.** Update [ARCHITECTURE.md](../architecture/ARCHITECTURE.md) (component table) and
   [`interfaces.md`](../srs/interfaces.md) if a public API is added.
7. **Verify & gate.** All gates green; `check_traceability.py --check` green.
8. **Release.** `feat:` commit.

## Checklist

- [ ] correct tier chosen; import rules respected (no sibling/cross-tier edges)
- [ ] reuses core/geometry/data/contract — no duplicated primitive
- [ ] import-linter contracts + `test_independence.py` updated to cover the new package
- [ ] integration tests (+ `realdata` if release-gated); `@pytest.mark.req`
- [ ] ARCHITECTURE.md + components.csv + interfaces.md updated
- [ ] DoD + traceability green
