# Playbook — Add a robust kernel `[PBK]`

> Recipe for adding an M-estimator / robust loss to the shared optimizer. Kernels live in the
> dependency-free core ([ADR-0001](../architecture/decisions/ADR-0001-layered-capability-pipeline.md),
> [ADR-0004](../architecture/decisions/ADR-0004-cv2-scipy-free-foundation.md)), so every calibration /
> BA path inherits them. Area: **CORE** (ARC-CORE).

## When

You need a new robust loss (e.g. a new Barron variant, a different scale estimator) used by the LM
solver and the bundle-adjustment drivers.

## Steps

1. **Requirement / decision.** Usually covered by the existing robustness NFRs; if it changes solver
   behaviour materially, note it (an ADR if it's a design choice such as a new default).
2. **Implement in `ds_msp/core/robust.py`.** Pure NumPy. A kernel provides the loss and its IRLS
   weight (ρ, ψ/influence → weight), composable with the existing scale estimation (`mad_scale`,
   studentized leverage) and the GNC schedule. Keep it a small, orthogonal addition — no edits to
   sibling kernels.
3. **Wire into the optimizer.** Expose it through `ds_msp/core/optimize.py` (`lm_solve` / `schur_lm`)
   via the `robust_kernel` / `robust_scale` selection so callers opt in by name; do not special-case it
   in any capability.
4. **Tests.**
   - Add identities/weight-curve tests next to the existing core robust tests
     (`tests/core/test_robust*`); assert the IRLS weight and scale behave as specified.
   - Add an outlier-resistance test (the kernel beats plain L2 under a known outlier fraction), in the
     style of the existing rig robustness tests.
   - Determinism via fixed seeds (NFR-REPRO-001); mark with `@pytest.mark.req(...)`.
5. **Docs.** If it becomes a selectable option in the public calibration API, note it in
   [`interfaces.md`](../srs/interfaces.md).
6. **Verify & gate.** Full gates green; `check_traceability.py --check` green.
7. **Release.** `feat:` (new option) or `fix:` (correcting an existing one).

## Checklist

- [ ] kernel in `core/robust.py`, pure NumPy, orthogonal to existing kernels
- [ ] selectable via `optimize.py` (no capability special-casing)
- [ ] weight/scale identities + outlier-resistance tests; seeded; `@pytest.mark.req`
- [ ] DoD + traceability green
