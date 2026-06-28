# Playbook — Add an IO / interop format `[PBK]`

> Recipe for adding a read/write adapter for an external format (alongside Kalibr, COLMAP,
> nerfstudio, MC-Calib). IO is an adapter layer: it may use `cv2`/`scipy` and depends on
> core+data+models, but nothing depends on it. Area: **IO** (ARC-IO).

## When

You need DS-MSP to interoperate with another ecosystem tool's calibration / reconstruction format.

## Steps

1. **Requirement.** Add an `FR-IO-00N` row to [`requirements.csv`](../srs/requirements.csv)
   (`arc_ref=ARC-IO`) describing the format and direction (read, write, or both).
2. **Implement `ds_msp/io/<format>.py`.** Convert between the external format and DS-MSP's neutral
   data containers / model parameters (`ds_msp/data`, the `CameraModel` params). Keep parsing/IO
   concerns here; do not leak format types into geometry or models. Add a public interface entry to
   [`interfaces.md`](../srs/interfaces.md) (IFC-06 family).
3. **Round-trip fidelity.** The core guarantee for IO is **lossless round-trip** where the formats
   allow it (write → read → compare). Document any intentional lossy mapping.
4. **Tests.** Add `tests/io/test_<format>.py`: a round-trip test (and a fixture parsing a small real
   sample where licensing allows). Mark with `@pytest.mark.req("FR-IO-00N")`.
5. **Docs.** Note the new format in the README/interop section.
6. **Verify & gate.** Full gates green; `check_traceability.py --check` green (the new FR is linked).
7. **Release.** `feat:` commit.

## Notes

- Keep large sample datasets out of the tracked tree; use a tiny synthetic or licence-clean fixture
  (CON-05, CON-06). No absolute local paths in tests (they must run in CI).
- A breaking change to an existing format's mapping is a breaking change to a public interface — use
  the `feat!:` / `BREAKING CHANGE` form ([CHANGE_RELEASE_MGMT.md](../management/CHANGE_RELEASE_MGMT.md)).

## Checklist

- [ ] `io/<format>.py` converts to/from neutral containers; no format types escape the adapter
- [ ] round-trip test (+ small licence-clean fixture); `@pytest.mark.req`
- [ ] `interfaces.md` (IFC-06) + README updated
- [ ] DoD + traceability green
