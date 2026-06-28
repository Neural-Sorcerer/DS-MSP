# Issue & Defect Process `[IDP]`

> Standards-informed after IEEE 1044 (classification of software anomalies). Defines how issues and
> defects are classified, prioritised, and tracked through to a verified fix. Issues live as **GitHub
> Issues**; their numbers are the `ISS-<n>` IDs in the traceability chain.

## 1. Item types

| Type | Meaning |
|------|---------|
| **Defect** | Behaviour deviates from a requirement / spec (wrong result, crash, regression). |
| **Enhancement** | New capability or improvement (maps to a new/changed FR/NFR). |
| **Task / chore** | Non-behavioural work (tooling, docs, refactor) — no requirement change. |

## 2. Defect attributes (IEEE-1044)

Every defect records:

- **Severity** — impact if it occurs (see §3).
- **Component** — the `ARC-*` component / package (e.g. ARC-CALIB, ARC-MODELS).
- **Defect type** — one of: *numerical-accuracy*, *logic*, *interface/contract*, *build/CI*,
  *documentation*, *performance*, *security*.
- **Affected requirement(s)** — the FR/NFR ID(s), if any.
- **Reproduction** — numbers, not screenshots: inputs, expected vs actual, seed/dataset.

## 3. Severity

| Severity | Definition | Example |
|----------|------------|---------|
| **S1 Critical** | Wrong numerical result on a supported path, data loss, or release blocker | calibration converges to a wrong basin; Jacobian regression |
| **S2 Major** | Significant function broken with no easy workaround | an IO format round-trip loses data |
| **S3 Minor** | Limited impact / workaround exists | misleading error message |
| **S4 Trivial** | Cosmetic / docs | typo, formatting |

Accuracy regressions and contract/Jacobian failures are **S1 by default** (they violate the project's
core numerical guarantees).

## 4. States

```
new → triaged → in-progress → in-review → verified → closed
                     │
                     └──→ wont-fix / duplicate (closed with reason)
```

- **triaged** — type, severity, component, defect type, affected requirement assigned.
- **verified** — the fix is merged *and* a regression test (fails-before / passes-after) is green in CI.
- A defect is never closed without either a regression test or an explicit, recorded reason.

## 5. Linkage to the rest of the system

- A defect that reveals a missing/incorrect requirement updates [`requirements.csv`](../srs/requirements.csv).
- A fix follows the [Definition of Done](../quality/DEFINITION_OF_DONE.md) (regression test mandatory).
- Recurring or high-impact defects are promoted to the [Risk Register](RISK_REGISTER.md).
- Security issues follow [`SECURITY.md`](../../../SECURITY.md), not the public issue tracker.

## 6. Templates

GitHub issue templates under `.github/ISSUE_TEMPLATE/` capture these fields (bug report with
severity/component/defect-type/reproduction; feature request linked to an FR/NFR).
