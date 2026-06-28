# Security Policy

## Supported versions

The latest released minor of DS-MSP on PyPI receives security fixes. Pre-release/`main` is fixed
directly.

## Reporting a vulnerability

**Please do not open a public issue for security problems.** Report privately via GitHub Security
Advisories:

➡️ https://github.com/Munna-Manoj/DS-MSP/security/advisories/new

Include, where possible: affected version, a minimal reproduction (inputs/seed), the impact, and any
suggested fix. You can expect an acknowledgement and, once triaged, a coordinated fix and disclosure.

## Scope

DS-MSP is a numerical / geometry library. The most relevant classes of issue are:

- Vulnerabilities in how the library **parses external files** (the IO adapters: Kalibr, COLMAP,
  nerfstudio, MC-Calib) — e.g. unsafe deserialization of untrusted input.
- Vulnerabilities introduced via a **third-party dependency** (numpy, opencv, scipy, pyyaml) —
  tracked as risk RSK-08.

Loading untrusted calibration/reconstruction files should be treated with the same caution as any
untrusted input; prefer the safe-loading paths the library provides.

## Handling

A confirmed vulnerability is tracked as an S1/S2 defect
(`docs/process/management/ISSUE_DEFECT_PROCESS.md`), fixed with a regression test, and shipped as a
patch release via the standard release path (`docs/process/management/CHANGE_RELEASE_MGMT.md`).
