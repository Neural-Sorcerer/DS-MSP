# Branching & Contribution Model `[BRN]`

> The branch model, commit conventions, and the gate every change passes before reaching `main`
> (ISO/IEC/IEEE 12207 §6.3.2). Thin by design — root [`CONTRIBUTING.md`](../../../CONTRIBUTING.md)
> points here.

## 1. Branch model

- `main` is always releasable and protected. **No direct commits to `main`** — every change lands via
  a reviewed pull request.
- Work happens on a short-lived **feature branch off `main`**, named by Conventional-Commit type:

  | Prefix | For |
  |--------|-----|
  | `feat/<slug>` | a new feature |
  | `fix/<slug>` | a bug fix |
  | `docs/<slug>` | documentation |
  | `chore/<slug>` / `refactor/<slug>` / `test/<slug>` | non-shipping work |

- Large multi-part work may stack PRs (each independently green) rather than one giant branch.

## 2. Worktrees for parallel work

Independent changes can be developed in parallel with **git worktrees** (`git worktree add`), one
branch per working directory, so unrelated work never shares a dirty tree. Each worktree still branches
off `main` and merges via its own PR.

## 3. Commit conventions

- [Conventional Commits](https://www.conventionalcommits.org/) — drives versioning & changelog
  ([CHANGE_RELEASE_MGMT.md](CHANGE_RELEASE_MGMT.md)).
- Keep unrelated concerns in **separate commits** (e.g. a lint/debt cleanup is its own commit, not
  buried in a feature) so history and review stay legible.
- Every commit ends with the attribution trailer
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` (where applicable).

## 4. The PR gate

A PR is mergeable only when the [Definition of Done](../quality/DEFINITION_OF_DONE.md) is met and CI is
green: lint + types + layering, the test matrix, and the `governance` job
([CICD_PIPELINE.md](CICD_PIPELINE.md)). The PR description links the FR/NFR ID(s) and ticks the
synthetic-verification and (where applicable) real-data-validation boxes. `docs/process/` and ADRs are
owner-reviewed via `CODEOWNERS`.

## 5. Develop → validate → publish

Every change follows the same flow before it is public:

```
branch off main → implement → verify (synthetic, all gates) → validate (real data, if release-gated)
              → review (CODEOWNERS) → merge → release-please
```

This is the contributor-facing shape of the lifecycle in [CICD_PIPELINE.md](CICD_PIPELINE.md); the
guarantee is that nothing reaches `main` unverified and nothing ships to PyPI without the required
validation.

## 6. Adding something new?

Use the matching [playbook](../playbooks/) — they walk REQ → ADR (if needed) → code → tests → docs →
release for the common cases (camera model, robust kernel, IO format, pipeline capability).
