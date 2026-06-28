#!/usr/bin/env python3
"""Tracked-tree structural hygiene gate (pure stdlib, CI-safe).

Fails if any *tracked* file is also matched by the repo's ignore rules — i.e. content that
is meant to stay local (declared in ``.gitignore``) has been committed. The set of
local-only paths is defined solely by ``.gitignore`` (the standard mechanism for declaring
them); this script enumerates none of them itself, so it reveals nothing and is safe to
ship. New local-only entries added to ``.gitignore`` are covered automatically.

The richer *semantic* denylist of internal codenames lives in a separate, local-only
pre-push guard — defense in depth.

Run in CI on every PR:  python tools/check_tree_hygiene.py
"""

from __future__ import annotations

import subprocess
import sys


def _git(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True, input=stdin)


def main() -> int:
    tracked = [ln for ln in _git("ls-files").stdout.splitlines() if ln.strip()]
    if not tracked:
        print("TREE HYGIENE: OK (no tracked files)")
        return 0

    # A tracked file that the ignore rules would match is a breach: local-only content has
    # been committed. `--no-index` makes check-ignore evaluate the rules even for paths that
    # are already tracked (tracking would otherwise mask them). Exit code: 0 = some matched,
    # 1 = none matched, >1 = error.
    res = _git("check-ignore", "--no-index", "--stdin", stdin="\n".join(tracked) + "\n")
    if res.returncode > 1:
        print(f"TREE HYGIENE: ERROR running git check-ignore:\n{res.stderr}", file=sys.stderr)
        return 2
    violations = sorted({ln.strip() for ln in res.stdout.splitlines() if ln.strip()})

    if violations:
        print("TREE HYGIENE: FAIL — local-only (git-ignored) paths are tracked:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print(f"TREE HYGIENE: OK ({len(tracked)} tracked files, none git-ignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
