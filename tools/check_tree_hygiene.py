#!/usr/bin/env python3
"""Tracked-tree structural hygiene gate (pure stdlib, CI-safe).

Fails if any tracked file would expose local-only working areas. It encodes ONLY
structural/path rules that are already public in ``.gitignore`` (so this script reveals
nothing sensitive and is safe to ship in the repo). The richer semantic denylist of
internal codenames stays in a local, unpublished pre-push guard — defense in depth.

Run in CI on every PR:  python tools/check_tree_hygiene.py
"""

from __future__ import annotations

import re
import subprocess
import sys

# Path patterns that must never appear among tracked files. All of these directory/file
# names are already declared in the public .gitignore, so listing them here leaks nothing.
FORBIDDEN_PATH = [
    (re.compile(r"(^|/)\.ai/"), "local R&D working area .ai/"),
    (re.compile(r"(^|/)\.claude/"), "local tooling .claude/"),
    (re.compile(r"(^|/)CLAUDE\.md$"), "local CLAUDE.md"),
    (re.compile(r"(^|/)CLAUDE\.local\.md$"), "local CLAUDE.local.md"),
    (re.compile(r"(^|/)CAREER_ROADMAP\.md$"), "local CAREER_ROADMAP.md"),
    (re.compile(r"(^|/)docs/research/"), "local docs/research/"),
    (re.compile(r"(^|/)simulations/"), "local simulations/"),
]
# Absolute home paths leaking into tracked content are also forbidden.
FORBIDDEN_PATH.append((re.compile(r"(^|/)Users/"), "absolute /Users/ path as a tracked file"))


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def main() -> int:
    violations = []
    for path in tracked_files():
        for rx, why in FORBIDDEN_PATH:
            if rx.search(path):
                violations.append(f"{path}  ({why})")
                break
    if violations:
        print("TREE HYGIENE: FAIL — local-only paths are tracked:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("TREE HYGIENE: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
