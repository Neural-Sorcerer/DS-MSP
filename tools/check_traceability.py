#!/usr/bin/env python3
"""Requirements <-> tests traceability governance gate (pure stdlib).

Validates the DS-MSP requirements registry against the architecture registry and the
test suite, and (re)generates the traceability matrix. Keeps governance live and current:
drift (orphan requirements, dangling/typo links, ADR-index or matrix out of sync) fails CI.

Modes:
  --check    (default, per-PR CI): structural + traceability rules. Fails on any drift.
  --release  (pre-release job): additionally require every release-gated requirement to
             have a real-data (``realdata``-marked) test linked. This is the machine teeth
             behind "nothing publishes without real-data validation".
  --write    (local): regenerate docs/process/traceability/TRACEABILITY.md, then exit.

The requirement<->test link is the ``@pytest.mark.req("FR-...", "NFR-...")`` marker (or a
module-level ``pytestmark``), discovered by AST so it cannot drift from a separate file.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ_CSV = ROOT / "docs/process/srs/requirements.csv"
COMP_CSV = ROOT / "docs/process/architecture/components.csv"
ADR_DIR = ROOT / "docs/process/architecture/decisions"
ADR_INDEX = ADR_DIR / "INDEX.md"
TESTS_DIR = ROOT / "tests"
MATRIX = ROOT / "docs/process/traceability/TRACEABILITY.md"

ID_RE = re.compile(r"^(FR|NFR)-[A-Z]+-\d{3}$")
ADR_FILE_RE = re.compile(r"^ADR-(\d{4})-[a-z0-9-]+\.md$")


# --------------------------------------------------------------------------- loaders
def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def scan_test_markers() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return (req_id -> [test locations], req_id -> [test locations that are realdata]).

    Detects both ``@pytest.mark.req(...)`` decorators and module-level
    ``pytestmark = pytest.mark.req(...)`` / ``pytestmark = [pytest.mark.req(...), ...]``.
    A location is flagged realdata when the same function/module also carries a
    ``realdata`` marker.
    """
    refs: dict[str, list[str]] = {}
    realdata_refs: dict[str, list[str]] = {}

    def mark_names_and_reqargs(call_or_attr):
        """Given an AST node for a marker expr, return (marker_name, [str args])."""
        node = call_or_attr
        args: list[str] = []
        if isinstance(node, ast.Call):
            func = node.func
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    args.append(a.value)
        else:
            func = node
        # walk attribute chain to get the trailing name, requiring 'mark' in the chain
        names = []
        f = func
        while isinstance(f, ast.Attribute):
            names.append(f.attr)
            f = f.value
        if names and "mark" in names:
            return names[0], args            # names[0] is the trailing attr (e.g. 'req')
        return None, args

    def collect_markers(decorator_list, extra_exprs):
        found: dict[str, list[str]] = {}
        is_realdata = False
        exprs = list(decorator_list) + list(extra_exprs)
        for d in exprs:
            name, args = mark_names_and_reqargs(d)
            if name == "req":
                found.setdefault("req", []).extend(args)
            elif name == "realdata":
                is_realdata = True
        return found.get("req", []), is_realdata

    for pyf in sorted(TESTS_DIR.rglob("test_*.py")):
        rel = pyf.relative_to(ROOT)
        try:
            tree = ast.parse(pyf.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        # module-level pytestmark
        module_reqs: list[str] = []
        module_realdata = False
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if "pytestmark" in targets:
                    exprs = (node.value.elts if isinstance(node.value, (ast.List, ast.Tuple))
                             else [node.value])
                    r, rd = collect_markers([], exprs)
                    module_reqs += r
                    module_realdata = module_realdata or rd
        for rid in module_reqs:
            refs.setdefault(rid, []).append(f"{rel} (module)")
            if module_realdata:
                realdata_refs.setdefault(rid, []).append(f"{rel} (module)")
        # per-function decorators
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                r, rd = collect_markers(node.decorator_list, [])
                loc = f"{rel}::{node.name}"
                for rid in r:
                    refs.setdefault(rid, []).append(loc)
                    if rd or module_realdata:
                        realdata_refs.setdefault(rid, []).append(loc)
    return refs, realdata_refs


# --------------------------------------------------------------------------- checks
def validate(release: bool) -> tuple[list[str], list[dict], dict[str, list[str]]]:
    errors: list[str] = []
    reqs = load_rows(REQ_CSV)
    comps = {c["id"] for c in load_rows(COMP_CSV)}
    refs, realdata_refs = scan_test_markers()

    seen: set[str] = set()
    known_ids = {r["id"] for r in reqs}

    for r in reqs:
        rid, status = r["id"], r["status"]
        # 1. ID format + uniqueness
        if not ID_RE.match(rid):
            errors.append(f"{rid}: malformed ID (expect FR-AREA-NNN / NFR-AREA-NNN)")
        if rid in seen:
            errors.append(f"{rid}: duplicate ID")
        seen.add(rid)
        # 2. ARC integrity
        if r["arc_ref"] not in comps:
            errors.append(f"{rid}: arc_ref '{r['arc_ref']}' not in components.csv")
        # 3. verify_method present
        vm = r["verify_method"].strip()
        if not vm:
            errors.append(f"{rid}: empty verify_method")
        if status not in ("implemented", "planned"):
            errors.append(f"{rid}: status '{status}' not in (implemented, planned)")
        # 4. implemented reqs must be linked/verified by an existing artifact
        if status == "implemented":
            if vm.startswith("tests/"):
                if rid not in refs:
                    errors.append(f"{rid}: orphan — no @pytest.mark.req(\"{rid}\") in tests/")
            else:
                # verified by infra (CI workflow / tool): the file must exist
                fpath = ROOT / vm.split("::")[0]
                if not fpath.exists():
                    errors.append(f"{rid}: verify_method file '{vm}' does not exist")
        # 5. release-gated coverage (pre-release only)
        if release and status == "implemented" and r["release_gated"].strip() == "yes":
            if rid not in realdata_refs:
                errors.append(f"{rid}: release-gated but has no realdata-marked test linked")

    # 6. dangling marker -> unknown requirement id
    for rid in refs:
        if rid not in known_ids:
            errors.append(f"test references unknown requirement '{rid}' "
                          f"(at {refs[rid][0]})")

    # 7. ADR index integrity (only if the decisions dir exists)
    if ADR_DIR.exists():
        adr_files = {p.name for p in ADR_DIR.glob("ADR-*.md")}
        nums = sorted(int(ADR_FILE_RE.match(n).group(1)) for n in adr_files
                      if ADR_FILE_RE.match(n))
        if not ADR_INDEX.exists():
            errors.append("architecture/decisions/INDEX.md missing while ADRs exist")
        else:
            index_text = ADR_INDEX.read_text(encoding="utf-8")
            for n in adr_files:
                if n.rsplit(".md", 1)[0] not in index_text:
                    errors.append(f"ADR {n} not listed in INDEX.md")
        for i, n in enumerate(nums):
            if i + 1 != n:
                errors.append(f"ADR numbering not contiguous from 0001 (gap/dupe near {n:04d})")
                break

    return errors, reqs, refs


# --------------------------------------------------------------------------- matrix
def render_matrix(reqs: list[dict], refs: dict[str, list[str]]) -> str:
    lines = [
        "# Traceability matrix",
        "",
        "Generated by `tools/check_traceability.py --write`. Do not edit by hand;",
        "CI fails if this file is out of sync with the registry and the test suite.",
        "",
        "| Requirement | Type | Area | ARC | Code module | Verification | Status | Release-gated | Linked tests |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(reqs, key=lambda x: x["id"]):
        linked = refs.get(r["id"], [])
        cell = "<br>".join(linked) if linked else "—"
        lines.append(
            f"| {r['id']} | {r['type']} | {r['area']} | {r['arc_ref']} | "
            f"`{r['code_module']}` | `{r['verify_method']}` | {r['status']} | "
            f"{r['release_gated']} | {cell} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="DS-MSP traceability governance gate")
    ap.add_argument("--write", action="store_true", help="regenerate the matrix and exit")
    ap.add_argument("--release", action="store_true",
                    help="also enforce real-data coverage for release-gated requirements")
    ap.add_argument("--check", action="store_true", help="validate (default)")
    args = ap.parse_args()

    errors, reqs, refs = validate(release=args.release)

    if args.write:
        MATRIX.parent.mkdir(parents=True, exist_ok=True)
        MATRIX.write_text(render_matrix(reqs, refs), encoding="utf-8")
        print(f"wrote {MATRIX.relative_to(ROOT)}")
        return 0

    # matrix in-sync check
    if not MATRIX.exists():
        errors.append("traceability matrix missing — run: python tools/check_traceability.py --write")
    elif MATRIX.read_text(encoding="utf-8") != render_matrix(reqs, refs):
        errors.append("traceability matrix out of sync — run: "
                      "python tools/check_traceability.py --write")

    if errors:
        print("TRACEABILITY: FAIL")
        for e in errors:
            print(f"  - {e}")
        return 1
    n_impl = sum(1 for r in reqs if r["status"] == "implemented")
    print(f"TRACEABILITY: OK ({len(reqs)} requirements, {n_impl} implemented, "
          f"{len(refs)} linked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
