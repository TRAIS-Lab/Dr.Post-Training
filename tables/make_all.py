#!/usr/bin/env python
"""Run (or --check) every table generator registered in tables/registry.py and audit the paper's Tables/ folder.

Usage: python tables/make_all.py [--check] [--venues ICLR] [--paper-root DIR] [--results-root DIR] [--only SUBSTR]
  default : regenerate every table whose generator exists on this checkout (writes into <paper root>/<venue>/Tables/)
  --check : regenerate in memory and diff against the files on disk instead of writing
Always reports: generators registered but missing on this checkout (results live on another machine), .tex files in
Tables/ that no registered family produces, and registered files the paper never \\input{}s. Exit code 1 if a --check
diff fails or an unregistered / orphaned table exists.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tables.common import paper_root  # noqa: E402
from tables.registry import FAMILIES, by_table  # noqa: E402


def inputs_in_paper(venue_dir: Path) -> set[str]:
    pat = re.compile(r"\\input\{Tables/([^}]+)\}")
    names = set()
    for tex in list(venue_dir.glob("*.tex")) + list((venue_dir / "appendix").glob("*.tex")):
        for line in tex.read_text(errors="replace").splitlines():
            if line.lstrip().startswith("%"):
                continue
            for m in pat.finditer(line):
                n = m.group(1)
                names.add(n if n.endswith(".tex") else n + ".tex")
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--venues", default="ICLR")
    ap.add_argument("--paper-root", default=None)
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--only", default=None, help="run only generators whose path contains this substring")
    a = ap.parse_args()
    proot = paper_root(a.paper_root)
    failed = 0
    print("== generators")
    for fam in FAMILIES:
        script = fam.get("script")
        if not script:
            continue
        if a.only and a.only not in script:
            continue
        if fam.get("kind") == "shelved":
            print(f"[shelved] {script}  (tables not in the paper; run it directly to regenerate them)")
            continue
        path = REPO / script
        if not path.exists():
            print(f"[MISSING] {script}  (runs on: {fam['machine']}; {fam.get('note', '')})")
            continue
        cmd = [sys.executable, str(path), "--venues", a.venues] + (["--check"] if a.check else [])
        if a.paper_root:
            cmd += ["--paper-root", a.paper_root]
        if a.results_root:
            cmd += ["--results-root", a.results_root]
        print(f"[run] {' '.join(cmd[1:])}")
        r = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
        sys.stdout.write("".join("    " + l + "\n" for l in (r.stdout + r.stderr).splitlines() if l.strip()))
        if r.returncode != 0:
            failed += 1
            print(f"    -> exit {r.returncode}")
    print("== audit of Tables/")
    reg = by_table()
    for venue in a.venues.split(","):
        vdir = proot / venue
        if not vdir.exists():
            print(f"[skip] {vdir} does not exist")
            continue
        on_disk = {p.name for p in (vdir / "Tables").glob("*.tex")}
        used = inputs_in_paper(vdir)
        for tex in sorted(on_disk - set(reg)):
            print(f"[UNREGISTERED] {venue}/Tables/{tex}  (no family in tables/registry.py produces it)"); failed += 1
        for tex in sorted(set(reg) - used):
            if reg[tex].get("kind") == "shelved":
                continue
            where = "on disk" if tex in on_disk else "not on disk"
            print(f"[ORPHAN] {venue}/Tables/{tex} is registered but never \\input{{}} by the paper ({where})"); failed += 1
        for tex in sorted(used - on_disk):
            print(f"[BROKEN INPUT] the paper \\input{{}}s {tex} but the file is missing"); failed += 1
        for tex in sorted(used - set(reg)):
            print(f"[UNREGISTERED] {venue} \\input{{}}s {tex} which is not in tables/registry.py"); failed += 1
        n_conc = sum(1 for t in reg.values() if t.get("kind") == "conceptual")
        n_shelved = sum(1 for t in reg.values() if t.get("kind") == "shelved")
        print(f"[summary] {venue}: {len(on_disk)} files in Tables/, {len(used)} \\input by the paper, {len(reg)} registered "
              f"({len(reg) - n_conc - n_shelved} generated, {n_conc} conceptual, {n_shelved} shelved)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
