"""Shared helpers for the paper-table generators (one script per table family under <setting>/tables/).

Every generator follows the same contract:

* a module docstring that states which Tables/*.tex files (and LaTeX labels) it writes, on which machine the
  experiments were run, which launcher produced the runs, and which result files it reads;
* it reads ONLY from the results root (``--results-root`` / ``$DRPT_RESULTS``; default ``$SCRATCH_DIR/Dr.Post-Training``)
  and writes ONLY to ``<paper root>/<venue>/Tables/`` (``--paper-root`` / ``$DRPT_PAPER``; default ``<repo>/Paper``);
* the first line of every written file is a provenance comment (script, date, results root);
* ``--check`` re-generates in memory and diffs against the files on disk instead of writing (comment lines ignored),
  so a table can be verified without touching the paper.

Statistics conventions (paper-wide): mean over seeds, SE = population std / sqrt(n) (ddof = 0),
cells ``\\(m\\,{\\scriptstyle\\pm se}\\)`` at one decimal, best per column in bold with ties at the displayed
precision all bold. Run any generator from the repo root, e.g. ``python SFT/tables/qwen_capability.py --check``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import difflib
import math
import os
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_VENUES = ["ICLR"]


# ----------------------------------------------------------------------------------------------------------------- paths
def results_root(cli: str | None = None) -> Path:
    """Where the result files live: --results-root, else $DRPT_RESULTS, else $SCRATCH_DIR/Dr.Post-Training."""
    for cand in (cli, os.environ.get("DRPT_RESULTS")):
        if cand:
            return Path(cand).expanduser()
    scratch = os.environ.get("SCRATCH_DIR") or _scratch_from_cluster_env()
    if scratch:
        return Path(scratch) / "Dr.Post-Training"
    sys.exit("results root unknown: pass --results-root, or export DRPT_RESULTS (or SCRATCH_DIR from cluster_env.sh)")


def _scratch_from_cluster_env() -> str | None:
    """SCRATCH_DIR from the (gitignored) cluster_env.sh at the repo root, so the generators work without sourcing it."""
    env = REPO / "cluster_env.sh"
    if not env.exists():
        return None
    m = re.search(r'^\s*export\s+SCRATCH_DIR="?([^"\n]+)"?', env.read_text(), re.M)
    return os.path.expandvars(m.group(1)) if m else None


def paper_root(cli: str | None = None) -> Path:
    """Where the venue folders (ICLR/, COLM/, ...) live: --paper-root, else $DRPT_PAPER, else <repo>/Paper."""
    for cand in (cli, os.environ.get("DRPT_PAPER")):
        if cand:
            return Path(cand).expanduser()
    return REPO / "Paper"


def add_common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--results-root", default=None, help="results root (default $DRPT_RESULTS or $SCRATCH_DIR/Dr.Post-Training)")
    ap.add_argument("--paper-root", default=None, help="paper root holding <venue>/Tables (default $DRPT_PAPER or <repo>/Paper)")
    ap.add_argument("--venues", default=",".join(DEFAULT_VENUES), help="comma-separated venue folders to write (default ICLR)")
    ap.add_argument("--check", action="store_true", help="do not write: diff the regenerated tables against the files on disk")
    ap.add_argument("--stdout", action="store_true", help="print the generated tables instead of writing them")
    return ap


# ------------------------------------------------------------------------------------------------------------ statistics
def mean_se(values):
    """(mean, SE, n) over the values of a {seed: value} dict or a list; SE = population std / sqrt(n); None if empty."""
    x = list(values.values()) if isinstance(values, dict) else list(values)
    x = [float(v) for v in x if v is not None]
    if not x:
        return None
    return st.mean(x), (st.pstdev(x) / math.sqrt(len(x)) if len(x) > 1 else 0.0), len(x)


def paired_gain(values, reference):
    """Mean of values[seed] - reference[seed] over the shared seeds (0.0 if none)."""
    shared = sorted(set(values) & set(reference))
    return st.mean([values[s] - reference[s] for s in shared]) if shared else 0.0


def best_mask(cols, prec: int = 1):
    """cols: list of value dicts/lists; True where the mean is maximal at the displayed precision (ties -> all True)."""
    means = [round(mean_se(c)[0], prec) if mean_se(c) else None for c in cols]
    mx = max((m for m in means if m is not None), default=None)
    return [m is not None and mx is not None and abs(m - mx) < 1e-9 for m in means]


# ---------------------------------------------------------------------------------------------------------------- LaTeX
def fmt_cell(values, bold: bool = False, se: bool = True, prec: int = 1, missing: str = "--") -> str:
    """``\\(m\\,{\\scriptstyle\\pm se}\\)`` (or ``\\(m\\)`` for one seed / se=False); ``--`` when there is no value."""
    ms = mean_se(values)
    if ms is None:
        return missing
    m, s, n = ms
    body = f"{m:.{prec}f}" if (n == 1 or not se) else f"{m:.{prec}f}\\,{{\\scriptstyle\\pm {s:.{prec}f}}}"
    if bold:
        body = f"\\mathbf{{{body}}}"
    return f"\\({body}\\)"


def mrow(n: int, txt: str, rule: bool = False) -> str:
    """multirow label centred over n rows; rule=True compensates for one booktabs cmidrule inside the span."""
    fix = r"[-0.5\dimexpr\aboverulesep+\belowrulesep+\cmidrulewidth\relax]" if rule else ""
    return f"\\multirow{{{n}}}{{*}}{fix}{{{txt}}}"


def provenance(script: str | Path, root: Path, note: str = "") -> str:
    rel = Path(script).resolve().relative_to(REPO) if str(script).startswith(str(REPO)) else Path(script).name
    date = _dt.date.today().isoformat()
    return f"% generated by {rel} on {date} from {root}{(' -- ' + note) if note else ''}; do not edit by hand\n"


# --------------------------------------------------------------------------------------------------------------- output
def _strip_comments(text: str) -> list[str]:
    return [l.rstrip() for l in text.splitlines() if not l.lstrip().startswith("%")]


def emit(files: dict[str, str], args, script: str | Path, root: Path, note: str = "") -> int:
    """Write / print / check the generated table bodies. Returns the number of files that differ (check mode)."""
    header = provenance(script, root, note)
    if args.stdout:
        for name, body in files.items():
            print(f"==> {name}\n{header}{body}")
        return 0
    proot = paper_root(args.paper_root)
    ndiff = 0
    for venue in [v for v in args.venues.split(",") if v]:
        tdir = proot / venue / "Tables"
        for name, body in files.items():
            path = tdir / name
            if args.check:
                old = path.read_text() if path.exists() else ""
                a, b = _strip_comments(old), _strip_comments(body)
                if a == b:
                    print(f"[ok]   {path}")
                else:
                    ndiff += 1
                    print(f"[DIFF] {path}" + ("" if path.exists() else " (missing on disk)"))
                    sys.stdout.writelines(difflib.unified_diff(a, b, "on disk", "regenerated", lineterm="", n=1))
                    print()
            else:
                tdir.mkdir(parents=True, exist_ok=True)
                path.write_text(header + body)
                print(f"wrote {path}")
    return ndiff
