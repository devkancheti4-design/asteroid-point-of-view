"""Compare two complete runs of the pipeline, file by file.

Usage:  python proofs/compare_runs.py <run A> <run B>      (each a project folder after ./run_all.sh)

  results/*.json   every number compared (max absolute difference), every other value compared exactly
  results/*.csv    same shape and columns; numeric columns by max absolute difference, others exactly
  paper/tables/*   generated LaTeX tables compared as text (they carry every number shown in the paper's tables)
A file counts as identical when every number agrees to TOL; otherwise its largest difference is printed.
Wall-clock timings (fields named *seconds*, runtime_s, ms or ms_per_*) differ between any two runs and are skipped.
Exit status 1 if any file is missing or differs by more than TOL.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-9
TIMING = re.compile(r"seconds|runtime_s|^ms$|ms_per_")


def walk(a, b, path="", out=None):
    """Recursively compare two JSON values; collect (path, kind, detail)."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if TIMING.search(k):
                continue
            if k not in a or k not in b:
                out.append((f"{path}/{k}", "missing", ""))
            else:
                walk(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, "length", f"{len(a)} vs {len(b)}"))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (math.isnan(a) and math.isnan(b)) and abs(a - b) > TOL:
            out.append((path, "number", abs(a - b)))
    elif a != b:
        out.append((path, "value", f"{str(a)[:40]!r} vs {str(b)[:40]!r}"))
    return out


def compare_csv(fa: Path, fb: Path) -> tuple[bool, str]:
    A, B = pd.read_csv(fa, low_memory=False), pd.read_csv(fb, low_memory=False)
    if A.shape != B.shape or list(A.columns) != list(B.columns):
        return False, f"shape/columns differ: {A.shape} vs {B.shape}"
    worst, worst_col = 0.0, ""
    for c in A.columns:
        if TIMING.search(c):
            continue
        if A[c].dtype.kind in "fiu" and B[c].dtype.kind in "fiu":
            d = np.nanmax(np.abs(A[c].to_numpy(float) - B[c].to_numpy(float))) if len(A) else 0.0
            if np.isfinite(d) and d > worst:
                worst, worst_col = float(d), c
            if (A[c].isna() != B[c].isna()).any():
                return False, f"missing values differ in {c}"
        elif not A[c].astype(str).equals(B[c].astype(str)):
            return False, f"text column {c} differs"
    return worst <= TOL, ("identical" if worst <= TOL else f"max |diff| {worst:.3g} in {worst_col}")


def main(a: str, b: str) -> None:
    A, B = Path(a), Path(b)
    rows, bad = [], 0
    files = sorted({p.relative_to(A) for p in (A / "results").glob("*") if p.suffix in (".json", ".csv")} |
                   {p.relative_to(B) for p in (B / "results").glob("*") if p.suffix in (".json", ".csv")} |
                   {p.relative_to(A) for p in (A / "paper" / "tables").glob("*.tex")} |
                   {p.relative_to(B) for p in (B / "paper" / "tables").glob("*.tex")})
    for rel in files:
        fa, fb = A / rel, B / rel
        if not (fa.exists() and fb.exists()):
            rows.append((str(rel), "MISSING in " + ("A" if not fa.exists() else "B")))
            bad += 1
            continue
        if rel.suffix == ".json":
            diffs = walk(json.loads(fa.read_text()), json.loads(fb.read_text()))
            if diffs:
                nums = [d for d in diffs if d[1] == "number"]
                other = [d for d in diffs if d[1] != "number"]
                msg = f"{len(diffs)} values differ"
                if nums:
                    p, _, v = max(nums, key=lambda d: d[2])
                    msg += f"; max |diff| {v:.3g} at {p}"
                if other:
                    msg += f"; e.g. {other[0][0]} ({other[0][1]})"
                rows.append((str(rel), msg))
                bad += 1
            else:
                rows.append((str(rel), "identical"))
        elif rel.suffix == ".csv":
            ok, msg = compare_csv(fa, fb)
            rows.append((str(rel), msg))
            bad += 0 if ok else 1
        else:
            same = fa.read_text() == fb.read_text()
            rows.append((str(rel), "identical text" if same else "TEXT DIFFERS"))
            bad += 0 if same else 1
    w = max(len(r[0]) for r in rows)
    for name, msg in rows:
        print(f"{name:<{w}}  {msg}")
    print(f"\n{len(rows) - bad} of {len(rows)} files identical (tolerance {TOL:g})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
