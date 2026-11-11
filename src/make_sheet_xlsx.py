"""Excel version of the received comparison sheet, with our method added.

Writes results/NASA_Asteroid_Comparative_Results_with_Ours.xlsx with the received sheet's tabs and columns
(Master Results, Accuracy, Precision, Recall, F1, ROC_AUC, Best Models, Dataset Summary), plus
'Original Sheet (as received)' and 'Notes'. Master Results holds the numbers from results/sheet_comparison.json;
every other results tab is Excel formulas over Master Results. openpyxl cannot store formula results, so the
computed values are written into the file as the formulas' cached values (Excel recalculates on open anyway;
file previewers show the cached values).
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from common import RESULTS

OUT = RESULTS / "NASA_Asteroid_Comparative_Results_with_Ours.xlsx"
OURS = "Ours (Asteroid's View)"
METHODS = [("CNN", "CNN"), ("GPBoost", "GPBoost"), ("Tuned RF", "Tuned Random Forest"),
           ("Ours: path rule (training-free)", OURS)]
METRICS = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC_AUC")]
MCOL = {name: get_column_letter(3 + i) for i, (_, name) in enumerate(METRICS)}      # Master Results columns C..G
FONT, BOLD = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
HEAD_FILL, OURS_FILL = PatternFill("solid", fgColor="D9D9D9"), PatternFill("solid", fgColor="E2F0D9")
NUM = "0.0000"


def blocks(d: dict) -> list[tuple[str, dict | None, str, str]]:
    """(dataset name, per-method metrics or None, key suffix, how it was evaluated)."""
    f = d["files"]
    cad, se, sc = f["cad_file"], f["sentry_file"], f["scout_file"]
    out = [
        ("NASA CAD", cad["methods"], "",
         f"cad.csv as delivered: {cad['rows_in_file']} close approaches (all within 0.05 au), {cad['positives']} PHAs; "
         "trained models scored by leave-one-out (pooled out-of-fold predictions); the random forest could not be "
         "tuned (at most 2 positives per training fold), so it uses library defaults with 400 trees."),
        ("NASA SENTRY", se["methods"], "_mean",
         f"sentry_risk_list.csv as delivered: {se['rows_in_file']:,} objects, of which {se['rows_matched_to_sbdb']:,} "
         f"are scored ({se['rows_in_file'] - se['rows_matched_to_sbdb']} has no absolute magnitude H in JPL's catalogue "
         f"and no PHA flag basis), {se['positives']} PHAs; 5 x 5 repeated stratified cross-validation, out-of-fold "
         "predictions pooled per repeat, mean of the 5 repeats."),
        ("NASA SCOUT", None, "",
         f"scout.csv as delivered: {sc['rows_in_file']} unconfirmed NEOCP candidates. None has a designation, JPL "
         "orbit or PHA flag yet, so there is no ground truth and no method can be scored (n/a)."),
    ]
    if "cad_full" in d:
        cf = d["cad_full"]
        out.append(("NASA CAD FULL", cf["protocols"]["asteroid_grouped"], "_mean",
                    f"Full NASA CAD table: {cf['rows']:,} Earth close approaches within 0.5 au (1910-2024) of "
                    f"{cf['asteroids']:,} asteroids, {cf['positive_rows']:,} PHA rows; 5-fold cross-validation with "
                    "every approach of an asteroid in the same fold (the paper's folds)."))
    return out


def style(ws, widths: dict, header_row: int = 1) -> None:
    for row in ws.iter_rows():
        for c in row:
            c.font = FONT
    for c in ws[header_row]:
        c.font, c.fill = BOLD, HEAD_FILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    ws.freeze_panes = ws.cell(row=header_row + 1, column=1)


def build(d: dict) -> tuple[Workbook, dict]:
    """Workbook plus the expected value of every formula cell: {sheet: {cell: value}}."""
    bl = blocks(d)
    datasets = [b[0] for b in bl]
    master = []                                          # (dataset, method, [metric values or 'n/a'])
    for ds, methods, suf, _ in bl:
        for key, name in METHODS:
            vals = [methods[key][f"{m}{suf}"] if methods is not None else "n/a" for m, _ in METRICS]
            master.append((ds, name, vals))
    last = 1 + len(master)
    rng = lambda col: f"'Master Results'!${col}$2:${col}${last}"             # noqa: E731
    lookup = {(ds, name): vals for ds, name, vals in master}
    wb, expected = Workbook(), {}

    ws = wb.active
    ws.title = "Master Results"
    ws.append(["Dataset", "Method"] + [n for _, n in METRICS])
    for ds, name, vals in master:
        ws.append([ds, name] + vals)
        if name == OURS:
            for c in ws[ws.max_row]:
                c.fill = OURS_FILL
    style(ws, {"A": 16, "B": 24, **{MCOL[n]: 12 for _, n in METRICS}})
    for row in ws.iter_rows(min_row=2, min_col=3):
        for c in row:
            c.number_format = NUM
            if c.row and ws.cell(c.row, 2).value == OURS:
                c.font = BOLD

    for _, mname in METRICS:                             # one tab per metric: Dataset x Method
        ws = wb.create_sheet(mname)
        ws.append(["Dataset"] + [n for _, n in METHODS])
        exp = expected.setdefault(mname, {})
        for r, ds in enumerate(datasets, start=2):
            ws.cell(r, 1, ds)
            for j, (_, name) in enumerate(METHODS, start=2):
                cell = ws.cell(r, j)
                cell.value = (f'=IFERROR(AVERAGEIFS({rng(MCOL[mname])},{rng("A")},$A{r},{rng("B")},'
                              f'{get_column_letter(j)}$1),"n/a")')
                cell.number_format = NUM
                exp[cell.coordinate] = lookup[(ds, name)][[n for _, n in METRICS].index(mname)]
        style(ws, {"A": 16, "B": 12, "C": 12, "D": 20, "E": 22})
        for r in range(2, len(datasets) + 2):
            ws.cell(r, len(METHODS) + 1).font = BOLD

    ws = wb.create_sheet("Best Models")                  # best method per dataset by F1, as in the received sheet
    ws.append(["Dataset", "Method"] + [n for _, n in METRICS])
    exp = expected.setdefault("Best Models", {})
    for r, ds in enumerate(datasets, start=2):
        ws.cell(r, 1, ds)
        ws.cell(r, 2).value = f'=IFERROR(INDEX(F1!$B$1:$E$1,MATCH(MAX(F1!$B{r}:$E{r}),F1!$B{r}:$E{r},0)),"n/a")'
        f1 = [lookup[(ds, n)][3] for _, n in METHODS]
        nums = [v for v in f1 if not isinstance(v, str)]
        best = METHODS[f1.index(max(nums))][1] if nums else "n/a"
        exp[f"B{r}"] = best
        for j, (_, mname) in enumerate(METRICS, start=3):
            cell = ws.cell(r, j)
            cell.value = (f'=IFERROR(AVERAGEIFS({rng(MCOL[mname])},{rng("A")},$A{r},{rng("B")},$B{r}),"n/a")')
            cell.number_format = NUM
            exp[cell.coordinate] = lookup[(ds, best)][j - 3] if best != "n/a" else "n/a"
    style(ws, {"A": 16, "B": 24, **{get_column_letter(c): 12 for c in range(3, 8)}})

    ws = wb.create_sheet("Dataset Summary")
    ws.append(["Dataset", "Number_of_Models"] + [f"Average_{n}" for _, n in METRICS])
    exp = expected.setdefault("Dataset Summary", {})
    for r, ds in enumerate(datasets, start=2):
        ws.cell(r, 1, ds)
        ws.cell(r, 2).value = f'=COUNTIFS({rng("A")},$A{r},{rng("C")},">=0")'
        rows = [lookup[(ds, n)] for _, n in METHODS]
        scored = [v for v in rows if not isinstance(v[0], str)]
        exp[f"B{r}"] = len(scored)
        for j, (_, mname) in enumerate(METRICS, start=3):
            cell = ws.cell(r, j)
            cell.value = f'=IFERROR(AVERAGEIFS({rng(MCOL[mname])},{rng("A")},$A{r}),"n/a")'
            cell.number_format = NUM
            exp[cell.coordinate] = sum(v[j - 3] for v in scored) / len(scored) if scored else "n/a"
    style(ws, {"A": 16, "B": 18, **{get_column_letter(c): 18 for c in range(3, 8)}})

    ws = wb.create_sheet("Original Sheet (as received)")
    sheet = d["sheet"]
    ws.append(["Dataset", "Method"] + [n for _, n in METRICS] + ["Test rows (decoded)", "Positive test rows (decoded)",
                                                                  "Rows in the delivered file", "Label"])
    for row in sheet["rows"]:
        t = sheet["test_sets"][row["dataset"]]
        ws.append([row["dataset"], row["model"], row["accuracy"], row["precision"], row["recall"], row["f1"],
                   row["roc_auc"], row["n"], row["positives"], t["rows_in_delivered_file"], "not stated"])
    for row in ws.iter_rows(min_row=2, min_col=3, max_col=7):
        for c in row:
            c.number_format = NUM
    note = ws.max_row + 2
    ws.cell(note, 1, "These are the numbers exactly as received. Their label is not stated and their test sets do not "
                     "match the delivered files (test rows decoded from the reported metrics), so they measure a "
                     "different task from the other tabs and are kept apart.")
    style(ws, {"A": 16, "B": 22, **{get_column_letter(c): 12 for c in range(3, 8)}, "H": 14, "I": 16, "J": 16, "K": 12})
    ws.cell(note, 1).font = Font(name="Arial", size=10, italic=True)

    ws = wb.create_sheet("Notes")
    lines = [
        ("What this workbook is", "The received sheet's tables with our method added. All four methods were rerun by "
         "us on the three delivered files, with one stated label and the same folds for every method."),
        ("Label (ground truth)", "JPL's PHA flag: Earth MOID <= 0.05 au and H <= 22 (JPL SBDB, retrieved 2026-11-11)."),
        (OURS, "Training-free rule: delta = exact distance between the asteroid's orbit and Earth's orbit, computed "
         "from the asteroid's orbital elements (JPL SBDB); PHA if and only if delta <= 0.05 au and H <= 22. No training."),
        ("CNN, GPBoost, Tuned Random Forest", "Trained on each file's own numeric columns. CNN: two 1-D convolution "
         "layers and a dense head (PyTorch). GPBoost: tree boosting, with a random effect per asteroid on the full "
         "CAD table. Tuned Random Forest: randomized search (F1) inside the training folds only. Threshold 0.5."),
    ] + [(ds, how) for ds, _, _, how in blocks(d)] + ([] if "cad_full" in d else [
        ("NASA CAD FULL", "Running now: the three models on all 306,590 NASA CAD approaches; rows will be added.")]) + [
        ("Why the received numbers are separate", "Decoded from the reported metrics, the received test sets have "
         + "; ".join(f"{ds} {t['n_test']} rows ({t['positives']} positive)" for ds, t in d["sheet"]["test_sets"].items())
         + ". The delivered files have 33, 72 and 2,242 rows, and only 15 of the 2,242 Sentry objects are PHAs, so the "
         "received label (about half positive) is not the PHA flag."),
        ("Source", "results/sheet_comparison.json, produced by src/sheet_comparison.py; this workbook is built by "
         "src/make_sheet_xlsx.py. Master Results holds the numbers; the other result tabs are formulas over it."),
    ]
    ws.append(["Item", "Details"])
    for k, v in lines:
        ws.append([k, v])
    style(ws, {"A": 34, "B": 120})
    for row in ws.iter_rows(min_row=2):
        row[0].font = BOLD
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
        row[0].alignment = Alignment(vertical="top")
    wb.calculation.fullCalcOnLoad = True
    return wb, expected


def add_cached_values(path: Path, wb: Workbook, expected: dict) -> int:
    """Write each formula's computed value into the saved file as its cached <v> value."""
    sheet_files = {ws.title: f"xl/worksheets/sheet{i}.xml" for i, ws in enumerate(wb.worksheets, start=1)}
    tmp = Path(tempfile.mkdtemp()) / path.name
    n = 0
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            title = next((t for t, f in sheet_files.items() if f == item.filename), None)
            if title in expected:
                xml = data.decode("utf-8")

                def fill(m: re.Match) -> str:
                    nonlocal n
                    attrs, ref, formula = m.group(1), m.group(2), m.group(3)
                    val = expected[title][ref]
                    n += 1
                    attrs = re.sub(r'\s+t="[^"]*"', "", attrs)
                    if isinstance(val, str):
                        return f'<c{attrs} t="str"><f>{formula}</f><v>{escape(val)}</v></c>'
                    return f"<c{attrs}><f>{formula}</f><v>{repr(float(val))}</v></c>"

                xml = re.sub(r'<c((?=[^>]*\sr="([A-Z]+[0-9]+)")[^>]*)><f>(.*?)</f>(?:<v\s*/>|<v>[^<]*</v>)?</c>', fill, xml)
                data = xml.encode("utf-8")
            zout.writestr(item, data)
    shutil.move(tmp, path)
    return n


def main() -> None:
    d = json.loads((RESULTS / "sheet_comparison.json").read_text())
    wb, expected = build(d)
    wb.save(OUT)
    n = add_cached_values(OUT, wb, expected)
    total = sum(len(v) for v in expected.values())
    assert n == total, f"cached {n} of {total} formula cells"
    print(f"wrote {OUT} ({total} formula cells, all with cached values)")


if __name__ == "__main__":
    main()
