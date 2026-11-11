"""Independent check of the score-sheet comparison (results/sheet_comparison.json and the Excel workbook).

Recomputes, without importing src/sheet_comparison.py:
  1. the received sheet's test sets and confusion counts by exact fractions (every reported ratio is k/m with
     small m, which pins down the test size), and that every reported ROC AUC is consistent with them;
  2. the delivered files' sizes, their PHA labels (JPL SBDB) and our rule's predictions, from a fresh
     path-to-path distance for every object;
  3. that every summary number in the JSON is the mean of its per-fold rows in results/sheet_comparison_folds.csv,
     and that the rule's rows on the full CAD table equal the paper's same-fold comparison;
  4. that the workbook's Master Results equal the JSON, every other results tab holds a formula whose cached
     value is what that formula must give, and the as-received tab equals the received file.
Exit status 1 on any failure.
"""
from __future__ import annotations

import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from orbit_geometry import moid_earth  # noqa: E402  (our solver; section 4 of verify_all.sh checks it)

EXT, RES = ROOT / "data" / "external", ROOT / "results"
RECEIVED = EXT / "NASA_Asteroid_Final_Comparative_Results.xlsx"
WORKBOOK = RES / "NASA_Asteroid_Comparative_Results_with_Ours.xlsx"
OURS_JSON, OURS_XLSX = "Ours: path rule (training-free)", "Ours (Asteroid's View)"
XLSX_NAME = {"CNN": "CNN", "GPBoost": "GPBoost", "Tuned RF": "Tuned Random Forest", OURS_JSON: OURS_XLSX}
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
HEAD = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC_AUC"}
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def frac(x: float) -> Fraction:
    f = Fraction(x).limit_denominator(100_000)
    assert abs(float(f) - x) < 1e-12, x
    return f


def metrics_from_counts(tp, fp, fn, tn) -> dict:
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"accuracy": (tp + tn) / n, "precision": prec, "recall": rec, "f1": f1,
            "mcc": (tp * tn - fp * fn) / den if den else 0.0}


# ---------------------------------------------------------------- 1. the received sheet
def part1(J: dict) -> None:
    print("\n-- 1. received sheet: test sets and confusion counts by exact fractions")
    sheet = pd.read_excel(RECEIVED, sheet_name="Master Results")
    rows_in_file = {"NASA CAD": len(pd.read_csv(EXT / "cad.csv")), "NASA SCOUT": len(pd.read_csv(EXT / "scout.csv")),
                    "NASA SENTRY": len(pd.read_csv(EXT / "sentry_risk_list.csv"))}
    js = {(r["dataset"], r["model"]): r for r in J["sheet"]["rows"]}
    for ds, g in sheet.groupby("Dataset", sort=False):
        n = math.lcm(*(frac(a).denominator for a in g["Accuracy"]))           # smallest common test size
        step = math.lcm(*(frac(r).denominator for r in g["Recall"] if r > 0))
        found = []
        for P in range(step, n, step):
            counts = {}
            for r in g.itertuples():
                tp = frac(r.Recall) * P
                tn_plus_tp = frac(r.Accuracy) * n
                if tp.denominator != 1 or tn_plus_tp.denominator != 1:
                    break
                tp, tn = int(tp), int(tn_plus_tp) - int(tp)
                fp = n - P - tn
                if r.Precision > 0 and frac(r.Precision) != Fraction(tp, tp + fp if tp + fp else 1):
                    break
                if r.Precision == 0 and tp > 0:
                    break
                if min(tp, tn, fp, P - tp) < 0:
                    break
                counts[r.Method] = (tp, fp, P - tp, tn)
            else:
                found.append((P, counts))
        check(f"{ds}: exactly one positives count fits all three models", len(found) == 1, f"n={n}, P={[f[0] for f in found]}")
        P, counts = found[0]
        for r in g.itertuples():
            tp, fp, fn, tn = counts[r.Method]
            j = js[(ds, r.Method)]
            check(f"{ds} {r.Method}: counts TP={tp} FP={fp} FN={fn} TN={tn} match the JSON",
                  (j["n"], j["positives"], j["TP"], j["FP"], j["FN"], j["TN"]) == (n, P, tp, fp, fn, tn))
            u2 = 2 * r.ROC_AUC * P * (n - P)
            check(f"{ds} {r.Method}: ROC AUC x P x N is a multiple of 1/2", abs(u2 - round(u2)) < 1e-6, f"2U={u2:.6f}")
        t = J["sheet"]["test_sets"][ds]
        check(f"{ds}: test set {n} rows vs {rows_in_file[ds]} rows in the delivered file (JSON agrees)",
              (t["n_test"], t["positives"], t["rows_in_delivered_file"]) == (n, P, rows_in_file[ds]))
    cad_n = J["sheet"]["test_sets"]["NASA CAD"]["n_test"]
    check("NASA CAD: the received test set is larger than the delivered cad.csv, so it cannot come from it",
          cad_n > rows_in_file["NASA CAD"], f"{cad_n} > {rows_in_file['NASA CAD']}")
    sc = {r["model"]: r for r in J["sheet"]["rows"] if r["dataset"] == "NASA SCOUT"}
    check("NASA SCOUT: the CNN predicted no positive (its 6/9 equals always answering no)",
          sc["CNN"]["predicted_positive"] == 0 and abs(sc["CNN"]["accuracy"] - sc["CNN"]["majority_class_accuracy"]) < 1e-12)
    check("NASA SCOUT: GPBoost and the tuned RF score below always answering no",
          all(sc[m]["accuracy"] < sc[m]["majority_class_accuracy"] for m in ("GPBoost", "Tuned Random Forest")))


# ---------------------------------------------------------------- 2. the delivered files
def part2(J: dict) -> None:
    print("\n-- 2. delivered files: PHA labels and our rule from a fresh path-to-path distance")
    sb = pd.read_csv(ROOT / "data" / "processed" / "sbdb_neo.csv", low_memory=False, dtype={"pdes": str})
    sb["pdes"] = sb["pdes"].str.strip()
    sb = sb.set_index("pdes")
    F = J["files"]
    for key, fname in (("cad_file", "cad.csv"), ("sentry_file", "sentry_risk_list.csv")):
        d = pd.read_csv(EXT / fname, dtype={"des": str})
        d["des"] = d["des"].str.strip()
        s_all = sb.loc[d["des"]]                                   # raises if any designation is missing
        has_h = s_all["H"].notna().to_numpy()                      # the pipeline scores asteroids that have an H
        no_h = s_all[~has_h]
        check(f"{fname}: {len(d)} rows, all in JPL SBDB; {int(has_h.sum())} have an H and are scored "
              f"(JSON agrees); those without H are not PHAs", F[key]["rows_in_file"] == len(d)
              and F[key]["rows_matched_to_sbdb"] == int(has_h.sum()) and bool((no_h["pha"] == "N").all()),
              f"no H: {list(no_h.index)}")
        s = s_all[has_h]
        y = (s["pha"] == "Y").to_numpy(int)
        delta = moid_earth(*(s[c].to_numpy(float) for c in ("a", "e", "i", "om", "w")))
        pred = ((delta <= 0.05) & (s["H"].to_numpy(float) <= 22.0)).astype(int)
        tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
        fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
        m = metrics_from_counts(tp, fp, fn, tn)
        ours = F[key]["methods"][OURS_JSON]
        sfx = "_mean" if key == "sentry_file" else ""
        check(f"{fname}: {int(y.sum())} PHAs among the scored objects (JSON agrees)", F[key]["positives"] == int(y.sum()))
        check(f"{fname}: our rule TP={tp} FP={fp} FN={fn} TN={tn}; JSON metrics agree",
              all(abs(ours[f"{k}{sfx}"] - m[k]) < 1e-12 for k in ("accuracy", "precision", "recall", "f1", "mcc")),
              ", ".join(f"{k}={m[k]:.4f}" for k in ("accuracy", "f1")))
        if key == "sentry_file":
            hy = (s["H"].to_numpy(float) <= 22.0).astype(int)
            check(f"{fname}: 'H <= 22' alone would make {int((hy != y).sum())} errors; the path geometry removes them",
                  int((hy != y).sum()) > 0 and fp + fn == 0)
    sc = pd.read_csv(EXT / "scout.csv")
    check("scout.csv: no Scout candidate is in JPL SBDB (no ground truth)",
          int(sc["objectName"].astype(str).isin(sb.index).sum()) == 0 and F["scout_file"]["ground_truth"] is None)
    flag = int(((sc["moid"] <= 0.05) & (sc["H"] <= 22.0)).sum())
    check(f"scout.csv: our rule on Scout's own MOID and H flags {flag} candidate(s) (JSON agrees)",
          flag == F["scout_file"]["rule_on_scout_nominal_moid_and_H"]["flagged"])


# ---------------------------------------------------------------- 3. summaries equal their per-fold rows
def part3(J: dict) -> None:
    print("\n-- 3. every summary equals the mean of its per-fold rows")
    folds = pd.read_csv(RES / "sheet_comparison_folds.csv")
    se = folds[folds["dataset"] == "Sentry file"]
    check("Sentry: 5 repeats x 4 methods of per-fold rows", len(se) == 20 and set(se["fold"]) == set(range(5)))
    worst = max(abs(se[se["method"] == m][k].mean() - v[f"{k}_mean"])
                for m, v in J["files"]["sentry_file"]["methods"].items() for k in METRICS)
    check("Sentry: every summary mean equals the mean of its repeats", worst < 1e-12, f"max diff {worst:.1e}")
    if "cad_full" not in J:
        check("full NASA CAD results present", False, "results/sheet_comparison.json has no cad_full yet")
        return
    cf = folds[folds["dataset"] == "NASA CAD (full)"]
    check("full CAD: 2 protocols x 5 folds x 4 methods of per-fold rows", len(cf) == 40)
    worst = max(abs(cf[(cf["protocol"] == p) & (cf["method"] == m)][k].mean() - v[f"{k}_mean"])
                for p, ms in J["cad_full"]["protocols"].items() for m, v in ms.items() for k in METRICS)
    check("full CAD: every summary mean equals the mean of its folds", worst < 1e-12, f"max diff {worst:.1e}")
    same = pd.read_csv(RES / "same_folds_rows.csv")
    ref = same[same["method"] == "rule: exact path geometry (ours)"].set_index("fold")
    mine = cf[(cf["protocol"] == "asteroid_grouped") & (cf["method"] == OURS_JSON)].set_index("fold")
    worst = max(abs(mine.loc[k, c] - ref.loc[k, c]) for k in range(5) for c in METRICS + ["mcc"])
    check("full CAD: our rule's grouped-fold rows equal the paper's same-fold comparison", worst < 1e-12,
          f"max diff {worst:.1e}")


# ---------------------------------------------------------------- 4. the workbook
def part4(J: dict) -> None:
    print("\n-- 4. workbook: values, formulas and the as-received tab")
    val, form = load_workbook(WORKBOOK, data_only=True), load_workbook(WORKBOOK)
    F = J["files"]
    blocks = [("NASA CAD", F["cad_file"]["methods"], ""), ("NASA SENTRY", F["sentry_file"]["methods"], "_mean"),
              ("NASA SCOUT", None, "")]
    if "cad_full" in J:
        blocks.append(("NASA CAD FULL", J["cad_full"]["protocols"]["asteroid_grouped"], "_mean"))
    expect = {}
    for ds, methods, sfx in blocks:
        for mj, mx in XLSX_NAME.items():
            expect[(ds, mx)] = [methods[mj][f"{k}{sfx}"] if methods is not None else "n/a" for k in METRICS]
    mr = list(val["Master Results"].iter_rows(min_row=2, values_only=True))
    got = {(r[0], r[1]): list(r[2:7]) for r in mr}
    check("Master Results holds exactly the JSON numbers", got == expect and len(mr) == len(expect))
    datasets = [b[0] for b in blocks]
    methods = list(XLSX_NAME.values())
    ok_val = ok_formula = True
    for k in METRICS:
        ws, wf = val[HEAD[k]], form[HEAD[k]]
        for r, ds in enumerate(datasets, start=2):
            for c, mx in enumerate(methods, start=2):
                ok_val &= ws.cell(r, c).value == expect[(ds, mx)][METRICS.index(k)]
                ok_formula &= str(wf.cell(r, c).value).startswith("=")
    check("per-metric tabs: every cell is a formula whose value equals Master Results", ok_val and ok_formula)
    ok = True
    for r, ds in enumerate(datasets, start=2):
        f1 = [expect[(ds, m)][3] for m in methods]
        nums = [v for v in f1 if not isinstance(v, str)]
        best = methods[f1.index(max(nums))] if nums else "n/a"
        row = [val["Best Models"].cell(r, c).value for c in range(1, 8)]
        ok &= row[:2] == [ds, best] and row[2:] == (expect[(ds, best)] if best != "n/a" else ["n/a"] * 5)
        ok &= all(str(form["Best Models"].cell(r, c).value).startswith("=") for c in range(2, 8))
        scored = [expect[(ds, m)] for m in methods if not isinstance(expect[(ds, m)][0], str)]
        srow = [val["Dataset Summary"].cell(r, c).value for c in range(1, 8)]
        avg = [sum(v[i] for v in scored) / len(scored) for i in range(5)] if scored else ["n/a"] * 5
        ok &= srow[0] == ds and srow[1] == len(scored) and all(
            (a == b) if isinstance(b, str) else abs(a - b) < 1e-12 for a, b in zip(srow[2:], avg))
    check("Best Models (highest F1) and Dataset Summary equal what their formulas must give", ok)
    orig = pd.read_excel(RECEIVED, sheet_name="Master Results")
    mine = pd.read_excel(WORKBOOK, sheet_name="Original Sheet (as received)").iloc[:len(orig)]
    cols = ["Dataset", "Method", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    check("as-received tab equals the received sheet's Master Results",
          bool((orig[cols].reset_index(drop=True) == mine[cols].reset_index(drop=True)).all().all()))


def main() -> None:
    J = json.loads((RES / "sheet_comparison.json").read_text())
    part1(J)
    part2(J)
    part3(J)
    part4(J)
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
