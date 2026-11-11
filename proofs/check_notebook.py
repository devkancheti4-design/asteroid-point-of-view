"""The Kaggle notebook reproduces the paper's numbers.

Runs every code cell of notebooks/asteroid_point_of_view_kaggle.ipynb on the data in data/ (the paper's snapshot
after ./run_all.sh --snapshot), with the Earth's-view machine-learning baselines and the JPL download switched off,
and compares what the notebook computes with results/: our path-to-path distance for every asteroid, the geometry
validation, the 5-fold and temporal rule results, the rule on the paper's grouped CAD folds, the delivered CAD and
Sentry files, and the uncertainty simulation. Exit status 1 on any difference.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "asteroid_point_of_view_kaggle.ipynb"
R = ROOT / "results"
TOL = 1e-12
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def run_notebook() -> dict:
    cells = ["".join(c["source"]) for c in json.loads(NB.read_text())["cells"] if c["cell_type"] == "code"]
    code = "\n\n".join(cells)
    for flag in ("RUN_EARTH_VIEW_BASELINES", "DOWNLOAD_IF_MISSING"):
        assert f"{flag} = True" in code, flag
        code = code.replace(f"{flag} = True", f"{flag} = False")
    ns = {"__name__": "__main__", "display": lambda *a, **k: None}
    os.environ["NEO_INPUT"] = str(ROOT / "data")
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            exec(compile(code, str(NB), "exec"), ns)
        finally:
            os.chdir(cwd)
    return ns


def close(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def main() -> None:
    ns = run_notebook()
    print("\n-- comparison with results/")
    neo = ns["neo"]
    feat = pd.read_csv(R / "asteroid_features.csv", usecols=["pdes", "moid_path", "d_lin", "d_node"], dtype={"pdes": str})
    check("same 42,888 asteroids in the same order as the pipeline",
          len(neo) == len(feat) and bool((neo["pdes"].to_numpy() == feat["pdes"].str.strip().to_numpy()).all()))
    for col_nb, col in (("delta", "moid_path"), ("d_lin", "d_lin"), ("d_node", "d_node")):
        d = close(neo[col_nb], feat[col])
        check(f"notebook {col_nb} equals the pipeline's {col} for every asteroid", d <= TOL, f"max diff {d:.1e}")

    names = {"Ours: exact path geometry": "rule: exact path geometry (ours)",
             "Closed form: tangent line": "rule: tangent-line closed form",
             "Upper bound: node gap": "rule: node gap (upper bound)",
             "Reference: the definition on JPL's own MOID": "rule: JPL MOID (definition, reference)"}
    av = pd.read_csv(R / "asteroid_view_summary.csv")
    av = av[av["method"].isin(names.values())].set_index("method")
    kf = ns["kfold"].set_index("method")
    cols = ["accuracy_mean", "accuracy_std", "f1_mean", "f1_std", "mcc_mean", "recall_mean", "precision_mean"]
    d = max(close(kf.loc[a, cols], av.loc[b, cols]) for a, b in names.items())
    check("5-fold rule results (4 rules, mean and std) equal results/asteroid_view_summary.csv", d <= TOL, f"max diff {d:.1e}")
    tp = pd.read_csv(R / "asteroid_view_temporal.csv").set_index("method")
    tn = ns["temporal"].set_index("method")
    cols = ["accuracy", "f1", "mcc", "recall", "precision"]
    d = max(close(tn.loc[a, cols], tp.loc[b, cols]) for a, b in names.items())
    check("temporal rule results equal results/asteroid_view_temporal.csv", d <= TOL, f"max diff {d:.1e}")

    sf = pd.read_csv(R / "same_folds_rows_summary.csv")
    ref = sf[sf["method"] == "rule: exact path geometry (ours)"].iloc[0]
    ev = ns["earth_view"]
    mine = ev[(ev["protocol"] == "asteroid-grouped (honest)") & (ev["method"] == "Ours: path rule (no training)")].iloc[0]
    cols = ["accuracy_mean", "f1_mean", "mcc_mean", "recall_mean", "precision_mean", "roc_auc_mean"]
    d = close(mine[cols], ref[cols])
    check("rule on the paper's grouped CAD folds equals results/same_folds_rows_summary.csv", d <= TOL, f"max diff {d:.1e}")

    sc = json.loads((R / "sheet_comparison.json").read_text())["files"]
    ours = "Ours: path rule (training-free)"
    t = ns["TABLES"]
    d1 = close([t["CAD table"][k] for k in cols_b()], [sc["cad_file"]["methods"][ours][k] for k in cols_b()])
    d2 = close([t["Sentry risk list"][k] for k in cols_b()],
               [sc["sentry_file"]["methods"][ours][k + "_mean"] for k in cols_b()])
    check("rule on the delivered CAD and Sentry files equals results/sheet_comparison.json", max(d1, d2) <= TOL)

    un = pd.read_csv(R / "uncertainty_cc8-9_summary.csv")
    un = un[un["scale"] == 1.0].set_index("method")
    nb_unc = ns["unc"].set_index("method")
    cols = ["accuracy_mean", "mcc_mean", "brier_mean", "f1_mean"]
    d = max(close(nb_unc.loc["point rule (perturbed orbit)", cols], un.loc["point rule (exact geometry)", cols]),
            close(nb_unc.loc["path cloud (ours)", cols], un.loc["path cloud (ours)", cols]))
    check("uncertainty simulation (short-arc profiles, scale 1) equals results/uncertainty_cc8-9_summary.csv",
          d <= TOL, f"max diff {d:.1e}")
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED'}")
    sys.exit(1 if FAILS else 0)


def cols_b() -> list[str]:
    return ["accuracy", "precision", "recall", "f1", "mcc"]


if __name__ == "__main__":
    main()
