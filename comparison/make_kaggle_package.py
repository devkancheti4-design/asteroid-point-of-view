"""Build the Kaggle upload: kaggle_upload/asteroid_model_comparison.zip, plus comparison/kaggle_notebook.ipynb.

The zip holds the full model code, the notebook that runs it, the optional notebook for the rest of the paper, the
guides, the data snapshot of 2026-11-11 and the paper's stored results. Its entries carry fixed timestamps, so the same
inputs always give the same zip. Upload it as one Kaggle dataset (comparison/KAGGLE_GUIDE.md).
Usage (from the repository root): .venv/bin/python comparison/make_kaggle_package.py
"""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "kaggle_upload"
ZIP = OUT_DIR / "asteroid_model_comparison.zip"
NOTEBOOK = ROOT / "comparison" / "kaggle_notebook.ipynb"

FILES = {   # path in the zip -> source in the repository
    "compare_models.py": "comparison/compare_models.py",
    "kaggle_notebook.ipynb": "comparison/kaggle_notebook.ipynb",
    "paper_notebook.ipynb": "notebooks/asteroid_point_of_view_kaggle.ipynb",
    "KAGGLE_GUIDE.md": "comparison/KAGGLE_GUIDE.md",
    "MODELS_AND_PAPERS.md": "comparison/README.md",
    "references.bib": "comparison/references.bib",
    "data/sbdb_neo.csv": "data/processed/sbdb_neo.csv",
    "data/cad_earth.csv": "data/processed/cad_earth.csv",
    "data/provenance.json": "data/processed/provenance.json",
    "data/cad.csv": "data/external/cad.csv",
    "data/sentry_risk_list.csv": "data/external/sentry_risk_list.csv",
    "data/scout.csv": "data/external/scout.csv",
    "paper_results/sheet_comparison_folds.csv": "results/sheet_comparison_folds.csv",
    "paper_results/sheet_comparison.json": "results/sheet_comparison.json",
    "paper_results/earth_view_folds.csv": "results/earth_view_folds.csv",
    "paper_results/sota_evidence.json": "results/sota_evidence.json",
}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""
# The Asteroid's Point of View: every method compared (Kaggle)

This notebook runs our **training-free path rule** against **nine Earth's-view models** on JPL data:

- the three further models of the paper's Table IX: a **CNN**, **GPBoost** and a **tuned random forest**;
- the six models of prior work from Table IV: logistic regression, k-NN, random forest, XGBoost, LightGBM and an MLP.

The label is JPL's PHA flag for every method. The run reproduces Tables IV and IX, checks every fold against the
paper's stored results and runs exact paired McNemar tests. All the code is in `compare_models.py` in the attached
dataset.

**Before running:**

1. **Settings → Internet → On**, to install GPBoost.
2. **Accelerator: None (CPU).**
3. Optional: **Add Input → Your Work → Datasets → asteroid-model-comparison** (the uploaded zip). If no dataset is
   attached, the notebook downloads the same code and data from the project's public GitHub repository.

**Full run:** Save Version → Save & Run All (Commit). Allow 2–4 hours on Kaggle's CPU.
**Quick check:** set `QUICK = True` in the third code cell and click Run All (about 20–30 minutes).
"""),
    code("""
# Step 1. Install GPBoost, the only library Kaggle does not preinstall (needs Settings > Internet > On).
# The paper used gpboost 1.8.0; if that version is not available here, the latest one is installed.
!pip install -q gpboost==1.8.0 || pip install -q gpboost
!pip list 2>/dev/null | grep -i -E "^(gpboost|torch|scikit-learn|xgboost|lightgbm|numpy|pandas) "
"""),
    code("""
# Step 2. Get the files: from the attached dataset (the uploaded zip) or, if none is attached, from GitHub
import glob, os, subprocess, zipfile

REPO = "https://github.com/devkancheti4-design/asteroid-point-of-view.git"   # public repository of the paper

def find(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True)
                  + glob.glob(f"/kaggle/working/pkg/**/{name}", recursive=True)
                  + glob.glob(f"/kaggle/working/repo/**/{name}", recursive=True))
    return hits[0] if hits else None

SCRIPT = find("compare_models.py")
if SCRIPT is None:      # the uploaded zip is still zipped: unpack it into the working folder
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        zipfile.ZipFile(z).extractall("/kaggle/working/pkg")
    SCRIPT = find("compare_models.py")
if SCRIPT is None:      # nothing attached (a fresh device): download the code and the data snapshot from GitHub
    if not os.path.isdir("/kaggle/working/repo/.git"):
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPO, "/kaggle/working/repo"], check=True)
    for z in glob.glob("/kaggle/working/repo/datasets/neo_datasets_*.zip"):   # JPL data of 2026-11-11
        zipfile.ZipFile(z).extractall("/kaggle/working/repo/data/snapshot_tmp")
    SCRIPT = find("compare_models.py")
assert SCRIPT, "compare_models.py not found: attach the dataset, or turn Internet on (Settings) to download it"
for name in ["compare_models.py", "sbdb_neo.csv", "cad_earth.csv", "cad.csv", "sentry_risk_list.csv",
             "sheet_comparison_folds.csv", "earth_view_folds.csv"]:
    print(f"{name:28s} {find(name) or 'MISSING'}")
"""),
    code("""
# Step 3. Run every method. Each unit runs in its own process and saves its results as soon as it finishes.
# QUICK = True: only the first fold of each split (about 20-30 minutes), to check that everything works.
# QUICK = False: the paper's full comparison (allow 2-4 hours on Kaggle's CPU).
QUICK = False
UNITS = "B-cad B-sentry E-grouped E-row C-grouped C-row"   # all six units (see KAGGLE_GUIDE.md)
extra = "--max-folds 1" if QUICK else ""
!python "{SCRIPT}" --units {UNITS} {extra}
"""),
    code("""
# Step 4. The report and one table per unit (all files are in /kaggle/working/comparison_output/)
import pandas as pd
OUT = "/kaggle/working/comparison_output"
print(open(f"{OUT}/report.txt").read())
for unit in ["E-grouped", "C-grouped", "B-sentry", "B-cad", "E-row", "C-row"]:
    path = f"{OUT}/{unit}.csv"
    if os.path.exists(path):
        f = pd.read_csv(path)
        print(f"\\n{unit} (mean over {f['fold'].nunique()} fold(s) or repeat(s))")
        display(f.groupby("method", sort=False)[["accuracy", "precision", "recall", "f1", "mcc", "roc_auc"]]
                .mean().round(4))
"""),
    md("""
## Reading the results

- **The rule gives exactly the paper's numbers on any computer.** It has no training, and its geometry is
  deterministic.
- **The trained models can differ slightly from the paper on Kaggle's processors and library versions.** The paper's
  runs used macOS on Apple silicon with the versions in `requirements.txt`, and on that machine every number matches.
  The report's verdict states the largest difference.
- **Honest evaluations:** folds grouped by asteroid, and the two small NASA tables, where no object repeats. On
  these, the rule beats every model in every fold. Only the leaky row split lets GPBoost score higher, because its
  random effect per asteroid memorises asteroids it has already seen.
- **Downloads:** the **Output** tab of this notebook version holds `comparison_output/`. It contains the report, every
  fold's metrics, the row-level predictions and the paired tests.

The methods, their settings and their research papers are in `MODELS_AND_PAPERS.md` and `references.bib` in the
dataset.
"""),
]


def write_notebook():
    nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i}"
    NOTEBOOK.write_text(json.dumps(nb, indent=1) + "\n")


def main():
    write_notebook()
    OUT_DIR.mkdir(exist_ok=True)
    sums = []
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for arc, src in FILES.items():
            data = (ROOT / src).read_bytes()
            if arc.startswith("data/"):
                sums.append(f"{hashlib.sha256(data).hexdigest()}  {arc.removeprefix('data/')}")
            info = zipfile.ZipInfo(arc, date_time=(2026, 11, 12, 0, 0, 0))
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            z.writestr(info, data)
        info = zipfile.ZipInfo("data/SHA256SUMS.txt", date_time=(2026, 11, 12, 0, 0, 0))
        info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
        z.writestr(info, "\n".join(sums) + "\n")
    digest = hashlib.sha256(ZIP.read_bytes()).hexdigest()
    print(f"wrote {ZIP.relative_to(ROOT)}: {ZIP.stat().st_size / 1e6:.1f} MB, {len(FILES) + 1} files, sha256 {digest}")
    print(f"wrote {NOTEBOOK.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
