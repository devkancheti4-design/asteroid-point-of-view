# Running every method on Kaggle: step by step

The notebook runs our training-free path rule against all nine Earth's-view models:

- the CNN, GPBoost and the tuned random forest;
- the six models of prior work: logistic regression, *k*-NN, random forest, XGBoost, LightGBM and an MLP.

It prints the paper's Tables IV and IX, a fold-by-fold check against the paper, and exact paired significance tests.
Everything runs in Kaggle, so the device you use needs only a web browser. There are two ways to start.

## Way A: a fresh device, with nothing to download or upload

1. Open this link and sign in to Kaggle. Kaggle opens a copy of the notebook:
   <https://www.kaggle.com/kernels/welcome?src=https://github.com/devkancheti4-design/asteroid-point-of-view/blob/main/comparison/kaggle_notebook.ipynb>
   If the link does not open it, go to **Create → New Notebook → File → Import Notebook** and paste
   `https://github.com/devkancheti4-design/asteroid-point-of-view/blob/main/comparison/kaggle_notebook.ipynb`.
2. In the right-hand panel, set **Internet: On** (Kaggle asks for phone verification the first time) and
   **Accelerator: None (CPU)**.
3. Click **Save Version → Save & Run All (Commit) → Save**, then continue at Step 4 below.

With no dataset attached, the notebook downloads the same code and data from the project's public GitHub repository
(about 50 MB) and runs.

## Way B: upload the two files

You upload `asteroid_model_comparison.zip` (about 30 MB), which holds the full model code, the data and the paper's
stored results, and you import `kaggle_notebook.ipynb`. Steps 1 to 4 below explain how.

## What is in the zip

| File | What it is |
|---|---|
| `compare_models.py` | the full model code for all ten methods (one file) |
| `kaggle_notebook.ipynb` | the notebook to import into Kaggle; it runs `compare_models.py` |
| `paper_notebook.ipynb` | optional: the rest of the paper (the asteroid's-view models, new discoveries, uncertain orbits) |
| `KAGGLE_GUIDE.md` | this guide |
| `MODELS_AND_PAPERS.md`, `references.bib` | every method's settings and its research papers, with links |
| `data/sbdb_neo.csv` | JPL Small-Body Database: 43,025 near-Earth asteroids (snapshot of 2026-11-11) |
| `data/cad_earth.csv` | JPL close-approach data: every Earth approach within 0.5 au, 1910–2024 |
| `data/cad.csv`, `data/sentry_risk_list.csv`, `data/scout.csv` | the three NASA tables of the comparison sheet |
| `data/provenance.json`, `data/SHA256SUMS.txt` | where the data came from, and checksums |
| `paper_results/` | the paper's stored results, used for the fold-by-fold check |

## Step 1: upload the zip as a Kaggle dataset

1. Sign in at [kaggle.com](https://www.kaggle.com). Click **Create** (the **+** button), then **New Dataset**.
2. Drag `asteroid_model_comparison.zip` into the upload box. Kaggle unpacks it.
3. Title: `asteroid-model-comparison`. Keep the visibility **Private**, because the paper is not published yet.
4. Click **Create** and wait until the dataset page shows the files.

## Step 2: create the notebook

1. On the dataset page, click **New Notebook**. Or use **Create → New Notebook** and then, in the right-hand panel,
   **Add Input → Your Work → Datasets → asteroid-model-comparison**.
2. Import the ready notebook: **File → Import Notebook**, then choose `kaggle_notebook.ipynb`. It is inside the zip
   and was also sent to you as a separate file. If importing replaces the notebook's inputs, add the dataset again as
   in point 1.
3. In the right-hand panel (**Settings** / **Session options**):
   - **Internet: On.** GPBoost is installed with pip. Kaggle asks for phone verification the first time.
   - **Accelerator: None (CPU).** The code runs on the CPU, so a GPU does not help.

## Step 3: run it

- **Full run (recommended):** click **Save Version → Save & Run All (Commit) → Save**. Kaggle runs the notebook in
  the background, so you can close the browser. Allow 2–4 hours. The same run takes about an hour on the authors'
  laptop, and Kaggle's CPUs are slower. This is well within Kaggle's per-session limit (12 hours for CPU sessions at
  the time of writing).
- **Quick check first (optional, about 20–30 minutes):** in the third code cell, set `QUICK = True` and click
  **Run All**. Only the first fold of each split runs. That shows everything works, but it is not comparable with
  the paper. Set `QUICK = False` again before the full run.

To run only part of the comparison, edit `UNITS` in the third code cell:

| Unit | What it runs |
|---|---|
| `B-cad` | the 33 NASA CAD approaches of the comparison sheet, leave-one-out: rule, CNN, GPBoost, tuned RF |
| `B-sentry` | the 2,241 Sentry objects, 5 × 5 repeated cross-validation: rule, CNN, GPBoost, tuned RF |
| `E-grouped` | 306,590 close approaches, folds grouped by asteroid (honest): rule and the six prior-work models |
| `E-row` | the same with folds over rows (the leaky prior-work split) |
| `C-grouped` | 306,590 close approaches, grouped folds: rule, CNN, GPBoost, tuned RF |
| `C-row` | the same with folds over rows |

## Step 4: read the results

The last cell prints the report and one table per unit. The report has four parts:

1. **Results**: the mean over folds for every method, next to the number printed in the paper.
2. **Fold-by-fold check**: every fold against `paper_results/`. Each line says IDENTICAL or DIFFERENT, with the largest
   difference.
3. **Claim check**: on every honest evaluation (grouped folds and the two small tables), the rule beats every model
   in every fold on accuracy, F1 and MCC.
4. **Exact paired McNemar test**: on the grouped folds, using the row-level predictions. It reports how many rows
   each model gets wrong where the rule is right and vice versa, with the exact p-value.

How to interpret it:

- **The rule gives exactly the paper's numbers on any computer.** It has no training, and its geometry is
  deterministic.
- **The trained models can differ slightly from the paper on Kaggle.** The paper's runs used macOS on Apple silicon,
  with the library versions in `requirements.txt`. Kaggle has other processors and library versions, and these change
  the CNN, GPBoost, the forests and the boosted trees a little. The verdict then shows a NOTE with the largest
  difference. This is expected, and it does not affect the conclusion. On the authors' laptop every number matches
  the paper.
- **The verdict must show PASS** for the rule, for the claim check and for the paired tests.

All outputs are in `/kaggle/working/comparison_output/`, in the notebook version's **Output** tab, where you can
download them:

| File | Contents |
|---|---|
| `report.txt` | the report |
| `<unit>.csv` | every fold's accuracy, balanced accuracy, precision, recall, F1, MCC, ROC AUC and PR AUC for every method |
| `predictions_<unit>.csv.gz` | the row-level predictions of every method (E and C units) |
| `paired_tests.json` | the paired tests |
| `<unit>_notes.json` | training details: the CNN's epochs, the forest's tuned settings, GPBoost's random-effect variance |

## The optional second notebook

`paper_notebook.ipynb` reproduces the rest of the paper from the same dataset:

- all 42,888 near-Earth asteroids over five folds;
- training on asteroids found before 2016 and testing on later ones;
- the leaky versus honest Earth's-view split;
- the three NASA tables;
- uncertain orbits.

Import it the same way (File → Import Notebook) and run all cells. It needs no Internet and no GPBoost.

## If something goes wrong

| Message | Fix |
|---|---|
| `ERROR: Could not find a version ... gpboost` or `No module named 'gpboost'` | Internet is off: Settings → Internet → On (needs a verified phone number), then run again |
| `compare_models.py not found` or `sbdb_neo.csv not found` | the dataset is not attached: Add Input → Your Work → Datasets → asteroid-model-comparison |
| the run stops after many hours | run fewer units per version: for example `UNITS = "C-row"` in one version and the other five in another |
| `UNITS THAT FAILED: ...` | the messages above it name the problem; the other units still ran, and the report covers them |
