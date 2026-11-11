# The Asteroid's Point of View

Code, data pipeline and paper for **"The Asteroid's Point of View: Path Geometry and Leakage-Free Evaluation for Potentially Hazardous Asteroid Classification"** by Ch. Satwik and Devieswar Kancheti ([paper/main.pdf](paper/main.pdf)).

Prior machine-learning work classifies potentially hazardous asteroids (PHAs) from Earth's point of view: one row per close approach, with miss distance and relative velocity. This project describes each asteroid only by its own orbit (path) and size, and judges hazard from the geometry of that path relative to Earth's orbit as a fixed curve, so the timing of individual encounters drops out.

## Main results (data retrieved 2026-11-11)

| Setting | Method | Accuracy | F1 (PHA) | MCC |
|---|---|---|---|---|
| Earth's view, row split (prior-work protocol) | Random forest | 94.74% | 0.676 | 0.650 |
| Earth's view, asteroid-grouped split (honest) | best of six models, per column | 92.63% | 0.428 | 0.410 |
| Same grouped folds, asteroid's view | training-free path rule | 99.79% | 0.988 | 0.987 |
| All 42,888 NEAs, 5-fold over asteroids | training-free path rule | 99.86% | 0.989 | 0.988 |
| All 42,888 NEAs, 5-fold over asteroids | closed-form tangent-line rule | 99.26% | 0.938 | 0.934 |
| Ceiling (definition on JPL's own MOID) | reference | 99.89% | 0.990 | 0.990 |

With each asteroid's real orbit uncertainties, the decision law (with the Earth-curve systematic measured on training folds) decides 99.43% of the 42,888 NEAs, agrees with the PHA definition evaluated on JPL's own MOID for every decided asteroid, and flags 244 (0.57%) for follow-up. Its only disagreements with JPL's flags are 48 flags that contradict JPL's own definition.

On the Kaggle table behind the highest published accuracy (`shrutimehta/nasa-asteroids-classification`, 4,687 records of 3,692 asteroids), the published record-level protocol scores 99.62% when reproduced, above the 99.55% that the PHA definition itself scores on that file, because it memorises inconsistent labels of repeated asteroids. Split by asteroid, the same random forest scores 99.42%, the same as the training-free rule (99.42%), which never reads the table's MOID column.

For uncertain orbits (simulation with the real error bars of short-arc NEAs), the path cloud is as accurate as the single-orbit rule (98.46% accuracy, MCC 0.949 for both) and lowers the Brier score of its hazard probabilities from 0.0154 to 0.0105.

### The received score sheet: CNN, GPBoost and a tuned random forest

A score sheet received on 11 November 2026 (`data/external/`) reports 72-75% accuracy for a CNN, GPBoost and a tuned random forest on "NASA CAD", "NASA SENTRY" and "NASA SCOUT", without stating the label or the test rows. Decoded from its own metrics, its test sets have 639 rows (CAD), 394 (Sentry) and 9 (Scout), about half of them positive in CAD and Sentry; the delivered files have 33, 2,242 and 72 rows, and only 15 Sentry objects are PHAs. So the sheet measures a different, unstated task. `src/sheet_comparison.py` reruns all four methods with one label (JPL's PHA flag) and the same folds for every method, first on the full NASA CAD table (306,590 approaches of 36,817 asteroids, mean of 5 folds):

| Method | Split | Accuracy | F1 (PHA) | MCC | ROC AUC |
|---|---|---|---|---|---|
| **Ours (training-free)** | asteroid-grouped (honest) | **99.79%** | **0.988** | **0.987** | **0.9998** |
| CNN | asteroid-grouped (honest) | 92.61% | 0.381 | 0.401 | 0.9416 |
| GPBoost | asteroid-grouped (honest) | 91.26% | 0.000 | 0.000 | 0.9193 |
| Tuned random forest | asteroid-grouped (honest) | 87.00% | 0.527 | 0.509 | 0.9374 |
| CNN | row split (leaky) | 92.67% | 0.380 | 0.405 | 0.9431 |
| GPBoost | row split (leaky) | 99.90% | 0.994 | 0.994 | 1.0000 |
| Tuned random forest | row split (leaky) | 89.42% | 0.595 | 0.585 | 0.9576 |

GPBoost's random effect per asteroid memorises the asteroids it has seen: under the row split, where 95.7% of test rows belong to an asteroid also in training, it scores 99.90%; on new asteroids it never predicts a PHA. Our rule needs no training, so the split does not change it (99.79% under both).

On the delivered files (the trained models see each file's own columns; our rule reads each asteroid's orbit from JPL's catalogue):

| File | Ours | CNN | GPBoost | Tuned random forest |
|---|---|---|---|---|
| CAD file: 33 approaches, 2 PHAs (leave-one-out) | **100.00%, F1 1.000** | 93.94%, F1 0.000 | 90.91%, F1 0.000 | 93.94%, F1 0.000 |
| Sentry list: 2,241 of 2,242 objects scored, 15 PHAs (5 x 5 CV) | **100.00%, F1 1.000** | 99.56%, F1 0.656 | 99.70%, F1 0.756 | 99.67%, F1 0.784 |

One Sentry object has no absolute magnitude in JPL's catalogue and is not scored. Scout lists unconfirmed candidates with no JPL orbit or PHA flag yet, so no method can be scored on it. The same comparison as an Excel workbook with the received sheet's tabs is `results/NASA_Asteroid_Comparative_Results_with_Ours.xlsx`.

## Layout

| Path | Contents |
|---|---|
| `src/fetch_data.py` | Downloads NEO orbits (SBDB Query API) and Earth close approaches (CAD API) from NASA/JPL |
| `src/orbit_geometry.py` | Path kernel, exact path-to-path (MOID) solver, closed-form node and tangent-line kernels, Öpik variables |
| `src/earth_view.py` | Earth's-view benchmark under row-wise and asteroid-grouped splits |
| `src/asteroid_view.py` | Asteroid's-view rules and models, temporal split, same-fold comparison |
| `src/uncertainty.py` | Simulation of hazard classification from uncertain orbits (path cloud) |
| `src/pha_core/` | Kernels K1–K5 (bounds, bound-checked exact MOID, signed MOID, delta method, sampling, score), the decision procedure, the decision law in C and Python, and property tests; specified in [docs/PHA_MATH_CORE.md](docs/PHA_MATH_CORE.md). `pha_pipeline.py` adds one policy setting, `s_sys` (Earth-curve systematic, default 0 = behaviour as specified). |
| `src/kernel_realdata.py` | Runs `pha_core` on the real catalogue: bound-checked MOID vs JPL, screening, score fit, decision law with and without `s_sys` |
| `src/kaggle_compare.py` | Same-file comparison on the Kaggle table behind the highest published accuracy, including the memorisation check |
| `src/sheet_comparison.py`, `src/cnn_worker.py` | The received score sheet's three models (CNN, GPBoost, tuned random forest) against ours: decodes the sheet, reruns all four methods on the delivered NASA CAD, Sentry and Scout files with one label and the same folds, and on the full NASA CAD table. The CNN trains in its own process (PyTorch's OpenMP runtime crashes next to the one GPBoost, LightGBM and XGBoost load) |
| `src/make_sheet_xlsx.py` | The same comparison as an Excel workbook with the received sheet's tabs (`results/NASA_Asteroid_Comparative_Results_with_Ours.xlsx`) |
| `data/external/` | The received score sheet and the three NASA files delivered with it, with SHA-256 sums |
| `src/verify_proofs.py` | Numerical check of every lemma, proposition and theorem proved in the paper's Appendix A, on the real catalogue and synthetic orbits |
| `proofs/verify_all.sh` | One command that reruns every check (decision law on all 2^32 inputs, kernel and pipeline tests, Appendix A statements, an audit of 100 numbers in the paper's text, the video's numbers, data checksums, the score-sheet comparison, the Kaggle notebook) and writes `proofs/REPORT.md` |
| `proofs/check_sheet_comparison.py` | Independent check of the score-sheet comparison: decodes the received sheet by exact fractions, recomputes labels and our rule from the raw JPL catalogue, and checks every summary and every workbook cell |
| `proofs/check_notebook.py` | Runs the Kaggle notebook on the snapshot and checks that it reproduces `results/` (every asteroid's distance, the 5-fold, temporal, same-fold and uncertainty results) |
| `proofs/compare_runs.py`, `proofs/REPRODUCTION.md` | File-by-file comparison of two complete runs (every number in `results/` and every generated table), and the record of the clean-clone reproductions |
| `presentation/make_video.py` | Narrated 4.5-minute presentation video (`presentation/asteroids_point_of_view.mp4`); needs macOS `say` and ffmpeg; checks every on-screen number against `results/` first |
| `notebooks/asteroid_point_of_view_kaggle.ipynb` | Self-contained Kaggle notebook of the method: path geometry for every NEA, the 5-fold and temporal evaluation, Earth's view vs the asteroid's view on NASA CAD, the CAD/Sentry/Scout tables and the orbit-uncertainty simulation; prints the paper's value next to each number |
| `src/make_figures.py`, `src/make_tables.py` | Every figure and table in the paper, generated from `results/` |
| `paper/` | LaTeX source (IEEE conference format), figures, generated tables |
| `prompts/kernel_prompt.md` | Prompt for designing additional mathematical kernels |

## Reproduce

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), a C compiler and [Tectonic](https://tectonic-typesetting.github.io/) for the PDF. On macOS, LightGBM, XGBoost and GPBoost also need OpenMP (`brew install libomp`).

To reproduce the paper exactly, from the NASA/JPL data downloaded on 11 November 2026 (`datasets/neo_datasets_2026-11-11.zip`, checked against its SHA-256 sums):

```bash
./run_all.sh --snapshot
```

To rerun everything on freshly downloaded data instead (JPL updates its catalogue daily, so the numbers shift slightly):

```bash
./run_all.sh
```

Either way the script creates `.venv`, installs `requirements.txt`, runs every experiment (about 1.5 hours on a laptop, 45 minutes of it the score-sheet comparison), compiles `paper/main.pdf` and finishes with `proofs/verify_all.sh`, which writes `proofs/REPORT.md`. Two clean-clone reproductions and their file-by-file comparison are recorded in [proofs/REPRODUCTION.md](proofs/REPRODUCTION.md).

## Data source

NASA/JPL Solar System Dynamics: [SBDB Query API](https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html) and [Close-Approach Data API](https://ssd-api.jpl.nasa.gov/doc/cad.html).
