# The Asteroid's Point of View

Code, data pipeline and paper for **"The Asteroid's Point of View: Path Geometry and Leakage-Free Evaluation for Potentially Hazardous Asteroid Classification"** by Ch. Satwik and Devieswar Kancheti ([paper/main.pdf](paper/main.pdf)).

Prior machine-learning work classifies potentially hazardous asteroids (PHAs) from Earth's point of view: one row per close approach, with miss distance and relative velocity. This project describes each asteroid only by its own orbit (path) and size, and judges hazard from the geometry of that path relative to Earth's orbit as a fixed curve, so the timing of individual encounters drops out.

## Main results (data retrieved 2026-11-11)

| Setting | Method | Accuracy | F1 (PHA) | MCC |
|---|---|---|---|---|
| Earth's view, row split (prior-work protocol) | Random forest | 94.74% | 0.676 | 0.650 |
| Earth's view, asteroid-grouped split (honest) | best model | 92.63% | 0.428 | 0.410 |
| Same grouped folds, asteroid's view | training-free path rule | 99.79% | 0.988 | 0.987 |
| All 42,888 NEAs, 5-fold over asteroids | training-free path rule | 99.86% | 0.989 | 0.988 |
| All 42,888 NEAs, 5-fold over asteroids | closed-form tangent-line rule | 99.26% | 0.938 | 0.934 |
| Ceiling (definition on JPL's own MOID) | reference | 99.89% | 0.990 | 0.990 |

With each asteroid's real orbit uncertainties, the decision law (with the Earth-curve systematic measured on training folds) decides 99.43% of the 42,888 NEAs, agrees with the PHA definition evaluated on JPL's own MOID for every decided asteroid, and flags 244 (0.57%) for follow-up. Its only disagreements with JPL's flags are 48 flags that contradict JPL's own definition.

On the Kaggle table behind the highest published accuracy (`shrutimehta/nasa-asteroids-classification`, 4,687 records of 3,692 asteroids), the published record-level protocol scores 99.62% when reproduced, above the 99.55% that the PHA definition itself scores on that file, because it memorises inconsistent labels of repeated asteroids. Split by asteroid, the same random forest scores 99.42%, the same as the training-free rule (99.42%), which never reads the table's MOID column.

For uncertain orbits (simulation with the real error bars of short-arc NEAs), the path cloud is as accurate as the single-orbit rule (98.46% accuracy, MCC 0.949 for both) and lowers the Brier score of its hazard probabilities from 0.0154 to 0.0105.

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
| `src/verify_proofs.py` | Numerical check of every lemma, proposition and theorem proved in the paper's Appendix A, on the real catalogue and synthetic orbits |
| `proofs/verify_all.sh` | One command that reruns every check (decision law on all 2^32 inputs, kernel and pipeline tests, Appendix A statements, an audit of 68 numbers in the paper's text, the video's numbers, data checksums) and writes `proofs/REPORT.md` |
| `presentation/make_video.py` | Narrated 4.5-minute presentation video (`presentation/asteroids_point_of_view.mp4`); needs macOS `say` and ffmpeg; checks every on-screen number against `results/` first |
| `src/make_figures.py`, `src/make_tables.py` | Every figure and table in the paper, generated from `results/` |
| `paper/` | LaTeX source (IEEE conference format), figures, generated tables |
| `prompts/kernel_prompt.md` | Prompt for designing additional mathematical kernels |

## Reproduce

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and [Tectonic](https://tectonic-typesetting.github.io/) for the PDF.

To reproduce the paper exactly, from the NASA/JPL data downloaded on 11 November 2026 (`datasets/neo_datasets_2026-11-11.zip`, checked against its SHA-256 sums):

```bash
./run_all.sh --snapshot
```

To rerun everything on freshly downloaded data instead (JPL updates its catalogue daily, so the numbers shift slightly):

```bash
./run_all.sh
```

Either way the script creates `.venv`, installs `requirements.txt`, runs every experiment (about 30-35 minutes on a laptop), compiles `paper/main.pdf` and finishes with `proofs/verify_all.sh`, which writes `proofs/REPORT.md`.

## Data source

NASA/JPL Solar System Dynamics: [SBDB Query API](https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html) and [Close-Approach Data API](https://ssd-api.jpl.nasa.gov/doc/cad.html).
