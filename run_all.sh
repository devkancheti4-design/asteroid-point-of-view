#!/usr/bin/env bash
# Rebuild every result, figure, table and the paper PDF, then rerun every check (proofs/REPORT.md).
#
#   ./run_all.sh --snapshot   reproduce the paper exactly, from the NASA/JPL data downloaded on 2026-11-11
#                             (datasets/neo_datasets_2026-11-11.zip, checked against its SHA-256 sums)
#   ./run_all.sh              download fresh data first (JPL updates daily, so numbers shift slightly)
#
# About 1.5 hours on a laptop (45 minutes of it the score-sheet comparison). Needs uv, a C compiler, tectonic and,
# on macOS, OpenMP for LightGBM, XGBoost and GPBoost (brew install libomp).
set -euo pipefail
cd "$(dirname "$0")"
SNAPSHOT=0
for arg in "$@"; do
  case "$arg" in
    --snapshot) SNAPSHOT=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ ! -x .venv/bin/python ]; then
  uv venv .venv
  uv pip install --python .venv/bin/python -r requirements.txt
fi
PY="$PWD/.venv/bin/python"
mkdir -p results data/processed

if [ "$SNAPSHOT" = 1 ]; then
  SNAP=datasets/neo_datasets_2026-11-11
  rm -rf data/snapshot_tmp && mkdir -p data/snapshot_tmp
  unzip -q "$SNAP.zip" -d data/snapshot_tmp
  cp data/snapshot_tmp/neo_datasets_2026-11-11/{sbdb_neo.csv,cad_earth.csv,provenance.json} data/processed/
  (cd data/processed && shasum -a 256 -c ../snapshot_tmp/neo_datasets_2026-11-11/SHA256SUMS.txt)
  rm -rf data/snapshot_tmp
  echo "using the 2026-11-11 snapshot"
fi

cd src
[ "$SNAPSHOT" = 1 ] || "$PY" fetch_data.py           # NASA/JPL SBDB + close-approach data
"$PY" earth_view.py                                  # Earth's-view benchmark, leaky vs grouped splits
"$PY" asteroid_view.py                               # asteroid's view: rules, ML, temporal, same folds
# K1-K5 kernels and the decision law (src/pha_core): property tests, exhaustive proof, real-catalogue run
(cd pha_core && "$PY" tests/test_kernels.py --n 300 --out ../../results/kernel_checks_synthetic.json \
   && "$PY" tests/test_pipeline.py)
cc -O2 -fwrapv -DPHA_LAW_MAIN -o ../results/pha_law_check pha_core/out/pha_law.c && ../results/pha_law_check
rm -f ../results/pha_law_check
"$PY" kernel_realdata.py                             # bound-checked MOID, screening, K5 fit, decision law on all NEAs
# Same-file comparison on the public Kaggle table behind the highest reported accuracy (CC0; no login needed)
K=../data/kaggle/nasa-asteroids-classification
if [ ! -f "$K/nasa.csv" ]; then
  mkdir -p "$K"
  curl -sL -o "$K/nasa-asteroids-classification.zip" \
    https://www.kaggle.com/api/v1/datasets/download/shrutimehta/nasa-asteroids-classification
  (cd "$K" && unzip -o -q nasa-asteroids-classification.zip nasa.csv)
fi
"$PY" -I kaggle_compare.py "$K/nasa.csv"
# The received score sheet's three models (CNN, GPBoost, tuned RF) vs ours on the delivered NASA files and full CAD
"$PY" sheet_comparison.py
"$PY" make_sheet_xlsx.py                             # the same comparison as an Excel workbook
"$PY" uncertainty.py --codes 5-7 --scales 0.25,0.5,1,2   # uncertain orbits, moderately determined profiles
"$PY" uncertainty.py --codes 8-9 --scales 0.25,0.5,1,2   # uncertain orbits, short-arc profiles
"$PY" verify_proofs.py                              # numerical checks of every statement in Appendix A
"$PY" make_figures.py
"$PY" make_tables.py
cd ../paper
tectonic -X compile main.tex
cd ..
./proofs/verify_all.sh                               # every check, one report: proofs/REPORT.md
echo "done: paper/main.pdf, proofs/REPORT.md (video: .venv/bin/python presentation/make_video.py)"
