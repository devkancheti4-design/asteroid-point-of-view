#!/usr/bin/env bash
# Re-run every check behind the paper's proofs and numbers, and write proofs/REPORT.md.
#   1. decision law vs its specification on all 2^32 int32 inputs (exhaustive, compiled C)
#   2. kernel property tests on synthetic orbits (independent brute-force MOID, gradients, probabilities)
#   3. decision-procedure tests (all ten acts, termination)
#   4. every lemma and proposition of Appendix A on the real catalogue (src/verify_proofs.py)
#   5. every key number in the paper's text recomputed from results/ (proofs/check_paper_numbers.py)
#   6. every number in the video narration recomputed from results/
#   7. SHA-256 checksums of the data the results were computed from
#   8. SHA-256 check of the received score sheet and the three NASA files delivered with it (data/external)
#   9. the score-sheet comparison recomputed independently (proofs/check_sheet_comparison.py)
#  10. the Kaggle notebook run on the snapshot reproduces results/ (proofs/check_notebook.py)
#  11. the statistical evidence against the state of the art recomputed independently (proofs/check_sota_evidence.py)
# Steps 4-6 and 9-11 read results/; run ./run_all.sh first on a fresh checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$PWD/.venv/bin/python"
OUT=proofs/REPORT.md
run() {  # run <title> <command...>: append a section with the command's output to the report
  local title="$1"; shift
  printf '\n## %s\n\n```\n' "$title" >> "$OUT"
  if "$@" >> "$OUT" 2>&1; then status=PASS; else status=FAIL; FAILED=1; fi
  printf '```\n\n**%s**\n' "$status" >> "$OUT"
  echo "$status  $title"
}
FAILED=0
{
  echo "# Verification report"
  echo
  echo "Generated $(date -u '+%Y-%m-%d %H:%M UTC') on $(uname -sm) by proofs/verify_all.sh."
} > "$OUT"
cc -O2 -fwrapv -Wall -DPHA_LAW_MAIN -o proofs/pha_law_check src/pha_core/out/pha_law.c
run "1. Decision law equals its specification on all 2^32 inputs" ./proofs/pha_law_check
run "2. Kernel property tests (synthetic orbits, independent brute force)" \
    bash -c "cd src/pha_core && '$PY' -u tests/test_kernels.py --n 300 --out ../../results/kernel_checks_synthetic.json"
run "3. Decision procedure tests" bash -c "cd src/pha_core && '$PY' -u tests/test_pipeline.py"
run "4. Appendix A statements on the real catalogue" bash -c "cd src && '$PY' verify_proofs.py"
run "5. Paper numbers recomputed from results/" "$PY" proofs/check_paper_numbers.py
run "6. Video narration numbers recomputed from results/" \
    "$PY" -c "import sys; sys.path.insert(0, 'presentation'); import make_video; make_video.check_numbers()"
run "7. Data checksums (SHA-256)" shasum -a 256 data/processed/sbdb_neo.csv data/processed/cad_earth.csv \
    data/kaggle/nasa-asteroids-classification/nasa.csv
run "8. Received score sheet and delivered NASA files (SHA-256 verified)" \
    bash -c "cd data/external && shasum -a 256 -c SHA256SUMS.txt"
run "9. Score-sheet comparison recomputed independently" "$PY" proofs/check_sheet_comparison.py
run "10. Kaggle notebook reproduces the results" "$PY" proofs/check_notebook.py
run "11. Statistical evidence against the state of the art recomputed independently" "$PY" proofs/check_sota_evidence.py
rm -f proofs/pha_law_check
echo
if [ "$FAILED" = 0 ]; then echo "ALL CHECKS PASS -> $OUT"; else echo "SOME CHECKS FAILED -> $OUT"; exit 1; fi
