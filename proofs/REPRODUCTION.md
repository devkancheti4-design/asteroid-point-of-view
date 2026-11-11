# Reproduction record

## Commit e9ded9d (the paper as first pushed), 2026-11-11

Two complete reproductions, each from a clean clone and the paper's data snapshot (`./run_all.sh --snapshot`,
which checks the snapshot against its SHA-256 sums before using it):

| Run | Source of the clone | Wall time | Outcome |
|---|---|---|---|
| A | `git clone` of the local repository | 58 min | every step ran, the paper compiled |
| B | `gh repo clone devkancheti4-design/asteroid-point-of-view` (GitHub) | 58 min | every step ran, the paper compiled |

Both ran at the same time on one laptop (macOS, Apple Silicon), alongside other jobs.

### Every result is identical

`proofs/compare_runs.py` compares every number in `results/*.json` and `results/*.csv` (absolute tolerance
1e-9) and every generated LaTeX table in `paper/tables/` (as text):

| Comparison | Files identical |
|---|---|
| run A vs run B | 33 of 33 |
| original run vs run A | 33 of 33 shared files |
| original run vs run B | 33 of 33 shared files |

The two files not shared, `results/sheet_comparison.json` and `results/sheet_comparison_folds.csv`, were added
after e9ded9d. Only wall-clock timings differ, and the comparison skips them: the bound-checked MOID kernel took a
median 2.08 ms (A) and 2.00 ms (B) per asteroid on the loaded machine, against 0.66 ms in the original run.

### Proof report

In both clones `proofs/verify_all.sh` passed every check except the paper-number audit, which at that commit
required the timing above to equal the paper's 0.66 ms exactly. The audit now checks wall-clock timings to within
a factor of 10 and every other number exactly; rerun on each clone's own results, it passes 68 of 68 in both.

### One stale file found and replaced

`results/uncertainty_cc5-7_meta.json` in the original run lacked the key `profile_condition_codes`, which the
committed `src/uncertainty.py` writes: the file came from an earlier version of that script. Every value the two
versions share was equal. The reproduced file replaced it.

## Commit 5522230 (score-sheet comparison, Kaggle notebook, proof checks 8-10), 2026-11-11

A fresh `gh repo clone` of the pushed commit, run end to end with `./run_all.sh --snapshot`: every step ran
(1 h 09 min on the same laptop, partly alongside other jobs), and all 10 sections of `proofs/verify_all.sh` passed
inside the clone, including the audit of the paper's numbers, the independent check of the score-sheet comparison
and the Kaggle notebook check.

| Comparison | Files identical (tolerance 1e-9) |
|---|---|
| original run vs the clone | 33 of 35 shared files |

The two files that differ are the score-sheet comparison's results (`results/sheet_comparison.json` and
`results/sheet_comparison_folds.csv`), and only in GPBoost's probability-based numbers. GPBoost estimates the
variance of its random effect per asteroid with multithreaded floating-point sums, whose order varies between runs;
the estimate differs by at most 6e-9, which moves GPBoost's ROC AUC by at most 8e-7 and its PR AUC by at most 9e-6
per fold. Its hard predictions, and so its accuracy, precision, recall, F1 and MCC, are identical, as is every
number of the CNN, the tuned random forest and the path rule. No digit reported in the paper or the README changes.
(`paper/tables/tab_sheet.tex`, the third file not shared, was added after this commit.)

## How to repeat

```bash
git clone https://github.com/devkancheti4-design/asteroid-point-of-view.git && cd asteroid-point-of-view
./run_all.sh --snapshot
python proofs/compare_runs.py <your first run> <your second run>
```
