# External files for the comparison with the score sheet

Used only by `src/sheet_comparison.py`. Checksums are in `SHA256SUMS.txt`.

| File | What it is | Rows |
|---|---|---|
| `NASA_Asteroid_Final_Comparative_Results.xlsx` | Score sheet received on 2026-11-11 (written by openpyxl at 05:21 UTC): accuracy, precision, recall, F1 and ROC AUC of a CNN, GPBoost and a tuned random forest on "NASA CAD", "NASA SCOUT" and "NASA SENTRY". It does not state the label, the split or the test rows. | 9 results |
| `cad.csv` | NASA/JPL Close-Approach Data API output: Earth approaches within 0.05 au, 2026-09-27 to 2026-11-19 (fields `des, orbit_id, jd, cd, dist, dist_min, dist_max, v_rel, v_inf, t_sigma_f, h`) | 33 |
| `sentry_risk_list.csv` | NASA/JPL Sentry impact-risk list (`des, diameter, h, ip, n_imp, ps_max, ps_cum, ts_max, v_inf, range, last_obs, ...`) | 2,242 |
| `scout.csv` | NASA/JPL Scout list of unconfirmed NEO candidates on the NEOCP, run of 2026-11-08 21:45 | 72 |

The NASA files are public data from the JPL Solar System Dynamics APIs (CAD, Sentry, Scout).
