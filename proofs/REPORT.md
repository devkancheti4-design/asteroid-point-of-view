# Verification report

Generated 2026-11-11 17:04 UTC on Darwin arm64 by proofs/verify_all.sh.

## 1. Decision law equals its specification on all 2^32 inputs

```
pha_law vs specification over all 4294967296 int32 inputs: 0 mismatches
  act 0: 4294967176
  act 1: 60
  act 2: 12
  act 3: 12
  act 4: 12
  act 5: 4
  act 6: 4
  act 7: 8
  act 8: 4
  act 9: 4
```

**PASS**

## 2. Kernel property tests (synthetic orbits, independent brute force)

```
K1: nodes on the reference plane, |r| law, orthonormal basis, rotation-matrix agreement
  worst |z at nodes| 1.04e-15 au, | |r| - p/(1+e cos f) | 7.11e-15, |B B^T - I| 4.44e-16, basis vs matrices 2.22e-16
K2: exact MOID vs the independent brute force; bound order L0 <= L_lon <= MOID <= U_lon <= U_node
  310 orbits (300 synthetic + 10 edge cases) in 44 s
  |MOID - brute force|: max 2.56e-11 au, median 2.22e-16;  bound order violated on 0;  uncertified 0
  certified gap U_lon - L_lon where MOID < 0.2 au: max 9.40e-05 au (n=110)
K2: vectorised moid_batch vs moid (elliptic)
  max |batch - exact| 1.19e-13 au (n=200)
K4: closed-form gradients vs central finite differences
  worst |FD - closed form|: nodal 7.16e-09, signed MOID (envelope theorem) 1.51e-07
K4: delta method vs Monte Carlo vs unscented (synthetic sigmas scaled so P is not 0 or 1)
  MOID 0.02969  s 9.81e-03  P_lin 0.9808  P_mc 0.9788 [0.9738, 0.9828]  P_ut 0.9809
  MOID 0.04082  s 4.42e-03  P_lin 0.9811  P_mc 0.9830 [0.9785, 0.9866]  P_ut 0.9811
  MOID 0.04331  s 4.67e-03  P_lin 0.9242  P_mc 0.9097 [0.9004, 0.9182]  P_ut 0.9114
  MOID 0.04614  s 3.37e-03  P_lin 0.8736  P_mc 0.8752 [0.8646, 0.8851]  P_ut 0.8746
  MOID 0.02580  s 9.91e-03  P_lin 0.9927  P_mc 0.9965 [0.9941, 0.9979]  P_ut 0.9927
  MOID 0.05288  s 1.27e-03  P_lin 0.0115  P_mc 0.0132 [0.0101, 0.0173]  P_ut 0.0116
  MOID 0.03885  s 5.17e-03  P_lin 0.9845  P_mc 0.9852 [0.9810, 0.9885]  P_ut 0.9845
  MOID 0.03110  s 1.37e-02  P_lin 0.9169  P_mc 0.9085 [0.8992, 0.9170]  P_ut 0.9116
  MOID 0.06563  s 1.03e-02  P_lin 0.0640  P_mc 0.0693 [0.0618, 0.0775]  P_ut 0.0663
  MOID 0.06715  s 9.56e-03  P_lin 0.0364  P_mc 0.0375 [0.0320, 0.0438]  P_ut 0.0355
  MOID 0.02059  s 1.33e-02  P_lin 0.9868  P_mc 0.9922 [0.9890, 0.9945]  P_ut 0.9885
  delta method inside the MC 95% interval (+-0.01) on 11 of 11; max |P_lin - P_mc| 0.0145, max |P_ut - P_mc| 0.0038
K4: bivariate nodal probability vs Monte Carlo of U_node (same synthetic sigmas)
  max |P_nodal (bivariate normal) - P_MC(U_node)| 0.0011 over 12 orbits
K5: the fitter recovers known (beta, w) from synthetic labels
  true beta 0.00070 w 0.00150 -> fitted beta 0.00059 w 0.00138
======================================================================
ALL CHECKS PASS
```

**PASS**

## 3. Decision procedure tests

```
invalid input (e < 0)                                HUMAN              HUMAN
faint (H = 24)                                       NOT_PHA_SIZE       NOT_PHA_SIZE
distant Amor, tiny sigma                             NOT_PHA_GEOMETRY   LINEAR -> NOT_PHA_GEOMETRY
                                                     P_lin=0.0000 s=0.0000 lin_ok=True
node pair inside, tiny sigma                         PHA                LINEAR -> PHA
                                                     P_lin=1.0000 s=0.0000 lin_ok=True
open screen, exact inside, tiny sigma                PHA                EXACT -> LINEAR -> PHA
                                                     moid=0.03230 P_lin=1.0000 s=0.0000 lin_ok=True
open screen, exact outside, tiny sigma               NOT_PHA_GEOMETRY   EXACT -> LINEAR -> NOT_PHA_GEOMETRY
                                                     moid=0.12516 P_lin=0.0000 s=0.0000 lin_ok=True
exact inside, near threshold, s = 0.002              OBSERVE_PHA        EXACT -> LINEAR -> OBSERVE_PHA
                                                     moid=0.04664 P_lin=0.9537 s=0.0020 lin_ok=True
exact outside, near threshold, s = 0.002             OBSERVE_NOT        EXACT -> LINEAR -> OBSERVE_NOT
                                                     moid=0.05312 P_lin=0.0594 s=0.0020 lin_ok=True
exact inside, s = 0.03 > s_lin: sampled              OBSERVE_PHA        EXACT -> LINEAR -> SAMPLE -> OBSERVE_PHA
                                                     moid=0.03892 P_lin=0.6426 s=0.0300 lin_ok=False P_mc=0.6072 [0.5920, 0.6223]
node pair inside, s = 0.004                          PHA                LINEAR -> PHA
                                                     P_lin=1.0000 s=0.0043 lin_ok=True
node pair just inside, s = 0.004: P needs exact      OBSERVE_PHA        LINEAR -> EXACT_FOR_P -> LINEAR -> OBSERVE_PHA
                                                     moid=0.04653 P_lin=0.8072 s=0.0040 lin_ok=True
radial gap 0.052 au, s ~ 0.002: P needs exact        NOT_PHA_GEOMETRY   LINEAR -> EXACT_FOR_P -> LINEAR -> NOT_PHA_GEOMETRY
                                                     moid=0.06014 P_lin=0.0000 s=0.0020 lin_ok=True
non-finite sigma                                     HUMAN              LINEAR -> HUMAN
----------------------------------------------------------------------------------------------------
acts reached: 10 of 10
ALL CASES PASS
```

**PASS**

## 4. Appendix A statements on the real catalogue

```
L1_radial_gap_violations: 0
L2_node_gap_violations: 0
L1_L2_n: 42888
L3_n: 37214
L3_lower_violations: 0
L3_upper_violations: 0
L3_width_max_for_delta_le_0.05_au: 1.9519244829881155e-05
L3_width_median_for_delta_le_0.05_au: 7.700360201146061e-07
P1_synthetic_orbits: 198
P1_samples_per_orbit: 40000
P1_min_of_closest_approach_minus_moid_au: 8.590233397609559e-06
P1_violations: 0
P1_jpl_approaches: 306590
P1_jpl_share_dist_ge_moid_minus_1e-4: 0.9875664568315992
P1_jpl_share_by_era: {'1910-1969': 0.9871048841604022, '1970-1999': 0.9923239197005784, '2000-2024': 0.9849635372415746}
P0_cad_predicted_seen_share_5fold: 0.9568679454632876
P0_cad_observed_seen_share_5fold: 0.9568968329038782
P0_kaggle_predicted_seen_share_10fold: 0.3476499607424792
P0_kaggle_observed_seen_share_10fold: 0.34883382234128507
P2_mismatches: 0
T1_PHA_with_H_gt_22: 0
T1_PHA_with_delta_gt_0.05: 0
T1_NOT_PHA_GEOMETRY_with_delta_le_0.05: 0
T1_NOT_PHA_SIZE_with_H_le_22: 0
T1_max_steps: 6
T1_decided: 42700
```

**PASS**

## 5. Paper numbers recomputed from results/

```
100 of 100 paper numbers recomputed from results/ and found in the LaTeX source
  timing, ms per asteroid: paper 0.66, this run 0.66 (machine-dependent; within 10x: True)
```

**PASS**

## 6. Video narration numbers recomputed from results/

```
all 27 narration numbers match results/
```

**PASS**

## 7. Data checksums (SHA-256)

```
8bb791a85ece76c2b43d347f110742b90abcd0baa36894d77d9aea74d6cba4dc  data/processed/sbdb_neo.csv
246b3d91a2c90a87968c1f9eb405f1f144a670c85041a3755d55263459f1c9fe  data/processed/cad_earth.csv
55db00905e91f0019a418eb24c8ac0045df88f2224d80e0354edf424cf3518f9  data/kaggle/nasa-asteroids-classification/nasa.csv
```

**PASS**

## 8. Received score sheet and delivered NASA files (SHA-256 verified)

```
cad.csv: OK
scout.csv: OK
sentry_risk_list.csv: OK
NASA_Asteroid_Final_Comparative_Results.xlsx: OK
```

**PASS**

## 9. Score-sheet comparison recomputed independently

```

-- 1. received sheet: test sets and confusion counts by exact fractions
PASS  NASA CAD: exactly one positives count fits all three models  [n=639, P=[328]]
PASS  NASA CAD CNN: counts TP=223 FP=71 FN=105 TN=240 match the JSON
PASS  NASA CAD CNN: ROC AUC x P x N is a multiple of 1/2  [2U=163276.000000]
PASS  NASA CAD GPBoost: counts TP=233 FP=82 FN=95 TN=229 match the JSON
PASS  NASA CAD GPBoost: ROC AUC x P x N is a multiple of 1/2  [2U=162984.000000]
PASS  NASA CAD Tuned Random Forest: counts TP=230 FP=65 FN=98 TN=246 match the JSON
PASS  NASA CAD Tuned Random Forest: ROC AUC x P x N is a multiple of 1/2  [2U=163030.000000]
PASS  NASA CAD: test set 639 rows vs 33 rows in the delivered file (JSON agrees)
PASS  NASA SCOUT: exactly one positives count fits all three models  [n=9, P=[3]]
PASS  NASA SCOUT CNN: counts TP=0 FP=0 FN=3 TN=6 match the JSON
PASS  NASA SCOUT CNN: ROC AUC x P x N is a multiple of 1/2  [2U=20.000000]
PASS  NASA SCOUT GPBoost: counts TP=0 FP=2 FN=3 TN=4 match the JSON
PASS  NASA SCOUT GPBoost: ROC AUC x P x N is a multiple of 1/2  [2U=18.000000]
PASS  NASA SCOUT Tuned Random Forest: counts TP=3 FP=5 FN=0 TN=1 match the JSON
PASS  NASA SCOUT Tuned Random Forest: ROC AUC x P x N is a multiple of 1/2  [2U=31.000000]
PASS  NASA SCOUT: test set 9 rows vs 72 rows in the delivered file (JSON agrees)
PASS  NASA SENTRY: exactly one positives count fits all three models  [n=394, P=[201]]
PASS  NASA SENTRY CNN: counts TP=139 FP=44 FN=62 TN=149 match the JSON
PASS  NASA SENTRY CNN: ROC AUC x P x N is a multiple of 1/2  [2U=62568.000000]
PASS  NASA SENTRY GPBoost: counts TP=142 FP=39 FN=59 TN=154 match the JSON
PASS  NASA SENTRY GPBoost: ROC AUC x P x N is a multiple of 1/2  [2U=62720.000000]
PASS  NASA SENTRY Tuned Random Forest: counts TP=133 FP=43 FN=68 TN=150 match the JSON
PASS  NASA SENTRY Tuned Random Forest: ROC AUC x P x N is a multiple of 1/2  [2U=61588.000000]
PASS  NASA SENTRY: test set 394 rows vs 2242 rows in the delivered file (JSON agrees)
PASS  NASA CAD: the received test set is larger than the delivered cad.csv, so it cannot come from it  [639 > 33]
PASS  NASA SCOUT: the CNN predicted no positive (its 6/9 equals always answering no)
PASS  NASA SCOUT: GPBoost and the tuned RF score below always answering no

-- 2. delivered files: PHA labels and our rule from a fresh path-to-path distance
PASS  cad.csv: 33 rows, all in JPL SBDB; 33 have an H and are scored (JSON agrees); those without H are not PHAs  [no H: []]
PASS  cad.csv: 2 PHAs among the scored objects (JSON agrees)
PASS  cad.csv: our rule TP=2 FP=0 FN=0 TN=31; JSON metrics agree  [accuracy=1.0000, f1=1.0000]
PASS  sentry_risk_list.csv: 2242 rows, all in JPL SBDB; 2241 have an H and are scored (JSON agrees); those without H are not PHAs  [no H: ['2010 MY112']]
PASS  sentry_risk_list.csv: 15 PHAs among the scored objects (JSON agrees)
PASS  sentry_risk_list.csv: our rule TP=15 FP=0 FN=0 TN=2226; JSON metrics agree  [accuracy=1.0000, f1=1.0000]
PASS  sentry_risk_list.csv: 'H <= 22' alone would make 6 errors; the path geometry removes them
PASS  scout.csv: no Scout candidate is in JPL SBDB (no ground truth)
PASS  scout.csv: our rule on Scout's own MOID and H flags 1 candidate(s) (JSON agrees)

-- 3. every summary equals the mean of its per-fold rows
PASS  Sentry: 5 repeats x 4 methods of per-fold rows
PASS  Sentry: every summary mean equals the mean of its repeats  [max diff 1.1e-16]
PASS  full CAD: 2 protocols x 5 folds x 4 methods of per-fold rows
PASS  full CAD: every summary mean equals the mean of its folds  [max diff 2.2e-16]
PASS  full CAD: our rule's grouped-fold rows equal the paper's same-fold comparison  [max diff 0.0e+00]

-- 4. workbook: values, formulas and the as-received tab
PASS  Master Results holds exactly the JSON numbers
PASS  per-metric tabs: every cell is a formula whose value equals Master Results
PASS  Best Models (highest F1) and Dataset Summary equal what their formulas must give
PASS  as-received tab equals the received sheet's Master Results

ALL PASS
```

**PASS**

## 10. Kaggle notebook reproduces the results

```
unpacked /Users/kanchetidevieswar/near earth objects (neo)/data/kaggle/nasa-asteroids-classification/nasa-asteroids-classification.zip
sbdb_neo.csv           /Users/kanchetidevieswar/near earth objects (neo)/data/processed/sbdb_neo.csv
cad_earth.csv          /Users/kanchetidevieswar/near earth objects (neo)/data/processed/cad_earth.csv
cad.csv                /Users/kanchetidevieswar/near earth objects (neo)/data/external/cad.csv
sentry_risk_list.csv   /Users/kanchetidevieswar/near earth objects (neo)/data/external/sentry_risk_list.csv
scout.csv              /Users/kanchetidevieswar/near earth objects (neo)/data/external/scout.csv
42,888 near-Earth asteroids (2,551 PHAs): path-to-path distance for all of them in 14.1 s   [paper: 42,888 asteroids, 2,551 PHAs]
|delta - JPL MOID|: median 1.03e-04 au, 99th percentile 6.03e-04 au, max 1.83e-03 au   [paper: 1.03e-04, 6.03e-04, 1.83e-03]
same PHA verdict from our delta as from JPL's MOID: 99.979%   [paper: 99.979%]
paper: ours accuracy 0.99865, F1 0.98859, MCC 0.98788; tangent line 0.99263 / 0.93811

asteroids first seen in 2016 or later: 27,817 (682 PHAs)   [paper: 27,817 (682)]
paper: ours accuracy 0.99942 (27,801 of 27,817), F1 0.98825
306,590 close approaches of 36,817 asteroids, 26,784 PHA rows   [paper: 306,590 approaches of 36,817 asteroids]
row split (prior work)    : 95.7% of test rows belong to an asteroid that is also in training
asteroid-grouped (honest) : 0.0% of test rows belong to an asteroid that is also in training
paper, row split: random forest 0.9474 / F1 0.676 (inflated by leakage)
paper, grouped: no baseline above 0.9263 accuracy (MLP) or F1 0.428 (random forest);
               ours 0.9979 / F1 0.9878 on the same folds
CAD table: 33 rows, 33 scored (an orbit and H in JPL SBDB), 2 PHAs -> TP 2, FP 0, FN 0, TN 31; accuracy 1.0000, F1 1.0000
Sentry risk list: 2,242 rows, 2,241 scored (an orbit and H in JPL SBDB), 15 PHAs -> TP 15, FP 0, FN 0, TN 2226; accuracy 1.0000, F1 1.0000
Scout: 72 unconfirmed candidates, 0 of them in JPL SBDB (no ground truth). The rule on Scout's own MOID and H flags 1:
11,054 well-determined asteroids (2,039 hazardous), 48 clones each, 107 s
paper (short-arc profiles, scale 1): point rule 0.9846 / MCC 0.9489 / Brier 0.0154; path cloud 0.9846 / 0.9487 / 0.0105

-- comparison with results/
PASS  same 42,888 asteroids in the same order as the pipeline
PASS  notebook delta equals the pipeline's moid_path for every asteroid  [max diff 0.0e+00]
PASS  notebook d_lin equals the pipeline's d_lin for every asteroid  [max diff 0.0e+00]
PASS  notebook d_node equals the pipeline's d_node for every asteroid  [max diff 0.0e+00]
PASS  5-fold rule results (4 rules, mean and std) equal results/asteroid_view_summary.csv  [max diff 0.0e+00]
PASS  temporal rule results equal results/asteroid_view_temporal.csv  [max diff 0.0e+00]
PASS  rule on the paper's grouped CAD folds equals results/same_folds_rows_summary.csv  [max diff 1.1e-16]
PASS  rule on the delivered CAD and Sentry files equals results/sheet_comparison.json
PASS  uncertainty simulation (short-arc profiles, scale 1) equals results/uncertainty_cc8-9_summary.csv  [max diff 0.0e+00]

ALL PASS
```

**PASS**
