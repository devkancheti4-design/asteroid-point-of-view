"""Same-file comparison on the Kaggle "NASA: Asteroids Classification" table (shrutimehta, nasa.csv),
the dataset behind Chandra (2025, arXiv:2508.15106: random forest, 99.7% accuracy, F2 0.987 under repeated
stratified 10-fold cross-validation over records) and Malakouti et al. (2023, MethodsX).

Usage:  python -I src/kaggle_compare.py data/kaggle/nasa-asteroids-classification/nasa.csv

  A. Prior-work protocol: random forest, all numeric columns (incl. the table's MOID), repeated stratified
     10-fold x 3 over records. Plus the share of test records whose asteroid also sits in training.
  B. The same model with folds grouped by asteroid (Neo Reference ID).
  C. Ours, training-free: exact path-to-path distance from the table's own orbital elements (not its MOID
     column) and the PHA definition; scored on every record.
  D. Ours, learned: LightGBM on path + kernels, grouped folds.
  E. Reference: the definition applied to the table's own MOID column (the table's ceiling).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # explicit: our own src/ only

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import fbeta_score  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold  # noqa: E402

from common import RESULTS, SEED, binary_metrics, save_json  # noqa: E402
from orbit_geometry import moid_earth, node_geometry, opik_variables  # noqa: E402

N_SPLITS, N_REPEATS = 10, 3


def metrics(y, pred, score=None) -> dict:
    m = binary_metrics(y, pred, score)
    m["f2"] = fbeta_score(y, pred, beta=2, zero_division=0)
    return m


def mean_of(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    return {c: float(df[c].mean()) for c in df.columns if df[c].dtype.kind in "fc"} | \
           {f"{c}_std": float(df[c].std()) for c in ("accuracy", "f1", "f2", "mcc") if c in df}


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    y = df["Hazardous"].astype(str).str.lower().eq("true").astype(int).to_numpy()
    ids = df["Neo Reference ID"].to_numpy()
    info = {"rows": int(len(df)), "asteroids": int(pd.unique(ids).size), "hazardous_rows": int(y.sum()),
            "hazardous_share": float(y.mean()), "columns": int(df.shape[1])}
    lab_per_id = pd.Series(y).groupby(ids).nunique()
    info["asteroids_with_conflicting_labels"] = int((lab_per_id > 1).sum())
    print(info)

    # numeric feature table as prior work would use it (identifiers and the label dropped)
    num = df.select_dtypes(include=[np.number]).drop(columns=["Neo Reference ID", "Name"], errors="ignore")
    X_all = num.to_numpy(float)
    feats_prior = list(num.columns)

    # A. prior-work protocol: record-level repeated stratified 10-fold x 3
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    rows_a, seen = [], []
    for tr, te in rskf.split(X_all, y):
        rf = RandomForestClassifier(random_state=SEED, n_jobs=-1).fit(X_all[tr], y[tr])
        rows_a.append(metrics(y[te], rf.predict(X_all[te]), rf.predict_proba(X_all[te])[:, 1]))
        seen.append(np.isin(ids[te], ids[tr]).mean())
    res = {"info": info, "features_prior": feats_prior,
           "A_prior_protocol_rf": mean_of(rows_a), "A_test_records_with_asteroid_in_train": float(np.mean(seen))}

    # B. the same random forest, folds grouped by asteroid (3 repeats with different shuffles)
    rows_b = []
    for rep in range(N_REPEATS):
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + rep)
        for tr, te in sgkf.split(X_all, y, ids):
            rf = RandomForestClassifier(random_state=SEED, n_jobs=-1).fit(X_all[tr], y[tr])
            rows_b.append(metrics(y[te], rf.predict(X_all[te]), rf.predict_proba(X_all[te])[:, 1]))
    res["B_grouped_rf"] = mean_of(rows_b)

    # C. ours, training-free, from the table's own elements (never its MOID column)
    a, e, i = df["Semi Major Axis"].to_numpy(float), df["Eccentricity"].to_numpy(float), df["Inclination"].to_numpy(float)
    om, w, H = df["Asc Node Longitude"].to_numpy(float), df["Perihelion Arg"].to_numpy(float), df["Absolute Magnitude"].to_numpy(float)
    delta = moid_earth(a, e, i, om, w)
    rule = ((delta <= 0.05) & (H <= 22.0)).astype(int)
    hi = np.maximum(delta / 0.05, 10.0 ** ((H - 22.0) / 5.0))
    res["C_ours_rule_all_records"] = metrics(y, rule, -np.log(hi))
    first = ~pd.Series(ids).duplicated().to_numpy()
    res["C_ours_rule_unique_asteroids"] = metrics(y[first], rule[first], -np.log(hi[first]))
    moid_tab = df["Minimum Orbit Intersection"].to_numpy(float)
    res["moid_vs_table"] = {"median_abs_diff_au": float(np.median(np.abs(delta - moid_tab))),
                            "p99_abs_diff_au": float(np.quantile(np.abs(delta - moid_tab), 0.99)),
                            "max_abs_diff_au": float(np.max(np.abs(delta - moid_tab)))}

    # E. reference: the definition on the table's own MOID column
    ref = ((moid_tab <= 0.05) & (H <= 22.0)).astype(int)
    res["E_definition_on_table_moid"] = metrics(y, ref)
    wrong = rule != y
    res["C_errors"] = {"records": int(wrong.sum()),
                       "of_which_table_label_contradicts_its_own_definition": int((wrong & (ref != y)).sum()),
                       "of_which_moids_straddle_0.05": int((wrong & ((delta <= 0.05) != (moid_tab <= 0.05))).sum())}

    # D. ours, learned: LightGBM on path + kernels, grouped folds
    F = pd.DataFrame({"a": a, "e": e, "i": i, "q": a * (1 - e), "ad": a * (1 + e), "H": H,
                      "sin_om": np.sin(np.deg2rad(om)), "cos_om": np.cos(np.deg2rad(om)),
                      "sin_w": np.sin(np.deg2rad(w)), "cos_w": np.cos(np.deg2rad(w))})
    for k, v in {**node_geometry(a, e, i, om, w), **opik_variables(a, e, i)}.items():
        F[k] = v
    F["cos_theta"] = F["cos_theta"].fillna(0.0)
    XF = F.to_numpy(float)
    rows_d = []
    for rep in range(N_REPEATS):
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + rep)
        for tr, te in sgkf.split(XF, y, ids):
            m = LGBMClassifier(n_estimators=600, num_leaves=31, learning_rate=0.05, random_state=SEED, verbose=-1)
            m.fit(XF[tr], y[tr])
            p = m.predict_proba(XF[te])[:, 1]
            rows_d.append(metrics(y[te], (p >= 0.5).astype(int), p))
    res["D_ours_lightgbm_grouped"] = mean_of(rows_d)

    # F. mechanism: does the record-level split let the forest memorise labels that contradict the definition?
    contra = ref != y

    def memorisation(splits) -> dict:
        out = dict(matched=0, n=0, matched_seen=0, n_seen=0, matched_unseen=0, n_unseen=0, err_consistent=0, n_consistent=0)
        for tr, te in splits:
            p = RandomForestClassifier(random_state=SEED, n_jobs=-1).fit(X_all[tr], y[tr]).predict(X_all[te])
            ok, c, seen_ = p == y[te], contra[te], np.isin(ids[te], ids[tr])
            out["matched"] += int(ok[c].sum()); out["n"] += int(c.sum())
            out["matched_seen"] += int(ok[c & seen_].sum()); out["n_seen"] += int((c & seen_).sum())
            out["matched_unseen"] += int(ok[c & ~seen_].sum()); out["n_unseen"] += int((c & ~seen_).sum())
            out["err_consistent"] += int((~ok[~c]).sum()); out["n_consistent"] += int((~c).sum())
        return out

    res["F_memorisation_record_split"] = memorisation(
        RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(X_all, y))
    res["F_memorisation_grouped_split"] = memorisation(
        [s for rep in range(N_REPEATS)
         for s in StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED + rep).split(X_all, y, ids)])
    counts = pd.Series(ids).value_counts()
    res["info"]["asteroids_with_more_than_one_record"] = int((counts > 1).sum())
    res["info"]["label_contradictions_records"] = int(contra.sum())
    res["info"]["label_contradictions_asteroids"] = int(pd.unique(ids[contra]).size)
    res["info"]["label_contradictions_H_range"] = [float(H[contra].min()), float(H[contra].max())]

    save_json(res, RESULTS / "kaggle_nasa_comparison.json")
    show = {k: v for k, v in res.items() if k != "features_prior"}
    import json
    print(json.dumps(show, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
