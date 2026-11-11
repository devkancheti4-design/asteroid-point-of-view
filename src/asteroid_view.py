"""Asteroid's-view benchmark: hazard from the asteroid's own path and size.

Inputs are the asteroid's osculating elements and H, plus Earth's ORBIT as a fixed curve. No close-approach
data, no Earth position at any time, and never JPL's `moid` or `pha` fields (except where a row is explicitly
labelled as a prior-work reproduction).

Outputs (results/):
  geometry_validation.json      our path-to-path distance vs JPL MOID
  asteroid_view_folds.csv/...   stratified 5-fold over asteroids (one row per asteroid -> no identity leakage)
  asteroid_view_temporal.csv    train on asteroids first seen before the cutoff, test on later discoveries
  same_folds_rows.csv           asteroid-view models on the exact Earth-view grouped folds, scored per approach row
  asteroid_features.csv         per-asteroid geometry features (reused by the uncertainty experiment)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common import H_LIMIT, MOID_LIMIT_AU, N_FOLDS, RESULTS, SEED, binary_metrics, load_sbdb, save_json, summarize_folds
from earth_view import build_dataset as build_earth_view
from orbit_geometry import moid_earth, node_geometry, opik_variables

TEMPORAL_CUTOFF = "2016-01-01"

RAW_PATH = ["a", "e", "i", "sin_om", "cos_om", "sin_w", "cos_w", "q", "ad", "H"]
KERNELS = ["r_asc", "r_desc", "dr_asc", "dr_desc", "d_node", "d_lin_asc", "d_lin_desc", "d_lin",
           "T_earth", "U", "cos_theta"]
FEATURE_SETS = {
    "path": RAW_PATH,                                   # raw path + size only
    "path+kernels": RAW_PATH + KERNELS,                 # + closed-form geometry kernels
    "path+kernels+exact": RAW_PATH + KERNELS + ["moid_path"],  # + exact path-to-path distance (ours)
}
PRIOR_WORK_ORBIT = ["a", "e", "i", "om", "w", "q", "ad", "per", "n", "ma", "H", "moid"]  # Kaggle-style, uses JPL moid


def add_geometry(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t0 = time.time()
    df["moid_path"] = moid_earth(df.a.values, df.e.values, df.i.values, df.om.values, df.w.values)
    print(f"exact path-to-path distance for {len(df):,} asteroids in {time.time() - t0:.1f}s")
    for k, v in node_geometry(df.a.values, df.e.values, df.i.values, df.om.values, df.w.values).items():
        df[k] = v
    for k, v in opik_variables(df.a.values, df.e.values, df.i.values).items():
        df[k] = v
    df["cos_theta"] = df["cos_theta"].fillna(0.0)
    for ang in ("om", "w"):
        rad = np.deg2rad(df[ang].values)
        df[f"sin_{ang}"], df[f"cos_{ang}"] = np.sin(rad), np.cos(rad)
    return df


def hazard_index(distance_au: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Dimensionless path-size hazard index: max(distance / 0.05 au, D_22 / D). PHA iff index <= 1.

    D_22 / D = 10**((H - 22) / 5) is the size ratio to an H = 22 body of the same albedo, so both terms are
    pure ratios and only the path geometry and size enter.
    """
    return np.maximum(np.asarray(distance_au) / MOID_LIMIT_AU, 10.0 ** ((np.asarray(H) - H_LIMIT) / 5.0))


RULES = {  # name -> distance column; all training-free
    "rule: exact path geometry (ours)": "moid_path",
    "rule: tangent-line closed form": "d_lin",
    "rule: node gap (upper bound)": "d_node",
    "rule: JPL MOID (definition, reference)": "moid",
}


def rule_scores(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    hi = hazard_index(df[col].values, df["H"].values)
    return (hi <= 1.0).astype(int), -np.log(hi)


def make_models() -> dict:
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.9, n_jobs=-1, random_state=SEED, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05, random_state=SEED, verbose=-1),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True,
                                                             max_iter=300, random_state=SEED)),
    }


def evaluate(train: pd.DataFrame, test: pd.DataFrame, tag: dict, rows: list, models: dict | None = None,
             feature_sets: dict | None = None, include_prior_work: bool = True) -> None:
    for rname, col in RULES.items():
        pred, score = rule_scores(test, col)
        rows.append({**tag, "method": rname, "features": col, **binary_metrics(test["label"], pred, score)})
    feature_sets = FEATURE_SETS if feature_sets is None else feature_sets
    sets = dict(feature_sets)
    if include_prior_work:
        sets["prior-work orbit+JPL moid"] = PRIOR_WORK_ORBIT
    for fs_name, cols in sets.items():
        for mname, model in (models or make_models()).items():
            model.fit(train[cols].values, train["label"].values)
            score = model.predict_proba(test[cols].values)[:, 1]
            m = binary_metrics(test["label"].values, (score >= 0.5).astype(int), score)
            rows.append({**tag, "method": mname, "features": fs_name, **m})


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    sbdb = load_sbdb()
    sbdb = sbdb[sbdb["moid"].notna()].reset_index(drop=True)
    df = add_geometry(sbdb)
    df.to_csv(RESULTS / "asteroid_features.csv", index=False)

    # 1) geometry validation against JPL
    diff = df["moid_path"] - df["moid"]
    lab_rule_jpl = ((df["moid"] <= MOID_LIMIT_AU) & (df["H"] <= H_LIMIT)).astype(int)
    lab_rule_ours = ((df["moid_path"] <= MOID_LIMIT_AU) & (df["H"] <= H_LIMIT)).astype(int)
    val = {
        "n_asteroids": int(len(df)),
        "n_pha": int(df["label"].sum()),
        "abs_diff_au": {q: float(np.quantile(np.abs(diff), p)) for q, p in
                        [("median", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1.0)]},
        "share_within_1e-3_au": float((np.abs(diff) < 1e-3).mean()),
        "pearson_r": float(np.corrcoef(df["moid_path"], df["moid"])[0, 1]),
        "rule_label_agreement_ours_vs_jplmoid": float((lab_rule_jpl == lab_rule_ours).mean()),
        "jpl_flag_vs_rule_on_jpl_moid_disagreements": int((lab_rule_jpl != df["label"]).sum()),
        "d_lin_abs_err_vs_moid_path_for_moid_below_0.1": {
            q: float(np.quantile(np.abs(df.loc[df.moid_path < 0.1, "d_lin"] - df.loc[df.moid_path < 0.1, "moid_path"]), p))
            for q, p in [("median", 0.5), ("p90", 0.9), ("p99", 0.99)]},
    }
    save_json(val, RESULTS / "geometry_validation.json")
    print(val)

    # 2) stratified K-fold over asteroids
    rows = []
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(skf.split(df, df["label"])):
        t0 = time.time()
        evaluate(df.iloc[tr], df.iloc[te], {"protocol": "asteroid_kfold", "fold": k}, rows)
        print(f"asteroid k-fold {k} done in {time.time() - t0:.0f}s", flush=True)
    folds = pd.DataFrame(rows)
    folds.to_csv(RESULTS / "asteroid_view_folds.csv", index=False)
    summary = summarize_folds(folds, ["protocol", "method", "features"])
    summary.to_csv(RESULTS / "asteroid_view_summary.csv", index=False)
    print(summary[["method", "features", "accuracy_mean", "f1_mean", "recall_mean", "precision_mean", "mcc_mean"]]
          .round(5).to_string(index=False))

    # 3) temporal split: learn from older discoveries, predict newer ones
    rows = []
    old = df[df["first_obs"] < TEMPORAL_CUTOFF]
    new = df[df["first_obs"] >= TEMPORAL_CUTOFF]
    print(f"temporal split: train {len(old):,} (PHA {old.label.sum()}), test {len(new):,} (PHA {new.label.sum()})")
    evaluate(old, new, {"protocol": f"temporal<{TEMPORAL_CUTOFF}", "fold": 0}, rows)
    temporal = pd.DataFrame(rows)
    temporal["n_train"], temporal["n_test"] = len(old), len(new)
    temporal.to_csv(RESULTS / "asteroid_view_temporal.csv", index=False)
    print(temporal[["method", "features", "accuracy", "f1", "recall", "precision", "mcc"]].round(5).to_string(index=False))

    # 4) identical grouped folds as the Earth-view benchmark, scored per close-approach row
    ev = build_earth_view()
    X_dummy = np.zeros(len(ev))
    groups = ev["des"].to_numpy()
    splits = list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X_dummy, ev["label"], groups))
    feat = df.set_index("pdes")
    rows = []
    for k, (tr, te) in enumerate(splits):
        tr_ids = pd.unique(groups[tr])
        te_rows = ev.iloc[te]
        tr_ids = [d for d in tr_ids if d in feat.index]
        train = feat.loc[tr_ids]
        test_rows = te_rows[te_rows["des"].isin(feat.index)]
        test_feat = feat.loc[test_rows["des"].values]
        y_rows = test_rows["label"].values
        for rname, col in RULES.items():
            pred, score = rule_scores(test_feat, col)
            rows.append({"protocol": "earthview_grouped_rows", "fold": k, "method": rname, "features": col,
                         **binary_metrics(y_rows, pred, score)})
        for fs_name in ("path", "path+kernels"):
            cols = FEATURE_SETS[fs_name]
            for mname in ("XGBoost", "LightGBM", "RandomForest"):
                model = make_models()[mname]
                model.fit(train[cols].values, train["label"].values)
                score = model.predict_proba(test_feat[cols].values)[:, 1]
                rows.append({"protocol": "earthview_grouped_rows", "fold": k, "method": mname, "features": fs_name,
                             **binary_metrics(y_rows, (score >= 0.5).astype(int), score)})
        print(f"same-fold comparison fold {k}: {len(test_rows):,} test rows", flush=True)
    same = pd.DataFrame(rows)
    same.to_csv(RESULTS / "same_folds_rows.csv", index=False)
    same_sum = summarize_folds(same, ["protocol", "method", "features"])
    same_sum.to_csv(RESULTS / "same_folds_rows_summary.csv", index=False)
    print(same_sum[["method", "features", "accuracy_mean", "f1_mean", "recall_mean", "mcc_mean"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
