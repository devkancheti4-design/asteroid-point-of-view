"""Earth's-view benchmark: reproduce the prior-work setup on JPL close-approach data.

Data: one row per Earth close approach (<= 0.5 au, 1910-2024) with the NeoWs/Kaggle feature set
(absolute magnitude, estimated diameter range, relative velocity, miss distance). Label: JPL PHA flag.

Protocols
  row_split         stratified K-fold over rows, as in prior work. The same asteroid's other
                    approaches can sit in the training folds (identity leakage).
  asteroid_grouped  stratified group K-fold: all approaches of one asteroid share a fold.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common import AU_KM, N_FOLDS, PROC, RESULTS, SEED, binary_metrics, load_sbdb, neows_diameters_km, save_json, summarize_folds

FEATURES = ["absolute_magnitude", "est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance"]


def build_dataset() -> pd.DataFrame:
    sbdb = load_sbdb()
    cad = pd.read_csv(PROC / "cad_earth.csv", low_memory=False)
    df = cad.merge(sbdb[["pdes", "label", "H"]], left_on="des", right_on="pdes", how="inner")
    df["absolute_magnitude"] = df["H"]
    dmin, dmax = neows_diameters_km(df["H"].to_numpy())
    df["est_diameter_min"], df["est_diameter_max"] = dmin, dmax
    df["relative_velocity"] = pd.to_numeric(df["v_rel"]) * 3600.0  # km/h, as in NeoWs
    df["miss_distance"] = pd.to_numeric(df["dist"]) * AU_KM  # km
    return df.reset_index(drop=True)


def make_models() -> dict:
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
            n_jobs=-1, random_state=SEED, eval_metric="logloss",
        ),
        "LightGBM": LGBMClassifier(n_estimators=400, num_leaves=63, learning_rate=0.05, random_state=SEED, verbose=-1),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=1024, early_stopping=True,
                          n_iter_no_change=5, max_iter=60, random_state=SEED),
        ),
    }


def main() -> None:
    df = build_dataset()
    X = df[FEATURES].to_numpy(float)
    y = df["label"].to_numpy(int)
    groups = df["des"].to_numpy()
    stats = {
        "rows": int(len(df)),
        "asteroids": int(df["des"].nunique()),
        "positive_rows": int(y.sum()),
        "positive_rate_rows": float(y.mean()),
        "positive_asteroids": int(df.groupby("des")["label"].first().sum()),
        "approaches_per_asteroid_mean": float(df.groupby("des").size().mean()),
        "approaches_per_asteroid_median": float(df.groupby("des").size().median()),
    }
    print(stats)

    splits = {
        "row_split": list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y)),
        "asteroid_grouped": list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y, groups)),
    }
    leakage = {}
    for proto, folds in splits.items():
        seen = [np.isin(groups[te], np.unique(groups[tr])).mean() for tr, te in folds]
        leakage[proto] = float(np.mean(seen))
        print(f"{proto}: share of test rows whose asteroid also appears in training = {leakage[proto]:.3f}")
    stats["test_rows_with_asteroid_in_train"] = leakage

    rows = []
    for proto, folds in splits.items():
        for k, (tr, te) in enumerate(folds):
            for name, model in make_models().items():
                t0 = time.time()
                model.fit(X[tr], y[tr])
                score = model.predict_proba(X[te])[:, 1]
                pred = (score >= 0.5).astype(int)
                m = binary_metrics(y[te], pred, score)
                rows.append({"protocol": proto, "model": name, "fold": k, **m, "fit_seconds": time.time() - t0})
                print(f"{proto:17s} fold {k} {name:12s} acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
                      f"rec={m['recall']:.4f} auc={m['roc_auc']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    folds_df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    folds_df.to_csv(RESULTS / "earth_view_folds.csv", index=False)
    summary = summarize_folds(folds_df, ["protocol", "model"])
    summary.to_csv(RESULTS / "earth_view_summary.csv", index=False)
    save_json(stats, RESULTS / "earth_view_stats.json")
    cols = ["protocol", "model", "accuracy_mean", "f1_mean", "recall_mean", "precision_mean", "mcc_mean", "roc_auc_mean"]
    print(summary[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
