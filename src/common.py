"""Shared helpers: paths, constants, data loading and metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics as skm

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = ROOT / "paper" / "figures"

AU_KM = 149_597_870.7
SEED = 20261111
N_FOLDS = 5

# Official PHA definition (JPL/CNEOS): Earth MOID <= 0.05 au and H <= 22.0.
MOID_LIMIT_AU = 0.05
H_LIMIT = 22.0

NUMERIC_SBDB = [
    "H", "diameter", "diameter_sigma", "albedo", "epoch", "e", "a", "q", "i", "om", "w", "ma",
    "ad", "n", "tp", "per", "moid", "moid_jup", "t_jup", "condition_code", "data_arc",
    "n_obs_used", "rms", "sigma_e", "sigma_a", "sigma_q", "sigma_i", "sigma_om", "sigma_w",
    "sigma_ma", "sigma_tp", "sigma_n", "sigma_per", "sigma_ad", "A1", "A2", "A3",
]


def load_sbdb(labelled_only: bool = True) -> pd.DataFrame:
    """One row per near-Earth asteroid, with numeric columns parsed and a 0/1 `label` (JPL PHA flag)."""
    df = pd.read_csv(PROC / "sbdb_neo.csv", low_memory=False)
    for c in NUMERIC_SBDB:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["label"] = df["pha"].map({"Y": 1, "N": 0})
    df["first_obs"] = pd.to_datetime(df["first_obs"], errors="coerce")
    if labelled_only:
        df = df[df["label"].notna() & df["H"].notna()].copy()
        df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def neows_diameters_km(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NeoWs convention: diameter range from H for geometric albedo 0.25 (min) to 0.05 (max)."""
    base = 10.0 ** (-np.asarray(H, dtype=float) / 5.0)
    return 1329.0 / np.sqrt(0.25) * base, 1329.0 / np.sqrt(0.05) * base


def binary_metrics(y_true, y_pred, y_score=None) -> dict:
    out = {
        "accuracy": skm.accuracy_score(y_true, y_pred),
        "balanced_accuracy": skm.balanced_accuracy_score(y_true, y_pred),
        "precision": skm.precision_score(y_true, y_pred, zero_division=0),
        "recall": skm.recall_score(y_true, y_pred, zero_division=0),
        "f1": skm.f1_score(y_true, y_pred, zero_division=0),
        "mcc": skm.matthews_corrcoef(y_true, y_pred),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = skm.roc_auc_score(y_true, y_score)
        out["pr_auc"] = skm.average_precision_score(y_true, y_score)
    return out


def summarize_folds(folds: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean and std of every metric column across folds."""
    metric_cols = [c for c in folds.columns if c not in by + ["fold"] and folds[c].dtype.kind in "fc"]
    agg = folds.groupby(by, sort=False)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
