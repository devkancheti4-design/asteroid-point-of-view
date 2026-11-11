"""The comparison sheet's three models (CNN, GPBoost, tuned random forest) against the asteroid's-view rule.

The sheet data/external/NASA_Asteroid_Final_Comparative_Results.xlsx lists accuracy, precision, recall, F1 and
ROC AUC for a CNN, GPBoost and a tuned random forest on "NASA CAD", "NASA SCOUT" and "NASA SENTRY". It does not
say which label the models predicted, how the data were split, or which rows were tested.

  A  decodes every row of the sheet into its exact confusion counts (test size, positives, TP, FP, FN, TN) and
     checks the counts against the reported ROC AUC;
  B  reruns the three model families and our training-free path rule on the three files as delivered
     (data/external/), with one stated label for every method (JPL's PHA flag, SBDB snapshot of 2026-11-11)
     and identical folds for every method;
  C  adds the three model families to the full NASA CAD benchmark (every Earth close approach within 0.5 au,
     1910-2024; the same row-split and asteroid-grouped folds as earth_view.py) and scores the rule on the
     identical folds.

The CNN trains in a separate process (cnn_worker.py): PyTorch's OpenMP runtime crashes next to the one GPBoost,
LightGBM and XGBoost load.

Writes results/sheet_comparison.json and results/sheet_comparison_folds.csv.
Usage (from src/):  python sheet_comparison.py [--parts ABC] [--max-folds K]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (GroupShuffleSplit, LeaveOneOut, RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit)
from sklearn.preprocessing import StandardScaler

from asteroid_view import rule_scores
from common import H_LIMIT, MOID_LIMIT_AU, N_FOLDS, RESULTS, ROOT, SEED, binary_metrics, save_json, summarize_folds
from earth_view import build_dataset as build_earth_view

EXT = ROOT / "data" / "external"
SHEET = EXT / "NASA_Asteroid_Final_Comparative_Results.xlsx"
WORKER = Path(__file__).with_name("cnn_worker.py")
N_JOBS = 6            # fixed thread counts keep reruns identical
TORCH_THREADS = 4
MODELS = ("CNN", "GPBoost", "Tuned RF")
RULE = "Ours: path rule (training-free)"

CAD_COLS = ["dist", "dist_min", "dist_max", "v_rel", "v_inf", "h"]          # the CAD API's own numeric fields
SENTRY_COLS = ["h", "diameter", "log10_ip", "n_imp", "ps_max", "ps_cum", "ts_max", "v_inf", "last_obs_jd",
               "range_first", "range_last"]                                 # the Sentry table's own fields
RF_SPACE = {
    "n_estimators": [200, 400],
    "max_depth": [None, 8, 16, 32],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", 0.5, 1.0],
    "class_weight": [None, "balanced", "balanced_subsample"],
}
# per data size: CNN epochs/patience/batch, GPBoost leaves/min leaf/rounds, RF search size and tuning subsample
PRESETS = {
    "tiny":   dict(cnn_full_batch_epochs=300, gb_leaves=7, gb_min_leaf=2, gb_rounds=200, rf_iter=0, rf_rows=None),
    "medium": dict(cnn_epochs=200, cnn_patience=15, cnn_batch=64, gb_leaves=15, gb_min_leaf=5, gb_rounds=200,
                   rf_iter=10, rf_rows=None),
    "large":  dict(cnn_epochs=25, cnn_patience=3, cnn_batch=2048, gb_leaves=63, gb_min_leaf=20, gb_rounds=300,
                   rf_iter=8, rf_rows=40_000),
}


# ---------------------------------------------------------------- A. what the sheet itself says
def decode_counts(acc, prec, rec, n, pos=None):
    """Every (P, TP, FP) on a test set of n rows that reproduces accuracy, precision and recall to 1e-9."""
    sols = []
    for P in ([pos] if pos else range(1, n)):
        tp = round(rec * P)
        if abs(tp / P - rec) > 1e-9:
            continue
        fps = ([round(tp / prec) - tp] if prec > 0 else []) if tp > 0 else range(0, n - P + 1)
        for fp in fps:
            tn = n - P - fp
            if fp < 0 or tn < 0 or (tp > 0 and abs(tp / (tp + fp) - prec) > 1e-9) or abs((tp + tn) / n - acc) > 1e-9:
                continue
            sols.append({"n": n, "positives": P, "TP": tp, "FP": fp, "FN": P - tp, "TN": tn})
    return sols


def wilson(k, n, z=1.959964):
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]


def part_a() -> dict:
    sheet = pd.read_excel(SHEET, sheet_name="Master Results")
    delivered = {"NASA CAD": len(pd.read_csv(EXT / "cad.csv")), "NASA SCOUT": len(pd.read_csv(EXT / "scout.csv")),
                 "NASA SENTRY": len(pd.read_csv(EXT / "sentry_risk_list.csv"))}
    out = {"file": SHEET.name, "sha256": hashlib.sha256(SHEET.read_bytes()).hexdigest(), "rows": [], "test_sets": {}}
    for ds, g in sheet.groupby("Dataset", sort=False):
        # smallest test set that reproduces every model with recall > 0 (the three share one test set)
        informative = g[g["Recall"] > 0]
        n_test = pos = None
        for n in range(2, 5001):
            sols = [decode_counts(r.Accuracy, r.Precision, r.Recall, n) for r in informative.itertuples()]
            if all(sols) and len({(s[0]["n"], s[0]["positives"]) for s in sols}) == 1 and all(len(s) == 1 for s in sols):
                n_test, pos = n, sols[0][0]["positives"]
                break
        assert n_test is not None, ds
        out["test_sets"][ds] = {"n_test": n_test, "positives": pos, "negatives": n_test - pos,
                                "rows_in_delivered_file": delivered[ds],
                                "test_share_of_delivered_file": n_test / delivered[ds],
                                "note": "smallest consistent size; any other must be an exact multiple with every count scaled"}
        for r in g.itertuples():
            s = decode_counts(r.Accuracy, r.Precision, r.Recall, n_test, pos)
            assert len(s) == 1, (ds, r.Method, s)
            c = s[0]
            u = r.ROC_AUC * c["positives"] * (n_test - c["positives"])   # Mann-Whitney U: a multiple of 1/2
            out["rows"].append({
                "dataset": ds, "model": r.Method, "accuracy": r.Accuracy, "precision": r.Precision, "recall": r.Recall,
                "f1": r.F1, "roc_auc": r.ROC_AUC, **{k: c[k] for k in ("n", "positives", "TP", "FP", "FN", "TN")},
                "auc_consistent_with_counts": bool(abs(2 * u - round(2 * u)) < 1e-6),
                "accuracy_wilson95": wilson(c["TP"] + c["TN"], n_test),
                "majority_class_accuracy": max(pos, n_test - pos) / n_test,
                "predicted_positive": c["TP"] + c["FP"],
            })
    return out


# ---------------------------------------------------------------- the three model families
def _impute(Xtr, Xte):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    return imp.transform(Xtr), imp.transform(Xte)


def tuned_rf_scores(Xtr, ytr, Xte, groups_tr, preset) -> tuple[np.ndarray, dict]:
    """Random forest tuned by randomized search (F1, inner 3-fold over the training fold only)."""
    Xtr, Xte = _impute(Xtr, Xte)
    n_pos, n_neg = int(ytr.sum()), int(len(ytr) - ytr.sum())
    if preset["rf_iter"] == 0 or min(n_pos, n_neg) < 3:      # too few positives to tune: library defaults, 400 trees
        rf = RandomForestClassifier(n_estimators=400, n_jobs=N_JOBS, random_state=SEED).fit(Xtr, ytr)
        return rf.predict_proba(Xte)[:, 1], {"tuned": False}
    idx = np.arange(len(ytr))
    if preset["rf_rows"] and len(idx) > preset["rf_rows"]:   # tune on a subsample (whole asteroids if grouped)
        frac = preset["rf_rows"] / len(idx)
        sp = (GroupShuffleSplit(1, train_size=frac, random_state=SEED).split(idx, ytr, groups_tr) if groups_tr is not None
              else StratifiedShuffleSplit(1, train_size=frac, random_state=SEED).split(idx, ytr))
        idx = next(sp)[0]
    inner = (StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(idx, ytr[idx], groups_tr[idx])
             if groups_tr is not None else StratifiedKFold(3, shuffle=True, random_state=SEED).split(idx, ytr[idx]))
    search = RandomizedSearchCV(RandomForestClassifier(n_jobs=N_JOBS, random_state=SEED), RF_SPACE,
                                n_iter=preset["rf_iter"], scoring="f1", cv=list(inner), random_state=SEED, refit=False)
    search.fit(Xtr[idx], ytr[idx])
    best = search.best_params_
    rf = RandomForestClassifier(n_jobs=N_JOBS, random_state=SEED, **best).fit(Xtr, ytr)
    return rf.predict_proba(Xte)[:, 1], {"tuned": True, "inner_f1": float(search.best_score_), **best}


def gpboost_scores(Xtr, ytr, Xte, gtr, gte, preset) -> tuple[np.ndarray, dict]:
    """GPBoost: tree boosting, plus a grouped random effect per asteroid when rows repeat an asteroid."""
    import gpboost as gpb
    params = {"learning_rate": 0.05, "num_leaves": preset["gb_leaves"], "min_data_in_leaf": preset["gb_min_leaf"],
              "verbose": -1, "seed": SEED, "num_threads": N_JOBS, "deterministic": True, "force_row_wise": True}
    data = gpb.Dataset(Xtr, ytr)
    if gtr is None:
        bst = gpb.train(params={**params, "objective": "binary"}, train_set=data, num_boost_round=preset["gb_rounds"])
        return np.asarray(bst.predict(Xte), float), {"random_effect": None}
    gp_model = gpb.GPModel(group_data=gtr, likelihood="bernoulli_probit", num_parallel_threads=N_JOBS)
    bst = gpb.train(params=params, train_set=data, gp_model=gp_model, num_boost_round=preset["gb_rounds"])
    pred = bst.predict(data=Xte, group_data_pred=gte, pred_latent=False)
    return np.asarray(pred["response_mean"], float), {"random_effect": "asteroid",
                                                      "re_variance": float(np.ravel(gp_model.get_cov_pars())[0])}


def cnn_job(Xtr, ytr, Xte, groups_tr, preset) -> dict:
    """Impute and standardize on the training fold, split off early-stopping rows, package for cnn_worker.py."""
    Xtr, Xte = _impute(Xtr, Xte)
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr).astype(np.float32), sc.transform(Xte).astype(np.float32)
    job = {"B": B, "seed": SEED, "threads": TORCH_THREADS}
    if "cnn_full_batch_epochs" in preset:                    # tiny tables: full batch, fixed epochs
        return {**job, "A_fit": A, "y_fit": ytr.astype(np.float32), "A_val": None, "y_val": None,
                "full_batch": True, "epochs": preset["cnn_full_batch_epochs"]}
    sp = (GroupShuffleSplit(1, test_size=0.1, random_state=SEED).split(A, ytr, groups_tr) if groups_tr is not None
          else StratifiedShuffleSplit(1, test_size=0.1, random_state=SEED).split(A, ytr))
    fit_idx, val_idx = next(sp)                              # 10% of the training rows (whole asteroids if grouped)
    return {**job, "A_fit": A[fit_idx], "y_fit": ytr[fit_idx].astype(np.float32), "A_val": A[val_idx],
            "y_val": ytr[val_idx].astype(np.float32), "full_batch": False, "epochs": preset["cnn_epochs"],
            "patience": preset["cnn_patience"], "batch": preset["cnn_batch"]}


def run_cnn_jobs(jobs: list[dict]) -> list[tuple[np.ndarray, dict]]:
    with tempfile.TemporaryDirectory() as d:
        inp, out = Path(d) / "jobs.pkl", Path(d) / "results.pkl"
        inp.write_bytes(pickle.dumps(jobs, protocol=5))
        subprocess.run([sys.executable, str(WORKER), str(inp), str(out)], check=True)
        return pickle.loads(out.read_bytes())


# ---------------------------------------------------------------- B. the three files as delivered
def load_features() -> pd.DataFrame:
    """Per-asteroid orbit, H, our path-to-path distance and JPL's PHA flag (written by asteroid_view.py)."""
    f = pd.read_csv(RESULTS / "asteroid_features.csv", low_memory=False, dtype={"pdes": str},
                    usecols=["pdes", "label", "H", "moid", "moid_path", "condition_code"])
    return f.set_index("pdes")


def oof_scores(X, y, folds, preset) -> tuple[dict, dict]:
    """Out-of-fold scores of the three model families on the given folds (CNN jobs batched into one worker)."""
    oof = {m: np.full(len(y), np.nan) for m in MODELS}
    notes = {m: [] for m in MODELS}
    for (tr, te), (s, info) in zip(folds, run_cnn_jobs([cnn_job(X[tr], y[tr], X[te], None, preset) for tr, te in folds])):
        oof["CNN"][te] = s
        notes["CNN"].append(info)
    for tr, te in folds:
        s, info = gpboost_scores(X[tr], y[tr], X[te], None, None, preset)
        oof["GPBoost"][te] = s
        notes["GPBoost"].append(info)
        s, info = tuned_rf_scores(X[tr], y[tr], X[te], None, preset)
        oof["Tuned RF"][te] = s
        notes["Tuned RF"].append(info)
    return oof, notes


def part_b(feat: pd.DataFrame, fold_rows: list) -> dict:
    out = {}
    # --- CAD file: 33 approaches, every one within 0.05 au; leave-one-out for the trained models
    cad_all = pd.read_csv(EXT / "cad.csv", dtype={"des": str})
    cad_all["des"] = cad_all["des"].str.strip()
    cad = cad_all[cad_all["des"].isin(feat.index)].reset_index(drop=True)
    X, y = cad[CAD_COLS].to_numpy(float), feat.loc[cad["des"], "label"].to_numpy(int)
    rule_pred, rule_score = rule_scores(feat.loc[cad["des"]], "moid_path")
    oof, notes = oof_scores(X, y, list(LeaveOneOut().split(X)), PRESETS["tiny"])
    res = {"rows_in_file": int(len(cad_all)), "rows_matched_to_sbdb": int(len(cad)), "positives": int(y.sum()),
           "label": "JPL PHA flag", "protocol": "leave-one-out (pooled out-of-fold predictions)",
           "features_for_trained_models": CAD_COLS, "methods": {RULE: binary_metrics(y, rule_pred, rule_score)},
           "tuned_rf_folds_tuned": int(sum(bool(i["tuned"]) for i in notes["Tuned RF"])),
           "rule_errors": cad.loc[rule_pred != y, ["des", "h", "dist"]].to_dict("records")}
    for name in MODELS:
        res["methods"][name] = binary_metrics(y, (oof[name] >= 0.5).astype(int), oof[name])
        res["methods"][name]["errors"] = cad.loc[(oof[name] >= 0.5).astype(int) != y, "des"].tolist()
    for name, m in res["methods"].items():
        print(f"[B] CAD file    {name:<32} acc={m['accuracy']:.4f} f1={m['f1']:.4f} mcc={m['mcc']:.4f}", flush=True)
    out["cad_file"] = res

    # --- Sentry file: 2,242 objects; 5 x 5 repeated stratified CV, out-of-fold predictions pooled per repeat
    se_all = pd.read_csv(EXT / "sentry_risk_list.csv", dtype={"des": str})
    se_all["des"] = se_all["des"].str.strip()
    se = se_all[se_all["des"].isin(feat.index)].reset_index(drop=True)
    se["log10_ip"] = np.log10(se["ip"])
    yrs = se["range"].str.split("-", expand=True).astype(int)
    se["range_first"], se["range_last"] = yrs[0], yrs[1]
    X, y = se[SENTRY_COLS].to_numpy(float), feat.loc[se["des"], "label"].to_numpy(int)
    sub = feat.loc[se["des"]]
    rule_pred, rule_score = rule_scores(sub, "moid_path")
    rskf = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(X, y))
    per_repeat = []
    for rep in range(5):
        oof, _ = oof_scores(X, y, rskf[rep * 5:(rep + 1) * 5], PRESETS["medium"])
        for name in MODELS:
            per_repeat.append({"part": "B", "dataset": "Sentry file", "protocol": "repeated_stratified", "fold": rep,
                               "method": name, **binary_metrics(y, (oof[name] >= 0.5).astype(int), oof[name])})
        per_repeat.append({"part": "B", "dataset": "Sentry file", "protocol": "repeated_stratified", "fold": rep,
                           "method": RULE, **binary_metrics(y, rule_pred, rule_score)})
        print(f"[B] Sentry repeat {rep}: " + ", ".join(f"{r['method'].split(':')[0]} f1={r['f1']:.3f}"
                                                      for r in per_repeat[-4:]), flush=True)
    fold_rows.extend(per_repeat)
    summ = summarize_folds(pd.DataFrame(per_repeat).drop(columns=["part", "dataset", "protocol"]), ["method"])
    wrong = rule_pred != y
    out["sentry_file"] = {
        "rows_in_file": int(len(se_all)), "rows_matched_to_sbdb": int(len(se)), "positives": int(y.sum()),
        "positives_with_H_le_22": int(((sub["H"] <= H_LIMIT) & (y == 1)).sum()),
        "objects_with_H_le_22": int((sub["H"] <= H_LIMIT).sum()),
        "objects_with_moid_path_le_0.05": int((sub["moid_path"] <= MOID_LIMIT_AU).sum()),
        "label": "JPL PHA flag", "protocol": "5x5 repeated stratified CV, out-of-fold predictions pooled per repeat",
        "features_for_trained_models": SENTRY_COLS, "methods": summ.set_index("method").to_dict("index"),
        "rule_errors": pd.DataFrame({"des": se["des"][wrong], "H": sub["H"].to_numpy()[wrong],
                                     "moid_path": sub["moid_path"].to_numpy()[wrong],
                                     "jpl_moid": sub["moid"].to_numpy()[wrong], "label": y[wrong]}).to_dict("records"),
    }

    # --- Scout file: unconfirmed objects (no designation, no JPL orbit, no PHA flag): no ground truth exists
    sc = pd.read_csv(EXT / "scout.csv")
    flag = (sc["moid"] <= MOID_LIMIT_AU) & (sc["H"] <= H_LIMIT)
    out["scout_file"] = {
        "rows_in_file": int(len(sc)), "ground_truth": None,
        "why": "Scout lists unconfirmed NEOCP candidates; they have no designation, JPL orbit or PHA flag yet",
        "names_found_in_sbdb": int(sc["objectName"].astype(str).isin(feat.index).sum()),
        "rule_on_scout_nominal_moid_and_H": {
            "flagged": int(flag.sum()),
            "flagged_objects": sc.loc[flag, ["objectName", "H", "moid", "phaScore"]].to_dict("records")},
        "scout_phaScore_gt_0": int((sc["phaScore"] > 0).sum()), "scout_phaScore_ge_50": int((sc["phaScore"] >= 50).sum()),
    }
    return out


# ---------------------------------------------------------------- C. full NASA CAD benchmark
def part_c(feat: pd.DataFrame, fold_rows: list, max_folds: int | None) -> dict:
    ev = build_earth_view()
    X = ev[CAD_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y, groups = ev["label"].to_numpy(int), ev["des"].to_numpy()
    gcode = pd.factorize(groups)[0]
    splits = {   # identical to earth_view.py (the split depends only on y, the groups and the seed)
        "row_split": list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y)),
        "asteroid_grouped": list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y, groups)),
    }
    rule_pred, rule_score = rule_scores(feat.loc[ev["des"]], "moid_path")
    same = pd.read_csv(RESULTS / "same_folds_rows.csv")
    ref = same[same["method"] == "rule: exact path geometry (ours)"].set_index("fold")["accuracy"]
    notes, rows = {}, []
    for proto, folds in splits.items():
        for k, (tr, te) in enumerate(folds[:max_folds]):
            m = binary_metrics(y[te], rule_pred[te], rule_score[te])
            if proto == "asteroid_grouped":
                assert abs(m["accuracy"] - ref.loc[k]) < 1e-12, "folds differ from the paper's same-fold comparison"
            rows.append({"part": "C", "dataset": "NASA CAD (full)", "protocol": proto, "fold": k, "method": RULE, **m})
            for name in MODELS:
                t0 = time.time()
                gtr = gcode[tr] if proto == "asteroid_grouped" else None   # early-stopping / tuning splits
                if name == "CNN":
                    s, info = run_cnn_jobs([cnn_job(X[tr], y[tr], X[te], gtr, PRESETS["large"])])[0]
                elif name == "GPBoost":
                    s, info = gpboost_scores(X[tr], y[tr], X[te], gcode[tr], gcode[te], PRESETS["large"])
                else:
                    s, info = tuned_rf_scores(X[tr], y[tr], X[te], gtr, PRESETS["large"])
                m = binary_metrics(y[te], (s >= 0.5).astype(int), s)
                rows.append({"part": "C", "dataset": "NASA CAD (full)", "protocol": proto, "fold": k, "method": name,
                             **m, "fit_seconds": time.time() - t0})
                notes.setdefault(f"{proto}/{name}", []).append(info)
                print(f"[C] {proto:16s} fold {k} {name:<9} acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
                      f"auc={m['roc_auc']:.4f} ({time.time() - t0:.0f}s) {info}", flush=True)
            fold_rows.extend(rows[-4:])
            pd.DataFrame(fold_rows).to_csv(RESULTS / "sheet_comparison_folds.csv", index=False)
    summ = summarize_folds(pd.DataFrame(rows).drop(columns=["part", "dataset"]), ["protocol", "method"])
    out = {"rows": int(len(ev)), "asteroids": int(ev["des"].nunique()), "positive_rows": int(y.sum()),
           "label": "JPL PHA flag", "features_for_trained_models": CAD_COLS,
           "folds": "same StratifiedKFold / StratifiedGroupKFold (seed 20261111) as earth_view.py",
           "rule_matches_same_folds_rows_csv": True, "model_notes": notes, "protocols": {}}
    for proto, g in summ.groupby("protocol", sort=False):
        out["protocols"][proto] = g.drop(columns="protocol").set_index("method").to_dict("index")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="ABC")
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args()
    out_path = RESULTS / "sheet_comparison.json"
    out = json.loads(out_path.read_text()) if (out_path.exists() and args.parts != "ABC") else {}
    fold_rows: list = []
    if args.parts != "ABC" and (RESULTS / "sheet_comparison_folds.csv").exists():
        keep = pd.read_csv(RESULTS / "sheet_comparison_folds.csv")     # keep rows of the parts not rerun
        fold_rows = keep[~keep["part"].isin(list(args.parts))].to_dict("records")
    if "A" in args.parts:
        out["sheet"] = part_a()
        for ds, t in out["sheet"]["test_sets"].items():
            print(f"[A] {ds:<12} test set {t['n_test']} rows ({t['positives']} positive); delivered file has "
                  f"{t['rows_in_delivered_file']} rows", flush=True)
    feat = load_features() if ("B" in args.parts or "C" in args.parts) else None
    if "B" in args.parts:
        out["files"] = part_b(feat, fold_rows)
    if "C" in args.parts:
        out["cad_full"] = part_c(feat, fold_rows, args.max_folds)
    if args.max_folds is None:
        pd.DataFrame(fold_rows).to_csv(RESULTS / "sheet_comparison_folds.csv", index=False)
        save_json(out, out_path)
    print("done", flush=True)


if __name__ == "__main__":
    main()
