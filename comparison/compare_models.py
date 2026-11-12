"""Full model code for the comparison: our training-free path rule against all nine Earth's-view models.

One self-contained file that imports nothing from src/. It runs on a laptop from the repository and on Kaggle from
the uploaded package (comparison/KAGGLE_GUIDE.md). It reproduces Tables IV and IX of the paper (Tables 4 and 9 of
the MDPI version) fold for fold, using the same data, label, features, folds, seeds and hyperparameters as
src/earth_view.py, src/sheet_comparison.py, src/cnn_worker.py, src/asteroid_view.py and src/orbit_geometry.py.

  Ours      a training-free rule on the exact distance between the asteroid's path and Earth's orbit (this paper)
  CNN       a one-dimensional convolutional network (LeCun et al. 1998; Kiranyaz et al. 2021), trained with Adam
            (Kingma and Ba 2015), with dropout (Srivastava et al. 2014) and early stopping
  GPBoost   tree boosting combined with a Gaussian random effect per asteroid (Sigrist 2022; Sigrist 2023 for the
            binary likelihood)
  Tuned RF  a random forest (Breiman 2001) tuned for F1 by randomized search (Bergstra and Bengio 2012)
  and the six models of prior work: logistic regression, k-nearest neighbours, random forest, XGBoost, LightGBM and
  a multilayer perceptron
Full citations, links and what each model takes from its paper: comparison/README.md and comparison/references.bib.

The label is JPL's PHA flag for every method. Each unit writes <output>/<unit>.csv:
  B-cad      the 33 Earth approaches of NASA CAD in the comparison sheet (September-November 2026), leave-one-out
  B-sentry   the 2,242-object NASA Sentry impact-risk list, 5 x 5 repeated stratified cross-validation
  E-grouped  306,590 JPL close approaches (Earth, <= 0.5 au, 1910-2024), the six prior-work models on the
             NeoWs-style features, stratified 5-fold grouped by asteroid (no asteroid in both train and test)
  E-row      the same, stratified 5-fold over rows: the prior-work split, where an asteroid's other approaches
             can sit in the training folds
  C-grouped  the same rows and grouped folds, CNN, GPBoost and the tuned random forest on the CAD table's columns
  C-row      the same with the row split
Our rule is scored in every unit. The report compares every fold with the paper's stored results and runs an exact
paired McNemar test of the rule against every model on the grouped folds, from the row-level predictions.

Usage, from the repository root with the project's environment (versions: requirements.txt):
  .venv/bin/python comparison/compare_models.py                    # all six units, then the report
  .venv/bin/python comparison/compare_models.py --units C-grouped  # one unit, then the report on what exists
  .venv/bin/python comparison/compare_models.py --report           # the report only, from the saved outputs
  .venv/bin/python comparison/compare_models.py --max-folds 1      # a quick smoke test, not comparable to the paper
The data files are found by name: data/processed/ and data/external/ in the repository (a fresh clone unpacks
datasets/neo_datasets_2026-11-11.zip by itself), data/ next to this file in the Kaggle package, or anywhere under
/kaggle/input. Outputs go to comparison/output/, or to
/kaggle/working/comparison_output/ on Kaggle.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

# The CNN trains in a child process of this same file that imports only numpy and torch. On macOS, PyTorch's OpenMP
# runtime crashes next to the one that GPBoost and scikit-learn load, so they never share a process.
CNN_WORKER = len(sys.argv) > 1 and sys.argv[1] == "--cnn-worker"
if not CNN_WORKER:
    import pandas as pd
    from sklearn import metrics as skm
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import (GroupShuffleSplit, LeaveOneOut, RandomizedSearchCV, RepeatedStratifiedKFold,
                                         StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit)
    from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                          # the repository root when this file sits in comparison/
KAGGLE_INPUT, KAGGLE_WORKING = Path("/kaggle/input"), Path("/kaggle/working")
EXTRA_ROOTS: list = []                      # --data
OUT = HERE / "output"                       # set in main()

SEED, N_FOLDS = 20261111, 5                 # the paper's seed and number of folds
MOID_LIMIT_AU, H_LIMIT = 0.05, 22.0         # PHA definition (JPL/CNEOS): MOID <= 0.05 au and H <= 22
AU_KM = 149_597_870.7
N_JOBS, TORCH_THREADS = 6, 4                # fixed thread counts keep reruns identical
MODELS = ("CNN", "GPBoost", "Tuned RF")
PRIOR = ("LogReg", "KNN", "RandomForest", "XGBoost", "LightGBM", "MLP")
RULE = "Ours: path rule (training-free)"
METHODS = (RULE, *MODELS)
UNITS = ("B-cad", "B-sentry", "E-grouped", "E-row", "C-grouped", "C-row")   # fastest first

CAD_COLS = ["dist", "dist_min", "dist_max", "v_rel", "v_inf", "h"]       # the CAD API's own numeric fields
SENTRY_COLS = ["h", "diameter", "log10_ip", "n_imp", "ps_max", "ps_cum", "ts_max", "v_inf", "last_obs_jd",
               "range_first", "range_last"]                              # the Sentry table's own fields
NEOWS_COLS = ["absolute_magnitude", "est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance"]


# =====================================================================================================================
# 1. Our method: the asteroid's path against Earth's orbit as a fixed curve, and a training-free rule
# =====================================================================================================================
# Earth's orbit: J2000 mean elements of the Earth-Moon barycentre (Standish, JPL SSD "Keplerian elements for
# approximate positions of the major planets", Table 1), a fixed Keplerian ellipse in the ecliptic. Only the
# asteroid's own osculating elements (a, e, i, node, perihelion argument) and H enter; no close-approach data, no
# Earth position at any time, and never JPL's MOID or PHA fields.
EARTH_A = 1.00000261                        # au
EARTH_E = 0.01671123
EARTH_VARPI = np.deg2rad(102.93768193)      # longitude of perihelion
EARTH_P = EARTH_A * (1.0 - EARTH_E**2)      # semi-latus rectum, au
TWO_PI = 2.0 * np.pi
GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0


def earth_radius(lam):
    """Heliocentric distance (au) of Earth's orbit at ecliptic longitude `lam` (rad)."""
    return EARTH_P / (1.0 + EARTH_E * np.cos(lam - EARTH_VARPI))


def orbit_basis(i, om, w):
    """Perifocal unit vectors P (towards perihelion) and Q (90 deg ahead) in the ecliptic frame; radians in."""
    ci, si = np.cos(i), np.sin(i)
    co, so = np.cos(om), np.sin(om)
    cw, sw = np.cos(w), np.sin(w)
    P = np.stack([cw * co - sw * so * ci, cw * so + sw * co * ci, sw * si], axis=-1)
    Q = np.stack([-sw * co - cw * so * ci, -sw * so + cw * co * ci, cw * si], axis=-1)
    return P, Q


def path_points(p, e, P, Q, f):
    """Heliocentric ecliptic positions on the asteroid's path at true anomalies f (shape (n, m)) -> (n, m, 3)."""
    r = p[:, None] / (1.0 + e[:, None] * np.cos(f))
    return r[..., None] * (np.cos(f)[..., None] * P[:, None, :] + np.sin(f)[..., None] * Q[:, None, :])


def distance_to_earth_orbit(X, newton_iters=4):
    """Exact distance (au) from points X (..., 3) to Earth's orbital ellipse.

    Minimizes D(lam) = |X - E(lam)|^2 over Earth's true longitude lam with Newton steps started at the point's own
    longitude (Earth's eccentricity is small, so a few steps converge).
    """
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)
    lam = phi.copy()
    for _ in range(newton_iters):
        psi = lam - EARTH_VARPI
        R = earth_radius(lam)
        R1 = (EARTH_E / EARTH_P) * R**2 * np.sin(psi)
        R2 = (EARTH_E / EARTH_P) * (2.0 * R * R1 * np.sin(psi) + R**2 * np.cos(psi))
        cd, sd = np.cos(lam - phi), np.sin(lam - phi)
        g1 = -2 * R1 * rho * cd + 2 * R * rho * sd + 2 * R * R1
        g2 = -2 * R2 * rho * cd + 4 * R1 * rho * sd + 2 * R * rho * cd + 2 * R1**2 + 2 * R * R2
        step = np.where(g2 > 0, g1 / np.where(g2 > 0, g2, 1.0), 0.0)
        lam = lam - step
    R = earth_radius(lam)
    D = rho**2 + z**2 - 2.0 * R * rho * np.cos(lam - phi) + R**2
    return np.sqrt(np.maximum(D, 0.0))


def moid_earth(a, e, i_deg, om_deg, w_deg, n_grid=2048, n_candidates=4, golden_iters=48, chunk=256):
    """Exact path-to-path minimum distance (au) between each asteroid's orbit and Earth's orbit.

    A grid over the asteroid's true anomaly, the exact distance to Earth's orbit at each grid point, then
    golden-section refinement of the `n_candidates` deepest local minima. Angles in degrees, as in JPL SBDB.
    """
    a, e = np.asarray(a, float), np.asarray(e, float)
    i, om, w = (np.deg2rad(np.asarray(v, float)) for v in (i_deg, om_deg, w_deg))
    out = np.empty(len(a))
    f_grid = np.linspace(0.0, TWO_PI, n_grid, endpoint=False)
    h = TWO_PI / n_grid
    for s in range(0, len(a), chunk):
        sl = slice(s, min(s + chunk, len(a)))
        p = a[sl] * (1.0 - e[sl] ** 2)
        P, Q = orbit_basis(i[sl], om[sl], w[sl])
        n = len(p)
        f = np.broadcast_to(f_grid, (n, n_grid))
        d = distance_to_earth_orbit(path_points(p, e[sl], P, Q, f))
        is_min = (d <= np.roll(d, 1, axis=1)) & (d <= np.roll(d, -1, axis=1))
        k = min(n_candidates, n_grid)
        idx = np.argsort(np.where(is_min, d, np.inf), axis=1)[:, :k]
        valid = np.take_along_axis(is_min, idx, axis=1)
        lo = f_grid[idx] - h
        hi = f_grid[idx] + h

        def dist_at(fv):
            return distance_to_earth_orbit(path_points(p, e[sl], P, Q, fv))

        # vectorized golden-section search: interior points c < d, one new evaluation per step
        c = hi - GOLDEN * (hi - lo)
        d_pt = lo + GOLDEN * (hi - lo)
        fc, fd = dist_at(c), dist_at(d_pt)
        for _ in range(golden_iters):
            left = fc < fd                                        # minimum in [lo, d]; otherwise in [c, hi]
            hi = np.where(left, d_pt, hi)
            lo = np.where(left, lo, c)
            new_c = np.where(left, hi - GOLDEN * (hi - lo), d_pt)  # left: fresh point; right: old d
            new_d = np.where(left, c, lo + GOLDEN * (hi - lo))     # left: old c; right: fresh point
            val = dist_at(np.where(left, new_c, new_d))
            fc, fd = np.where(left, val, fd), np.where(left, fc, val)
            c, d_pt = new_c, new_d
        refined = np.minimum(fc, fd)
        refined = np.where(valid, refined, np.inf)
        out[sl] = np.minimum(refined.min(axis=1), d.min(axis=1))
    return out


def hazard_index(distance_au, H):
    """max(distance / 0.05 au, D_22 / D): PHA if and only if the index is <= 1.

    D_22 / D = 10**((H - 22) / 5) is the size ratio to an H = 22 body of the same albedo, so both terms are pure
    ratios and only the path geometry and the size enter.
    """
    return np.maximum(np.asarray(distance_au) / MOID_LIMIT_AU, 10.0 ** ((np.asarray(H) - H_LIMIT) / 5.0))


def rule_scores(feat):
    """The training-free rule on the exact path distance: 0/1 prediction and a score (-log index) for ROC AUC."""
    hi = hazard_index(feat["moid_path"].values, feat["H"].values)
    return (hi <= 1.0).astype(int), -np.log(hi)


# =====================================================================================================================
# 2. Model 1: CNN (PyTorch), run in a child process
# =====================================================================================================================
def cnn_build(n_features):
    """1-D CNN over the standardized feature vector: two convolution layers, a dense head and one logit."""
    import torch
    nn = torch.nn
    return nn.Sequential(nn.Unflatten(1, (1, n_features)),
                         nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
                         nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                         nn.Flatten(), nn.Linear(64 * n_features, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))


def cnn_train(job):
    """Adam (lr 1e-3, weight decay 1e-5) on binary cross-entropy; early stopping on the validation rows."""
    import torch
    t0 = time.time()
    torch.manual_seed(job["seed"])
    rng = np.random.default_rng(job["seed"])
    net = cnn_build(job["A_fit"].shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    xf, yf = torch.from_numpy(job["A_fit"]), torch.from_numpy(job["y_fit"])
    info = {}
    if job["full_batch"]:                       # tiny tables: full batch, fixed number of epochs
        for _ in range(job["epochs"]):
            net.train()
            opt.zero_grad()
            loss_fn(net(xf).squeeze(1), yf).backward()
            opt.step()
        info["epochs"] = job["epochs"]
    else:                                       # mini-batches, early stopping on the validation rows
        xv, yv = torch.from_numpy(job["A_val"]), torch.from_numpy(job["y_val"])
        best, best_state, wait, epoch = math.inf, copy.deepcopy(net.state_dict()), 0, 0
        for epoch in range(job["epochs"]):
            net.train()
            perm = torch.from_numpy(rng.permutation(len(yf)))
            for s in range(0, len(perm), job["batch"]):
                b = perm[s:s + job["batch"]]
                opt.zero_grad()
                loss_fn(net(xf[b]).squeeze(1), yf[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = loss_fn(net(xv).squeeze(1), yv).item()
            if v < best - 1e-6:
                best, best_state, wait = v, copy.deepcopy(net.state_dict()), 0
            else:
                wait += 1
                if wait >= job["patience"]:
                    break
        net.load_state_dict(best_state)
        info.update(epochs=epoch + 1, val_loss=best)
    net.eval()
    with torch.no_grad():
        scores = torch.sigmoid(net(torch.from_numpy(job["B"])).squeeze(1)).numpy().astype(float)
    info["fit_seconds"] = time.time() - t0
    return scores, info


def cnn_worker(jobs_path, out_path):
    """Child-process entry point: train every job in jobs_path and pickle (scores, info) pairs to out_path."""
    import torch
    with open(jobs_path, "rb") as f:
        jobs = pickle.load(f)                   # written by run_cnn_jobs() of this file, in the same run
    torch.set_num_threads(jobs[0]["threads"] if jobs else 1)
    torch.use_deterministic_algorithms(True)
    results = [cnn_train(job) for job in jobs]
    with open(out_path, "wb") as f:
        pickle.dump(results, f, protocol=5)


def impute(Xtr, Xte):
    """Median imputation fitted on the training fold only."""
    imp = SimpleImputer(strategy="median").fit(Xtr)
    return imp.transform(Xtr), imp.transform(Xte)


def cnn_job(Xtr, ytr, Xte, groups_tr, preset):
    """Impute and standardize on the training fold, hold out early-stopping rows, and package one CNN job."""
    Xtr, Xte = impute(Xtr, Xte)
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


def run_cnn_jobs(jobs):
    """Train CNN jobs in a child process of this file and return [(test scores, info), ...] in order."""
    with tempfile.TemporaryDirectory() as d:
        inp, out = Path(d) / "jobs.pkl", Path(d) / "results.pkl"
        inp.write_bytes(pickle.dumps(jobs, protocol=5))
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--cnn-worker", str(inp), str(out)], check=True)
        return pickle.loads(out.read_bytes())


# =====================================================================================================================
# 3. Model 2: GPBoost
# =====================================================================================================================
def gpboost_scores(Xtr, ytr, Xte, gtr, gte, preset):
    """GPBoost: tree boosting, plus a grouped random effect per asteroid when rows repeat an asteroid.

    With groups, the model is F(X) + b_asteroid with a Bernoulli-probit likelihood, trained jointly (Sigrist 2022,
    2023). An asteroid that is not in the training rows gets b = 0, the prior mean, at prediction time.
    """
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


# =====================================================================================================================
# 4. Model 3: random forest tuned by randomized search
# =====================================================================================================================
RF_SPACE = {
    "n_estimators": [200, 400],
    "max_depth": [None, 8, 16, 32],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", 0.5, 1.0],
    "class_weight": [None, "balanced", "balanced_subsample"],
}


def tuned_rf_scores(Xtr, ytr, Xte, groups_tr, preset):
    """Random forest tuned by randomized search for F1, with an inner 3-fold split of the training fold only."""
    Xtr, Xte = impute(Xtr, Xte)
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


# Per data size: CNN epochs/patience/batch, GPBoost leaves/minimum leaf size/rounds, RF search size and tuning rows.
# Fixed in advance; nothing is tuned on test folds, and the decision threshold is 0.5 for every model.
PRESETS = {
    "tiny":   dict(cnn_full_batch_epochs=300, gb_leaves=7, gb_min_leaf=2, gb_rounds=200, rf_iter=0, rf_rows=None),
    "medium": dict(cnn_epochs=200, cnn_patience=15, cnn_batch=64, gb_leaves=15, gb_min_leaf=5, gb_rounds=200,
                   rf_iter=10, rf_rows=None),
    "large":  dict(cnn_epochs=25, cnn_patience=3, cnn_batch=2048, gb_leaves=63, gb_min_leaf=20, gb_rounds=300,
                   rf_iter=8, rf_rows=40_000),
}


# =====================================================================================================================
# 5. The six models of prior work (scikit-learn, XGBoost, LightGBM), with their fixed settings
# =====================================================================================================================
def prior_work_models():
    """Fresh instances of the six Earth's-view models of prior work (as in src/earth_view.py)."""
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from xgboost import XGBClassifier
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


# =====================================================================================================================
# 6. Data, metrics and the six units
# =====================================================================================================================
def locate(name, subdir):
    """A data or reference file by name: repository layout first, then the Kaggle package, then /kaggle/input."""
    for p in (ROOT / subdir / name, HERE / "data" / name, HERE / "paper_results" / name, HERE / name):
        if p.is_file():
            return p
    for r in (*EXTRA_ROOTS, HERE, ROOT / "data", KAGGLE_INPUT, KAGGLE_WORKING / "pkg"):
        if Path(r).is_dir():
            hits = sorted(Path(r).rglob(name))
            if hits:
                return hits[0]
    return None


def data_file(name, subdir):
    p = locate(name, subdir)
    if p is None:   # a fresh clone of the repository: unpack the JPL snapshot that ships with it in datasets/
        for z in sorted((ROOT / "datasets").glob("neo_datasets_*.zip"))[:1]:
            print(f"unpacking {z.name} into data/snapshot_tmp/", flush=True)
            zipfile.ZipFile(z).extractall(ROOT / "data" / "snapshot_tmp")
            p = locate(name, subdir)
    if p is None:
        raise FileNotFoundError(f"{name} not found: in the repository it belongs in {subdir}/, in the Kaggle package "
                                f"in data/ (attach the uploaded dataset with Add Input), or pass --data DIR")
    return p


def load_catalogue():
    """Every near-Earth asteroid with a PHA flag and an H (JPL SBDB snapshot of 2026-11-11), with our path distance.

    label = JPL's PHA flag (1 = PHA), the label for every method. Our rule reads only moid_path (computed here from
    the asteroid's own elements) and H.
    """
    df = pd.read_csv(data_file("sbdb_neo.csv", "data/processed"), low_memory=False)
    for c in ("H", "a", "e", "i", "om", "w"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["label"] = df["pha"].map({"Y": 1, "N": 0})
    df = df[df["label"].notna() & df["H"].notna()].copy()
    df["label"] = df["label"].astype(int)
    df = df.reset_index(drop=True)
    t0 = time.time()
    df["moid_path"] = moid_earth(df.a.values, df.e.values, df.i.values, df.om.values, df.w.values)
    print(f"exact path-to-path distance for {len(df):,} asteroids in {time.time() - t0:.1f} s", flush=True)
    return df


def close_approaches(cat):
    """The 306,590 Earth close approaches (<= 0.5 au, 1910-2024, JPL CAD) of labelled asteroids, in file order."""
    cad = pd.read_csv(data_file("cad_earth.csv", "data/processed"), low_memory=False)
    return cad.merge(cat[["pdes", "label", "H"]], left_on="des", right_on="pdes", how="inner").reset_index(drop=True)


def binary_metrics(y_true, y_pred, y_score=None):
    """Accuracy, balanced accuracy, and precision, recall and F1 of the PHA class; MCC; ROC and PR AUC."""
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


def unit_close_approaches(unit, cat, max_folds):
    """E-* and C-*: the 306,590 approaches with the row split or the grouped split.

    E units train the six prior-work models on the NeoWs-style features (absolute magnitude, the diameter range that
    NeoWs derives from it, relative velocity in km/h and miss distance in km); C units train the CNN, GPBoost and the
    tuned random forest on the CAD table's own columns. Both use identical folds, which depend only on the labels,
    the asteroids and the seed.
    """
    sbdb = cat.set_index("pdes")
    ev = close_approaches(cat)
    y, groups = ev["label"].to_numpy(int), ev["des"].to_numpy()
    gcode = pd.factorize(groups)[0]
    grouped = unit.endswith("grouped")
    if unit.startswith("E"):
        base = 10.0 ** (-ev["H"].to_numpy(float) / 5.0)        # NeoWs: diameters for albedo 0.25 (min) to 0.05 (max)
        ev["absolute_magnitude"] = ev["H"]
        ev["est_diameter_min"], ev["est_diameter_max"] = 1329.0 / np.sqrt(0.25) * base, 1329.0 / np.sqrt(0.05) * base
        ev["relative_velocity"] = pd.to_numeric(ev["v_rel"]) * 3600.0   # km/h, as in NeoWs
        ev["miss_distance"] = pd.to_numeric(ev["dist"]) * AU_KM          # km
        X, names = ev[NEOWS_COLS].to_numpy(float), PRIOR
    else:
        X, names = ev[CAD_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float), MODELS
    folds = (list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y, groups)) if grouped
             else list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y)))
    print(f"[{unit}] {len(ev):,} approaches of {ev['des'].nunique():,} asteroids, {int(y.sum()):,} by PHAs", flush=True)
    rule_pred, rule_score = rule_scores(sbdb.loc[ev["des"]])
    pred = {m: np.full(len(y), -1, np.int8) for m in (RULE, *names)}
    fold_of = np.full(len(y), -1, np.int8)
    rows, notes = [], []
    for k, (tr, te) in enumerate(folds[:max_folds]):
        fold_of[te] = k
        pred[RULE][te] = rule_pred[te]
        rows.append({"unit": unit, "fold": k, "method": RULE, **binary_metrics(y[te], rule_pred[te], rule_score[te])})
        prior = prior_work_models() if unit.startswith("E") else None
        for name in names:
            t0 = time.time()
            gtr = gcode[tr] if grouped else None           # early-stopping and tuning splits keep asteroids whole
            info = {}
            if prior is not None:
                prior[name].fit(X[tr], y[tr])
                s = prior[name].predict_proba(X[te])[:, 1]
            elif name == "CNN":
                s, info = run_cnn_jobs([cnn_job(X[tr], y[tr], X[te], gtr, PRESETS["large"])])[0]
            elif name == "GPBoost":                        # the random effect per asteroid is used in both splits
                s, info = gpboost_scores(X[tr], y[tr], X[te], gcode[tr], gcode[te], PRESETS["large"])
            else:
                s, info = tuned_rf_scores(X[tr], y[tr], X[te], gtr, PRESETS["large"])
            pred[name][te] = (s >= 0.5).astype(np.int8)
            m = binary_metrics(y[te], (s >= 0.5).astype(int), s)
            rows.append({"unit": unit, "fold": k, "method": name, **m, "fit_seconds": time.time() - t0})
            notes.append({"fold": k, "method": name, **info})
            print(f"[{unit}] fold {k} {name:<12} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    done = fold_of >= 0
    preds = pd.DataFrame({"des": ev["des"], "fold": fold_of, "label": y, **pred})[done]
    preds.to_csv(OUT / f"predictions_{unit}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    return rows, notes


def oof_scores(X, y, folds, preset):
    """Out-of-fold scores of the three models on the given folds (all CNN jobs in one child process)."""
    oof = {m: np.full(len(y), np.nan) for m in MODELS}
    notes = []
    for (tr, te), (s, info) in zip(folds, run_cnn_jobs([cnn_job(X[tr], y[tr], X[te], None, preset) for tr, te in folds])):
        oof["CNN"][te] = s
        notes.append({"method": "CNN", **info})
    for tr, te in folds:
        s, info = gpboost_scores(X[tr], y[tr], X[te], None, None, preset)
        oof["GPBoost"][te] = s
        notes.append({"method": "GPBoost", **info})
        s, info = tuned_rf_scores(X[tr], y[tr], X[te], None, preset)
        oof["Tuned RF"][te] = s
        notes.append({"method": "Tuned RF", **info})
    return oof, notes


def unit_cad_file(cat):
    """B-cad: the comparison sheet's NASA CAD table (33 approaches within 0.05 au); leave-one-out, pooled."""
    feat = cat.set_index("pdes")
    cad_all = pd.read_csv(data_file("cad.csv", "data/external"), dtype={"des": str})
    cad_all["des"] = cad_all["des"].str.strip()
    cad = cad_all[cad_all["des"].isin(feat.index)].reset_index(drop=True)
    X, y = cad[CAD_COLS].to_numpy(float), feat.loc[cad["des"], "label"].to_numpy(int)
    print(f"[B-cad] {len(cad)} of {len(cad_all)} approaches matched to the catalogue, {int(y.sum())} by PHAs", flush=True)
    rule_pred, rule_score = rule_scores(feat.loc[cad["des"]])
    oof, notes = oof_scores(X, y, list(LeaveOneOut().split(X)), PRESETS["tiny"])
    rows = [{"unit": "B-cad", "fold": 0, "method": RULE, **binary_metrics(y, rule_pred, rule_score)}]
    for name in MODELS:
        rows.append({"unit": "B-cad", "fold": 0, "method": name,
                     **binary_metrics(y, (oof[name] >= 0.5).astype(int), oof[name])})
    return rows, notes


def unit_sentry(cat, max_folds):
    """B-sentry: the Sentry impact-risk list; 5 x 5 repeated stratified CV, out-of-fold predictions per repeat."""
    feat = cat.set_index("pdes")
    se_all = pd.read_csv(data_file("sentry_risk_list.csv", "data/external"), dtype={"des": str})
    se_all["des"] = se_all["des"].str.strip()
    se = se_all[se_all["des"].isin(feat.index)].reset_index(drop=True)
    se["log10_ip"] = np.log10(se["ip"])
    yrs = se["range"].str.split("-", expand=True).astype(int)
    se["range_first"], se["range_last"] = yrs[0], yrs[1]
    X, y = se[SENTRY_COLS].to_numpy(float), feat.loc[se["des"], "label"].to_numpy(int)
    print(f"[B-sentry] {len(se)} of {len(se_all)} objects matched to the catalogue, {int(y.sum())} PHAs", flush=True)
    rule_pred, rule_score = rule_scores(feat.loc[se["des"]])
    rskf = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(X, y))
    rows, notes = [], []
    for rep in range(min(5, max_folds or 5)):
        oof, nts = oof_scores(X, y, rskf[rep * 5:(rep + 1) * 5], PRESETS["medium"])
        notes += [{"repeat": rep, **n} for n in nts]
        for name in MODELS:
            rows.append({"unit": "B-sentry", "fold": rep, "method": name,
                         **binary_metrics(y, (oof[name] >= 0.5).astype(int), oof[name])})
        rows.append({"unit": "B-sentry", "fold": rep, "method": RULE, **binary_metrics(y, rule_pred, rule_score)})
        print(f"[B-sentry] repeat {rep}: " + ", ".join(f"{r['method'].split(':')[0]} f1={r['f1']:.3f}"
                                                     for r in rows[-4:]), flush=True)
    return rows, notes


def run_unit(unit, cat, max_folds):
    t0 = time.time()
    if unit.startswith(("C-", "E-")):
        rows, notes = unit_close_approaches(unit, cat, max_folds)
    elif unit == "B-cad":
        rows, notes = unit_cad_file(cat)
    else:
        rows, notes = unit_sentry(cat, max_folds)
    pd.DataFrame(rows).to_csv(OUT / f"{unit}.csv", index=False)
    (OUT / f"{unit}_notes.json").write_text(json.dumps({"max_folds": max_folds, "seconds": time.time() - t0,
                                                         "notes": notes}, indent=1, default=float))
    print(f"[{unit}] done in {time.time() - t0:.0f} s", flush=True)


# =====================================================================================================================
# 7. Report: Tables IV and IX, a fold-by-fold check against the paper's stored results, and exact paired tests
# =====================================================================================================================
# As printed in the paper: (accuracy %, F1, MCC, ROC AUC) for C units, (accuracy %, F1, MCC) for E units
# (Table IV) and (accuracy %, F1) for B units (Table IX, bottom)
RULE_CAD = (99.79, 0.988, 0.987, 1.000)
PAPER = {
    ("C-row", "CNN"): (92.67, 0.380, 0.405, 0.943), ("C-row", "GPBoost"): (99.90, 0.994, 0.994, 1.000),
    ("C-row", "Tuned RF"): (89.42, 0.595, 0.585, 0.958), ("C-row", RULE): RULE_CAD,
    ("C-grouped", "CNN"): (92.61, 0.381, 0.401, 0.942), ("C-grouped", "GPBoost"): (91.26, 0.000, 0.000, 0.919),
    ("C-grouped", "Tuned RF"): (87.00, 0.527, 0.509, 0.937), ("C-grouped", RULE): RULE_CAD,
    ("E-row", "LogReg"): (91.12, 0.279, 0.268), ("E-row", "KNN"): (92.48, 0.514, 0.479),
    ("E-row", "MLP"): (92.67, 0.400, 0.413), ("E-row", "LightGBM"): (93.21, 0.479, 0.480),
    ("E-row", "XGBoost"): (93.27, 0.513, 0.500), ("E-row", "RandomForest"): (94.74, 0.676, 0.650),
    ("E-row", RULE): RULE_CAD[:3],
    ("E-grouped", "LogReg"): (91.09, 0.277, 0.266), ("E-grouped", "KNN"): (91.09, 0.425, 0.381),
    ("E-grouped", "MLP"): (92.63, 0.398, 0.410), ("E-grouped", "LightGBM"): (92.02, 0.397, 0.382),
    ("E-grouped", "XGBoost"): (91.84, 0.414, 0.388), ("E-grouped", "RandomForest"): (90.76, 0.428, 0.380),
    ("E-grouped", RULE): RULE_CAD[:3],
    ("B-cad", "CNN"): (93.94, 0.000), ("B-cad", "GPBoost"): (90.91, 0.000), ("B-cad", "Tuned RF"): (93.94, 0.000),
    ("B-cad", RULE): (100.00, 1.000),
    ("B-sentry", "CNN"): (99.56, 0.656), ("B-sentry", "GPBoost"): (99.70, 0.756),
    ("B-sentry", "Tuned RF"): (99.67, 0.784), ("B-sentry", RULE): (100.00, 1.000),
}
COLS = {"C": ["accuracy", "f1", "mcc", "roc_auc"], "E": ["accuracy", "f1", "mcc"], "B": ["accuracy", "f1"]}
METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc"]
STORED = {"C-row": ("C", "NASA CAD (full)", "row_split"), "C-grouped": ("C", "NASA CAD (full)", "asteroid_grouped"),
          "B-sentry": ("B", "Sentry file", "repeated_stratified")}
HONEST = ("B-cad", "B-sentry", "E-grouped", "C-grouped")    # no asteroid in both train and test
# GPBoost estimates its random-effect variance with multithreaded sums whose order varies between runs; this moves
# its ROC and PR AUC by up to 9e-6 per fold and never changes a prediction (proofs/REPRODUCTION.md)
TOL, TOL_GPBOOST_AUC = 1e-9, 2e-5


def log10_mcnemar(b, c):
    """log10 of the exact one-sided McNemar p-value P(X <= c), X ~ Binomial(b + c, 1/2), in integer arithmetic.

    b: rows the model gets wrong and the rule right; c: rows the rule gets wrong and the model right.
    """
    n = b + c
    return math.log10(sum(math.comb(n, k) for k in range(c + 1))) - n * math.log10(2)


def stored_reference(unit):
    """The paper's per-fold results for a unit (results/ in the repository, paper_results/ in the Kaggle package)."""
    if unit.startswith("E"):
        p = locate("earth_view_folds.csv", "results")
        if p is None:
            return None
        proto = "asteroid_grouped" if unit == "E-grouped" else "row_split"
        ref = pd.read_csv(p).rename(columns={"model": "method"})
        ref = ref[ref["protocol"] == proto]
        q = locate("sheet_comparison_folds.csv", "results")   # the rule on the same folds, stored with Table IX
        if q is not None:
            s = pd.read_csv(q)
            ref = pd.concat([ref, s[(s["part"] == "C") & (s["protocol"] == proto) & (s["method"] == RULE)]])
        return ref
    if unit == "B-cad":
        p = locate("sheet_comparison.json", "results")
        if p is None:
            return None
        methods = json.loads(p.read_text())["files"]["cad_file"]["methods"]
        return pd.DataFrame([{"fold": 0, "method": m, **{c: methods[m][c] for c in METRICS}} for m in METHODS])
    p = locate("sheet_comparison_folds.csv", "results")
    if p is None:
        return None
    s = pd.read_csv(p)
    part, ds, proto = STORED[unit]
    return s[(s["part"] == part) & (s["dataset"] == ds) & (s["protocol"] == proto)]


def report():
    lines = []

    def say(s=""):
        lines.append(s)
        print(s, flush=True)

    frames = {u: pd.read_csv(OUT / f"{u}.csv") for u in UNITS if (OUT / f"{u}.csv").exists()}
    if not frames:
        print("no unit has run yet")
        return False
    full = {u: f["fold"].nunique() == (1 if u == "B-cad" else 5) for u, f in frames.items()}
    say("RESULTS (means over folds or repeats; Table IV and Table IX of the paper)")
    table_ok = True
    for unit, f in frames.items():
        cols = COLS[unit[0]]
        say(f"  {unit}: {f['fold'].nunique()} fold(s) or repeat(s){'' if full[unit] else ' (partial run)'}")
        say(f"    {'method':<32} " + "  ".join(f"{c:>8}" for c in cols) + "   paper")
        for name in f["method"].drop_duplicates():
            got = f[f["method"] == name][cols].mean()
            vals = [100 * got[c] if c == "accuracy" else got[c] for c in cols]
            txt = [f"{v:.{2 if c == 'accuracy' else 3}f}" for v, c in zip(vals, cols)]
            paper = [f"{v:.{2 if c == 'accuracy' else 3}f}" for v, c in zip(PAPER[(unit, name)], cols)]
            same = txt == paper
            table_ok &= same or not full[unit]
            say(f"    {name:<32} " + "  ".join(f"{t:>8}" for t in txt) +
                f"   {'same' if same else ('paper: ' + ' '.join(paper)) if full[unit] else '-'}")

    say("\nFOLD-BY-FOLD CHECK against the paper's stored results")
    rule_identical, model_diff, checked = True, {}, False
    for unit, f in frames.items():
        ref = stored_reference(unit)
        if ref is None:
            say(f"  {unit}: the paper's stored results were not found; only the printed means above were compared")
            continue
        checked = True
        mrg = f.merge(ref, on=["fold", "method"], suffixes=("", "_paper"))
        for name in f["method"].drop_duplicates():
            g = mrg[mrg["method"] == name]
            worst = {c: float(np.max(np.abs(g[c] - g[f"{c}_paper"]))) for c in METRICS
                     if c in g and f"{c}_paper" in g and len(g) and g[c].notna().all()}
            tol = {c: TOL_GPBOOST_AUC if (name == "GPBoost" and c in ("roc_auc", "pr_auc")) else TOL for c in worst}
            same = len(g) == (f["method"] == name).sum() and all(worst[c] <= tol[c] for c in worst)
            big = max(worst, key=worst.get) if worst else "-"
            if name == RULE:
                rule_identical &= same
            else:
                model_diff[(unit, name)] = 0.0 if same else worst.get(big, float("nan"))
            say(f"  {'IDENTICAL' if same else 'DIFFERENT':<9} {unit:<9} {name:<32} {len(g)} fold(s), largest difference "
                f"{worst.get(big, float('nan')):.1e} ({big})")

    say("\nCLAIM CHECK: on every honest evaluation the rule beats each model in every fold on accuracy, F1 and MCC")
    claim_ok, n_cmp = True, 0
    for unit, f in frames.items():
        r = f[f["method"] == RULE].set_index("fold")
        for name in f["method"].drop_duplicates():
            if name == RULE:
                continue
            m = f[f["method"] == name].set_index("fold")
            wins = all((r.loc[m.index, c] > m[c]).all() for c in ("accuracy", "f1", "mcc"))
            if unit in HONEST:
                claim_ok &= wins
                n_cmp += 1
            if not wins:
                say(f"  {'FAILS' if unit in HONEST else 'note '} {unit}: {name} is not beaten in every fold"
                    + ("" if unit in HONEST else " (leaky row split: a model can memorise asteroids it has seen)"))
    say(f"  {'PASS' if claim_ok else 'FAIL'}: {n_cmp} model-unit comparisons on honest evaluations")

    say("\nEXACT PAIRED McNEMAR TEST on the grouped folds, from the row-level predictions")
    preds = [pd.read_csv(OUT / f"predictions_{u}.csv.gz") for u in ("E-grouped", "C-grouped")
             if (OUT / f"predictions_{u}.csv.gz").exists()]
    paired_ok = None
    if preds:
        p = preds[0]
        for q in preds[1:]:                        # same rows in the same order: identical folds over one table
            assert (q["des"].values == p["des"].values).all() and (q["fold"].values == p["fold"].values).all()
            p = pd.concat([p, q.drop(columns=["des", "fold", "label", RULE])], axis=1)
        y, rp = p["label"].to_numpy(), p[RULE].to_numpy()
        rule_wrong = rp != y
        bp = locate("sota_evidence.json", "results")
        bound = {m["model"]: m for m in json.loads(bp.read_text())["paired"]["models"]} if bp else {}
        say(f"  {len(p):,} rows, {p['fold'].nunique()} fold(s); b = model wrong and rule right, c = rule wrong and "
            "model right; p is one-sided")
        say(f"  {'method':<32} {'errors':>7} {'missed':>7} {'false+':>7}   {'b / c (all rows)':>17} {'log10 p':>9}   "
            f"{'b / c (PHA rows)':>17} {'log10 p':>9}")
        fn, fp = int(((y == 1) & (rp == 0)).sum()), int(((y == 0) & (rp == 1)).sum())
        say(f"  {RULE:<32} {fn + fp:>7,} {fn:>7,} {fp:>7,}")
        res, paired_ok, worst_p, worst_pm, counts_same = {}, True, -math.inf, -math.inf, True
        for name in [c for c in p.columns if c not in ("des", "fold", "label", RULE)]:
            pr = p[name].to_numpy()
            fn, fp = int(((y == 1) & (pr == 0)).sum()), int(((y == 0) & (pr == 1)).sum())
            wrong = pr != y
            b, c = int((wrong & ~rule_wrong).sum()), int((rule_wrong & ~wrong).sum())
            pos = y == 1
            bm, cm = int((pos & (pr == 0) & (rp == 1)).sum()), int((pos & (rp == 0) & (pr == 1)).sum())
            le, lm = log10_mcnemar(b, c), log10_mcnemar(bm, cm)
            worst_p, worst_pm = max(worst_p, le), max(worst_pm, lm)
            paired_ok &= b > c and bm > cm
            res[name] = dict(errors=fn + fp, missed=fn, false_alarms=fp, b_errors=b, c_errors=c, log10_p_errors=le,
                             b_missed=bm, c_missed=cm, log10_p_missed=lm)
            say(f"  {name:<32} {fn + fp:>7,} {fn:>7,} {fp:>7,}   {f'{b:,} / {c:,}':>17} {le:>9.1f}   "
                f"{f'{bm:,} / {cm:,}':>17} {lm:>9.1f}")
            if name in bound and p["fold"].nunique() == N_FOLDS:
                bd = bound[name]
                same = bd["fn"] == fn and bd["fp"] == fp
                counts_same &= same
                if same:                           # then the actual p cannot exceed the least-favourable bound
                    paired_ok &= le <= bd["log10_p_errors"] + 1e-9 and lm <= bd["log10_p_missed"] + 1e-9
        say(f"  every model makes more errors and misses more PHA approaches than the rule; every p < "
            f"10^{math.ceil(worst_p)} (all rows) and < 10^{math.ceil(worst_pm)} (PHA rows)")
        if bound and p["fold"].nunique() == N_FOLDS:
            if counts_same:
                say("  the confusion counts that the paper decoded from its stored metrics (Table X) equal these "
                    "predictions for every model, and every actual p is at or below the paper's least-favourable bound")
            else:
                say("  for some models these counts differ from those in the paper's Table X (expected on other "
                    "processors or library versions), so the paper's bound is not compared")
        (OUT / "paired_tests.json").write_text(json.dumps(res, indent=1))
    else:
        say("  run E-grouped and/or C-grouped to get the row-level predictions")

    say("\nVERDICT")
    say(f"  {'PASS' if rule_identical and checked else ('-   ' if not checked else 'FAIL')} our training-free rule "
        f"gives the paper's numbers in every fold" + ("" if checked else " (no stored results to compare with)"))
    diffs = {k: v for k, v in model_diff.items() if v > 0}
    if checked and not diffs:
        say("  PASS every trained model gives the paper's numbers in every fold")
    elif diffs:
        say(f"  NOTE {len(diffs)} model-unit result(s) differ from the paper by at most {max(diffs.values()):.3g}. The "
            "paper's runs used macOS arm64 and the versions in requirements.txt; other processors and library versions "
            "change trained models slightly, while the rule stays identical")
    say(f"  {'PASS' if claim_ok else 'FAIL'} on every honest evaluation the rule beats every model in every fold "
        "(accuracy, F1 and MCC)")
    if paired_ok is not None:
        say(f"  {'PASS' if paired_ok else 'FAIL'} exact paired tests: the rule is more accurate and misses fewer PHA "
            f"approaches than every model (p < 10^{math.ceil(max(worst_p, worst_pm))})")
    complete = all(full.get(u, False) for u in UNITS)
    say(f"  {'all six units are complete' if complete else 'units run: ' + ', '.join(u for u in UNITS if u in frames)}"
        f"{'' if table_ok else '; some means differ from the printed tables'}")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    return checked and rule_identical and not diffs and claim_ok and paired_ok is not False and table_ok


def default_out():
    """comparison/output/ next to this file; /kaggle/working/comparison_output/ on Kaggle (input is read-only)."""
    return KAGGLE_WORKING / "comparison_output" if KAGGLE_WORKING.is_dir() else HERE / "output"


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--units", nargs="+", choices=UNITS, default=list(UNITS), help="default: all six")
    ap.add_argument("--max-folds", type=int, default=None, help="quick smoke test: only the first K folds")
    ap.add_argument("--report", action="store_true", help="only print the report from the saved outputs")
    ap.add_argument("--data", type=Path, default=None, help="an extra folder to search for the data files")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: comparison/output/)")
    ap.add_argument("--strict", action="store_true",
                    help="exit with an error unless every fold matches the paper and every check passes")
    ap.add_argument("--no-report", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    OUT = args.out or default_out()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.data:
        EXTRA_ROOTS.insert(0, args.data)
    print(f"outputs: {OUT}", flush=True)
    if not args.report:
        if len(args.units) == 1:
            run_unit(args.units[0], load_catalogue(), args.max_folds)
        else:   # one process per unit, so GPBoost and LightGBM/XGBoost never share a process
            failed = []
            for unit in args.units:
                cmd = [sys.executable, str(Path(__file__).resolve()), "--units", unit, "--out", str(OUT), "--no-report"]
                cmd += ["--max-folds", str(args.max_folds)] if args.max_folds else []
                cmd += ["--data", str(args.data)] if args.data else []
                if subprocess.run(cmd).returncode != 0:
                    failed.append(unit)
            if failed:
                print(f"\nUNITS THAT FAILED: {', '.join(failed)} (see the messages above)\n", flush=True)
    if not args.no_report and not report() and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    if CNN_WORKER:
        cnn_worker(sys.argv[2], sys.argv[3])
    else:
        main()
