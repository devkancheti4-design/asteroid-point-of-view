"""The Asteroid's Point of View: hazard from path geometry (Ch. Satwik, Devieswar Kancheti).

Single-cell version of notebooks/asteroid_point_of_view_kaggle.ipynb: paste all of it into ONE Kaggle code
cell and run it. Attach datasets/neo_datasets_2026-11-11.zip (sbdb_neo.csv, cad_earth.csv) and, optionally,
cad.csv, sentry_risk_list.csv and scout.csv; the files are found anywhere under /kaggle/input. Without
sbdb_neo.csv / cad_earth.csv the code downloads them from JPL if the notebook's Internet setting is on.
"""

# ====================================================================================================
# The Asteroid's Point of View: hazard from path geometry
# ====================================================================================================

import glob
import math
import os
import time

import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

try:
    display
except NameError:  # running as a plain script
    display = print
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)

SEED, N_FOLDS = 20261111, 5            # the paper's seed and number of folds
MOID_LIMIT_AU, H_LIMIT = 0.05, 22.0    # official PHA definition (JPL/CNEOS): MOID <= 0.05 au and H <= 22
AU_KM = 149_597_870.7

RUN_EARTH_VIEW_BASELINES = True        # section 4: ML on close-approach rows, leaky vs honest split
BASELINES = ["RandomForest", "XGBoost", "LightGBM"]  # add "LogReg", "KNN", "MLP" for the paper's full table
RUN_UNCERTAINTY = True                 # section 6: uncertain orbits, point rule vs path cloud
DOWNLOAD_IF_MISSING = True             # fetch sbdb_neo.csv / cad_earth.csv from JPL if not attached (Internet on)

INPUT = os.environ.get("NEO_INPUT", "/kaggle/input")
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
INPUT_DIRS = [INPUT, os.path.join(WORK, "unzipped"), os.path.join(WORK, "jpl_download")]


def find(name):
    """First file with this name anywhere under the input folders."""
    for root in INPUT_DIRS:
        hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True))
        if hits:
            return hits[0]
    return None


def unpack_zips():
    """Kaggle normally unpacks uploaded zips; unpack any that are still zipped into the working folder."""
    import zipfile
    for z in glob.glob(os.path.join(INPUT, "**", "*.zip"), recursive=True):
        dest = os.path.join(WORK, "unzipped", os.path.splitext(os.path.basename(z))[0])
        if not os.path.isdir(dest):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
            print(f"unpacked {z}")


def jpl_get(url, params, retries=4):
    import requests
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=600)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # network errors and bad JSON
            if attempt == retries - 1:
                raise
            print(f"  JPL request failed ({exc}); retrying")
            time.sleep(5 * 2**attempt)


def download(name):
    """The same JPL queries that built the paper's snapshot (src/fetch_data.py), on today's catalogue."""
    out = os.path.join(WORK, "jpl_download", name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if name == "sbdb_neo.csv":
        fields = ["spkid", "pdes", "full_name", "neo", "pha", "class", "H", "diameter", "diameter_sigma", "albedo",
                  "epoch", "e", "a", "q", "i", "om", "w", "ma", "ad", "n", "tp", "per", "moid", "moid_jup", "t_jup",
                  "condition_code", "data_arc", "first_obs", "last_obs", "n_obs_used", "rms", "sigma_e", "sigma_a",
                  "sigma_q", "sigma_i", "sigma_om", "sigma_w", "sigma_ma", "sigma_tp", "sigma_n", "sigma_per",
                  "sigma_ad", "two_body", "A1", "A2", "A3"]
        js = jpl_get("https://ssd-api.jpl.nasa.gov/sbdb_query.api",
                     {"fields": ",".join(fields), "sb-group": "neo", "sb-kind": "a", "full-prec": "true"})
        df = pd.DataFrame(js["data"], columns=js["fields"])
    else:  # cad_earth.csv: Earth approaches within 0.5 au, 1910-2024, in 5-year chunks
        frames = []
        for y0 in range(1910, 2025, 5):
            js = jpl_get("https://ssd-api.jpl.nasa.gov/cad.api",
                         {"date-min": f"{y0}-01-01", "date-max": f"{min(y0 + 4, 2024)}-12-31", "dist-max": "0.5",
                          "body": "Earth", "neo": "true", "fullname": "true"})
            if int(js.get("count", 0)):
                frames.append(pd.DataFrame(js["data"], columns=js["fields"]))
        df = pd.concat(frames, ignore_index=True)
    df.to_csv(out, index=False)
    print(f"downloaded {name} from JPL: {len(df):,} rows")
    return out


unpack_zips()
NAMES = ("sbdb_neo.csv", "cad_earth.csv", "cad.csv", "sentry_risk_list.csv", "scout.csv")
FILES = {n: find(n) for n in NAMES}
if FILES["sbdb_neo.csv"] is None:
    attached = [p for p in glob.glob(os.path.join(INPUT, "**", "*"), recursive=True) if os.path.isfile(p)]
    print(f"sbdb_neo.csv is not attached. Files under {INPUT}: {attached[:20] or 'none'}")
FRESH = False
for n in ("sbdb_neo.csv", "cad_earth.csv"):
    if FILES[n] is None and DOWNLOAD_IF_MISSING:
        try:
            FILES[n], FRESH = download(n), True
        except Exception as exc:
            print(f"could not download {n} from JPL ({exc})")
if FILES["sbdb_neo.csv"] is None:
    raise FileNotFoundError(
        "sbdb_neo.csv is needed. Either attach datasets/neo_datasets_2026-11-11.zip from the repository "
        "(Add data -> Upload; it holds sbdb_neo.csv and cad_earth.csv), or turn on Internet in the notebook "
        "settings so the catalogue can be downloaded from JPL.")
for n, p in FILES.items():
    print(f"{n:22s} {p or '- not attached, its section is skipped'}")
if FRESH:
    print("\nNOTE: data downloaded from JPL today, not the paper's 2026-11-11 snapshot: JPL updates its catalogue "
          "daily, so the numbers will differ slightly from the [paper: ...] values.")

# ====================================================================================================
# 1. The method
# ====================================================================================================

# ---- Earth's orbit as a fixed curve, and the asteroid's path (identical to src/orbit_geometry.py) ----
EARTH_A = 1.00000261                     # au
EARTH_E = 0.01671123
EARTH_VARPI = np.deg2rad(102.93768193)   # longitude of perihelion
EARTH_P = EARTH_A * (1.0 - EARTH_E**2)   # semi-latus rectum, au
TWO_PI = 2.0 * np.pi
GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0


def earth_radius(lam):
    """Heliocentric distance (au) of Earth's orbit at ecliptic longitude lam (rad)."""
    return EARTH_P / (1.0 + EARTH_E * np.cos(lam - EARTH_VARPI))


def orbit_basis(i, om, w):
    """Perifocal unit vectors P (towards perihelion) and Q (90 deg ahead) in the ecliptic frame (radians in)."""
    ci, si = np.cos(i), np.sin(i)
    co, so = np.cos(om), np.sin(om)
    cw, sw = np.cos(w), np.sin(w)
    P = np.stack([cw * co - sw * so * ci, cw * so + sw * co * ci, sw * si], axis=-1)
    Q = np.stack([-sw * co - cw * so * ci, -sw * so + cw * co * ci, cw * si], axis=-1)
    return P, Q


def path_points(p, e, P, Q, f):
    """Heliocentric ecliptic positions on the path at true anomalies f; shapes (n,), (n,3), (n,m) -> (n,m,3)."""
    r = p[:, None] / (1.0 + e[:, None] * np.cos(f))
    return r[..., None] * (np.cos(f)[..., None] * P[:, None, :] + np.sin(f)[..., None] * Q[:, None, :])


def distance_to_earth_orbit(X, newton_iters=4):
    """Exact distance (au) from points X (..., 3) to Earth's orbit: Newton steps in Earth's true longitude."""
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
    """Exact path-to-path minimum distance delta (au) between each asteroid's orbit and Earth's orbit."""
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
        lo, hi = f_grid[idx] - h, f_grid[idx] + h

        def dist_at(fv):
            return distance_to_earth_orbit(path_points(p, e[sl], P, Q, fv))

        c = hi - GOLDEN * (hi - lo)            # vectorized golden-section search on each candidate bracket
        d_pt = lo + GOLDEN * (hi - lo)
        fc, fd = dist_at(c), dist_at(d_pt)
        for _ in range(golden_iters):
            left = fc < fd
            hi = np.where(left, d_pt, hi)
            lo = np.where(left, lo, c)
            new_c = np.where(left, hi - GOLDEN * (hi - lo), d_pt)
            new_d = np.where(left, c, lo + GOLDEN * (hi - lo))
            val = dist_at(np.where(left, new_c, new_d))
            fc, fd = np.where(left, val, fd), np.where(left, fc, val)
            c, d_pt = new_c, new_d
        refined = np.where(valid, np.minimum(fc, fd), np.inf)
        out[sl] = np.minimum(refined.min(axis=1), d.min(axis=1))
    return out


def node_geometry(a, e, i_deg, om_deg, w_deg):
    """Closed forms at the nodes: radial gaps, node-gap upper bound d_node and tangent-line estimate d_lin."""
    a, e = np.asarray(a, float), np.asarray(e, float)
    i, om, w = (np.deg2rad(np.asarray(v, float)) for v in (i_deg, om_deg, w_deg))
    p = a * (1.0 - e**2)
    out, d_lin_nodes = {}, []
    for name, f_node, lam in (("asc", -w, om), ("desc", np.pi - w, om + np.pi)):
        r = p / (1.0 + e * np.cos(f_node))
        dr = r - earth_radius(lam)
        k_a = e * np.sin(f_node) / (1.0 + e * np.cos(f_node))       # asteroid v_r / v_theta at the node
        F = lam - EARTH_VARPI
        k_e = EARTH_E * np.sin(F) / (1.0 + EARTH_E * np.cos(F))     # Earth's v_r / v_theta there
        si, ci = np.abs(np.sin(i)), np.cos(i)
        denom = np.sqrt(si**2 * (1.0 + k_e**2) + (k_a - k_e * ci) ** 2)
        d_lin = np.abs(dr) * si / np.maximum(denom, 1e-300)
        out[f"r_{name}"], out[f"dr_{name}"], out[f"d_lin_{name}"] = r, dr, d_lin
        d_lin_nodes.append(d_lin)
    out["d_node"] = np.minimum(np.abs(out["dr_asc"]), np.abs(out["dr_desc"]))
    out["d_lin"] = np.minimum(*d_lin_nodes)
    return out

# ---- the training-free rule and the evaluation metrics ----
def hazard_index(delta_au, H):
    """Phi = max(delta / 0.05 au, 10**((H - 22) / 5)); PHA if and only if Phi <= 1."""
    return np.maximum(np.asarray(delta_au, float) / MOID_LIMIT_AU, 10.0 ** ((np.asarray(H, float) - H_LIMIT) / 5.0))


def rule(delta_au, H):
    """PHA prediction (0/1) and a ranking score -log(Phi) (higher = more hazardous)."""
    phi = hazard_index(delta_au, H)
    return (phi <= 1.0).astype(int), -np.log(phi)


def binary_metrics(y_true, y_pred, y_score=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {"accuracy": skm.accuracy_score(y_true, y_pred),
           "balanced_accuracy": skm.balanced_accuracy_score(y_true, y_pred),
           "precision": skm.precision_score(y_true, y_pred, zero_division=0),
           "recall": skm.recall_score(y_true, y_pred, zero_division=0),
           "f1": skm.f1_score(y_true, y_pred, zero_division=0),
           "mcc": skm.matthews_corrcoef(y_true, y_pred)}
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = skm.roc_auc_score(y_true, y_score)
        out["pr_auc"] = skm.average_precision_score(y_true, y_score)
    return out


def summarize(folds, by):
    """Mean and standard deviation of every metric across folds."""
    cols = [c for c in folds.columns if c not in by + ["fold"] and folds[c].dtype.kind in "fc"]
    agg = folds.groupby(by, sort=False)[cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


SHOW = ["accuracy_mean", "f1_mean", "mcc_mean", "recall_mean", "precision_mean"]

# ====================================================================================================
# 2. Every near-Earth asteroid's path geometry
# ====================================================================================================

sbdb = pd.read_csv(FILES["sbdb_neo.csv"], low_memory=False, dtype={"pdes": str})
sbdb["pdes"] = sbdb["pdes"].str.strip()
for c in ["H", "a", "e", "i", "om", "w", "moid", "condition_code", "sigma_a", "sigma_e", "sigma_i", "sigma_om",
          "sigma_w"]:
    sbdb[c] = pd.to_numeric(sbdb[c], errors="coerce")
sbdb["label"] = sbdb["pha"].map({"Y": 1, "N": 0})                       # JPL's PHA flag
sbdb["first_obs"] = pd.to_datetime(sbdb["first_obs"], errors="coerce")
labelled = sbdb[sbdb["label"].notna() & sbdb["H"].notna()].reset_index(drop=True)
labelled["label"] = labelled["label"].astype(int)
neo = labelled[labelled["moid"].notna()].reset_index(drop=True)

t0 = time.time()
neo["delta"] = moid_earth(neo["a"], neo["e"], neo["i"], neo["om"], neo["w"])
for k, v in node_geometry(neo["a"], neo["e"], neo["i"], neo["om"], neo["w"]).items():
    neo[k] = v
print(f"{len(neo):,} near-Earth asteroids ({neo['label'].sum():,} PHAs): path-to-path distance for all of them "
      f"in {time.time() - t0:.1f} s   [paper: 42,888 asteroids, 2,551 PHAs]")

diff = (neo["delta"] - neo["moid"]).abs()
ours, _ = rule(neo["delta"], neo["H"])
jpl, _ = rule(neo["moid"], neo["H"])
print(f"|delta - JPL MOID|: median {diff.median():.2e} au, 99th percentile {diff.quantile(0.99):.2e} au, "
      f"max {diff.max():.2e} au   [paper: 1.03e-04, 6.03e-04, 1.83e-03]")
print(f"same PHA verdict from our delta as from JPL's MOID: {np.mean(ours == jpl):.3%}   [paper: 99.979%]")

# ====================================================================================================
# 3. All near-Earth asteroids: 5 folds over asteroids, and new discoveries
# ====================================================================================================

RULES = {"Ours: exact path geometry": "delta",
         "Closed form: tangent line": "d_lin",
         "Upper bound: node gap": "d_node",
         "Reference: the definition on JPL's own MOID": "moid"}
rows = []
for k, (tr, te) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(neo, neo["label"])):
    t = neo.iloc[te]
    for name, col in RULES.items():
        pred, score = rule(t[col], t["H"])
        rows.append({"method": name, "fold": k, **binary_metrics(t["label"], pred, score)})
kfold = summarize(pd.DataFrame(rows), ["method"])
display(kfold[["method"] + SHOW].round(5))
print("paper: ours accuracy 0.99865, F1 0.98859, MCC 0.98788; tangent line 0.99263 / 0.93811")

new = neo[neo["first_obs"] >= "2016-01-01"]
temporal = pd.DataFrame([{"method": name, **binary_metrics(new["label"], *rule(new[col], new["H"]))}
                         for name, col in RULES.items()])
print(f"\nasteroids first seen in 2016 or later: {len(new):,} ({new['label'].sum():,} PHAs)   [paper: 27,817 (682)]")
display(temporal[["method", "accuracy", "f1", "mcc", "recall", "precision"]].round(5))
print("paper: ours accuracy 0.99942 (27,801 of 27,817), F1 0.98825")

# ====================================================================================================
# 4. Earth's view vs the asteroid's view on NASA CAD
# ====================================================================================================

def make_model(name):
    """The paper's Earth's-view baselines (src/earth_view.py)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "LogReg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if name == "KNN":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                             n_jobs=-1, random_state=SEED, eval_metric="logloss")
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=400, num_leaves=63, learning_rate=0.05, random_state=SEED, verbose=-1)
    if name == "MLP":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=1024,
                                                             early_stopping=True, n_iter_no_change=5, max_iter=60,
                                                             random_state=SEED))
    raise ValueError(name)


if FILES["cad_earth.csv"]:
    cad = pd.read_csv(FILES["cad_earth.csv"], low_memory=False, dtype={"des": str})
    cad["des"] = cad["des"].str.strip()
    ev = cad.merge(labelled[["pdes", "label", "H"]], left_on="des", right_on="pdes", how="inner").reset_index(drop=True)
    base = 10.0 ** (-ev["H"] / 5.0)                                     # NASA NeoWs feature set
    ev["absolute_magnitude"] = ev["H"]
    ev["est_diameter_min"], ev["est_diameter_max"] = 1329.0 / np.sqrt(0.25) * base, 1329.0 / np.sqrt(0.05) * base
    ev["relative_velocity"] = pd.to_numeric(ev["v_rel"]) * 3600.0       # km/h
    ev["miss_distance"] = pd.to_numeric(ev["dist"]) * AU_KM             # km
    FEATURES = ["absolute_magnitude", "est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance"]
    X, y, groups = ev[FEATURES].to_numpy(float), ev["label"].to_numpy(int), ev["des"].to_numpy()
    print(f"{len(ev):,} close approaches of {ev['des'].nunique():,} asteroids, {y.sum():,} PHA rows   "
          f"[paper: 306,590 approaches of 36,817 asteroids]")
    splits = {
        "row split (prior work)": list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y)),
        "asteroid-grouped (honest)": list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
                                          .split(X, y, groups)),
    }
    for proto, folds in splits.items():
        seen = np.mean([np.isin(groups[te], np.unique(groups[tr])).mean() for tr, te in folds])
        print(f"{proto:26s}: {seen:.1%} of test rows belong to an asteroid that is also in training")

    feat = neo.set_index("pdes")
    rows = []
    for proto, folds in splits.items():
        for k, (tr, te) in enumerate(folds):
            t = ev.iloc[te]
            t = t[t["des"].isin(feat.index)]
            f = feat.loc[t["des"]]
            pred, score = rule(f["delta"].to_numpy(), f["H"].to_numpy())
            rows.append({"protocol": proto, "fold": k, "method": "Ours: path rule (no training)",
                         **binary_metrics(t["label"].to_numpy(), pred, score)})
            if RUN_EARTH_VIEW_BASELINES:
                for name in BASELINES:
                    t0 = time.time()
                    model = make_model(name).fit(X[tr], y[tr])
                    s = model.predict_proba(X[te])[:, 1]
                    rows.append({"protocol": proto, "fold": k, "method": name,
                                 **binary_metrics(y[te], (s >= 0.5).astype(int), s)})
                    print(f"  {proto:26s} fold {k} {name:12s} {time.time() - t0:5.0f} s", flush=True)
    earth_view = summarize(pd.DataFrame(rows), ["protocol", "method"])
    display(earth_view[["protocol", "method"] + SHOW].round(4))
    print("paper, row split: random forest 0.9474 / F1 0.676 (inflated by leakage)\n"
          "paper, grouped: no baseline above 0.9263 accuracy (MLP) or F1 0.428 (random forest);\n"
          "               ours 0.9979 / F1 0.9878 on the same folds")
else:
    print("cad_earth.csv not attached: section skipped")

# ====================================================================================================
# 5. The NASA CAD, Sentry and Scout tables
# ====================================================================================================

feat = neo.set_index("pdes")
TABLES = {}


def score_table(label, designations):
    des = pd.Series(designations).astype(str).str.strip()
    known = des.isin(feat.index)
    f = feat.loc[des[known]]
    pred, score = rule(f["delta"].to_numpy(), f["H"].to_numpy())
    y = f["label"].to_numpy()
    m = binary_metrics(y, pred, score)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    print(f"{label}: {len(des):,} rows, {known.sum():,} scored (an orbit and H in JPL SBDB), {y.sum()} PHAs -> "
          f"TP {tp}, FP {fp}, FN {fn}, TN {tn}; accuracy {m['accuracy']:.4f}, F1 {m['f1']:.4f}")
    TABLES[label] = m
    return m


if FILES["cad.csv"]:
    score_table("CAD table", pd.read_csv(FILES["cad.csv"], dtype={"des": str})["des"])
if FILES["sentry_risk_list.csv"]:
    sentry = pd.read_csv(FILES["sentry_risk_list.csv"], dtype={"des": str})
    score_table("Sentry risk list", sentry["des"])
if FILES["scout.csv"]:
    scout = pd.read_csv(FILES["scout.csv"])
    flag = (scout["moid"] <= MOID_LIMIT_AU) & (scout["H"] <= H_LIMIT)
    print(f"Scout: {len(scout)} unconfirmed candidates, {int(scout['objectName'].astype(str).isin(feat.index).sum())}"
          f" of them in JPL SBDB (no ground truth). The rule on Scout's own MOID and H flags {int(flag.sum())}:")
    display(scout.loc[flag, ["objectName", "H", "moid", "phaScore"]])

# ====================================================================================================
# 6. Uncertain orbits: point rule vs path cloud
# ====================================================================================================

def perturb(a, e, i, om, w, S, eps):
    """Element perturbation, identical to src/uncertainty.py."""
    a2 = np.clip(a * (1.0 + S[..., 0] * eps[..., 0]), 0.3, 50.0)
    e2 = np.clip(e + S[..., 1] * eps[..., 1], 0.0, 0.995)
    i2 = np.abs(i + S[..., 2] * eps[..., 2])
    i2 = np.where(i2 > 180.0, 360.0 - i2, i2)
    om2 = np.mod(om + S[..., 3] * eps[..., 3], 360.0)
    w2 = np.mod(w + S[..., 4] * eps[..., 4], 360.0)
    return a2, e2, i2, om2, w2


if RUN_UNCERTAINTY:
    N_CLONES, N_GRID, SCALE = 48, 1024, 1.0
    CAPS = {"rel_a": 1.0, "e": 0.5, "i": 30.0, "om": 90.0, "w": 90.0}
    t_start = time.time()
    truth = neo[neo["condition_code"] <= 2].reset_index(drop=True)
    pool = neo[neo["condition_code"].between(8, 9)]
    prof = pd.DataFrame({"rel_a": pool["sigma_a"] / pool["a"], "e": pool["sigma_e"], "i": pool["sigma_i"],
                         "om": pool["sigma_om"], "w": pool["sigma_w"]}).dropna()
    for k, v in CAPS.items():
        prof[k] = prof[k].clip(upper=v)
    a, e, i, om, w, H = (truth[c].to_numpy(float) for c in ("a", "e", "i", "om", "w", "H"))
    y = (hazard_index(moid_earth(a, e, i, om, w), H) <= 1.0).astype(int)

    rng = np.random.default_rng(SEED)
    S = prof.to_numpy()[rng.integers(0, len(prof), len(truth))] * SCALE
    eps_obs = rng.standard_normal((len(truth), 5))
    eps_clone = rng.standard_normal((len(truth), N_CLONES, 5))
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))

    ao, eo, io, omo, wo = perturb(a, e, i, om, w, S, eps_obs)
    point = (hazard_index(moid_earth(ao, eo, io, omo, wo, n_grid=N_GRID), H) <= 1.0).astype(int)
    rep = lambda v: np.repeat(v[:, None], N_CLONES, axis=1)  # noqa: E731
    ac, ec, ic, omc, wc = perturb(rep(ao), rep(eo), rep(io), rep(omo), rep(wo), S[:, None, :], eps_clone)
    mc = moid_earth(ac.ravel(), ec.ravel(), ic.ravel(), omc.ravel(), wc.ravel(), n_grid=N_GRID)
    p_cloud = (hazard_index(mc.reshape(len(truth), N_CLONES), H[:, None]) <= 1.0).mean(axis=1)

    rows = []
    for k, (tr, te) in enumerate(folds):
        rows.append({"method": "point rule (perturbed orbit)", "fold": k,
                     **binary_metrics(y[te], point[te], point[te].astype(float)),
                     "brier": float(np.mean((point[te] - y[te]) ** 2))})
        taus = np.linspace(0.02, 0.98, 49)
        tau = float(taus[int(np.argmax([((p_cloud[tr] >= t).astype(int) == y[tr]).mean() for t in taus]))])
        pred = (p_cloud[te] >= tau).astype(int)
        rows.append({"method": "path cloud (ours)", "fold": k, **binary_metrics(y[te], pred, p_cloud[te]),
                     "brier": float(np.mean((p_cloud[te] - y[te]) ** 2))})
    unc = summarize(pd.DataFrame(rows), ["method"])
    print(f"{len(truth):,} well-determined asteroids ({y.sum():,} hazardous), {N_CLONES} clones each, "
          f"{time.time() - t_start:.0f} s")
    display(unc[["method", "accuracy_mean", "mcc_mean", "brier_mean"]].round(4))
    print("paper (short-arc profiles, scale 1): point rule 0.9846 / MCC 0.9489 / Brier 0.0154; "
          "path cloud 0.9846 / 0.9487 / 0.0105")

# ====================================================================================================
# Summary
# ====================================================================================================
