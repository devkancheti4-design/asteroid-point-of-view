"""Orbit-uncertainty experiment: hazard classification when the path is not exactly known.

Ground truth: asteroids with well-determined orbits (JPL condition code 0-2). Their true label is the PHA
definition applied to the exact path geometry of the true orbit.

Simulated discovery-time orbits: each true orbit is perturbed with a realistic uncertainty profile, i.e. the
1-sigma element uncertainties of a randomly drawn poorly determined NEO (condition code 5-7), scaled by a
multiplier. Errors are independent Gaussians per element (a limitation: real errors are correlated).

Methods, all seeing only the perturbed orbit, its sigmas and H:
  point rule (exact)      PHA definition on the exact path geometry of the perturbed orbit
  point rule (closed)     same with the tangent-line closed form
  path cloud (ours)       sample clones from N(perturbed, sigma), P(hazard) = share of clones with index <= 1;
                          threshold chosen on the training folds
  LightGBM                learned from perturbed orbit + sigmas + kernels
  LightGBM + cloud        the same plus the path-cloud probability as a feature
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

from common import N_FOLDS, RESULTS, SEED, binary_metrics, load_sbdb, save_json, summarize_folds
from asteroid_view import hazard_index
from orbit_geometry import moid_earth, node_geometry, opik_variables

SCALES = [0.25, 0.5, 1.0, 2.0]
N_CLONES = 48
N_GRID = 1024
PROFILE_CAPS = {"rel_a": 1.0, "e": 0.5, "i": 30.0, "om": 90.0, "w": 90.0}


def perturb(a, e, i, om, w, S, eps):
    a2 = np.clip(a * (1.0 + S[..., 0] * eps[..., 0]), 0.3, 50.0)
    e2 = np.clip(e + S[..., 1] * eps[..., 1], 0.0, 0.995)
    i2 = np.abs(i + S[..., 2] * eps[..., 2])
    i2 = np.where(i2 > 180.0, 360.0 - i2, i2)
    om2 = np.mod(om + S[..., 3] * eps[..., 3], 360.0)
    w2 = np.mod(w + S[..., 4] * eps[..., 4], 360.0)
    return a2, e2, i2, om2, w2


def features(a, e, i, om, w, H, S) -> pd.DataFrame:
    f = {"a": a, "e": e, "i": i, "q": a * (1 - e), "ad": a * (1 + e), "H": H,
         "sin_om": np.sin(np.deg2rad(om)), "cos_om": np.cos(np.deg2rad(om)),
         "sin_w": np.sin(np.deg2rad(w)), "cos_w": np.cos(np.deg2rad(w)),
         "sig_rel_a": S[:, 0], "sig_e": S[:, 1], "sig_i": S[:, 2], "sig_om": S[:, 3], "sig_w": S[:, 4]}
    f.update(node_geometry(a, e, i, om, w))
    f.update(opik_variables(a, e, i))
    f["cos_theta"] = np.nan_to_num(f["cos_theta"])
    f["moid_point"] = moid_earth(a, e, i, om, w, n_grid=N_GRID)
    return pd.DataFrame(f)


def main(pool_codes: tuple[int, int] = (5, 7), scales: list[float] = SCALES, tag: str = "cc5-7") -> None:
    t_start = time.time()
    df = load_sbdb()
    df = df[df["moid"].notna()].reset_index(drop=True)
    truth = df[df["condition_code"] <= 2].reset_index(drop=True)
    pool = df[df["condition_code"].between(*pool_codes)]
    prof = pd.DataFrame({"rel_a": pool["sigma_a"] / pool["a"], "e": pool["sigma_e"], "i": pool["sigma_i"],
                         "om": pool["sigma_om"], "w": pool["sigma_w"]}).dropna()
    capped = {k: float((prof[k] > v).mean()) for k, v in PROFILE_CAPS.items()}
    for k, v in PROFILE_CAPS.items():
        prof[k] = prof[k].clip(upper=v)

    a, e, i, om, w, H = (truth[c].to_numpy(float) for c in ("a", "e", "i", "om", "w", "H"))
    moid_true = moid_earth(a, e, i, om, w)
    y = (hazard_index(moid_true, H) <= 1.0).astype(int)
    meta = {"profile_condition_codes": list(pool_codes), "n_truth": int(len(truth)), "n_truth_pha": int(y.sum()),
            "n_profiles": int(len(prof)), "profile_median": prof.median().to_dict(), "profile_share_capped": capped,
            "n_clones": N_CLONES, "scales": list(scales)}
    print(meta, flush=True)

    rng = np.random.default_rng(SEED)
    P = prof.to_numpy()[rng.integers(0, len(prof), len(truth))]
    eps_obs = rng.standard_normal((len(truth), 5))
    eps_clone = rng.standard_normal((len(truth), N_CLONES, 5))
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))

    rows, flips = [], {}
    for scale in scales:
        t0 = time.time()
        S = P * scale
        ao, eo, io, omo, wo = perturb(a, e, i, om, w, S, eps_obs)
        X = features(ao, eo, io, omo, wo, H, S)
        point_exact = (hazard_index(X["moid_point"].values, H) <= 1.0).astype(int)
        point_closed = (hazard_index(X["d_lin"].values, H) <= 1.0).astype(int)
        flips[scale] = float((point_exact != y).mean())

        rep = lambda v: np.repeat(v[:, None], N_CLONES, axis=1)  # noqa: E731
        ac, ec, ic, omc, wc = perturb(rep(ao), rep(eo), rep(io), rep(omo), rep(wo), S[:, None, :], eps_clone)
        mc = moid_earth(ac.ravel(), ec.ravel(), ic.ravel(), omc.ravel(), wc.ravel(), n_grid=N_GRID)
        p_cloud = (hazard_index(mc.reshape(len(truth), N_CLONES), H[:, None]) <= 1.0).mean(axis=1)
        X["p_cloud"] = p_cloud
        print(f"scale {scale}: features + {N_CLONES} clones/asteroid in {time.time() - t0:.0f}s; "
              f"point-rule label flips vs truth: {flips[scale]:.4f}", flush=True)

        feat_ml = [c for c in X.columns if c != "p_cloud"]
        for k, (tr, te) in enumerate(folds):
            fold_key = {"scale": scale, "fold": k}
            for name, pred in (("point rule (exact geometry)", point_exact), ("point rule (closed form)", point_closed)):
                m = binary_metrics(y[te], pred[te], pred[te].astype(float))
                rows.append({**fold_key, "method": name, **m, "brier": float(np.mean((pred[te] - y[te]) ** 2))})
            # path cloud: choose the probability threshold on the training fold only
            taus = np.linspace(0.02, 0.98, 49)
            acc_tr = [((p_cloud[tr] >= t).astype(int) == y[tr]).mean() for t in taus]
            tau = float(taus[int(np.argmax(acc_tr))])
            pred = (p_cloud[te] >= tau).astype(int)
            rows.append({**fold_key, "method": "path cloud (ours)", **binary_metrics(y[te], pred, p_cloud[te]),
                         "brier": float(np.mean((p_cloud[te] - y[te]) ** 2)), "tau": tau})
            for name, cols in (("LightGBM", feat_ml), ("LightGBM + path cloud (ours)", feat_ml + ["p_cloud"])):
                model = LGBMClassifier(n_estimators=500, num_leaves=63, learning_rate=0.05, random_state=SEED, verbose=-1)
                model.fit(X.iloc[tr][cols].values, y[tr])
                prob = model.predict_proba(X.iloc[te][cols].values)[:, 1]
                rows.append({**fold_key, "method": name, **binary_metrics(y[te], (prob >= 0.5).astype(int), prob),
                             "brier": float(np.mean((prob - y[te]) ** 2))})

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"uncertainty_{tag}_folds.csv", index=False)
    summary = summarize_folds(out.drop(columns=["tau"], errors="ignore"), ["scale", "method"])
    summary.to_csv(RESULTS / f"uncertainty_{tag}_summary.csv", index=False)
    meta["point_rule_label_flip_rate"] = flips
    meta["runtime_s"] = time.time() - t_start
    save_json(meta, RESULTS / f"uncertainty_{tag}_meta.json")
    print(summary[["scale", "method", "accuracy_mean", "f1_mean", "recall_mean", "mcc_mean", "brier_mean"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--codes", default="5-7", help="condition-code range of the uncertainty profiles, e.g. 5-7 or 8-9")
    ap.add_argument("--scales", default=",".join(str(s) for s in SCALES))
    args = ap.parse_args()
    lo, hi = (int(v) for v in args.codes.split("-"))
    main((lo, hi), [float(s) for s in args.scales.split(",")], tag=f"cc{lo}-{hi}")
