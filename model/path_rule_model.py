"""The training-free path rule: deterministic hazard classification of asteroids from their orbits.

From "The Asteroid's Point of View" (Ch. Satwik and Devieswar Kancheti). NASA/JPL calls an asteroid potentially
hazardous (PHA) when its orbit comes within 0.05 au of Earth's orbit (the MOID) and its absolute magnitude is
H <= 22 (roughly 140 m or larger). This model needs no training and no close-approach data:
  1. from the asteroid's own orbital elements (a, e, i, node, perihelion argument) compute the exact minimum
     distance between its path and Earth's orbit, both treated as fixed curves in space;
  2. hazard index = max(distance / 0.05 au, 10**((H - 22) / 5)); PHA if and only if the index is <= 1.
It never reads JPL's MOID or PHA fields, uses no random numbers and gives the same answer on any computer.
On all 42,888 labelled near-Earth asteroids it reaches 99.86% accuracy (F1 0.989) in about 12 seconds.

Use from Python (numpy arrays, angles in degrees as in JPL's Small-Body Database):
    from path_rule_model import predict
    pha, score, distance_au = predict(a, e, i, om, w, H)
Run the demo on the NASA/JPL catalogue (the repository's data snapshot, or a CSV given on the command line):
    python model/path_rule_model.py [sbdb.csv]
Requires numpy (and pandas for the demo). The geometry is the same code as src/orbit_geometry.py.
"""
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

MOID_LIMIT_AU, H_LIMIT = 0.05, 22.0     # the PHA definition of NASA/JPL CNEOS

# Earth's orbit: J2000 mean elements of the Earth-Moon barycentre (Standish, JPL SSD "Keplerian elements for
# approximate positions of the major planets", Table 1), a fixed Keplerian ellipse in the ecliptic.
EARTH_A = 1.00000261
EARTH_E = 0.01671123
EARTH_VARPI = np.deg2rad(102.93768193)
EARTH_P = EARTH_A * (1.0 - EARTH_E**2)
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



def predict(a, e, i_deg, om_deg, w_deg, H):
    """For arrays of orbital elements (angles in degrees) and H: PHA prediction (0/1), score, path distance (au)."""
    distance = moid_earth(a, e, i_deg, om_deg, w_deg)
    index = hazard_index(distance, H)
    return (index <= 1.0).astype(int), -np.log(index), distance


EXAMPLES = {"99942": "Apophis", "101955": "Bennu", "4179": "Toutatis", "433": "Eros", "1036": "Ganymed",
            "2024 YR4": "2024 YR4"}


def load_catalogue(path=None):
    """NASA/JPL near-Earth asteroids with a PHA flag and an H: a CSV path, or the repository's snapshot of 2026-11-11."""
    import pandas as pd
    root = Path(__file__).resolve().parents[1]
    if path is None:
        for p in (root / "data" / "processed" / "sbdb_neo.csv",
                  root / "1_Datasets" / "Labels_NASA_SBDB" / "sbdb_near_earth_asteroids.csv"):
            if p.is_file():
                path = p
                break
    if path is not None:
        df = pd.read_csv(path, low_memory=False, dtype={"pdes": str})
    else:   # a fresh clone: read the catalogue straight from the data snapshot zip in datasets/
        with zipfile.ZipFile(root / "datasets" / "neo_datasets_2026-11-11.zip") as z:
            with z.open("neo_datasets_2026-11-11/sbdb_neo.csv") as f:
                df = pd.read_csv(f, low_memory=False, dtype={"pdes": str})
    for c in ("H", "a", "e", "i", "om", "w"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["label"] = df["pha"].map({"Y": 1, "N": 0})
    return df[df["label"].notna() & df["H"].notna()].reset_index(drop=True)


def main():
    """Classify every labelled near-Earth asteroid and compare with NASA's PHA flag."""
    df = load_catalogue(sys.argv[1] if len(sys.argv) > 1 else None)
    y = df["label"].astype(int).to_numpy()
    t0 = time.time()
    pred, _, distance = predict(df.a.values, df.e.values, df.i.values, df.om.values, df.w.values, df.H.values)
    seconds = time.time() - t0
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    print(f"{len(y):,} near-Earth asteroids ({int(y.sum()):,} PHAs according to NASA/JPL), no training, {seconds:.1f} s")
    print(f"accuracy {100 * (tp + tn) / len(y):.2f}%   F1 {2 * tp / (2 * tp + fp + fn):.3f}   PHAs found {tp:,} of "
          f"{tp + fn:,}   missed {fn}   false alarms {fp} among {fp + tn:,} other asteroids\n")
    print(f"{'asteroid':<10} {'H':>6} {'path distance':>14}  {'ours':<12} NASA")
    for des, name in EXAMPLES.items():
        k = np.flatnonzero(df["pdes"].to_numpy() == des)
        if len(k):
            k = k[0]
            print(f"{name:<10} {df.H[k]:>6.2f} {distance[k]:>11.4f} au  {'PHA' if pred[k] else 'not a PHA':<12} "
                  f"{'PHA' if y[k] else 'not a PHA'}")


if __name__ == "__main__":
    main()
