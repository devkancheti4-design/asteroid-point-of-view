"""Numerical checks of every lemma and proposition proved in the paper's appendix.

A proof is the argument; these checks confirm that the implemented quantities obey the proved inequalities on
real and synthetic orbits (floating-point tolerance TOL), so a gap between proof and code would show up here.

  L1  radial gap:      L0 <= delta                                  all labelled NEAs
  L2  node gap:        delta <= delta_node                          all labelled NEAs
  L3  bracket:         L_lon <= delta <= U_lon, and its width       NEAs with delta < 0.2 au
  P1  miss distance:   min over time of |r_ast(t) - r_E(t)| >= delta, two-body propagation of synthetic orbits;
                       and, for context, JPL close approaches versus delta of the current osculating orbit
  P2  hazard index:    Phi <= 1  <=>  delta <= 0.05 au and H <= 22  all labelled NEAs
  T1  decision law:    every decided class agrees with its certificate; the procedure ends within six steps
Writes results/proof_checks.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "pha_core"))
import pha_kernels as K  # noqa: E402

from common import PROC, RESULTS, load_sbdb, save_json  # noqa: E402
from orbit_geometry import EARTH_A, EARTH_E, EARTH_VARPI, moid_earth  # noqa: E402

TOL = 1e-9  # au
ROOT_DATA = PROC.parent


def kepler_positions(a, e, i, om, w, M):
    """Heliocentric ecliptic positions for mean anomalies M (two-body), elements in au and degrees."""
    E = M.copy()
    for _ in range(30):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    f = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    r = a * (1 - e * e) / (1 + e * np.cos(f))
    i, om, w = np.deg2rad(i), np.deg2rad(om), np.deg2rad(w)
    u = w + f
    return np.stack([r * (np.cos(om) * np.cos(u) - np.sin(om) * np.sin(u) * np.cos(i)),
                     r * (np.sin(om) * np.cos(u) + np.cos(om) * np.sin(u) * np.cos(i)),
                     r * np.sin(u) * np.sin(i)], axis=-1)


def main() -> None:
    out = {}
    kr = pd.read_csv(RESULTS / "kernel_rows.csv")
    d = kr["moid_k"].to_numpy()

    # L1, L2: closed-form bounds on all labelled NEAs
    out["L1_radial_gap_violations"] = int((kr["L0"].to_numpy() > d + TOL).sum())
    out["L2_node_gap_violations"] = int((d > kr["U_node"].to_numpy() + TOL).sum())
    out["L1_L2_n"] = int(len(kr))

    # L3: same-longitude bracket, recomputed for every NEA with delta < 0.2 au
    sb = load_sbdb()
    sb = sb[sb["moid"].notna()].reset_index(drop=True)
    close = sb[d < 0.2]
    lo_v = hi_v = 0
    widths = []
    for _, row in close.iterrows():
        el = K.elements(row)
        b = K.lon_bounds(el)
        dk = d[_]
        lo_v += int(b["L_lon"] > dk + K.TOL_D)
        hi_v += int(dk > b["U_lon"] + K.TOL_D)
        if dk <= 0.05:
            widths.append(b["U_lon"] - b["L_lon"])
    out["L3_n"] = int(len(close))
    out["L3_lower_violations"], out["L3_upper_violations"] = lo_v, hi_v
    out["L3_width_max_for_delta_le_0.05_au"] = float(np.max(widths))
    out["L3_width_median_for_delta_le_0.05_au"] = float(np.median(widths))

    # P1 (synthetic): two-body motion of both bodies never brings them closer than the MOID
    rng = np.random.default_rng(1)
    n_orb, n_t = 300, 40000
    a = rng.uniform(0.6, 3.5, n_orb)
    e = rng.uniform(0.0, 0.9, n_orb)
    inc = np.degrees(np.arccos(rng.uniform(np.cos(np.deg2rad(60)), 1, n_orb)))
    om, w = rng.uniform(0, 360, n_orb), rng.uniform(0, 360, n_orb)
    keep = a * (1 - e) < 1.3
    a, e, inc, om, w = a[keep], e[keep], inc[keep], om[keep], w[keep]
    moid = moid_earth(a, e, inc, om, w)
    t = np.linspace(0, 400.0, n_t)  # years
    nE = 2 * np.pi / EARTH_A ** 1.5
    ME = nE * t
    earth = kepler_positions(np.full_like(ME, EARTH_A), np.full_like(ME, EARTH_E), np.zeros_like(ME),
                             np.zeros_like(ME), np.full_like(ME, np.degrees(EARTH_VARPI)), ME)
    worst, ratio = np.inf, []
    for k in range(len(a)):
        M = rng.uniform(0, 2 * np.pi) + 2 * np.pi / a[k] ** 1.5 * t
        ast = kepler_positions(np.full_like(M, a[k]), np.full_like(M, e[k]), np.full_like(M, inc[k]),
                               np.full_like(M, om[k]), np.full_like(M, w[k]), M)
        dmin = float(np.min(np.linalg.norm(ast - earth, axis=1)))
        worst = min(worst, dmin - moid[k])
        ratio.append(dmin / max(moid[k], 1e-12))
    out["P1_synthetic_orbits"], out["P1_samples_per_orbit"] = int(len(a)), n_t
    out["P1_min_of_closest_approach_minus_moid_au"] = float(worst)
    out["P1_violations"] = int(worst < -TOL)

    # P1 (context): JPL close approaches use the orbits of their own epoch; the MOID uses today's osculating orbit
    cad = pd.read_csv(PROC / "cad_earth.csv", usecols=["des", "cd", "dist"])
    m = cad.merge(kr[["pdes", "moid_k"]], left_on="des", right_on="pdes")
    m["year"] = m["cd"].str[:4].astype(int)
    below = m["dist"] < m["moid_k"] - 1e-4
    out["P1_jpl_approaches"] = int(len(m))
    out["P1_jpl_share_dist_ge_moid_minus_1e-4"] = float((~below).mean())
    out["P1_jpl_share_by_era"] = {era: float((~below[(m.year >= lo) & (m.year < hi)]).mean())
                                  for era, (lo, hi) in {"1910-1969": (1910, 1970), "1970-1999": (1970, 2000),
                                                        "2000-2024": (2000, 2025)}.items()}

    # P0: leakage of a random split. With test fraction rho, a test row of an asteroid with m rows has its asteroid
    #     in training with probability 1 - rho^(m-1), so the expected share over test rows is
    #     sum_a m_a (1 - rho^(m_a - 1)) / sum_a m_a.
    def predicted_seen(m: np.ndarray, rho: float) -> float:
        return float(np.sum(m * (1 - rho ** (m - 1))) / np.sum(m))

    m_cad = m.groupby("des").size().to_numpy()
    out["P0_cad_predicted_seen_share_5fold"] = predicted_seen(m_cad, 0.2)
    import json as _json
    out["P0_cad_observed_seen_share_5fold"] = _json.loads((RESULTS / "earth_view_stats.json").read_text())[
        "test_rows_with_asteroid_in_train"]["row_split"]
    kag = ROOT_DATA / "kaggle" / "nasa-asteroids-classification" / "nasa.csv"
    if kag.exists():
        ids = pd.read_csv(kag, usecols=["Neo Reference ID"])["Neo Reference ID"]
        out["P0_kaggle_predicted_seen_share_10fold"] = predicted_seen(ids.value_counts().to_numpy(), 0.1)
        out["P0_kaggle_observed_seen_share_10fold"] = _json.loads(
            (RESULTS / "kaggle_nasa_comparison.json").read_text())["A_test_records_with_asteroid_in_train"]

    # P2: hazard index equivalence
    H = kr["H"].to_numpy()
    phi = np.maximum(d / 0.05, 10.0 ** ((H - 22.0) / 5.0))
    out["P2_mismatches"] = int(((phi <= 1.0) != ((d <= 0.05) & (H <= 22.0))).sum())

    # T1: decision-law soundness and termination on the real catalogue (law as specified)
    pr = pd.read_csv(RESULTS / "pipeline_rows.csv").merge(kr[["pdes", "moid_k", "U_node", "L0"]], on="pdes",
                                                          suffixes=("", "_k"))
    pha, geo, size = pr["act"].eq("PHA"), pr["act"].eq("NOT_PHA_GEOMETRY"), pr["act"].eq("NOT_PHA_SIZE")
    out["T1_PHA_with_H_gt_22"] = int((pha & (pr["H"] > 22)).sum())
    out["T1_PHA_with_delta_gt_0.05"] = int((pha & (pr["moid_k"] > 0.05 + TOL)).sum())
    out["T1_NOT_PHA_GEOMETRY_with_delta_le_0.05"] = int((geo & (pr["moid_k"] <= 0.05)).sum())
    out["T1_NOT_PHA_SIZE_with_H_le_22"] = int((size & (pr["H"] <= 22)).sum())
    out["T1_max_steps"] = int(pr["steps"].max())
    out["T1_decided"] = int((pha | geo | size).sum())

    save_json(out, RESULTS / "proof_checks.json")
    for k_, v in out.items():
        print(f"{k_}: {v}")


if __name__ == "__main__":
    main()
