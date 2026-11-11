"""Asteroid's-view geometry: orbit paths, Earth-orbit proximity and Opik encounter variables.

Everything here depends only on an asteroid's own heliocentric path (osculating elements) and on
Earth's ORBIT as a fixed curve. Earth's position at any particular time never enters.

Earth's orbit: J2000 mean elements of the Earth-Moon barycentre (Standish, JPL SSD
"Keplerian elements for approximate positions of the major planets", Table 1), treated as a fixed
Keplerian ellipse in the ecliptic; its 1.5e-5 deg inclination is neglected.

Conventions: element angles are passed in DEGREES (as in JPL SBDB) and converted here.
All functions are vectorized over asteroids.
"""
from __future__ import annotations

import numpy as np

EARTH_A = 1.00000261  # au
EARTH_E = 0.01671123
EARTH_VARPI = np.deg2rad(102.93768193)  # longitude of perihelion
EARTH_P = EARTH_A * (1.0 - EARTH_E**2)  # semi-latus rectum, au
TWO_PI = 2.0 * np.pi
GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0


def earth_radius(lam):
    """Heliocentric distance (au) of Earth's orbit at ecliptic longitude `lam` (rad)."""
    return EARTH_P / (1.0 + EARTH_E * np.cos(lam - EARTH_VARPI))


def orbit_basis(i, om, w):
    """Perifocal unit vectors P (towards perihelion) and Q (90 deg ahead) in the ecliptic frame.

    Inputs in radians, shape (n,). Returns two arrays of shape (n, 3).
    """
    ci, si = np.cos(i), np.sin(i)
    co, so = np.cos(om), np.sin(om)
    cw, sw = np.cos(w), np.sin(w)
    P = np.stack([cw * co - sw * so * ci, cw * so + sw * co * ci, sw * si], axis=-1)
    Q = np.stack([-sw * co - cw * so * ci, -sw * so + cw * co * ci, cw * si], axis=-1)
    return P, Q


def path_points(p, e, P, Q, f):
    """K1 path kernel: heliocentric ecliptic positions at true anomalies `f`.

    p, e: shape (n,); P, Q: shape (n, 3); f: shape (n, m). Returns shape (n, m, 3).
    """
    r = p[:, None] / (1.0 + e[:, None] * np.cos(f))
    return r[..., None] * (np.cos(f)[..., None] * P[:, None, :] + np.sin(f)[..., None] * Q[:, None, :])


def distance_to_earth_orbit(X, newton_iters: int = 4):
    """Exact distance (au) from points X (..., 3) to Earth's orbital ellipse.

    Minimizes D(lam) = |X - E(lam)|^2 over Earth's true longitude lam with Newton steps started at the
    point's own longitude (Earth's eccentricity is small, so this converges in a few steps).
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


def moid_earth(a, e, i_deg, om_deg, w_deg, n_grid: int = 2048, n_candidates: int = 4,
               golden_iters: int = 48, chunk: int = 256):
    """Exact path-to-path minimum distance (au) between each asteroid orbit and Earth's orbit.

    Grid over the asteroid's true anomaly, exact distance to Earth's orbit at each grid point,
    then golden-section refinement of the `n_candidates` deepest local minima.
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

        # Vectorized golden-section search: interior points c < d, one new evaluation per step.
        c = hi - GOLDEN * (hi - lo)
        d_pt = lo + GOLDEN * (hi - lo)
        fc, fd = dist_at(c), dist_at(d_pt)
        for _ in range(golden_iters):
            left = fc < fd  # minimum lies in [lo, d]; otherwise in [c, hi]
            hi = np.where(left, d_pt, hi)
            lo = np.where(left, lo, c)
            new_c = np.where(left, hi - GOLDEN * (hi - lo), d_pt)  # left: fresh point; right: old d
            new_d = np.where(left, c, lo + GOLDEN * (hi - lo))  # left: old c; right: fresh point
            val = dist_at(np.where(left, new_c, new_d))
            fc, fd = np.where(left, val, fd), np.where(left, fc, val)
            c, d_pt = new_c, new_d
        refined = np.minimum(fc, fd)
        refined = np.where(valid, refined, np.inf)
        out[sl] = np.minimum(refined.min(axis=1), d.min(axis=1))
    return out


def node_geometry(a, e, i_deg, om_deg, w_deg):
    """Closed-form quantities at the two nodes (where the path pierces the ecliptic).

    Returns a dict of arrays:
      r_asc, r_desc     heliocentric distance of the asteroid at its ascending / descending node
      dr_asc, dr_desc   radial gap to Earth's orbit at the same longitude (signed, au)
      d_node            min |radial gap|: exact distance between the two orbits along the node line,
                        an upper bound on the MOID
      d_lin             first-order (tangent-line) MOID estimate at the better node: the distance between
                        the two orbits' tangent lines there (cf. Bonanno 2000). Accurate when the
                        orbits are nearly straight over the gap; unreliable at very low inclination.
    """
    a, e = np.asarray(a, float), np.asarray(e, float)
    i, om, w = (np.deg2rad(np.asarray(v, float)) for v in (i_deg, om_deg, w_deg))
    p = a * (1.0 - e**2)
    out = {}
    d_lin_nodes = []
    for name, f_node, lam in (("asc", -w, om), ("desc", np.pi - w, om + np.pi)):
        r = p / (1.0 + e * np.cos(f_node))
        R = earth_radius(lam)
        dr = r - R
        k_a = e * np.sin(f_node) / (1.0 + e * np.cos(f_node))  # asteroid v_r / v_theta at the node
        F = lam - EARTH_VARPI
        k_e = EARTH_E * np.sin(F) / (1.0 + EARTH_E * np.cos(F))  # Earth's v_r / v_theta there
        si, ci = np.abs(np.sin(i)), np.cos(i)
        denom = np.sqrt(si**2 * (1.0 + k_e**2) + (k_a - k_e * ci) ** 2)
        d_lin = np.abs(dr) * si / np.maximum(denom, 1e-300)
        out[f"r_{name}"], out[f"dr_{name}"], out[f"d_lin_{name}"] = r, dr, d_lin
        d_lin_nodes.append(d_lin)
    out["d_node"] = np.minimum(np.abs(out["dr_asc"]), np.abs(out["dr_desc"]))
    out["d_lin"] = np.minimum(*d_lin_nodes)
    return out


def opik_variables(a, e, i_deg):
    """Opik encounter variables w.r.t. Earth (units: Earth's orbital radius and speed).

    T: Tisserand parameter relative to Earth; U: unperturbed encounter speed sqrt(3 - T)
    (0 if T >= 3, i.e. no encounter possible in Opik's approximation); cos_theta: angle between the
    encounter velocity and Earth's heliocentric velocity. Time-independent: they describe any encounter
    the path allows, not a particular one.
    """
    a, e = np.asarray(a, float), np.asarray(e, float)
    i = np.deg2rad(np.asarray(i_deg, float))
    T = 1.0 / a + 2.0 * np.sqrt(np.maximum(a * (1.0 - e**2), 0.0)) * np.cos(i)
    U = np.sqrt(np.maximum(3.0 - T, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta = np.where(U > 0, (1.0 - U**2 - 1.0 / a) / (2.0 * U), np.nan)
    return {"T_earth": T, "U": U, "cos_theta": np.clip(cos_theta, -1.0, 1.0)}
