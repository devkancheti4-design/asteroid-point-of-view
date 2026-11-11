"""PHA geometry kernels K1-K5: the asteroid's point of view, with Earth's ORBIT as a fixed curve.

Reference implementation of PHA_MATH_CORE.md (numpy, plus scipy.special / scipy.optimize / scipy.stats).
Inputs are SBDB fields. Angles arrive in degrees and are converted once, in elements().

No kernel reads ma, epoch, moid, pha, any close-approach quantity, or Earth's position at a time.
MOID depends on the two CURVES only, so neither body's position along its orbit can enter.

Units: au, radians (after elements()), magnitudes, km for diameters."""
import math
import numpy as np
from math import pi
from scipy.special import ndtr
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal

D0 = 0.05          # au   PHA MOID threshold (definition; never fitted)
H0 = 22.0          # mag  PHA absolute-magnitude threshold (definition; never fitted)
TWO_PI = 2.0 * pi

# ------------------------------------------------------------------ Earth's orbit, a fixed curve
# Standish, JPL "Approximate Positions of the Planets", Table 1 (1800-2050 AD), Earth-Moon barycenter,
# mean ecliptic and equinox of J2000. The tabulated inclination (-0.00001531 deg) is set to 0, so the
# curve lies in the reference plane; the vertical offset this drops is below 3e-7 au.
EARTH = dict(a=1.00000261, e=0.01671123, i_deg=-0.00001531, varpi_deg=102.93768193, node_deg=0.0)
A_E, E_E = EARTH["a"], EARTH["e"]
VARPI_E = math.radians(EARTH["varpi_deg"])
P_E = A_E * (1.0 - E_E ** 2)               # semi-latus rectum
B_E = A_E * math.sqrt(1.0 - E_E ** 2)      # semi-minor axis
QP_E = A_E * (1.0 - E_E)                   # perihelion distance
QA_E = A_E * (1.0 + E_E)                   # aphelion distance
K_E = P_E * E_E / (1.0 - E_E) ** 2         # >= max |dR_E/dlambda|: Lipschitz constant of R_E in longitude
XE = np.array([math.cos(VARPI_E), math.sin(VARPI_E), 0.0])     # Earth's perihelion direction
YE = np.array([-math.sin(VARPI_E), math.cos(VARPI_E), 0.0])


def R_earth(lam):
    """Earth's heliocentric distance (au) at ecliptic longitude lam (rad)."""
    return P_E / (1.0 + E_E * np.cos(lam - VARPI_E))


def dR_earth(lam):
    """dR_E/dlambda (au/rad)."""
    t = lam - VARPI_E
    return P_E * E_E * np.sin(t) / (1.0 + E_E * np.cos(t)) ** 2


def earth_point(psi):
    """Earth's orbit at eccentric anomaly psi: (3, n)."""
    psi = np.atleast_1d(psi)
    return A_E * (np.cos(psi) - E_E) * XE[:, None] + B_E * np.sin(psi) * YE[:, None]


def earth_tangent(psi):
    psi = np.atleast_1d(psi)
    return -A_E * np.sin(psi) * XE[:, None] + B_E * np.cos(psi) * YE[:, None]


# ------------------------------------------------------------------ input
def elements(row):
    """SBDB fields -> au/radians. p = q (1 + e) stays well conditioned as e -> 1 (a(1-e^2) does not)."""
    e = np.asarray(row["e"], float)
    q = np.asarray(row["q"], float) if "q" in row else np.asarray(row["a"], float) * (1.0 - e)
    a = np.asarray(row["a"], float) if "a" in row else q / (1.0 - e)
    return dict(a=a, e=e, q=q, p=q * (1.0 + e),
                i=np.deg2rad(np.asarray(row["i"], float)),
                om=np.deg2rad(np.mod(np.asarray(row["om"], float), 360.0)),
                w=np.deg2rad(np.mod(np.asarray(row["w"], float), 360.0)))


def sigmas(row):
    """1-sigma of (a, e, i, Omega, omega) in au, -, rad, rad, rad.  sigma_ma is not used: MOID does not
    depend on the mean anomaly."""
    return np.array([row["sigma_a"], row["sigma_e"], math.radians(row["sigma_i"]),
                     math.radians(row["sigma_om"]), math.radians(row["sigma_w"])], float)


def valid(el, H):
    """True where the input is a usable conic: finite, e >= 0, q > 0, 0 <= i <= pi, H finite."""
    ok = np.ones(np.shape(el["e"]), bool)
    for k in ("e", "q", "i", "om", "w"):
        ok &= np.isfinite(el[k])
    ok &= (el["e"] >= 0) & (el["q"] > 0) & (el["i"] >= 0) & (el["i"] <= pi) & np.isfinite(H)
    ok &= np.where(el["e"] < 1, el["a"] > 0, True)
    return ok


# ------------------------------------------------------------------ K1  path kernel
def basis(i, om, w):
    """Perifocal unit vectors P (to perihelion), Q (90 deg ahead in the motion) and the orbit normal h."""
    ci, si, cO, sO, cw, sw = np.cos(i), np.sin(i), np.cos(om), np.sin(om), np.cos(w), np.sin(w)
    P = np.stack([cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si])
    Q = np.stack([-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si])
    h = np.stack([sO * si, -cO * si, ci])
    return P, Q, h


def f_window(e):
    """None for an ellipse (all f); for e >= 1 the half-width of the allowed true anomalies."""
    return None if e < 1 else math.acos(-1.0 / e) * (1.0 - 1e-9)


def path(p, e, i, om, w, f):
    """K1: heliocentric ecliptic position (au) at true anomaly f (rad). Scalar elements, array f -> (3, n)."""
    P, Q, _ = basis(i, om, w)
    f = np.atleast_1d(np.asarray(f, float))
    r = p / (1.0 + e * np.cos(f))
    return r * (np.cos(f) * P[:, None] + np.sin(f) * Q[:, None])


def path_tangent(p, e, i, om, w, f):
    """d(position)/df (au/rad)."""
    P, Q, _ = basis(i, om, w)
    f = np.atleast_1d(np.asarray(f, float))
    c, s = np.cos(f), np.sin(f)
    r = p / (1.0 + e * c)
    dr = r * r * e * s / p
    return dr * (c * P[:, None] + s * Q[:, None]) + r * (-s * P[:, None] + c * Q[:, None])


def cyl(p, e, i, om, w, f):
    """(rho, lambda, z, r): projected distance, ecliptic longitude, height and radius at true anomaly f."""
    r = p / (1.0 + e * np.cos(f))
    u = w + f
    X, Y = np.cos(u), np.sin(u) * np.cos(i)
    return r * np.hypot(X, Y), om + np.arctan2(Y, X), r * np.sin(u) * np.sin(i), r


# ------------------------------------------------------------------ K2(a)  closed forms
def radial_gap(el):
    """L0 = max(0, q - Q_E, q_E - Q): a rigorous lower bound on MOID (Q = inf for e >= 1).
    Returns L0 and which side certifies it (+1 outside, -1 inside, 0 none)."""
    q, e = el["q"], el["e"]
    Q = np.where(e < 1, q * (1 + e) / np.where(e < 1, 1 - e, 1.0), np.inf)
    out, inn = q - QA_E, QP_E - Q
    L0 = np.maximum(0.0, np.maximum(out, inn))
    side = np.where(out > 0, 1, np.where(inn > 0, -1, 0))
    return L0, side


def nodal(el, u_max=0.2):
    """Signed nodal distances, the nodal upper bound U_node and the linearised node correction d_lin.

    delta(+/-) = r(+/-) - R_E(Omega, Omega+pi),  r(+/-) = p / (1 +/- e cos w)
    U_node = min |delta|  >= MOID  (two actual points, one on each curve)
    d_lin  = |delta| r sin i / sqrt(s^2 + r^2 sin^2 i)  (first order about the node; valid if |u*| <= u_max)"""
    p, e, i, om, w = el["p"], el["e"], el["i"], el["om"], el["w"]
    cw, sw, si, ci = np.cos(w), np.sin(w), np.sin(i), np.cos(i)
    res = {}
    for sgn, nm in ((1, "asc"), (-1, "desc")):
        den = 1.0 + sgn * e * cw
        ex = den > 0
        r = np.where(ex, p / np.where(ex, den, 1.0), np.inf)
        lam = om if sgn > 0 else om + pi
        R, dR = R_earth(lam), dR_earth(lam)
        delta = np.where(ex, r - R, np.inf)
        s = np.where(ex, -sgn * r * r * e * sw / p - ci * dR, np.nan)
        q2 = s * s + (r * si) ** 2
        good = ex & (q2 > 0)
        safe = np.where(good, q2, 1.0)
        dlin = np.where(good, np.abs(delta) * r * si / np.sqrt(safe), np.nan)
        ust = np.where(good, -s * delta / safe, np.nan)
        res[nm] = dict(r=r, R=R, delta=delta, s=s, dlin=dlin, ustar=ust, exists=ex,
                       lin_ok=good & (np.abs(ust) <= u_max))
    a, d = res["asc"], res["desc"]
    U = np.minimum(np.abs(a["delta"]), np.abs(d["delta"]))
    dl = np.fmin(np.where(a["lin_ok"], a["dlin"], np.nan), np.where(d["lin_ok"], d["dlin"], np.nan))
    return dict(asc=a, desc=d, U_node=U, d_lin=dl)


def smooth_proximity(delta_p, delta_m, eps=1e-4, tau=1e-3):
    """Differentiable stand-in for min(|delta+|, |delta-|): smoothed |.| (eps, au) and a soft minimum
    (tau, au). It lies within tau*ln 2 + eps of min(|delta+|, |delta-|)."""
    ap, am = np.sqrt(delta_p ** 2 + eps ** 2), np.sqrt(delta_m ** 2 + eps ** 2)
    m = np.minimum(ap, am)
    return m - tau * np.log(np.exp(-(ap - m) / tau) + np.exp(-(am - m) / tau))


# ------------------------------------------------------------------ 1-D machinery
def _grid(e, w, N):
    """True-anomaly grid over the allowed range, with the two node anomalies added exactly."""
    fw = f_window(e)
    nodes = np.mod(np.array([-w, pi - w]) + pi, TWO_PI) - pi
    if fw is None:
        f = np.linspace(-pi, pi, N, endpoint=False)
        circ = True
    else:
        f = np.linspace(-fw, fw, N)
        nodes = nodes[np.abs(nodes) < fw]
        circ = False
    return np.unique(np.concatenate([f, nodes])), circ


def _local_minima(v, circ):
    if circ:
        return np.nonzero((v <= np.roll(v, 1)) & (v <= np.roll(v, -1)))[0]
    k = np.nonzero((v[1:-1] <= v[:-2]) & (v[1:-1] <= v[2:]))[0] + 1
    return k


def _bracket(f, k, circ):
    n = len(f)
    if circ:
        lo = f[k - 1] if k > 0 else f[-1] - TWO_PI
        hi = f[k + 1] if k < n - 1 else f[0] + TWO_PI
    else:
        lo, hi = f[max(k - 1, 0)], f[min(k + 1, n - 1)]
    return lo, hi


def _refine(fun, f, v, circ, keep):
    """Refine the `keep` smallest grid minima with bounded Brent; return [(value, f)] sorted, deduplicated."""
    ks = _local_minima(v, circ)
    ks = ks[np.argsort(v[ks])][:keep]
    out = []
    for k in ks:
        lo, hi = _bracket(f, k, circ)
        r = minimize_scalar(fun, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-13))
        x, val = (r.x, r.fun) if r.fun <= v[k] else (f[k], v[k])
        x = (x + pi) % TWO_PI - pi if circ else x
        if all(abs(((x - y + pi) % TWO_PI) - pi) > 1e-7 for _, y in out):
            out.append((float(val), float(x)))
    return sorted(out)


def _scalar_el(el):
    return tuple(float(el[k]) for k in ("p", "e", "i", "om", "w"))


def _cyl_s(t, p, e, i, om, w):
    r = p / (1.0 + e * math.cos(t))
    u = w + t
    X, Y = math.cos(u), math.sin(u) * math.cos(i)
    return r * math.hypot(X, Y), om + math.atan2(Y, X), r * math.sin(u) * math.sin(i)


def _lon_s(t, el5, which):
    rho, lam, z = _cyl_s(t, *el5)
    g = rho - P_E / (1.0 + E_E * math.cos(lam - VARPI_E))
    if which == "up":
        return math.sqrt(z * z + g * g)
    c = 4.0 * rho * QP_E / (pi * pi)
    return math.sqrt(z * z + c / (c + K_E * K_E) * g * g)


def lon_fun(el, f, which):
    rho, lam, z, _ = cyl(float(el["p"]), float(el["e"]), float(el["i"]), float(el["om"]), float(el["w"]), f)
    g = rho - R_earth(lam)
    if which == "up":
        return np.sqrt(z * z + g * g)
    c = 4.0 * rho * QP_E / pi ** 2
    return np.sqrt(z * z + c / (c + K_E ** 2) * g * g)


# ------------------------------------------------------------------ K2(a')  certified 1-D bounds
def lon_bounds(el, N=720, keep=4):
    """Same-longitude bounds from ONE minimisation over the asteroid's true anomaly:
         L_lon = min_f sqrt(z^2 + kappa g^2)  <=  MOID  <=  U_lon = min_f sqrt(z^2 + g^2)
       g = rho - R_E(lambda); kappa = c/(c + K_E^2), c = 4 rho q_E / pi^2. Exact (L = U = MOID) for a
       circular Earth orbit."""
    el5 = _scalar_el(el)
    f, circ = _grid(el5[1], el5[4], N)
    out = {}
    for which in ("up", "lo"):
        v = lon_fun(el, f, which)
        mins = _refine(lambda t: _lon_s(t, el5, which), f, v, circ, keep)
        out[which] = mins[0]
    return dict(U_lon=out["up"][0], f_up=out["up"][1], L_lon=out["lo"][0], f_lo=out["lo"][1])


# ------------------------------------------------------------------ K2(b)  exact MOID
def nearest_on_earth(X3, iters=7):
    """Inner minimisation over Earth's eccentric anomaly psi for each column of X3 (3, n).

    Newton on F(psi) = (a X + a^2 e) sin psi - b Y cos psi - (a^2 - b^2) sin psi cos psi = 0 from the central
    angle psi0 = atan2(Y/b, (X + a e)/a). For any point whose ecliptic projection lies outside the evolute of
    Earth's ellipse (an astroid of half-size ~ a e^2 about the ellipse's centre) the minimum is unique."""
    X, Y, Z = XE @ X3, YE @ X3, X3[2]
    psi = np.arctan2(Y / B_E, (X + A_E * E_E) / A_E)
    c1, c2, c3 = A_E * X + A_E * A_E * E_E, B_E * Y, A_E * A_E - B_E * B_E
    for _ in range(iters):
        s, c = np.sin(psi), np.cos(psi)
        F = c1 * s - c2 * c - c3 * s * c
        dF = c1 * c + c2 * s - c3 * (c * c - s * s)
        psi = psi - F / dF
    d2 = (X - A_E * (np.cos(psi) - E_E)) ** 2 + (Y - B_E * np.sin(psi)) ** 2 + Z * Z
    return psi, d2


def _near_s(x, y, z, iters=7):
    X = XE[0] * x + XE[1] * y
    Y = YE[0] * x + YE[1] * y
    psi = math.atan2(Y / B_E, (X + A_E * E_E) / A_E)
    c1, c2, c3 = A_E * X + A_E * A_E * E_E, B_E * Y, A_E * A_E - B_E * B_E
    for _ in range(iters):
        s, c = math.sin(psi), math.cos(psi)
        psi -= (c1 * s - c2 * c - c3 * s * c) / (c1 * c + c2 * s - c3 * (c * c - s * s))
    return (X - A_E * (math.cos(psi) - E_E)) ** 2 + (Y - B_E * math.sin(psi)) ** 2 + z * z, psi


def _G_s(t, el5):
    p, e, i, om, w = el5
    r = p / (1.0 + e * math.cos(t))
    u = w + t
    cu, su, cO, sO, ci = math.cos(u), math.sin(u), math.cos(om), math.sin(om), math.cos(i)
    x = r * (cO * cu - sO * su * ci)
    y = r * (sO * cu + cO * su * ci)
    z = r * su * math.sin(i)
    return _near_s(x, y, z)[0]


def moid(el, N=720, keep=4, sin_min=0.05):
    """K2(b): MOID = min over (f, psi) of |r_ast(f) - r_E(psi)|, nested: inner Newton on Earth's ellipse,
    outer 1-D global search over f (grid + Brent on the `keep` best basins). Returns the MOID, the signed
    MOID (Gronchi-Tommei style: sign of (t_E x t_a).Delta), the minimising anomalies, the second-best local
    minimum, and the tangent angle (signed MOID is smooth only if the tangents are not parallel)."""
    el5 = _scalar_el(el)
    p, e, i, om, w = el5
    f, circ = _grid(e, w, N)
    _, d2 = nearest_on_earth(path(p, e, i, om, w, f))
    mins = _refine(lambda t: _G_s(t, el5), f, d2, circ, keep)
    fbest = mins[0][1]
    X3 = path(p, e, i, om, w, fbest)
    psi, d2b = nearest_on_earth(X3)
    Delta = (X3 - earth_point(psi))[:, 0]
    ta = path_tangent(p, e, i, om, w, fbest)[:, 0]
    te = earth_tangent(psi)[:, 0]
    nvec = np.cross(te, ta)
    nn = np.linalg.norm(nvec)
    sin_t = nn / (np.linalg.norm(te) * np.linalg.norm(ta))
    nhat = nvec / nn if nn > 0 else np.full(3, np.nan)
    D = math.sqrt(max(float(d2b[0]), 0.0))
    return dict(moid=D, signed=float(nhat @ Delta), f=float(fbest), psi=float(psi[0]), r_a=X3[:, 0],
                nhat=nhat, sin_tangent=float(sin_t), signed_ok=bool(sin_t >= sin_min),
                second=math.sqrt(mins[1][0]) if len(mins) > 1 else math.inf,
                minima=[(math.sqrt(max(v, 0.0)), x) for v, x in mins])


TOL_D = 2e-8      # au: a distance computed from squared differences of ~1 au vectors is resolved to
                  # about sqrt(machine eps) * 1 au near zero, so certificates compare at this tolerance


def moid_certified(el, N=720, keep=4):
    """K2(b) with its certificate: L <= MOID <= U from K2(a), K2(a') must hold; one denser retry if not."""
    L0, _ = radial_gap(el)
    nd = nodal(el)
    for n in (N, 4 * N):
        m, b = moid(el, n, keep), lon_bounds(el, n, keep)
        L = max(float(L0), b["L_lon"])
        U = min(float(nd["U_node"]), b["U_lon"])
        ok = (L - TOL_D <= m["moid"] <= U + TOL_D)
        if ok:
            break
    m.update(L=L, U=U, certified=ok, U_lon=b["U_lon"], L_lon=b["L_lon"])
    return m


def moid_batch(P, E, I, OM, W, N=720, iters=40):
    """Vectorised MOID for many ELLIPTIC orbits at once (sampling): grid over f for all orbits, inner Newton,
    golden-section refinement of each orbit's best grid cell. Returns (moid, f*)."""
    P, E, I, OM, W = (np.asarray(x, float)[:, None] for x in (P, E, I, OM, W))
    f = np.linspace(-pi, pi, N, endpoint=False)[None, :]

    def G(ff):
        r = P / (1.0 + E * np.cos(ff))
        u = W + ff
        cu, su = np.cos(u), np.sin(u)
        x = r * (np.cos(OM) * cu - np.sin(OM) * su * np.cos(I))
        y = r * (np.sin(OM) * cu + np.cos(OM) * su * np.cos(I))
        z = r * su * np.sin(I)
        X3 = np.stack([x.ravel(), y.ravel(), z.ravel()])
        return nearest_on_earth(X3)[1].reshape(x.shape)

    d2 = G(f)
    k = d2.argmin(axis=1)
    h = TWO_PI / N
    lo = f[0, k][:, None] - h
    hi = f[0, k][:, None] + h
    g = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    fc, fd = G(c), G(d)
    for _ in range(iters):
        left = fc < fd
        hi = np.where(left, d, hi)
        lo = np.where(left, lo, c)
        c_new = hi - g * (hi - lo)
        d_new = lo + g * (hi - lo)
        c, d = np.where(left, c_new, d), np.where(left, c, d_new)
        fc, fd = G(c), G(d)          # one of the two is new each step; recomputing both keeps it short
    x = (lo + hi) / 2
    return np.sqrt(np.maximum(G(x), 0.0))[:, 0], x[:, 0]


# ------------------------------------------------------------------ K3  size
C_D = 1329.0     # km: D = C_D / sqrt(p_V) * 10^(-H/5)


def diameter_km(H, pV):
    return C_D / np.sqrt(pV) * 10.0 ** (-np.asarray(H, float) / 5.0)


def size_label(H, H0=H0):
    """The label's own size test: exactly H <= H0 (catalog H as given)."""
    return np.asarray(H, float) <= H0


def p_large_lognormal(H, mu_lnp, s_lnp, Dstar_km):
    """P(D >= D*) when ln p_V ~ N(mu, s^2):  ln D = ln C_D - (H/5) ln 10 - (1/2) ln p_V is Gaussian."""
    z = (math.log(C_D / Dstar_km) - np.asarray(H, float) * math.log(10.0) / 5.0 - mu_lnp / 2.0) / (s_lnp / 2.0)
    return ndtr(z)


def p_large_mixture(H, weights, mus, ss, Dstar_km):
    return sum(wk * p_large_lognormal(H, mk, sk, Dstar_km) for wk, mk, sk in zip(weights, mus, ss))


def fit_lognormal_albedo(pV_train):
    """mu, s of ln p_V from the TRAINING split's measured albedos (selection-biased sample: say so)."""
    x = np.log(np.asarray(pV_train, float)[np.asarray(pV_train, float) > 0])
    return float(x.mean()), float(x.std(ddof=1))


# ------------------------------------------------------------------ K4  uncertainty
def nodal_grad(el):
    """d(delta+, delta-)/d(a, e, i, Omega, omega): closed form. (i does not enter the nodal distances.)"""
    a, e, om, w = float(el["a"]), float(el["e"]), float(el["om"]), float(el["w"])
    cw, sw = math.cos(w), math.sin(w)
    d1, d2 = 1 + e * cw, 1 - e * cw
    g_asc = np.array([(1 - e * e) / d1, -a * (2 * e + (1 + e * e) * cw) / d1 ** 2, 0.0,
                      -float(dR_earth(om)), a * (1 - e * e) * e * sw / d1 ** 2])
    g_desc = np.array([(1 - e * e) / d2, a * ((1 + e * e) * cw - 2 * e) / d2 ** 2, 0.0,
                       -float(dR_earth(om + pi)), -a * (1 - e * e) * e * sw / d2 ** 2])
    return g_asc, g_desc


def radial_grad(el, side):
    """d(q - Q_E)/dx (side +1) or d(q_E - Q)/dx (side -1)."""
    a, e = float(el["a"]), float(el["e"])
    return np.array([1 - e, -a, 0, 0, 0.0]) if side > 0 else np.array([-(1 + e), -a, 0, 0, 0.0])


def moid_grad(el, m):
    """Envelope theorem: d(signed MOID)/d(a, e, i, Omega, omega) = nhat . d r_ast/dx at the minimising f
    (Earth's orbit fixed). d r/da = r/a, d r/de = (dr/de) r_hat, d r/di = n_node x r, d r/dOmega = z x r,
    d r/domega = h x r."""
    a, e, i, om = float(el["a"]), float(el["e"]), float(el["i"]), float(el["om"])
    f, ra, n = m["f"], m["r_a"], m["nhat"]
    r = float(np.linalg.norm(ra))
    dr_de = -a * (2 * e + (1 + e * e) * math.cos(f)) / (1 + e * math.cos(f)) ** 2
    node = np.array([math.cos(om), math.sin(om), 0.0])
    hvec = np.array([math.sin(om) * math.sin(i), -math.cos(om) * math.sin(i), math.cos(i)])
    zhat = np.array([0.0, 0.0, 1.0])
    return np.array([n @ (ra / a), n @ (dr_de * ra / r), n @ np.cross(node, ra),
                     n @ np.cross(zhat, ra), n @ np.cross(hvec, ra)])


def p_fold(mean, s, d0=D0):
    """P(|X| <= d0), X ~ N(mean, s^2)."""
    if not s > 0:
        return float(abs(mean) <= d0)
    return float(ndtr((d0 - mean) / s) - ndtr((-d0 - mean) / s))


def p_below(mean, s, d0=D0):
    """P(X <= d0), X ~ N(mean, s^2)."""
    if not s > 0:
        return float(mean <= d0)
    return float(ndtr((d0 - mean) / s))


def p_nodal(el, sig, d0=D0):
    """K4(a) on the nodal bound: P(min(|delta+|, |delta-|) <= d0) under the linearised bivariate normal.
    U_node >= MOID, so this is a LOWER bound on P(MOID <= d0)."""
    nd = nodal(el)
    ga, gd = nodal_grad(el)
    S = np.diag(sig ** 2)
    m = np.array([float(nd["asc"]["delta"]), float(nd["desc"]["delta"])])
    C = np.array([[ga @ S @ ga, ga @ S @ gd], [gd @ S @ ga, gd @ S @ gd]])
    sa, sd = math.sqrt(C[0, 0]), math.sqrt(C[1, 1])
    pa = p_fold(m[0], sa, d0) if np.isfinite(m[0]) else 0.0
    pd = p_fold(m[1], sd, d0) if np.isfinite(m[1]) else 0.0
    if not (np.isfinite(m).all() and sa > 0 and sd > 0):
        return dict(P=1 - (1 - pa) * (1 - pd) if pa * pd else max(pa, pd), s=(sa, sd), mean=m)
    rho = C[0, 1] / (sa * sd)
    if abs(rho) > 1 - 1e-12:
        both = min(pa, pd)
    else:
        both = multivariate_normal.cdf(np.array([d0, d0]), mean=m, cov=C, abseps=1e-10, releps=1e-8,
                                       lower_limit=np.array([-d0, -d0]), rng=np.random.default_rng(0))
        both = float(min(max(both, 0.0), min(pa, pd)))
    return dict(P=float(pa + pd - both), s=(sa, sd), mean=m)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return 0.0, 1.0
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - hw), min(1.0, c + hw)


def sample_elements(el, sig, n, rng):
    """Independent Gaussian draws of (a, e, i, Omega, omega); e < 0 and i outside [0, pi] are folded onto the
    same geometric orbit; draws with a <= 0 or e >= 1 (unbound) are rejected and counted."""
    a = float(el["a"]) + sig[0] * rng.standard_normal(n)
    e = float(el["e"]) + sig[1] * rng.standard_normal(n)
    i = float(el["i"]) + sig[2] * rng.standard_normal(n)
    om = float(el["om"]) + sig[3] * rng.standard_normal(n)
    w = float(el["w"]) + sig[4] * rng.standard_normal(n)
    neg = e < 0
    e = np.abs(e)
    w = np.where(neg, w + pi, w)
    i = np.mod(i, TWO_PI)
    flip = i > pi
    i = np.where(flip, TWO_PI - i, i)
    om = np.where(flip, om + pi, om)
    w = np.where(flip, w + pi, w)
    ok = (a > 0) & (e < 1)
    return dict(a=a[ok], e=e[ok], i=i[ok], om=np.mod(om[ok], TWO_PI), w=np.mod(w[ok], TWO_PI),
                p=(a * (1 - e * e))[ok]), int((~ok).sum())


def p_sampled(el, sig, n=4000, d0=D0, seed=0, N=720):
    """K4(b) Monte Carlo: P(MOID <= d0) as a fraction of draws, with its 95% Wilson interval."""
    rng = np.random.default_rng(seed)
    s, rej = sample_elements(el, sig, n, rng)
    m, _ = moid_batch(s["p"], s["e"], s["i"], s["om"], s["w"], N=N)
    k = int((m <= d0).sum())
    lo, hi = wilson(k, len(m))
    return dict(P=k / max(len(m), 1), lo=lo, hi=hi, n=len(m), rejected=rej)


def p_unscented(el, sig, d0=D0, kappa=None, N=720):
    """K4(b') unscented transform: 2n+1 = 11 sigma points of (a, e, i, Omega, omega), signed MOID at each,
    then the folded normal. Flagged invalid if a sigma point lands in another basin (non-smooth)."""
    n = 5
    kappa = 3 - n if kappa is None else kappa
    lam = kappa
    x0 = np.array([float(el[k]) for k in ("a", "e", "i", "om", "w")])
    m0 = moid(el, N)
    pts, wts = [x0], [lam / (n + lam)]
    for k in range(n):
        dx = np.zeros(n)
        dx[k] = math.sqrt(n + lam) * sig[k]
        pts += [x0 + dx, x0 - dx]
        wts += [1 / (2 * (n + lam))] * 2
    vals, same = [], True
    for x in pts:
        a, e, i, om, w = x
        if not (a > 0 and 0 <= e < 1 and 0 <= i <= pi):
            return dict(P=float("nan"), ok=False, why="sigma point outside the element domain")
        ek = dict(a=a, e=e, i=i, om=om, w=w, p=a * (1 - e * e))
        mk = moid(ek, N)
        same &= abs(((mk["f"] - m0["f"] + pi) % TWO_PI) - pi) < 0.5
        vals.append(mk["signed"] * np.sign(m0["signed"]) if m0["signed"] != 0 else mk["moid"])
    vals, wts = np.array(vals), np.array(wts)
    mean = float(wts @ vals)
    var = float(wts @ (vals - mean) ** 2)
    ok = same and var >= 0
    return dict(P=p_fold(mean, math.sqrt(max(var, 0.0)), d0), mean=mean, s=math.sqrt(max(var, 0.0)), ok=ok,
                why="" if ok else "sigma points in different basins or negative variance")


# ------------------------------------------------------------------ K5  combination
def score(dtilde, s, H, beta=0.0, w=0.0, d0=D0, H0=H0):
    """S = 1[H <= H0] * P(|d~ + noise| <= d0 + beta), noise ~ N(0, s^2 + w^2).  In [0, 1]; reads no time."""
    se = math.sqrt(s * s + w * w)
    if H > H0:
        return 0.0
    if se == 0:
        return float(abs(dtilde) <= d0 + beta)
    return float(ndtr((d0 + beta - dtilde) / se) - ndtr((-d0 - beta - dtilde) / se))


def fit_score(dtilde, s, H, y, d0=D0, H0=H0):
    """(beta, w) by maximum likelihood on TRAINING labels y (the pha flag); never the moid column."""
    from scipy.optimize import minimize
    dtilde, s, H, y = (np.asarray(v, float) for v in (dtilde, s, H, y))
    keep = H <= H0                       # faint objects carry no information about beta, w
    dt, ss, yy = dtilde[keep], s[keep], y[keep]

    def nll(th):
        beta, w = th[0], math.exp(th[1])
        se = np.sqrt(ss * ss + w * w)
        Sv = ndtr((d0 + beta - dt) / se) - ndtr((-d0 - beta - dt) / se)
        Sv = np.clip(Sv, 1e-12, 1 - 1e-12)
        return -float(np.sum(yy * np.log(Sv) + (1 - yy) * np.log(1 - Sv)))
    r = minimize(nll, x0=[0.0, math.log(1e-4)], method="Nelder-Mead",
                 options=dict(xatol=1e-10, fatol=1e-10, maxiter=4000))
    return float(r.x[0]), float(math.exp(r.x[1])), r
