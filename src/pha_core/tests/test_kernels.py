"""Property tests for K1-K5 on SYNTHETIC orbits (no SBDB data; nothing here is an asteroid result).

    python3 -u tests/test_kernels.py [--n 300]

The independent reference is a brute-force MOID: both orbits parametrised by ECCENTRIC anomaly, rotated by
explicit rotation matrices (not pha_kernels.basis), a full 2-D grid, then a 2-D polish of the best cells."""
import sys, os, math, argparse, time, json
import numpy as np
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import pha_kernels as K

FAILS = []


def check(name, cond, info=""):
    if not cond:
        FAILS.append((name, info))
        print("  FAIL %s %s" % (name, info))


# ------------------------------------------------------------------ independent reference
def R1(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def R3(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ref_orbit(a, e, i, om, w, Ecc):
    """position at eccentric anomaly Ecc: R3(Omega) R1(i) R3(omega) [a(cos E - e), b sin E, 0]"""
    b = a * math.sqrt(1 - e * e)
    pf = np.stack([a * (np.cos(Ecc) - e), b * np.sin(Ecc), np.zeros_like(Ecc)])
    return R3(om) @ R1(i) @ R3(w) @ pf


def ref_moid(a, e, i, om, w, n=1500, top=6):
    E1 = np.linspace(0, 2 * math.pi, n, endpoint=False)
    A = ref_orbit(a, e, i, om, w, E1)                                    # asteroid
    B = ref_orbit(K.A_E, K.E_E, 0.0, 0.0, K.VARPI_E, E1)               # Earth (Omega=0, omega=varpi)
    D2 = ((A[:, :, None] - B[:, None, :]) ** 2).sum(0)
    flat = np.argsort(D2, axis=None)[: top * 40]
    seeds, best = [], math.inf
    for idx in flat:
        r, c = divmod(int(idx), n)
        if all(min(abs(r - r2), n - abs(r - r2)) > 3 or min(abs(c - c2), n - abs(c - c2)) > 3 for r2, c2 in seeds):
            seeds.append((r, c))
        if len(seeds) >= top:
            break
    Ra, Rb = R3(om) @ R1(i) @ R3(w), R3(K.VARPI_E)
    ba, bb = a * math.sqrt(1 - e * e), K.A_E * math.sqrt(1 - K.E_E ** 2)

    def fg(x):                                          # squared distance and its exact gradient
        s1, c1, s2, c2 = math.sin(x[0]), math.cos(x[0]), math.sin(x[1]), math.cos(x[1])
        d = Ra @ np.array([a * (c1 - e), ba * s1, 0.0]) - Rb @ np.array([K.A_E * (c2 - K.E_E), bb * s2, 0.0])
        g1 = Ra @ np.array([-a * s1, ba * c1, 0.0])
        g2 = Rb @ np.array([-K.A_E * s2, bb * c2, 0.0])
        return float(d @ d), np.array([2 * d @ g1, -2 * d @ g2])
    for r, c in seeds:
        res = minimize(fg, x0=[E1[r], E1[c]], jac=True, method="BFGS", options=dict(gtol=1e-15, maxiter=500))
        best = min(best, res.fun, D2[r, c])
    return math.sqrt(max(best, 0.0))


# ------------------------------------------------------------------ synthetic orbits
def synth(n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        a = rng.uniform(0.55, 3.5)
        e = rng.uniform(0, 0.97) if k % 5 else rng.uniform(0, 0.02)
        i = rng.choice([rng.uniform(0, 70), rng.uniform(0, 2), rng.uniform(100, 179)], p=[0.7, 0.2, 0.1])
        rows.append(dict(a=a, e=e, i=i, om=rng.uniform(0, 360), w=rng.uniform(0, 360)))
    for r in rows:
        r["q"] = r["a"] * (1 - r["e"])
    return rows


EDGES = [
    dict(a=1.3, e=0.35, i=0.0, om=40.0, w=250.0),          # exactly coplanar
    dict(a=1.3, e=0.35, i=1e-7, om=40.0, w=250.0),         # nodes ill-defined
    dict(a=1.05, e=0.0, i=12.0, om=70.0, w=0.0),           # circular: omega undefined
    dict(a=1.05, e=1e-9, i=12.0, om=70.0, w=123.0),
    dict(a=40.0, e=0.9755, i=20.0, om=10.0, w=100.0),      # e near 1 (q ~ 0.98)
    dict(a=0.70, e=0.20, i=15.0, om=200.0, w=30.0),        # Atira-like, wholly inside (Q = 0.84)
    dict(a=1.80, e=0.30, i=5.0, om=300.0, w=60.0),         # distant Amor, wholly outside (q = 1.26)
    dict(a=1.20, e=0.20, i=170.0, om=80.0, w=200.0),       # retrograde
    dict(a=1.00000261, e=0.01671123, i=1e-4, om=0.0, w=102.93768193),   # Earth's own orbit (MOID ~ 0)
    dict(a=1.25, e=0.25, i=30.0, om=359.9999, w=0.0001),   # angle wrap
]
for r in EDGES:
    r["q"] = r["a"] * (1 - r["e"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "out", "kernel_checks.json"))
    a = ap.parse_args()
    rep = {}

    print("K1: nodes on the reference plane, |r| law, orthonormal basis, rotation-matrix agreement")
    worst = dict(z=0.0, r=0.0, orth=0.0, rot=0.0)
    for row in synth(200, 1) + EDGES:
        el = K.elements(row)
        p, e, i, om, w = (float(el[k]) for k in ("p", "e", "i", "om", "w"))
        X = K.path(p, e, i, om, w, np.array([-w, math.pi - w]))
        worst["z"] = max(worst["z"], float(np.abs(X[2]).max()))
        f = np.linspace(-3, 3, 7)
        X = K.path(p, e, i, om, w, f)
        worst["r"] = max(worst["r"], float(np.abs(np.linalg.norm(X, axis=0) - p / (1 + e * np.cos(f))).max()))
        P, Q, h = K.basis(i, om, w)
        M = np.stack([P, Q, h])
        worst["orth"] = max(worst["orth"], float(np.abs(M @ M.T - np.eye(3)).max()))
        Rm = R3(om) @ R1(i) @ R3(w)
        worst["rot"] = max(worst["rot"], float(np.abs(Rm[:, 0] - P).max() + np.abs(Rm[:, 1] - Q).max() + np.abs(Rm[:, 2] - h).max()))
    print("  worst |z at nodes| %.2e au, | |r| - p/(1+e cos f) | %.2e, |B B^T - I| %.2e, basis vs matrices %.2e" %
          (worst["z"], worst["r"], worst["orth"], worst["rot"]))
    check("K1", max(worst.values()) < 1e-12, worst)
    rep["K1"] = worst

    print("K2: exact MOID vs the independent brute force; bound order L0 <= L_lon <= MOID <= U_lon <= U_node")
    rows = synth(a.n, 7) + EDGES
    t0 = time.time()
    dev, order_bad, cert_bad, gaps, ms = [], 0, 0, [], []
    for k, row in enumerate(rows):
        el = K.elements(row)
        m = K.moid_certified(el)
        ref = ref_moid(row["a"], row["e"], math.radians(row["i"]), math.radians(row["om"]), math.radians(row["w"]))
        dev.append(abs(m["moid"] - ref))
        L0, _ = K.radial_gap(el)
        U_node = float(K.nodal(el)["U_node"])
        tol = K.TOL_D
        ok = (float(L0) <= m["moid"] + tol) and (m["L_lon"] <= m["moid"] + tol) and \
             (m["moid"] <= m["U_lon"] + tol) and (m["U_lon"] <= U_node + tol)
        order_bad += (not ok)
        cert_bad += (not m["certified"])
        gaps.append(m["U_lon"] - m["L_lon"])
        ms.append(m["moid"])
        if dev[-1] > 1e-8 or not ok:
            print("  row %d dev %.2e ok %s  %s moid %.9f ref %.9f" % (k, dev[-1], ok, row, m["moid"], ref))
    dev = np.array(dev)
    print("  %d orbits (%d synthetic + %d edge cases) in %.0f s" % (len(rows), a.n, len(EDGES), time.time() - t0))
    print("  |MOID - brute force|: max %.2e au, median %.2e;  bound order violated on %d;  uncertified %d" %
          (dev.max(), np.median(dev), order_bad, cert_bad))
    g, mm = np.array(gaps), np.array(ms)
    near = mm < 0.2
    print("  certified gap U_lon - L_lon where MOID < 0.2 au: max %.2e au (n=%d)" % (g[near].max(), near.sum()))
    check("K2 vs brute force", dev.max() < 1e-8, dev.max())
    check("K2 bound order", order_bad == 0, order_bad)
    rep["K2"] = dict(n=len(rows), max_dev=float(dev.max()), median_dev=float(np.median(dev)),
                     order_violations=order_bad, uncertified=cert_bad, max_gap_near=float(g[near].max()))

    print("K2: vectorised moid_batch vs moid (elliptic)")
    rr = synth(200, 11)
    els = [K.elements(r) for r in rr]
    mb, _ = K.moid_batch([e["p"] for e in els], [e["e"] for e in els], [e["i"] for e in els],
                         [e["om"] for e in els], [e["w"] for e in els], N=1440)
    ex = np.array([K.moid(e)["moid"] for e in els])
    d = np.abs(mb - ex)
    print("  max |batch - exact| %.2e au (n=%d)" % (d.max(), len(d)))
    check("moid_batch", d.max() < 1e-9, d.max())
    rep["moid_batch_max_dev"] = float(d.max())

    print("K4: closed-form gradients vs central finite differences")
    worst_n, worst_m = 0.0, 0.0
    for row in synth(60, 21):
        el = K.elements(row)
        ga, gd = K.nodal_grad(el)
        x0 = np.array([float(el[k]) for k in ("a", "e", "i", "om", "w")])
        m0 = K.moid(el)
        gm = K.moid_grad(el, m0)
        for j in range(5):
            hstep = 1e-6 * (1 if j else max(1.0, x0[0]))
            vals = []
            for sgn in (1, -1):
                x = x0.copy(); x[j] += sgn * hstep
                ek = dict(a=x[0], e=x[1], i=x[2], om=x[3], w=x[4], p=x[0] * (1 - x[1] ** 2))
                nd = K.nodal(ek)
                mk = K.moid(ek)
                sgn_m = np.sign(mk["nhat"] @ m0["nhat"])
                vals.append((float(nd["asc"]["delta"]), float(nd["desc"]["delta"]), sgn_m * mk["signed"]))
            fd = [(vals[0][c] - vals[1][c]) / (2 * hstep) for c in range(3)]
            worst_n = max(worst_n, abs(fd[0] - ga[j]), abs(fd[1] - gd[j]))
            if m0["signed_ok"] and m0["second"] - m0["moid"] > 1e-3:
                worst_m = max(worst_m, abs(fd[2] - gm[j]))
    print("  worst |FD - closed form|: nodal %.2e, signed MOID (envelope theorem) %.2e" % (worst_n, worst_m))
    check("nodal_grad", worst_n < 1e-6, worst_n)
    check("moid_grad", worst_m < 1e-5, worst_m)
    rep["grad"] = dict(nodal=worst_n, moid=worst_m)

    print("K4: delta method vs Monte Carlo vs unscented (synthetic sigmas scaled so P is not 0 or 1)")
    rng = np.random.default_rng(5)
    agree, tested, rows4, worst_lin, worst_ut = 0, 0, [], 0.0, 0.0
    for row in synth(120, 31):
        el = K.elements(row)
        m0 = K.moid(el)
        if not (0.01 < m0["moid"] < 0.12 and m0["signed_ok"]) or tested >= 16:
            continue
        pattern = np.array([1e-4 * row["a"], 1e-4, math.radians(1e-3), math.radians(5e-3), math.radians(5e-3)])
        s1 = float(math.sqrt(np.sum((K.moid_grad(el, m0) * pattern) ** 2)))
        target = abs(m0["moid"] - K.D0) / rng.uniform(0.3, 2.5)          # P lands between ~0.01 and ~0.6
        sig = pattern * (target / s1)
        s = target
        if m0["second"] - m0["moid"] < 5 * s or s > 0.02:
            continue
        Plin = K.p_fold(m0["signed"], s)
        mc = K.p_sampled(el, sig, n=4000, seed=tested)
        ut = K.p_unscented(el, sig)
        inside = mc["lo"] - 0.01 <= Plin <= mc["hi"] + 0.01
        agree += inside
        tested += 1
        worst_lin = max(worst_lin, abs(Plin - mc["P"]))
        if ut["ok"]:
            worst_ut = max(worst_ut, abs(ut["P"] - mc["P"]))
        rows4.append(dict(moid=m0["moid"], s=s, P_lin=Plin, P_mc=mc["P"], mc_lo=mc["lo"], mc_hi=mc["hi"],
                          P_ut=ut["P"], ut_ok=ut["ok"]))
        print("  MOID %.5f  s %.2e  P_lin %.4f  P_mc %.4f [%.4f, %.4f]  P_ut %.4f%s" %
              (m0["moid"], s, Plin, mc["P"], mc["lo"], mc["hi"], ut["P"], "" if ut["ok"] else " (UT flagged)"))
    print("  delta method inside the MC 95%% interval (+-0.01) on %d of %d; max |P_lin - P_mc| %.4f, "
          "max |P_ut - P_mc| %.4f" % (agree, tested, worst_lin, worst_ut))
    check("K4 lin vs MC", agree == tested and tested >= 8, (agree, tested))
    rep["K4"] = dict(tested=tested, agree=agree, max_lin_vs_mc=worst_lin, max_ut_vs_mc=worst_ut, rows=rows4)

    print("K4: bivariate nodal probability vs Monte Carlo of U_node (same synthetic sigmas)")
    worst_node, nn = 0.0, 0
    for row in synth(60, 41):
        el = K.elements(row)
        nd = K.nodal(el)
        if not (0.005 < float(nd["U_node"]) < 0.15):
            continue
        sig = np.array([3e-3 * row["a"], 3e-3, math.radians(0.05), math.radians(0.2), math.radians(0.2)]) * rng.uniform(0.3, 3)
        pn = K.p_nodal(el, sig)
        smp, _ = K.sample_elements(el, sig, 20000, np.random.default_rng(nn))
        U = K.nodal(dict(p=smp["p"], e=smp["e"], i=smp["i"], om=smp["om"], w=smp["w"]))["U_node"]
        Pmc = float((U <= K.D0).mean())
        worst_node = max(worst_node, abs(pn["P"] - Pmc))
        nn += 1
        if nn >= 12:
            break
    print("  max |P_nodal (bivariate normal) - P_MC(U_node)| %.4f over %d orbits" % (worst_node, nn))
    rep["K4_nodal"] = dict(n=nn, max_dev=worst_node)

    print("K5: the fitter recovers known (beta, w) from synthetic labels")
    rng = np.random.default_rng(9)
    n = 20000
    dt = rng.uniform(0, 0.12, n); s = rng.uniform(0, 0.004, n); H = rng.uniform(16, 26, n)
    beta_t, w_t = 0.0007, 0.0015
    Sv = np.array([K.score(dt[j], s[j], H[j], beta_t, w_t) for j in range(n)])
    y = (rng.uniform(size=n) < Sv).astype(float)
    b, w, _ = K.fit_score(dt, s, H, y)
    print("  true beta %.5f w %.5f -> fitted beta %.5f w %.5f" % (beta_t, w_t, b, w))
    check("K5 fit", abs(b - beta_t) < 3e-4 and abs(w - w_t) < 5e-4, (b, w))
    rep["K5"] = dict(true=(beta_t, w_t), fitted=(b, w))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(report=rep, fails=FAILS), open(a.out, "w"), indent=1, default=float)
    print("=" * 70)
    print("ALL CHECKS PASS" if not FAILS else "FAILURES: %s" % FAILS)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
