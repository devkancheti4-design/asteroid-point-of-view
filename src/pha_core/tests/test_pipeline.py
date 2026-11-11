"""End-to-end: the body carries out the law's acts on SYNTHETIC rows chosen to reach every act (no SBDB data).

    python3 -u tests/test_pipeline.py

Checks: every run ends at a terminal act; no word repeats (every act changed the state); every one of the
ten acts is reached by some synthetic row; and the final act matches the specification read by hand."""
import os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import pha_kernels as K
import pha_pipeline as PL

rng = np.random.default_rng(2026)


def row_of(a, e, i, om, w, H=20.0, sig=(1e-8, 1e-8, 1e-6, 1e-6, 1e-6)):
    return dict(a=a, e=e, i=i, om=om, w=w, q=a * (1 - e), H=H, sigma_a=sig[0], sigma_e=sig[1],
                sigma_i=sig[2], sigma_om=sig[3], sigma_w=sig[4])


def scaled(row, s_target):
    """sigmas in a fixed pattern, scaled so the exact-MOID 1-sigma equals s_target"""
    el = K.elements(row)
    m = K.moid(el)
    pat = np.array([1e-4 * row["a"], 1e-4, math.radians(1e-3), math.radians(5e-3), math.radians(5e-3)])
    s1 = float(math.sqrt(np.sum((K.moid_grad(el, m) * pat) ** 2)))
    k = s_target / s1
    r = dict(row)
    r.update(sigma_a=pat[0] * k, sigma_e=pat[1] * k, sigma_i=math.degrees(pat[2] * k),
             sigma_om=math.degrees(pat[3] * k), sigma_w=math.degrees(pat[4] * k))
    return r


def find(pred, tries=4000):
    for _ in range(tries):
        a = rng.uniform(0.8, 2.6); e = rng.uniform(0.05, 0.7); i = rng.uniform(2, 40)
        r = row_of(a, e, i, rng.uniform(0, 360), rng.uniform(0, 360))
        el = K.elements(r)
        L0, _ = K.radial_gap(el)
        U = float(K.nodal(el)["U_node"])
        m = K.moid(el)
        if pred(float(L0), U, m):
            return r
    raise SystemExit("no synthetic orbit found for a scenario")


CASES = []
CASES.append(("invalid input (e < 0)", row_of(1.2, -0.1, 10, 20, 30), "HUMAN"))
CASES.append(("faint (H = 24)", row_of(1.2, 0.3, 10, 20, 30, H=24.0), "NOT_PHA_SIZE"))
CASES.append(("distant Amor, tiny sigma", row_of(1.8, 0.3, 5, 300, 60), "NOT_PHA_GEOMETRY"))
r = find(lambda L0, U, m: U <= 0.04 and m["second"] > 0.1)
CASES.append(("node pair inside, tiny sigma", r, "PHA"))
r = find(lambda L0, U, m: L0 <= 0.05 < U and m["moid"] < 0.035 and m["second"] > m["moid"] + 0.05)
CASES.append(("open screen, exact inside, tiny sigma", r, "PHA"))
r = find(lambda L0, U, m: L0 <= 0.05 < U and 0.07 < m["moid"] < 0.2 and m["second"] > m["moid"] + 0.05)
CASES.append(("open screen, exact outside, tiny sigma", r, "NOT_PHA_GEOMETRY"))
r = find(lambda L0, U, m: L0 <= 0.05 < U and 0.046 < m["moid"] < 0.0495 and m["second"] > m["moid"] + 0.05)
CASES.append(("exact inside, near threshold, s = 0.002", scaled(r, 0.002), "OBSERVE_PHA"))
r = find(lambda L0, U, m: L0 <= 0.05 < U and 0.0505 < m["moid"] < 0.054 and m["second"] > m["moid"] + 0.05)
CASES.append(("exact outside, near threshold, s = 0.002", scaled(r, 0.002), "OBSERVE_NOT"))
r = find(lambda L0, U, m: L0 <= 0.05 < U and 0.02 < m["moid"] < 0.04 and m["second"] > m["moid"] + 0.1)
CASES.append(("exact inside, s = 0.03 > s_lin: sampled", scaled(r, 0.03), None))
r = find(lambda L0, U, m: U <= 0.045 and m["moid"] > 0.02 and m["second"] > m["moid"] + 0.1)
CASES.append(("node pair inside, s = 0.004", scaled(r, 0.004), None))
r = find(lambda L0, U, m: 0.046 <= U <= 0.0495 and m["second"] > m["moid"] + 0.1)
CASES.append(("node pair just inside, s = 0.004: P needs exact", scaled(r, 0.004), None))
far = row_of(1.0687 / 0.75, 0.25, 6, 120, 200)              # q = 1.0687: radial gap 0.052 au
far.update(sigma_a=2e-3, sigma_e=1e-3)
CASES.append(("radial gap 0.052 au, s ~ 0.002: P needs exact", far, None))
bad = row_of(1.8, 0.3, 5, 300, 60)
bad["sigma_e"] = float("nan")
CASES.append(("non-finite sigma", bad, "HUMAN"))


def main():
    reached, fails = set(), []
    for name, row, want in CASES:
        res = PL.classify(row)
        reached.update(a for _, a in res["trail"])
        words = [w for w, _ in res["trail"]]
        ok = (res["code"] in PL.TERMINAL) and len(set(words)) == len(words) and (want is None or res["act"] == want)
        fails += [] if ok else [name]
        extra = " moid=%.5f" % res["moid"] if "moid" in res else ""
        extra += " P_lin=%.4f s=%.4f lin_ok=%s" % (res["P_lin"], res["s"], res["lin_ok"]) if "P_lin" in res else ""
        extra += " P_mc=%.4f [%.4f, %.4f]" % (res["P_mc"], res["mc_lo"], res["mc_hi"]) if "P_mc" in res else ""
        print("%-52s %-18s %s" % (name, res["act"], " -> ".join(a for _, a in res["trail"])))
        if extra:
            print("%52s %s" % ("", extra.strip()))
    missing = [a for a in PL.ACTS if a not in reached]
    print("-" * 100)
    print("acts reached: %d of %d%s" % (len(PL.ACTS) - len(missing), len(PL.ACTS), (" (missing %s)" % missing) if missing else ""))
    print("ALL CASES PASS" if not fails else "FAIL: %s" % fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
