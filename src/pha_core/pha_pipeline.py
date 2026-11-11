"""The body: carries out the PHA decision law's acts until a terminal act.

    classify(row) -> dict(act=..., trail=[(word, act), ...], quantities...)

The law (out/pha_law.py, machine-authored and proven over all 2^32 words) reads only the situation word.
This body computes the kernels, packs the word, and executes what the law rules. Every act changes the
word: EXACT and EXACT_FOR_P set CERT to EXACT_* or BAD and reset the probability; LINEAR sets UNC to 1
or 2; SAMPLE sets UNC to 3. So the loop ends, in at most 6 steps.

POLICY holds the named, non-fitted policy parameters (defaults stated in PHA_MATH_CORE.md, section K5)."""
import os, sys, math
import numpy as np
from scipy.special import ndtr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "law")); sys.path.insert(0, os.path.join(HERE, "out"))
import pha_kernels as K
from spec import pack, ACTS, CERT, UNC, PB
from pha_law import pha_law

POLICY = dict(alpha=float(ndtr(-3.0)),   # a call is "sure" beyond 3 sigma (one-sided tail 0.00135)
              s_lin=0.01,                # au: largest 1-sigma proximity spread for which K4(a) is trusted
              k_gap=5.0,                 # the runner-up local minimum must sit >= k_gap sigma above the MOID
              sin_min=0.05,              # tangent angle at the minimum >= ~2.9 deg (signed MOID smooth)
              n_mc=4000, seed=0,
              s_sys=0.0)                 # au: 1-sigma systematic between our fixed Earth curve and the label's
                                         # (PHA_MATH_CORE.md section 0, item 7). 0 reproduces the law as delivered.
                                         # When > 0: it is added in quadrature to every linearised spread, and a
                                         # cheap certificate is accepted only if it clears d0 by 3 * s_sys.
TERMINAL = {0, 1, 5, 6, 8, 9}


def _pb_two_sided(P, alpha):
    if not np.isfinite(P):
        return PB["BAD"]
    return PB["SURE_IN"] if P >= 1 - alpha else PB["SURE_OUT"] if P <= alpha else PB["MIDDLE"]


def classify(row, policy=POLICY, law=pha_law):
    el = K.elements(row)
    H = float(row["H"])
    st = dict(inv=0, faint=0, cert=CERT["OPEN"], unc=UNC["NONE"], pb=0)
    q = dict()
    if not bool(K.valid(el, H)):
        st["inv"] = 1
    elif H > K.H0:
        st["faint"] = 1
    else:
        L0, side = K.radial_gap(el)
        nd = K.nodal(el)
        q.update(L0=float(L0), side=int(side), U_node=float(nd["U_node"]))
        margin = 3.0 * policy.get("s_sys", 0.0)
        st["cert"] = (CERT["FAR"] if q["L0"] > K.D0 + margin else
                      CERT["NEAR"] if q["U_node"] <= K.D0 - margin else CERT["OPEN"])
    try:
        sig = K.sigmas(row)
    except (KeyError, TypeError):
        sig = np.full(5, np.nan)
    trail, seen = [], set()
    for _ in range(8):
        n = pack(**st)
        act = law(n)
        trail.append((n, ACTS[act]))
        assert n not in seen, "the body repeated a word: an act did nothing (%s)" % trail
        seen.add(n)
        if act in TERMINAL:
            return dict(act=ACTS[act], code=act, trail=trail, **q)
        if act in (2, 7):                                    # EXACT, EXACT_FOR_P
            m = K.moid_certified(el)
            q.update(moid=m["moid"], signed=m["signed"], L=m["L"], U=m["U"], certified=m["certified"])
            q["_m"] = m
            st["cert"] = (CERT["EXACT_NEAR"] if m["moid"] <= K.D0 else CERT["EXACT_FAR"]) if m["certified"] else 5
            st["unc"], st["pb"] = UNC["NONE"], 0             # the probability must be recomputed on the exact MOID
        elif act == 3:                                       # LINEAR: K4(a) on whichever certificate is held
            if not np.all(np.isfinite(sig)):
                st["unc"], st["pb"] = UNC["LINEAR"], PB["BAD"]
                continue
            c = st["cert"]
            s_sys2 = policy.get("s_sys", 0.0) ** 2
            if c == CERT["FAR"]:                             # upper bound on P(hazard): P(L0 quantity <= d0)
                g = K.radial_grad(el, q["side"])
                s = float(math.sqrt(np.sum((g * sig) ** 2) + s_sys2))
                P = K.p_below(q["L0"], s)
                pb = PB["SURE_OUT"] if P <= policy["alpha"] else PB["MIDDLE"]
                lin_ok = s <= policy["s_lin"]
            elif c == CERT["NEAR"]:                          # lower bound on P(hazard): P(U_node <= d0)
                r = K.p_nodal(el, sig)
                s, P = max(r["s"]), r["P"]
                pb = PB["SURE_IN"] if P >= 1 - policy["alpha"] else PB["MIDDLE"]
                lin_ok = s <= policy["s_lin"]
            else:                                            # exact: folded normal on the signed MOID
                m = q["_m"]
                gm = K.moid_grad(el, m)
                s = float(math.sqrt(np.sum((gm * sig) ** 2) + s_sys2))
                P = K.p_fold(m["signed"], s)
                pb = _pb_two_sided(P, policy["alpha"])
                lin_ok = (m["signed_ok"] and (m["second"] - m["moid"]) >= policy["k_gap"] * s and s <= policy["s_lin"])
            q.update(P_lin=P, s=s, lin_ok=bool(lin_ok))
            if not np.isfinite(P):
                st["unc"], st["pb"] = UNC["LINEAR"], PB["BAD"]
            elif lin_ok:
                st["unc"], st["pb"] = UNC["LINEAR"], pb
            else:
                st["unc"], st["pb"] = UNC["NEEDS_SAMPLE"], 0
        elif act == 4:                                       # SAMPLE: K4(b) Monte Carlo on the exact MOID
            mc = K.p_sampled(el, sig, n=policy["n_mc"], seed=policy["seed"])
            q.update(P_mc=mc["P"], mc_lo=mc["lo"], mc_hi=mc["hi"], mc_n=mc["n"], mc_rejected=mc["rejected"])
            if mc["n"] == 0:
                pb = PB["BAD"]
            else:
                a_ = policy["alpha"]
                pb = PB["SURE_IN"] if mc["lo"] >= 1 - a_ else PB["SURE_OUT"] if mc["hi"] <= a_ else PB["MIDDLE"]
            st["unc"], st["pb"] = UNC["SAMPLED"], pb
    raise AssertionError("no terminal act in 8 steps: %s" % trail)
