"""Real-data checks of the K1-K5 kernels and the PHA decision law (src/pha_core) on the JPL NEA catalogue.

1. Certified exact MOID, K2(b), for every labelled NEA: against JPL's MOID column and against our own solver
   (src/orbit_geometry.py; same fixed Earth curve, independent implementation).
2. Certified screening: share of bright NEAs (H <= 22) settled by the closed-form bounds alone
   (radial gap L0 > 0.05 au, or node pair U_node <= 0.05 au), and their agreement with JPL's MOID.
3. K5(a) score: beta, w fitted by maximum likelihood on training folds (object-level, s = 0, i.e. label
   reproduction), evaluated on held-out folds.
4. The decision procedure (pha_pipeline.classify) on every NEA with its real 1-sigma element uncertainties:
   counts per act, coverage, and agreement of the decided classes with JPL's flag.
5. Delta method vs Monte Carlo on real orbits whose hazard probability is neither 0 nor 1.

JPL's `moid` and `pha` columns are used only for evaluation (and `pha` as the K5 training target).
"""
from __future__ import annotations

import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent / "pha_core"))
import pha_kernels as K  # noqa: E402
import pha_pipeline as PL  # noqa: E402

from common import N_FOLDS, RESULTS, SEED, binary_metrics, load_sbdb, save_json  # noqa: E402

FIELDS = ["pdes", "a", "e", "q", "i", "om", "w", "H", "sigma_a", "sigma_e", "sigma_i", "sigma_om", "sigma_w"]
N_WORKERS = 10
N_MC_CHECK = 300


def kernel_row(row: dict) -> dict:
    t0 = time.perf_counter()
    el = K.elements(row)
    L0, _ = K.radial_gap(el)
    nd = K.nodal(el)
    m = K.moid_certified(el)
    out = dict(pdes=row["pdes"], L0=float(L0), U_node=float(nd["U_node"]), d_lin_k=float(nd["d_lin"]),
               moid_k=m["moid"], signed=m["signed"], L=m["L"], U=m["U"], certified=bool(m["certified"]),
               signed_ok=bool(m["signed_ok"]), second=m["second"])
    try:
        sig = K.sigmas(row)
        out["s_delta"] = float(math.sqrt(np.sum((K.moid_grad(el, m) * sig) ** 2))) if np.all(np.isfinite(sig)) else np.nan
    except (KeyError, TypeError, ValueError):
        out["s_delta"] = np.nan
    out["ms"] = 1e3 * (time.perf_counter() - t0)
    return out


def pipeline_row(row: dict) -> dict:
    t0 = time.perf_counter()
    r = PL.classify(row)
    keep = {k: r.get(k) for k in ("L0", "U_node", "moid", "certified", "P_lin", "s", "lin_ok", "P_mc", "mc_lo",
                                  "mc_hi", "mc_n", "mc_rejected")}
    return dict(pdes=row["pdes"], act=r["act"], steps=len(r["trail"]), trail=" > ".join(a for _, a in r["trail"]),
                ms=1e3 * (time.perf_counter() - t0), **keep)


def pipeline_row_sys(args) -> dict:
    row, s_sys = args
    r = PL.classify(row, policy={**PL.POLICY, "s_sys": s_sys})
    return dict(pdes=row["pdes"], act=r["act"], steps=len(r["trail"]), moid=r.get("moid"), P_lin=r.get("P_lin"))


def decision_summary(pr: pd.DataFrame) -> dict:
    """Coverage and agreement of decided classes with JPL's flag and with the definition on JPL's MOID."""
    dec_pha, dec_not = pr["act"].eq("PHA"), pr["act"].isin(["NOT_PHA_SIZE", "NOT_PHA_GEOMETRY"])
    dec, cls = dec_pha | dec_not, dec_pha.astype(int)
    wrong = dec & (cls != pr["label"])
    contradict = pr["label"] != pr["def_jpl"]
    return {
        "acts": {k: int(v) for k, v in pr["act"].value_counts().items()},
        "coverage": float(dec.mean()),
        "decided_agreement_with_jpl_flag": float((cls[dec] == pr["label"][dec]).mean()),
        "decided_disagreements_with_flag": int(wrong.sum()),
        "of_which_flag_contradicts_definition": int((wrong & contradict).sum()),
        "decided_agreement_with_definition_on_jpl_moid": float((cls[dec] == pr["def_jpl"][dec]).mean()),
        "decided_disagreements_with_definition": int((dec & (cls != pr["def_jpl"])).sum()),
        "unresolved": int(pr["act"].isin(["OBSERVE_PHA", "OBSERVE_NOT"]).sum()),
        "human": int(pr["act"].eq("HUMAN").sum()),
    }


def mc_check_row(row: dict) -> dict:
    el, sig = K.elements(row), K.sigmas(row)
    mc = K.p_sampled(el, sig, n=4000, seed=0)
    return dict(pdes=row["pdes"], P_mc=mc["P"], mc_lo=mc["lo"], mc_hi=mc["hi"], mc_n=mc["n"])


def run_parallel(fn, rows: list[dict], label: str) -> pd.DataFrame:
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        out = list(ex.map(fn, rows, chunksize=64))
    print(f"{label}: {len(rows):,} objects in {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(out)


def main() -> None:
    sbdb = load_sbdb()
    sbdb = sbdb[sbdb["moid"].notna()].reset_index(drop=True)
    ours = pd.read_csv(RESULTS / "asteroid_features.csv", usecols=["pdes", "moid_path"])
    rows = sbdb[FIELDS].to_dict("records")
    y = sbdb["label"].to_numpy(int)
    bright = sbdb["H"].to_numpy() <= K.H0

    # 1) certified exact MOID on the real catalogue
    kr = run_parallel(kernel_row, rows, "K2(b) certified MOID")
    kr = kr.merge(ours, on="pdes", how="left")
    kr["moid_jpl"], kr["H"], kr["label"] = sbdb["moid"].values, sbdb["H"].values, y
    kr.to_csv(RESULTS / "kernel_rows.csv", index=False)
    d_jpl = (kr["moid_k"] - kr["moid_jpl"]).abs()
    d_ours = (kr["moid_k"] - kr["moid_path"]).abs()
    rule_k = ((kr["moid_k"] <= K.D0) & (kr["H"] <= K.H0)).astype(int)
    summary = {
        "n": int(len(kr)),
        "certified_share": float(kr["certified"].mean()),
        "ms_per_object_median": float(kr["ms"].median()),
        "abs_diff_vs_jpl_au": {q: float(d_jpl.quantile(p)) for q, p in [("median", .5), ("p99", .99), ("max", 1.0)]},
        "abs_diff_vs_our_solver_au": {q: float(d_ours.quantile(p)) for q, p in [("median", .5), ("p99", .99), ("max", 1.0)]},
        "rule_on_kernel_moid": binary_metrics(y, rule_k.values),
        "rule_label_disagreements_vs_our_solver": int((rule_k != ((kr["moid_path"] <= K.D0) & (kr["H"] <= K.H0))).sum()),
    }

    # 2) certified screening of bright NEAs by closed-form bounds
    b = kr[bright]
    far, near = b["L0"] > K.D0, b["U_node"] <= K.D0
    summary["screening"] = {
        "bright": int(len(b)),
        "far_share": float(far.mean()), "near_share": float(near.mean()), "open_share": float((~far & ~near).mean()),
        "far_but_jpl_moid_le_d0": int((far & (b["moid_jpl"] <= K.D0)).sum()),
        "near_but_jpl_moid_gt_d0": int((near & (b["moid_jpl"] > K.D0)).sum()),
        "d_lin_defined_share": float(b["d_lin_k"].notna().mean()),
    }

    # 3) K5(a) score: (beta, w) by maximum likelihood on training folds, s = 0 (label reproduction)
    k5 = []
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    zeros = np.zeros(len(kr))
    for k, (tr, te) in enumerate(skf.split(kr, y)):
        beta, w, _ = K.fit_score(kr["signed"].values[tr], zeros[tr], kr["H"].values[tr], y[tr])
        S = np.array([K.score(d, 0.0, h, beta, w) for d, h in zip(kr["signed"].values[te], kr["H"].values[te])])
        m_fit = binary_metrics(y[te], (S >= 0.5).astype(int), S)
        m_raw = binary_metrics(y[te], rule_k.values[te])
        k5.append({"fold": k, "beta_au": beta, "w_au": w, **{f"fit_{a}": v for a, v in m_fit.items()},
                   **{f"rule_{a}": v for a, v in m_raw.items()}})
    k5 = pd.DataFrame(k5)
    k5.to_csv(RESULTS / "kernel_k5_folds.csv", index=False)
    summary["k5"] = {c: float(k5[c].mean()) for c in k5.columns if c != "fold"}
    summary["k5"].update(beta_std=float(k5["beta_au"].std()), w_std=float(k5["w_au"].std()))
    print("K5 folds:\n", k5[["fold", "beta_au", "w_au", "fit_accuracy", "fit_f1", "rule_accuracy", "rule_f1"]].round(6))

    # 4) the decision procedure with real 1-sigma uncertainties
    pr = run_parallel(pipeline_row, rows, "decision procedure")
    pr["label"], pr["H"] = y, sbdb["H"].values
    pr.to_csv(RESULTS / "pipeline_rows.csv", index=False)
    counts = pr["act"].value_counts().to_dict()
    decided_pha, decided_not = pr["act"].eq("PHA"), pr["act"].isin(["NOT_PHA_SIZE", "NOT_PHA_GEOMETRY"])
    decided = decided_pha | decided_not
    nominal = (pr["act"].isin(["PHA", "OBSERVE_PHA"])).astype(int)
    summary["decision"] = {
        "acts": {k: int(v) for k, v in counts.items()},
        "coverage": float(decided.mean()),
        "decided_agreement_with_jpl": float(((decided_pha & (pr["label"] == 1)) | (decided_not & (pr["label"] == 0)))[decided].mean()),
        "decided_disagreements": int(((decided_pha & (pr["label"] == 0)) | (decided_not & (pr["label"] == 1))).sum()),
        "unresolved": int(pr["act"].isin(["OBSERVE_PHA", "OBSERVE_NOT"]).sum()),
        "human": int(pr["act"].eq("HUMAN").sum()),
        "nominal_label_metrics_all": binary_metrics(y[pr["act"] != "HUMAN"], nominal[pr["act"] != "HUMAN"]),
        "steps_max": int(pr["steps"].max()),
        "ms_per_object_median_bright": float(pr.loc[bright, "ms"].median()),
    }
    dis = pr[(decided_pha & (pr["label"] == 0)) | (decided_not & (pr["label"] == 1))]
    summary["decision"]["disagreement_H_range"] = [float(dis["H"].min()), float(dis["H"].max())] if len(dis) else None

    # 5) delta method vs Monte Carlo on real orbits with a non-trivial probability
    cand = pr[(pr["lin_ok"] == True) & pr["P_lin"].between(0.01, 0.99)]  # noqa: E712
    cand = cand.sample(min(N_MC_CHECK, len(cand)), random_state=SEED)
    by_id = {r["pdes"]: r for r in rows}
    mc = run_parallel(mc_check_row, [by_id[p] for p in cand["pdes"]], "Monte Carlo check") if len(cand) else pd.DataFrame()
    if len(mc):
        cmp = cand[["pdes", "P_lin"]].merge(mc, on="pdes")
        cmp["abs_diff"] = (cmp["P_lin"] - cmp["P_mc"]).abs()
        cmp["inside"] = (cmp["P_lin"] >= cmp["mc_lo"] - 0.01) & (cmp["P_lin"] <= cmp["mc_hi"] + 0.01)
        cmp.to_csv(RESULTS / "kernel_delta_vs_mc.csv", index=False)
        summary["delta_vs_mc"] = {"n": int(len(cmp)), "inside_share": float(cmp["inside"].mean()),
                                  "median_abs_diff": float(cmp["abs_diff"].median()), "max_abs_diff": float(cmp["abs_diff"].max())}

    # 6) the same procedure with the Earth-curve systematic s_sys, measured on the training folds only
    #    (RMS of our MOID minus JPL's for training asteroids with JPL MOID < 0.2 au; JPL's MOID never enters a
    #    test decision)
    def_jpl = ((sbdb["moid"] <= K.D0) & (sbdb["H"] <= K.H0)).astype(int).to_numpy()
    pr["def_jpl"] = def_jpl
    summary["decision"] = {**summary["decision"], **decision_summary(pr)}
    parts = []
    for k, (tr, te) in enumerate(skf.split(kr, y)):
        trk = kr.iloc[tr]
        close = trk["moid_jpl"] < 0.2
        s_sys = float(np.sqrt(np.mean((trk.loc[close, "moid_k"] - trk.loc[close, "moid_jpl"]) ** 2)))
        out = run_parallel(pipeline_row_sys, [(rows[j], s_sys) for j in te], f"decision procedure, fold {k}, s_sys {s_sys:.2e} au")
        out["fold"], out["s_sys"], out["label"], out["def_jpl"] = k, s_sys, y[te], def_jpl[te]
        parts.append(out)
    ps = pd.concat(parts, ignore_index=True)
    ps.to_csv(RESULTS / "pipeline_rows_sys.csv", index=False)
    summary["decision_sys"] = {"s_sys_au_mean": float(ps.groupby("fold")["s_sys"].first().mean()),
                               "s_sys_au_by_fold": [float(v) for v in ps.groupby("fold")["s_sys"].first()],
                               **decision_summary(ps)}
    save_json(summary, RESULTS / "kernel_summary.json")
    print(json_dump(summary))


def json_dump(obj) -> str:
    import json
    return json.dumps(obj, indent=1, default=float)


if __name__ == "__main__":
    main()
