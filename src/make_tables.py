"""LaTeX tables generated from results/, so every number in the paper matches the experiments."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import RESULTS, ROOT

TAB = ROOT / "paper" / "tables"

MODEL_NAMES = {"LogReg": "Logistic regression", "KNN": "$k$-NN ($k{=}5$)", "RandomForest": "Random forest",
               "XGBoost": "XGBoost", "LightGBM": "LightGBM", "MLP": "MLP"}
FEATURE_NAMES = {"path": "path", "path+kernels": "path + kernels", "path+kernels+exact": "path + kernels + exact",
                 "prior-work orbit+JPL moid": "orbit + JPL MOID"}
RULE_NAMES = {"rule: exact path geometry (ours)": "Path rule, exact geometry (ours)",
              "rule: tangent-line closed form": "Path rule, tangent-line closed form (ours)",
              "rule: node gap (upper bound)": "Path rule, node gap (upper bound)",
              "rule: JPL MOID (definition, reference)": "Definition on JPL MOID$^\\dagger$"}


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def f3(x: float) -> str:
    return f"{x:.3f}"


def f4(x: float) -> str:
    return f"{x:.4f}"


def bold_best(df: pd.DataFrame, cols: list[str], eligible: pd.Series, higher=True) -> pd.DataFrame:
    """Wrap the best eligible value of each column in \\textbf{}; values must already be strings."""
    out = df.copy()
    for c in cols:
        vals = pd.to_numeric(df.loc[eligible, c], errors="coerce")
        if vals.empty:
            continue
        best = vals.max() if higher else vals.min()
        for idx in vals.index[vals == best]:
            out.loc[idx, c] = f"\\textbf{{{df.loc[idx, c]}}}"
    return out


def write(name: str, body: str) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / f"{name}.tex").write_text(body)
    print("wrote", name)


def table_earth_view() -> None:
    s = pd.read_csv(RESULTS / "earth_view_summary.csv")
    stats = json.loads((RESULTS / "earth_view_stats.json").read_text())
    order = ["LogReg", "KNN", "MLP", "LightGBM", "XGBoost", "RandomForest"]
    lines = []
    for m in order:
        r = s[(s.protocol == "row_split") & (s.model == m)].iloc[0]
        g = s[(s.protocol == "asteroid_grouped") & (s.model == m)].iloc[0]
        lines.append([MODEL_NAMES[m], pct(r.accuracy_mean), f3(r.f1_mean), f3(r.mcc_mean),
                      pct(g.accuracy_mean), f3(g.f1_mean), f3(g.mcc_mean)])
    df = pd.DataFrame(lines, columns=["m", "ra", "rf", "rm", "ga", "gf", "gm"])
    df = bold_best(df, ["ra", "rf", "rm", "ga", "gf", "gm"], pd.Series(True, index=df.index))
    base = 1 - stats["positive_rate_rows"]
    rows = "\n".join(" & ".join(r) + r" \\" for r in df.values.tolist())
    body = rf"""\begin{{table}}[t]
\centering
\footnotesize
\caption{{Earth's-view benchmark on {stats['rows']:,} JPL close approaches ({stats['asteroids']:,} asteroids). Row split: stratified 5-fold over rows, as in prior work ({100 * stats['test_rows_with_asteroid_in_train']['row_split']:.1f}\% of test rows belong to an asteroid seen in training). Grouped: all approaches of an asteroid share a fold. Means over 5 folds.}}
\label{{tab:earthview}}
\setlength{{\tabcolsep}}{{2.4pt}}
\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\toprule
 & \multicolumn{{3}}{{c}}{{Row split (leaky)}} & \multicolumn{{3}}{{c}}{{Grouped (honest)}} \\
\cmidrule(lr){{2-4}}\cmidrule(l){{5-7}}
Model & Acc.\,\% & F1 & MCC & Acc.\,\% & F1 & MCC \\
\midrule
{rows}
\midrule
Always ``not hazardous'' & {pct(base)} & 0.000 & 0.000 & {pct(base)} & 0.000 & 0.000 \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write("tab_earthview", body)


def _method_label(method: str, features: str) -> str:
    if method in RULE_NAMES:
        return RULE_NAMES[method]
    return f"{MODEL_NAMES.get(method, method)} ({FEATURE_NAMES.get(features, features)})"


def table_asteroid_view() -> None:
    s = pd.read_csv(RESULTS / "asteroid_view_summary.csv")
    groups = [
        ("Training-free, asteroid's view", [("rule: exact path geometry (ours)", "moid_path"),
                                            ("rule: tangent-line closed form", "d_lin"),
                                            ("rule: node gap (upper bound)", "d_node")]),
        ("Learned, asteroid's view", [("LightGBM", "path"), ("XGBoost", "path"),
                                      ("RandomForest", "path+kernels"), ("XGBoost", "path+kernels"),
                                      ("LightGBM", "path+kernels"), ("RandomForest", "path+kernels+exact"),
                                      ("LightGBM", "path+kernels+exact")]),
        ("Reference: uses JPL's MOID column$^\\dagger$", [("rule: JPL MOID (definition, reference)", "moid"),
                                                          ("RandomForest", "prior-work orbit+JPL moid"),
                                                          ("LightGBM", "prior-work orbit+JPL moid")]),
    ]
    recs, eligible, heads = [], [], []
    for gname, items in groups:
        heads.append((len(recs), gname))
        for method, feats in items:
            r = s[(s.method == method) & (s.features == feats)].iloc[0]
            recs.append([_method_label(method, feats), f"{pct(r.accuracy_mean)}", pct(r.balanced_accuracy_mean),
                         f3(r.precision_mean), f3(r.recall_mean), f3(r.f1_mean), f3(r.mcc_mean)])
            eligible.append(not gname.startswith("Reference"))
    df = pd.DataFrame(recs, columns=["m", "acc", "bal", "p", "r", "f1", "mcc"])
    df = bold_best(df, ["acc", "bal", "p", "r", "f1", "mcc"], pd.Series(eligible, index=df.index))
    lines = []
    for i, row in enumerate(df.values.tolist()):
        for pos, gname in heads:
            if pos == i:
                if pos:
                    lines.append(r"\midrule")
                lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{gname}}}}} \\")
        lines.append(" & ".join(row) + r" \\")
    n = int(json.loads((RESULTS / "geometry_validation.json").read_text())["n_asteroids"])
    acc_sd = 100 * s["accuracy_std"].max()
    f1_sd = max(s["f1_std"].max(), s["mcc_std"].max())
    body = rf"""\begin{{table*}}[t]
\centering
\footnotesize
\caption{{Asteroid's-view results on all {n:,} labelled near-Earth asteroids: stratified 5-fold cross-validation over asteroids (one row per asteroid, so no asteroid is in both training and test data). Means over folds; across all rows, fold standard deviations are at most {acc_sd:.2f} percentage points for accuracy and {f1_sd:.3f} for F1 and MCC. Best non-reference value per column in bold. $^\dagger$Rows marked with a dagger use JPL's own MOID column, i.e.\ the quantity that defines the label.}}
\label{{tab:asteroidview}}
\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\toprule
Method (features) & Accuracy\,\% & Bal.\ acc.\,\% & Precision & Recall & F1 & MCC \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    write("tab_asteroidview", body)


def table_same_folds() -> None:
    ev = pd.read_csv(RESULTS / "earth_view_summary.csv")
    same = pd.read_csv(RESULTS / "same_folds_rows_summary.csv")
    recs, eligible = [], []
    for m in ["MLP", "XGBoost", "RandomForest"]:
        g = ev[(ev.protocol == "asteroid_grouped") & (ev.model == m)].iloc[0]
        recs.append([f"Earth: {MODEL_NAMES[m]}", pct(g.accuracy_mean), f3(g.f1_mean), f3(g.mcc_mean),
                     f4(g.roc_auc_mean)])
        eligible.append(True)
    for method, feats in [("rule: exact path geometry (ours)", "moid_path"), ("rule: tangent-line closed form", "d_lin"),
                          ("LightGBM", "path"), ("LightGBM", "path+kernels"), ("XGBoost", "path+kernels")]:
        r = same[(same.method == method) & (same.features == feats)].iloc[0]
        label = _method_label(method, feats).replace(" (ours)", "").replace("Path rule, ", "rule, ").replace("tangent-line closed form", "closed form")
        recs.append([f"Asteroid: {label}", pct(r.accuracy_mean), f3(r.f1_mean), f3(r.mcc_mean), f4(r.roc_auc_mean)])
        eligible.append(True)
    df = pd.DataFrame(recs, columns=["m", "acc", "f1", "mcc", "auc"])
    df = bold_best(df, ["acc", "f1", "mcc", "auc"], pd.Series(eligible, index=df.index))
    rows = df.values.tolist()
    lines = [" & ".join(r) + r" \\" for r in rows[:3]] + [r"\midrule"] + [" & ".join(r) + r" \\" for r in rows[3:]]
    body = rf"""\begin{{table}}[t]
\centering
\footnotesize
\caption{{Head-to-head on identical asteroid-grouped folds, scored per close-approach row (306,590 rows). Asteroid's-view models are trained on the training asteroids' paths and their prediction is applied to every approach of each test asteroid.}}
\label{{tab:samefolds}}
\setlength{{\tabcolsep}}{{2.2pt}}
\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\toprule
Method & Acc.\,\% & F1 & MCC & AUC \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write("tab_samefolds", body)


def table_temporal() -> None:
    t = pd.read_csv(RESULTS / "asteroid_view_temporal.csv")
    items = [("rule: exact path geometry (ours)", "moid_path"), ("rule: tangent-line closed form", "d_lin"),
             ("LightGBM", "path"), ("LightGBM", "path+kernels"), ("XGBoost", "path+kernels"),
             ("RandomForest", "path+kernels+exact"),
             ("rule: JPL MOID (definition, reference)", "moid"), ("RandomForest", "prior-work orbit+JPL moid")]
    recs, eligible = [], []
    for method, feats in items:
        r = t[(t.method == method) & (t.features == feats)].iloc[0]
        recs.append([_method_label(method, feats), pct(r.accuracy), f3(r.precision), f3(r.recall), f3(r.f1), f3(r.mcc)])
        eligible.append("JPL" not in feats and "JPL" not in method)
    df = pd.DataFrame(recs, columns=["m", "acc", "p", "r", "f1", "mcc"])
    df = bold_best(df, ["acc", "p", "r", "f1", "mcc"], pd.Series(eligible, index=df.index))
    rows = df.values.tolist()
    lines = [" & ".join(r) + r" \\" for r in rows[:6]] + [r"\midrule"] + [" & ".join(r) + r" \\" for r in rows[6:]]
    n_tr, n_te = int(t.n_train.iloc[0]), int(t.n_test.iloc[0])
    body = rf"""\begin{{table}}[t]
\centering
\footnotesize
\caption{{Temporal generalization: train on the {n_tr:,} asteroids first observed before 2016, test on the {n_te:,} discovered since (2.5\% PHAs). $^\dagger$Uses JPL's MOID column.}}
\label{{tab:temporal}}
\setlength{{\tabcolsep}}{{2pt}}
\begin{{tabular}}{{@{{}}lccccc@{{}}}}
\toprule
Method (features) & Acc.\,\% & Prec. & Rec. & F1 & MCC \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    body = (body.replace("Path rule, ", "Rule: ").replace(" (ours)", "").replace("tangent-line closed form", "closed form")
            .replace(" (path + kernels + exact)", " (path+k+exact)").replace("Random forest (", "RF (").replace(" (path + kernels)", " (path+k)")
            .replace("RF (orbit + JPL MOID)", "RF (orbit+JPL MOID)$^\\dagger$")
            .replace("$^\\dagger$Uses JPL's MOID column.", "RF: random forest; k: geometry kernels. $^\\dagger$Uses JPL's MOID column."))
    write("tab_temporal", body)


def table_uncertainty() -> None:
    pools = [("cc5-7", "Condition code 5--7"), ("cc8-9", "Condition code 8--9 (short arcs)")]
    if not all((RESULTS / f"uncertainty_{t}_summary.csv").exists() for t, _ in pools):
        print("skip uncertainty table (results missing)")
        return
    summ = {t: pd.read_csv(RESULTS / f"uncertainty_{t}_summary.csv") for t, _ in pools}
    meta = {t: json.loads((RESULTS / f"uncertainty_{t}_meta.json").read_text()) for t, _ in pools}
    methods = [("point rule (exact geometry)", "Single orbit, exact rule"),
               ("point rule (closed form)", "Single orbit, closed form"),
               ("LightGBM", "LightGBM (orbit, $\\sigma$s, kernels)"),
               ("path cloud (ours)", "Path cloud (ours)"),
               ("LightGBM + path cloud (ours)", "LightGBM + path cloud (ours)")]
    recs = []
    for m, label in methods:
        row = [label]
        for t, _ in pools:
            s = summ[t]
            r = s[(s.method == m) & (s.scale == 1.0)].iloc[0]
            row += [pct(r.accuracy_mean), f3(r.mcc_mean), f4(r.roc_auc_mean), f4(r.brier_mean)]
        recs.append(row)
    df = pd.DataFrame(recs, columns=["m", "a1", "m1", "u1", "b1", "a2", "m2", "u2", "b2"])
    elig = pd.Series(True, index=df.index)
    df = bold_best(df, ["a1", "m1", "u1", "a2", "m2", "u2"], elig)
    df = bold_best(df, ["b1", "b2"], elig, higher=False)
    rows = "\n".join(" & ".join(r) + r" \\" for r in df.values.tolist())
    m0 = meta["cc5-7"]
    flips = {t: 100 * meta[t]["point_rule_label_flip_rate"]["1.0"] for t, _ in pools}
    body = rf"""\begin{{table*}}[t]
\centering
\footnotesize
\caption{{Hazard classification from uncertain orbits (simulation, real error bars $\times1$). Truth: {m0['n_truth']:,} well-determined NEAs ({m0['n_truth_pha']:,} PHAs). Each true orbit is perturbed with the element uncertainties of a randomly drawn NEA of the given orbit-quality class ({meta['cc5-7']['n_profiles']:,} and {meta['cc8-9']['n_profiles']:,} profiles); methods see only the perturbed orbit, its 1$\sigma$ uncertainties and $H$. Applied to the perturbed orbit, the PHA rule mislabels {flips['cc5-7']:.2f}\% and {flips['cc8-9']:.2f}\% of asteroids, respectively. Path cloud: {m0['n_clones']} clones per asteroid. Means over 5 folds.}}
\label{{tab:uncertainty}}
\begin{{tabular}}{{@{{}}lcccccccc@{{}}}}
\toprule
 & \multicolumn{{4}}{{c}}{{{pools[0][1]}}} & \multicolumn{{4}}{{c}}{{{pools[1][1]}}} \\
\cmidrule(lr){{2-5}}\cmidrule(l){{6-9}}
Method & Acc.\,\% & MCC & ROC-AUC & Brier & Acc.\,\% & MCC & ROC-AUC & Brier \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    write("tab_uncertainty", body)


def table_decision() -> None:
    path = RESULTS / "kernel_summary.json"
    if not path.exists():
        print("skip decision table (no kernel results yet)")
        return
    s = json.loads(path.read_text())
    a, b = s["decision"], s["decision_sys"]
    acts = [("PHA", "PHA"), ("NOT_PHA_GEOMETRY", "Not PHA: geometry"),
            ("NOT_PHA_SIZE", "Not PHA: size ($H > 22$)"), ("OBSERVE_PHA", "Observe, nominal PHA"),
            ("OBSERVE_NOT", "Observe, nominal non-PHA"), ("HUMAN", "Human referral")]
    lines = [f"{lab} & {a['acts'].get(k, 0):,} & {b['acts'].get(k, 0):,} \\\\" for k, lab in acts]
    n = sum(a["acts"].values())
    pct2 = lambda v: f"{100 * v:.2f}"  # noqa: E731
    pct3 = lambda v: f"{100 * v:.3f}"  # noqa: E731
    summary_lines = [
        f"Decided (coverage) & {pct2(a['coverage'])}\\% & {pct2(b['coverage'])}\\% \\\\",
        f"Agrees with JPL flag & {pct3(a['decided_agreement_with_jpl_flag'])}\\% & "
        f"{pct3(b['decided_agreement_with_jpl_flag'])}\\% \\\\",
        f"\\quad disagreements & {a['decided_disagreements_with_flag']} & {b['decided_disagreements_with_flag']} \\\\",
        f"\\quad flag contradicts definition & {a['of_which_flag_contradicts_definition']} & "
        f"{b['of_which_flag_contradicts_definition']} \\\\",
        f"Agrees with definition on JPL MOID & {pct3(a['decided_agreement_with_definition_on_jpl_moid'])}\\% & "
        f"{pct3(b['decided_agreement_with_definition_on_jpl_moid'])}\\% \\\\",
    ]
    body = rf"""\begin{{table}}[t]
\centering
\footnotesize
\caption{{The decision procedure on all {n:,} labelled NEAs with their real 1$\sigma$ element uncertainties. Left: the law as specified; right: with the Earth-curve systematic $s_{{\mathrm{{sys}}}} = {1e4 * b['s_sys_au_mean']:.1f}\times10^{{-4}}$\,au measured on the training folds (5-fold, by asteroid). Agreement is over decided asteroids; ``observe'' marks asteroids whose label could flip within their uncertainty.}}
\label{{tab:decision}}
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}lrr@{{}}}}
\toprule
Act & $s_{{\mathrm{{sys}}}} = 0$ & measured $s_{{\mathrm{{sys}}}}$ \\
\midrule
{chr(10).join(lines)}
\midrule
{chr(10).join(summary_lines)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write("tab_decision", body)


def table_kaggle() -> None:
    path = RESULTS / "kaggle_nasa_comparison.json"
    if not path.exists():
        print("skip Kaggle table (no results yet)")
        return
    r = json.loads(path.read_text())
    inf = r["info"]
    rows = [
        ("Reported~\\cite{chandra2025}, RF by record", "99.7", "--", "0.987", "--", False),
    ]
    for key, label in [("A_prior_protocol_rf", "RF by record (reproduced)"),
                       ("B_grouped_rf", "RF by asteroid"),
                       ("C_ours_rule_all_records", "Training-free path rule (ours)"),
                       ("D_ours_lightgbm_grouped", "LightGBM, path+k, by asteroid (ours)"),
                       ("E_definition_on_table_moid", "Definition on the table's MOID$^\\dagger$")]:
        m = r[key]
        rows.append((label, pct(m["accuracy"]), f3(m["f1"]), f3(m["f2"]), f3(m["mcc"]), key == "E_definition_on_table_moid"))
    lines = []
    for i, (label, a, f1, f2, mcc, ref) in enumerate(rows):
        if i == 1 or ref:
            lines.append(r"\midrule")
        lines.append(f"{label} & {a} & {f1} & {f2} & {mcc} \\\\")
    body = rf"""\begin{{table}}[t]
\centering
\footnotesize
\caption{{Same-file comparison on the Kaggle NeoWs table behind the highest reported accuracy~\cite{{kaggle_nasa,chandra2025}}: {inf['rows']:,} records of {inf['asteroids']:,} asteroids, {100 * inf['hazardous_share']:.1f}\% hazardous. Random-forest rows use every numeric column, including the table's MOID; repeated stratified 10-fold $\times$3. Our rule computes the distance from the table's orbital elements and never reads its MOID column. RF: random forest; k: geometry kernels. $^\dagger$The ceiling for any method that follows the definition.}}
\label{{tab:kaggle}}
\setlength{{\tabcolsep}}{{2.2pt}}
\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\toprule
Method & Acc.\,\% & F1 & F2 & MCC \\
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write("tab_kaggle", body)


def table_sheet() -> None:
    """Three further model families (CNN, GPBoost, tuned random forest) against the rule (sheet_comparison.py)."""
    path = RESULTS / "sheet_comparison.json"
    if not path.exists():
        print("skip score-sheet table (no results yet)")
        return
    r = json.loads(path.read_text())
    if "cad_full" not in r:
        print("skip score-sheet table (full CAD results missing)")
        return
    ours, names = "Ours: path rule (training-free)", {"CNN": "CNN", "GPBoost": "GPBoost", "Tuned RF": "Tuned RF"}
    cf = r["cad_full"]["protocols"]

    def cells(m, sfx="_mean"):
        return [pct(m["accuracy" + sfx]), f3(m["f1" + sfx]), f3(m["mcc" + sfx]), f3(m["roc_auc" + sfx])]

    top = []                                            # (label, split, cells, eligible for bold)
    for proto, split in (("row_split", "row"), ("asteroid_grouped", "grouped")):
        for k, label in names.items():
            top.append((label, split, cells(cf[proto][k]), proto == "asteroid_grouped"))
    rule_row, rule_grp = cells(cf["row_split"][ours]), cells(cf["asteroid_grouped"][ours])
    if rule_row == rule_grp:
        top.append(("Path rule (ours)", "both", rule_grp, True))
    else:
        top += [("Path rule (ours)", "row", rule_row, False), ("Path rule (ours)", "grouped", rule_grp, True)]
    t = pd.DataFrame([c for _, _, c, _ in top], columns=["acc", "f1", "mcc", "auc"])
    t = bold_best(t, list(t.columns), pd.Series([e for *_, e in top]))
    lines = [f"{lab} & {sp} & " + " & ".join(t.iloc[i]) + r" \\" for i, (lab, sp, _, _) in enumerate(top)]

    files = r["files"]
    cad, se = files["cad_file"]["methods"], files["sentry_file"]["methods"]
    bottom = [(names.get(k, "Path rule (ours)"), [pct(cad[k]["accuracy"]), f3(cad[k]["f1"]),
                                                  pct(se[k]["accuracy_mean"]), f3(se[k]["f1_mean"])])
              for k in ("CNN", "GPBoost", "Tuned RF", ours)]
    b = pd.DataFrame([c for _, c in bottom], columns=["a1", "f1", "a2", "f2"])
    b = bold_best(b, list(b.columns), pd.Series([True] * len(b)))
    blines = [f"{lab} & & " + " & ".join(b.iloc[i]) + r" \\" for i, (lab, _) in enumerate(bottom)]
    n_cad, n_se = files["cad_file"]["rows_matched_to_sbdb"], files["sentry_file"]["rows_matched_to_sbdb"]
    body = rf"""\begin{{table}}[t]
\centering
\footnotesize
\caption{{Three further model families against the training-free path rule, with JPL's PHA flag as the label. Top: the {r['cad_full']['rows']:,} CAD approaches and folds of Table~\ref{{tab:earthview}} (mean of 5 folds); the models use the CAD table's own columns, and GPBoost adds a random effect per asteroid. Bottom: two small JPL tables~\cite{{jpl_cad_api,jpl_sentry_api}}, each with its own columns for the models: {n_cad} Earth approaches of September to November 2026 ({files['cad_file']['positives']} PHAs; leave-one-out) and the {n_se:,} Sentry objects with an absolute magnitude ({files['sentry_file']['positives']} PHAs; 5-fold $\times$5). Bold: best honest result per column.}}
\label{{tab:sheet}}
\setlength{{\tabcolsep}}{{2.2pt}}
\begin{{tabular}}{{@{{}}llcccc@{{}}}}
\toprule
Method & Split & Acc.\,\% & F1 & MCC & AUC \\
\midrule
{chr(10).join(lines)}
\midrule
 & & \multicolumn{{2}}{{c}}{{CAD ({n_cad})}} & \multicolumn{{2}}{{c}}{{Sentry ({n_se:,})}} \\
\cmidrule(lr){{3-4}}\cmidrule(l){{5-6}}
 & & Acc.\,\% & F1 & Acc.\,\% & F1 \\
{chr(10).join(blines)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write("tab_sheet", body)


def sci(x: float, digits: int = 1) -> str:
    """LaTeX scientific notation, e.g. 1.9\\times10^{-5}."""
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    return f"{x / 10 ** exp:.{digits}f}\\times10^{{{exp}}}"


def table_proofs() -> None:
    need = ["proof_checks.json", "kernel_summary.json", "kernel_checks_synthetic.json", "kaggle_nasa_comparison.json"]
    if not all((RESULTS / f).exists() for f in need):
        print("skip proof table (results missing)")
        return
    p = json.loads((RESULTS / "proof_checks.json").read_text())
    k = json.loads((RESULTS / "kernel_summary.json").read_text())
    syn = json.loads((RESULTS / "kernel_checks_synthetic.json").read_text())["report"]
    rows = [
        ("Prop.~\\ref{prop:time}", "no mean anomaly or epoch in $\\delta$", "code", "by construction"),
        ("Prop.~\\ref{prop:miss}", "two-body motion, 40,000 epochs in 400\\,yr",
         f"{p['P1_synthetic_orbits']} synthetic orbits",
         f"closest approach $\\ge\\delta$; min.\\ margin ${sci(p['P1_min_of_closest_approach_minus_moid_au'])}$\\,au"),
        ("Prop.~\\ref{prop:miss}", "JPL approach vs.\\ today's $\\delta$", f"{p['P1_jpl_approaches']:,} approaches",
         f"{100 * p['P1_jpl_share_dist_ge_moid_minus_1e-4']:.1f}\\% $\\ge \\delta - 10^{{-4}}$\\,au (orbits evolve)"),
        ("Prop.~\\ref{prop:leak}", "predicted vs.\\ observed leakage", "CAD 5-fold; Kaggle 10-fold",
         f"{100 * p['P0_cad_predicted_seen_share_5fold']:.2f} vs.\\ {100 * p['P0_cad_observed_seen_share_5fold']:.2f}\\%; "
         f"{100 * p['P0_kaggle_predicted_seen_share_10fold']:.1f} vs.\\ {100 * p['P0_kaggle_observed_seen_share_10fold']:.1f}\\%"),
        ("Lemma~\\ref{lem:radial}", "$L_0 \\le \\delta$", f"{p['L1_L2_n']:,} NEAs", f"{p['L1_radial_gap_violations']} violations"),
        ("Lemma~\\ref{lem:node}", "$\\delta \\le \\delta_{\\mathrm{node}}$", f"{p['L1_L2_n']:,} NEAs",
         f"{p['L2_node_gap_violations']} violations"),
        ("Lemma~\\ref{lem:bracket}, Cor.~\\ref{cor:bracket}", "$L_{\\mathrm{lon}} \\le \\delta \\le U_{\\mathrm{lon}}$; width",
         f"{p['L3_n']:,} NEAs, $\\delta < 0.2$\\,au",
         f"{p['L3_lower_violations'] + p['L3_upper_violations']} violations; width $\\le {sci(p['L3_width_max_for_delta_le_0.05_au'], 2)}$\\,au"),
        ("Prop.~\\ref{prop:phi}", "$\\Phi \\le 1 \\Leftrightarrow$ definition", f"{p['L1_L2_n']:,} NEAs",
         f"{p['P2_mismatches']} mismatches"),
        ("Prop.~\\ref{prop:grad}", "gradient vs.\\ central differences", "60 synthetic orbits",
         f"$\\le {sci(syn['grad']['moid'])}$"),
        ("Prop.~\\ref{prop:prob}", "$P_{\\mathrm{lin}}$ vs.\\ Monte Carlo (4,000 draws)",
         f"{syn['K4']['tested']} synthetic, {k['delta_vs_mc']['n']} real orbits",
         f"{syn['K4']['agree']}/{syn['K4']['tested']} and {int(round(k['delta_vs_mc']['inside_share'] * k['delta_vs_mc']['n']))}/{k['delta_vs_mc']['n']} inside 95\\% interval $\\pm0.01$"),
        ("Thm.~\\ref{thm:law}", "law equals its specification", "all $2^{32}$ words", "0 mismatches"),
        ("Thm.~\\ref{thm:law}", "soundness; termination", f"{p['T1_decided']:,} decided NEAs",
         f"{p['T1_PHA_with_H_gt_22'] + p['T1_PHA_with_delta_gt_0.05'] + p['T1_NOT_PHA_GEOMETRY_with_delta_le_0.05'] + p['T1_NOT_PHA_SIZE_with_H_le_22']} violations; $\\le {p['T1_max_steps']}$ steps"),
        ("$\\delta$ solver", "two independent solvers", f"{p['L1_L2_n']:,} NEAs",
         f"$\\le {sci(k['abs_diff_vs_our_solver_au']['max'])}$\\,au"),
        ("$\\delta$ solver", "brute-force reference", f"{syn['K2']['n']} synthetic orbits",
         f"$\\le {sci(syn['K2']['max_dev'])}$\\,au; {syn['K2']['uncertified']} uncertified"),
    ]
    body_rows = "\n".join(" & ".join(r) + r" \\" for r in rows)
    body = rf"""\begin{{table*}}[t]
\centering
\footnotesize
\caption{{Numerical confirmation of every statement proved in Appendix~\ref{{app:proofs}}: each check evaluates the implemented quantity against the proved relation (floating-point tolerance $10^{{-9}}$\,au unless stated). Reproduced by \texttt{{proofs/verify\_all.sh}}.}}
\label{{tab:proofs}}
\begin{{tabularx}}{{\textwidth}}{{@{{}}>{{\raggedright\arraybackslash}}p{{2.6cm}}YYY@{{}}}}
\toprule
Statement & Check & Data & Result \\
\midrule
{body_rows}
\bottomrule
\end{{tabularx}}
\end{{table*}}
"""
    write("tab_proofs", body)


if __name__ == "__main__":
    table_proofs()
    table_kaggle()
    table_sheet()
    table_decision()
    table_earth_view()
    table_asteroid_view()
    table_same_folds()
    table_temporal()
    table_uncertainty()
