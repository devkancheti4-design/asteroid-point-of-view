"""Statistical evidence against the state of the art, from the stored fold results only.

1. Paired test on identical rows. The training-free path rule and the nine Earth's-view models (six in
   earth_view.py, three in sheet_comparison.py) were scored on the same asteroid-grouped folds of the 306,590
   NASA CAD approaches. Each fold's test size n and positive count P are decoded exactly from the stored
   metrics (accuracy * n and recall * P must be integers for every method at once), which gives every
   method's confusion counts. For each model, an exact one-sided McNemar test asks whether the rule makes
   no fewer errors (and misses no fewer PHA approaches). Row-level predictions are not stored, so the test is
   evaluated in the least favourable case, where the two error sets are disjoint; the true p is smaller.
2. Summary table of every benchmark against the best published result.

Writes results/sota_evidence.json, paper/tables/tab_sota.tex and paper/tables/tab_sota_summary.tex.
"""
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAB = ROOT / "paper" / "tables"
RULE = "rule: exact path geometry (ours)"
NAMES = {"LogReg": "Logistic reg.", "KNN": "$k$-NN", "MLP": "MLP", "RandomForest": "Random forest",
         "XGBoost": "XGBoost", "LightGBM": "LightGBM", "CNN": "CNN", "GPBoost": "GPBoost",
         "Tuned RF": "Tuned RF"}


def decode(values, lo, hi, tol=1e-6):
    """The unique integer N in [lo, hi) for which every value * N is an integer."""
    hits = [n for n in range(lo, hi) if all(abs(v * n - round(v * n)) < tol for v in values)]
    assert len(hits) == 1, (hits[:5], len(hits))
    return hits[0]


def fold_counts(frame, n_range, p_range):
    """Confusion counts per method and fold, decoded from accuracy and recall."""
    out = {}
    for fold, g in frame.groupby("fold"):
        n = decode(g["accuracy"].tolist(), *n_range)
        p = decode([r for r in g["recall"] if 0 < r < 1], *p_range)
        for _, r in g.iterrows():
            tp = round(r["recall"] * p)
            err = n - round(r["accuracy"] * n)
            fn = p - tp
            fp = err - fn
            assert fp >= 0
            prec = tp / (tp + fp) if tp + fp else 0.0
            assert abs(prec - r["precision"]) < 1e-9, (r["method"], fold, prec, r["precision"])
            out.setdefault(r["method"], []).append(dict(fold=int(fold), n=n, p=p, tp=tp, fp=fp, fn=fn,
                                                        tn=n - p - fp, accuracy=r["accuracy"],
                                                        f1=r["f1"], mcc=r["mcc"]))
    return out


def log10_mcnemar(b, c):
    """log10 of the exact one-sided McNemar p-value P(X <= c), X ~ Binomial(b + c, 1/2)."""
    n = b + c
    terms = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2)
             for k in range(c + 1)]
    m = max(terms)
    return (m + math.log(sum(math.exp(t - m) for t in terms))) / math.log(10)


def paired():
    ev = pd.read_csv(RES / "earth_view_folds.csv")
    ev = ev[ev.protocol == "asteroid_grouped"].rename(columns={"model": "method"})
    sh = pd.read_csv(RES / "sheet_comparison_folds.csv")
    sh = sh[(sh.part == "C") & (sh.dataset == "NASA CAD (full)") & (sh.protocol == "asteroid_grouped")
            & sh.method.isin(["CNN", "GPBoost", "Tuned RF"])]
    sf = pd.read_csv(RES / "same_folds_rows.csv")
    sf = sf[(sf.protocol == "earthview_grouped_rows") & (sf.method == RULE)]
    cols = ["fold", "method", "accuracy", "precision", "recall", "f1", "mcc"]
    frame = pd.concat([sf[cols], ev[cols], sh[cols]], ignore_index=True)
    counts = fold_counts(frame, (40000, 80000), (2000, 10000))
    stats = json.load(open(RES / "earth_view_stats.json"))
    rule = counts[RULE]
    assert sum(f["n"] for f in rule) == stats["rows"] and sum(f["p"] for f in rule) == stats["positive_rows"]

    def pooled(fs):
        t = {k: sum(f[k] for f in fs) for k in ("n", "p", "tp", "fp", "fn", "tn")}
        t["errors"] = t["fp"] + t["fn"]
        return t

    R = pooled(rule)
    models = []
    for name, fs in counts.items():
        if name == RULE:
            continue
        assert [f["n"] for f in fs] == [f["n"] for f in rule]  # identical test rows in every fold
        M = pooled(fs)
        wins = {k: sum(r[k] > m[k] for r, m in zip(rule, fs)) for k in ("accuracy", "f1", "mcc")}
        models.append(dict(model=name, **M, folds_won=wins,
                           log10_p_errors=log10_mcnemar(M["errors"], R["errors"]),
                           log10_p_missed=log10_mcnemar(M["fn"], R["fn"])))
    models.sort(key=lambda m: m["errors"])
    return dict(rows=R["n"], positive_rows=R["p"], folds=len(rule), rule=R, models=models)


def asteroid_level():
    """The rule on all 42,888 NEAs (5-fold over asteroids), as counts."""
    af = pd.read_csv(RES / "asteroid_view_folds.csv")
    af = af[af.protocol == "asteroid_kfold"]
    counts = fold_counts(af, (5000, 12000), (200, 1000))
    R = {k: sum(f[k] for f in counts[RULE]) for k in ("n", "p", "tp", "fp", "fn", "tn")}
    return R


def pct(x, d=2):
    return f"{100 * x:.{d}f}\\%"


def p_bound(log10p):
    return f"$<10^{{{math.ceil(log10p)}}}$"


def write_tables(ev):
    R = ev["paired"]["rule"]
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\caption{Paired test against the nine Earth's-view models on identical rows: the "
             f"{ev['paired']['rows']:,} NASA CAD approaches ({ev['paired']['positive_rows']:,} to PHAs) of "
             r"the asteroid-grouped folds of Table~\ref{tab:earthview}, counts pooled over the 5 folds. $p$: "
             r"exact one-sided McNemar test of the hypothesis that the rule makes no fewer errors (misses no "
             r"fewer PHA approaches) than the model, evaluated in the least favourable case, where the two "
             r"error sets are disjoint; the true $p$ is smaller. The rule wins all 5 folds on accuracy, F1 and "
             r"MCC against every model.}",
             r"\label{tab:sota}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lrrrll@{}}", r"\toprule",
             r"Method & Errors & Missed & False & $p$ & $p$ \\",
             r" & & PHA & alarms & (errors) & (missed) \\", r"\midrule",
             rf"Path rule (ours) & \textbf{{{R['errors']:,}}} & \textbf{{{R['fn']:,}}} & {R['fp']:,} & -- & -- \\",
             r"\midrule"]
    for m in ev["paired"]["models"]:
        lines.append(f"{NAMES[m['model']]} & {m['errors']:,} & {m['fn']:,} & {m['fp']:,} & "
                     f"{p_bound(m['log10_p_errors'])} & {p_bound(m['log10_p_missed'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TAB / "tab_sota.tex").write_text("\n".join(lines))

    s = ev["summary"]
    rows = [
        (f"NASA CAD, {s['cad']['rows']:,} approaches", f"RF, row split: {pct(s['cad']['prior_acc'])}, F1 {s['cad']['prior_f1']:.3f}",
         f"{pct(s['cad']['ml_acc'])}, F1 {s['cad']['ml_f1']:.3f} (9 models)",
         f"{pct(s['cad']['rule_acc'])}, F1 {s['cad']['rule_f1']:.3f}",
         f"{pct(s['cad']['ceiling_acc'])}, F1 {s['cad']['ceiling_f1']:.3f}"),
        (f"SBDB, {s['sbdb']['n']:,} NEAs", r"graph NN~\cite{jacques2025}: 99\%, F1 0.37",
         f"RF, path features: {pct(s['sbdb']['ml_acc'])}, F1 {s['sbdb']['ml_f1']:.3f}",
         f"{pct(s['sbdb']['rule_acc'])}, F1 {s['sbdb']['rule_f1']:.3f}",
         f"{pct(s['sbdb']['ceiling_acc'])}, F1 {s['sbdb']['ceiling_f1']:.3f}"),
        (f"Kaggle NeoWs table, {s['kaggle']['rows']:,} records",
         rf"RF by record~\cite{{chandra2025}}: 99.7\% ({pct(s['kaggle']['reproduced_acc'])} reproduced)",
         f"RF by asteroid: {pct(s['kaggle']['ml_acc'])}, F1 {s['kaggle']['ml_f1']:.3f}",
         f"{pct(s['kaggle']['rule_acc'])}, F1 {s['kaggle']['rule_f1']:.3f}",
         f"{pct(s['kaggle']['ceiling_acc'])}, F1 {s['kaggle']['ceiling_f1']:.3f}"),
        ("Received sheet: CAD file, 33 approaches", r"CNN, GPBoost, RF: 72--75\%, task unstated",
         f"{pct(s['cad_file']['ml_acc'])}, F1 {s['cad_file']['ml_f1']:.3f}",
         f"{pct(s['cad_file']['rule_acc'])}, F1 {s['cad_file']['rule_f1']:.3f}", "--"),
        (f"Received sheet: Sentry list, {s['sentry']['n']:,} objects", r"CNN, GPBoost, RF: 72--75\%, task unstated",
         f"{pct(s['sentry']['ml_acc'])}, F1 {s['sentry']['ml_f1']:.3f}",
         f"{pct(s['sentry']['rule_acc'])}, F1 {s['sentry']['rule_f1']:.3f}", "--"),
    ]
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\caption{Every benchmark against the best published result. ``Honest'': split by asteroid, so no "
             r"test asteroid is seen in training; best ML values are the best of the models tried, per column. "
             r"Ceiling: the PHA definition applied to JPL's (Kaggle: the table's) own MOID. Published numbers "
             r"use the protocols of Table~\ref{tab:literature}.}",
             r"\label{tab:sotasummary}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{3.1cm}YYY>{\raggedright\arraybackslash}p{2.2cm}@{}}",
             r"\toprule",
             r"Benchmark & Best published (protocol) & Best ML, honest split & Our training-free rule & Ceiling \\",
             r"\midrule"]
    for b, pub, ml, rule, ceil in rows:
        lines.append(f"{b} & {pub} & {ml} & \\textbf{{{rule}}} & {ceil} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""]
    (TAB / "tab_sota_summary.tex").write_text("\n".join(lines))


def summary():
    evs = pd.read_csv(RES / "earth_view_summary.csv")
    rf_row = evs[(evs.protocol == "row_split") & (evs.model == "RandomForest")].iloc[0]
    grp = evs[evs.protocol == "asteroid_grouped"]
    sh = json.load(open(RES / "sheet_comparison.json"))
    gp = sh["cad_full"]["protocols"]["asteroid_grouped"]
    ml3 = [v for k, v in gp.items() if not k.startswith("Ours")]
    sfs = pd.read_csv(RES / "same_folds_rows_summary.csv").set_index("method")
    avs = pd.read_csv(RES / "asteroid_view_summary.csv")
    kf = avs[avs.protocol == "asteroid_kfold"].set_index("method")
    learned = kf[~kf.index.str.startswith("rule") & ~kf.features.str.contains("jpl", case=False)]
    ref = kf[kf.index.str.contains("JPL MOID")].iloc[0]
    kg = json.load(open(RES / "kaggle_nasa_comparison.json"))
    files = sh["files"]
    cad_ml = [v for k, v in files["cad_file"]["methods"].items() if not k.startswith("Ours")]
    se_ml = [v for k, v in files["sentry_file"]["methods"].items() if not k.startswith("Ours")]
    se_rule = files["sentry_file"]["methods"]["Ours: path rule (training-free)"]
    cad_rule = files["cad_file"]["methods"]["Ours: path rule (training-free)"]
    return dict(
        cad=dict(rows=sh["cad_full"]["rows"], prior_acc=rf_row.accuracy_mean, prior_f1=rf_row.f1_mean,
                 ml_acc=max(grp.accuracy_mean.max(), max(v["accuracy_mean"] for v in ml3)),
                 ml_f1=max(grp.f1_mean.max(), max(v["f1_mean"] for v in ml3)),
                 rule_acc=sfs.loc[RULE, "accuracy_mean"], rule_f1=sfs.loc[RULE, "f1_mean"],
                 ceiling_acc=sfs.loc["rule: JPL MOID (definition, reference)", "accuracy_mean"],
                 ceiling_f1=sfs.loc["rule: JPL MOID (definition, reference)", "f1_mean"]),
        sbdb=dict(n=42888, ml_acc=learned.accuracy_mean.max(), ml_f1=learned.f1_mean.max(),
                  rule_acc=kf.loc[RULE, "accuracy_mean"], rule_f1=kf.loc[RULE, "f1_mean"],
                  ceiling_acc=ref.accuracy_mean, ceiling_f1=ref.f1_mean),
        kaggle=dict(rows=kg["info"]["rows"], reproduced_acc=kg["A_prior_protocol_rf"]["accuracy"],
                    ml_acc=kg["B_grouped_rf"]["accuracy"], ml_f1=kg["B_grouped_rf"]["f1"],
                    rule_acc=kg["C_ours_rule_all_records"]["accuracy"], rule_f1=kg["C_ours_rule_all_records"]["f1"],
                    ceiling_acc=kg["E_definition_on_table_moid"]["accuracy"],
                    ceiling_f1=kg["E_definition_on_table_moid"]["f1"]),
        cad_file=dict(ml_acc=max(v["accuracy"] for v in cad_ml), ml_f1=max(v["f1"] for v in cad_ml),
                      rule_acc=cad_rule["accuracy"], rule_f1=cad_rule["f1"]),
        sentry=dict(n=files["sentry_file"]["rows_matched_to_sbdb"],
                    ml_acc=max(v["accuracy_mean"] for v in se_ml), ml_f1=max(v["f1_mean"] for v in se_ml),
                    rule_acc=se_rule["accuracy_mean"], rule_f1=se_rule["f1_mean"]),
    )


def main():
    ev = dict(paired=paired(), asteroid_level=asteroid_level(), summary=summary())
    for m in ev["paired"]["models"]:
        assert m["errors"] > ev["paired"]["rule"]["errors"] and m["fn"] > ev["paired"]["rule"]["fn"]
        assert all(v == ev["paired"]["folds"] for v in m["folds_won"].values()), m
    json.dump(ev, open(RES / "sota_evidence.json", "w"), indent=1)
    write_tables(ev)
    R, A = ev["paired"]["rule"], ev["asteroid_level"]
    print(f"rule on {R['n']:,} identical rows: errors {R['errors']}, missed PHA {R['fn']} of {R['p']}, FP {R['fp']}")
    for m in ev["paired"]["models"]:
        print(f"  {m['model']:13s} errors {m['errors']:6d} missed {m['fn']:6d} FP {m['fp']:6d} "
              f"log10p {m['log10_p_errors']:9.1f} {m['log10_p_missed']:9.1f} wins {m['folds_won']}")
    print(f"rule on all {A['n']:,} NEAs: missed {A['fn']} of {A['p']} PHAs, false alarms {A['fp']} of {A['n'] - A['p']}")
    print(json.dumps(ev["summary"], indent=1))


if __name__ == "__main__":
    main()
