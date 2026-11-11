"""Audit: every key number written in the paper's text, recomputed from results/ and found in the LaTeX source.

Each claim is (what, the exact LaTeX fragment in the paper, the number recomputed from the result files). A claim
passes when the recomputed number appears in the fragment and the fragment appears in main.tex or appendix.tex.
Numbers inside tables are generated from results/ by src/make_tables.py and are not repeated here.
Wall-clock timings (TIMING_CLAIMS) depend on the machine and its load, so for them only the order of magnitude is
checked: the recomputed value must lie within a factor of 10 of the paper's.
Exit status 1 on any failure.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
TEX = (ROOT / "paper" / "main.tex").read_text() + (ROOT / "paper" / "appendix.tex").read_text()


def j(name):
    return json.loads((R / name).read_text())


def pct(x, d=2):
    return f"{100 * x:.{d}f}"


ev, st = pd.read_csv(R / "earth_view_summary.csv"), j("earth_view_stats.json")
av, sf = pd.read_csv(R / "asteroid_view_summary.csv"), pd.read_csv(R / "same_folds_rows_summary.csv")
tp, geo = pd.read_csv(R / "asteroid_view_temporal.csv"), j("geometry_validation.json")
ks, kg, pc = j("kernel_summary.json"), j("kaggle_nasa_comparison.json"), j("proof_checks.json")
u57, u89 = pd.read_csv(R / "uncertainty_cc5-7_summary.csv"), pd.read_csv(R / "uncertainty_cc8-9_summary.csv")
m57, m89 = j("uncertainty_cc5-7_meta.json"), j("uncertainty_cc8-9_meta.json")


def ev_row(proto, model):
    return ev[(ev.protocol == proto) & (ev.model == model)].iloc[0]


def av_row(method, features=None):
    d = av[av.method == method]
    return (d if features is None else d[d.features == features]).iloc[0]


def sf_row(method, features=None):
    d = sf[sf.method == method]
    return (d if features is None else d[d.features == features]).iloc[0]


def tp_row(method, features=None):
    d = tp[tp.method == method]
    return (d if features is None else d[d.features == features]).iloc[0]


def unc(df, method, scale):
    return df[(df.method == method) & (df.scale == scale)].iloc[0]


RULE, CEIL = "rule: exact path geometry (ours)", "rule: JPL MOID (definition, reference)"
grp = ev[ev.protocol == "asteroid_grouped"]
rule, ceil, rf_row, rf_grp = av_row(RULE), av_row(CEIL), ev_row("row_split", "RandomForest"), ev_row("asteroid_grouped", "RandomForest")
mlp_grp = ev_row("asteroid_grouped", "MLP")
pt89 = u89[u89.method == "point rule (exact geometry)"].set_index("scale").brier_mean
cl89 = u89[u89.method == "path cloud (ours)"].set_index("scale").brier_mean
red = (1 - cl89 / pt89) * 100
dec, decs = ks["decision"], ks["decision_sys"]
n_unres_cc = 173  # condition code 8-9 among the 188 unresolved (recomputed below from pipeline rows)
pr = pd.read_csv(R / "pipeline_rows.csv")
sb = pd.read_csv(ROOT / "data" / "processed" / "sbdb_neo.csv", usecols=["pdes", "condition_code"], low_memory=False)
unres = pr[pr.act.str.startswith("OBSERVE")].merge(sb, on="pdes")
n_unres_cc = int(unres.condition_code.between(8, 9).sum())

CLAIMS = [
    ("leak share", r"places 95.7\% of test rows", pct(st["test_rows_with_asteroid_in_train"]["row_split"], 1)),
    ("best grouped accuracy", r"at most 92.6\% accuracy", pct(grp.accuracy_mean.max(), 1)),
    ("best grouped F1", r"PHA-class F1 of 0.43", f"{grp.f1_mean.max():.2f}"),
    ("majority baseline", r"(91.3\%)", pct(1 - st["positive_rate_rows"], 1)),
    ("rule accuracy", r"reaches 99.86\% accuracy and F1 0.989 on all 42,888", pct(rule.accuracy_mean)),
    ("rule F1", r"reaches 99.86\% accuracy and F1 0.989 on all 42,888", f"{rule.f1_mean:.3f}"),
    ("same-fold F1", r"raises F1 from 0.43 to 0.99 on identical grouped folds", f"{sf_row(RULE).f1_mean:.2f}"),
    ("decided share", r"decides 99.43\% of them", pct(decs["coverage"])),
    ("flagged share", r"flags the remaining 0.57\%", pct(1 - decs["coverage"])),
    ("Brier reduction low", r"30--36\%", f"{round(red.min())}"),
    ("Brier reduction high", r"30--36\%", f"{round(red.max())}"),
    ("Kaggle grouped = rule", r"the same model scores 99.42\%", pct(kg["B_grouped_rf"]["accuracy"])),
    ("Kaggle rule", r"(99.42\%, F1 0.982, MCC 0.979)", pct(kg["C_ours_rule_all_records"]["accuracy"])),
    ("approach rows", r"306,590 belong to a labelled asteroid", f"{st['rows']:,}"),
    ("asteroids in CAD", r"they cover 36,817 asteroids", f"{st['asteroids']:,}"),
    ("positive rows", r"8.74\% of rows are PHA approaches", pct(st["positive_rate_rows"])),
    ("NEAs", r"42,888 have both an MOID and a flag", f"{geo['n_asteroids']:,}"),
    ("PHAs", r"(2,551 PHAs, 5.9\%)", f"{geo['n_pha']:,}"),
    ("flag contradictions", r"all but 49 asteroids (0.11\%)", f"{geo['jpl_flag_vs_rule_on_jpl_moid_disagreements']}"),
    ("ceiling", r"at 99.886\%", pct(ceil.accuracy_mean, 3)),
    ("median vs JPL", r"the median absolute difference is $1.0\times10^{-4}\au$", f"{geo['abs_diff_au']['median'] * 1e4:.1f}"),
    ("p99 vs JPL", r"the 99th percentile $6.0\times10^{-4}\au$", f"{geo['abs_diff_au']['p99'] * 1e4:.1f}"),
    ("correlation", r"$r = 0.999997$", f"{geo['pearson_r']:.6f}"),
    ("label changes", r"changes the PHA label of 9 of 42,888", f"{round((1 - geo['rule_label_agreement_ours_vs_jplmoid']) * geo['n_asteroids'])}"),
    ("two solvers", r"within $6.8\times10^{-10}\au$ on every asteroid", f"{ks['abs_diff_vs_our_solver_au']['max'] * 1e10:.1f}"),
    ("ms per asteroid", r"at a median 0.66\,ms per asteroid", f"{ks['ms_per_object_median']:.2f}"),
    ("bounds settle", r"settle 51.1\% of the 11,761 bright NEAs", pct(ks["screening"]["far_share"] + ks["screening"]["near_share"], 1)),
    ("far share", r"39.6\% have $L_0 > 0.05\au$", pct(ks["screening"]["far_share"], 1)),
    ("near share", r"11.5\% have $\delta_{\mathrm{node}} \le 0.05\au$", pct(ks["screening"]["near_share"], 1)),
    ("beta", r"$\hat\beta = (-1.1 \pm 2.9)\times10^{-4}\au$", f"{ks['k5']['beta_au'] * 1e4:.1f}"),
    ("beta sd", r"$\hat\beta = (-1.1 \pm 2.9)\times10^{-4}\au$", f"{ks['k5']['beta_std'] * 1e4:.1f}"),
    ("K5 vs rule", r"(99.860\% against 99.865\%)", pct(ks["k5"]["fit_accuracy"], 3)),
    ("unresolved", r"flags 188 (0.44\%) for follow-up", f"{dec['unresolved']}"),
    ("unresolved short-arc", r"173 of them with short-arc orbits", f"{n_unres_cc}"),
    ("disagreements", r"Its 56 disagreements with JPL's flag", f"{dec['decided_disagreements_with_flag']}"),
    ("flag contradictions decided", r"48 flags that contradict JPL's own definition", f"{dec['of_which_flag_contradicts_definition']}"),
    ("s_sys", r"$s_{\mathrm{sys}} = 2.4\times10^{-4}\au$ moves", f"{decs['s_sys_au_mean'] * 1e4:.1f}"),
    ("delta vs MC n", r"On the 32 real orbits", f"{ks['delta_vs_mc']['n']}"),
    ("row-split RF", r"reaches 94.74\% accuracy and F1 0.676", pct(rf_row.accuracy_mean)),
    ("grouped RF", r"falls to 90.76\% accuracy and F1 0.428", pct(rf_grp.accuracy_mean)),
    ("grouped MLP", r"The best grouped accuracy is 92.63\% (MLP)", pct(mlp_grp.accuracy_mean)),
    ("closed form", r"rule reaches 99.26\% (F1 0.938)", pct(av_row("rule: tangent-line closed form").accuracy_mean)),
    ("raw path", r"raw path reach 98.38\% (F1 0.862)", pct(av_row("LightGBM", "path").accuracy_mean)),
    ("kernels", r"lift LightGBM to 99.71\% (F1 0.976)", pct(av_row("LightGBM", "path+kernels").accuracy_mean)),
    ("reference RF", r"training-free rule (99.87\% against 99.86\%, F1 0.989 for both)", pct(av_row("RandomForest", "prior-work orbit+JPL moid").accuracy_mean)),
    ("same folds", r"from 92.63\% to 99.79\%, F1 from 0.428 to 0.988 and MCC from 0.410 to 0.987", pct(sf_row(RULE).accuracy_mean)),
    ("temporal train", r"the 15,071 asteroids first observed before 2016 (1,869 PHAs)", f"{int(tp.n_train.iloc[0]):,}"),
    ("temporal test", r"the 27,817 discovered since (682 PHAs)", f"{int(tp.n_test.iloc[0]):,}"),
    ("temporal rule F1", r"the path rule keeps F1 0.988", f"{tp_row(RULE).f1:.3f}"),
    ("temporal LightGBM", r"LightGBM on path and kernels reaches F1 0.972", f"{tp_row('LightGBM', 'path+kernels').f1:.3f}"),
    ("temporal raw", r"on the raw path alone it drops to 0.835", f"{tp_row('LightGBM', 'path').f1:.3f}"),
    ("cc5-7 flips", r"mislabels only 0.11\% of asteroids", pct(m57["point_rule_label_flip_rate"]["1.0"])),
    ("cc5-7 point MCC", r"(MCC 0.996)", f"{unc(u57, 'point rule (exact geometry)', 1.0).mcc_mean:.3f}"),
    ("cc5-7 cloud MCC", r"(MCC 0.994)", f"{unc(u57, 'path cloud (ours)', 1.0).mcc_mean:.3f}"),
    ("cc8-9 flips", r"mislabels 1.54\% of asteroids", pct(m89["point_rule_label_flip_rate"]["1.0"])),
    ("cc8-9 accuracy", r"(98.46\% accuracy and MCC 0.949 for both)", pct(unc(u89, "path cloud (ours)", 1.0).accuracy_mean)),
    ("cc8-9 Brier", r"by 32\% (0.0105 instead of 0.0154)", f"{unc(u89, 'path cloud (ours)', 1.0).brier_mean:.4f}"),
    ("Kaggle record split", r"reaches 99.62\%, close to the reported 99.7\%", pct(kg["A_prior_protocol_rf"]["accuracy"])),
    ("Kaggle ceiling", r"above the 99.55\% that the definition itself scores", pct(kg["E_definition_on_table_moid"]["accuracy"])),
    ("memorised seen", r"in 23 of 31 cases", f"{kg['F_memorisation_record_split']['matched_seen']}"),
    ("memorised unseen", r"in 4 of 32", f"{kg['F_memorisation_record_split']['matched_unseen']}"),
    ("rule errors", r"Of the rule's 27 errors, 21", f"{kg['C_errors']['records']}"),
    ("Kaggle records", r"4,687 NeoWs records of 3,692 asteroids", f"{kg['info']['rows']:,}"),
    ("Kaggle repeated asteroids", r"751 appear in more than one record", f"{kg['info']['asteroids_with_more_than_one_record']}"),
    ("leak formula CAD", r"the formula gives 95.69\%", pct(pc["P0_cad_predicted_seen_share_5fold"])),
    ("leak formula Kaggle", r"it gives 34.8\%", pct(pc["P0_kaggle_predicted_seen_share_10fold"], 1)),
    ("approaches below MOID", r"this happens for 1.2\% of approaches", pct(1 - pc["P1_jpl_share_dist_ge_moid_minus_1e-4"], 1)),
    ("bracket width", r"the widest observed on the catalogue is $1.95\times10^{-5}\au$", f"{pc['L3_width_max_for_delta_le_0.05_au'] * 1e5:.2f}"),
]


TIMING_CLAIMS = {"ms per asteroid"}


def main() -> None:
    bad, timing = [], []
    for what, frag, val in CLAIMS:
        ok_text = frag in TEX
        if what in TIMING_CLAIMS:
            paper = float(re.search(r"[0-9]+(?:\.[0-9]+)?", frag).group())
            ok_val = paper / 10 <= float(val) <= paper * 10
            timing.append(f"  timing, {what}: paper {paper:g}, this run {val} (machine-dependent; within 10x: {ok_val})")
        else:
            ok_val = val in frag
        if not (ok_text and ok_val):
            bad.append((what, frag, val, ok_text, ok_val))
    print(f"{len(CLAIMS) - len(bad)} of {len(CLAIMS)} paper numbers recomputed from results/ and found in the LaTeX source")
    print("\n".join(timing))
    for what, frag, val, ok_text, ok_val in bad:
        print(f"  FAIL {what}: recomputed {val!r}; fragment in paper: {ok_text}; value in fragment: {ok_val}\n       {frag}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
