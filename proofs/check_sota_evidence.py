"""Independent check of src/sota_evidence.py: re-decodes every fold's confusion counts in a different way and
recomputes the McNemar bounds with a separate implementation, then checks the JSON and both LaTeX tables.

Decoding here uses the F1 identity instead of accuracy: with P positives, TP = recall * P and the predicted
positives are TP / precision; n is then the unique test size for which every method's accuracy * n is an integer.
The McNemar tail is summed with exact integer arithmetic (Python big integers), not floating-point log-gamma.
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, TAB = ROOT / "results", ROOT / "paper" / "tables"
RULE = "rule: exact path geometry (ours)"
ok = True


def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg)
    ok &= bool(cond)


ev = pd.read_csv(RES / "earth_view_folds.csv").rename(columns={"model": "method"})
ev = ev[ev.protocol == "asteroid_grouped"]
sh = pd.read_csv(RES / "sheet_comparison_folds.csv")
sh = sh[(sh.part == "C") & (sh.dataset == "NASA CAD (full)") & (sh.protocol == "asteroid_grouped")
        & sh.method.isin(["CNN", "GPBoost", "Tuned RF"])]
sf = pd.read_csv(RES / "same_folds_rows.csv")
sf = sf[(sf.protocol == "earthview_grouped_rows") & (sf.method == RULE)]
frame = pd.concat([sf, ev, sh], ignore_index=True)

counts = {}
for fold, g in frame.groupby("fold"):
    # P: smallest positive count consistent with every method's recall
    rec = [r for r in g.recall if 0 < r < 1]
    P = next(p for p in range(1, 20000) if all(abs(r * p - round(r * p)) < 1e-7 for r in rec))
    accs = g.accuracy.tolist()
    n = next(m for m in range(P, 200000) if all(abs(a * m - round(a * m)) < 1e-7 for a in accs))
    for _, r in g.iterrows():
        tp = round(r.recall * P)
        pp = round(tp / r.precision) if r.precision > 0 else None
        err = n - round(r.accuracy * n)
        fp = err - (P - tp)
        if pp is not None:
            assert pp == tp + fp, (r.method, fold)
        c = counts.setdefault(r.method, dict(n=0, p=0, fn=0, fp=0))
        c["n"] += n; c["p"] += P; c["fn"] += P - tp; c["fp"] += fp


def log10_tail(b, c):
    """log10 P(X <= c), X ~ Bin(b + c, 1/2), from an exact integer sum."""
    n = b + c
    s = sum(math.comb(n, k) for k in range(c + 1))
    return (math.log10(s) if s < 10**300 else len(str(s)) - 1 + math.log10(int(str(s)[:17]) / 10**16)) - n * math.log10(2)


js = json.load(open(RES / "sota_evidence.json"))
R = counts[RULE]
stats = json.load(open(RES / "earth_view_stats.json"))
check(R["n"] == stats["rows"] and R["p"] == stats["positive_rows"], f"fold sizes add up to {R['n']:,} rows, {R['p']:,} PHA rows")
check(js["paired"]["rule"]["fn"] == R["fn"] and js["paired"]["rule"]["fp"] == R["fp"],
      f"rule: {R['fn']} missed, {R['fp']} false alarms, as in sota_evidence.json")
tab = (TAB / "tab_sota.tex").read_text()
for m in js["paired"]["models"]:
    c = counts[m["model"]]
    same = c["fn"] == m["fn"] and c["fp"] == m["fp"]
    le = log10_tail(c["fn"] + c["fp"], R["fn"] + R["fp"])
    lm = log10_tail(c["fn"], R["fn"])
    close = abs(le - m["log10_p_errors"]) < 1e-6 and abs(lm - m["log10_p_missed"]) < 1e-6
    in_tab = f"$<10^{{{math.ceil(le)}}}$" in tab and f"$<10^{{{math.ceil(lm)}}}$" in tab and f"{c['fn']:,}" in tab
    check(same and close and in_tab, f"{m['model']}: errors {c['fn'] + c['fp']:,}, missed {c['fn']:,}, "
          f"log10 p {le:.1f} / {lm:.1f}, in JSON and Table")
summ = (TAB / "tab_sota_summary.tex").read_text()
s = js["summary"]
check(f"{100 * s['cad']['rule_acc']:.2f}" in summ and f"{s['sbdb']['rule_f1']:.3f}" in summ
      and f"{100 * s['kaggle']['ceiling_acc']:.2f}" in summ, "summary table carries the JSON values")
print("all checks passed" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
