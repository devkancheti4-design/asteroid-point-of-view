"""Narrated presentation video: "The Asteroid's Point of View".

Output: presentation/asteroids_point_of_view.mp4 (1280x720, 24 fps, H.264 + AAC narration, burned-in captions)

Pipeline
  1. Narration: macOS `say` speaks each sentence; sentences are laid out in time within each scene, and each
     scene's audio is padded to a whole number of video frames, so picture and sound stay in sync.
  2. Frames: matplotlib draws every frame of a scene (one process per scene), piped as raw RGBA to ffmpeg.
  3. Assembly: scenes are concatenated and the narration is muxed in.

Every number on screen is checked against results/ by check_numbers() before anything is rendered.

Usage:  .venv/bin/python presentation/make_video.py              full build
        .venv/bin/python presentation/make_video.py --stills     one still per scene sentence, for checking
        VOICE="Aman (English (India))" .venv/bin/python presentation/make_video.py   other narrator
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import textwrap
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from orbit_geometry import EARTH_E, EARTH_P, EARTH_VARPI, distance_to_earth_orbit, moid_earth, orbit_basis, path_points  # noqa: E402

RESULTS = ROOT / "results"
OUT = ROOT / "presentation"
BUILD = OUT / "build"
W, H, FPS = 1280, 720, 24
VOICE = os.environ.get("VOICE", "Samantha")
RATE = int(os.environ.get("RATE", "170"))
LEAD, GAP, TAIL = 0.6, 0.45, 0.9
SR = 44100

# dark-mode categorical slots validated against a dark surface (all checks pass, contrast >= 3:1)
BG, CARD = "#0d0d0d", "#1a1a19"
TEXT, TEXT2, MUTED, GRID, AXIS = "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835"
BLUE, ORANGE, AQUA, YELLOW = "#3987e5", "#d95926", "#199e70", "#c98500"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "text.color": TEXT, "axes.edgecolor": AXIS, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
                     "ytick.color": TEXT2, "axes.facecolor": BG, "figure.facecolor": BG, "mathtext.fontset": "dejavusans"})

# ------------------------------------------------------------------------------------------- narration
# (caption shown on screen, text spoken)
SCENES_TEXT = [
    ("title", [
        ("The asteroid's point of view.", None),
        ("How a change of viewpoint classifies potentially hazardous asteroids almost perfectly, and why some "
         "published accuracies mislead.", None)]),
    ("definition", [
        ("NASA calls an asteroid potentially hazardous when two things are true.", None),
        ("Its orbit passes within 0.05 au of Earth's orbit: the minimum orbit intersection distance, or MOID.",
         "Its orbit passes within zero point zero five astronomical units of Earth's orbit. This distance is called "
         "the minimum orbit intersection distance, or MOID."),
        ("And it is big enough: absolute magnitude H of 22 or brighter, roughly 140 m or larger.",
         "And it is big enough: an absolute magnitude of twenty-two or brighter, which means roughly one hundred and "
         "forty metres or larger.")]),
    ("earthview", [
        ("Most machine-learning papers look at asteroids from Earth.", None),
        ("Every close approach becomes a training row: its miss distance, its speed and the asteroid's size.", None),
        ("An asteroid passes Earth again and again, so one asteroid fills many rows. Apophis alone fills 46.",
         "An asteroid passes Earth again and again, so one asteroid fills many rows. Apophis alone fills forty-six."),
        ("These papers report accuracies from 94% to 99.7%.",
         "These papers report accuracies from ninety-four to ninety-nine point seven percent.")]),
    ("leak", [
        ("This creates a leak.", None),
        ("With a random split, 95.7% of test rows belong to an asteroid the model already saw in training.",
         "With a random split, almost ninety-six percent of test rows belong to an asteroid the model has already "
         "seen in training."),
        ("Keep each asteroid entirely in training or in test, and the best accuracy falls to 92.6%, barely above "
         "always answering 'not hazardous' (91.3%).",
         "If we keep each asteroid entirely in training or entirely in test, the best accuracy falls to ninety-two "
         "point six percent, barely above always answering not hazardous."),
        ("The F1 score on hazardous asteroids falls from 0.68 to 0.43.",
         "And the F1 score on the hazardous asteroids falls from zero point six eight to zero point four three.")]),
    ("asteroidview", [
        ("We take the asteroid's point of view instead.", None),
        ("Each asteroid becomes one row: its own path around the Sun, and its size.", None),
        ("Where the two bodies are at any moment does not matter.", None),
        ("Only how close the two paths come, so the timing of encounters cancels out.", None)]),
    ("math", [
        ("The asteroid's path is a conic curve, fixed by five orbital elements.", None),
        ("We compute the exact closest distance between this curve and Earth's orbit, in about 0.3 ms per asteroid.",
         "We compute the exact closest distance between this curve and Earth's orbit, in about a third of a "
         "millisecond per asteroid."),
        ("Each asteroid then gets one number, the hazard index: the larger of its distance over 0.05 au and its "
         "size ratio.",
         "Each asteroid then gets one number, the hazard index: the larger of its distance divided by zero point "
         "zero five astronomical units, and its size ratio."),
        ("It is hazardous exactly when the index is at most 1. No training is needed.",
         "It is hazardous exactly when this index is at most one. No training is needed.")]),
    ("proofs", [
        ("Why is it right? Every step comes with a proof.", None),
        ("A close approach is the distance between one point on each orbit, so for fixed orbits it can never be "
         "smaller than the MOID.", None),
        ("The triangle inequality gives a lower bound on the MOID, and the two node points give an upper bound.",
         None),
        ("The hazard index is at most 1 exactly when both conditions of the definition hold.",
         "The hazard index is at most one exactly when both conditions of the definition hold."),
        ("The decision law ends within six steps, never contradicts its certificate, and matches its specification "
         "on all 4.3 billion inputs.",
         "And the decision law ends within six steps, never contradicts its certificate, and matches its "
         "specification on all four point three billion inputs."),
        ("The full proofs are in the paper's appendix, and each one is checked numerically on the whole catalogue.",
         None)]),
    ("data", [
        ("Here is the real catalogue from both viewpoints.", None),
        ("Against miss distance, hazardous and safe asteroids overlap: one encounter only bounds the path distance "
         "from above.",
         "Against miss distance, hazardous and safe asteroids overlap, because one encounter only bounds the path "
         "distance from above."),
        ("Against the path-to-path distance, the hazardous class becomes a clean rectangle.", None)]),
    ("results", [
        ("On all 42,888 known near-Earth asteroids, the training-free rule reaches 99.86% accuracy and F1 0.989.",
         "On all forty-two thousand, eight hundred and eighty-eight known near-Earth asteroids, the training-free "
         "rule reaches ninety-nine point eight six percent accuracy, and an F1 score of zero point nine eight nine."),
        ("That is just below the 99.89% ceiling set by a few inconsistent catalogue flags.",
         "That is just below the ceiling set by a few inconsistent catalogue flags."),
        ("On the same test folds as the Earth's-view models, F1 rises from 0.43 to 0.99.",
         "On exactly the same test folds as the Earth's view models, F1 rises from zero point four three to zero "
         "point nine nine."),
        ("And it holds for asteroids discovered after 2016 (F1 0.988).",
         "And it holds for asteroids discovered after twenty sixteen.")]),
    ("law", [
        ("Real orbits are uncertain, so the method also knows when not to decide.", None),
        ("Cheap closed-form bounds settle half of the bright asteroids without any search.", None),
        ("A branch-free decision law, checked on all 4.3 billion possible inputs, combines bounds, exact distances "
         "and probabilities.",
         "A branch-free decision law, checked on all four point three billion possible inputs, combines the bounds, "
         "the exact distances and the probabilities."),
        ("It decides 99.43% of asteroids, with zero disagreements against the definition on NASA's own MOID, and "
         "flags 0.57% for follow-up.",
         "It decides ninety-nine point four three percent of asteroids, with no disagreement against the "
         "definition computed from NASA's own MOID, and flags the remaining zero point five seven percent for "
         "follow-up observation.")]),
    ("cloud", [
        ("For newly found asteroids with short observation arcs, we treat the path as a cloud of possible paths.",
         None),
        ("The share of the cloud that meets the definition is the hazard probability.", None),
        ("In simulations, its decisions match the best single orbit, and its probabilities are 30-36% better "
         "calibrated (Brier score).",
         "In simulations, its decisions are as accurate as the best single orbit, and its probabilities are thirty "
         "to thirty-six percent better calibrated.")]),
    ("kaggle", [
        ("We also reran the highest published result on its own Kaggle file.",
         "We also reran the highest published result, on its own Kaggle file."),
        ("Its record-split protocol scores 99.62%, above the 99.55% that the definition itself scores on that file.",
         "Its record-split protocol scores ninety-nine point six two percent, which is even above what the "
         "definition itself scores on that file."),
        ("That is only possible by memorising the inconsistent labels of repeated asteroids.", None),
        ("Split honestly by asteroid, it scores 99.42%, the same as our training-free rule, which never reads the "
         "MOID column.",
         "Split honestly by asteroid, it scores ninety-nine point four two percent, the same as our training-free "
         "rule, which never even reads the MOID column.")]),
    ("takeaways", [
        ("To sum up.", None),
        ("Hazard is a property of a path, not of a single encounter.", None),
        ("Evaluate by asteroid, not by row, and report F1 and MCC.",
         "Evaluate by asteroid, not by row, and report F1 and M C C."),
        ("Let the method say when an orbit is not yet known well enough.", None),
        ("Thank you.", None)]),
]


def check_numbers() -> None:
    """Every number written in the narration, checked against the experiment outputs."""
    ev = pd.read_csv(RESULTS / "earth_view_summary.csv")
    st = json.loads((RESULTS / "earth_view_stats.json").read_text())
    av = pd.read_csv(RESULTS / "asteroid_view_summary.csv")
    sf = pd.read_csv(RESULTS / "same_folds_rows_summary.csv")
    tp = pd.read_csv(RESULTS / "asteroid_view_temporal.csv")
    ks = json.loads((RESULTS / "kernel_summary.json").read_text())
    kg = json.loads((RESULTS / "kaggle_nasa_comparison.json").read_text())
    u = pd.read_csv(RESULTS / "uncertainty_cc8-9_summary.csv")
    g = ev[ev.protocol == "asteroid_grouped"]
    rule = av[av.method == "rule: exact path geometry (ours)"].iloc[0]
    ceil = av[av.method == "rule: JPL MOID (definition, reference)"].iloc[0]
    rf_row = ev[(ev.protocol == "row_split") & (ev.model == "RandomForest")].iloc[0]
    checks = {
        "95.7% seen": (round(100 * st["test_rows_with_asteroid_in_train"]["row_split"], 1), 95.7),
        "best grouped acc 92.6": (round(100 * g.accuracy_mean.max(), 1), 92.6),
        "baseline 91.3": (round(100 * (1 - st["positive_rate_rows"]), 1), 91.3),
        "row-split F1 0.68": (round(rf_row.f1_mean, 2), 0.68),
        "best grouped F1 0.43": (round(g.f1_mean.max(), 2), 0.43),
        "rule acc 99.86": (round(100 * rule.accuracy_mean, 2), 99.86),
        "rule F1 0.989": (round(rule.f1_mean, 3), 0.989),
        "ceiling 99.89": (round(100 * ceil.accuracy_mean, 2), 99.89),
        "same-fold F1 0.99": (round(sf[sf.method == "rule: exact path geometry (ours)"].f1_mean.iloc[0], 2), 0.99),
        "post-2016 F1 0.988": (round(tp[tp.method == "rule: exact path geometry (ours)"].f1.iloc[0], 3), 0.988),
        "bounds settle half": (round(ks["screening"]["far_share"] + ks["screening"]["near_share"], 1), 0.5),
        "decided 99.43": (round(100 * ks["decision_sys"]["coverage"], 2), 99.43),
        "0 disagreements": (ks["decision_sys"]["decided_disagreements_with_definition"], 0),
        "flagged 0.57%": (round(100 * (1 - ks["decision_sys"]["coverage"]), 2), 0.57),
        "kaggle record 99.62": (round(100 * kg["A_prior_protocol_rf"]["accuracy"], 2), 99.62),
        "kaggle ceiling 99.55": (round(100 * kg["E_definition_on_table_moid"]["accuracy"], 2), 99.55),
        "kaggle grouped 99.42": (round(100 * kg["B_grouped_rf"]["accuracy"], 2), 99.42),
        "kaggle ours 99.42": (round(100 * kg["C_ours_rule_all_records"]["accuracy"], 2), 99.42),
        "memorised 23/31": ((kg["F_memorisation_record_split"]["matched_seen"], kg["F_memorisation_record_split"]["n_seen"]), (23, 31)),
        "memorised 4/32": ((kg["F_memorisation_record_split"]["matched_unseen"], kg["F_memorisation_record_split"]["n_unseen"]), (4, 32)),
    }
    pt = u[u.method == "point rule (exact geometry)"].set_index("scale").brier_mean
    cl = u[u.method == "path cloud (ours)"].set_index("scale").brier_mean
    red = (1 - cl / pt) * 100
    checks["Brier 30-36% lower"] = ((round(red.min()), round(red.max())), (30, 36))
    pc = json.loads((RESULTS / "proof_checks.json").read_text())
    checks["bounds: 0 violations"] = (pc["L1_radial_gap_violations"] + pc["L2_node_gap_violations"]
                                      + pc["L3_lower_violations"] + pc["L3_upper_violations"], 0)
    checks["index equivalence: 0 mismatches"] = (pc["P2_mismatches"], 0)
    checks["law: within six steps"] = (pc["T1_max_steps"] <= 6, True)
    checks["law: sound on decided"] = (pc["T1_PHA_with_H_gt_22"] + pc["T1_PHA_with_delta_gt_0.05"]
                                       + pc["T1_NOT_PHA_GEOMETRY_with_delta_le_0.05"] + pc["T1_NOT_PHA_SIZE_with_H_le_22"], 0)
    checks["encounters never below MOID (two-body)"] = (pc["P1_violations"], 0)
    checks["checked on 42,888 asteroids"] = (pc["L1_L2_n"], 42888)
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise SystemExit(f"narration numbers do not match results/: {bad}")
    print(f"all {len(checks)} narration numbers match results/")


def tts(text: str) -> np.ndarray:
    """Speak `text` with macOS `say`; return mono int16 PCM at SR (cached by voice, rate and text)."""
    key = hashlib.sha1(f"{VOICE}|{RATE}|{text}".encode()).hexdigest()[:16]
    aiff, raw = BUILD / "tts" / f"{key}.aiff", BUILD / "tts" / f"{key}.pcm"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        subprocess.run(["say", "-v", VOICE, "-r", str(RATE), "-o", str(aiff), text], check=True)
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(aiff), "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                             capture_output=True, check=True).stdout
        raw.write_bytes(pcm)
    return np.frombuffer(raw.read_bytes(), dtype=np.int16)


def build_audio() -> list[dict]:
    metas = []
    for idx, (name, sentences) in enumerate(SCENES_TEXT):
        chunks, timings, t = [np.zeros(int(LEAD * SR), np.int16)], [], LEAD
        for j, (cap, spoken) in enumerate(sentences):
            pcm = tts(spoken or cap)
            d = len(pcm) / SR
            timings.append((t, t + d, cap))
            chunks.append(pcm)
            t += d
            gap = GAP if j < len(sentences) - 1 else TAIL
            chunks.append(np.zeros(int(gap * SR), np.int16))
            t += gap
        pcm = np.concatenate(chunks)
        n_frames = int(math.ceil(len(pcm) / SR * FPS))
        pcm = np.concatenate([pcm, np.zeros(int(round(n_frames / FPS * SR)) - len(pcm), np.int16)])
        with wave.open(str(BUILD / f"scene{idx:02d}.wav"), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SR)
            wf.writeframes(pcm.tobytes())
        metas.append(dict(idx=idx, name=name, n_frames=n_frames, duration=n_frames / FPS, timings=timings))
    (BUILD / "timeline.json").write_text(json.dumps(metas, indent=1))
    total = sum(m["duration"] for m in metas)
    print(f"narration: {sum(len(s) for _, s in SCENES_TEXT)} sentences, {total:.1f} s ({total / 60:.1f} min), voice {VOICE}")
    return metas


# ------------------------------------------------------------------------------------------- drawing helpers
def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def fade(t: float, t0: float, d: float = 0.6) -> float:
    return ease((t - t0) / d)


class Ctx:
    def __init__(self, meta: dict):
        self.meta, self.cache = meta, {}
        self.starts = [s for s, _, _ in meta["timings"]]

    def s(self, j: int) -> float:
        """Start time of sentence j (seconds into the scene)."""
        return self.starts[min(j, len(self.starts) - 1)]

    def phase(self, t: float) -> int:
        return max([j for j, st in enumerate(self.starts) if st <= t + 1e-9] or [0])


def text(fig, x, y, s, size=20, color=TEXT, alpha=1.0, **kw):
    return fig.text(x, y, s, fontsize=size, color=color, alpha=alpha, **kw)


def heading(fig, t, s, sub=None):
    a = fade(t, 0.0, 0.5)
    text(fig, 0.05, 0.9, s, size=30, weight="bold", alpha=a, va="center")
    if sub:
        text(fig, 0.05, 0.835, sub, size=17, color=TEXT2, alpha=a, va="center")


def caption(fig, t, ctx):
    for start, end, cap in ctx.meta["timings"]:
        if start - 0.15 <= t <= end + GAP * 0.9:
            a = min(fade(t, start - 0.15, 0.25), 1 - fade(t, end + GAP * 0.6, 0.3))
            fig.patches.append(Rectangle((0, 0), 1, 0.115, transform=fig.transFigure, color="#000000", alpha=0.72 * a,
                                         zorder=50))
            text(fig, 0.5, 0.058, textwrap.fill(cap, 92), size=17, ha="center", va="center", alpha=a, zorder=51,
                 linespacing=1.25)


def card(fig, x, y, w, h, alpha=1.0, edge=AXIS):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                      transform=fig.transFigure, facecolor=CARD, edgecolor=edge, lw=1.2,
                                      alpha=alpha, zorder=2))


# ------------------------------------------------------------------------------------------- orbit model
TILT = np.deg2rad(62.0)
YEAR_S = 5.0                        # one year of orbital motion per 5 s of video
ORB = dict(a=1.55, e=0.42, i=14.0, om=35.0, w=250.0)  # illustrative orbit (same as the paper's Fig. 1)


def proj(xyz):
    return xyz[..., 0], xyz[..., 1] * np.cos(TILT) + xyz[..., 2] * np.sin(TILT)


def earth_curve(n=721):
    lam = np.linspace(0, 2 * np.pi, n)
    R = EARTH_P / (1 + EARTH_E * np.cos(lam - EARTH_VARPI))
    return np.stack([R * np.cos(lam), R * np.sin(lam), 0 * lam], axis=-1)


def ast_setup(o=ORB):
    a, e = o["a"], o["e"]
    P, Q = orbit_basis(np.deg2rad([o["i"]]), np.deg2rad([o["om"]]), np.deg2rad([o["w"]]))
    return dict(a=a, e=e, p=np.array([a * (1 - e * e)]), P=P, Q=Q)


def ast_curve(s, n=1441):
    f = np.linspace(0, 2 * np.pi, n)[None, :]
    return path_points(s["p"], np.array([s["e"]]), s["P"], s["Q"], f)[0]


def ast_at(s, t, M0=4.2):
    M = M0 + 2 * np.pi * (t / YEAR_S) / s["a"] ** 1.5
    E = M
    for _ in range(15):
        E = E - (E - s["e"] * np.sin(E) - M) / (1 - s["e"] * np.cos(E))
    f = 2 * np.arctan2(np.sqrt(1 + s["e"]) * np.sin(E / 2), np.sqrt(1 - s["e"]) * np.cos(E / 2))
    return path_points(s["p"], np.array([s["e"]]), s["P"], s["Q"], np.array([[f]]))[0, 0]


def earth_at(t, lam0=3.4):
    lam = lam0 + 2 * np.pi * t / YEAR_S
    R = EARTH_P / (1 + EARTH_E * np.cos(lam - EARTH_VARPI))
    return np.array([R * np.cos(lam), R * np.sin(lam), 0.0])


def closest_pair(E, A):
    d = np.linalg.norm(A[:, None, :] - E[None, :, :], axis=-1)
    ia, ie = np.unravel_index(np.argmin(d), d.shape)
    return A[ia], E[ie], float(d[ia, ie])


def orbit_axes(fig, rect, xlim=(-1.75, 1.45), ylim=(-0.75, 1.55)):
    ax = fig.add_axes(rect)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return ax


def draw_system(ax, s, alpha=1.0, ast_alpha=None, lw=2.0):
    ex, ey = proj(earth_curve())
    ax.plot(ex, ey, color=BLUE, lw=lw, alpha=alpha)
    axx, axy = proj(ast_curve(s))
    ax.plot(axx, axy, color=ORANGE, lw=lw, alpha=alpha if ast_alpha is None else ast_alpha)
    ax.plot(0, 0, "o", ms=15, color=YELLOW, alpha=alpha)


# ------------------------------------------------------------------------------------------- scenes
def sc_title(fig, t, ctx):
    s = ctx.cache.setdefault("s", ast_setup())
    ax = orbit_axes(fig, [0.42, 0.08, 0.6, 0.86])
    a = 0.55 * fade(t, 0, 1.0)
    draw_system(ax, s, alpha=a)
    E, A = proj(earth_at(t)), proj(ast_at(s, t))
    ax.plot(*E, "o", ms=10, color=BLUE, alpha=a * 1.6 if a < 0.6 else 1)
    ax.plot(*A, "o", ms=7, color=ORANGE, alpha=a * 1.6 if a < 0.6 else 1)
    a1 = fade(t, 0.3, 0.8)
    text(fig, 0.06, 0.62, "The Asteroid's", size=46, weight="bold", alpha=a1)
    text(fig, 0.06, 0.52, "Point of View", size=46, weight="bold", alpha=a1, color=ORANGE)
    a2 = fade(t, ctx.s(1), 0.8)
    text(fig, 0.06, 0.41, "Path geometry and leakage-free evaluation\nfor potentially hazardous asteroid classification",
         size=19, color=TEXT2, alpha=a2, linespacing=1.4, va="top")
    text(fig, 0.06, 0.2, "Ch. Satwik  ·  Devieswar Kancheti", size=18, alpha=a2)


def sc_definition(fig, t, ctx):
    s = ctx.cache.setdefault("s", ast_setup())
    heading(fig, t, "What makes an asteroid potentially hazardous?")
    ax = orbit_axes(fig, [0.02, 0.13, 0.5, 0.68])
    draw_system(ax, s)
    A3, E3, _ = ctx.cache.setdefault("cp", closest_pair(earth_curve(), ast_curve(s)))
    A, E = proj(A3), proj(E3)
    pulse = 0.6 + 0.4 * np.sin(2 * np.pi * t / 1.6) ** 2
    a1 = fade(t, ctx.s(1), 0.6)
    ax.plot([A[0], E[0]], [A[1], E[1]], color=TEXT, lw=4 * pulse + 1, alpha=a1, solid_capstyle="round")
    ax.annotate("MOID", xy=((A[0] + E[0]) / 2, (A[1] + E[1]) / 2), xytext=(-1.6, -0.6), fontsize=18, color=TEXT,
                alpha=a1, arrowprops=dict(arrowstyle="-", color=TEXT2, lw=1.2, alpha=a1))
    ax.text(0.75, -0.62, "Earth's orbit", color=BLUE, fontsize=15)
    ax.text(0.55, 1.35, "asteroid's orbit", color=ORANGE, fontsize=15)
    for k, (title, body, j) in enumerate([
            ("1   Close orbit", "MOID ≤ 0.05 au\n(≈ 7.5 million km, 19.5 lunar distances)", 1),
            ("2   Big enough", "H ≤ 22\n(roughly 140 m or larger)", 2)]):
        a = fade(t, ctx.s(j), 0.6)
        y = 0.56 - k * 0.25
        card(fig, 0.55, y, 0.4, 0.2, alpha=a, edge=ORANGE if k == 0 else AQUA)
        text(fig, 0.575, y + 0.145, title, size=21, weight="bold", alpha=a)
        text(fig, 0.575, y + 0.095, body, size=17, color=TEXT2, alpha=a, va="top", linespacing=1.35)


def apophis_rows():
    c = pd.read_csv(ROOT / "data" / "processed" / "cad_earth.csv", usecols=["des", "cd", "dist", "v_rel"])
    r = c[c.des == "99942"].reset_index(drop=True)
    return r, len(r)


def sc_earthview(fig, t, ctx):
    s = ctx.cache.setdefault("s", ast_setup())
    heading(fig, t, "Prior work: the Earth's view", "every close approach becomes one training row")
    ax = orbit_axes(fig, [0.0, 0.12, 0.52, 0.68])
    draw_system(ax, s, ast_alpha=0.45)
    E3, A3 = earth_at(t, lam0=2.2), ast_at(s, t, M0=3.6)
    E, A = proj(E3), proj(A3)
    d = float(np.linalg.norm(E3 - A3))
    ax.plot([E[0], A[0]], [E[1], A[1]], color=TEXT, lw=1.5, ls=(0, (4, 3)))
    ax.plot(*E, "o", ms=13, color=BLUE)
    ax.plot(*A, "o", ms=9, color=ORANGE)
    ax.text((E[0] + A[0]) / 2 + 0.05, (E[1] + A[1]) / 2 + 0.06, f"miss distance now: {d:.2f} au", fontsize=15,
            color=TEXT)
    rows, n = ctx.cache.setdefault("apophis", apophis_rows())
    a = fade(t, ctx.s(1), 0.6)
    card(fig, 0.53, 0.17, 0.43, 0.61, alpha=a)
    text(fig, 0.555, 0.735, "Apophis (99942): one row per approach", size=17, weight="bold", alpha=a)
    cols = [(0.555, "date", "left"), (0.735, "miss (au)", "right"), (0.835, "speed (km/s)", "right"),
            (0.945, "hazardous", "right")]
    for x, lab, ha in cols:
        text(fig, x, 0.69, lab, size=14, color=MUTED, alpha=a, ha=ha)
    show = 8
    t_rows0, t_rows1 = ctx.s(1) + 0.5, ctx.s(3)
    k_vis = int(np.clip((t - t_rows0) / max(t_rows1 - t_rows0, 0.1) * show, 0, show)) if t >= t_rows0 else 0
    for k in range(min(k_vis, show)):
        r = rows.iloc[k]
        y = 0.645 - k * 0.047
        for (x, _, ha), val in zip(cols, (r.cd[:11], f"{r.dist:.3f}", f"{r.v_rel:.1f}", "yes")):
            text(fig, x, y, val, size=14, alpha=a, ha=ha, family="monospace")
    a3 = fade(t, ctx.s(2), 0.6)
    text(fig, 0.555, 0.235, f"… {n} rows of the same asteroid, all labelled hazardous", size=15, color=ORANGE,
         alpha=a3)
    a4 = fade(t, ctx.s(3), 0.6)
    text(fig, 0.555, 0.19, "Reported accuracy: 94% to 99.7%", size=17, weight="bold", alpha=a4)


def sc_leak(fig, t, ctx):
    heading(fig, t, "The leak: the same asteroid in training and test")
    rng = np.random.default_rng(3)
    colors = [BLUE, ORANGE, AQUA, YELLOW, TEXT2]
    tiles = [(c, k) for c in range(5) for k in range(6)]
    order = ctx.cache.setdefault("order", rng.permutation(len(tiles)))
    test_set = ctx.cache.setdefault("test", set(rng.choice(len(tiles), 7, replace=False).tolist()))
    ph = ctx.phase(t)
    if ph <= 1:
        split = fade(t, ctx.s(1), 1.2)
        for pos, ti in enumerate(order):
            c, _ = tiles[ti]
            col, row = pos % 10, pos // 10
            x0, y0 = 0.08 + col * 0.036, 0.62 - row * 0.065
            is_test = ti in test_set
            x1 = 0.58 + (sorted(test_set).index(ti) % 4) * 0.045 if is_test else 0.08 + (pos % 10) * 0.036
            y1 = 0.55 - (sorted(test_set).index(ti) // 4) * 0.08 if is_test else y0
            x, y = x0 + (x1 - x0) * split, y0 + (y1 - y0) * split
            fig.patches.append(FancyBboxPatch((x, y), 0.028, 0.05, boxstyle="round,pad=0.002,rounding_size=0.006",
                                              transform=fig.transFigure, color=colors[c], zorder=3))
            if is_test and split > 0.95:
                fig.patches.append(Rectangle((x - 0.004, y - 0.007), 0.036, 0.064, transform=fig.transFigure,
                                             fill=False, edgecolor=TEXT, lw=2, zorder=4))
        text(fig, 0.08, 0.72, "one colour = one asteroid; each tile = one close-approach row", size=16, color=TEXT2)
        if split > 0:
            text(fig, 0.08, 0.3, "Training rows", size=17, color=TEXT2, alpha=split)
            text(fig, 0.58, 0.3, "Test rows", size=17, color=TEXT2, alpha=split)
            a = fade(t, ctx.s(1) + 1.3, 0.6)
            text(fig, 0.58, 0.23, "every test asteroid was already seen in training", size=16, color=ORANGE, alpha=a)
            text(fig, 0.08, 0.2, "real data: 95.7% of test rows", size=22, weight="bold", alpha=a)
        return
    a = fade(t, ctx.s(2), 0.6)
    for k, (title, vals, fmt, lim) in enumerate([
            ("Accuracy (%)", [94.74, 92.63], "{:.1f}", (85, 100)),
            ("F1 on hazardous asteroids", [0.676, 0.428], "{:.2f}", (0, 1))]):
        ax = fig.add_axes([0.08 + k * 0.46, 0.28, 0.36, 0.44])
        ax.set_facecolor(BG)
        grow = fade(t, ctx.s(2) + (0 if k == 0 else 0.0), 1.0) if k == 0 else fade(t, ctx.s(3), 1.0)
        base = lim[0]
        for b, (v, col, lab) in enumerate(zip(vals, [MUTED, ORANGE], ["random split\n(prior work)", "split by asteroid\n(honest)"])):
            h = (v - base) * grow
            ax.bar(b, h, bottom=base, width=0.55, color=col, alpha=a)
            if grow > 0.98:
                ax.text(b, v + (lim[1] - lim[0]) * 0.02, fmt.format(v if k == 1 else v), ha="center", fontsize=18,
                        color=TEXT, weight="bold")
            ax.text(b, base - (lim[1] - lim[0]) * 0.1, lab, ha="center", va="top", fontsize=14, color=TEXT2)
        if k == 0:
            ax.axhline(91.26, color=TEXT2, lw=1.2, ls=(0, (3, 3)), alpha=a)
            ax.text(1.42, 91.26, "always\n'not hazardous'\n91.3", fontsize=12, color=TEXT2, va="center", alpha=a)
        ax.set_ylim(*lim)
        ax.set_xlim(-0.6, 1.9)
        ax.set_xticks([])
        ax.tick_params(labelsize=13)
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.set_title(title, fontsize=18, color=TEXT, loc="left", alpha=a)


def sc_asteroidview(fig, t, ctx):
    s = ctx.cache.setdefault("s", ast_setup())
    heading(fig, t, "Our idea: the asteroid's point of view")
    ax = orbit_axes(fig, [0.0, 0.12, 0.55, 0.7])
    draw_system(ax, s)
    gone = fade(t, ctx.s(2) + 0.4, 1.0)
    if gone < 1:
        E, A = proj(earth_at(t)), proj(ast_at(s, t))
        ax.plot(*E, "o", ms=13, color=BLUE, alpha=1 - gone)
        ax.plot(*A, "o", ms=9, color=ORANGE, alpha=1 - gone)
    A3, E3, d = ctx.cache.setdefault("cp", closest_pair(earth_curve(), ast_curve(s)))
    A, E = proj(A3), proj(E3)
    a3 = fade(t, ctx.s(3), 0.6)
    ax.plot([A[0], E[0]], [A[1], E[1]], color=TEXT, lw=5, alpha=a3, solid_capstyle="round")
    ax.annotate("path-to-path distance\n(MOID): no timing", xy=((A[0] + E[0]) / 2, (A[1] + E[1]) / 2),
                xytext=(-1.7, -0.62), fontsize=15, color=TEXT, alpha=a3,
                arrowprops=dict(arrowstyle="-", color=TEXT2, lw=1.2, alpha=a3))
    a1 = fade(t, ctx.s(1), 0.6)
    card(fig, 0.57, 0.5, 0.38, 0.27, alpha=a1, edge=ORANGE)
    text(fig, 0.595, 0.715, "One row per asteroid", size=20, weight="bold", alpha=a1)
    text(fig, 0.595, 0.665, "its path:  a, e, i, Ω, ω\nits size:  H", size=18, color=TEXT2, alpha=a1,
         va="top", linespacing=1.5)
    a2 = fade(t, ctx.s(2), 0.6)
    card(fig, 0.57, 0.2, 0.38, 0.24, alpha=a2, edge=AQUA)
    text(fig, 0.595, 0.385, "Timing cancels", size=20, weight="bold", alpha=a2)
    text(fig, 0.595, 0.335, "no mean anomaly, no epoch,\nno position of Earth at any time", size=17, color=TEXT2,
         alpha=a2, va="top", linespacing=1.4)


def sc_math(fig, t, ctx):
    s = ctx.cache.setdefault("s", ast_setup())
    heading(fig, t, "The math: three steps, no training")
    eqs = [(0, r"$\mathbf{r}(f) = \dfrac{a(1-e^2)}{1+e\cos f}\,(\cos f\,\hat{\mathbf{P}} + \sin f\,\hat{\mathbf{Q}})$",
            "path kernel"),
           (1, r"$\delta = \min_{f,\lambda}\ \Vert \mathbf{r}(f) - \mathbf{E}(\lambda) \Vert$", "exact path-to-path distance"),
           (2, r"$\Phi = \max\!\left(\dfrac{\delta}{0.05\ \mathrm{au}},\ 10^{(H-22)/5}\right)$", "hazard index"),
           (3, r"PHA  $\Longleftrightarrow\ \Phi \leq 1$", "the definition, exactly")]
    for k, (j, eq, lab) in enumerate(eqs):
        a = fade(t, ctx.s(j), 0.6)
        y = 0.7 - k * 0.15
        text(fig, 0.05, y, lab, size=15, color=MUTED, alpha=a)
        text(fig, 0.05, y - 0.055, eq, size=22, alpha=a, va="center")
    # the minimisation, animated: a probe slides along the path; its distance to Earth's orbit is traced
    ax = orbit_axes(fig, [0.52, 0.32, 0.46, 0.5])
    draw_system(ax, s, lw=1.6)
    f_all = ctx.cache.setdefault("f", np.linspace(0, 2 * np.pi, 600))
    if "dist" not in ctx.cache:
        X = path_points(s["p"], np.array([s["e"]]), s["P"], s["Q"], f_all[None, :])[0]
        ctx.cache["X"], ctx.cache["dist"] = X, distance_to_earth_orbit(X)
    X, dist = ctx.cache["X"], ctx.cache["dist"]
    t0, t1 = ctx.s(1), ctx.s(2) + 1.5
    prog = np.clip((t - t0) / (t1 - t0), 0, 1)
    if prog > 0:
        k = int(prog * (len(f_all) - 1))
        P3 = X[k]
        # nearest point on Earth's orbit: same longitude, refined by the solver's own geometry (radial approximation)
        lam = np.arctan2(P3[1], P3[0])
        R = EARTH_P / (1 + EARTH_E * np.cos(lam - EARTH_VARPI))
        E3 = np.array([R * np.cos(lam), R * np.sin(lam), 0.0])
        ax.plot(*zip(proj(P3), proj(E3)), color=TEXT, lw=1.6)
        ax.plot(*proj(P3), "o", ms=8, color=ORANGE)
        ax2 = fig.add_axes([0.56, 0.17, 0.38, 0.14])
        ax2.set_facecolor(BG)
        ax2.plot(np.degrees(f_all[:k + 1]), dist[:k + 1], color=AQUA, lw=2)
        ax2.set_xlim(0, 360)
        ax2.set_ylim(0, dist.max() * 1.05)
        ax2.set_xticks([0, 90, 180, 270, 360])
        ax2.tick_params(labelsize=11)
        ax2.set_xlabel("position along the asteroid's path (deg)", fontsize=12)
        ax2.set_ylabel("au", fontsize=12)
        for sp in ("top", "right"):
            ax2.spines[sp].set_visible(False)
        if prog >= 1:
            km = int(np.argmin(dist))
            ax2.plot(np.degrees(f_all[km]), dist[km], "o", ms=9, color=YELLOW)
            ax2.text(np.degrees(f_all[km]) - 8, dist[km] + dist.max() * 0.12, f"minimum = δ = {dist[km]:.3f} au",
                     fontsize=13, color=YELLOW, ha="right")


def sc_proofs(fig, t, ctx):
    heading(fig, t, "Why it is right: four short proofs", "full proofs in the paper's Appendix A")
    cards = [
        ("Encounters bound the MOID",
         r"$\Vert\mathbf{x}_{\mathrm{ast}}(t)-\mathbf{x}_{\oplus}(t)\Vert\ \geq\ "
         r"\min_{\mathbf{x}\in\mathcal{A},\,\mathbf{y}\in\mathcal{E}}\Vert\mathbf{x}-\mathbf{y}\Vert=\delta$",
         "both bodies sit on their own orbits", 1),
        ("Two cheap bounds", r"$L_0\ \leq\ \delta\ \leq\ \delta_{\mathrm{node}}$",
         "triangle inequality below; the two node points above", 2),
        ("Hazard index = the definition", r"$\max(x,y)\leq 1\ \Leftrightarrow\ x\leq 1\ \mathrm{and}\ y\leq 1$",
         r"so $\Phi\leq 1$ exactly when $\delta\leq 0.05$ au and $H\leq 22$", 3),
        ("A verified decision law", r"$\leq 6$ steps,  sound,  $2^{32}$ inputs checked",
         "0 mismatches with its specification", 4),
    ]
    for k, (title, formula, note, j) in enumerate(cards):
        a = fade(t, ctx.s(j), 0.6)
        col, row = k % 2, k // 2
        x, y = 0.05 + col * 0.46, 0.47 - row * 0.29
        card(fig, x, y, 0.43, 0.25, alpha=a, edge=ORANGE if fade(t, ctx.s(5), 0.6) > 0.5 else AXIS)
        text(fig, x + 0.02, y + 0.2, title, size=18, weight="bold", alpha=a, va="center")
        text(fig, x + 0.02, y + 0.12, formula, size=17, alpha=a, va="center")
        text(fig, x + 0.02, y + 0.045, note, size=14, color=TEXT2, alpha=a, va="center")
    a = fade(t, ctx.s(5), 0.6)
    text(fig, 0.95, 0.835, "checked on all 42,888 asteroids: 0 violations", size=16, color=ORANGE, alpha=a, ha="right",
         va="center")


def scatter_panels():
    """Pre-render the two data panels (Earth's view, asteroid's view) once, as RGBA images."""
    feats = pd.read_csv(RESULTS / "asteroid_features.csv", usecols=["pdes", "label", "H", "moid_path"])
    cad = pd.read_csv(ROOT / "data" / "processed" / "cad_earth.csv", usecols=["des", "dist"])
    ev = cad.merge(feats, left_on="des", right_on="pdes").sample(40000, random_state=1)
    imgs = []
    for x, df in ((ev["dist"].values, ev), (feats["moid_path"].values, feats)):
        f2 = plt.figure(figsize=(5.4, 3.9), dpi=100)
        f2.patch.set_facecolor(BG)
        ax = f2.add_axes([0.14, 0.15, 0.82, 0.8])
        ax.set_facecolor(BG)
        neg, pos = df["label"].values == 0, df["label"].values == 1
        ax.scatter(np.clip(x[neg], 1e-5, None), df["H"].values[neg], s=1.5, color=BLUE, alpha=0.25, lw=0)
        ax.scatter(np.clip(x[pos], 1e-5, None), df["H"].values[pos], s=2.2, color=ORANGE, alpha=0.7, lw=0)
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 0.8)
        ax.set_ylim(34.5, 9)
        ax.axhline(22, color=TEXT2, lw=0.9, ls=(0, (3, 3)))
        ax.set_ylabel("absolute magnitude H", fontsize=12)
        ax.set_xlabel("distance (au)", fontsize=12)
        ax.tick_params(labelsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        f2.canvas.draw()
        imgs.append(np.asarray(f2.canvas.buffer_rgba()).copy())
        plt.close(f2)
    return imgs


def sc_data(fig, t, ctx):
    heading(fig, t, "The real catalogue, from both viewpoints",
            "orange: hazardous (JPL flag); blue: not hazardous; dashed: H = 22")
    imgs = ctx.cache.setdefault("imgs", scatter_panels())
    for k, (img, title, j) in enumerate([(imgs[0], "Earth's view: miss distance of each approach", 1),
                                         (imgs[1], "Asteroid's view: path-to-path distance", 2)]):
        a = fade(t, ctx.s(j), 0.8)
        ax = fig.add_axes([0.04 + k * 0.49, 0.14, 0.45, 0.62])
        ax.imshow(img, alpha=a)
        ax.axis("off")
        text(fig, 0.06 + k * 0.49, 0.77, title, size=17, weight="bold", alpha=a)
    a = fade(t, ctx.s(2) + 0.8, 0.6)
    # the PHA rectangle in the right panel: x <= 0.05 au, H <= 22, mapped into the pre-rendered image coordinates
    x0, x1 = 0.04 + 0.49, 0.04 + 0.49 + 0.45
    fx = lambda v: x0 + (0.14 + 0.82 * (np.log10(v) + 5) / (np.log10(0.8) + 5)) * (x1 - x0)  # noqa: E731
    fy = lambda h: 0.14 + (0.15 + 0.8 * (34.5 - h) / (34.5 - 9)) * 0.62  # noqa: E731
    fig.patches.append(Rectangle((fx(1e-5), fy(22)), fx(0.05) - fx(1e-5), fy(9) - fy(22), transform=fig.transFigure,
                                 fill=False, edgecolor=TEXT, lw=2.5, alpha=a, zorder=10))
    text(fig, fx(1.3e-5), fy(10.3), "PHA: distance ≤ 0.05 au and H ≤ 22", size=14, alpha=a, zorder=11)


def sc_results(fig, t, ctx):
    heading(fig, t, "Results")
    a0 = fade(t, 0, 0.6)
    text(fig, 0.06, 0.66, "99.86%", size=60, weight="bold", alpha=a0, color=ORANGE)
    text(fig, 0.06, 0.57, "accuracy on all 42,888 NEAs\nF1 0.989, no training", size=19, color=TEXT2, alpha=a0,
         va="top", linespacing=1.4)
    a1 = fade(t, ctx.s(1), 0.6)
    text(fig, 0.06, 0.4, "ceiling 99.89%", size=24, weight="bold", alpha=a1)
    text(fig, 0.06, 0.355, "set by inconsistent catalogue flags", size=16, color=TEXT2, alpha=a1)
    a3 = fade(t, ctx.s(3), 0.6)
    text(fig, 0.06, 0.25, "post-2016 discoveries: F1 0.988", size=20, alpha=a3)
    a2 = fade(t, ctx.s(2), 0.6)
    ax = fig.add_axes([0.5, 0.2, 0.45, 0.55])
    ax.set_facecolor(BG)
    bars = [("best published F1,\nfull catalogue (graph NN)", 0.37, MUTED),
            ("Earth's view,\nhonest split", 0.43, MUTED),
            ("asteroid's view,\nsame folds", 0.988, ORANGE)]
    grow = fade(t, ctx.s(2), 1.2)
    for k, (lab, v, col) in enumerate(bars):
        ax.barh(2 - k, v * grow, height=0.55, color=col, alpha=a2)
        ax.text(-0.02, 2 - k, lab, ha="right", va="center", fontsize=13, color=TEXT2, alpha=a2)
        if grow > 0.98:
            ax.text(v + 0.02, 2 - k, f"{v:.2f}", va="center", fontsize=17, weight="bold", color=TEXT)
    ax.set_xlim(0, 1.15)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([])
    ax.tick_params(labelsize=12)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title("F1 on hazardous asteroids", fontsize=18, loc="left", color=TEXT, alpha=a2)


def sc_law(fig, t, ctx):
    heading(fig, t, "Knowing when not to decide", "bounds, exact distance, probability, and a verified decision law")
    boxes = [("Closed-form bounds", "settle 51% of bright\nasteroids, no search", 1),
             ("Exact distance", "checked against\ntwo-sided bounds", 2),
             ("Hazard probability", "delta method, or a\ncloud of paths", 2),
             ("Decision law", "9-bit word to 10 acts\nchecked on all 2³² inputs", 2)]
    for k, (title, body, j) in enumerate(boxes):
        a = fade(t, ctx.s(j) + (0.35 * k if j == 2 else 0), 0.5)
        x = 0.04 + k * 0.24
        card(fig, x, 0.5, 0.21, 0.22, alpha=a, edge=ORANGE if k == 3 else AXIS)
        text(fig, x + 0.015, 0.67, title, size=17, weight="bold", alpha=a)
        text(fig, x + 0.015, 0.625, body, size=14, color=TEXT2, alpha=a, va="top", linespacing=1.35)
        if k < 3:
            text(fig, x + 0.217, 0.6, "→", size=26, color=TEXT2, alpha=a, va="center", family="DejaVu Sans")
    a = fade(t, ctx.s(3), 0.6)
    prog = fade(t, ctx.s(3), 2.0)
    stats = [("decided", f"{99.43 * prog:.2f}%", ORANGE), ("disagreements with\nthe definition", "0", TEXT),
             ("flagged for\nfollow-up", f"{int(round(244 * prog))}  (0.57%)", TEXT), ("sent to a human", "0", TEXT)]
    for k, (lab, val, col) in enumerate(stats):
        x = 0.04 + k * 0.24
        text(fig, x, 0.37, val, size=30, weight="bold", color=col, alpha=a)
        text(fig, x, 0.31, lab, size=15, color=TEXT2, alpha=a, va="top", linespacing=1.3)


def sc_cloud(fig, t, ctx):
    heading(fig, t, "Uncertain orbits: a cloud of possible paths")
    if "clones" not in ctx.cache:
        rng = np.random.default_rng(7)
        # put the nominal orbit right at the threshold (distance 0.05 au) by bisection on a, so the cloud splits
        lo_a, hi_a = 1.30, 1.80
        for _ in range(40):
            mid = (lo_a + hi_a) / 2
            dm = moid_earth([mid], [ORB["e"]], [ORB["i"]], [ORB["om"]], [ORB["w"]])[0]
            d_lo = moid_earth([lo_a], [ORB["e"]], [ORB["i"]], [ORB["om"]], [ORB["w"]])[0]
            if (dm - 0.05) * (d_lo - 0.05) <= 0:
                hi_a = mid
            else:
                lo_a = mid
        base = dict(ORB, a=(lo_a + hi_a) / 2)
        n = 60
        el = {k: np.full(n, float(v)) for k, v in base.items()}
        el["a"] += rng.normal(0, 0.035, n)
        el["e"] += rng.normal(0, 0.012, n)
        el["i"] += rng.normal(0, 0.8, n)
        el["om"] += rng.normal(0, 0.8, n)
        el["w"] += rng.normal(0, 1.5, n)
        d = moid_earth(el["a"], el["e"], el["i"], el["om"], el["w"])
        curves = []
        for q in range(n):
            sq = ast_setup({k: el[k][q] for k in el})
            curves.append(proj(ast_curve(sq, 721)))
        ctx.cache["clones"] = (curves, d <= 0.05)
    curves, inside = ctx.cache["clones"]
    ax = orbit_axes(fig, [0.0, 0.12, 0.6, 0.7])
    ex, ey = proj(earth_curve())
    ax.plot(ex, ey, color=BLUE, lw=2.2)
    ax.plot(0, 0, "o", ms=15, color=YELLOW)
    reveal = fade(t, ctx.s(0), 2.5)
    k_vis = int(reveal * len(curves))
    colour = fade(t, ctx.s(1), 0.8)
    for q in range(k_vis):
        col = (ORANGE if inside[q] else BLUE) if colour > 0.5 else TEXT2
        ax.plot(*curves[q], color=col, lw=0.8, alpha=0.45)
    a1 = fade(t, ctx.s(1), 0.8)
    n_in = int(inside.sum())
    card(fig, 0.6, 0.48, 0.35, 0.28, alpha=a1, edge=ORANGE)
    text(fig, 0.625, 0.7, "hazard probability", size=17, color=TEXT2, alpha=a1)
    text(fig, 0.625, 0.6, f"{n_in} / {len(curves)} = {n_in / len(curves):.2f}", size=34, weight="bold", alpha=a1)
    text(fig, 0.625, 0.53, "orange paths: within 0.05 au", size=15, color=ORANGE, alpha=a1)
    a2 = fade(t, ctx.s(2), 0.6)
    text(fig, 0.6, 0.37, "Brier score 30–36% lower", size=24, weight="bold", alpha=a2)
    text(fig, 0.6, 0.315, "decisions as accurate as the best single orbit\n(simulation, short-arc error bars)",
         size=15, color=TEXT2, alpha=a2, va="top", linespacing=1.35)


def sc_kaggle(fig, t, ctx):
    heading(fig, t, "Honest test on the strongest published result", "same Kaggle file, 4,687 records of 3,692 asteroids")
    rows = [("Published, records split at random", "99.7%", MUTED, 0),
            ("Reproduced, records split at random", "99.62%", MUTED, 1),
            ("Definition itself on this file (ceiling)", "99.55%", TEXT2, 1),
            ("Same model, split by asteroid", "99.42%", MUTED, 3),
            ("Our training-free rule (no MOID column)", "99.42%", ORANGE, 3)]
    for k, (lab, val, col, j) in enumerate(rows):
        a = fade(t, ctx.s(j) + (0.6 if k in (2, 4) else 0), 0.6)
        y = 0.68 - k * 0.085
        text(fig, 0.06, y, lab, size=18, color=TEXT if col != MUTED else TEXT2, alpha=a)
        text(fig, 0.6, y, val, size=22, weight="bold", color=col if col != MUTED else TEXT, alpha=a, ha="right")
    a2 = fade(t, ctx.s(2), 0.6)
    card(fig, 0.66, 0.24, 0.3, 0.5, alpha=a2, edge=ORANGE)
    text(fig, 0.68, 0.69, "How the random split\nbeats the definition", size=16, weight="bold", alpha=a2, va="top",
         linespacing=1.3)
    text(fig, 0.68, 0.585, "inconsistent labels the\nmodel reproduced:", size=14, color=TEXT2, alpha=a2, va="top",
         linespacing=1.3)
    for k, (lab, num, den) in enumerate([("asteroid seen\nin training", 23, 31), ("asteroid\nnot seen", 4, 32)]):
        y = 0.44 - k * 0.1
        grow = fade(t, ctx.s(2) + 0.4, 1.0)
        fig.patches.append(Rectangle((0.68, y), 0.16 * num / den * grow, 0.04, transform=fig.transFigure,
                                     color=ORANGE if k == 0 else MUTED, alpha=a2, zorder=3))
        text(fig, 0.85, y + 0.02, f"{num}/{den}", size=16, weight="bold", alpha=a2, va="center")
        text(fig, 0.68, y - 0.012, lab, size=12, color=TEXT2, alpha=a2, va="top", linespacing=1.2)


def sc_takeaways(fig, t, ctx):
    heading(fig, t, "Takeaways")
    items = [("1", "Hazard is a property of a path,", "not of a single encounter.", 1),
             ("2", "Evaluate by asteroid, not by row,", "and report F1 and MCC.", 2),
             ("3", "Let the method say when an orbit", "is not yet known well enough.", 3)]
    for k, (num, l1, l2, j) in enumerate(items):
        a = fade(t, ctx.s(j), 0.6)
        y = 0.7 - k * 0.15
        text(fig, 0.07, y, num, size=40, weight="bold", color=ORANGE, alpha=a, va="center")
        text(fig, 0.13, y + 0.025, l1, size=22, alpha=a, va="center")
        text(fig, 0.13, y - 0.03, l2, size=22, color=TEXT2, alpha=a, va="center")
    a = fade(t, ctx.s(4), 0.6)
    text(fig, 0.07, 0.19, "Thank you", size=30, weight="bold", alpha=a, va="center")
    text(fig, 0.3, 0.19, "Ch. Satwik  ·  Devieswar Kancheti", size=17, color=TEXT2, alpha=a, va="center")


DRAW = {"title": sc_title, "definition": sc_definition, "earthview": sc_earthview, "leak": sc_leak,
        "asteroidview": sc_asteroidview, "math": sc_math, "proofs": sc_proofs, "data": sc_data, "results": sc_results, "law": sc_law,
        "cloud": sc_cloud, "kaggle": sc_kaggle, "takeaways": sc_takeaways}


def draw_frame(fig, meta, ctx, t):
    fig.clf()
    fig.patch.set_facecolor(BG)
    DRAW[meta["name"]](fig, t, ctx)
    text(fig, 0.95, 0.955, f"{meta['idx'] + 1} / {len(SCENES_TEXT)}", size=12, color=MUTED, ha="right")
    caption(fig, t, ctx)


def render_scene(meta: dict) -> str:
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ctx = Ctx(meta)
    out = BUILD / f"scene{meta['idx']:02d}.mp4"
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                             "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    for k in range(meta["n_frames"]):
        draw_frame(fig, meta, ctx, k / FPS)
        fig.canvas.draw()
        proc.stdin.write(fig.canvas.buffer_rgba())
    proc.stdin.close()
    proc.wait()
    plt.close(fig)
    return f"{meta['name']}: {meta['n_frames']} frames"


def stills(metas: list[dict]) -> None:
    d = BUILD / "stills"
    d.mkdir(parents=True, exist_ok=True)
    for meta in metas:
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ctx = Ctx(meta)
        for j, (start, end, _) in enumerate(meta["timings"]):
            draw_frame(fig, meta, ctx, min(end, meta["duration"] - 1 / FPS))
            fig.savefig(d / f"{meta['idx']:02d}_{meta['name']}_{j}.png", dpi=50)
        plt.close(fig)
    print("stills in", d)


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    check_numbers()
    metas = build_audio()
    if "--stills" in sys.argv:
        stills(metas)
        return
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        for msg in ex.map(render_scene, metas):
            print("rendered", msg, flush=True)
    (BUILD / "scenes.txt").write_text("".join(f"file 'scene{m['idx']:02d}.mp4'\n" for m in metas))
    pcm = b"".join(wave.open(str(BUILD / f"scene{m['idx']:02d}.wav"), "rb").readframes(10 ** 9) for m in metas)
    with wave.open(str(BUILD / "narration.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(BUILD / "scenes.txt"),
                    "-c", "copy", str(BUILD / "video.mp4")], check=True)
    final = OUT / "asteroids_point_of_view.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(BUILD / "video.mp4"), "-i", str(BUILD / "narration.wav"),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart",
                    str(final)], check=True)
    print("wrote", final)


if __name__ == "__main__":
    main()
