"""Paper figures (PDF + PNG) from the results/ tables.

Palette: validated categorical slots (blue, orange, aqua, yellow, magenta) on a light surface; lighter slots
fall below 3:1 contrast, so every figure also carries marker shapes and direct labels.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from common import AU_KM, FIGS, H_LIMIT, MOID_LIMIT_AU, RESULTS
from orbit_geometry import EARTH_E, EARTH_P, EARTH_VARPI, orbit_basis, path_points

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "text.color": INK, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})
COL_W, PAGE_W = 3.5, 7.16  # IEEE column / page width, inches


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Fig. 1: concept
def fig_concept() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.55))
    lam = np.linspace(0, 2 * np.pi, 721)
    R = EARTH_P / (1 + EARTH_E * np.cos(lam - EARTH_VARPI))
    view = np.deg2rad(62.0)  # oblique camera: tilt the ecliptic so heights become visible

    def project(xyz):
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        return x, y * np.cos(view) + z * np.sin(view)

    earth = np.stack([R * np.cos(lam), R * np.sin(lam), 0 * lam], axis=-1)
    a, e, inc, om, w = 1.55, 0.42, 14.0, 35.0, 250.0  # illustrative orbit, not a real object
    p = np.array([a * (1 - e**2)])
    P, Q = orbit_basis(np.deg2rad([inc]), np.deg2rad([om]), np.deg2rad([w]))
    f = np.linspace(0, 2 * np.pi, 1441)[None, :]
    ast = path_points(p, np.array([e]), P, Q, f)[0]
    # closest pair of points between the two paths (brute force, for the drawing only)
    d = np.linalg.norm(ast[:, None, :] - earth[None, :, :], axis=-1)
    ia, ie = np.unravel_index(np.argmin(d), d.shape)

    # (a) Earth's view: one snapshot per close approach
    ax = axes[0]
    ex, ey = project(earth)
    ax.plot(ex, ey, color=BLUE, lw=1.1)
    ax.plot(*project(ast), color=ORANGE, lw=1.1, alpha=0.5, ls=(0, (3, 2)))
    ke, ka = ie + 50, ia  # Earth arrives weeks after the asteroid passes the closest-path point
    E_pt, A_pt = project(earth[ke % len(earth)]), project(ast[ka % len(ast)])
    ax.plot(0, 0, marker="o", ms=7, color=YELLOW, mec=INK2, mew=0.4)
    ax.annotate("", xy=A_pt, xytext=E_pt, arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9, shrinkA=4, shrinkB=3,
                                                          mutation_scale=7))
    ax.plot(*E_pt, "o", ms=6, color=BLUE, mec="white", mew=0.8, zorder=4)
    ax.plot(*A_pt, "o", ms=4.5, color=ORANGE, mec="white", mew=0.8, zorder=4)
    mid = ((E_pt[0] + A_pt[0]) / 2, (E_pt[1] + A_pt[1]) / 2)
    ax.annotate("miss distance\nat time $t$", xy=mid, xytext=(-1.25, -0.66), fontsize=7, color=INK, ha="left",
                va="top", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=1, shrinkB=2))
    ax.text(E_pt[0] + 0.06, E_pt[1] - 0.12, "Earth", fontsize=7, color=BLUE, ha="left", va="top")
    ax.text(A_pt[0] - 0.08, A_pt[1] + 0.07, "asteroid", fontsize=7, color=ORANGE, ha="right", va="bottom")
    ax.set_title("(a) Earth's view: one row per close approach", loc="left")
    ax.text(-1.55, -1.32, "Features change with every encounter (timing,\nspeed, distance); the same asteroid appears many times.",
            fontsize=6.5, color=INK2, ha="left", va="bottom")

    # (b) Asteroid's view: two fixed paths
    ax = axes[1]
    ax.plot(ex, ey, color=BLUE, lw=1.1)
    ax.plot(*project(ast), color=ORANGE, lw=1.3)
    ax.plot(0, 0, marker="o", ms=7, color=YELLOW, mec=INK2, mew=0.4)
    # nodes: where the asteroid path pierces the ecliptic plane
    sign = np.sign(ast[:, 2])
    for k in np.where(np.diff(sign) != 0)[0]:
        ax.plot(*project(ast[k]), marker="D", ms=3.8, color="white", mec=ORANGE, mew=1.0)
    A_c, E_c = project(ast[ia]), project(earth[ie])
    ax.plot([A_c[0], E_c[0]], [A_c[1], E_c[1]], color=INK, lw=1.6, solid_capstyle="round")
    ax.annotate("path-to-path distance (MOID):\nno timing involved", xy=((A_c[0] + E_c[0]) / 2, (A_c[1] + E_c[1]) / 2),
                xytext=(-1.25, -0.66), fontsize=7, color=INK, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=1, shrinkB=2))
    ax.text(1.08, -0.62, "Earth's orbit", fontsize=7, color=BLUE, ha="left")
    ax.text(1.3, 0.95, "asteroid's path", fontsize=7, color=ORANGE, ha="left")
    ax.text(-1.55, -1.32, "Each asteroid is one row: its own path and size.\nDiamonds: nodes, where the path crosses Earth's orbital plane.",
            fontsize=6.5, color=INK2, ha="left", va="bottom")
    ax.set_title("(b) Asteroid's view: one row per asteroid", loc="left")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-1.65, 2.25)
        ax.set_ylim(-1.35, 1.5)
        ax.axis("off")
    save(fig, "fig1_concept")


# ---------------------------------------------------------------- Fig. 2: the two viewpoints in data
def fig_viewpoints() -> None:
    feats = pd.read_csv(RESULTS / "asteroid_features.csv", usecols=["pdes", "label", "H", "moid_path"])
    cad = pd.read_csv(RESULTS.parent / "data" / "processed" / "cad_earth.csv", usecols=["des", "dist"])
    ev = cad.merge(feats, left_on="des", right_on="pdes", how="inner")
    ev = ev.sample(60000, random_state=1)
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.45), sharey=True)
    panels = [(axes[0], ev["dist"].values, ev, "(a) Earth's view: miss distance of each close approach (au)"),
              (axes[1], feats["moid_path"].values, feats, "(b) Asteroid's view: path-to-path distance (au)")]
    for ax, x, df, title in panels:
        neg, pos = df["label"].values == 0, df["label"].values == 1
        ax.scatter(np.clip(x[neg], 1e-5, None), df["H"].values[neg], s=1.2, color=BLUE, alpha=0.18, lw=0,
                   rasterized=True, label="not hazardous")
        ax.scatter(np.clip(x[pos], 1e-5, None), df["H"].values[pos], s=1.8, color=ORANGE, alpha=0.55, lw=0,
                   rasterized=True, label="PHA (JPL flag)")
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 0.8)
        ax.set_ylim(34.5, 9)
        ax.set_title(title, loc="left")
        ax.axhline(H_LIMIT, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax = axes[1]
    ax.axvline(MOID_LIMIT_AU, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.fill_between([1e-5, MOID_LIMIT_AU], 9, H_LIMIT, color=ORANGE, alpha=0.07, lw=0)
    ax.text(1.4e-5, 10.2, "PHA region:\ndistance ≤ 0.05 au and H ≤ 22", fontsize=6.5, color=INK, va="top")
    axes[0].text(1.4e-5, 10.2, "Classes overlap: a miss distance only\nbounds the path distance from above",
                 fontsize=6.5, color=INK, va="top")
    axes[0].set_ylabel("absolute magnitude H (brighter = larger)")
    leg = axes[1].legend(loc="lower right", markerscale=5, handletextpad=0.3)
    for h in leg.legend_handles:
        h.set_alpha(1)
    save(fig, "fig2_viewpoints")


# ---------------------------------------------------------------- Fig. 3: leakage
def fig_leakage() -> None:
    ev = pd.read_csv(RESULTS / "earth_view_summary.csv")
    same = pd.read_csv(RESULTS / "same_folds_rows_summary.csv")
    stats = json.loads((RESULTS / "earth_view_stats.json").read_text())
    models = ["LogReg", "KNN", "MLP", "LightGBM", "XGBoost", "RandomForest"]
    best_ast = same[(same.method == "LightGBM") & (same.features == "path+kernels")].iloc[0]
    rule = same[same.method == "rule: exact path geometry (ours)"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.2), sharey=True)
    for ax, metric, label in ((axes[0], "accuracy", "accuracy"), (axes[1], "mcc", "Matthews correlation (MCC)")):
        for yi, mname in enumerate(models):
            leaky = ev[(ev.protocol == "row_split") & (ev.model == mname)][f"{metric}_mean"].iloc[0]
            honest = ev[(ev.protocol == "asteroid_grouped") & (ev.model == mname)][f"{metric}_mean"].iloc[0]
            ax.plot([honest, leaky], [yi, yi], color=AXIS, lw=1.2, zorder=1)
            ax.plot(leaky, yi, "o", ms=5, color=ORANGE, mec="white", mew=0.6, zorder=3,
                    label="Earth's view, row split (prior work)" if yi == 0 else None)
            ax.plot(honest, yi, "s", ms=4.6, color=BLUE, mec="white", mew=0.6, zorder=3,
                    label="Earth's view, asteroid-grouped split" if yi == 0 else None)
        yb = len(models) + 0.4
        ax.plot(best_ast[f"{metric}_mean"], yb, "D", ms=5, color=AQUA, mec="white", mew=0.6, zorder=3,
                label="asteroid's view, same grouped folds")
        ax.plot(rule[f"{metric}_mean"], yb + 0.9, "^", ms=5.4, color=INK, mec="white", mew=0.6, zorder=3,
                label="training-free path rule, same folds")
        ax.set_xlabel(label)
        if metric == "accuracy":
            base = 1 - stats["positive_rate_rows"]
            ax.axvline(base, color=INK2, lw=0.6, ls=(0, (2, 2)))
            ax.text(base - 0.002, -0.95, "always 'not hazardous'", fontsize=6.3, color=INK2, ha="right")
            ax.set_xlim(0.89, 1.002)
        else:
            ax.set_xlim(0.2, 1.02)
    axes[0].set_yticks(list(range(len(models))) + [len(models) + 0.4, len(models) + 1.3])
    axes[0].set_yticklabels(["Logistic reg.", "k-NN", "MLP", "LightGBM", "XGBoost", "Random forest",
                             "LightGBM (path)", "Path rule"])
    axes[0].set_ylim(-1.3, len(models) + 1.9)
    axes[1].legend(loc="lower right", fontsize=6.3, handletextpad=0.3, borderaxespad=0.2)
    save(fig, "fig3_leakage")


# ---------------------------------------------------------------- Fig. 4: geometry validation
def fig_geometry() -> None:
    f = pd.read_csv(RESULTS / "asteroid_features.csv", usecols=["moid", "moid_path", "d_lin", "d_node"])
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.45))
    lo = 1e-6
    ax = axes[0]
    hb = ax.hexbin(np.clip(f.moid, lo, None), np.clip(f.moid_path, lo, None), xscale="log", yscale="log",
                   gridsize=60, cmap=SEQ_BLUE, norm=LogNorm(), mincnt=1, linewidths=0, rasterized=True)
    ax.plot([lo, 1], [lo, 1], color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel("JPL MOID (au)")
    ax.set_ylabel("our exact path-to-path distance (au)")
    ax.set_title("(a) Exact path geometry vs. JPL", loc="left")
    ax = axes[1]
    sub = f[f.moid_path < 0.3]
    ax.hexbin(np.clip(sub.moid_path, lo, None), np.clip(sub.d_lin, lo, None), xscale="log", yscale="log",
              gridsize=60, cmap=SEQ_BLUE, norm=LogNorm(), mincnt=1, linewidths=0, rasterized=True)
    ax.plot([lo, 0.3], [lo, 0.3], color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.axvline(MOID_LIMIT_AU, color=ORANGE, lw=0.7)
    ax.axhline(MOID_LIMIT_AU, color=ORANGE, lw=0.7)
    ax.text(0.056, 2e-6, "0.05 au", color=ORANGE, fontsize=6.5)
    ax.set_xlabel("exact path-to-path distance (au)")
    ax.set_ylabel("closed-form tangent-line estimate (au)")
    ax.set_title("(b) Closed form vs. exact (distance < 0.3 au)", loc="left")
    cb = fig.colorbar(hb, ax=axes, shrink=0.85, pad=0.015)
    cb.set_label("asteroids per cell")
    cb.outline.set_linewidth(0.4)
    save(fig, "fig4_geometry")


# ---------------------------------------------------------------- Fig. 5: uncertainty
def fig_uncertainty() -> None:
    # The closed-form single-orbit rule (MCC ~0.94) is left to the table so the y-range can resolve the rest.
    series = [("point rule (exact geometry)", BLUE, "o", "single orbit, exact rule"),
              ("path cloud (ours)", ORANGE, "D", "path cloud (ours)"),
              ("LightGBM", AQUA, "s", "LightGBM"),
              ("LightGBM + path cloud (ours)", YELLOW, "^", "LightGBM + path cloud (ours)")]
    pools = [("cc5-7", "moderately determined (condition code 5-7)"),
             ("cc8-9", "short-arc orbits (condition code 8-9)")]
    fig, axes = plt.subplots(2, 2, figsize=(PAGE_W, 4.1), sharex=True)
    for r, (tag, pool_label) in enumerate(pools):
        s = pd.read_csv(RESULTS / f"uncertainty_{tag}_summary.csv")
        for c, (metric, label) in enumerate((("mcc", "MCC (higher is better)"), ("brier", "Brier score (lower is better)"))):
            ax = axes[r, c]
            for name, color, marker, lab in series:
                d = s[s.method == name].sort_values("scale")
                ax.plot(d.scale, d[f"{metric}_mean"], color=color, lw=1.4, marker=marker, ms=4.5, mec="white",
                        mew=0.6, label=lab)
            ax.set_xscale("log", base=2)
            ticks = sorted(s.scale.unique())
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"×{t:g}" for t in ticks])
            ax.minorticks_off()
            ax.set_ylabel(label)
            ax.set_title(f"({'abcd'[2 * r + c]}) {pool_label}", loc="left")
            if r == 1:
                ax.set_xlabel("error bars, multiple of the real ones")
    axes[0, 0].legend(loc="lower left", fontsize=6.3, handletextpad=0.3)
    fig.tight_layout(h_pad=1.2)
    save(fig, "fig5_uncertainty")


if __name__ == "__main__":
    import sys

    wanted = sys.argv[1:] or ["concept", "viewpoints", "leakage", "geometry", "uncertainty"]
    for name in wanted:
        globals()[f"fig_{name}"]()
