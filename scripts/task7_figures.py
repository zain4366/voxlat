"""Task 7 figures (results/figures/task7_*.png, 300 dpi).

  task7_error_vs_N.png         main paper figure: discrepancy delta vs 1/N for k_n, K_t, C_nn, G_t, G_sz
                               (top: gyroid & diamond, bottom: blend; phase range as bars, physical-model lines)
  task7_discrepancy_models.png CV error of the physical form vs physical + GP (RQ1 second half)
  task7_velocity_slices.png    near-wall flow: streamwise velocity slices of strips (N = 2) and the
                               plane-averaged velocity profile vs the bulk cell (same cut)
  task7_graded_error.png       graded strips: gradient effect vs |grad rho*| L
  task7_convergence.png        resolution (n = 24/32/48) and wall-thickness checks

Needs data/finite_gap.csv (scripts/task7_finite_gap.py) and models/finite_gap_correction.json
(scripts/task7_fit_discrepancy.py). The velocity fields are computed once (~2-4 min) and cached in
results/task7_velocity_fields.npz.

  python scripts/task7_figures.py            # all figures
  python scripts/task7_figures.py --only error_vs_N graded
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from voxlat.homogenization.finite_gap import (  # noqa: E402
    PRIMARY_QUANTITIES,
    distinct_cuts,
    FiniteGapCorrection,
    GapSpec,
    build_strip,
    bulk_cell,
    compute_properties,
)
from voxlat.utils import results_dir, save_figure  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task7_fit_discrepancy import load_table  # noqa: E402

# ---- style (validated categorical slots 1-3, light surface) ------------------------
SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
MORPH_COLOR = {"G": "#2a78d6", "D": "#eb6834", "B0.50": "#1baf7a"}
MORPH_NAME = {"G": "gyroid", "D": "diamond", "B0.50": "blend w = 0.5"}
RHO_MARKER = {0.25: "o", 0.35: "s", 0.45: "^"}
RHO_DASH = {0.25: (0, (4, 2)), 0.35: "-", 0.45: (0, (1, 1.5))}
QLABEL = {
    "k_n": "through-gap conductivity $k_{rr}$",
    "K_t": "in-plane permeability $K_t$",
    "C_nn": "normal stiffness $C_{rrrr}$",
    "G_t": "transverse shear $G_t$ (walls slide)",
    "G_sz": "in-plane shear $G_{sz}$",
}
BLUES = LinearSegmentedColormap.from_list(
    "vox_blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 1.6,
    "lines.solid_capstyle": "round",
})


def _uniform(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df[(df["n"] == n) & (df["wall"] == 0.5) & (df["gradient"] == 0.0) & (df["a_z"] == 1.0)]


def _prod_n(df: pd.DataFrame) -> int:
    corr = Path("models/finite_gap_correction.json")
    if corr.exists():
        return int(FiniteGapCorrection.load(corr).meta["n"])
    return int(df["n"].mode().iloc[0])


def _jacket_band(ax, invN: bool = True) -> None:
    # h = 6 mm, L = 2...6 mm -> N = 1...3
    lo, hi = (1 / 3, 1.0) if invN else (1.0, 3.0)
    ax.axvspan(lo, hi, color="#f0efec", zorder=0, lw=0)


# =============================================================================
# Figure 1: discrepancy vs 1/N
# =============================================================================
def fig_error_vs_N(df: pd.DataFrame, corr: FiniteGapCorrection | None) -> Path:
    n = _prod_n(df)
    u = distinct_cuts(_uniform(df, n))
    rows = (("pure", ("G", "D"), "gyroid & diamond"), ("blend", ("B0.50",), "blend w = 0.5"))
    fig, axes = plt.subplots(2, 5, figsize=(15.5, 7.4), constrained_layout=True)
    for i, (_, morphs, rlabel) in enumerate(rows):
        for k, q in enumerate(PRIMARY_QUANTITIES):
            ax = axes[i, k]
            _jacket_band(ax)
            ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
            for (m, rho), g in u[u["morphology"].isin(morphs)].groupby(["morphology", "rho"]):
                st = g.groupby("N")[f"delta_{q}"].agg(["mean", "min", "max"]) * 100
                x = 1.0 / st.index.to_numpy(float)
                c = MORPH_COLOR.get(m, INK2)
                ax.errorbar(x, st["mean"], yerr=[st["mean"] - st["min"], st["max"] - st["mean"]], fmt="none",
                            ecolor=c, elinewidth=0.9, capsize=0, alpha=0.55, zorder=2)
                ax.plot(x, st["mean"], ls="none", marker=RHO_MARKER[round(rho, 2)], ms=4.8, mfc=c, mec=SURFACE,
                        mew=0.8, zorder=3)
                if corr is not None and q in corr.physical.models:
                    xx = np.linspace(0.0, 1.0, 101)[1:]
                    yy = corr.physical.predict(q, 1.0 / xx, g["w"].iloc[0], rho) * 100
                    ax.plot(xx, yy, color=c, ls=RHO_DASH[round(rho, 2)], lw=1.1, zorder=2)
            ax.set_xlim(0, 1.04)
            if i == 1:
                ax.set_xlabel("1 / N   (N = cells across the gap)")
            if k == 0:
                ax.set_ylabel(f"{rlabel}\n" + r"$\delta$ = q$_{strip}$ / q$_{bulk}$ - 1  [%]")
            ax.set_title(f"({'abcdefghij'[5 * i + k]}) {QLABEL[q]}", loc="left", fontsize=8.8)
            top = ax.secondary_xaxis("top")
            top.set_xticks([1 / 8, 1 / 4, 1 / 3, 1 / 2, 1.0])
            top.set_xticklabels(["N=8", "4", "3", "2", "1"], fontsize=7.5, color=MUTED)
            top.tick_params(length=0)
            top.spines["top"].set_visible(False)
    handles = [Line2D([], [], color=MORPH_COLOR[m], lw=1.6, label=MORPH_NAME[m]) for m in MORPH_COLOR]
    handles += [Line2D([], [], color=INK2, marker=RHO_MARKER[r], ls=RHO_DASH[r], lw=1.0, ms=4.8,
                       label=f"$\\rho^*$ = {r:.2f}") for r in RHO_MARKER]
    handles += [plt.Rectangle((0, 0), 1, 1, color="#f0efec", label="jacket range: h = 6 mm, L = 2-6 mm (N = 1-3)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=7, fontsize=8)
    fig.suptitle("Finite-gap discrepancy of bulk closures: lattice strips between bonded solid walls "
                 f"(n = {n} voxels/cell). Markers = mean over symmetry-distinct cut phases, bars = their range, "
                 "lines = physical model 1/N form", fontsize=10, color=INK)
    return save_figure(fig, "task7_error_vs_N")


def fig_models() -> Path | None:
    cvp = Path(results_dir()) / "task7_model_cv.csv"
    if not cvp.exists():
        return None
    cv = pd.read_csv(cvp)
    cv = cv[cv["split"] == "theta"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), constrained_layout=True, sharey=False)
    for ax, (stratum, title) in zip(axes, (("G/D", "(a) gyroid & diamond"), ("blend", "(b) blend w = 0.5"))):
        c = cv[cv["stratum"] == stratum].set_index("quantity").reindex(PRIMARY_QUANTITIES)
        x = np.arange(len(c))
        wbar = 0.26
        bars = [("none", "bulk closure, uncorrected", "#b8b6ae"), ("physical", "physical 1/N form", "#2a78d6"),
                ("physical+gp", "physical + GP on residuals", "#eb6834")]
        for j, (col, lab, colr) in enumerate(bars):
            if col in c:
                ax.bar(x + (j - 1) * wbar, c[col] * 100, width=wbar * 0.88, color=colr, label=lab, zorder=2)
        for i, v in enumerate(c["phase_floor"] * 100):
            ax.plot([i - 1.5 * wbar, i + 1.5 * wbar], [v, v], color=INK, lw=1.1, zorder=3)
        ax.plot([], [], color=INK, lw=1.1, label="scatter over cut phases (floor)")
        ax.set_xticks(x, ["$k_{rr}$", "$K_t$", "$C_{rrrr}$", "$G_t$", "$G_{sz}$"])
        ax.set_ylabel(r"rms error of $\delta$ [percentage points]")
        ax.set_title(f"{title}: leave-one-(w, $\\rho^*$, $a_z$)-out CV", loc="left")
        ax.grid(axis="x", visible=False)
        if stratum == "G/D":
            ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Can a simple correction remove the finite-gap error? (all N = 1-8, uniform + graded + stretched strips)",
                 fontsize=10)
    return save_figure(fig, "task7_discrepancy_models")


# =============================================================================
# Figure 2: velocity slices
# =============================================================================
def _velocity_fields(n: int = 48, cache: Path | None = None) -> dict[str, np.ndarray]:
    cache = cache or Path(results_dir()) / "task7_velocity_fields.npz"
    if cache.exists():
        d = dict(np.load(cache))
        if int(d.get("n", 0)) == n:
            return d
    out: dict[str, np.ndarray] = {"n": np.array(n)}
    for m, w in (("G", 0.0), ("D", 1.0), ("B0.50", 0.5)):
        spec = GapSpec(w, 0.35, 2.0, 0.0, n=n)
        strip = build_strip(spec)
        p = compute_properties(strip.solid, n, spec.n_lattice, spec.n_wall, quantities=("K",), keep_velocity=True)
        cell = bulk_cell(w, 0.35, n, 0.0)
        pb = compute_properties(cell, n, n, 0, quantities=("K",), keep_velocity=True)
        # staggered face velocities (k: 0 = force e_y, 1 = force e_z); keep the streamwise component
        out[f"{m}_solid"] = strip.solid
        out[f"{m}_uy"] = p.velocity[0, 1].astype(np.float32)
        out[f"{m}_uz"] = p.velocity[1, 2].astype(np.float32)
        out[f"{m}_bulk_uy"] = pb.velocity[0, 1].astype(np.float32)
        out[f"{m}_bulk_uz"] = pb.velocity[1, 2].astype(np.float32)
        out[f"{m}_Kt"] = p.K_t
        out[f"{m}_Kb"] = pb.K_t
        out[f"{m}_T"] = np.array(p.T)
    np.savez_compressed(cache, **out)
    return out


def fig_velocity_slices(n: int = 48) -> Path:
    d = _velocity_fields(n)
    n = int(d["n"])
    morphs = ("G", "D", "B0.50")
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.2), width_ratios=[1.35, 1.0], constrained_layout=True)
    for i, m in enumerate(morphs):
        solid = d[f"{m}_solid"]
        nl = solid.shape[0] - int(round(0.5 * n))
        uz = d[f"{m}_uz"].astype(float)
        # cell-centred streamwise velocity (u_z on +z faces -> average the two faces of each cell)
        uzc = 0.5 * (uz + np.roll(uz, 1, axis=2))
        Ub = float(d[f"{m}_bulk_uz"].mean())  # bulk superficial velocity (same force, same grid)
        k = solid.shape[2] // 2
        sl = np.where(solid[:, :, k], np.nan, uzc[:, :, k] / (Ub / (1 - 0.35)))  # / bulk interstitial mean
        # show the lattice plus a slice of each wall (periodic image)
        nwv = min(n // 4, solid.shape[0] - nl)
        img = np.concatenate([sl[-nwv:], sl[:nl], sl[nl:nl + nwv]], axis=0)
        sol = np.concatenate([solid[-nwv:, :, k], solid[:nl, :, k], solid[nl:nl + nwv, :, k]], axis=0)
        x0 = -nwv / n
        ext = [x0, x0 + img.shape[0] / n, 0, solid.shape[1] / n]
        ax = axes[i, 0]
        ax.imshow(np.where(sol, 1.0, np.nan).T, origin="lower", extent=ext, cmap=ListedColormap(["#c9c7bf"]),
                  vmin=0, vmax=1, interpolation="nearest")
        vmax = np.nanpercentile(img, 99.5)
        im = ax.imshow(img.T, origin="lower", extent=ext, cmap=BLUES, vmin=0, vmax=vmax, interpolation="nearest")
        for xw in (0.0, nl / n):
            ax.axvline(xw, color=INK, lw=1.0)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel("x / L  (gap normal r; walls at 0 and 2)")
        ax.set_ylabel("y / L  (s)")
        ax.set_title(f"({'ace'[i]}) {MORPH_NAME[m]}, $\\rho^*$ = 0.35, N = 2: axial velocity $u_z$ "
                     f"(slice z = L/2)", loc="left")
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
        cb.set_label(r"$u_z$ / bulk mean interstitial", color=INK2)
        cb.outline.set_visible(False)
        # plane-averaged superficial velocity profiles
        ax = axes[i, 1]
        x = (np.arange(nl) + 0.5) / n
        for comp, ls, lab in (("uz", "-", "flow along z (axial)"), ("uy", (0, (4, 2)), "flow along y (s)")):
            us = d[f"{m}_{comp}"].astype(float)
            ub = d[f"{m}_bulk_{comp}"].astype(float)
            ds = 2 if comp == "uz" else 1
            ps = us.mean(axis=(1, 2))[:nl]
            pb = np.tile(ub.mean(axis=(1, 2)), int(np.ceil(nl / n)))[:nl]
            Um = float(ub.mean())
            ax.plot(x, pb / Um, color=MUTED, ls=ls, lw=1.2)
            ax.plot(x, ps / Um, color=MORPH_COLOR[m], ls=ls, lw=1.8)
            ax.fill_between(x, ps / Um, pb / Um, color=MORPH_COLOR[m], alpha=0.10, lw=0)
            del ds
        Kt, Kb = d[f"{m}_Kt"], d[f"{m}_Kb"]
        ax.set_xlim(0, nl / n)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("x / L")
        ax.set_ylabel("plane-avg. superficial velocity / bulk mean")
        ax.set_title(f"({'bdf'[i]}) strip (color) vs bulk, same cut (gray): "
                     f"$K_{{zz}}$ x{Kt[1, 1] / Kb[1, 1]:.2f}, $K_{{ss}}$ x{Kt[0, 0] / Kb[0, 0]:.2f}", loc="left")
        if i == 0:
            ax.legend(handles=[Line2D([], [], color=INK2, ls="-", label="flow along z (axial)"),
                               Line2D([], [], color=INK2, ls=(0, (4, 2)), label="flow along y (s)"),
                               Line2D([], [], color=MUTED, label="bulk cell (periodic)"),
                               Line2D([], [], color=MORPH_COLOR[m], label="strip between walls")],
                      fontsize=7.5, loc="lower center")
    fig.suptitle(f"Near-wall flow in finite-gap strips (Stokes, unit body force; n = {n} voxels/cell; "
                 "gray = solid)", fontsize=10)
    return save_figure(fig, "task7_velocity_slices")


# =============================================================================
# Figure 3: graded strips
# =============================================================================
def fig_graded(df: pd.DataFrame, corr: FiniteGapCorrection | None) -> Path | None:
    gp = Path(results_dir()) / "task7_graded.csv"
    if not gp.exists():
        return None
    gt = pd.read_csv(gp)
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.6), constrained_layout=True)
    # (a) illustration: density profile of a graded strip
    ax = axes.flat[0]
    spec = GapSpec(0.0, 0.35, 4.0, 0.0, gradient=0.075, n=32)
    strip = build_strip(spec)
    x = spec.layer_x
    ax.plot(x, strip.layer_density, color=MUTED, lw=0.9, label="planar solid fraction (voxels)")
    ax.plot(x, strip.layer_density.reshape(4, 32).mean(axis=1).repeat(32), color=MORPH_COLOR["G"], lw=1.8,
            label="cell average")
    ax.plot(x, spec.layer_rho, color=INK, lw=1.2, ls=(0, (4, 2)), label=r"target $\rho^*(x)$")
    ax.set_xlabel("x / L")
    ax.set_ylabel("relative density")
    ax.set_title(r"(a) graded gyroid strip, N = 4, $|\nabla\rho^*| L$ = 0.075", loc="left")
    ax.legend(fontsize=7.5, loc="upper left")
    for k, q in enumerate(PRIMARY_QUANTITIES):
        ax = axes.flat[k + 1]
        ax.axhline(0, color=MUTED, lw=0.8)
        for (m, N), g in gt.groupby(["morphology", "N"]):
            st = g.groupby("gradient")[f"Delta_{q}"].agg(["mean", "min", "max"]) * 100
            gx = st.index.to_numpy(float)
            c = MORPH_COLOR.get(m, INK2)
            ax.errorbar(gx, st["mean"], yerr=[st["mean"] - st["min"], st["max"] - st["mean"]], fmt="none",
                        ecolor=c, elinewidth=0.9, alpha=0.6)
            ax.plot(gx, st["mean"], ls="-" if N == 4 else (0, (4, 2)), lw=1.0, color=c,
                    marker="o" if N == 4 else "s", ms=4.5, mfc=c, mec=SURFACE, mew=0.8)
        if corr is not None and q in corr.physical.models:
            gg = np.linspace(0, 0.16, 50)
            ax.plot(gg, corr.physical.models[q].c_g * gg**2 * 100, color=INK, lw=1.0, ls=(0, (1, 1.5)))
        ax.axvline(0.05, color=MUTED, lw=0.8, ls=(0, (3, 2)))
        ax.set_xlabel(r"grading gradient $|\nabla\rho^*|\,L$ (change per cell)")
        ax.set_ylabel(r"gradient effect $\Delta_g$ [%]")
        ax.set_title(f"({'bcdef'[k]}) {QLABEL[q]}", loc="left")
    handles = [Line2D([], [], color=MORPH_COLOR[m], lw=1.6, label=MORPH_NAME[m]) for m in MORPH_COLOR]
    handles += [Line2D([], [], color=INK2, marker="o", lw=1.0, ms=4.5, label="N = 4"),
                Line2D([], [], color=INK2, marker="s", ls=(0, (4, 2)), lw=1.0, ms=4.5, label="N = 2"),
                Line2D([], [], color=INK, ls=(0, (1, 1.5)), lw=1.0, label=r"fit $c_g g^2$ (gyroid & diamond)"),
                Line2D([], [], color=MUTED, ls=(0, (3, 2)), lw=0.8, label="config limit 0.05 (kept)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=7, fontsize=8)
    fig.suptitle(r"Graded strips: local-closure error from the density gradient, "
                 r"$\Delta_g = (1+\delta_{graded})/(1+\delta_{uniform}) - 1$ (same N, mid-gap $\rho^*$ = 0.35, cut phase)",
                 fontsize=10)
    return save_figure(fig, "task7_graded_error")


# =============================================================================
# Figure 4: convergence and wall thickness
# =============================================================================
def fig_convergence(df: pd.DataFrame) -> Path:
    n = _prod_n(df)
    base = df[(df["gradient"] == 0) & (df["a_z"] == 1.0) & (df["rho"] == 0.35) & (df["phase"] == 0.0)
              & df["w"].isin([0.0, 1.0])]
    fig, axes = plt.subplots(2, 5, figsize=(13.0, 5.6), constrained_layout=True)
    for k, q in enumerate(PRIMARY_QUANTITIES):
        ax = axes[0, k]
        sub = base[(base["wall"] == 0.5)]
        for (m, N), g in sub.groupby(["morphology", "N"]):
            if N not in (1.0, 2.0, 4.0) or g["n"].nunique() < 2:
                continue
            g = g.sort_values("n")
            ax.plot(1.0 / g["n"], g[f"delta_{q}"] * 100, color=MORPH_COLOR[m], marker="o" if N == 1 else ("s" if N == 2 else "^"),
                    ms=4.5, mec=SURFACE, mew=0.8, lw=1.2)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("1 / n  (n = voxels per cell)")
        ax.set_title(QLABEL[q], loc="left", fontsize=8.5)
        if k == 0:
            ax.set_ylabel(r"$\delta$ [%] (phase 0, $\rho^*$ = 0.35)")
        ax.set_xticks([1 / 48, 1 / 32, 1 / 24], ["1/48", "1/32", "1/24"])
        ax = axes[1, k]
        sub = base[base["n"] == n]
        for (m, N), g in sub.groupby(["morphology", "N"]):
            if g["wall"].nunique() < 2:
                continue
            g = g.sort_values("wall")
            ax.plot(g["wall"], g[f"delta_{q}"] * 100, color=MORPH_COLOR[m], marker="o" if N == 1 else "s",
                    ms=4.5, mec=SURFACE, mew=0.8, lw=1.2, ls="-" if N == 1 else (0, (4, 2)))
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)))
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.125, 0.25, 0.5, 1.0], ["1/8", "1/4", "1/2", "1"])
        ax.set_xlabel(r"wall thickness $t_w$ / L")
        if k == 0:
            ax.set_ylabel(rf"$\delta$ [%] (n = {n})")
    handles = [Line2D([], [], color=MORPH_COLOR[m], lw=1.6, label=MORPH_NAME[m]) for m in ("G", "D")]
    handles += [Line2D([], [], color=INK2, marker=mk, ls="none", ms=4.5, label=f"N = {N}")
                for mk, N in (("o", 1), ("s", 2), ("^", 4))]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=8)
    fig.suptitle("Checks: resolution (top; strip and bulk on the same grid) and wall thickness (bottom; "
                 "default 0.5 L dashed)", fontsize=10)
    return save_figure(fig, "task7_convergence")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of: error_vs_N velocity graded convergence")
    ap.add_argument("--velocity-n", type=int, default=48)
    args = ap.parse_args(argv)
    df = load_table()
    cp = Path("models/finite_gap_correction.json")
    corr = FiniteGapCorrection.load(cp) if cp.exists() else None
    todo = args.only or ["error_vs_N", "models", "velocity", "graded", "convergence"]
    if "error_vs_N" in todo:
        print(fig_error_vs_N(df, corr))
    if "models" in todo:
        print(fig_models())
    if "graded" in todo:
        print(fig_graded(df, corr))
    if "convergence" in todo:
        print(fig_convergence(df))
    if "velocity" in todo:
        print(fig_velocity_slices(args.velocity_n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
