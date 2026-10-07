"""Task 1 figures: TPMS cells, surface-area convergence, morphology vs density, feasible (rho, L) map.

Run:  python scripts/task1_tpms_figures.py            (~1-2 min; needs the morphology table,
                                                      see scripts/task1_build_tables.py)
Out:  results/figures/task1_tpms_cells.png
      results/figures/task1_asf_convergence.png
      results/figures/task1_morphology_vs_density.png
      results/figures/task1_feasible_region.png
      results/task1_asf_convergence.csv
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from voxlat.geometry.tpms import (
    DIAMOND_AREA_EXACT,
    GYROID_AREA_EXACT,
    TPMSParams,
    feasible_region,
    load_morphology_table,
    sample_level_set,
    specific_surface_area,
    threshold_from_table,
    voxelize,
)
from voxlat.utils import get_logger, load_config, results_dir, save_figure

LOG = get_logger("task1.figures")
CASES = [(0.0, "Gyroid (w = 0)"), (1.0, "Diamond (w = 1)"), (0.5, "Blend (w = 0.5)")]
COLORS = {0.0: "#1f5aa6", 0.5: "#7a3fa0", 1.0: "#c4581a"}
RHO = 0.30


def _closed_mesh(cell_field: np.ndarray, c: float):
    """Marching cubes of the solid {g <= c}, capped at the cell faces (for rendering)."""
    g = np.pad(cell_field - c, 1, mode="constant", constant_values=1.0)
    verts, faces, _n, _v = marching_cubes(g, level=0.0)
    n = cell_field.shape[0]
    verts = (verts - 0.5) / n  # voxel-centre coordinates -> cell coordinates
    return np.clip(verts, 0, 1), faces


def fig_cells() -> None:
    n = 40
    fig = plt.figure(figsize=(10.5, 6.8))
    for j, (w, title) in enumerate(CASES):
        solid, phi, c = voxelize(TPMSParams(w=w, rho=RHO), n, return_field=True)
        verts, faces = _closed_mesh(phi, c)
        ax = fig.add_subplot(2, 3, j + 1, projection="3d")
        mesh = Poly3DCollection(verts[faces], linewidths=0)
        # simple Lambert shading from a fixed light
        tri = verts[faces]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        light = np.array([0.4, -0.5, 0.75]) / np.linalg.norm([0.4, -0.5, 0.75])
        shade = 0.35 + 0.65 * np.abs(nrm @ light)
        base = np.array(matplotlib.colors.to_rgb(COLORS[w]))
        mesh.set_facecolor(np.clip(base[None, :] * shade[:, None] + 0.15 * (1 - shade[:, None]), 0, 1))
        ax.add_collection3d(mesh)
        ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_zlim(0, 1)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=24, azim=-58)
        ax.set_xticks([0, 1]), ax.set_yticks([0, 1]), ax.set_zticks([0, 1])
        ax.tick_params(labelsize=7, pad=0)
        ax.set_title(f"{title}\nrho* = {solid.mean():.3f}, c = {c:+.3f}", fontsize=9)

        ax2 = fig.add_subplot(2, 3, j + 4)
        k = n // 2
        ax2.imshow(solid[:, :, k].T, origin="lower", cmap="Greys", vmin=-0.15, vmax=1.6,
                   extent=(0, 1, 0, 1), interpolation="nearest")
        fine = sample_level_set((160, 160, 160), w)
        # slice of the fine field through the same physical plane z = (k + 1/2) / n
        zf = int(round((k + 0.5) / n * 160 - 0.5))
        xs = (np.arange(160) + 0.5) / 160
        ax2.contour(xs, xs, fine[:, :, zf].T, levels=[c], colors=[COLORS[w]], linewidths=1.4, linestyles="solid")
        ax2.set_title(f"slice z = {(k + 0.5) / n:.3f} L  ({n}$^2$ voxels)", fontsize=8.5)
        ax2.set_xlabel("x / L", fontsize=8), ax2.set_ylabel("y / L", fontsize=8)
        ax2.tick_params(labelsize=7)
    fig.suptitle(f"Network TPMS unit cells at relative density {RHO}: solid (grey voxels) "
                 "and exact level set (line)", fontsize=10)
    fig.tight_layout()
    LOG.info("saved %s", save_figure(fig, "task1_tpms_cells"))


def fig_asf_convergence() -> pd.DataFrame:
    ns = np.array([16, 24, 32, 48, 64, 96, 128])
    rows = []
    for w, title in CASES:
        c = float(threshold_from_table(w, RHO))
        for n in ns:
            t0 = time.perf_counter()
            a = specific_surface_area(sample_level_set((n,) * 3, w), c, n=n)
            rows.append(dict(w=w, n=n, rho=RHO, c=c, a_sf_L=a, seconds=time.perf_counter() - t0))
    # c = 0 (minimal surfaces) for G and D: compare with the exact minimal-surface areas
    for w, exact in ((0.0, GYROID_AREA_EXACT), (1.0, DIAMOND_AREA_EXACT)):
        for n in ns:
            a = specific_surface_area(sample_level_set((n,) * 3, w), 0.0, n=n)
            rows.append(dict(w=w, n=n, rho=0.5, c=0.0, a_sf_L=a, seconds=np.nan, exact=exact))
    df = pd.DataFrame(rows)
    # Richardson extrapolation (p = 2) from the two finest grids
    df["a_inf"] = np.nan
    for (w, rho), g in df.groupby(["w", "rho"]):
        a1, a2 = g.sort_values("n")["a_sf_L"].values[-2:]
        a_inf = a2 + (a2 - a1) / ((128 / 96) ** 2 - 1)
        df.loc[g.index, "a_inf"] = a_inf
    df["rel_err"] = (df["a_sf_L"] - df["a_inf"]).abs() / df["a_inf"]

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for w, title in CASES:
        g = df[(df.w == w) & (df.rho == RHO)].sort_values("n")
        axs[0].plot(g.n, g.a_sf_L, "o-", color=COLORS[w], label=title, ms=4)
        axs[0].axhline(g.a_inf.iloc[0], color=COLORS[w], lw=0.8, ls=":")
        axs[1].loglog(g.n, g.rel_err, "o-", color=COLORS[w], label=title, ms=4)
    ref_n = np.array([16, 128])
    axs[1].loglog(ref_n, 0.02 * (16 / ref_n) ** 2, "k--", lw=1, label="slope -2")
    for ax in axs:
        ax.set_xlabel("voxels per cell n")
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel(r"$a_{sf} L$  (marching cubes)")
    axs[0].set_title(rf"Specific surface area, $\rho^*$ = {RHO} (dotted: extrapolated)", fontsize=9)
    axs[1].set_ylabel("relative error vs Richardson limit")
    axs[1].set_title("Convergence: second order in h = L/n", fontsize=9)
    axs[1].legend(fontsize=7, frameon=False)
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    LOG.info("saved %s", save_figure(fig, "task1_asf_convergence"))
    out = results_dir() / "task1_asf_convergence.csv"
    df.to_csv(out, index=False)
    LOG.info("saved %s", out)
    for w, exact in ((0.0, GYROID_AREA_EXACT), (1.0, DIAMOND_AREA_EXACT)):
        g = df[(df.w == w) & (df.rho == 0.5)]
        LOG.info("w=%.0f, c=0: a_inf = %.4f vs exact minimal surface %.4f (%.3f %%)",
                 w, g.a_inf.iloc[0], exact, 100 * (g.a_inf.iloc[0] / exact - 1))
    for n in (24, 32, 48, 64):
        e = df[(df.n == n) & (df.rho == RHO)].rel_err.max()
        LOG.info("rho=%.2f: max rel. error of a_sf at n=%d: %.4f %%", RHO, n, 100 * e)
    return df


def fig_morphology() -> None:
    tab = load_morphology_table()
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
    rho_f = np.linspace(0.05, 0.95, 181)
    for w, title in CASES:
        axs[0].plot(rho_f, threshold_from_table(w, rho_f), color=COLORS[w], label=title)
        i = int(np.argmin(np.abs(tab.w - w)))
        r = tab.rho
        axs[1].plot(r, tab.data["wall_min"][i], "-", color=COLORS[w], label=f"{title}: min LT")
        axs[1].plot(r, tab.data["wall_throat"][i], "--", color=COLORS[w], lw=1, label="throat")
        axs[2].plot(r, tab.data["pore_throat"][i], "-", color=COLORS[w], label=f"{title}: throat")
        axs[2].plot(r, tab.data["pore_min"][i], ":", color=COLORS[w], lw=1, label="min LT")
    axs[0].set_xlabel(r"relative density $\rho^*$"), axs[0].set_ylabel("level c")
    axs[0].set_title(r"Threshold table c(w, $\rho^*$)", fontsize=9)
    axs[1].set_title("Solid: thinnest wall (min local thickness)", fontsize=9)
    axs[1].set_ylabel("thickness / L")
    axs[2].set_title("Fluid: throat (passable sphere)", fontsize=9)
    axs[2].set_ylabel("pore size / L")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.set_xlabel(r"relative density $\rho^*$")
    for ax in axs[1:]:
        ax.axvspan(0.2, 0.5, color="0.9", zorder=-1)
        ax.set_xlim(tab.rho[0], tab.rho[-1])
    axs[0].legend(fontsize=7, frameon=False)
    axs[1].legend(fontsize=6, frameon=False, ncol=1)
    axs[2].legend(fontsize=6, frameon=False, ncol=1)
    fig.suptitle(f"Network TPMS morphology vs density (n = {tab.n} voxels per cell; "
                 "grey band = design range)", fontsize=10)
    fig.tight_layout()
    LOG.info("saved %s", save_figure(fig, "task1_morphology_vs_density"))


def fig_feasible() -> None:
    cfg = load_config()
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (w, title) in zip(axs, CASES):
        fr = feasible_region(w, cfg=cfg)
        ax.contourf(fr.rho, fr.L * 1e3, fr.mask.T.astype(float), levels=[-0.5, 0.5, 1.5],
                    colors=["#f2d7d5", "#d5ecd4"])
        ax.plot(fr.rho, fr.L_min_wall * 1e3, color="#b03a2e", lw=1.4, label="wall >= 0.35 mm")
        ax.plot(fr.rho, fr.L_min_pore * 1e3, color="#1f5aa6", lw=1.4, label="pore throat >= 0.8 mm")
        ax.axhspan(1.0, fr.L_bounds[0] * 1e3, color="0.85", zorder=-1)
        ax.text(0.21, 0.5 * (1.0 + fr.L_bounds[0] * 1e3), "below design bound L = 2 mm",
                fontsize=7, va="center", color="0.3")
        ax.set_xlim(*fr.rho_bounds)
        ax.set_ylim(1.0, fr.L_bounds[1] * 1e3)
        ax.set_xlabel(r"relative density $\rho^*$")
        ax.set_title(f"{title}: {100 * fr.feasible_fraction():.0f} % of box feasible", fontsize=9)
        ax.grid(alpha=0.3)
        LOG.info("w=%.1f feasible fraction %.3f; L_min(rho=0.2)=%.2f mm, L_min(rho=0.5)=%.2f mm",
                 w, fr.feasible_fraction(), fr.L_min[0] * 1e3, fr.L_min[-1] * 1e3)
    axs[0].set_ylabel("cell size L [mm]")
    axs[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Manufacturable region (green) of the design box: L >= L_min(w, rho*)", fontsize=10)
    fig.tight_layout()
    LOG.info("saved %s", save_figure(fig, "task1_feasible_region"))


def main() -> None:
    t0 = time.perf_counter()
    fig_cells()
    fig_asf_convergence()
    fig_morphology()
    fig_feasible()
    LOG.info("done in %.0f s", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
