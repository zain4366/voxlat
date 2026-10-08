"""Task 4 verification figure: the MAC Stokes solver against exact and published solutions.

(a) plane Poiseuille: discrete velocity profile vs parabola; relative K error vs H
    (exactly 2/H^2: second order, mirror-ghost walls);
(b) square and 2:1 rectangular ducts vs the series solution (second order) and inclined
    slits (45 deg: second order; slopes 1:2 and 1:3: first-order staircase error);
(c) simple-cubic sphere arrays: K/l^2 vs solid fraction c against Zick & Homsy (1982) and
    the Sangani & Acrivos (1982) series; voxel spheres at n = 24, 32, 48, 64;
(d) sphere-array error vs n (reference interpolated to the realized voxel fraction).

Run:  python scripts/task4_verification.py [--quick]     (~2 min on 2 cores; --quick ~30 s)
      python scripts/task4_verification.py --replot    (figure from the saved CSVs)
Out:  results/task4_verification_channels.csv, results/task4_verification_spheres.csv
      results/figures/task4_verification.png
"""

from __future__ import annotations

import argparse

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from voxlat.homogenization.stokes import (
    ZICK_HOMSY_SC,
    inclined_slit_cell,
    inclined_slit_reference,
    permeability,
    rectangular_duct_mean_velocity,
    sangani_acrivos_sc_drag,
    sc_sphere_permeability,
    slit_permeability,
    sphere_array_cell,
    zick_homsy_sc_drag,
)
from voxlat.utils import get_logger, results_dir, save_figure

LOG = get_logger("task4.verification")


def channel_rows(quick: bool) -> list[dict]:
    rows = []
    for H in (2, 4, 8, 16, 32, 64):
        cell = np.zeros((1, H + 1, 1), bool)
        cell[:, 0] = True
        r = permeability(cell, tol=1e-12)
        ex = float(slit_permeability(H)) * H / (H + 1)
        rows.append({"case": "plates", "N": H, "h_over_D": 1 / H, "K": r.K[0, 0], "K_exact": ex,
                     "err": r.K[0, 0] / ex - 1})
    for aspect in (1, 2):
        for m in (4, 8, 16, 32, 64) if not quick else (4, 8, 16):
            cell = np.zeros((m + 1, aspect * m + 1, 1), bool)
            cell[0] = True
            cell[:, 0] = True
            r = permeability(cell, tol=1e-11)
            ex = rectangular_duct_mean_velocity(m, aspect * m) * m * aspect * m / ((m + 1) * (aspect * m + 1))
            rows.append({"case": f"duct {aspect}:1", "N": m, "h_over_D": 1 / m, "K": r.K[2, 2],
                         "K_exact": ex, "err": r.K[2, 2] / ex - 1})
    for slope in (1, 2, 3):
        for N in (10, 20, 40, 80, 160) if not quick else (10, 20, 40):
            cell = inclined_slit_cell(N, 0.5, slope=slope)
            r = permeability(cell, tol=1e-11)
            k, t, nrm = inclined_slit_reference(N, 0.5, slope=slope)
            H = 0.5 * N / np.sqrt(1 + slope**2)  # fluid layer thickness in voxels
            for name, val in (("t", t @ r.K @ t), ("z", r.K[2, 2])):
                rows.append({"case": f"slit 1:{slope} along {name}", "N": N, "h_over_D": 1 / H,
                             "K": val, "K_exact": k, "err": val / k - 1,
                             "K_normal_rel": nrm @ r.K @ nrm / k})
    return rows


def _sphere(n: int, c: float) -> dict:
    cell = sphere_array_cell(n, c)
    cv = float(cell.mean())
    r = permeability(cell, voxel_size=1 / n, symmetry="cubic")
    k_ref_c = float(sc_sphere_permeability(c, ZICK_HOMSY_SC[c]))
    k_ref_v = float(sc_sphere_permeability(cv, zick_homsy_sc_drag(cv)))
    return {"c": c, "n": n, "c_voxel": cv, "K_l2": r.K[0, 0], "K_ref_target_c": k_ref_c,
            "K_ref_voxel_c": k_ref_v, "err_vs_target": r.K[0, 0] / k_ref_c - 1,
            "err_vs_voxel_c": r.K[0, 0] / k_ref_v - 1, "tortuosity": r.tortuosity[0],
            "iters": r.iterations[0], "time_s": r.wall_time}


def plot(ch: pd.DataFrame, sph: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    # (a) Poiseuille profile + error
    ax = axes[0, 0]
    H = 8
    cell = np.zeros((1, H + 1, 1), bool)
    cell[:, 0] = True
    r = permeability(cell, tol=1e-12, directions=[0], return_fields=True)
    u = r.velocity[0, 0][0, 1:, 0]
    y = np.arange(1, H + 1) - 0.5
    yy = np.linspace(0, H, 200)
    ax.plot(yy / H, yy * (H - yy) / 2 / (H * H / 8), "k-", lw=1, label="exact parabola")
    ax.plot(y / H, u / (H * H / 8), "o", color="#1f5aa6", label=f"MAC, H = {H} voxels")
    ax.set_xlabel("y / H")
    ax.set_ylabel(r"$u / u_{max}$")
    ax.set_title("(a) Plane Poiseuille", fontsize=10, loc="left")
    ins = ax.inset_axes([0.33, 0.08, 0.36, 0.38])
    p = ch[ch.case == "plates"]
    ins.loglog(p.N, p.err, "o-", color="#1f5aa6", ms=3)
    ins.loglog(p.N, 2 / p.N**2, "k:", lw=1)
    ins.set_xticks([2, 8, 32], ["2", "8", "32"])
    ins.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ins.set_xlabel("H [voxels]", fontsize=7)
    ins.set_ylabel("K error", fontsize=7)
    ins.tick_params(labelsize=6)
    ins.text(0.45, 0.7, r"$2/H^2$ (exact)", transform=ins.transAxes, fontsize=7)
    ax.legend(fontsize=8, loc="upper right")

    # (b) ducts and inclined slits
    ax = axes[0, 1]
    styles = {"duct 1:1": ("#1f5aa6", "s-"), "duct 2:1": ("#5b8fd0", "s--"),
              "slit 1:1 along t": ("#2b8a3e", "^-"), "slit 1:2 along t": ("#c4581a", "o-"),
              "slit 1:3 along t": ("#7a3fa0", "D-")}
    # (the "along z" slit errors change sign at coarse N -> kept in the CSV, not plotted on |err|)
    for case, (col, st) in styles.items():
        g = ch[ch.case == case]
        if len(g):
            ax.loglog(g.h_over_D, g.err.abs(), st, color=col, ms=4, label=case)
    xs = np.array([0.004, 0.25])
    ax.loglog(xs, 0.25 * xs, "k:", lw=1)
    ax.loglog(xs, 2.0 * xs**2, "k--", lw=1)
    ax.text(0.006, 0.25 * 0.006 * 1.6, "slope 1", fontsize=8)
    ax.text(0.012, 2 * 0.012**2 * 0.35, "slope 2", fontsize=8)
    ax.set_xlabel("h / D   (D = side of duct / fluid-layer thickness)")
    ax.set_ylabel("|K error|")
    ax.set_title("(b) Ducts (series) and inclined slits", fontsize=10, loc="left")
    ax.legend(fontsize=7, ncol=2)

    # (c) sphere arrays
    ax = axes[1, 0]
    cs = np.linspace(0.005, 0.5236, 300)
    ax.semilogy(cs, sc_sphere_permeability(cs, zick_homsy_sc_drag(cs)), "k-", lw=0.8,
                label="Zick & Homsy (1982), interpolated")
    csa = np.linspace(0.005, 0.3, 100)
    ax.semilogy(csa, sc_sphere_permeability(csa, sangani_acrivos_sc_drag(csa)), "k--", lw=0.8,
                label="Sangani & Acrivos (1982) series")
    czh = np.array(sorted(ZICK_HOMSY_SC))
    ax.semilogy(czh, sc_sphere_permeability(czh, [ZICK_HOMSY_SC[c] for c in czh]), "ks", mfc="none",
                ms=7, label="Zick & Homsy table")
    cols = {24: "#9ecae1", 32: "#6baed6", 48: "#3182bd", 64: "#08519c"}
    for n, g in sph.groupby("n"):
        ax.semilogy(g.c_voxel, g.K_l2, "o", color=cols.get(n, "gray"), ms=4, label=f"MAC, n = {n}")
    ax.set_xlabel("solid fraction c")
    ax.set_ylabel(r"$K / \ell^2$")
    ax.set_title("(c) Simple-cubic sphere arrays", fontsize=10, loc="left")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    for c, g in sph.groupby("c"):
        g = g.sort_values("n")
        ax.plot(g.n, g.err_vs_voxel_c * 100, "o-", ms=4, label=f"c = {c}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(-1, 1, color="0.9", zorder=0)
    ax.set_xlabel("n (voxels per period)")
    ax.set_ylabel("K error vs Zick & Homsy at the voxel c  [%]")
    ax.set_title("(d) Sphere-array error", fontsize=10, loc="left")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = save_figure(fig, "task4_verification")
    LOG.info("figure -> %s", path)


def main(quick: bool = False) -> None:
    ch = pd.DataFrame(channel_rows(quick))
    ns = (16, 24, 32, 48, 64) if not quick else (16, 24, 32)
    cs = (0.027, 0.064, 0.125, 0.216, 0.343, 0.45) if not quick else (0.064, 0.216)
    jobs = sorted([(n, c) for n in ns for c in cs], key=lambda t: -t[0])
    sph = pd.DataFrame(Parallel(n_jobs=-1, verbose=2)(delayed(_sphere)(n, c) for n, c in jobs))
    sph = sph.sort_values(["c", "n"])
    out = results_dir()
    ch.to_csv(out / "task4_verification_channels.csv", index=False)
    sph.to_csv(out / "task4_verification_spheres.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(ch.pivot_table(index="N", columns="case", values="err"))
        print(sph.pivot_table(index="n", columns="c", values="err_vs_voxel_c"))
    plot(ch, sph[sph.n >= 24])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true")
    a = ap.parse_args()
    if a.replot:
        out = results_dir()
        sph = pd.read_csv(out / "task4_verification_spheres.csv")
        plot(pd.read_csv(out / "task4_verification_channels.csv"), sph[sph.n >= 24])
    else:
        main(a.quick)
