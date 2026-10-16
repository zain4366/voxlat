"""Task 3 figures: stiffness, elastic anisotropy and stress localization of gyroid, diamond
and a 50/50 blend (network solids, AlSi10Mg E = 70 GPa, nu = 0.33) vs relative density.

Estimator: production two-grid Richardson R(32, 64), p = 1, on the default grid
(``extrapolated_elasticity_tpms``; choice from scripts/task3_convergence.py, STATUS.md,
Task 3). Localization statistics come from the fine grid (n = 64), not extrapolated.

Run:  python scripts/task3_stiffness_vs_density.py [--quick]   (~15 min on 2 cores; --quick ~20 s)
Out:  results/task3_stiffness_vs_density.csv, results/task3_powerlaw_fits.csv
      results/figures/task3_youngs_vs_density.png
      results/figures/task3_anisotropy_vs_density.png
      results/figures/task3_localization_vs_density.png
"""

from __future__ import annotations

import argparse
import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from voxlat.geometry import TPMSParams
from voxlat.homogenization.elasticity import (
    extrapolated_elasticity_tpms,
    cubic_deviation,
    hashin_shtrikman_porous,
    voigt_reuss_hill,
    youngs_extremes,
)
from voxlat.utils import get_logger, load_config, results_dir, save_figure

LOG = get_logger("task3.density")
MORPH = {0.0: ("Gyroid (w = 0)", "#1f5aa6", "o"), 1.0: ("Diamond (w = 1)", "#c4581a", "s"),
         0.5: ("Blend (w = 0.5)", "#7a3fa0", "^")}
N_PAIR = (32, 64)
# Maskery et al. (2018), Polymer 152, 62-71: PA2200 SLS lattices at rho* = 0.3, compression
# tests, relative moduli E*/E_s = 0.060 (network gyroid), 0.059 (network diamond).
MASKERY = {0.0: 0.060, 1.0: 0.059}


def _one(w: float, rho: float, n_pair: tuple[int, int]) -> dict:
    cfg = load_config()
    E = cfg.material.youngs_modulus
    a = extrapolated_elasticity_tpms(TPMSParams(w=w, rho=rho), n_pair)
    C = a.C_eff / E
    vrh = voigt_reuss_hill(C)
    emin, emax = youngs_extremes(C)
    v = np.array([1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3])
    E111 = 1.0 / (v @ np.linalg.inv(C) @ v)
    c11 = np.mean(np.diag(C)[:3])
    c12 = np.mean([C[0, 1], C[0, 2], C[1, 2]])
    c44 = np.mean(np.diag(C)[3:])
    row = {
        "w": w, "rho": rho, "n1": n_pair[0], "n2": n_pair[1],
        "solid_fraction": a.fine.solid_fraction, "correction": a.correction,
        "E_axial_raw_n2": a.fine.E_axial / E, "E_axial_raw_n1": a.coarse.E_axial / E,
        "E_axial": a.E_axial / E, "E_x": a.youngs_moduli[0] / E, "E_z": a.youngs_moduli[2] / E,
        "E_111": E111, "E_min": emin, "E_max": emax, "E_hill": vrh["E_H"], "K_hill": vrh["K_H"],
        "G_hill": vrh["G_H"], "A_U": vrh["A_U"], "C11": c11, "C12": c12, "C44": c44,
        "zener": 2 * c44 / (c11 - c12), "cubic_dev": cubic_deviation(C),
        "time_s": a.wall_time, "iters_n2": max(a.fine.iterations),
    }
    for name, loc in a.fine.localization.items():
        for k in ("max", "p99", "p999", "mean"):
            row[f"loc_{name}_{k}"] = getattr(loc, k)
    return row


def main(quick: bool = False) -> None:
    rhos = np.round(np.arange(0.15, 0.551, 0.05), 3) if not quick else np.array([0.2, 0.35, 0.5])
    n_pair = N_PAIR if not quick else (12, 24)
    jobs = list(itertools.product(MORPH, rhos))
    rows = Parallel(n_jobs=2, verbose=5)(delayed(_one)(w, r, n_pair) for w, r in jobs)
    df = pd.DataFrame(rows).sort_values(["w", "rho"])
    out = results_dir()
    df.to_csv(out / "task3_stiffness_vs_density.csv", index=False)

    fits = []
    for w, g in df.groupby("w"):
        m = (g.rho >= 0.2 - 1e-9) & (g.rho <= 0.5 + 1e-9)
        for q in ("E_axial", "E_hill", "G_hill", "K_hill", "C44"):
            sl, ic = np.polyfit(np.log(g.rho[m]), np.log(g[q][m]), 1)
            fits.append({"w": w, "quantity": q, "C": float(np.exp(ic)), "m": float(sl),
                         "rho_range": "0.20-0.50"})
            # bending-dominated scaling with m fixed at 2 (Gibson-Ashby), for comparison with Maskery
            if q == "E_axial":
                c2 = float(np.exp(np.mean(np.log(g[q][m]) - 2 * np.log(g.rho[m]))))
                fits.append({"w": w, "quantity": "E_axial (m fixed = 2)", "C": c2, "m": 2.0,
                             "rho_range": "0.20-0.50"})
    fdf = pd.DataFrame(fits)
    fdf.to_csv(out / "task3_powerlaw_fits.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 40, "display.precision", 4):
        print(fdf)
        print(df[["w", "rho", "solid_fraction", "E_axial", "E_min", "E_max", "E_hill", "zener", "A_U",
                  "cubic_dev", "loc_uniaxial_z_p99", "loc_uniaxial_z_max", "loc_shear_xz_p99",
                  "loc_shear_xz_max", "correction", "time_s"]])

    # ---- figure 1: Young's modulus ------------------------------------------------
    cfg = load_config()
    phi = np.linspace(0.12, 0.58, 200)
    hs = hashin_shtrikman_porous(phi, 1.0, cfg.material.poisson_ratio)
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.plot(phi, hs["E"], color="0.45", lw=1.0, label="Hashin-Shtrikman upper bound")
    for w, (lab, col, mk) in MORPH.items():
        g = df[df.w == w]
        f = fdf[(fdf.w == w) & (fdf.quantity == "E_axial")].iloc[0]
        if w == 0.5:
            ax.fill_between(g.rho, g.E_min, g.E_max, color=col, alpha=0.15, lw=0,
                            label="blend: min-max directional E")
        ax.plot(g.rho, g.E_axial, "-", color=col, marker=mk, ms=5,
                label=f"{lab}: {f.C:.2f} ρ*^{f.m:.2f}" if w != 0.5 else f"{lab} (along x, y, z)")
        if w in MASKERY:
            ax.plot([0.293 if w == 0 else 0.307], [MASKERY[w]], marker=mk, mfc="white", mec=col, ms=9,
                    ls="none",
                    label=f"Maskery et al. 2018 exp., {lab.split()[0].lower()}")
    rr = np.array([0.15, 0.55])
    ax.plot(rr, 0.75 * rr**2, "k--", lw=0.8, label="slope 2 (bending, Khaderi et al. 2014)")
    ax.plot(rr, 0.5 * rr**1, "k:", lw=0.8, label="slope 1 guide (stretching-dominated)")
    ax.axvspan(0.2, 0.5, color="#2a9d8f", alpha=0.06, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.14, 0.58)
    ax.set_ylim(3e-3, 0.5)
    ax.set_xticks([0.15, 0.2, 0.3, 0.4, 0.5])
    ax.set_xticklabels(["0.15", "0.2", "0.3", "0.4", "0.5"])
    ax.set_xlabel(r"relative density $\rho^*$")
    ax.set_ylabel(r"$E^*/E_s$  (mean axial Young's modulus)")
    ax.set_title(f"Network TPMS stiffness, voxel FEA, Richardson R{n_pair}", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "task3_youngs_vs_density")

    # ---- figure 2: anisotropy ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for w, (lab, col, mk) in MORPH.items():
        g = df[df.w == w]
        if w != 0.5:
            axes[0].plot(g.rho, g.zener, "-", color=col, marker=mk, ms=5, label=lab)
        axes[1].plot(g.rho, g.A_U, "-", color=col, marker=mk, ms=5, label=lab)
    axes[0].axhline(1.0, color="k", lw=0.8)
    axes[0].set_ylabel(r"Zener ratio $A = 2C_{44}/(C_{11}-C_{12})$")
    axes[0].set_title("(a) Cubic anisotropy (A = 1: isotropic)", fontsize=10)
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"universal anisotropy index $A^U$")
    axes[1].set_title("(b) $A^U$ (any symmetry; blend is trigonal, not cubic)", fontsize=10)
    for ax in axes:
        ax.set_xlabel(r"relative density $\rho^*$")
        ax.axvspan(0.2, 0.5, color="#2a9d8f", alpha=0.06, lw=0)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "task3_anisotropy_vs_density")

    # ---- figure 3: localization ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=False)
    for ax, case, title in ((axes[0], "uniaxial_z", r"(a) uniaxial $\Sigma_{zz}$"),
                            (axes[1], "shear_xz", r"(b) shear $\Sigma_{xz}$")):
        for w, (lab, col, mk) in MORPH.items():
            g = df[df.w == w]
            ax.plot(g.rho, g[f"loc_{case}_p99"], "-", color=col, marker=mk, ms=5, label=f"{lab}: 99th pct")
            ax.plot(g.rho, g[f"loc_{case}_max"], ":", color=col, marker=mk, ms=4, mfc="white",
                    label=f"{lab}: max")
            ax.plot(g.rho, g[f"loc_{case}_mean"], "--", color=col, lw=0.8, alpha=0.7)
        ax.plot(rhos, 1 / rhos, "k-", lw=0.7, label=r"$1/\rho^*$ (uniform stress in solid)")
        ax.set_yscale("log")
        ax.set_xlabel(r"relative density $\rho^*$")
        ax.set_ylabel(r"localization factor $\sigma_{vm}/\Sigma_{vm}$")
        ax.set_title(title + f", n = {n_pair[1]} (dashed: solid-volume mean)", fontsize=10)
        ax.axvspan(0.2, 0.5, color="#2a9d8f", alpha=0.06, lw=0)
        ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    save_figure(fig, "task3_localization_vs_density")
    LOG.info("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
