"""Task 2 figure: effective conductivity of gyroid, diamond and a 50/50 blend vs relative density.

k_s, k_f from configs/reference.yaml (AlSi10Mg 130, water-glycol 0.40 W/(m K) -> k_f/k_s = 1/325),
plus the insulating-fluid limit k_f = 0. Production estimator: two-grid Richardson
R(32, 64), p = 1 (see scripts/task2_convergence.py and STATUS.md, Task 2).

Run:  python scripts/task2_keff_vs_density.py [--quick]      (~4 min on 2 cores; --quick ~30 s)
Out:  results/task2_keff_vs_density.csv
      results/figures/task2_keff_vs_density.png
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
from voxlat.homogenization.conduction import (
    extrapolated_conductivity_tpms,
    hashin_shtrikman_bounds,
)
from voxlat.utils import get_logger, load_config, results_dir, save_figure

LOG = get_logger("task2.keff")
MORPH = {0.0: ("Gyroid (w = 0)", "#1f5aa6", "o"), 1.0: ("Diamond (w = 1)", "#c4581a", "s"),
         0.5: ("Blend (w = 0.5)", "#7a3fa0", "^")}
N_PAIR = (32, 64)


def _one(w: float, rho: float, kf_ratio: float, n_pair: tuple[int, int]) -> dict:
    cfg = load_config()
    k_s = cfg.material.conductivity
    r = extrapolated_conductivity_tpms(TPMSParams(w=w, rho=rho), n_pair, k_s, kf_ratio * k_s)
    ev = r.eigenvalues / k_s
    return {
        "w": w, "rho": rho, "kf_ks": kf_ratio, "n1": n_pair[0], "n2": n_pair[1],
        "k_mean": r.mean / k_s, "k_min": ev[0], "k_max": ev[-1],
        "k_mean_raw_n2": r.fine.mean / k_s, "k_mean_raw_n1": r.coarse.mean / k_s,
        "solid_fraction": r.fine.solid_fraction, "correction": r.correction,
        "time_s": r.wall_time,
        **{f"k{a}{b}": r.k_eff[i, j] / k_s for (i, a), (j, b) in
           itertools.combinations_with_replacement(enumerate("xyz"), 2)},
    }


def main(quick: bool = False) -> None:
    cfg = load_config()
    kf_cfg = cfg.coolant.conductivity / cfg.material.conductivity
    rhos = np.round(np.arange(0.15, 0.551, 0.05), 3) if not quick else np.array([0.2, 0.35, 0.5])
    n_pair = N_PAIR if not quick else (16, 32)
    jobs = list(itertools.product(MORPH, rhos, (kf_cfg, 0.0)))
    rows = Parallel(n_jobs=-1, verbose=5)(delayed(_one)(w, r, k, n_pair) for w, r, k in jobs)
    df = pd.DataFrame(rows).sort_values(["kf_ks", "w", "rho"])
    df.to_csv(results_dir() / "task2_keff_vs_density.csv", index=False)

    # power-law fits k/k_s = C rho^m over the design range (insulating limit and config)
    for (kfr, w), g in df.groupby(["kf_ks", "w"]):
        m = (g.rho >= 0.2) & (g.rho <= 0.5)
        sl, ic = np.polyfit(np.log(g.rho[m]), np.log(g.k_mean[m]), 1)
        LOG.info("k_f/k_s=%.2e w=%.1f: k/k_s ~ %.3f rho^%.3f", kfr, w, np.exp(ic), sl)

    phi = np.linspace(0.12, 0.58, 200)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    lo, hi = hashin_shtrikman_bounds(phi, 1.0, kf_cfg)
    ax.fill_between(phi, lo, hi, color="0.92", zorder=0, label="HS bounds (k_f/k_s = 1/325)")
    ax.plot(phi, hi, color="0.45", lw=1.0)
    ax.plot(phi, lo, color="0.45", lw=1.0)
    ax.plot(phi, hashin_shtrikman_bounds(phi, 1.0, 0.0)[1], color="0.45", lw=1.0, ls=":",
            label="HS upper, k_f = 0")
    for w, (lab, col, mk) in MORPH.items():
        for kfr, ls, fill in ((kf_cfg, "-", col), (0.0, ":", "white")):
            g = df[(df.w == w) & (df.kf_ks == kfr)]
            ax.plot(g.rho, g.k_mean, ls, color=col, marker=mk, ms=5, mfc=fill,
                    label=lab if kfr > 0 else None)
            if w == 0.5 and kfr > 0:
                ax.fill_between(g.rho, g.k_min, g.k_max, color=col, alpha=0.15, lw=0,
                                label="blend: min-max eigenvalue")
    ax.axvspan(0.2, 0.5, color="#2a9d8f", alpha=0.06, lw=0)
    ax.set_yscale("log")
    ax.set_ylim(3e-3, 0.5)
    ax.set_xlim(0.12, 0.58)
    ax.set_xlabel(r"relative density $\rho^*$")
    ax.set_ylabel(r"$k_{eff}/k_s$  (mean eigenvalue)")
    ax.set_title("(a) Effective conductivity; solid: k_f/k_s = 1/325, open: k_f = 0", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[1]
    for w, (lab, col, mk) in MORPH.items():
        for kfr, ls, fill in ((kf_cfg, "-", col), (0.0, ":", "white")):
            g = df[(df.w == w) & (df.kf_ks == kfr)]
            _lo, _hi = hashin_shtrikman_bounds(g.solid_fraction.values, 1.0, kfr)
            ax.plot(g.rho, g.k_mean / _hi, ls, color=col, marker=mk, ms=5, mfc=fill,
                    label=lab + (" (config)" if kfr > 0 else " (k_f = 0)"))
            if kfr > 0:
                gr = g.k_mean_raw_n2 / _hi
                ax.plot(g.rho, gr, ls, color=col, lw=0.6, alpha=0.5)
    ax.axvspan(0.2, 0.5, color="#2a9d8f", alpha=0.06, lw=0)
    ax.set_xlabel(r"relative density $\rho^*$")
    ax.set_ylabel(r"$k_{eff}\,/\,k_{HS}^{+}$")
    ax.set_title(f"(b) Fraction of the HS upper bound (thin: raw n = {n_pair[1]})", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_xlim(0.12, 0.58)
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "task2_keff_vs_density")
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 4):
        print(df[["w", "rho", "kf_ks", "k_mean", "k_min", "k_max", "k_mean_raw_n2", "correction", "time_s"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
