"""Task 4 figure: permeability K/L^2 of gyroid, diamond and a 50/50 blend vs porosity.

Network solids, rho* = 0.15 ... 0.55 (porosity phi = 1 - rho* = 0.45 ... 0.85). Production
estimator: two-grid Richardson R(32, 64), p = 1, on the offset-0 grid (choice and accuracy:
scripts/task4_convergence.py and STATUS.md, Task 4). Gyroid/diamond are cubic -> one solve
(K = k I); the blend is trigonal -> three solves, full tensor (principal values plotted).
Also reported: effective Kozeny constant c_K = phi^3 / (K a_sf^2) with the continuum a_sf
of Task 1 (marching cubes, n = 64), and the hydraulic tortuosity <|u|>/|<u>|.

Run:  python scripts/task4_permeability_vs_porosity.py [--quick]   (~15 min on 2 cores; --quick ~40 s)
Out:  results/task4_permeability_vs_porosity.csv
      results/figures/task4_permeability_vs_porosity.png
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

from voxlat.geometry import TPMSParams, compute_metrics
from voxlat.homogenization.stokes import extrapolated_permeability_tpms, kozeny_constant
from voxlat.utils import get_logger, results_dir, save_figure

LOG = get_logger("task4.K")
MORPH = {0.0: ("Gyroid (w = 0)", "#1f5aa6", "o"), 1.0: ("Diamond (w = 1)", "#c4581a", "s"),
         0.5: ("Blend (w = 0.5)", "#7a3fa0", "^")}
N_PAIR = (32, 64)


def _one(w: float, rho: float, n_pair: tuple[int, int]) -> dict:
    p = TPMSParams(w=w, rho=rho)
    r = extrapolated_permeability_tpms(p, n_pair)
    a_sf = compute_metrics(p, n=64, thickness=False, connectivity=False).a_sf_L
    ev = r.eigenvalues
    phi = r.porosity
    row = {
        "w": w, "rho": rho, "porosity": phi, "n1": n_pair[0], "n2": n_pair[1],
        "K_mean": r.mean, "K_min": ev[0], "K_max": ev[-1],
        "K_mean_raw_n1": r.coarse.mean, "K_mean_raw_n2": r.fine.mean, "correction": r.correction,
        "a_sf_L": a_sf, "kozeny_c": float(kozeny_constant(r.mean, phi, a_sf)),
        "tortuosity": float(np.nanmean(list(r.tortuosity.values()))),
        "interstitial_u_mean": r.mean / phi,  # <u>_f per unit f L^2 / mu, along the force
        "anisotropy": (ev[-1] - ev[0]) / ev.mean(),
        "time_s": r.wall_time, "n_solves": len(r.fine.directions),
    }
    row.update({f"K{a}{b}": r.K[i, j] for (i, a), (j, b) in
                itertools.combinations_with_replacement(enumerate("xyz"), 2)})
    return row


def plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    for w, (lab, col, mk) in MORPH.items():
        g = df[df.w == w].sort_values("porosity")
        ax.semilogy(g.porosity, g.K_mean, "-" + mk, color=col, ms=5, label=lab)
        if w == 0.5:
            ax.fill_between(g.porosity, g.K_min, g.K_max, color=col, alpha=0.18, lw=0,
                            label="blend: principal values")
    ax.set_xlabel(r"porosity $\phi = 1 - \rho^*$")
    ax.set_ylabel(r"$K / L^2$   (mean principal value)")
    ax.set_title("(a) Permeability, R(32, 64)", fontsize=10, loc="left")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    for w, (lab, col, mk) in MORPH.items():
        g = df[df.w == w].sort_values("porosity")
        ax.plot(g.porosity, g.kozeny_c, "-" + mk, color=col, ms=5, label=lab)
    ax.axhline(5.0, color="k", ls=":", lw=1)
    ax.text(0.46, 5.08, "Carman (packed beds) = 5", fontsize=8)
    ax.set_xlabel(r"porosity $\phi$")
    ax.set_ylabel(r"Kozeny constant  $c_K = \phi^3 / (K\, a_{sf}^2)$")
    ax.set_title("(b) Effective Kozeny constant", fontsize=10, loc="left")
    ax.legend(fontsize=8)

    ax = axes[2]
    for w, (lab, col, mk) in MORPH.items():
        g = df[df.w == w].sort_values("porosity")
        ax.plot(g.porosity, g.tortuosity, "-" + mk, color=col, ms=5, label=lab)
    ax.set_xlabel(r"porosity $\phi$")
    ax.set_ylabel(r"hydraulic tortuosity  $\langle|u|\rangle / |\langle u \rangle|$")
    ax.set_title("(c) Tortuosity (fine grid)", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = save_figure(fig, "task4_permeability_vs_porosity")
    LOG.info("figure -> %s", path)


def main(quick: bool = False) -> None:
    rhos = np.round(np.arange(0.15, 0.551, 0.05), 3) if not quick else np.array([0.2, 0.35, 0.5])
    n_pair = N_PAIR if not quick else (16, 32)
    jobs = sorted(itertools.product(MORPH, rhos), key=lambda t: -(3 if t[0] == 0.5 else 1))
    rows = Parallel(n_jobs=-1, verbose=5)(delayed(_one)(w, r, n_pair) for w, r in jobs)
    df = pd.DataFrame(rows).sort_values(["w", "rho"])
    out = results_dir() / "task4_permeability_vs_porosity.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 220, "display.precision", 4):
        print(df[["w", "rho", "porosity", "K_mean", "K_min", "K_max", "correction", "kozeny_c",
                  "tortuosity", "anisotropy", "time_s"]])
    # power-law fits K/L^2 = A phi^m over the design range (rho* = 0.2-0.5)
    for w in (0.0, 1.0):
        g = df[(df.w == w) & (df.rho >= 0.2) & (df.rho <= 0.5)]
        m, lnA = np.polyfit(np.log(g.porosity), np.log(g.K_mean), 1)
        LOG.info("w = %.1f: K/L^2 ~ %.4f phi^%.2f (rho* 0.2-0.5)", w, np.exp(lnA), m)
    plot(df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
