"""Task 0 figure: axial heat-flux profile q''(z) of the reference problem.

Run:  python scripts/task0_heat_flux_profile.py
Out:  results/figures/task0_heat_flux_profile.png (300 dpi) + a printed summary.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from voxlat.utils import get_logger, load_config, save_figure


def main() -> None:
    log = get_logger("task0")
    cfg = load_config()
    j = cfg.jacket
    L = j.axial_length

    z = np.linspace(0.0, L, 401)
    q = cfg.q_wall(z)
    Q_num = integrate.quad(cfg.q_wall, 0.0, L)[0] * 2 * math.pi * j.stator_radius

    log.info("Lattice gap      : r = %.1f ... %.1f mm (h = %.1f mm)",
             j.lattice_inner_radius * 1e3, j.lattice_outer_radius * 1e3, j.lattice_gap * 1e3)
    log.info("Mean heat flux   : %.0f W/m^2", cfg.mean_heat_flux)
    log.info("q''(0) / q''(L/2): %.3f", cfg.q_wall(0.0) / cfg.q_wall(L / 2))
    log.info("Integral check   : Q = %.4f W (config %.1f W)", Q_num, cfg.motor.heat_to_jacket)
    log.info("Coolant Pr       : %.2f", cfg.coolant.prandtl)
    log.info("Total motor loss : %.0f W  (jacket share %.0f %%)",
             cfg.motor.total_loss, 100 * cfg.motor.heat_to_jacket / cfg.motor.total_loss)

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(z * 1e3, q / 1e3, lw=2, color="#1f5aa6", label=r"$q''(z)$")
    ax.axhline(cfg.mean_heat_flux / 1e3, ls="--", lw=1, color="0.4",
               label=rf"$\bar q = Q/(2\pi r_s L)$ = {cfg.mean_heat_flux / 1e3:.1f} kW/m$^2$")
    ax.set_xlabel("Axial position z [mm]")
    ax.set_ylabel(r"Heat flux $q''$ [kW/m$^2$]")
    ax.set_xlim(0, L * 1e3)
    ax.set_ylim(0, None)
    ax.set_title(f"Reference heat-flux profile (Q = {cfg.motor.heat_to_jacket:.0f} W)")
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    ax.grid(alpha=0.3)
    path = save_figure(fig, "task0_heat_flux_profile")
    log.info("Saved %s", path)


if __name__ == "__main__":
    main()
