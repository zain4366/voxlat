"""Task 8: literature closures (Forchheimer C_F, interstitial Nu) - tables and figures.

Outputs
  results/task8_literature_table.csv     every correlation considered (source, TPMS, ranges, formula, access)
  results/task8_operating_range.csv      jacket operating box: Re_Dh, Re_K, C_F, Fo, Nu, h_sf over
                                         w in {0, 0.5, 1}, rho* = 0.20-0.50, L = 2/4/6 mm, 1-5 L/min
  results/task8_nusselt_comparison.csv   Nu_Dh(Re) at Pr = 20.7 for the candidate correlations
  results/figures/task8_forchheimer.png  C_F vs porosity (sources), literature K vs our Task 4 K,
                                         inertial share of the pressure drop vs flow rate
  results/figures/task8_nusselt.png      Nu_Dh vs Re_Dh at the coolant Pr: recommended closure with its
                                         Pr-exponent band, the other variants and packed-bed references

Uses our K and a_sf from results/task4_permeability_vs_porosity.csv (Task 4, R(32, 64)), the reference
config (coolant, gap, axial length) and voxlat.closures.empirical. Runs in a few seconds.

  python scripts/task8_literature_closures.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from voxlat.closures.empirical import (  # noqa: E402
    GAJETTI2025_TABLE2,
    RATHORE2023_CF,
    CorrelationRangeWarning,
    equivalent_particle_diameter,
    forchheimer_coefficient,
    forchheimer_number,
    friction_factor_re,
    gajetti2025_forchheimer,
    gajetti2025_permeability,
    hydraulic_diameter,
    literature_table,
    nusselt_gnielinski_packed_bed,
    nusselt_interstitial,
    nusselt_wakao_kaguei,
    reynolds_hydraulic,
    reynolds_permeability,
    savoldi2026_forchheimer,
    savoldi2026_permeability,
)
from voxlat.utils import load_config, results_dir, save_figure  # noqa: E402

# reference palette (dataviz skill, light mode): categorical slots 1-3, text inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
BLUE_SEQ = ["#9ec5f4", "#5a9be6", "#1f5fae"]  # light -> dark (L = 2, 4, 6 mm)
RANGE_FILL = "#ecebe6"


def load_task4() -> pd.DataFrame:
    path = results_dir() / "task4_permeability_vs_porosity.csv"
    df = pd.read_csv(path)
    return df[(df.rho >= 0.199) & (df.rho <= 0.501)].reset_index(drop=True)


def operating_range(cfg, t4: pd.DataFrame) -> pd.DataFrame:
    """Jacket operating box. Mean superficial velocity of the two parallel half-annulus paths:
    U_s = Q / (2 h L_ax)."""
    h, l_ax = cfg.jacket.lattice_gap, cfg.jacket.axial_length
    c = cfg.coolant
    rows = []
    for _, r in t4.iterrows():
        for L in (2e-3, 4e-3, 6e-3):
            for q in cfg.operating.flow_rate_sweep:
                u_s = q / (2 * h * l_ax)
                phi = float(r.porosity)
                a_sf = float(r.a_sf_L) / L
                K = float(r.K_mean) * L**2
                re = float(reynolds_hydraulic(u_s, a_sf, c.kinematic_viscosity))
                re_k = float(reynolds_permeability(u_s, K, c.kinematic_viscosity))
                cf = float(forchheimer_coefficient(phi, float(r.w), on_extrapolation="ignore"))
                cf_hi = float(forchheimer_coefficient(phi, float(r.w), source="gajetti2025_table", on_extrapolation="ignore"))
                dh = float(hydraulic_diameter(phi, a_sf))
                fre = float(friction_factor_re(re, phi, dh, K, cf))
                nu1 = float(nusselt_interstitial(re, phi, fre, c.prandtl, variant=1, w=float(r.w), on_extrapolation="ignore"))
                nu2 = float(nusselt_interstitial(re, phi, fre, c.prandtl, variant=2, w=float(r.w), on_extrapolation="ignore"))
                fo = float(forchheimer_number(cf, re_k))
                rows.append(
                    dict(
                        w=float(r.w), rho=float(r.rho), porosity=phi, L_mm=L * 1e3, flow_lpm=q * 6e4,
                        U_s=u_s, Re_Dh=re, Re_K=re_k, C_F=cf, C_F_table=cf_hi, Fo=fo,
                        inertial_share=fo / (1 + fo), dp_dx=c.dynamic_viscosity * u_s / K * (1 + fo),
                        Nu_v1=nu1, Nu_v2=nu2, h_sf=nu1 * c.conductivity / dh, h_sf_a_sf=nu1 * c.conductivity / dh * a_sf,
                    )
                )
    return pd.DataFrame(rows)


def nusselt_curves(cfg, t4: pd.DataFrame, re: np.ndarray) -> pd.DataFrame:
    pr = cfg.coolant.prandtl
    rows = []
    for (w, rho) in ((0.0, 0.35), (0.0, 0.2), (1.0, 0.35), (1.0, 0.2), (0.0, 0.5), (1.0, 0.5)):
        r = t4[(np.isclose(t4.w, w)) & (np.isclose(t4.rho, rho))].iloc[0]
        phi, asl, kl = float(r.porosity), float(r.a_sf_L), float(r.K_mean)
        dh, dp = 4 * phi / asl, 6 * (1 - phi) / asl  # in units of L
        cf = float(forchheimer_coefficient(phi, w, on_extrapolation="ignore"))
        fre = friction_factor_re(re, phi, dh, kl, cf)
        out = dict(w=w, rho=rho, porosity=phi)
        for v in (1, 2, 3):
            out[f"v{v}"] = nusselt_interstitial(re, phi, fre, pr, variant=v, w=w, on_extrapolation="ignore")
        out["v1_pr04"] = nusselt_interstitial(re, phi, fre, pr, variant=1, pr_exponent=0.4, w=w, on_extrapolation="ignore")
        out["gnielinski"] = nusselt_gnielinski_packed_bed(re * dp / dh, pr, phi, on_extrapolation="ignore") * dh / dp
        out["wakao_kaguei"] = nusselt_wakao_kaguei(re * dp / dh * phi, pr, on_extrapolation="ignore") * dh / dp
        for i, x in enumerate(re):
            rows.append({**{k: v for k, v in out.items() if np.ndim(v) == 0}, "Re_Dh": x,
                         **{k: float(v[i]) for k, v in out.items() if np.ndim(v) == 1}})
    return pd.DataFrame(rows)


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)


def figure_forchheimer(cfg, t4: pd.DataFrame, ops: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), constrained_layout=True)
    # (a) C_F vs porosity
    ax = axes[0]
    phi_in = np.linspace(0.3, 0.6, 50)
    phi_out = np.linspace(0.6, 0.8, 30)
    for tpms, col in (("gyroid", BLUE), ("diamond", ORANGE)):
        ax.plot(phi_in, gajetti2025_forchheimer(phi_in, tpms), color=col, lw=2, label=f"{tpms.capitalize()}: Gajetti 2025 fit")
        ax.plot(phi_out, gajetti2025_forchheimer(phi_out, tpms), color=col, lw=2, ls=(0, (4, 2)))
        t = GAJETTI2025_TABLE2[tpms]
        ax.plot(t[:, 0], t[:, 2], "o", ms=6, color=col, mec="white", mew=1.0, zorder=5)
        p, _, cf = RATHORE2023_CF[tpms]
        ax.plot([p], [cf], "^", ms=7, color=col, mec="white", mew=1.0, zorder=5)
    ax.plot([0.6, 0.8], [0.19, 0.19], color=ORANGE, lw=1.2, ls=":", label="Diamond: Table 2 held (plateau)")
    phi_s = np.linspace(0.3, 0.8, 60)
    ax.plot(phi_s, savoldi2026_forchheimer(phi_s), color=INK2, lw=1.2, ls="-.", label="Gyroid: Savoldi 2026 fit")
    ax.axvspan(0.5, 0.8, color=RANGE_FILL, zorder=0)
    ax.text(0.65, 1.12, "jacket\n(rho* 0.2-0.5)", ha="center", va="top", fontsize=7.5, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(0.07, 1.3)
    ax.set_xlabel("porosity phi", fontsize=9, color=INK)
    ax.set_ylabel("Forchheimer coefficient C_F", fontsize=9, color=INK)
    ax.set_title("(a) inertial drag (dashed = extrapolated)", fontsize=9.5, loc="left", color=INK)
    h, lab = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D

    h += [Line2D([], [], ls="", marker="o", color=MUTED, mec="white"), Line2D([], [], ls="", marker="^", color=MUTED, mec="white")]
    lab += ["Gajetti 2025 Table 2 (CFD)", "Rathore 2023 (4-cell channel)"]
    ax.legend(h, lab, fontsize=6.8, frameon=False, loc="lower left")
    _style(ax)

    # (b) our Task 4 K vs literature permeability fits
    ax = axes[1]
    for w, tpms, col in ((0.0, "gyroid", BLUE), (1.0, "diamond", ORANGE)):
        d = t4[np.isclose(t4.w, w)].sort_values("porosity")
        ax.plot(d.porosity, d.K_mean, "o", ms=6, color=col, mec="white", mew=1.0, zorder=5, label=f"{tpms.capitalize()}: VoxLat Task 4")
        ax.plot(phi_in, gajetti2025_permeability(phi_in, tpms), color=col, lw=2)
        ax.plot(phi_out, gajetti2025_permeability(phi_out, tpms), color=col, lw=2, ls=(0, (4, 2)))
    ax.plot(phi_s, savoldi2026_permeability(phi_s), color=INK2, lw=1.2, ls="-.", label="Gyroid: Savoldi 2026 fit")
    ax.axvspan(0.5, 0.8, color=RANGE_FILL, zorder=0)
    ax.set_yscale("log")
    ax.set_xlabel("porosity phi", fontsize=9, color=INK)
    ax.set_ylabel("K / L^2", fontsize=9, color=INK)
    ax.set_title("(b) our K vs published CFD fits (lines)", fontsize=9.5, loc="left", color=INK)
    ax.legend(fontsize=6.8, frameon=False, loc="upper left")
    _style(ax)

    # (c) inertial share of the pressure drop vs flow rate (gyroid, rho* = 0.35)
    ax = axes[2]
    for L, col in zip((2.0, 4.0, 6.0), BLUE_SEQ):
        d = ops[(ops.w == 0) & np.isclose(ops.rho, 0.35) & np.isclose(ops.L_mm, L)].sort_values("flow_lpm")
        ax.plot(d.flow_lpm, 100 * d.inertial_share, "-o", color=col, lw=2, ms=5, mec="white", label=f"L = {L:g} mm")
        dd = ops[(ops.w == 1) & np.isclose(ops.rho, 0.35) & np.isclose(ops.L_mm, L)].sort_values("flow_lpm")
        ax.plot(dd.flow_lpm, 100 * dd.inertial_share, ls=(0, (4, 2)), color=col, lw=1.4)
    ax.set_ylim(0, 100)
    ax.set_xlabel("coolant flow [L/min]", fontsize=9, color=INK)
    ax.set_ylabel("inertial share of dp, Fo/(1+Fo) [%]", fontsize=9, color=INK)
    ax.set_title("(c) rho* = 0.35: gyroid (solid), diamond (dashed)", fontsize=9.5, loc="left", color=INK)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    _style(ax)
    return save_figure(fig, "task8_forchheimer")


def figure_nusselt(cfg, curves: pd.DataFrame, ops: pd.DataFrame) -> Path:
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

    cases = ((0.0, 0.35, "Gyroid, rho* = 0.35 (phi = 0.65)"), (0.0, 0.2, "Gyroid, rho* = 0.20 (phi = 0.80)"),
             (1.0, 0.35, "Diamond, rho* = 0.35 (phi = 0.65)"))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.9, bottom=0.3, wspace=0.08)
    re_lo, re_hi = ops.Re_Dh.min(), ops.Re_Dh.max()
    for ax, (w, rho, title) in zip(axes, cases):
        d = curves[np.isclose(curves.w, w) & np.isclose(curves.rho, rho)]
        re = d.Re_Dh.to_numpy()
        ax.axvspan(20, 100, color=RANGE_FILL, zorder=0)
        ax.fill_between(re, d.v1, d.v1_pr04, color=BLUE, alpha=0.18, lw=0, label="variant 1 with Pr^0.4 (band)")
        ax.plot(re, d.v1, color=BLUE, lw=2.2, label="Savoldi 2026 variant 1 x Pr^1/3 (recommended)")
        ax.plot(re, d.v2, color=ORANGE, lw=2, label="variant 2 x Pr^1/3 (low-h bound)")
        ax.plot(re, d.v3, color=AQUA, lw=2, label="variant 3 x Pr^1/3")
        ax.plot(re, d.gnielinski, color=INK2, lw=1.4, ls=(0, (4, 2)), label="Gnielinski packed bed (valid Pr 0.4-1000)")
        ax.plot(re, d.wakao_kaguei, color=INK2, lw=1.2, ls=":", label="Wakao & Kaguei packed bed")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(re.min(), re.max())
        ax.set_ylim(10, 220)
        ax.xaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 200, 500]))
        ax.yaxis.set_major_locator(FixedLocator([10, 20, 30, 50, 100, 200]))
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(ScalarFormatter())
            axis.set_minor_formatter(NullFormatter())
        ax.text(np.sqrt(20 * 100), 0.97, "fitted box\n(Pr = 1, gyroid)", transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=7, color=INK2)
        ax.annotate("", xy=(re_hi, 11.0), xytext=(re_lo, 11.0), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9))
        ax.text(np.sqrt(re_lo * re_hi), 11.5, "jacket operating range", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_title(title, fontsize=9.5, loc="left", color=INK)
        ax.set_xlabel("Re_Dh = U_b D_h / nu", fontsize=9, color=INK)
        _style(ax)
    axes[0].set_ylabel(f"Nu_Dh = h_sf D_h / k_f   (Pr = {cfg.coolant.prandtl:.1f})", fontsize=9, color=INK)
    h, lab = axes[0].get_legend_handles_labels()
    order = [1, 0, 2, 3, 4, 5]
    fig.legend([h[i] for i in order], [lab[i] for i in order], loc="lower center", ncol=3, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    return save_figure(fig, "task8_nusselt")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    warnings.simplefilter("ignore", CorrelationRangeWarning)  # extrapolation is the subject here
    cfg = load_config()
    t4 = load_task4()
    out = results_dir()

    table = literature_table()
    table.to_csv(out / "task8_literature_table.csv", index=False)

    ops = operating_range(cfg, t4)
    ops.to_csv(out / "task8_operating_range.csv", index=False)

    re = np.geomspace(10, 700, 60)
    curves = nusselt_curves(cfg, t4, re)
    curves.to_csv(out / "task8_nusselt_comparison.csv", index=False)

    print(f"coolant Pr = {cfg.coolant.prandtl:.2f}")
    print("Operating box (w in {0, .5, 1}, rho* 0.2-0.5, L 2-6 mm, 1-5 L/min):")
    for col in ("Re_Dh", "Re_K", "C_F", "Fo", "inertial_share", "Nu_v1", "h_sf", "h_sf_a_sf"):
        print(f"  {col:15s} {ops[col].min():10.4g} ... {ops[col].max():10.4g}   (median {ops[col].median():.4g})")
    nom = ops[np.isclose(ops.flow_lpm, 3.0) & np.isclose(ops.L_mm, 4.0) & np.isclose(ops.rho, 0.35)]
    print("Nominal 3 L/min, L = 4 mm, rho* = 0.35:")
    print(nom[["w", "Re_Dh", "Re_K", "C_F", "Fo", "inertial_share", "Nu_v1", "Nu_v2", "h_sf"]].round(3).to_string(index=False))
    d = ops[(ops.w == 1) & (ops.porosity > 0.6)]
    print(f"Diamond, phi > 0.6: C_F plateau / power law = {np.min(d.C_F_table / d.C_F):.2f} ... {np.max(d.C_F_table / d.C_F):.2f}")
    print(f"Fraction of operating points with Re_Dh > 100: {np.mean(ops.Re_Dh > 100):.0%}; < 20: {np.mean(ops.Re_Dh < 20):.0%}")
    for p in (figure_forchheimer(cfg, t4, ops), figure_nusselt(cfg, curves, ops)):
        print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
