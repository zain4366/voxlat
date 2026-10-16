"""Task 9: homogenized jacket device model - verification, grid study, reference results, figures.

Outputs
  results/task9_verification.csv          exact-reference checks (dp, eps-NTU, bulk flux solution, balances)
  results/task9_grid_convergence.csv      R, R_lattice, dp vs grid spacing (uniform gyroid, reference problem)
  results/task9_flow_sweep.csv            R, dp, pump power, T_out, margins at 1-5 L/min for uniform G / D / blend
  results/task9_sensitivity.csv           closure / modelling sensitivity of the uniform gyroid at 3 L/min
  results/task9_reference_gyroid.json     full output dict of the reference design (gyroid, rho* 0.35, L 4 mm, 3 L/min)
  results/figures/task9_fields_uniform_gyroid.png   pressure, velocity, coolant and wall temperature maps
  results/figures/task9_verification.png           eps-NTU convergence, dp vs analytic, grid convergence of R
  results/figures/task9_flow_sweep.png             R, dp, pump power vs flow rate (G, D, blend)

  python scripts/task9_jacket_model.py            # ~1-2 min
  python scripts/task9_jacket_model.py --quick    # coarser grid study (~20 s)
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from voxlat.device.jacket2d import (  # noqa: E402
    INLET,
    LATTICE,
    OUTLET,
    JacketModel,
    JacketOptions,
    ntu_reference,
    uniform_flow_reference,
)
from voxlat.utils import load_config, results_dir, save_figure  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
REF = dict(rho=0.35, w=0.0, L=4e-3)


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


# -----------------------------------------------------------------------------
def verification(cfg):
    rows = []
    T_in = cfg.operating.inlet_temperature
    # dp vs the exact two-path Darcy-Forchheimer solution
    m = JacketModel(cfg)
    for Q in cfg.operating.flow_rate_sweep:
        out = m.evaluate(**REF, flow_rate=Q)
        ref = uniform_flow_reference(m, **REF, flow_rate=Q)
        rows.append(dict(check="dp_uniform", flow_lpm=Q * 6e4, model=out["delta_p"], reference=ref["dp"],
                         rel_error=out["delta_p"] / ref["dp"] - 1, extra=ref["dp_darcy"]))
        rows.append(dict(check="mass_balance", flow_lpm=Q * 6e4, model=out["mass_balance_error"], reference=0.0,
                         rel_error=out["mass_balance_error"]))
        rows.append(dict(check="energy_balance", flow_lpm=Q * 6e4, model=out["energy_balance_error"], reference=0.0,
                         rel_error=out["energy_balance_error"]))
    # eps-NTU at fixed wall temperature, several grids
    ntu_rows = []
    for sp in (10e-3, 5e-3, 2.5e-3, 1.25e-3, 0.625e-3):
        mm = JacketModel(cfg, JacketOptions(spacing=sp, wall_condition="temperature", wall_temperature=T_in + 10,
                                            manifold_htc=0.0, fluid_conduction=False, lattice_conduction=False))
        out = mm.evaluate(**REF)
        r = ntu_reference(mm, **REF)
        n = int((mm.grid.kind[:, 0] == LATTICE).sum() // 2)
        eps = (out["T_out"] - T_in) / 10
        disc = 1 - (1 + r["ntu"] / n) ** (-n)
        ntu_rows.append(dict(spacing_mm=sp * 1e3, cells_per_path=n, ntu=r["ntu"], eps_model=eps, eps_discrete=disc,
                             eps_exact=r["effectiveness"]))
        rows.append(dict(check="eps_NTU", flow_lpm=3.0, spacing_mm=sp * 1e3, model=eps, reference=r["effectiveness"],
                         rel_error=eps / r["effectiveness"] - 1, extra=disc / eps - 1))
    # imposed flux, z-uniform: bulk analytic solution mid-path
    cflat = load_config(overrides={"heat_flux": {"amplitude": 0.0}})
    mf = JacketModel(cflat)
    out = mf.evaluate(**REF, return_fields=True)
    f, g = out["fields"], mf.grid
    r = ntu_reference(mf, **REF)
    q_in = cflat.mean_heat_flux * cflat.jacket.stator_radius / g.r_mean
    i = int(np.argmin(np.abs(g.theta_plot_deg() - 90)))
    j = g.n_z // 2
    rows.append(dict(check="bulk_Tw_minus_Tf", model=f.T_w[i, j] - f.T_f[i, j], reference=q_in / r["U_eff"],
                     rel_error=(f.T_w[i, j] - f.T_f[i, j]) / (q_in / r["U_eff"]) - 1))
    rows.append(dict(check="bulk_eta", model=(f.T_s_mean[i, j] - f.T_f[i, j]) / (f.T_w[i, j] - f.T_f[i, j]),
                     reference=r["eta"], rel_error=(f.T_s_mean[i, j] - f.T_f[i, j]) / (f.T_w[i, j] - f.T_f[i, j]) / r["eta"] - 1))
    rcp = cfg.coolant.density * cfg.coolant.specific_heat
    dT = cfg.motor.heat_to_jacket / (rcp * cfg.operating.nominal_flow_rate)
    rows.append(dict(check="T_out_minus_T_in", model=out["T_out"] - T_in, reference=dT, rel_error=(out["T_out"] - T_in) / dT - 1))
    return pd.DataFrame(rows), pd.DataFrame(ntu_rows), (mf, out, r, q_in)


def grid_study(cfg, quick):
    rows = []
    specs = [(5e-3, 2.5e-3), (2.5e-3, 2.5e-3), (1.25e-3, 2.5e-3), (0.625e-3, 2.5e-3)]
    if not quick:
        specs += [(0.3125e-3, 2.5e-3), (2.5e-3, 1.25e-3), (1.25e-3, 1.25e-3), (1.25e-3, 0.625e-3)]
    for sp, spz in specs:
        m = JacketModel(cfg, JacketOptions(spacing=sp, spacing_z=spz))
        m.evaluate(**REF)
        t0 = time.perf_counter()
        out = m.evaluate(**REF)
        rows.append(dict(spacing_s_mm=sp * 1e3, spacing_z_mm=spz * 1e3, n_s=m.grid.n_s, n_z=m.grid.n_z,
                         R=out["thermal_resistance"],
                         R_lattice=(out["T_wall_max_lattice"] - out["T_in"]) / out["heat_input"],
                         T_wall_mean=out["T_wall_mean"], delta_p=out["delta_p"], time_s=time.perf_counter() - t0))
    df = pd.DataFrame(rows)
    s_series = df[df.spacing_z_mm == 2.5].sort_values("spacing_s_mm", ascending=False)
    R = s_series.R.to_numpy()
    p = math.log2((R[-3] - R[-2]) / (R[-2] - R[-1]))
    R_inf = R[-1] + (R[-1] - R[-2]) / (2**p - 1)
    df["observed_order_s"] = p
    df["R_extrapolated"] = R_inf
    df["R_error_vs_extrapolated"] = df.R / R_inf - 1
    return df


def flow_sweep(cfg):
    m = JacketModel(cfg)
    rows = []
    for name, w in (("gyroid", 0.0), ("blend w=0.5", 0.5), ("diamond", 1.0)):
        for Q in cfg.operating.flow_rate_sweep:
            out = m.evaluate(0.35, w, 4e-3, flow_rate=Q)
            d = out["diagnostics"]
            rows.append(dict(design=name, w=w, rho=0.35, L_mm=4.0, flow_lpm=Q * 6e4,
                             R=out["thermal_resistance"], R_lattice=(out["T_wall_max_lattice"] - out["T_in"]) / 770.0,
                             T_wall_max_C=out["T_wall_max"] - 273.15, T_out_C=out["T_out"] - 273.15,
                             delta_p=out["delta_p"], pump_power=out["pump_power"], mass=out["mass"],
                             min_structural_margin=out["min_structural_margin"], lattice_margin=out["lattice_margin"],
                             Re_median=d["reynolds_median"], Fo_max=d["forchheimer_number_range"][1],
                             h_sf_mean=d["h_sf_mean"], U_eff_mean=d["U_eff_mean"], fin_eta=d["fin_efficiency_mean"],
                             fgK=d["finite_gap_factor_K_mean"], picard_iterations=out["picard_iterations"],
                             time_s=out["wall_time"]))
    return pd.DataFrame(rows)


def sensitivity(cfg):
    base = JacketModel(cfg)
    cases = [
        ("baseline", {}),
        ("no finite-gap correction", dict(finite_gap=False)),
        ("Nu variant 2 (low-h bound)", dict(nu_variant=2)),
        ("Pr exponent 0.4", dict(pr_exponent=0.4)),
        ("C_F Table-2 plateau", dict(cf_source="gajetti2025_table")),
        ("isothermal fins (bound)", dict(fin_model="isothermal")),
        ("no exposed-wall convection", dict(wall_exposed_convection=False)),
        ("manifold h_m = 1000 W/m2K", dict(manifold_htc=1000.0)),
        ("manifold h_m = 5000 W/m2K", dict(manifold_htc=5000.0)),
        ("GP surrogate", dict(backend="gp")),
        ("no lattice in-plane conduction", dict(lattice_conduction=False)),
    ]
    rows = []
    b = None
    for name, kw in cases:
        out = base.with_options(**kw).evaluate(**REF)
        row = dict(case=name, R=out["thermal_resistance"],
                   R_lattice=(out["T_wall_max_lattice"] - out["T_in"]) / 770.0, delta_p=out["delta_p"],
                   pump_power=out["pump_power"], lattice_margin=out["lattice_margin"])
        if b is None:
            b = row
        row.update({f"{k}_change": row[k] / b[k] - 1 for k in ("R", "R_lattice", "delta_p")})
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
def slot_shading(ax, g):
    tf = g.theta_faces_deg()
    for k in (INLET, OUTLET):
        cols = np.flatnonzero(np.any(g.kind == k, axis=1))
        ax.axvspan(tf[cols.min()], tf[cols.max() + 1], color="#e8e7e3", zorder=0, lw=0)


def fig_fields(model, out):
    f = out["fields"]
    g = model.grid
    th_f = g.theta_faces_deg()
    zf = g.z_faces * 1e3
    lat = g.lattice
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 8.6), sharex=True, constrained_layout=True)

    def panel(ax, data, cmap, label, mask_slots=True):
        d = np.where(lat, data, np.nan) if mask_slots else data
        slot_shading(ax, g)
        lo, hi = np.nanmin(d), np.nanmax(d)
        if hi - lo < 1e-6 * max(abs(hi), 1e-30):  # uniform field: show the value, not round-off
            lo, hi = 0.9 * lo, 1.1 * hi
        pc = ax.pcolormesh(th_f, zf, d.T, cmap=cmap, shading="flat", rasterized=True, vmin=lo, vmax=hi)
        cb = fig.colorbar(pc, ax=ax, pad=0.01, aspect=12)
        cb.set_label(label, fontsize=8, color=INK2)
        cb.ax.tick_params(labelsize=7, colors=INK2)
        cb.outline.set_visible(False)
        ax.set_ylabel("z [mm]", fontsize=8, color=INK2)
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
        return pc

    p_kpa = f.pressure / 1e3
    panel(axes[0], p_kpa, "Blues", "pressure [kPa]")
    axes[0].set_title(f"(a) pressure, Δp = {out['delta_p']:.0f} Pa", fontsize=9, loc="left", color=INK)
    panel(axes[1], f.speed * 1e3, "Blues", "|U| superficial [mm/s]")
    th = g.theta_plot_deg()
    zc = g.z_centres * 1e3
    step_s = max(1, g.n_s // 36)
    step_z = max(1, g.n_z // 6)
    I = np.arange(0, g.n_s, step_s)
    J = np.arange(step_z // 2, g.n_z, step_z)
    U = np.where(lat, f.U_s, np.nan)[np.ix_(I, J)]
    V = np.where(lat, f.U_z, np.nan)[np.ix_(I, J)]
    axes[1].quiver(th[I][:, None] * np.ones((1, len(J))), np.ones((len(I), 1)) * zc[J][None], U, V,
                   color=INK, width=0.0022, scale=1.6, scale_units="inches", angles="uv", alpha=0.75)
    axes[1].set_title("(b) superficial velocity |U|; arrows show direction (two half-annulus paths)",
                      fontsize=9, loc="left", color=INK)
    panel(axes[2], f.T_f - 273.15, "Oranges", "coolant T_f [°C]")
    axes[2].set_title(f"(c) coolant temperature, outlet mixed {out['T_out'] - 273.15:.2f} °C", fontsize=9,
                      loc="left", color=INK)
    panel(axes[3], f.T_wall - 273.15, "Oranges", "sleeve T_wall [°C]", mask_slots=False)
    k = np.unravel_index(np.argmax(f.T_wall), f.T_wall.shape)
    axes[3].plot(th[k[0]], zc[k[1]], marker="o", ms=8, mfc="none", mec=INK, mew=1.4)
    axes[3].annotate(f"max {f.T_wall[k] - 273.15:.1f} °C\n(unfinned sleeve under the outlet slot)",
                     (th[k[0]], zc[k[1]]), xytext=(14, 18), textcoords="offset points", fontsize=7.5, color=INK,
                     ha="left", va="bottom", arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    axes[3].set_title(f"(d) sleeve temperature (stator side), R = {out['thermal_resistance'] * 1e3:.1f} mK/W; "
                      "incl. slots", fontsize=9, loc="left", color=INK)
    axes[3].set_xlabel("θ [deg] (inlet slot at 0°, outlet slot at 180°; grey bands = manifold slots)", fontsize=8,
                       color=INK2)
    axes[3].set_xticks(np.arange(0, 361, 45))
    fig.suptitle("Uniform gyroid jacket: ρ* = 0.35, L = 4 mm, 3 L/min, Q = 770 W", fontsize=10, color=INK)
    return save_figure(fig, "task9_fields_uniform_gyroid")


def fig_verification(ver, ntu, grid, cfg):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), constrained_layout=True)
    # (a) eps-NTU
    ax = axes[0]
    style(ax)
    h = ntu.spacing_mm.to_numpy()
    e_model = (ntu.eps_exact - ntu.eps_model) / ntu.eps_exact
    ax.loglog(h, e_model, "o-", color=BLUE, lw=2, ms=8, label="2-D model")
    ax.loglog(h, (ntu.eps_exact - ntu.eps_discrete) / ntu.eps_exact, "x", color=INK, ms=9, mew=1.4,
              label="exact discrete upwind")
    ax.loglog(h, e_model.iloc[0] * h / h[0], ":", color=INK2, lw=1, label="slope 1")
    ax.set_xlabel("grid spacing [mm]", fontsize=8, color=INK2)
    ax.set_ylabel("relative error of ε vs ε-NTU", fontsize=8, color=INK2)
    ax.set_title(f"(a) fixed wall temperature, NTU = {ntu.ntu.iloc[0]:.2f}", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)
    # (b) dp vs analytic
    ax = axes[1]
    style(ax)
    d = ver[ver.check == "dp_uniform"]
    ax.plot(d.flow_lpm, d.reference / 1e3, "-", color=BLUE, lw=2, label="Darcy-Forchheimer (analytic)")
    ax.plot(d.flow_lpm, d.extra / 1e3, "--", color=ORANGE, lw=2, label="Darcy term only")
    ax.plot(d.flow_lpm, d.model / 1e3, "o", color=INK, ms=8, mfc="none", mew=1.4, label="2-D model")
    ax.set_xlabel("flow rate [L/min]", fontsize=8, color=INK2)
    ax.set_ylabel("Δp [kPa]", fontsize=8, color=INK2)
    ax.set_title(f"(b) uniform gyroid: max |error| {d.rel_error.abs().max():.0e}", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)
    # (c) grid convergence of R
    ax = axes[2]
    style(ax)
    gs = grid[grid.spacing_z_mm == 2.5].sort_values("spacing_s_mm")
    err = np.abs(gs.R_error_vs_extrapolated.to_numpy())
    hs = gs.spacing_s_mm.to_numpy()
    ax.loglog(hs, err * 100, "o-", color=BLUE, lw=2, ms=8, label="R (overall max)")
    ax.loglog(hs, err[-1] * 100 * (hs / hs[-1]) ** 2, ":", color=INK2, lw=1, label="slope 2")
    ax.axvline(cfg_default_spacing() * 1e3, color=ORANGE, lw=1.5, ls="--")
    ax.text(cfg_default_spacing() * 1e3 * 1.08, err.max() * 100 * 0.8, "default", color=ORANGE, fontsize=7.5)
    ax.set_xlabel("grid spacing in s [mm] (z: 2.5 mm)", fontsize=8, color=INK2)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(hs))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("|R - R_extrapolated| / R [%]", fontsize=8, color=INK2)
    ax.set_title(f"(c) grid convergence, order {gs.observed_order_s.iloc[0]:.2f}", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)
    return save_figure(fig, "task9_verification")


def cfg_default_spacing():
    return JacketOptions().spacing


def fig_sweep(sw):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
    cols = {"gyroid": BLUE, "blend w=0.5": ORANGE, "diamond": AQUA}
    for ax, (key, lab, scale) in zip(axes, (("R", "thermal resistance R [mK/W]", 1e3),
                                            ("delta_p", "Δp [kPa]", 1e-3), ("pump_power", "pump power [W]", 1.0))):
        style(ax)
        for name, d in sw.groupby("design", sort=False):
            ax.plot(d.flow_lpm, d[key] * scale, "o-", color=cols[name], lw=2, ms=6, label=name)
        ax.set_xlabel("flow rate [L/min]", fontsize=8, color=INK2)
        ax.set_ylabel(lab, fontsize=8, color=INK2)
        ax.set_xlim(0.8, 5.2)
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_title("(a) R = (T_wall,max - T_in)/Q", fontsize=9, loc="left")
    axes[1].set_title("(b) pressure drop", fontsize=9, loc="left")
    axes[2].set_title("(c) pump power (η_pump = 0.3)", fontsize=9, loc="left")
    fig.suptitle("Uniform lattices, ρ* = 0.35, L = 4 mm", fontsize=10)
    return save_figure(fig, "task9_flow_sweep")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    cfg = load_config()
    res = results_dir()
    t0 = time.time()
    ver, ntu, _ = verification(cfg)
    ver.to_csv(res / "task9_verification.csv", index=False)
    print(ver.to_string())
    grid = grid_study(cfg, args.quick)
    grid.to_csv(res / "task9_grid_convergence.csv", index=False)
    print(grid.to_string())
    sw = flow_sweep(cfg)
    sw.to_csv(res / "task9_flow_sweep.csv", index=False)
    print(sw[["design", "flow_lpm", "R", "R_lattice", "delta_p", "pump_power", "min_structural_margin", "time_s"]].to_string())
    sens = sensitivity(cfg)
    sens.to_csv(res / "task9_sensitivity.csv", index=False)
    print(sens.to_string())
    model = JacketModel(cfg)
    out = model.evaluate(**REF, return_fields=True)
    js = {k: v for k, v in out.items() if k != "fields"}
    (res / "task9_reference_gyroid.json").write_text(json.dumps(js, indent=2, default=float))
    print(fig_fields(model, out))
    print(fig_verification(ver, ntu, grid, cfg))
    print(fig_sweep(sw))
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
