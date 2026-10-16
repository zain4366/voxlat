"""Task 10: baselines B1-B3 - tuning, verification, comparison at 1/3/5 L/min, figures.

Selection rule (same for every family): minimum thermal resistance R at 3 L/min subject to
manufacturable, min structural margin >= 1 and pump power <= the pump power of the Task 9
reference design (uniform gyroid rho* = 0.35, L = 4 mm: 0.306 W).

Outputs
  results/task10_b1_grid.csv, task10_b1s_grid.csv, task10_b2_grid.csv, task10_b3_search.csv   all candidates
  results/task10_comparison.csv          selected designs at 1, 3, 5 L/min (+ Task 9 reference)
  results/task10_equal_pump_power.csv    best R and R_active per family at pump-power caps 0.1-2 W
  results/task10_fronts.csv              non-dominated (R, pump power, mass) sets per family at 3 L/min
  results/task10_verification.csv        B1 model checks against exact / hand solutions
  results/task10_sensitivity.csv         B1 / B1s modelling sensitivity
  results/baselines/{B1,B1s,B2,B3}.json  selected designs + metrics (design_from_dict-compatible)
  results/figures/task10_temperature_maps.png, task10_fronts.png, task10_flow_comparison.png

  python scripts/task10_baselines.py            # ~5 min
  python scripts/task10_baselines.py --quick    # coarse grids, short B3 search (~1 min)
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from voxlat.device.baselines import (  # noqa: E402
    ChannelDesign,
    ChannelJacketModel,
    ChannelOptions,
    GradedDensityDesign,
    UniformLatticeDesign,
    b1_candidates,
    channel_hydraulics,
    evaluate,
    pareto_front,
    select_best,
    summarize,
    tune_b1,
    tune_b2,
    tune_b3,
)
from voxlat.device.jacket2d import INLET, OUTLET, JacketModel, JacketOptions  # noqa: E402
from voxlat.utils import load_config, results_dir, run_record, save_figure  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2", "#8c8b87"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COLORS = {"B1": BLUE, "B1s": VIOLET, "B2": ORANGE, "B3": AQUA, "ref": GREY}
LABELS = {"B1": "B1 helical channels (end-ring manifolds)", "B1s": "B1s circumferential channels (lattice slots)",
          "B2": "B2 best uniform gyroid", "B3": "B3 density-graded gyroid", "ref": "Task 9 reference gyroid"}
REF = UniformLatticeDesign(0.35, 4e-3, 0.0)
CAPS = (0.1, 0.25, 0.5, 1.0, 2.0)


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
    """B1 model checks (all exact references)."""
    rows = []
    rcp = cfg.coolant.density * cfg.coolant.specific_heat
    T_in = cfg.operating.inlet_temperature
    Q = cfg.operating.nominal_flow_rate
    d = ChannelDesign(2.0e-3, 0.5e-3, 16, "helical")
    # (1) helical, no sleeve conduction: T_f(z) = T_in + C/(rho c_p Q) sum q dz (discrete upwind), T_w - T_f = q/U
    m = ChannelJacketModel(cfg, ChannelOptions(sleeve_conduction=False))
    out = m.evaluate(d, return_fields=True)
    f, g = out["fields"], m.grid("helical")
    q = f.q_in[0]
    T_exact = T_in + g.circumference / (rcp * Q) * np.cumsum(q * g.dz)
    rows.append(dict(check="helical T_f(z) vs exact marching", model=float(np.max(np.abs(f.T_f - T_exact[None, :]))),
                     reference=0.0, rel_error=float(np.max(np.abs(f.T_f - T_exact[None, :]))) / (T_exact[-1] - T_in)))
    dT = f.T_w - f.T_f
    ref = q[None, :] / out["diagnostics"]["U_eff"]
    rows.append(dict(check="helical T_w - T_f = q/U_eff", model=float(np.max(dT)), reference=float(np.max(ref)),
                     rel_error=float(np.max(np.abs(dT / ref - 1)))))
    # (2) balances, default model, both layouts, 1 and 5 L/min
    mm = ChannelJacketModel(cfg)
    for lay, dd in (("helical", d), ("circumferential", ChannelDesign.from_passes(20, 0.5e-3, 1, "circumferential"))):
        for Qx in (cfg.operating.flow_rate_sweep[0], cfg.operating.flow_rate_sweep[-1]):
            o = mm.evaluate(dd, Qx)
            rows.append(dict(check=f"energy balance ({lay})", flow_lpm=Qx * 6e4, model=o["energy_balance_error"],
                             reference=0.0, rel_error=o["energy_balance_error"]))
            rows.append(dict(check=f"T_out - T_in = Q/(rho c_p V) ({lay})", flow_lpm=Qx * 6e4, model=o["T_out"] - T_in,
                             reference=770.0 / (rcp * Qx), rel_error=(o["T_out"] - T_in) / (770.0 / (rcp * Qx)) - 1))
            rows.append(dict(check=f"mass balance ({lay})", flow_lpm=Qx * 6e4, model=o["mass_balance_error"],
                             reference=0.0, rel_error=o["mass_balance_error"]))
    # (3) circumferential, fixed wall T, no plenum exchange: eps vs exact discrete upwind and eps-NTU
    dc = ChannelDesign.from_passes(20, 0.5e-3, 1, "circumferential")
    for sp in (5e-3, 2.5e-3, 1.25e-3, 0.625e-3):
        mt = ChannelJacketModel(cfg, ChannelOptions(spacing=sp, wall_condition="temperature", wall_temperature=T_in + 10,
                                                    manifold_htc=0.0))
        o = mt.evaluate(dc)
        gg = mt.grid("circumferential")
        n = int(np.sum((gg.kind[:, 0] == 0)) // 2)
        U = o["diagnostics"]["U_eff"]
        ell = math.pi * gg.r_mean - cfg.manifolds.width
        ntu = U * ell * gg.axial_length / (rcp * Q / 2.0)
        eps = (o["T_out"] - T_in) / 10.0
        disc = 1 - (1 + ntu / n) ** (-n)
        rows.append(dict(check="circumferential eps vs discrete upwind", spacing_mm=sp * 1e3, model=eps, reference=disc,
                         rel_error=eps / disc - 1, extra=1 - math.exp(-ntu)))
    # (4) dp by hand for one helical design
    hyd = channel_hydraulics(d, cfg, Q)
    C = 2 * math.pi * cfg.jacket.lattice_mean_radius
    P = d.pitch
    ell = cfg.jacket.axial_length / P * math.hypot(C, P)
    U = Q / d.n_starts / (d.channel_width * cfg.jacket.lattice_gap)
    dp_hand = (hyd["friction"] * ell / hyd["D_h"] + 1.25) * 0.5 * cfg.coolant.density * U**2
    rows.append(dict(check="helical dp vs hand calculation", model=hyd["delta_p"], reference=dp_hand,
                     rel_error=hyd["delta_p"] / dp_hand - 1))
    return pd.DataFrame(rows)


def sensitivity(cfg, b1, b1s):
    cases = [("baseline", {}), ("Nu: straight duct, no curvature", dict(nu_model="straight")),
             ("Nu: developing-flow mean (optimistic)", dict(nu_model="developing_mean")),
             ("outer wall as fin", dict(outer_wall_fin=True)), ("no entrance loss (K_inc = 0)", dict(k_inc=0.0)),
             ("rib root K_t = 1.5", dict(root_stress_factor=1.5)), ("rib root K_t = 4", dict(root_stress_factor=4.0)),
             ("manifold h_m = 1000 W/m2K", dict(manifold_htc=1000.0)), ("manifold h_m = 5000 W/m2K", dict(manifold_htc=5000.0))]
    rows = []
    for fam, d in (("B1", b1), ("B1s", b1s)):
        base = None
        for name, kw in cases:
            o = ChannelJacketModel(cfg, ChannelOptions(**kw)).evaluate(d)
            r = dict(family=fam, case=name, R=o["thermal_resistance"], R_active=o["R_active"], delta_p=o["delta_p"],
                     pump_power=o["pump_power"], min_margin=o["min_structural_margin"], Nu=o["diagnostics"]["nusselt"])
            base = base or r
            r.update(R_change=r["R"] / base["R"] - 1, R_active_change=r["R_active"] / base["R_active"] - 1,
                     dp_change=r["delta_p"] / base["delta_p"] - 1)
            rows.append(r)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
def comparison(cfg, selected):
    rows = []
    for fam, d in selected.items():
        for Q in (cfg.operating.flow_rate_sweep[0], cfg.operating.nominal_flow_rate, cfg.operating.flow_rate_sweep[-1]):
            o = evaluate(d, Q)
            r = summarize(o)
            r.update(family=fam, label=LABELS[fam], design=json.dumps(d.to_dict()))
            rows.append(r)
    df = pd.DataFrame(rows)
    cols = ["family", "flow_lpm", "thermal_resistance", "R_active", "delta_p", "pump_power", "mass",
            "min_structural_margin", "manufacturable", "T_wall_max", "T_out", "reynolds", "U_eff", "label", "design"]
    return df[cols + [c for c in df.columns if c not in cols]]


def equal_pump_power(tables):
    rows = []
    for fam, df in tables.items():
        for cap in CAPS:
            try:
                i = select_best(df, cap)
            except ValueError:
                continue
            rows.append(dict(family=fam, pump_power_cap=cap, R=df.thermal_resistance[i], R_active=df.R_active[i],
                             pump_power=df.pump_power[i], mass=df.mass[i], design=df.design[i]))
    return pd.DataFrame(rows)


def fronts(tables):
    out = []
    for fam, df in tables.items():
        m = pareto_front(df)
        sub = df[m].copy()
        sub["family"] = fam
        out.append(sub[["family", "thermal_resistance", "R_active", "pump_power", "mass", "min_structural_margin", "design"]])
    return pd.concat(out, ignore_index=True)


# -----------------------------------------------------------------------------
def fig_maps(cfg, selected, outs):
    fams = list(selected)
    fig, axes = plt.subplots(len(fams), 1, figsize=(7.4, 2.05 * len(fams) + 0.6), sharex=True, constrained_layout=True)
    allT = np.concatenate([outs[f]["fields"].T_wall.ravel() for f in fams]) - 273.15
    vmin, vmax = float(np.min(allT)), float(np.max(allT))
    for ax, fam in zip(axes, fams):
        o = outs[fam]
        fld = o["fields"]
        g = fld.grid
        th = g.theta_faces_deg()
        T = fld.T_wall - 273.15
        pc = ax.pcolormesh(th, g.z_faces * 1e3, T.T, cmap="inferno", vmin=vmin, vmax=vmax, shading="flat", rasterized=True)
        if np.any(g.kind):
            for k in (INLET, OUTLET):
                cols = np.flatnonzero(np.any(g.kind == k, axis=1))
                tf = g.theta_faces_deg()
                for x in (tf[cols.min()], tf[cols.max() + 1]):
                    ax.axvline(x, color="white", lw=0.6, ls=":")
        k = np.unravel_index(np.argmax(T), T.shape)
        tc = 0.5 * (th[1:] + th[:-1])
        ax.plot(tc[k[0]], g.z_centres[k[1]] * 1e3, marker="o", ms=7, mfc="none", mec="white", mew=1.3)
        d = o["design"]
        if fam == "B1":
            desc = f"w_c {d['channel_width']*1e3:.2f} mm, t_r {d['rib_thickness']*1e3:.2f} mm, {d['n_starts']} starts"
        elif fam == "B1s":
            desc = f"w_c {d['channel_width']*1e3:.2f} mm, t_r {d['rib_thickness']*1e3:.2f} mm"
        elif fam == "B2":
            desc = f"ρ* {d['rho']:.3f}, L {d['L']*1e3:.2f} mm"
        else:
            desc = f"ρ* path {', '.join(f'{v:.2f}' for v in d['rho_path'])}, amp_z {d['amp_z']:+.3f}, L {d['L']*1e3:.2f} mm"
        ax.set_title(f"{LABELS[fam]} — {desc}\nR = {o['thermal_resistance']*1e3:.1f} mK/W (active region "
                     f"{o['R_active']*1e3:.1f}), pump {o['pump_power']:.2f} W, max {np.max(T):.1f} °C",
                     fontsize=8, loc="left", color=INK)
        ax.set_ylabel("z [mm]", fontsize=8, color=INK2)
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(pc, ax=axes, pad=0.01, aspect=30, shrink=0.9)
    cb.set_label("sleeve T_wall, stator side [°C]", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)
    axes[-1].set_xlabel("θ [deg] (lattice slots at 0° and 180°, dotted; B1 has end-ring manifolds, no slots)",
                        fontsize=8, color=INK2)
    axes[-1].set_xticks(np.arange(0, 361, 45))
    axes[-1].set_xlim(-6.0, 360.0)
    fig.suptitle("Baselines at 3 L/min, Q = 770 W — same colour scale (circle = hottest point)", fontsize=10, color=INK)
    return save_figure(fig, "task10_temperature_maps")


def fig_fronts(tables, selected_rows, ref_row):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for ax, key, title in ((axes[0], "thermal_resistance", "(a) R = (T_wall,max − T_in)/Q, incl. slot strips"),
                           (axes[1], "R_active", "(b) R_active: hottest point of the cooled region")):
        style(ax)
        for fam, df in tables.items():
            ok = df.manufacturable.astype(bool) & (df.min_structural_margin >= 1)
            ax.scatter(df.pump_power[ok], df[key][ok] * 1e3, s=6, color=COLORS[fam], alpha=0.18, lw=0)
            m = pareto_front(df, ("thermal_resistance", "pump_power")) if key == "thermal_resistance" else \
                pareto_front(df, ("R_active", "pump_power"))
            fr = df[m].sort_values("pump_power")
            ax.plot(fr.pump_power, fr[key] * 1e3, "-", color=COLORS[fam], lw=2, label=LABELS[fam])
        for fam, r in selected_rows.items():
            ax.plot(r["pump_power"], r[key] * 1e3, marker="o", ms=9, mfc="white", mec=COLORS[fam], mew=2, zorder=5)
        ax.plot(ref_row["pump_power"], ref_row[key] * 1e3, marker="*", ms=12, color=INK, zorder=6,
                label=LABELS["ref"])
        ax.axvline(ref_row["pump_power"], color=INK2, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(5e-3, 5)
        ax.set_xlabel("pump power at 3 L/min [W] (dotted: selection cap)", fontsize=8, color=INK2)
        ax.set_ylabel("thermal resistance [mK/W]", fontsize=8, color=INK2)
        ax.set_title(title, fontsize=9, loc="left", color=INK)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Baseline families at 3 L/min: feasible candidates (dots), R–pump-power fronts (lines), "
                 "selected designs (circles)", fontsize=10, color=INK)
    return save_figure(fig, "task10_fronts")


def fig_flow(comp):
    fams = list(dict.fromkeys(comp.family))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
    flows = sorted(comp.flow_lpm.unique())
    width = 0.8 / len(fams)
    for ax, key, lab, scale, logy in ((axes[0], "thermal_resistance", "R [mK/W]", 1e3, False),
                                      (axes[1], "R_active", "R_active [mK/W]", 1e3, False),
                                      (axes[2], "pump_power", "pump power [W]", 1.0, True)):
        style(ax)
        for k, fam in enumerate(fams):
            sub = comp[comp.family == fam].sort_values("flow_lpm")
            x = np.arange(len(flows)) + (k - (len(fams) - 1) / 2) * width
            ax.bar(x, sub[key] * scale, width=width * 0.92, color=COLORS[fam], label=LABELS[fam] if ax is axes[0] else None)
        ax.set_xticks(np.arange(len(flows)))
        ax.set_xticklabels([f"{f:.0f} L/min" for f in flows])
        ax.set_ylabel(lab, fontsize=8, color=INK2)
        if logy:
            ax.set_yscale("log")
    axes[0].set_title("(a) thermal resistance", fontsize=9, loc="left")
    axes[1].set_title("(b) cooled-region resistance", fontsize=9, loc="left")
    axes[2].set_title("(c) pump power (η_pump = 0.3)", fontsize=9, loc="left")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.12))
    return save_figure(fig, "task10_flow_comparison")


# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t_all = time.perf_counter()
    cfg = load_config()
    res = results_dir()
    (res / "baselines").mkdir(parents=True, exist_ok=True)
    lattice = JacketModel(cfg, JacketOptions(check_bounds="ignore"))
    channel = ChannelJacketModel(cfg)

    ref_out = evaluate(REF, model=lattice)
    cap = float(ref_out["pump_power"])
    print(f"selection cap = pump power of the Task 9 reference = {cap:.4f} W")

    print("verification ...")
    ver = verification(cfg)
    ver.to_csv(res / "task10_verification.csv", index=False)
    print(ver.to_string())

    t0 = time.perf_counter()
    grid_kw = dict(n_pass=(4, 6, 8, 12, 16, 20, 30), n_starts=(1, 4, 8, 16, 24, 32)) if args.quick else {}
    r1 = tune_b1(layout="helical", pump_power_cap=cap, model=channel, **grid_kw)
    r1s = tune_b1(layout="circumferential", pump_power_cap=cap, model=channel,
                  **({"n_pass": grid_kw["n_pass"]} if args.quick else {}))
    print(f"B1 / B1s grids: {len(r1.table)} + {len(r1s.table)} candidates, {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter()
    b2_kw = dict(rho_values=np.arange(0.20, 0.501, 0.05), L_values=np.arange(2e-3, 6.01e-3, 0.5e-3)) if args.quick else {}
    r2 = tune_b2(pump_power_cap=cap, model=lattice, **b2_kw)
    print(f"B2 grid: {len(r2.table)} candidates, {time.perf_counter() - t0:.1f} s; best {r2.best}")
    t0 = time.perf_counter()
    b2d = r2.best
    start = GradedDensityDesign((b2d.rho,) * 4, 0.0, b2d.L, 0.0)
    r3 = tune_b3(L=b2d.L, pump_power_cap=cap, start=start, model=lattice,
                 n_random=16 if args.quick else 64, maxfev=40 if args.quick else 200)
    print(f"B3 search: {len(r3.table)} evaluations, {time.perf_counter() - t0:.1f} s; best {r3.best}")

    for r, name in ((r1, "b1"), (r1s, "b1s"), (r2, "b2"), (r3, "b3")):
        r.table.to_csv(res / f"task10_{name}_grid.csv" if name != "b3" else res / "task10_b3_search.csv", index=False)
    selected = {"B1": r1.best, "B1s": r1s.best, "B2": r2.best, "B3": r3.best}
    tables = {"B1": r1.table, "B1s": r1s.table, "B2": r2.table, "B3": r3.table}

    comp = comparison(cfg, {**selected, "ref": REF})
    comp.to_csv(res / "task10_comparison.csv", index=False)
    epp = equal_pump_power(tables)
    epp.to_csv(res / "task10_equal_pump_power.csv", index=False)
    fr = fronts(tables)
    fr.to_csv(res / "task10_fronts.csv", index=False)
    sens = sensitivity(cfg, r1.best, r1s.best)
    sens.to_csv(res / "task10_sensitivity.csv", index=False)

    outs = {}
    for fam, d in selected.items():
        o = evaluate(d, return_fields=True)
        outs[fam] = o
        rec = {k: v for k, v in o.items() if k not in ("fields",)}
        rec["selection_rule"] = dict(objective="thermal_resistance at 3 L/min", pump_power_cap=cap,
                                     constraints=["manufacturable", "min_structural_margin >= 1"])
        rec["run_record"] = run_record({"family": fam, **d.to_dict()}, resolution=list(o["fields"].grid.shape),
                                       wall_time=o["wall_time"])
        (res / "baselines" / f"{fam}.json").write_text(json.dumps(rec, indent=2, default=lambda x: x.tolist()
                                                                  if hasattr(x, "tolist") else str(x)))
    sel_rows = {fam: summarize(o) for fam, o in outs.items()}
    fig_maps(cfg, selected, outs)
    fig_fronts(tables, sel_rows, summarize(ref_out))
    fig_flow(comp)

    pd.set_option("display.width", 220)
    show = comp[["family", "flow_lpm", "thermal_resistance", "R_active", "delta_p", "pump_power", "mass",
                 "min_structural_margin", "reynolds"]].copy()
    show[["thermal_resistance", "R_active"]] *= 1e3
    print(show.round(3).to_string(index=False))
    print(epp.drop(columns="design").assign(R=lambda x: x.R * 1e3, R_active=lambda x: x.R_active * 1e3).round(3).to_string())
    print(sens.round(4).to_string())
    print(f"done in {time.perf_counter() - t_all:.0f} s")


if __name__ == "__main__":
    main()
