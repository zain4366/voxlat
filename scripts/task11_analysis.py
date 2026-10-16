"""Task 11 analysis: hypervolume curves (RQ3), Pareto fronts vs baselines B1-B3 (RQ2),
representative designs (knee, min-R, min-mass) with field maps, design JSONs, tables.

Reads results/task11/<layout>/*.csv.gz (scripts/task11_optimize.py) and the Task 10 baselines
(results/task10_fronts.csv, results/baselines/*.json). Writes
    results/figures/task11_hypervolume.png, task11_pareto_fronts.png, task11_designs_<layout>.png
    results/task11_hv_curves.csv, task11_hv_summary.csv, task11_runs.csv, task11_front_<layout>.csv,
    task11_equal_pump_power.csv, task11_representative.csv
    results/designs/task11_<layout>_<label>.json
Runtime ~3-5 min (re-evaluates 6 designs over the flow sweep and checks their critical cells at n = 48).

    python scripts/task11_uniform_references.py   # once, ~2-6 min (uniform G/D references, both layouts)
    python scripts/task11_analysis.py [--root results/task11] [--no-designs]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from voxlat.optimize import ObjectiveSpace, pareto_mask  # noqa: E402
from voxlat.optimize.study import (  # noqa: E402
    METHODS,
    combined_front,
    design_record,
    equal_pump_power,
    evaluate_x,
    hv_curves,
    load_runs,
    representative_designs,
    rescreen_cell_size,
    x_of,
)
from voxlat.utils import load_config, results_dir, save_figure  # noqa: E402

warnings.simplefilter("ignore")

# reference palette (dataviz skill, light mode): categorical slots 1-3 validate all-pairs
COL = {"nsga2": "#2a78d6", "qnehvi": "#eb6834", "sobol": "#1baf7a"}
LABEL = {"nsga2": "NSGA-II", "qnehvi": "qNEHVI", "sobol": "Sobol random"}
INK, INK2, GRID, BASE = "#0b0b0b", "#52514e", "#d9d8d4", "#8c8b87"
BMARK = {"B1": "D", "B1s": "s", "B2": "o", "B3": "^"}
BOFF = {"B1": (6, -10), "B1s": (-14, 6), "B2": (6, 4), "B3": (6, -11)}
BLABEL = {"B1": "B1 helical (end rings)", "B1s": "B1s circumf. (slots)", "B2": "B2 uniform gyroid", "B3": "B3 density-graded"}
LAYOUT_TITLE = {"slots": "Axial slot manifolds (reference)", "end_rings": "End-ring manifolds"}
CAPS = (0.01, 0.03, 0.1, 0.25, 0.3055, 0.5, 1.0, 2.0)
REF_CAP = 0.3055  # Task 10 selection cap = pump power of the Task 9 reference design
FAMILIES = {"slots": ("B1s", "B2", "B3", "U-G", "U-D", "B1"), "end_rings": ("B1", "U-G", "U-D", "B1s", "B2", "B3")}
UMARK = {"U-G": "x", "U-D": "+"}
ULABEL = {"U-G": "uniform gyroid, same manifolds", "U-D": "uniform diamond, same manifolds"}


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def method_front(df: pd.DataFrame) -> pd.DataFrame:
    f = df[df["feasible"].astype(bool)]
    F = f[["thermal_resistance", "pump_power", "mass"]].to_numpy(float)
    return f[pareto_mask(F)] if len(f) else f


def load_baselines() -> tuple[pd.DataFrame, dict]:
    fr = pd.read_csv(results_dir() / "task10_fronts.csv")
    fr["layout"] = "task10"
    sel = {}
    for fam in ("B1", "B1s", "B2", "B3"):
        p = results_dir() / "baselines" / f"{fam}.json"
        if p.is_file():
            d = json.loads(p.read_text())
            sel[fam] = {k: d[k] for k in ("thermal_resistance", "R_active", "pump_power", "mass", "min_structural_margin")}
    return fr, sel


def load_uniform(margin: float = 0.01, fine: bool = True) -> pd.DataFrame:
    """Uniform gyroid / diamond references on both layouts (scripts/task11_uniform_references.py), fronts only,
    screened with the same cell-size rule as the optimized designs (L >= (1 + margin) L_min_fine)."""
    from voxlat.geometry import min_cell_size

    p = results_dir() / "task11_uniform_refs.csv"
    if not p.is_file():
        return pd.DataFrame(columns=["layout", "family", "thermal_resistance", "pump_power", "mass"])
    u = pd.read_csv(p)
    u["cell_ratio_fine"] = u["L"] / np.asarray(min_cell_size(u["w"].to_numpy(), u["rho"].to_numpy(), fine=fine))
    # (uniform G / D only: w = 0, 1 are free of valleys)
    u = u[u["feasible"].astype(bool) & (u["cell_ratio_fine"] >= 1.0 + margin)]
    keep = []
    for _, g in u.groupby(["layout", "family"]):
        keep.append(g[pareto_mask(g[["thermal_resistance", "pump_power", "mass"]].to_numpy(float))])
    return pd.concat(keep, ignore_index=True)


def hv_figure(curves: dict[str, pd.DataFrame], bo_budget: int) -> None:
    fig, axes = plt.subplots(1, len(curves), figsize=(5.2 * len(curves), 3.6), squeeze=False)
    for ax, (layout, c) in zip(axes[0], curves.items()):
        for m in METHODS:
            g = c[c["method"] == m]
            if g.empty:
                continue
            st = g.groupby("n_evals")["hv"].agg(med="median", q1=lambda v: v.quantile(0.25), q3=lambda v: v.quantile(0.75),
                                                n="count")
            st = st[st["n"] >= max(1, st["n"].max() // 2)]  # keep checkpoints most seeds reached
            ax.fill_between(st.index, st["q1"], st["q3"], color=COL[m], alpha=0.18, lw=0)
            ax.plot(st.index, st["med"], color=COL[m], lw=2, label=f"{LABEL[m]} (n={g['seed'].nunique()})")
            ax.annotate(LABEL[m], (st.index[-1], st["med"].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=INK2, va="center")
        ax.axvline(bo_budget, color=INK2, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("expensive (device-model) evaluations", fontsize=8, color=INK2)
        ax.set_ylabel("hypervolume (normalized R, log10 P, mass)", fontsize=8, color=INK2)
        ax.set_title(LAYOUT_TITLE[layout], fontsize=9, color=INK, loc="left")
        style(ax)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("RQ3 - hypervolume vs evaluations (median and interquartile range over seeds)", fontsize=10, color=INK)
    fig.tight_layout()
    save_figure(fig, "task11_hypervolume")


def fronts_figure(runs: dict[str, pd.DataFrame], fr: pd.DataFrame, sel: dict, uni: pd.DataFrame) -> None:
    layouts = list(runs)
    fig, axes = plt.subplots(2, len(layouts), figsize=(5.4 * len(layouts), 7.2), squeeze=False)
    for j, layout in enumerate(layouts):
        df = runs[layout]
        for i, (ycol, xcol, xlab, xs) in enumerate((("thermal_resistance", "pump_power", "pump power at 3 L/min [W]", "log"),
                                                    ("thermal_resistance", "mass", "jacket mass [kg]", "linear"))):
            ax = axes[i, j]
            # baselines: candidate fronts (grey) + selected designs (labelled)
            for fam in ("B1", "B1s", "B2", "B3"):
                b = fr[fr["family"] == fam]
                ax.scatter(b[xcol], b[ycol] * 1e3, s=10, marker=BMARK[fam], facecolor="none", edgecolor=BASE, lw=0.6,
                           alpha=0.7, zorder=2)
                if fam in sel:
                    s = sel[fam]
                    ax.scatter(s[xcol], s[ycol] * 1e3, s=60, marker=BMARK[fam], color=INK, edgecolor="white", lw=1.2, zorder=6)
                    ax.annotate(fam, (s[xcol], s[ycol] * 1e3), xytext=BOFF[fam], textcoords="offset points", fontsize=7,
                                color=INK, zorder=7)
            for fam in ("U-G", "U-D"):
                u = uni[(uni["layout"] == layout) & (uni["family"] == fam)]
                ax.scatter(u[xcol], u[ycol] * 1e3, s=16, marker=UMARK[fam], color=INK2, lw=0.8, zorder=3)
            for m in ("sobol", "nsga2", "qnehvi"):
                g = df[df["method"] == m]
                if g.empty:
                    continue
                f = method_front(g)
                ax.scatter(f[xcol], f[ycol] * 1e3, s={"sobol": 10, "nsga2": 14, "qnehvi": 9}[m], color=COL[m], edgecolor="white", lw=0.4, alpha=0.9, zorder=4,
                           label=f"{LABEL[m]} front" if (i == 0 and j == 0) else None)
            ax.set_xscale(xs)
            if xcol == "pump_power":
                ax.set_xlim(1e-3, 4.0)  # B1 candidates reach 1e3 W; irrelevant region clipped
            else:
                ax.set_xlim(0.23, 0.42)
            ax.set_ylim(5, 45)
            ax.set_xlabel(xlab, fontsize=8, color=INK2)
            ax.set_ylabel("thermal resistance R [mK/W]", fontsize=8, color=INK2)
            if i == 0:
                ax.axvline(REF_CAP, color=INK2, lw=0.8, ls=":")
                ax.set_title(LAYOUT_TITLE[layout], fontsize=9, color=INK, loc="left")
            style(ax)
    h = [plt.Line2D([], [], ls="", marker="o", color=COL[m], label=f"{LABEL[m]} (optimized lattice, union of seeds)")
         for m in METHODS]
    h += [plt.Line2D([], [], ls="", marker=BMARK[f], color=INK, label=f"{BLABEL[f]} - Task 10 pick") for f in BMARK]
    h += [plt.Line2D([], [], ls="", marker="o", mfc="none", color=BASE, label="Task 10 candidate fronts")]
    h += [plt.Line2D([], [], ls="", marker=UMARK[f], color=INK2, label=f"{ULABEL[f]} (front)") for f in UMARK]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.suptitle("RQ2 - feasible Pareto fronts at 3 L/min vs baselines B1-B3 (dotted: 0.306 W selection cap)",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    save_figure(fig, "task11_pareto_fronts")


def design_maps(layout: str, designs: dict[str, tuple[np.ndarray, dict]]) -> None:
    labels = list(designs)
    fig, axes = plt.subplots(len(labels), 4, figsize=(15, 2.5 * len(labels)), squeeze=False, sharex=True, sharey=True)
    specs = (("rho", "relative density rho*", "Blues", None), ("w", "blend w (0 G, 1 D)", "Purples", (0, 1)),
             ("L", "cell size L [mm]", "Greens", None), ("T_wall", "sleeve T, stator side [°C]", "inferno", None))
    Tmin = min(np.nanmin(o["fields"].T_wall) for _, o in designs.values()) - 273.15
    Tmax = max(np.nanmax(o["fields"].T_wall) for _, o in designs.values()) - 273.15
    for r, lab in enumerate(labels):
        x, out = designs[lab]
        f = out["fields"]
        g = f.grid
        th = g.theta_faces_deg()
        lat = g.lattice
        for c, (key, title, cmap, lim) in enumerate(specs):
            ax = axes[r, c]
            if key == "T_wall":
                V = f.T_wall - 273.15
                V = np.where(g.heated, V, np.nan)
                vmin, vmax = Tmin, Tmax
            else:
                V = getattr(f, key) * (1e3 if key == "L" else 1.0)
                V = np.where(lat, V, np.nan)
                vmin, vmax = lim if lim else (np.nanmin(V), np.nanmax(V) + 1e-9)
                if key == "rho":
                    vmin, vmax = 0.2, 0.5
                if key == "L":
                    vmin, vmax = 2.0, 6.0
            pc = ax.pcolormesh(th, g.z_faces * 1e3, V.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat", rasterized=True)
            cb = fig.colorbar(pc, ax=ax, pad=0.01, aspect=15)
            cb.ax.tick_params(labelsize=6, colors=INK2)
            if key == "T_wall":
                k = np.nanargmax(V)
                i, jz = np.unravel_index(k, V.shape)
                ax.plot(g.theta_plot_deg()[i], g.z_centres[jz] * 1e3, "o", mfc="none", mec="white", ms=7)
            if r == 0:
                ax.set_title(title, fontsize=8, color=INK)
            if c == 0:
                R, P, m = out["thermal_resistance"] * 1e3, out["pump_power"], out["mass"]
                ax.set_ylabel(f"{lab}\nR {R:.1f} mK/W, P {P:.3g} W\nm {m:.3f} kg\nz [mm]", fontsize=7, color=INK)
            if r == len(labels) - 1:
                ax.set_xlabel("theta [deg]", fontsize=8, color=INK2)
            ax.tick_params(colors=INK2, labelsize=7)
    fig.suptitle(f"Task 11 representative designs - {LAYOUT_TITLE[layout].lower()}, 3 L/min (circle = hottest point)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    save_figure(fig, f"task11_designs_{layout}")


def critical_cell_check(out: dict, n_cells: int = 3, n_blend: int = 3) -> dict:
    """Direct Task 1 check_manufacturable (n = 48) on the cells with the smallest L / L_min (fine table), plus
    the tightest cells inside the blend band 0.3 < w < 0.9 (where pinch necks live)."""
    from voxlat.geometry import TPMSParams, check_manufacturable, min_cell_size

    f = out["fields"]
    lat = f.grid.lattice
    w, rho, L = f.w[lat], f.rho[lat], f.L[lat]
    ratio = L / np.asarray(min_cell_size(np.clip(w, 0, 1), rho, fine=True))
    band = (w > 0.3) & (w < 0.9)
    cells, seen = [], set()

    def visit(order, limit, tag):
        k_done = 0
        for k in order:
            key = (round(w[k], 3), round(rho[k], 3), round(L[k] * 1e4))
            if key in seen:
                continue
            seen.add(key)
            rep = check_manufacturable(TPMSParams.from_design(float(np.clip(w[k], 0, 1)), float(rho[k]), 1.0),
                                       float(L[k]), n=48)
            cells.append({"w": float(w[k]), "rho": float(rho[k]), "L_m": float(L[k]), "L_over_Lmin_fine": float(ratio[k]),
                          "selected_as": tag, "ok_direct_n48": bool(rep.ok), "reasons": list(getattr(rep, "reasons", []))})
            k_done += 1
            if k_done >= limit:
                break

    visit(np.argsort(ratio), n_cells, "smallest L/L_min")
    bi = np.flatnonzero(band)
    visit(bi[np.argsort(ratio[bi])], n_blend, "blend band 0.3<w<0.9")
    return {"method": "Task 1 check_manufacturable at n = 48 on the most critical cells (fine table) and the tightest "
                      "blend-band cells", "cells": cells, "all_ok": all(c["ok_direct_n48"] for c in cells)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--layouts", nargs="+", default=["slots", "end_rings"])
    ap.add_argument("--no-designs", action="store_true")
    ap.add_argument("--bo-budget", type=int, default=200)
    ap.add_argument("--cell-margin", type=float, default=0.01,
                    help="re-screen: require L >= (1 + margin) L_min in every cell (default 0.01)")
    ap.add_argument("--coarse-table", action="store_true",
                    help="re-screen with the coarse Task 1 table instead of the fine-w table")
    a = ap.parse_args(argv)
    space = ObjectiveSpace()
    fr, sel = load_baselines()
    uni = load_uniform(a.cell_margin, fine=not a.coarse_table)
    screen_rows = []
    runs, curves, run_rows, fronts, eqp, reps = {}, {}, [], {}, [], []
    for layout in a.layouts:
        df = load_runs(layout, a.root)
        if df.empty:
            print(f"no runs for {layout}")
            continue
        df = rescreen_cell_size(df, layout, margin=a.cell_margin, fine=not a.coarse_table)
        for (m, run), g in df.groupby(["method", "run"]):
            e = g[g["expensive"].astype(bool)]
            fc = combined_front(e.assign(feasible=e["feasible_coarse"]), space) if e["feasible_coarse"].any() else e.iloc[:0]
            screen_rows.append({"layout": layout, "method": m, "run": run, "expensive": len(e),
                                "feasible_coarse": int(e["feasible_coarse"].sum()), "feasible_fine": int(e["feasible"].sum()),
                                "coarse_front": len(fc),
                                "coarse_front_rejected_fine": int(((fc["cell_ratio_fine"] < 1.0 + a.cell_margin)
                                                                   | (fc["valley_cells"] > 0)).sum()),
                                "coarse_front_valley": int((fc["valley_cells"] > 0).sum())})
        runs[layout] = df
        n_max = int(df[df["expensive"]].groupby("run").size().max())
        cps = sorted(set(np.unique(np.round(np.geomspace(4, n_max, 60)).astype(int))) | {50, 100, a.bo_budget, n_max})
        c = hv_curves(df, [k for k in cps if k <= n_max], space)
        c["layout"] = layout
        curves[layout] = c
        for run, g in df.groupby("run"):
            e = g[g["expensive"].astype(bool)]
            meta = json.loads((Path(a.root or results_dir() / "task11") / layout / f"{run}.json").read_text())
            cc = c[c["run"] == run]
            f = method_front(g)
            atcap = f[f["pump_power"] <= REF_CAP]
            run_rows.append({"layout": layout, "run": run, "method": g["method"].iloc[0], "seed": int(g["seed"].iloc[0]),
                             "status": meta.get("status"), "proposals": len(g), "expensive": len(e),
                             "feasible": int(e["feasible"].sum()), "wall_time_s": meta.get("wall_time_s"),
                             "bo_backend": meta.get("bo_backend", ""), "hv_final": cc["hv"].iloc[-1] if len(cc) else np.nan,
                             "hv_at_bo_budget": cc.loc[cc["n_evals"] == a.bo_budget, "hv"].max(),
                             "best_R_at_ref_cap": atcap["thermal_resistance"].min() if len(atcap) else np.nan,
                             "min_R": f["thermal_resistance"].min(), "min_P": f["pump_power"].min(), "min_mass": f["mass"].min(),
                             "front_size": len(f), "mean_eval_time_s": e["wall_time"].mean()})
        front = combined_front(df, space)
        front.to_csv(results_dir() / f"task11_front_{layout}.csv", index=False, float_format="%.8g")
        fronts[layout] = front
        base = pd.concat([fr, uni[uni["layout"] == layout]], ignore_index=True)
        t = equal_pump_power(front, base, CAPS, FAMILIES[layout])
        t.insert(0, "layout", layout)
        eqp.append(t)
    if not runs:
        return
    pd.DataFrame(screen_rows).to_csv(results_dir() / "task11_rescreen.csv", index=False)
    hv = pd.concat(curves.values(), ignore_index=True)
    hv.to_csv(results_dir() / "task11_hv_curves.csv", index=False, float_format="%.6g")
    rr = pd.DataFrame(run_rows)
    rr.to_csv(results_dir() / "task11_runs.csv", index=False, float_format="%.6g")
    # HV summary per layout/method (+ evaluations needed to reach the qNEHVI median HV at its budget)
    summ = []
    for (layout, m), g in hv.groupby(["layout", "method"]):
        row = {"layout": layout, "method": m, "seeds": g["seed"].nunique()}
        for k in (50, 100, a.bo_budget):
            v = g[g["n_evals"] == k]["hv"]
            row[f"hv_{k}_median"] = v.median() if len(v) else np.nan
            row[f"hv_{k}_iqr"] = (v.quantile(0.75) - v.quantile(0.25)) if len(v) else np.nan
        last = g.sort_values("n_evals").groupby("run").tail(1)
        row["n_evals_final_median"] = last["n_evals"].median()
        row["hv_final_median"] = last["hv"].median()
        summ.append(row)
    summ = pd.DataFrame(summ)
    for layout in summ["layout"].unique():
        g = hv[hv["layout"] == layout]
        target = g[(g["method"] == "qnehvi") & (g["n_evals"] == a.bo_budget)]["hv"].median()
        for m in METHODS:
            reach = []
            for run, h in g[g["method"] == m].groupby("run"):
                hit = h[h["hv"] >= target]
                reach.append(hit["n_evals"].min() if len(hit) else np.nan)
            sel_ = (summ["layout"] == layout) & (summ["method"] == m)
            summ.loc[sel_, "evals_to_reach_qnehvi_hv200_median"] = np.nanmedian(reach) if np.isfinite(reach).any() else np.nan
            summ.loc[sel_, "seeds_reaching"] = int(np.isfinite(reach).sum())
    summ.to_csv(results_dir() / "task11_hv_summary.csv", index=False, float_format="%.6g")
    eq = pd.concat(eqp, ignore_index=True)
    eq.to_csv(results_dir() / "task11_equal_pump_power.csv", index=False, float_format="%.6g")
    hv_figure(curves, a.bo_budget)
    fronts_figure(runs, fr, sel, uni)
    if not a.no_designs:
        out_dir = results_dir() / "designs"
        out_dir.mkdir(parents=True, exist_ok=True)
        for layout, front in fronts.items():
            rep = representative_designs(front, space)
            capped = front[front["pump_power"] <= REF_CAP]
            if len(capped):
                i_cap = int(front.index.get_loc(capped["thermal_resistance"].idxmin()))
                if i_cap not in rep.values():
                    rep["best_at_cap"] = i_cap  # the graded design behind the equal-pump-power headline
            n_var = sum(1 for c in front.columns if c.startswith("x") and c[1:].isdigit())
            designs = {}
            for lab in [k for k in ("min_R", "best_at_cap", "knee", "min_mass") if k in rep]:
                row = front.iloc[rep[lab]]
                x = x_of(row, n_var)
                out, F, prob = evaluate_x(layout, x, return_fields=True)
                chk = critical_cell_check(out)
                src = {"run": row["run"], "method": row["method"], "seed": int(row["seed"]), "eval_id": int(row["eval_id"]),
                       "selection": {"min_R": "minimum R on the combined feasible front",
                                     "best_at_cap": f"minimum R with pump power <= {REF_CAP} W (Task 10 selection cap)",
                                     "min_mass": "minimum mass on the combined feasible front",
                                     "knee": "closest to the ideal point, objectives (R, log10 P, mass) normalized to [0,1] over the front"}[lab]}
                rec = design_record(layout, lab, x, source=src, manufacturability_check=chk)
                (out_dir / f"task11_{layout}_{lab}.json").write_text(json.dumps(rec, indent=1))
                designs[lab] = (x, out)
                m = rec["metrics_nominal_flow"]
                reps.append({"layout": layout, "design": lab, "R_mK_W": m["thermal_resistance"] * 1e3,
                             "R_active_mK_W": m["R_active"] * 1e3, "pump_power_W": m["pump_power"], "delta_p_Pa": m["delta_p"],
                             "mass_kg": m["mass"], "min_margin": m["min_structural_margin"], "T_wall_max_C": m["T_wall_max"] - 273.15,
                             **{f"{n}_{s}": rec["field_stats"][n][s] for n in ("rho", "w", "L") for s in ("min", "max", "area_mean")},
                             "critical_cells_ok_n48": chk["all_ok"], "source_run": src["run"]})
            design_maps(layout, designs)
        pd.DataFrame(reps).to_csv(results_dir() / "task11_representative.csv", index=False, float_format="%.6g")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(rr.groupby(["layout", "method"])[["expensive", "proposals", "hv_final", "hv_at_bo_budget", "best_R_at_ref_cap",
                                                "wall_time_s"]].median())
        print(summ)
        print(eq[["layout", "pump_power_cap", "family", "R_baseline", "R_opt", "improvement_pct", "improvement_pct_mass_le"]])
        if reps:
            print(pd.DataFrame(reps)[["layout", "design", "R_mK_W", "pump_power_W", "mass_kg", "min_margin",
                                      "critical_cells_ok_n48"]])


if __name__ == "__main__":
    sys.exit(main())
