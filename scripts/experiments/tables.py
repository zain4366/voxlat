"""Paper tables T01-T09 (results/tables/<id>_<name>.csv + .md) from the committed result files.

    python scripts/experiments/tables.py all        # every table (< 10 s)
    python scripts/experiments/tables.py T07        # one table

Pure post-processing: no solver is run here. Each table states its source files in the Markdown header so a
reader can trace every number back to the CSV/JSON written by the task scripts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import ITEMS, PAPER_DESIGN, item  # noqa: E402

RES = ROOT / "results"
OUT = RES / "tables"


def _read(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def _md(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for v in row.values:
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write(table_id: str, df: pd.DataFrame, sources: list[str], notes: str = "") -> Path:
    it = item(table_id)
    path = ROOT / it.path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6g")
    md = [f"# {it.id} - {it.name.replace('_', ' ')}", "", it.caption, "",
          "Sources: " + ", ".join(f"`{s}`" for s in sources), ""]
    if notes:
        md += [notes, ""]
    md.append(_md(df))
    path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------- T01
def t01() -> Path:
    rows = []
    k = _read("results/task2_convergence_summary.csv")
    for _, r in k.iterrows():
        rows.append(dict(quantity="k_eff/k_s", w=r.w, rho=r.rho, case=f"k_f/k_s={r.kf_ks:.4g}", reference=r.k_ref,
                         p_observed=r.p_observed, err_raw_n32=r.get("err raw n=32", np.nan), err_raw_n64=r.get("err raw n=64", np.nan),
                         err_R32_64=r.get("err R(32,64)", np.nan)))
    c = _read("results/task3_convergence_summary.csv")
    for _, r in c[c.quantity.isin(["E_axial", "C11c", "C12c", "C44c"])].iterrows():
        rows.append(dict(quantity=f"C_eff {r.quantity}", w=r.w, rho=r.rho, case="", reference=r.ref,
                         p_observed=r.p_fit, err_raw_n32=r.get("err raw n=32", np.nan), err_raw_n64=r.get("err raw n=64", np.nan),
                         err_R32_64=r.get("err R(32,64)", np.nan)))
    s = _read("results/task4_convergence_summary.csv")
    for _, r in s[s.quantity.isin(["K_mean"])].iterrows():
        rows.append(dict(quantity="K/L^2", w=r.w, rho=r.rho, case="", reference=r.ref, p_observed=r.p_fit,
                         err_raw_n32=r.get("err raw n=32", np.nan), err_raw_n64=r.get("err raw n=64", np.nan), err_R32_64=r.get("err R(32,64)", np.nan)))
    df = pd.DataFrame(rows)
    return _write("T01", df, ["results/task2_convergence_summary.csv", "results/task3_convergence_summary.csv",
                              "results/task4_convergence_summary.csv"],
                  "Errors are relative to the extrapolated reference of each case (signed).")


# ---------------------------------------------------------------------------------------------- T02
def t02() -> Path:
    m = _read("results/task6_metrics.csv")
    groups = {"K principal": ["K_eig1", "K_eig2", "K_eig3"], "k_eff": ["keff_xx", "keff_yy", "keff_zz"],
              "C_eff": [c for c in m.quantity.unique() if str(c).startswith("C") and len(str(c)) == 3],
              "E": ["E_x", "E_y", "E_z"], "a_sf L": ["a_sf_L", "asf_L", "a_sf"],
              "localization p99": [q for q in m.quantity.unique() if str(q).startswith("loc_")]}
    rows = []
    for model in ["gp+cal", "ensemble+cal"]:
        for split in ["cv5", "band", "sparse"]:
            for g, qs in groups.items():
                sub = m[(m.model == model) & (m.split == split) & m.quantity.isin(qs)]
                if sub.empty:
                    continue
                rows.append(dict(model=model, split=split, group=g, n_targets=len(sub),
                                 R2_median=sub.R2.median(), R2_min=sub.R2.min(), MAPE_median=sub.MAPE.median(),
                                 MAPE_max=sub.MAPE.max(), cov95_median=sub.cov95.median()))
    return _write("T02", pd.DataFrame(rows), ["results/task6_metrics.csv"],
                  "Medians over the targets of each group; MAPE only for positive quantities.")


# ---------------------------------------------------------------------------------------------- T03
def t03() -> Path:
    w = _read("results/task7_wall_coefficients.csv")
    keep = ["morphology", "w", "rho", "a_z"]
    for q in ["k_n", "K_t", "C_nn", "G_t", "G_sz"]:
        keep += [c for c in (f"a_{q}", f"a_{q}_phase_std", f"a_{q}_fit") if c in w.columns]
    return _write("T03", w[keep], ["results/task7_wall_coefficients.csv", "models/finite_gap_correction.json"],
                  "delta = (P_strip - P_homogenized)/P_homogenized ~ a/N; a_*_fit = physical-form fit used by Task 9.")


# ---------------------------------------------------------------------------------------------- T04
def t04() -> Path:
    t = _read("results/task8_literature_table.csv")
    cols = [c for c in ["key", "quantity", "source", "doi", "tpms", "method", "Re_range", "porosity_range",
                        "Pr_range", "implemented"] if c in t.columns]
    return _write("T04", t[cols], ["results/task8_literature_table.csv", "docs/task8_literature_closures.md"])


# ---------------------------------------------------------------------------------------------- T05
def t05() -> Path:
    c = _read("results/task10_comparison.csv")
    c = c[c.family != "ref"].copy()
    df = pd.DataFrame(dict(family=c.family, flow_lpm=c.flow_lpm, R_mK_W=c.thermal_resistance * 1e3,
                           R_active_mK_W=c.R_active * 1e3, delta_p_Pa=c.delta_p, pump_power_W=c.pump_power,
                           mass_kg=c.mass, min_margin=c.min_structural_margin, label=c.label))
    return _write("T05", df, ["results/task10_comparison.csv", "results/baselines/*.json"])


# ---------------------------------------------------------------------------------------------- T06
def t06() -> Path:
    h = _read("results/task11_hv_summary.csv")
    return _write("T06", h, ["results/task11_hv_summary.csv", "results/task11_hv_curves.csv"])


# ---------------------------------------------------------------------------------------------- T07
def _paper_design_metrics() -> dict:
    d = json.loads((ROOT / PAPER_DESIGN).read_text())
    m = d["metrics_nominal_flow"]
    return dict(R=m["thermal_resistance"], R_active=m["R_active"], P=m["pump_power"], mass=m["mass"],
                margin=m["min_structural_margin"])


def _best_below(cap: float, layout: str) -> dict[str, dict]:
    """Best feasible design of each baseline family with pump power <= cap (Task 10/11 selection rule)."""
    out: dict[str, dict] = {}
    fr = _read("results/task10_fronts.csv")
    fr = fr[(fr.min_structural_margin >= 1.0) & (fr.pump_power <= cap)]
    for fam, g in fr.groupby("family"):
        r = g.loc[g.thermal_resistance.idxmin()]
        out[str(fam)] = dict(R=r.thermal_resistance, P=r.pump_power, mass=r.mass)
    u = _read("results/task11_uniform_refs.csv")
    u = u[(u.layout == layout) & u.feasible.astype(bool) & (u.pump_power <= cap)]
    for fam, g in u.groupby("family"):
        r = g.loc[g.thermal_resistance.idxmin()]
        out[str(fam)] = dict(R=r.thermal_resistance, P=r.pump_power, mass=r.mass)
    return out


def t07() -> Path:
    e = _read("results/task11_equal_pump_power.csv")
    rows = []
    for _, r in e.iterrows():
        if not np.isfinite(r.R_opt):
            continue
        rows.append(dict(layout=r.layout, design="Task 11 optimum (archive)", cap_W=r.pump_power_cap,
                         family=r.family, R_baseline_mK_W=r.R_baseline * 1e3, P_baseline_W=r.P_baseline,
                         R_opt_mK_W=r.R_opt * 1e3, P_opt_W=r.P_opt, mass_opt_kg=r.mass_opt,
                         R_change_pct=-r.improvement_pct))
    g = _paper_design_metrics()
    cap_note = []
    for cap, label in [(0.3055, "0.306 W cap (Task 11 rule)"), (g["P"], "matched: cap = P of the gyroid optimum")]:
        base = _best_below(cap, "slots")
        for fam in ["B1s", "B2", "B3", "U-G", "U-D", "B1"]:
            if fam not in base:
                continue
            b = base[fam]
            rows.append(dict(layout="slots", design=f"best_at_cap_gyroid, {label}", cap_W=round(cap, 4),
                             family=fam, R_baseline_mK_W=b["R"] * 1e3, P_baseline_W=b["P"], R_opt_mK_W=g["R"] * 1e3,
                             P_opt_W=g["P"], mass_opt_kg=g["mass"], R_change_pct=100 * (g["R"] / b["R"] - 1)))
        cap_note.append(label)
    df = pd.DataFrame(rows)
    notes = ("R_change_pct < 0 = optimized design better. Rows 'best_at_cap_gyroid' use the manufacturable slot "
             f"optimum `{PAPER_DESIGN}` (w = 0; P = {g['P']:.4f} W, i.e. {100 * (g['P'] / 0.3055 - 1):+.1f} % above "
             "the 0.3055 W cap), against the baselines below the 0.3055 W cap and below its own pump power; "
             "baseline families from results/task10_fronts.csv (B1, B1s, B2, B3) and results/task11_uniform_refs.csv "
             "(U-G, U-D, slots layout, corrected cell-size rule).")
    return _write("T07", df, ["results/task11_equal_pump_power.csv", PAPER_DESIGN, "results/task10_fronts.csv",
                              "results/task11_uniform_refs.csv"], notes)


# ---------------------------------------------------------------------------------------------- T08
def t08() -> Path:
    r = _read("results/task11_representative.csv")
    d = json.loads((ROOT / PAPER_DESIGN).read_text())
    m = d["metrics_nominal_flow"]
    base = r[(r.layout == "slots") & (r.design == "best_at_cap")].iloc[0].to_dict()
    base.update(design="best_at_cap_gyroid (manufacturable, exported)", R_mK_W=m["thermal_resistance"] * 1e3,
                R_active_mK_W=m["R_active"] * 1e3, pump_power_W=m["pump_power"], delta_p_Pa=m["delta_p"],
                mass_kg=m["mass"], min_margin=m["min_structural_margin"], T_wall_max_C=m["T_wall_max"] - 273.15,
                w_min=0.0, w_max=0.0, w_area_mean=0.0, critical_cells_ok_n48=True, source_run="Task 12 (w = 0)")
    df = pd.concat([r, pd.DataFrame([base])], ignore_index=True)
    for c in ["L_min", "L_max", "L_area_mean"]:
        df[c.replace("L_", "L_") + "_mm"] = df[c] * 1e3
        df = df.drop(columns=c)
    return _write("T08", df, ["results/task11_representative.csv", PAPER_DESIGN],
                  "Graded-field statistics of best_at_cap_gyroid equal those of best_at_cap (same rho*/L nets).")


# ---------------------------------------------------------------------------------------------- T09
def _get(d: dict, *keys, default=np.nan):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def t09() -> Path:
    rows = []
    for p in sorted((RES / "export").glob("*_checks.json")):
        d = json.loads(p.read_text())
        ch = d.get("checks", d)
        src = "PicoGK (laptop)" if "picogk_report" in d else "numpy mirror"
        mode = "full jacket" if "_full_" in p.name else "30 deg sector"
        rep = d.get("picogk_report") or {}
        vf = d.get("volume_fraction_raw") or {}
        rows.append(dict(
            design=p.name.split("_sector30")[0].split("_full")[0], geometry=mode, source=src,
            voxel_mm=round(float(_get(ch, "voxel_mm")), 4),
            min_wall_mm=_get(ch, "wall", "min_wall_mm") if _get(ch, "wall_resolved", default=False) else np.nan,
            sliver_min_mm=_get(ch, "interface_slivers", "min_wall_mm"),
            throat_mm=_get(ch, "min_throat_mm"),
            connected=_get(ch, "connected"),
            pockets=_get(ch, "trapped_pockets", "count"),
            voxel_scale_pockets=_get(ch, "trapped_pockets", "voxel_scale_count"),
            loose_solid=_get(ch, "loose_solid", "count"),
            all_pass=_get(ch, "all_pass"),
            vf_err_vs_design_pct=100 * vf["relative_error"] if "relative_error" in vf else np.nan,
            iou_vs_numpy=_get(d, "numpy_vs_picogk_raw", "iou"),
            picogk_total_s=_get(rep, "timings_s", "total"),
            triangles=_get(rep, "triangles"),
            file=f"results/export/{p.name}"))
    df = pd.DataFrame(rows)
    return _write("T09", df, ["results/export/*_checks.json"],
                  "Empty cells: quantity not recorded in that report (see the JSON). STATUS Task 12 holds the "
                  "laptop numbers in prose; this table is generated from the JSON reports only.")


TABLES = {"T01": t01, "T02": t02, "T03": t03, "T04": t04, "T05": t05, "T06": t06, "T07": t07, "T08": t08, "T09": t09}


def main(argv: list[str] | None = None) -> int:
    args = (argv if argv is not None else sys.argv[1:]) or ["all"]
    ids = list(TABLES) if "all" in args else [a.upper() for a in args]
    for t in ids:
        p = TABLES[t]()
        print(f"{t}: {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
