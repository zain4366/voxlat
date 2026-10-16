"""Audit: do the committed figure/table data agree with the numbers quoted in STATUS.md?

    python scripts/experiments/check_consistency.py          # writes results/reproduce/consistency_report.{md,csv}

Each check reads one number from the data behind a paper figure/table and compares it with the value quoted in
STATUS.md (to the precision quoted there). A check is FLAGGED when they differ by more than that precision. Also
checks that every catalogued figure/table file exists and that the figure files are not older than their data.
Exit code 1 if anything is flagged (run_all reports it as FLAGS, not as a failure).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from catalog import ITEMS, PAPER_DESIGN  # noqa: E402


def csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def js(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def row(df: pd.DataFrame, **kw) -> pd.Series:
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        col = df[k]
        m &= np.isclose(col, v) if isinstance(v, float) else (col == v).to_numpy()
    sub = df[m]
    if len(sub) != 1:
        raise LookupError(f"{len(sub)} rows match {kw}")
    return sub.iloc[0]


@dataclass
class Check:
    id: str
    items: str            # figure/table ids the number belongs to
    claim: str            # where in STATUS.md / what is claimed
    expected: float
    tol: float            # absolute tolerance (= precision quoted in STATUS)
    actual: Callable[[], float]
    note: str = ""


def _eq_pump(layout, cap, fam, col="improvement_pct"):
    return lambda: row(csv("results/task11_equal_pump_power.csv"), layout=layout, pump_power_cap=cap, family=fam)[col]


def _rep(layout, design, col):
    return lambda: row(csv("results/task11_representative.csv"), layout=layout, design=design)[col]


def _hv(layout, method, col):
    return lambda: row(csv("results/task11_hv_summary.csv"), layout=layout, method=method)[col]


def _cmp(fam, flow, col, scale=1.0):
    return lambda: row(csv("results/task10_comparison.csv"), family=fam, flow_lpm=flow)[col] * scale


def _metric(model, split, q, col):
    return lambda: row(csv("results/task6_metrics.csv"), model=model, split=split, quantity=q)[col]


def _gyroid(key, scale=1.0):
    return lambda: js(PAPER_DESIGN)["metrics_nominal_flow"][key] * scale


CHECKS: list[Check] = [
    # ---- Task 1 -----------------------------------------------------------------------------------------
    Check("t1_asf_gyroid", "F03", "Task 1/Task 5: gyroid rho*=0.35 a_sf L = 2.970 (dataset regression row)", 2.970, 0.002,
          lambda: float(row(csv("results/task4_permeability_vs_porosity.csv"), w=0.0, rho=0.35)["a_sf_L"]),
          "a_sf L of the Task 4 table (n = 64 geometry)"),
    # ---- Task 2 -----------------------------------------------------------------------------------------
    Check("t2_keff_G035", "F06", "Task 2 results table: gyroid rho 0.35 k_eff/k_s = 0.208", 0.208, 0.0006,
          lambda: row(csv("results/task2_keff_vs_density.csv"), w=0.0, rho=0.35, kf_ks=1 / 325)["k_mean"]),
    Check("t2_keff_D050", "F06", "Task 2 results table: diamond rho 0.50 k_eff/k_s = 0.336", 0.336, 0.0006,
          lambda: row(csv("results/task2_keff_vs_density.csv"), w=1.0, rho=0.5, kf_ks=1 / 325)["k_mean"]),
    Check("t2_keff_B030", "F06", "Task 2 results table: blend rho 0.30 k_eff/k_s = 0.097", 0.097, 0.0006,
          lambda: row(csv("results/task2_keff_vs_density.csv"), w=0.5, rho=0.3, kf_ks=1 / 325)["k_mean"]),
    Check("t2_R3264_max", "F07/T01", "Task 2: R(32,64) max |error| over the 8 cases = 0.7 %", 0.7, 0.06,
          lambda: 100 * csv("results/task2_convergence_summary.csv")["err R(32,64)"].abs().max()),
    Check("t2_raw32_max", "F07/T01", "Task 2: raw n = 32 up to 17 % low", 17.0, 0.6,
          lambda: 100 * csv("results/task2_convergence_summary.csv")["err raw n=32"].abs().max()),
    # ---- Task 3 -----------------------------------------------------------------------------------------
    Check("t3_E_G035", "F08", "Task 3 results: gyroid rho 0.35 E*/E_s = 0.0998", 0.0998, 0.00006,
          lambda: row(csv("results/task3_stiffness_vs_density.csv"), w=0.0, rho=0.35)["E_axial"]),
    Check("t3_E_D050", "F08", "Task 3 results: diamond rho 0.50 E*/E_s = 0.1789", 0.1789, 0.00006,
          lambda: row(csv("results/task3_stiffness_vs_density.csv"), w=1.0, rho=0.5)["E_axial"]),
    Check("t3_fit_G_m", "F08", "Task 3: gyroid E*/E_s = 0.90 rho^2.13", 2.13, 0.006,
          lambda: row(csv("results/task3_powerlaw_fits.csv"), w=0.0, quantity="E_axial")["m"]),
    Check("t3_fit_D_m", "F08", "Task 3: diamond E*/E_s = 0.67 rho^1.93", 1.93, 0.006,
          lambda: row(csv("results/task3_powerlaw_fits.csv"), w=1.0, quantity="E_axial")["m"]),
    Check("t3_zener_D020", "F09", "Task 3 results: diamond Zener A at rho 0.20 = 2.29", 2.29, 0.006,
          lambda: row(csv("results/task3_stiffness_vs_density.csv"), w=1.0, rho=0.2)["zener"]),
    Check("t3_loc_G035", "F10", "Task 3 results: gyroid rho 0.35 K_p99 uniaxial z = 15", 15.0, 0.6,
          lambda: row(csv("results/task3_stiffness_vs_density.csv"), w=0.0, rho=0.35)["loc_uniaxial_z_p99"]),
    Check("t3_loc_B020", "F10", "Task 3 results: blend rho 0.20 K_p99 uniaxial z = 95", 95.0, 0.6,
          lambda: row(csv("results/task3_stiffness_vs_density.csv"), w=0.5, rho=0.2)["loc_uniaxial_z_p99"]),
    # ---- Task 4 -----------------------------------------------------------------------------------------
    Check("t4_K_G035", "F14", "Task 4 results: gyroid rho 0.35 K/L^2 = 4.887e-3", 4.887e-3, 6e-7,
          lambda: row(csv("results/task4_permeability_vs_porosity.csv"), w=0.0, rho=0.35)["K_mean"]),
    Check("t4_K_D020", "F14", "Task 4 results: diamond rho 0.20 K/L^2 = 6.651e-3", 6.651e-3, 6e-7,
          lambda: row(csv("results/task4_permeability_vs_porosity.csv"), w=1.0, rho=0.2)["K_mean"]),
    Check("t4_tort_D050", "F14", "Task 4 results: diamond rho 0.50 tortuosity = 1.29", 1.29, 0.006,
          lambda: row(csv("results/task4_permeability_vs_porosity.csv"), w=1.0, rho=0.5)["tortuosity"]),
    Check("t4_R3264_max", "F13/T01", "Task 4: R(32,64) max |error| = 1.0 %", 1.0, 0.06,
          lambda: 100 * csv("results/task4_convergence_summary.csv").query("quantity == 'K_mean'")["err R(32,64)"]
          .abs().max()),
    # ---- Task 6 -----------------------------------------------------------------------------------------
    Check("t6_cnn_band_E", "F19", "Task 6 CNN table: band split E_star CNN MAPE = 107.2 %", 107.2, 0.06,
          lambda: 100 * row(csv("results/task6_cnn_metrics.csv"), split="band", model="cnn", quantity="E_star")["MAPE"]),
    Check("t6_cnn_cv5_K_gp", "F19", "Task 6 CNN table: cv5 K_star GP MAPE = 0.7 %", 0.7, 0.06,
          lambda: 100 * row(csv("results/task6_cnn_metrics.csv"), split="cv5", model="gp", quantity="K_star")["MAPE"]),
    Check("t6_asf_cv5_R2", "F16/T02", "Task 6 cv5 table: a_sf L GP R^2 = 0.999", 0.999, 0.0006,
          lambda: _first(csv("results/task6_metrics.csv"), ("gp+cal", "gp"), "cv5", ("a_sf_L", "asf_L", "a_sf"), "R2")),
    # ---- Task 7 -----------------------------------------------------------------------------------------
    Check("t7_KtG035_N1", "F20/T03", "Task 7: gyroid rho 0.35 K_t delta at N = 1 = -26.6 %", -26.6, 0.06,
          lambda: 100 * _t7("G", 0.35)["delta_K_t_N1_mean"]),
    Check("t7_KtD045_N1", "F20/T03", "Task 7: diamond rho 0.45 K_t delta at N = 1 = -18.7 %", -18.7, 0.06,
          lambda: 100 * _t7("D", 0.45)["delta_K_t_N1_mean"]),
    Check("t7_CnnB035_N1", "F20/T03", "Task 7: blend rho 0.35 C_nn delta at N = 1 = +86 %", 86.0, 0.6,
          lambda: 100 * _t7("B0.50", 0.35)["delta_C_nn_N1_mean"]),
    Check("t7_beta_Kt", "F20/T03", "Task 7: K_t wall coefficient beta_1 = +0.2663 (models/finite_gap_correction.json)",
          0.2663, 0.00006, lambda: _t7_beta("K_t")),
    # ---- Task 9 -----------------------------------------------------------------------------------------
    Check("t9_R_ref", "F27/F29", "Task 9: uniform gyroid rho 0.35, L 4 mm, 3 L/min R = 24.37 mK/W", 24.37, 0.006,
          lambda: js("results/task9_reference_gyroid.json")["thermal_resistance"] * 1e3),
    Check("t9_dp_ref", "F27/F29", "Task 9: reference dp = 1833 Pa", 1833.0, 0.6,
          lambda: js("results/task9_reference_gyroid.json")["delta_p"]),
    Check("t9_P_ref", "F29", "Task 9/10: reference pump power 0.306 W (selection cap)", 0.306, 0.0006,
          lambda: js("results/task9_reference_gyroid.json")["pump_power"]),
    Check("t9_grid_default", "F28", "Task 9 grid table: default 1.25/2.5 mm grid R = 24.37 mK/W", 24.37, 0.006,
          lambda: row(csv("results/task9_grid_convergence.csv"), spacing_s_mm=1.25, spacing_z_mm=2.5)["R"] * 1e3),
    Check("t9_sweep_D3", "F29", "Task 9 flow table: diamond R at 3 L/min = 24.3 mK/W", 24.3, 0.06,
          lambda: row(csv("results/task9_flow_sweep.csv"), design="diamond", flow_lpm=3.0)["R"] * 1e3),
    # ---- Task 10 ----------------------------------------------------------------------------------------
    Check("t10_B1_R3", "F32/T05", "Task 10: B1 R at 3 L/min = 13.3 mK/W", 13.3, 0.06, _cmp("B1", 3.0, "thermal_resistance", 1e3)),
    Check("t10_B1s_R3", "F32/T05", "Task 10: B1s R at 3 L/min = 25.7 mK/W", 25.7, 0.06, _cmp("B1s", 3.0, "thermal_resistance", 1e3)),
    Check("t10_B3_R3", "F32/T05", "Task 10: B3 R at 3 L/min = 22.7 mK/W", 22.7, 0.06, _cmp("B3", 3.0, "thermal_resistance", 1e3)),
    Check("t10_B2_R3", "F32/T05", "Task 10: B2 (= Task 9 reference) R at 3 L/min = 24.4 mK/W", 24.4, 0.06,
          _cmp("B2", 3.0, "thermal_resistance", 1e3)),
    Check("t10_B2_mass", "F32/T05", "Task 10: B2 mass 0.344 kg", 0.344, 0.0006, _cmp("B2", 3.0, "mass")),
    # ---- Task 11 (RQ3) ----------------------------------------------------------------------------------
    Check("t11_hv_slots_q200", "F33/T06", "Task 11 RQ3 table: slots qNEHVI HV at 200 = 0.1998", 0.1998, 0.00006,
          _hv("slots", "qnehvi", "hv_200_median")),
    Check("t11_hv_slots_nsga_final", "F33/T06", "Task 11 RQ3 table: slots NSGA-II final HV = 0.2093", 0.2093, 0.00006,
          _hv("slots", "nsga2", "hv_final_median")),
    Check("t11_hv_er_q200", "F33/T06", "Task 11 RQ3 table: end rings qNEHVI HV at 200 = 0.6748", 0.6748, 0.00006,
          _hv("end_rings", "qnehvi", "hv_200_median")),
    Check("t11_evals_1368_slots", "F33/T06", "Task 11: NSGA-II needs a median 1 368 evaluations (both layouts)", 1368, 0.5,
          _hv("slots", "nsga2", "evals_to_reach_qnehvi_hv200_median")),
    Check("t11_evals_1368_er", "F33/T06", "Task 11: NSGA-II needs a median 1 368 evaluations (both layouts)", 1368, 0.5,
          _hv("end_rings", "nsga2", "evals_to_reach_qnehvi_hv200_median")),
    # ---- Task 11 (RQ2) ----------------------------------------------------------------------------------
    Check("t11_cap_vs_B1s", "F34/T07", "Task 11 RQ2: 0.306 W cap, slots, -14.5 % vs B1s", 14.5, 0.06,
          _eq_pump("slots", 0.3055, "B1s")),
    Check("t11_cap_vs_B2", "F34/T07", "Task 11 RQ2: 0.306 W cap, slots, -10.3 % vs B2", 10.3, 0.06,
          _eq_pump("slots", 0.3055, "B2")),
    Check("t11_cap_vs_B3", "F34/T07", "Task 11 RQ2: 0.306 W cap, slots, -3.2 % vs B3", 3.2, 0.06,
          _eq_pump("slots", 0.3055, "B3")),
    Check("t11_cap_er_vs_B1", "F34/T07", "Task 11 RQ2: end rings -35.3 % vs B1 at 0.306 W", 35.3, 0.06,
          _eq_pump("end_rings", 0.3055, "B1")),
    Check("t11_rep_bac_R", "F35/T08", "Task 11 representative: slots best_at_cap R = 21.94 mK/W", 21.94, 0.006,
          _rep("slots", "best_at_cap", "R_mK_W")),
    Check("t11_rep_bac_mass", "F35/T08", "Task 11 representative: slots best_at_cap mass 0.310 kg", 0.310, 0.0006,
          _rep("slots", "best_at_cap", "mass_kg")),
    Check("t11_rep_er_minR", "F36/T08", "Task 11 representative: end rings min_R R = 8.57 mK/W", 8.57, 0.006,
          _rep("end_rings", "min_R", "R_mK_W")),
    Check("t11_rep_er_knee_P", "F36/T08", "Task 11 representative: end rings knee pump power 0.0056 W", 0.0056, 0.00006,
          _rep("end_rings", "knee", "pump_power_W")),
    # ---- Task 12 / paper design -------------------------------------------------------------------------
    Check("t11_B2_cap_eq_t10", "T07/F34",
          "Task 10 B2 (rho* 0.35, L 4 mm) R = 24.37 mK/W at 0.3055 W is the best uniform gyroid at the 0.306 W cap; "
          "Task 11 uses the same cap for its B2 baseline", 24.37, 0.006,
          lambda: row(csv("results/task11_equal_pump_power.csv"), layout="slots", pump_power_cap=0.3055,
                      family="B2")["R_baseline"] * 1e3,
          "Task 11 cap 0.3055 W is below the reference design's 0.305541 W, so Task 11's B2 baseline is a different "
          "uniform gyroid (24.46 mK/W at 0.293 W); gains vs B2 are overstated by ~0.3 pp"),
    Check("t11_cap_gyr_P", "T07", "Task 12: best_at_cap_gyroid stays within the 0.3055 W selection cap", 0.3055, 1e-6,
          _gyroid("pump_power"),
          "0.3076 W: 0.7 % above the cap (STATUS Task 12 finding 1 says so); T07 compares at matched pump power"),
    Check("t6_cv5_Kprinc_cov", "F16/T02", "Task 6 cv5: K principal values GP cov95 = 93 %", 93.0, 0.6,
          lambda: 100 * csv("results/task6_metrics.csv").query("model == 'gp+cal' and split == 'cv5' and "
                                                                "quantity in ['K_eig1','K_eig2','K_eig3']").cov95.mean()),
    Check("t12_full_vol", "T09", "Task 12 laptop run 3: full jacket volume 136 345 mm^3 raw", 136345, 0.6,
          lambda: js("results/export/task12_slots_best_at_cap_gyroid_full_checks.json")["picogk_report"]["volume_raw_mm3"]),
    Check("t12_gyr_R", "T07/T08", "Task 12: best_at_cap_gyroid R = 21.92 mK/W", 21.92, 0.006, _gyroid("thermal_resistance", 1e3)),
    Check("t12_gyr_P", "T07/T08", "Task 12: best_at_cap_gyroid pump power 0.3076 W", 0.3076, 0.00006, _gyroid("pump_power")),
    Check("t12_gyr_margin", "T07/T08", "Task 12: best_at_cap_gyroid margin 9.8", 9.8, 0.06, _gyroid("min_structural_margin")),
    Check("t12_iou", "T09/F37", "Task 12 laptop run 2: PicoGK vs numpy IoU 0.99676", 0.99676, 0.000006,
          lambda: js("results/export/task12_slots_best_at_cap_gyroid_sector30_checks.json")["numpy_vs_picogk_raw"]["iou"]),
    Check("t12_minwall", "T09", "Task 12 laptop run 2: min wall 0.482 mm", 0.482, 0.0006,
          lambda: js("results/export/task12_slots_best_at_cap_gyroid_sector30_checks.json")["checks"]["wall"]["min_wall_mm"]),
    Check("t12_throat", "T09", "Task 12 laptop run 2: min throat 1.07 mm", 1.07, 0.006,
          lambda: js("results/export/task12_slots_best_at_cap_gyroid_sector30_checks.json")["checks"]["min_throat_mm"]),
    Check("t12_full_throat", "T09", "Task 12 laptop run 3: full jacket throat 1.22 mm", 1.22, 0.006,
          lambda: js("results/export/task12_slots_best_at_cap_gyroid_full_checks.json")["checks"]["min_throat_mm"]),
    Check("t12_full_tri", "T09", "Task 12 laptop run 3: 15.0 M triangles", 15.0, 0.06,
          lambda: js("results/export/task12_slots_best_at_cap_gyroid_full_checks.json")["picogk_report"]["triangles"] / 1e6),
    Check("t12_bac_minwall", "F39/T09", "Task 12: slots/best_at_cap numpy sector min wall 0.306 mm (fails)", 0.306, 0.0006,
          lambda: js("results/export/slots_best_at_cap_sector30_numpy_checks.json")["checks"]["wall"]["min_wall_mm"]),
    Check("t12_erminR_throat", "T09", "Task 12: end_rings/min_R throat 0.785 mm", 0.785, 0.0006,
          lambda: js("results/export/end_rings_min_R_sector30_numpy_checks.json")["checks"]["min_throat_mm"]),
]


def _first(df, models, split, quantities, col):
    for m in models:
        for q in quantities:
            sub = df[(df.model == m) & (df.split == split) & (df.quantity == q)]
            if len(sub):
                return float(sub.iloc[0][col])
    raise LookupError(f"none of {quantities} for {models}/{split}")


def _t7(morph, rho):
    w = csv("results/task7_wall_coefficients.csv")
    sub = w[(w.morphology == morph) & np.isclose(w.rho, rho) & np.isclose(w.a_z, 1.0)]
    if len(sub) != 1:
        raise LookupError(f"{len(sub)} rows for {morph} {rho}")
    return sub.iloc[0]


def _t7_beta(q):
    d = js("models/finite_gap_correction.json")
    for path in (("physical", "models", q, "beta"), ("quantities", q, "beta"), (q, "beta")):
        x = d
        try:
            for k in path:
                x = x[k]
        except (KeyError, TypeError):
            continue
        return float(x[0] if isinstance(x, list) else x.get("1", x.get("const")))
    raise LookupError(f"beta for {q} not found; keys {list(d)[:8]}")


# ------------------------------------------------------------------------------------- structural checks
def structural() -> list[dict]:
    out = []
    for it in ITEMS:
        p = ROOT / it.path
        if not p.exists():
            out.append(dict(id=f"exists_{it.id}", items=it.id, status="FLAG", claim=f"{it.path} exists",
                            expected="yes", actual="missing", note="run the item's script"))
            continue
        if it.kind != "figure":
            continue
        for e in it.extra:
            q = ROOT / e
            if q.exists() and q.suffix in (".csv", ".gz", ".json", ".parquet") and q.stat().st_mtime > p.stat().st_mtime + 3600:
                out.append(dict(id=f"stale_{it.id}", items=it.id, status="FLAG", claim=f"{it.path} newer than its data",
                                expected="figure newer", actual=f"{e} is newer",
                                note="mtimes from git checkout are unreliable; re-run --replot to be sure"))
    return out


def main() -> int:
    rows = []
    for c in CHECKS:
        try:
            a = float(c.actual())
            ok = abs(a - c.expected) <= c.tol + 1e-12
            rows.append(dict(id=c.id, items=c.items, status="ok" if ok else "FLAG", claim=c.claim,
                             expected=c.expected, actual=round(a, 6), diff=round(a - c.expected, 6), note=c.note))
        except Exception as e:  # noqa: BLE001
            rows.append(dict(id=c.id, items=c.items, status="ERROR", claim=c.claim, expected=c.expected,
                             actual="", diff="", note=f"{type(e).__name__}: {e}"))
    rows += structural()
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "reproduce"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "consistency_report.csv", index=False)
    flagged = df[df.status != "ok"]
    md = ["# Consistency audit: figure/table data vs STATUS.md", "",
          f"{len(df)} checks, {int((df.status == 'ok').sum())} ok, {len(flagged)} flagged.", ""]
    if len(flagged):
        md += ["## Flagged", "", "| id | items | claim (STATUS) | expected | actual | note |", "|---|---|---|---|---|---|"]
        for _, r in flagged.iterrows():
            md.append(f"| {r.id} | {r['items']} | {r.claim} | {r.expected} | {r.actual} | {r.note} |")
        md.append("")
    md += ["## All checks", "", "| id | items | status | expected | actual |", "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r.id} | {r['items']} | {r.status} | {r.expected} | {r.actual} |")
    (out / "consistency_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{len(df)} checks: {int((df.status == 'ok').sum())} ok, {len(flagged)} flagged -> "
          f"{(out / 'consistency_report.md').relative_to(ROOT)}")
    for _, r in flagged.iterrows():
        print(f"  {r.status:5s} {r.id:22s} {r['items']:9s} expected {r.expected} actual {r.actual}  {r.note}")
    return 1 if len(flagged) else 0


if __name__ == "__main__":
    raise SystemExit(main())
