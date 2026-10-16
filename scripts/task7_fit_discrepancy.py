"""Task 7: fit and assess the finite-gap discrepancy models; write tables and the correction file.

Reads data/finite_gap.csv (scripts/task7_finite_gap.py) and writes
  models/finite_gap_correction.json     physical-form correction (FiniteGapCorrection, used by Task 9)
  results/task7_wall_coefficients.csv   observed wall coefficient a per (morphology, rho*, a_z), phase mean/std
  results/task7_one_over_N.csv          per (theta, phase): fit of delta(N) by a/N vs a/N + b/N^2
  results/task7_model_cv.csv            leave-one-theta-out / leave-one-N-out CV: none, physical, physical+GP
  results/task7_graded.csv              graded strips: gradient effect relative to the uniform strip
  results/task7_convergence.csv         resolution check (n = 24 / 32 / 48)
  results/task7_wall_thickness.csv      wall-thickness check (t_w / L = 0.125 ... 1)
  results/task7_coefficients.txt        human-readable summary (copied into STATUS.md)

Examples
  python scripts/task7_fit_discrepancy.py              # production resolution = largest n with a complete uniform suite
  python scripts/task7_fit_discrepancy.py --n 32
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from voxlat.homogenization.finite_gap import (
    FEATURE_NAMES,
    PRIMARY_QUANTITIES,
    QUANTITY_TYPE,
    FiniteGapCorrection,
    PhysicalDiscrepancyModel,
    distinct_cuts,
    grouped_cv,
    reduced_phase,
)
from voxlat.homogenization.finite_gap_study import StudySettings, default_paths, study_design
from voxlat.utils import results_dir
from voxlat.utils.provenance import git_hash

Q = PRIMARY_QUANTITIES


def load_table(path: Path | None = None) -> pd.DataFrame:
    from voxlat.closures.dataset import read_table

    p = Path(path) if path is not None else default_paths()["table"]
    df = read_table(p) if p.exists() else read_table(p.with_suffix(".parquet"))
    return df[df["status"] == "ok"].reset_index(drop=True)


def production_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rows used for fitting: resolution n, wall 0.5 L (uniform, stretch, graded), one row per
    symmetry-distinct cut (``distinct_cuts``: for G/D the cuts 1/3 and 2/3 are the same strip)."""
    return distinct_cuts(df[(df["n"] == n) & (df["wall"] == 0.5)])


def wall_coefficient_obs(q: str, delta: np.ndarray, N: np.ndarray) -> np.ndarray:
    if QUANTITY_TYPE[q] == "series":
        return N * (1.0 / (1.0 + delta) - 1.0)
    return -N * delta


def one_over_N_check(df: pd.DataFrame) -> pd.DataFrame:
    """Per (theta, phase) group: rms residual in delta of the 1-parameter (a/N) and 2-parameter fits."""
    rows = []
    uni = df[df["gradient"] == 0.0]
    for key, g in uni.groupby(["w", "rho", "a_z", "phase"]):
        if g["N"].nunique() < 3:
            continue
        N = g["N"].to_numpy(float)
        for q in Q:
            d = g[f"delta_{q}"].to_numpy(float)
            y = wall_coefficient_obs(q, d, N) / N  # = a/N (+ b/N^2)
            x1 = 1.0 / N
            a1 = float(np.sum(x1 * y) / np.sum(x1 * x1))
            X2 = np.stack([1.0 / N, 1.0 / N**2], axis=1)
            (a2, b2), *_ = np.linalg.lstsq(X2, y, rcond=None)

            def back(yhat: np.ndarray) -> np.ndarray:  # transformed -> delta
                return 1.0 / (1.0 + yhat) - 1.0 if QUANTITY_TYPE[q] == "series" else -yhat

            r1 = d - back(a1 * x1)
            r2 = d - back(X2 @ np.array([a2, b2]))
            rows.append(dict(w=key[0], rho=key[1], a_z=key[2], phase=key[3], quantity=q, a_1term=a1,
                             rms_1term=float(np.sqrt(np.mean(r1**2))), max_1term=float(np.max(np.abs(r1))),
                             a_2term=float(a2), b_2term=float(b2), rms_2term=float(np.sqrt(np.mean(r2**2))),
                             delta_N1=float(d[N == 1.0][0]) if (N == 1.0).any() else np.nan))
    return pd.DataFrame(rows)


def wall_coefficient_table(df: pd.DataFrame, pm: PhysicalDiscrepancyModel) -> pd.DataFrame:
    rows = []
    uni = df[df["gradient"] == 0.0]
    for (m, w, rho, az), g in uni.groupby(["morphology", "w", "rho", "a_z"]):
        row = dict(morphology=m, w=w, rho=rho, a_z=az, rows=len(g))
        for q in Q:
            a = wall_coefficient_obs(q, g[f"delta_{q}"].to_numpy(float), g["N"].to_numpy(float))
            # phase mean of a per N, then mean / spread over N and phases
            row[f"a_{q}"] = float(np.mean(a))
            row[f"a_{q}_phase_std"] = float(g.assign(a=a).groupby("N")["a"].std(ddof=1).mean())
            row[f"a_{q}_fit"] = float(pm.wall_coefficient(q, w, rho, az))
            d1 = g.loc[g["N"] == 1.0, f"delta_{q}"]
            row[f"delta_{q}_N1_mean"] = float(d1.mean()) if len(d1) else np.nan
            row[f"delta_{q}_N1_min"] = float(d1.min()) if len(d1) else np.nan
            row[f"delta_{q}_N1_max"] = float(d1.max()) if len(d1) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def graded_table(df: pd.DataFrame, pm: PhysicalDiscrepancyModel) -> pd.DataFrame:
    """Gradient effect: Delta_g = (1 + delta_graded) / (1 + delta_uniform(same N, rho_mid, phase)) - 1."""
    gr = df[df["gradient"] != 0.0]
    uni = df[(df["gradient"] == 0.0) & (df["a_z"] == 1.0)]
    rows = []
    for _, r in gr.iterrows():
        pr = float(reduced_phase(r["w"], r["phase"]))  # the uniform twin may be stored under an equivalent cut
        u = uni[(uni["w"] == r["w"]) & (np.isclose(uni["rho"], r["rho"])) & (uni["N"] == r["N"])
                & (np.isclose(uni["phase_r"], pr))]
        row = dict(id=r["id"], morphology=r["morphology"], w=r["w"], rho=r["rho"], N=r["N"], phase=r["phase"],
                   gradient=r["gradient"], rho_min=r["rho"] - r["gradient"] * r["N"] / 2,
                   rho_max=r["rho"] + r["gradient"] * r["N"] / 2)
        for q in Q:
            row[f"delta_{q}"] = r[f"delta_{q}"]
            row[f"Delta_{q}"] = ((1 + r[f"delta_{q}"]) / (1 + u[f"delta_{q}"].iloc[0]) - 1) if len(u) else np.nan
            row[f"pred_{q}"] = float(pm.predict(q, r["N"], r["w"], r["rho"], 1.0, r["gradient"]))
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=None, help="production resolution (default: auto)")
    ap.add_argument("--table", default=None)
    ap.add_argument("--no-gp", action="store_true", help="skip the GP comparison (fast)")
    ap.add_argument("--gradient-tol", type=float, default=0.02,
                    help="max |gradient effect| defining the recommended grading limit (default 2 %%)")
    args = ap.parse_args(argv)

    df = load_table(Path(args.table) if args.table else None)
    full_uniform = len(study_design(["uniform"], n=32)["uniform"])  # 231 strips per resolution
    if args.n is None:
        cand = [n for n, g in df[(df["gradient"] == 0) & (df["wall"] == 0.5) & (df["a_z"] == 1.0)].groupby("n")
                if len(g) >= full_uniform]
        n = max(cand) if cand else int(df["n"].mode().iloc[0])
    else:
        n = args.n
    prod = production_rows(df, n)
    print(f"production resolution n = {n}: {len(prod)} distinct strips "
          f"({(prod['gradient'] == 0).sum()} uniform incl. stretch, {(prod['gradient'] != 0).sum()} graded)")
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)

    pm = PhysicalDiscrepancyModel(Q).fit(prod)
    settings = StudySettings.from_config()
    meta = dict(n=n, rows=len(prod), fitted=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                git=git_hash(), settings_fingerprint=settings.fingerprint(), wall_over_L=0.5,
                form={"series": "q/q_bulk = (1 + c_g g^2) / (1 + a/N)", "parallel": "q/q_bulk = (1 + c_g g^2)(1 - a/N)"},
                quantities={q: QUANTITY_TYPE[q] for q in Q}, features=list(FEATURE_NAMES),
                validity="N >= 1, w in [0, 1], rho* in [0.25, 0.45] (fit; 0.2-0.5 graded), a_z in [0.75, 1.5], "
                         "|grad rho*| L <= 0.15, wall-bonded network TPMS")
    corr = FiniteGapCorrection(pm, meta)
    path = corr.save()
    print(f"saved {path}")

    lines = [f"Task 7 finite-gap correction (n = {n}, {len(prod)} strips, wall 0.5 L)", ""]
    lines.append("wall coefficient a(theta) = sum beta_j f_j,  f = " + ", ".join(FEATURE_NAMES))
    lines.append("  r = (rho*-0.35)/0.1, b = 4w(1-w)")
    for q in Q:
        m = pm.models[q]
        beta = ", ".join(f"{FEATURE_NAMES[j]}: {b:+.4f}" for j, b in enumerate(m.beta) if m.active[j])
        lines.append(f"  {q:5s} ({m.qtype:8s}) beta = [{beta}]; c_g = {m.c_g:+.2f}; "
                     f"phase scatter of a = {m.a_scatter:.4f}; rms residual of a = {m.rmse_a:.4f}")

    wt = wall_coefficient_table(prod, pm)
    wt.to_csv(out / "task7_wall_coefficients.csv", index=False)
    on = one_over_N_check(prod)
    on.to_csv(out / "task7_one_over_N.csv", index=False)
    lines += ["", "1/N form per (theta, phase) group: rms / max |delta residual| of a/N (and of a/N + b/N^2):"]
    for q in Q:
        s = on[on["quantity"] == q]
        lines.append(f"  {q:5s} a/N: rms {100 * s['rms_1term'].mean():.2f} pp (worst max {100 * s['max_1term'].max():.2f} pp); "
                     f"a/N + b/N^2: rms {100 * s['rms_2term'].mean():.2f} pp")

    lines += ["", "phase-mean delta at N = 1 / 2 / 3 [%] (uniform, a_z = 1):"]
    uni = prod[(prod["gradient"] == 0) & (prod["a_z"] == 1.0)]
    for (m, rho), g in uni.groupby(["morphology", "rho"]):
        parts = []
        for q in Q:
            v = [100 * g.loc[g["N"] == N, f"delta_{q}"].mean() for N in (1.0, 2.0, 3.0)]
            parts.append(f"{q} {v[0]:+.1f}/{v[1]:+.1f}/{v[2]:+.1f}")
        lines.append(f"  {m:5s} rho {rho:.2f}: " + "; ".join(parts))

    cv_rows = []
    for groups in ("theta", "N"):
        sub = (prod if groups == "theta" else prod[prod["gradient"] == 0]).reset_index(drop=True)
        pure = sub["w"].isin([0.0, 1.0]).to_numpy()
        strata = {"all": np.ones(len(sub), bool), "G/D": pure, "blend": ~pure}
        cv = grouped_cv(sub, Q, groups=groups, use_gp=not args.no_gp, restarts=1, strata=strata)
        for stratum, res in cv.items():
            for q, r in res.items():
                cv_rows.append(dict(split=groups, stratum=stratum, quantity=q, **r))
    cvdf = pd.DataFrame(cv_rows)
    cvdf.to_csv(out / "task7_model_cv.csv", index=False)
    lines += ["", "leave-one-group-out CV, rms error of delta [pp] (none = bulk closure uncorrected; "
              "phase floor = pooled scatter over cuts):"]
    for _, r in cvdf.iterrows():
        gp = f", physical+GP {100 * r['physical+gp']:5.2f}" if np.isfinite(r.get("physical+gp", np.nan)) else ""
        lines.append(f"  {r['split']:5s} {r['stratum']:5s} {r['quantity']:5s}: none {100 * r['none']:5.2f}, "
                     f"physical {100 * r['physical']:5.2f}{gp}, phase floor {100 * r['phase_floor']:5.2f}  (n = {r['n']})")

    gt = graded_table(prod, pm)
    if len(gt):
        gt.to_csv(out / "task7_graded.csv", index=False)
        lines += ["", "graded strips: gradient effect Delta_g = (1+delta_g)/(1+delta_uniform) - 1, phase mean [%]:"]
        agg = gt.groupby(["morphology", "N", "gradient"])[[f"Delta_{q}" for q in Q]].mean() * 100
        lines += ["  " + s for s in agg.round(2).to_string().splitlines()]
        for label, sel, qs in (("G/D, all quantities", gt["w"].isin([0.0, 1.0]), Q),
                               ("blend, k_n and K_t", ~gt["w"].isin([0.0, 1.0]), ("k_n", "K_t")),
                               ("blend, all quantities", ~gt["w"].isin([0.0, 1.0]), Q)):
            worst = gt[sel].groupby("gradient")[[f"Delta_{q}" for q in qs]].apply(lambda x: x.abs().max().max())
            ok = [g for g, v in worst.items() if v <= args.gradient_tol]
            rec = max(ok) if ok else None
            lines.append(f"  {label}: worst |Delta_g| " + ", ".join(f"g={g:g}: {100 * v:.2f} %" for g, v in worst.items())
                         + f"  -> largest g with |Delta_g| <= {100 * args.gradient_tol:.0f} %: {rec}")

    for name, mask in (("convergence", (df["wall"] == 0.5) & (df["gradient"] == 0) & (df["a_z"] == 1.0)
                        & (df["rho"] == 0.35) & (df["phase"] == 0.0) & df["w"].isin([0.0, 1.0])
                        & df["N"].isin([1.0, 2.0, 4.0])),
                       ("wall_thickness", (df["n"] == n) & (df["gradient"] == 0) & (df["a_z"] == 1.0)
                        & (df["rho"] == 0.35) & (df["phase"] == 0.0) & df["w"].isin([0.0, 1.0])
                        & df["N"].isin([1.0, 2.0]))):
        sub = df[mask].sort_values(["w", "N", "n", "wall"])
        cols = ["morphology", "N", "n", "wall"] + [f"delta_{q}" for q in Q]
        sub[cols].to_csv(out / f"task7_{name}.csv", index=False)
        lines += ["", f"{name} check (delta [%]):"]
        t = sub[cols].copy()
        for q in Q:
            t[f"delta_{q}"] = (100 * t[f"delta_{q}"]).round(2)
        lines += ["  " + s for s in t.to_string(index=False).splitlines()]

    txt = "\n".join(lines)
    (out / "task7_coefficients.txt").write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
