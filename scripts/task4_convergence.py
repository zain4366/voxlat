"""Task 4: grid convergence of the voxel Stokes permeability K (and tortuosity).

Cases: gyroid (w = 0) and diamond (w = 1) at rho* = 0.2, 0.35, 0.5 (network solids,
porosity 0.8 / 0.65 / 0.5), plus the blend w = 0.5 at rho* = 0.35 (trigonal, full tensor).
Resolutions n = 16, 24, 32, 48, 64 voxels per cell; every n is solved on the default
grid (offset 0) and on 3 seeded random rigid shifts of the TPMS, so alignment noise can
be separated from the systematic staircase error. Anchor: n = 96 (2 offsets) for G and D
at rho* = 0.35, used as an independent check of the reference values.

Reference per quantity: least-squares fit f(n) = f_inf + C/n (p = 1) to the
offset-averaged values at n >= 32; a free-order fit over n >= 24 is reported (p_fit).
Estimator errors are evaluated on offset-0 grids (what one production run gives).
Quantity for the blend: mean eigenvalue tr(K)/3 and the anisotropy.

Run:  python scripts/task4_convergence.py [--quick] [--no-anchor]   (~11 min on 2 cores; --quick ~15 s)
      python scripts/task4_convergence.py --replot     (figure + summary from the saved CSV)
Out:  results/task4_convergence.csv            one row per (case, n, offset)
      results/task4_convergence_summary.csv    references, p_fit, n96 check, estimator errors, noise
      results/figures/task4_convergence.png
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from voxlat.geometry import TPMSParams
from voxlat.homogenization.convergence import fit_convergence, richardson
from voxlat.homogenization.elasticity import grid_offsets
from voxlat.homogenization.stokes import permeability_tpms
from voxlat.utils import get_logger, results_dir, save_figure

LOG = get_logger("task4.convergence")
N_ALL = (16, 24, 32, 48, 64)
N_REF = (32, 48, 64)
ESTIMATORS = {
    "raw n=32": (32,), "raw n=48": (48,), "raw n=64": (64,),
    "R(16,32)": (16, 32), "R(24,48)": (24, 48), "R(32,64)": (32, 64), "R(48,64)": (48, 64),
}
COLORS = {0.0: "#1f5aa6", 1.0: "#c4581a", 0.5: "#7a3fa0"}
LABEL = {0.0: "G", 1.0: "D", 0.5: "w=0.5"}


def _one(w: float, rho: float, n: int, j: int, off: tuple[float, float, float]) -> dict:
    r = permeability_tpms(TPMSParams(w=w, rho=rho), n, offset=off)
    ev = r.eigenvalues
    return {
        "w": w, "rho": rho, "n": n, "offset_id": j,
        "K_mean": r.mean, "K_min": ev[0], "K_max": ev[-1], "anisotropy": r.anisotropy,
        "K_xx": r.K[0, 0], "K_xy": r.K[0, 1],
        "tortuosity": float(np.nanmean(list(r.tortuosity.values()))),
        "porosity": r.porosity, "dofs": r.n_dofs, "iters": max(r.iterations),
        "residual": max(r.residuals), "agreement": r.agreement, "time_s": r.wall_time,
        "n_solves": len(r.directions),
    }


def summarize(df: pd.DataFrame, n_ref: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for (w, rho), g in df[df.n < 96].groupby(["w", "rho"]):
        avg = g.groupby("n")[["K_mean", "tortuosity"]].mean()
        std = g.groupby("n")[["K_mean", "tortuosity"]].std(ddof=1)
        off0 = g[g.offset_id == 0].set_index("n")
        for q in ("K_mean", "tortuosity"):
            ns = [n for n in n_ref if n in avg.index]
            ref = fit_convergence(ns, avg.loc[ns, q].values, p=1.0)
            nf = [n for n in avg.index if n >= 24]
            p_fit = fit_convergence(nf, avg.loc[nf, q].values, p=None).p if len(nf) >= 4 else np.nan
            row = {"w": w, "rho": rho, "quantity": q, "ref": ref.f_inf, "ref_std": ref.f_inf_std,
                   "slope_C": ref.C, "p_fit": p_fit}
            anchor = df[(df.w == w) & (df.rho == rho) & (df.n == 96)]
            if len(anchor):
                row["n96_avg_vs_fit"] = anchor[q].mean() / ref.predict(96) - 1
            for n in avg.index:
                row[f"avg_err_n{n}"] = avg.loc[n, q] / ref.f_inf - 1
                row[f"noise_n{n}"] = std.loc[n, q] / ref.f_inf
            for name, ns_e in ESTIMATORS.items():
                if not all(n in off0.index for n in ns_e):
                    continue
                if len(ns_e) == 1:
                    est = off0.loc[ns_e[0], q]
                else:
                    est = richardson(ns_e[0], off0.loc[ns_e[0], q], ns_e[1], off0.loc[ns_e[1], q], p=1)
                row[f"err {name}"] = est / ref.f_inf - 1
            if 48 in avg.index:
                row["err avg3 n=48"] = g[(g.n == 48) & (g.offset_id < 3)][q].mean() / ref.f_inf - 1
            rows.append(row)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, summ: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    for (w, rho), g in df[df.n < 96].groupby(["w", "rho"]):
        s = summ[(summ.w == w) & (summ.rho == rho) & (summ.quantity == "K_mean")].iloc[0]
        avg = g.groupby("n")["K_mean"].mean() / s.ref
        o0 = g[g.offset_id == 0].set_index("n")["K_mean"] / s.ref
        alpha = {0.2: 0.55, 0.35: 0.8, 0.45: 0.8, 0.5: 1.0}[rho]
        lab = f"{LABEL[w]} ρ*={rho:.2f}"
        ax.plot(1 / avg.index.values, avg.values, "-s", ms=4, color=COLORS[w], alpha=alpha, label=lab)
        ax.plot(1 / o0.index.values, o0.values, ":o", ms=3, color=COLORS[w], alpha=alpha)
        a96 = df[(df.w == w) & (df.rho == rho) & (df.n == 96)]
        if len(a96):
            ax.plot([1 / 96], [a96.K_mean.mean() / s.ref], "*", ms=10, color=COLORS[w], mec="k", mew=0.5)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xlim(0, None)
    ax.set_xlabel("1 / n   (n = voxels per cell)")
    ax.set_ylabel(r"$K(n)/K_{ref}$   (mean principal permeability)")
    ax.set_title("K: offset-averaged (solid), offset 0 (dotted), n = 96 (★)", fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    est = ["raw n=32", "raw n=48", "raw n=64", "avg3 n=48", "R(24,48)", "R(32,64)"]
    sk = summ[summ.quantity == "K_mean"]
    xs = np.arange(len(est))
    width = 0.8 / len(sk)
    for i, (_, s) in enumerate(sk.iterrows()):
        vals = [abs(s.get(f"err {e}", np.nan)) * 100 for e in est]
        ax.bar(xs + (i - len(sk) / 2 + 0.5) * width, vals, width, color=COLORS[s.w],
               alpha={0.2: 0.5, 0.35: 0.75, 0.45: 0.75, 0.5: 1.0}[s.rho],
               label=f"{LABEL[s.w]} ρ*={s.rho:.2f}")
    ax.set_xticks(xs, est, rotation=20)
    ax.set_ylabel("|error| vs reference  [%]")
    ax.set_title("Estimator error, K (offset-0 grids)", fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[2]
    for (w, rho), g in df[df.n < 96].groupby(["w", "rho"]):
        s = summ[(summ.w == w) & (summ.rho == rho) & (summ.quantity == "tortuosity")].iloc[0]
        avg = g.groupby("n")["tortuosity"].mean()
        alpha = {0.2: 0.55, 0.35: 0.8, 0.45: 0.8, 0.5: 1.0}[rho]
        ax.plot(avg.index.values, avg.values, "-o", ms=4, color=COLORS[w], alpha=alpha,
                label=f"{LABEL[w]} ρ*={rho:.2f}")
    ax.set_xlabel("n")
    ax.set_ylabel(r"hydraulic tortuosity  $\langle|u|\rangle/|\langle u\rangle|$")
    ax.set_title("Tortuosity, offset-averaged", fontsize=10)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.45 * (hi - lo))  # room for the legend above the curves
    ax.legend(fontsize=7, ncol=3, loc="upper center")
    fig.tight_layout()
    path = save_figure(fig, "task4_convergence")
    LOG.info("figure -> %s", path)


def replot() -> None:
    df = pd.read_csv(results_dir() / "task4_convergence.csv")
    summ = summarize(df, N_REF)
    summ.to_csv(results_dir() / "task4_convergence_summary.csv", index=False)
    plot(df, summ)


def main(quick: bool = False, anchor: bool = True) -> None:
    if quick:
        cases = [(0.0, 0.35)]
        n_all, n_ref, n_off = (16, 24, 32), (24, 32), {16: 2, 24: 2, 32: 2}
        anchor = False
    else:
        cases = [(w, r) for w in (0.0, 1.0) for r in (0.2, 0.35, 0.5)] + [(0.5, 0.35)]
        n_all, n_ref = N_ALL, N_REF
        n_off = {n: 4 for n in n_all}
    offs = grid_offsets(max(n_off.values()))
    jobs = [(c, n, j, tuple(offs[j])) for c in cases for n in n_all for j in range(n_off[n])]
    if anchor:
        jobs += [((w, 0.35), 96, j, tuple(offs[j])) for w in (0.0, 1.0) for j in range(2)]
    jobs.sort(key=lambda t: -(t[1] ** 3) * (3 if t[0][0] == 0.5 else 1))  # big first
    LOG.info("%d solves", len(jobs))
    rows = Parallel(n_jobs=-1, verbose=5)(
        delayed(_one)(c[0], c[1], n, j, off) for c, n, j, off in jobs
    )
    df = pd.DataFrame(rows).sort_values(["w", "rho", "n", "offset_id"])
    out = results_dir()
    df.to_csv(out / "task4_convergence.csv", index=False)
    summ = summarize(df, n_ref)
    summ.to_csv(out / "task4_convergence_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 4):
        cols = ["w", "rho", "quantity", "ref", "p_fit"] + [c for c in summ.columns if c.startswith("err")]
        cols += [c for c in ("n96_avg_vs_fit",) if c in summ.columns]
        print(summ[cols])
        print(df.groupby("n")[["time_s", "iters"]].max())
    plot(df, summ)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-anchor", action="store_true")
    ap.add_argument("--replot", action="store_true")
    a = ap.parse_args()
    replot() if a.replot else main(a.quick, not a.no_anchor)
