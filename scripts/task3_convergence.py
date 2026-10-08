"""Task 3: grid convergence of the voxel-FEA stiffness C_eff and of the stress localization.

Cases: gyroid (w = 0) and diamond (w = 1) at rho* = 0.2, 0.35, 0.5 (network solids).
Resolutions n = 16, 24, 32, 40, 48, 64 voxels per cell; every n is solved on the
default grid (offset 0) plus 3 seeded random rigid shifts of the TPMS against the grid,
so alignment noise can be separated from the systematic staircase error.
(n = 96 is checked separately for two cases, scripts/task3_n96_anchor.py: the
assembled matrix needs ~1-2 GB there.)

Reference value per quantity: least-squares fit f(n) = f_inf + C/n (p = 1, as in
Task 2) of the offset-averaged values at n >= 32. A free-order fit over the same
points is reported as a check (p_fit). Estimator errors are evaluated on the
offset-0 grids (what one production run gives).

Run:  python scripts/task3_convergence.py [--quick]     (~30 min on 2 cores; --quick ~10 s)
Out:  results/task3_convergence.csv            one row per (case, n, offset)
      results/task3_convergence_summary.csv    reference values, p_fit, estimator errors, noise
      results/figures/task3_convergence.png
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

from voxlat.geometry import TPMSParams
from voxlat.homogenization.convergence import fit_convergence, richardson
from voxlat.homogenization.elasticity import effective_elasticity_tpms
from voxlat.utils import get_logger, load_config, results_dir, save_figure, set_seed

LOG = get_logger("task3.convergence")
N_ALL = (16, 24, 32, 40, 48, 64)
N_REF = (32, 40, 48, 64)
QUANT = ["E_axial", "C11c", "C12c", "C44c", "zener", "K_bulk"]
LOCQ = ["loc_uniaxial_z_max", "loc_uniaxial_z_p99", "loc_uniaxial_z_mean",
        "loc_shear_xz_max", "loc_shear_xz_p99", "loc_shear_xz_mean"]
ESTIMATORS = {
    "raw n=24": (24,), "raw n=32": (32,), "raw n=40": (40,), "raw n=48": (48,), "raw n=64": (64,),
    "R(24,48)": (24, 48), "R(32,64)": (32, 64), "R(16,32)": (16, 32),
}


def _offsets(k: int, seed: int = 2026) -> np.ndarray:
    rng = set_seed(seed)
    off = rng.random((k, 3))
    off[0] = 0.0
    return off


def _one(w: float, rho: float, n: int, j: int, off) -> dict:
    cfg = load_config()
    E = cfg.material.youngs_modulus
    r = effective_elasticity_tpms(TPMSParams(w=w, rho=rho), n, offset=off)
    row = {"w": w, "rho": rho, "n": n, "offset_id": j}
    row.update(r.as_row(E_ref=E))
    row.update({"solid_fraction": r.solid_fraction, "dofs": r.n_dofs, "time_s": r.wall_time,
                "solve_s": r.solve_time, "iters": max(r.iterations), "components": r.n_components})
    return row


def main(quick: bool = False) -> None:
    if quick:
        cases = [(0.0, 0.35)]
        n_all, n_ref, n_off = (16, 24, 32), (24, 32), {16: 2, 24: 2, 32: 2}
    else:
        cases = list(itertools.product((0.0, 1.0), (0.2, 0.35, 0.5)))
        n_all, n_ref = N_ALL, N_REF
        n_off = {n: 4 for n in n_all}
    offs = _offsets(max(n_off.values()))
    jobs = [(c, n, j, tuple(offs[j])) for c in cases for n in n_all for j in range(n_off[n])]
    jobs.sort(key=lambda t: -t[1])  # big grids first
    LOG.info("%d solves", len(jobs))
    rows = Parallel(n_jobs=2, verbose=5)(delayed(_one)(c[0], c[1], n, j, o) for c, n, j, o in jobs)
    df = pd.DataFrame(rows).sort_values(["w", "rho", "n", "offset_id"])
    out = results_dir()
    df.to_csv(out / "task3_convergence.csv", index=False)

    summ = []
    for (w, rho), g in df.groupby(["w", "rho"]):
        avg = g.groupby("n")[QUANT + LOCQ].mean()
        sd = g.groupby("n")[QUANT + LOCQ].std()
        o0 = g[g.offset_id == 0].set_index("n")
        nr = [n for n in n_ref if n in avg.index]
        for q in QUANT:
            fit = fit_convergence(nr, avg.loc[nr, q].values, p=1.0)
            try:
                p_fit = fit_convergence(nr, avg.loc[nr, q].values, p=None).p if len(nr) >= 4 else np.nan
            except Exception:  # noqa: BLE001
                p_fit = np.nan
            ref = fit.f_inf
            row = {"w": w, "rho": rho, "quantity": q, "ref": ref, "ref_std": fit.f_inf_std,
                   "p_fit": p_fit, "slope_C_rel": fit.C / ref,
                   # alignment noise: std over offsets / ref, at n = 32 and 48
                   "noise_n32": sd.loc[32, q] / abs(ref) if 32 in sd.index else np.nan,
                   "noise_n48": sd.loc[48, q] / abs(ref) if 48 in sd.index else np.nan}
            for name, ns in ESTIMATORS.items():
                if not all(n in o0.index for n in ns):
                    continue
                est = o0.loc[ns[0], q] if len(ns) == 1 else richardson(ns[0], o0.loc[ns[0], q],
                                                                       ns[1], o0.loc[ns[1], q], 1.0)
                row[f"err {name}"] = est / ref - 1.0
            for n in avg.index:
                row[f"avg_err n={n}"] = avg.loc[n, q] / ref - 1.0
            summ.append(row)
        for q in LOCQ:
            row = {"w": w, "rho": rho, "quantity": q, "ref": np.nan}
            for n in avg.index:
                row[f"avg n={n}"] = avg.loc[n, q]
                row[f"sd n={n}"] = sd.loc[n, q]
            summ.append(row)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(out / "task3_convergence_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 4):
        print(sdf[sdf.quantity.isin(QUANT)].drop(columns=[c for c in sdf.columns if c.startswith(("avg n", "sd n"))]))
        print(sdf[sdf.quantity.isin(LOCQ)][["w", "rho", "quantity"] + [c for c in sdf.columns if c.startswith("avg n")]])

    # ---- figure ---------------------------------------------------------------
    colors = {0.0: "#1f5aa6", 1.0: "#c4581a"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for (w, rho), g in df.groupby(["w", "rho"]):
        s = sdf[(sdf.w == w) & (sdf.rho == rho) & (sdf.quantity == "E_axial")].iloc[0]
        lab = f"{'G' if w == 0 else 'D'} ρ*={rho:.2f}"
        alpha = 0.35 + rho
        o0 = g[g.offset_id == 0].set_index("n").E_axial
        avg = g.groupby("n").E_axial.mean()
        axes[0].plot(1 / o0.index.values, o0.values / s.ref, "o:", ms=3.5, color=colors[w], alpha=alpha)
        axes[0].plot(1 / avg.index.values, avg.values / s.ref, "s-", ms=4.5, color=colors[w], alpha=alpha, label=lab)
        for q, ls in (("loc_uniaxial_z_p99", "-"), ("loc_uniaxial_z_max", ":")):
            a = g.groupby("n")[q].mean()
            axes[2].plot(a.index, a.values, "o" + ls, ms=3.5, color=colors[w], alpha=alpha,
                         label=lab if ls == "-" else None)
        a = g.groupby("n").zener.mean()
        sz = sdf[(sdf.w == w) & (sdf.rho == rho) & (sdf.quantity == "zener")].iloc[0]
        axes[1].plot(1 / a.index.values, a.values / sz.ref, "s-", ms=4, color=colors[w], alpha=alpha, label=lab)
    for ax in axes[:2]:
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xlim(left=0)
        ax.set_xlabel("1 / n   (n = voxels per cell)")
    axes[0].set_ylabel(r"$E^*(n)\,/\,E^*_{ref}$  (mean axial Young's modulus)")
    axes[0].set_title("E*: offset-0 grid (dotted) vs offset-averaged (solid)", fontsize=10)
    axes[1].set_ylabel(r"$A(n)\,/\,A_{ref}$  (Zener ratio)")
    axes[1].set_title("Zener ratio, offset-averaged", fontsize=10)
    axes[2].set_xlabel("n")
    axes[2].set_ylabel(r"$\sigma_{vm}/\Sigma_{vm}$ under uniaxial $\Sigma_{zz}$")
    axes[2].set_title("Localization: 99th percentile (solid) and max (dotted)", fontsize=10)
    axes[0].legend(fontsize=7.5, ncol=2)
    axes[2].legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    save_figure(fig, "task3_convergence")
    LOG.info("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
