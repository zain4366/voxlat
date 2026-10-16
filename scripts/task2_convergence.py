"""Task 2: grid convergence of k_eff for gyroid / diamond, with Richardson estimates.

Two sequences per case (w, rho, k_f/k_s):
  A  offset 0 (the default grid), n = 24, 32, 40, 48, 56, 64, full tensor
     -> what a single production run gives, and two-grid Richardson estimators.
  B  averaged over 3 grid offsets (0 + 2 seeded random shifts), n = 24, 48, 96
     -> observed order p from the constant-ratio triple and the reference
        value k_ref = Richardson(48, 96, p = 1). Offset averaging removes most of
        the grid-alignment noise, so the observed order is clean.

Run:  python scripts/task2_convergence.py [--quick]     (~15 min full on 2 cores; --quick ~1 min)
Out:  results/task2_convergence.csv            one row per (case, sequence, n)
      results/task2_convergence_summary.csv    estimator errors vs k_ref
      results/figures/task2_convergence.png
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
from voxlat.homogenization.conduction import effective_conductivity_tpms
from voxlat.homogenization.convergence import fit_convergence, observed_order, richardson
from voxlat.utils import get_logger, load_config, results_dir, save_figure, set_seed

LOG = get_logger("task2.convergence")
N_A = (24, 32, 40, 48, 56, 64)
N_B = (24, 48, 96)
N_OFFSETS = 3
ESTIMATORS = {  # name -> (n1, n2) two-grid Richardson with p = 1, or (n,) raw
    "raw n=32": (32,),
    "raw n=48": (48,),
    "raw n=64": (64,),
    "R(24,48)": (24, 48),
    "R(32,48)": (32, 48),
    "R(32,64)": (32, 64),
    "R(48,64)": (48, 64),
}


def _offsets(seed: int = 2026) -> np.ndarray:
    rng = set_seed(seed)
    off = rng.random((N_OFFSETS, 3))
    off[0] = 0.0
    return off


def _one(w: float, rho: float, kf_ratio: float, n: int, seq: str, j: int, off) -> dict:
    cfg = load_config()
    k_s = cfg.material.conductivity
    r = effective_conductivity_tpms(TPMSParams(w=w, rho=rho), n, k_s, kf_ratio * k_s, offset=off)
    return {
        "w": w, "rho": rho, "kf_ks": kf_ratio, "seq": seq, "n": n, "offset_id": j,
        "k_mean_rel": r.mean / k_s, "anisotropy": r.anisotropy,
        "agreement": r.agreement, "iters": max(r.iterations), "time_s": r.wall_time,
        "solid_fraction": r.solid_fraction,
    }


def main(quick: bool = False) -> None:
    cfg = load_config()
    kf_cfg = cfg.coolant.conductivity / cfg.material.conductivity
    if quick:
        cases = [(0.0, 0.3, kf_cfg)]
        n_a, n_b = (24, 32, 48), (16, 32, 64)
    else:
        cases = [(w, rho, kf_cfg) for w, rho in itertools.product((0.0, 1.0), (0.2, 0.35, 0.5))]
        cases += [(0.0, 0.35, 0.0), (1.0, 0.35, 0.0)]
        n_a, n_b = N_A, N_B
    offs = _offsets()
    jobs = [(c, n, "A", 0, None) for c in cases for n in n_a]
    jobs += [(c, n, "B", j, tuple(offs[j])) for c in cases for n in n_b for j in range(N_OFFSETS)]
    jobs.sort(key=lambda t: -t[1])  # big grids first for load balance
    LOG.info("%d solves", len(jobs))
    rows = Parallel(n_jobs=-1, verbose=5)(
        delayed(_one)(c[0], c[1], c[2], n, s, j, o) for c, n, s, j, o in jobs
    )
    df = pd.DataFrame(rows).sort_values(["w", "rho", "kf_ks", "seq", "n", "offset_id"])
    out = results_dir()
    df.to_csv(out / "task2_convergence.csv", index=False)

    summ = []
    for (w, rho, kfr), g in df.groupby(["w", "rho", "kf_ks"]):
        A = g[g.seq == "A"].set_index("n").k_mean_rel
        B = g[g.seq == "B"].groupby("n").k_mean_rel.mean()
        nb = list(B.index)
        p_obs = observed_order(nb, [B[k] for k in nb])
        k_ref = richardson(nb[1], B[nb[1]], nb[2], B[nb[2]], p=1.0)
        k_ref_p = richardson(nb[1], B[nb[1]], nb[2], B[nb[2]], p=p_obs) if np.isfinite(p_obs) else np.nan
        fitA = fit_convergence(A.index.values, A.values, p=1.0)
        row = {"w": w, "rho": rho, "kf_ks": kfr, "p_observed": p_obs, "k_ref": k_ref,
               "k_ref_obs_p": k_ref_p, "k_fitA_p1": fitA.f_inf,
               "noise_rms_A": fitA.residual_rms / k_ref}
        for name, ns in ESTIMATORS.items():
            if not all(n in A.index for n in ns):
                continue
            est = A[ns[0]] if len(ns) == 1 else richardson(ns[0], A[ns[0]], ns[1], A[ns[1]], 1.0)
            row[f"err {name}"] = est / k_ref - 1.0
        summ.append(row)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(out / "task2_convergence_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 4):
        print(sdf)

    # ---- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    colors = {0.0: "#1f5aa6", 1.0: "#c4581a"}
    for (w, rho, kfr), g in df.groupby(["w", "rho", "kf_ks"]):
        s = sdf[(sdf.w == w) & (sdf.rho == rho) & (sdf.kf_ks == kfr)].iloc[0]
        A = g[g.seq == "A"].set_index("n").k_mean_rel
        B = g[g.seq == "B"].groupby("n").k_mean_rel.mean()
        ls = "-" if kfr > 0 else ":"
        lab = f"{'G' if w == 0 else 'D'} rho={rho:.2f}" + ("" if kfr > 0 else " k_f=0")
        axes[0].plot(1 / A.index.values, A.values / s.k_ref, "o" + ls, ms=3.5, color=colors[w],
                     alpha=0.35 + rho, label=lab)
        axes[1].loglog(B.index.values, 1 - B.values / s.k_ref, "s" + ls, ms=4, color=colors[w],
                       alpha=0.35 + rho, label=lab)
    axes[0].axhline(1.0, color="k", lw=0.8)
    axes[0].set_xlabel("1 / n   (n = voxels per cell)")
    axes[0].set_ylabel(r"$\bar k_{eff}(n)\, /\, k_{ref}$")
    axes[0].set_title("Offset-0 grids: first-order (linear in 1/n) + alignment noise")
    axes[0].set_xlim(left=0)
    nn = np.array([20, 110])
    axes[1].loglog(nn, 1.5 / nn, "k--", lw=0.8, label="slope -1")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel(r"$1 - \bar k_{eff}(n)/k_{ref}$")
    axes[1].set_title(f"Offset-averaged ({N_OFFSETS} shifts): observed order")
    axes[1].legend(fontsize=7, ncol=2)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save_figure(fig, "task2_convergence")
    LOG.info("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
