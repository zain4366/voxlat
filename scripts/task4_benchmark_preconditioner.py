"""Task 4: time the Stokes solver with the Jacobi and the AMG (pyamg) preconditioner.

The default ``preconditioner="auto"`` picks AMG whenever pyamg is installed. This script
checks that AMG is actually faster on this machine and that both give the same K
(expected agreement ~1e-5 of K or better; the discretization error is ~1e-2).
If AMG turns out slower, pass ``preconditioner="jacobi"`` in the Task 5 dataset script.

Run:  python scripts/task4_benchmark_preconditioner.py [--quick]   (~3-5 min; --quick ~30 s)
Out:  printed table + results/task4_benchmark_preconditioner.csv
"""

from __future__ import annotations

import argparse
import platform

import numpy as np
import pandas as pd

from voxlat.geometry import TPMSParams
from voxlat.homogenization.stokes import permeability_tpms, pyamg_available
from voxlat.utils import results_dir

CASES = [  # (label, w, rho, n)
    ("gyroid", 0.0, 0.3, 48),
    ("gyroid", 0.0, 0.3, 64),
    ("diamond", 1.0, 0.3, 64),
    ("blend", 0.5, 0.35, 48),  # trigonal: three body-force directions
]


def main(quick: bool = False) -> None:
    if not pyamg_available():
        print("pyamg is not installed: only Jacobi is available (pip install -e \".[amg]\").")
    pcs = ["jacobi"] + (["amg"] if pyamg_available() else [])
    cases = [("gyroid", 0.0, 0.3, 32), ("blend", 0.5, 0.35, 24)] if quick else CASES
    rows = []
    for lab, w, rho, n in cases:
        res = {}
        for pc in pcs:
            r = permeability_tpms(TPMSParams(w=w, rho=rho), n, preconditioner=pc)
            res[pc] = r
            rows.append({
                "case": lab, "w": w, "rho": rho, "n": n, "preconditioner": pc,
                "directions": len(r.directions), "time_s": round(r.wall_time, 2),
                "iterations_max": max(r.iterations), "true_residual_max": max(r.residuals),
                "K_mean": r.mean,
            })
            print(f"{lab:8s} n={n:3d} {pc:7s}: {r.wall_time:7.1f} s, iterations {r.iterations}, "
                  f"K/L^2 = {r.mean:.6e}", flush=True)
        if "amg" in res:
            dK = np.max(np.abs(res["amg"].K - res["jacobi"].K)) / res["jacobi"].mean
            speed = res["jacobi"].wall_time / res["amg"].wall_time
            print(f"{'':8s} -> AMG {speed:.2f}x the speed of Jacobi, max|dK| = {dK:.1e} of K\n", flush=True)
            rows[-1]["speedup_vs_jacobi"] = speed
            rows[-1]["K_diff_rel"] = dK
    df = pd.DataFrame(rows)
    out = results_dir() / "task4_benchmark_preconditioner.csv"
    df.to_csv(out, index=False)
    print(f"machine: {platform.platform()}, {platform.processor()}")
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
