"""Task 11: uniform-lattice references on BOTH manifold layouts (B2-style grid search).

Task 10's B2 is the uniform gyroid with slot manifolds. For the end-ring layout there is no
uniform reference, and the optimized designs lean toward diamond, so this script evaluates
uniform gyroid (w = 0) and uniform diamond (w = 1) over rho* 0.20:0.025:0.50 x L 2:0.25:6 mm on
both layouts (4 x 221 designs, ~6 min on 2 cores). Output: results/task11_uniform_refs.csv
(family U-G / U-D, layout, manufacturable/margin flags) - used by task11_analysis.py.

    python scripts/task11_uniform_references.py [--n-jobs 2]
"""

from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

from voxlat.utils import results_dir


def _grid(layout: str, w: float, rhos: list, Ls: list) -> list[dict]:
    warnings.simplefilter("ignore")
    from voxlat.device import JacketModel, JacketOptions

    m = JacketModel(options=JacketOptions(check_bounds="ignore", manifold_layout=layout))
    rows = []
    for r in rhos:
        for L in Ls:
            o = m.evaluate(r, w, L)
            rows.append({"layout": layout, "family": "U-G" if w == 0 else "U-D", "rho": r, "w": w, "L": L,
                         "thermal_resistance": o["thermal_resistance"],
                         "R_active": (o["T_wall_max_lattice"] - o["T_in"]) / m.cfg.motor.heat_to_jacket,
                         "pump_power": o["pump_power"], "delta_p": o["delta_p"], "mass": o["mass"],
                         "min_structural_margin": o["min_structural_margin"], "manufacturable": o["manufacturable"]})
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args(argv)
    from joblib import Parallel, delayed

    rhos = list(np.round(np.arange(0.20, 0.5001, 0.025), 4))
    Ls = list(np.round(np.arange(2e-3, 6e-3 + 1e-12, 0.25e-3), 6))
    jobs = [(lay, w, rhos[i::2]) for lay in ("slots", "end_rings") for w in (0.0, 1.0) for i in range(2)]
    out = Parallel(n_jobs=a.n_jobs)(delayed(_grid)(lay, w, rs, Ls) for lay, w, rs in jobs)
    df = pd.DataFrame([r for rows in out for r in rows]).sort_values(["layout", "family", "rho", "L"])
    df["feasible"] = df["manufacturable"].astype(bool) & (df["min_structural_margin"] >= 1.0)
    df.to_csv(results_dir() / "task11_uniform_refs.csv", index=False, float_format="%.8g")
    print(df.groupby(["layout", "family"])["feasible"].agg(["size", "sum"]))


if __name__ == "__main__":
    sys.exit(main())
