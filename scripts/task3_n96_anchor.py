"""Task 3: n = 96 anchor points for the convergence study (gyroid and diamond, rho* = 0.35).

Two grid offsets (0 and the first random shift of scripts/task3_convergence.py) at
n = 96, assembled matrix (~1 GB, ~2 GB peak during assembly), one process at a time.
Used to check that the n <= 64 reference fit (f_inf + C/n) extrapolates correctly.

Run:  python scripts/task3_n96_anchor.py          (~20 min on 1 core, ~2 GB RAM)
Out:  results/task3_n96_anchor.csv
"""

from __future__ import annotations

import pandas as pd

from voxlat.geometry import TPMSParams
from voxlat.homogenization.elasticity import effective_elasticity_tpms, grid_offsets
from voxlat.utils import get_logger, load_config, results_dir

LOG = get_logger("task3.n96")


def main() -> None:
    E = load_config().material.youngs_modulus
    offs = grid_offsets(4)  # same seed/sequence as task3_convergence.py
    rows = []
    for w in (0.0, 1.0):
        for j in (0, 1):
            r = effective_elasticity_tpms(TPMSParams(w=w, rho=0.35), 96, offset=tuple(offs[j]),
                                          operator="assembled")
            row = {"w": w, "rho": 0.35, "n": 96, "offset_id": j, **r.as_row(E_ref=E),
                   "dofs": r.n_dofs, "time_s": r.wall_time, "iters": max(r.iterations)}
            LOG.info("w=%.0f offset %d: E*/E_s = %.5f, %.0f s", w, j, row["E_axial"], r.wall_time)
            rows.append(row)
    pd.DataFrame(rows).to_csv(results_dir() / "task3_n96_anchor.csv", index=False)


if __name__ == "__main__":
    main()
