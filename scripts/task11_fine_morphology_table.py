"""Task 11: build the fine-w TPMS morphology table (w = 0:0.025:1 x rho* = 0.15:0.025:0.55, n = 48).

The Task 1 table steps w by 0.125, which interpolates over the pinch-neck spikes of L_min for blends; the
optimizer found designs there that fail a direct manufacturability check (STATUS Task 11). This table (same
method and resolution as Task 1, 5x finer in w) is used by scripts/task11_analysis.py to re-screen every
evaluated design. Output: src/voxlat/geometry/data/tpms_morphology_table_fine.npz (~0.1 MB).
Runtime: 697 points x ~3.5 s / n_jobs (~25-40 min on 2 cores).

    python scripts/task11_fine_morphology_table.py [--n-jobs 4]
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

import numpy as np

from voxlat.geometry.tpms import FINE_MORPHOLOGY_TABLE_FILE, build_morphology_table


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    t = time.time()
    w = np.round(np.arange(0, 1.0001, 0.025), 10)
    rho = np.round(np.arange(0.15, 0.5501, 0.025), 10)
    build_morphology_table(w_grid=w, rho_grid=rho, n=48, n_jobs=a.n_jobs, path=FINE_MORPHOLOGY_TABLE_FILE, progress=True)
    print(f"{len(w)} x {len(rho)} table in {time.time() - t:.0f} s -> {FINE_MORPHOLOGY_TABLE_FILE}")


if __name__ == "__main__":
    sys.exit(main())
