"""Task 1: build the precomputed TPMS tables shipped with the package.

Run:  python scripts/task1_build_tables.py [--n 48] [--jobs -1] [--thresholds-only]
Out:  src/voxlat/geometry/data/tpms_threshold_table.npz   (c(w, rho), network + sheet; ~10 s)
      src/voxlat/geometry/data/tpms_morphology_table.npz  (wall / pore / a_sf vs (w, rho); ~10 min on 1 core)

Both files are committed to the repo, so you only need to re-run this if the
level-set definitions or the thickness algorithm change.
"""

from __future__ import annotations

import argparse
import time

from voxlat.geometry.tpms import (
    MORPHOLOGY_TABLE_FILE,
    THRESHOLD_TABLE_FILE,
    build_morphology_table,
    save_threshold_tables,
)
from voxlat.utils import get_logger


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=48, help="voxels per cell for the morphology table")
    ap.add_argument("--n-ref", type=int, default=128, help="voxels per cell for the threshold table")
    ap.add_argument("--jobs", type=int, default=-1, help="joblib workers (-1 = all cores)")
    ap.add_argument("--thresholds-only", action="store_true")
    args = ap.parse_args()
    log = get_logger("task1.tables")

    t0 = time.perf_counter()
    path = save_threshold_tables(THRESHOLD_TABLE_FILE, n_ref=args.n_ref)
    log.info("threshold table -> %s (%.1f s)", path, time.perf_counter() - t0)
    if args.thresholds_only:
        return
    t0 = time.perf_counter()
    tab = build_morphology_table(n=args.n, n_jobs=args.jobs, path=MORPHOLOGY_TABLE_FILE)
    log.info("morphology table %s x %s at n=%d -> %s (%.0f s)",
             len(tab.w), len(tab.rho), tab.n, MORPHOLOGY_TABLE_FILE, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
