"""Task 7: finite-gap (scale-separation) study - compute the strip suites (resumable).

Builds TPMS lattice strips of N cells between solid walls (voxlat.homogenization.finite_gap),
solves them with the Task 2-4 solvers, compares with the bulk ("plain-wall")
homogenized prediction and stores one JSON part per strip in data/finite_gap/strips/
(bulk reference cells in data/finite_gap/bulk/). The consolidated table is
data/finite_gap.csv (+ .parquet if pyarrow is installed). Suites: see
voxlat.homogenization.finite_gap_study (uniform, graded, convergence, wall, stretch).

Workflow
  1. --estimate-only times one N = 2 strip per morphology at the chosen n and prints the
     runtime of everything still pending on this machine (the timed strips are kept).
  2. A normal run computes the bulk cells, then the strips (largest first) on --n-jobs
     workers. Every finished strip is saved at once: rerun the SAME command to resume.
  3. The table is (re)written at the end; --table-only just consolidates.

Examples (Windows PowerShell, repo root, venv active)
  python scripts/task7_finite_gap.py --quick                       # plumbing test, n = 16, ~1 min
  python scripts/task7_finite_gap.py --estimate-only --n 48        # time 3 strips, print the estimate
  python scripts/task7_finite_gap.py --suite all --n-jobs 2        # n = 32, everything (~1.5 h on 2 cores)
  python scripts/task7_finite_gap.py --suite uniform graded stretch --n 48 --n-jobs 4   # production n = 48 (~3-4 h)
  python scripts/task7_finite_gap.py --table-only

Then: python scripts/task7_fit_discrepancy.py  and  python scripts/task7_figures.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from voxlat.homogenization.finite_gap import PRIMARY_QUANTITIES, GapSpec
from voxlat.homogenization.finite_gap_study import (
    SUITES,
    StudySettings,
    bulk_requirements,
    compute_bulk,
    compute_strip,
    consolidate,
    default_paths,
    run_study,
    strip_cost,
    study_design,
    _read_json,
)
from voxlat.utils import get_logger

LOG = get_logger("task7.finite_gap")


def _pending(specs: list[GapSpec], paths: dict[str, Path], settings: StudySettings) -> list[GapSpec]:
    fp = settings.fingerprint()
    out = []
    for s in specs:
        r = _read_json(paths["strips"] / f"{s.case_id}.json")
        if r is None or r.get("fingerprint") != fp or r.get("status") != "ok":
            out.append(s)
    return out


def _estimate(specs: list[GapSpec], paths: dict[str, Path], settings: StudySettings, n_jobs: int) -> float:
    """Time one N = 2 strip per morphology (kept), extrapolate by voxel count."""
    pend = _pending(specs, paths, settings)
    if not pend:
        print("nothing pending")
        return 0.0
    per_morph: dict[str, float] = {}
    for w in sorted({s.w for s in pend}):
        cand = [s for s in pend if s.w == w and not s.graded and s.wall == 0.5 and s.a_z == 1.0]
        if not cand:
            cand = [s for s in pend if s.w == w]
        probe = min(cand, key=lambda s: abs(s.N - 2.0) + abs(s.phase))
        for k in bulk_requirements([probe]):
            compute_bulk(k, settings, paths["bulk"])
        t0 = time.perf_counter()
        rec = compute_strip(probe, settings, paths["bulk"], paths["strips"])
        dt = time.perf_counter() - t0
        per_morph[probe.morphology] = dt / strip_cost(probe)
        print(f"  timed {probe.case_id}: {dt:.1f} s ({rec['status']})")
    rate = {m: v for m, v in per_morph.items()}
    default_rate = float(np.mean(list(rate.values())))
    pend = _pending(specs, paths, settings)
    total = sum(strip_cost(s) * rate.get(s.morphology, default_rate) for s in pend)
    # bulk cells: ~ cost of a strip with N = 1 and no wall
    nb = len([k for k in bulk_requirements(pend)])
    total += nb * default_rate * (pend[0].n ** 3 if pend else 0)
    # 2 workers on 2 physical cores run ~1.3x slower each (memory bandwidth, Task 4)
    wall = total / max(1, n_jobs) * (1.3 if n_jobs > 1 else 1.0)
    print(f"pending: {len(pend)} strips + {nb} bulk cells; CPU time ~{total / 3600:.1f} h; "
          f"wall time on {n_jobs} worker(s) ~{wall / 3600:.1f} h (parallel overhead included)")
    return wall


def _summary(df) -> None:
    if df is None or len(df) == 0:
        return
    ok = df[(df["status"] == "ok") & (df["gradient"] == 0.0) & (df["wall"] == 0.5) & (df["a_z"] == 1.0)]
    if ok.empty:
        return
    cols = [f"delta_{q}" for q in PRIMARY_QUANTITIES]
    g = ok.groupby(["morphology", "rho", "N", "n"])[cols].mean() * 100
    with np.printoptions(precision=1, suppress=True):
        print("\nphase-mean discrepancy delta = q_strip/q_bulk - 1 [%]:")
        print(g.round(1).to_string())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", nargs="+", default=["all"], help=f"suites: {SUITES} or 'all'")
    ap.add_argument("--n", type=int, default=32, help="voxels per cell (default 32)")
    ap.add_argument("--n-jobs", type=int, default=1, help="parallel workers")
    ap.add_argument("--preconditioner", default="two_level", choices=["two_level", "auto", "amg", "jacobi"])
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny plumbing test (n = 16, 4 strips)")
    ap.add_argument("--force", action="store_true", help="recompute finished strips")
    ap.add_argument("--data-root", default=None, help="alternative data folder (default data/)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    settings = StudySettings.from_config(preconditioner=args.preconditioner)
    paths = default_paths(args.data_root)
    suites = list(SUITES) if "all" in args.suite else args.suite
    if args.quick:
        if args.data_root is None:
            paths = default_paths(Path("results") / "task7_quick")
        design = {"uniform": [GapSpec(w, 0.35, N, 0.0, n=16) for w in (0.0, 1.0) for N in (1.0, 2.0)]}
    else:
        design = study_design(suites, n=args.n)
    specs = list({s.case_id: s for v in design.values() for s in v}.values())
    print(f"Task 7 finite-gap study: suites {list(design)}, {len(specs)} strips, n = {args.n}, "
          f"settings {settings.fingerprint()}, preconditioner {settings.preconditioner}")
    print(f"parts: {paths['strips']}  table: {paths['table']}")

    if args.table_only:
        df = consolidate(study_design(SUITES, n=args.n), paths, settings)
        print(f"table: {len(df)} rows -> {paths['table']}")
        _summary(df)
        return 0
    if args.estimate_only:
        _estimate(specs, paths, settings, args.n_jobs)
        return 0

    t0 = time.perf_counter()
    recs = run_study(specs, settings, n_jobs=args.n_jobs, paths=paths, force=args.force)
    bad = [r for r in recs if r.get("status") != "ok"]
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min: {len(recs) - len(bad)} ok, {len(bad)} failed")
    for r in bad[:5]:
        print("  FAILED", r["id"], r.get("error", "")[:300])
    full = study_design(SUITES, n=args.n) if not args.quick else design
    if not args.quick:  # also label strips from other suites/resolutions already on disk
        for m in (24, 32, 48):
            if m != args.n:
                for k, v in study_design(SUITES, n=m).items():
                    full.setdefault(k, []).extend(v)
    df = consolidate(full, paths, settings)
    print(f"table: {len(df)} rows -> {paths['table']}")
    _summary(df)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
