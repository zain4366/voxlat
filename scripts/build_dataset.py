"""Task 5: build the closure dataset data/closures.parquet (overnight run, resumable).

For every design point theta = (w, rho*, a_z) - 256 scrambled-Sobol points + 8 corners
+ 12 edge midpoints + 14 pure-line points (w = 0/1, a_z = 1) by default - compute the
Task 1 geometry metrics and the production closures of Tasks 2-4 (two-grid Richardson
R(32, 64)): k_eff/k_s, C_eff/E_s + stress localization (6 unit stresses), K/L^2.
Details, column meanings and design choices: voxlat.closures.dataset and STATUS.md (Task 5).

Workflow
  1. Times 3 representative pending samples on one core (they are real samples and are
     kept) and prints a runtime estimate for this machine.
  2. Runs the remaining samples on --n-jobs workers, longest first. Every finished sample
     is saved at once to data/closures_parts/<id>.json, so Ctrl+C / a crash / a reboot
     loses at most the samples in progress: rerun the SAME command to resume.
  3. Consolidates everything into data/closures.parquet and draws
     results/figures/task5_closures_quicklook.png.

Examples (Windows PowerShell, repo root, venv active)
  python scripts/build_dataset.py --estimate-only            # time 3 samples, print estimate, stop
  python scripts/build_dataset.py --n-jobs 4                 # full overnight run (resumable)
  python scripts/build_dataset.py --n-sobol 8 --no-boundary --n-jobs 2   # small trial (reused later)
  python scripts/build_dataset.py --plot-only                # redraw the figure from the table
  python scripts/build_dataset.py --quick                    # 4 samples at R(8, 16): plumbing test, ~10 s

A log of every run is appended to data/closures_build.log (next to the table; parts in
data/closures_parts/ - both derive from --out).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from voxlat.closures.dataset import (
    DEFAULT_DESIGN_SEED,
    DatasetSettings,
    DesignSpace,
    SettingsMismatchError,
    build_design,
    consolidate,
    estimate_runtime,
    load_existing,
    parquet_available,
    pending_samples,
    pick_timing_samples,
    plot_quicklook,
    read_table,
    relative_cost,
    run_samples,
)
from voxlat.utils import data_dir, get_logger, set_seed
from voxlat.utils.config import load_config

LOG = get_logger("task5.dataset")


def _default_jobs() -> int:
    n = os.cpu_count() or 2
    return max(1, n // 2)  # ~ physical cores; the solvers are memory-bandwidth bound


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("design")
    g.add_argument("--n-sobol", type=int, default=256, help="scrambled Sobol points (default 256)")
    g.add_argument("--seed", type=int, default=DEFAULT_DESIGN_SEED, help="Sobol scrambling seed")
    g.add_argument("--no-boundary", action="store_true", help="omit corners, edges and pure lines")
    g.add_argument("--edge-points", type=int, default=1, help="interior points per box edge (default 1)")
    g.add_argument("--pure-line-points", type=int, default=7,
                   help="rho* points on the w = 0 and w = 1, a_z = 1 lines (default 7)")
    g = p.add_argument_group("resolution / solvers")
    g.add_argument("--n-coarse", type=int, default=32)
    g.add_argument("--n-fine", type=int, default=64)
    g.add_argument("--preconditioner", default="auto", choices=["auto", "amg", "jacobi"],
                   help="'auto' = AMG if pyamg is installed (results identical to ~1e-6)")
    g.add_argument("--quick", action="store_true",
                   help="plumbing test: 4 Sobol samples at R(8, 16) into data/closures_quick.* (seconds)")
    g = p.add_argument_group("run")
    g.add_argument("--n-jobs", type=int, default=_default_jobs(),
                   help=f"parallel workers (default cpu_count/2 = {_default_jobs()})")
    g.add_argument("--timing-samples", type=int, default=3, help="samples timed before the run (default 3)")
    g.add_argument("--skip-timing", action="store_true", help="no timing step / estimate")
    g.add_argument("--estimate-only", action="store_true", help="time samples, print the estimate, stop")
    g.add_argument("--limit", type=int, default=None, help="compute at most this many pending samples")
    g.add_argument("--retry-errors", action="store_true", help="recompute samples whose status is not 'ok'")
    g.add_argument("--force", action="store_true",
                   help="recompute records made with other settings/design (instead of stopping)")
    g.add_argument("--save-fields", action="store_true",
                   help="also save fine-grid fields to data/fields/<id>.npz (solid, velocity, von Mises "
                        "factors; ~3-15 MB per sample)")
    g = p.add_argument_group("output")
    g.add_argument("--out", type=Path, default=None, help="table (default data/closures.parquet)")
    g.add_argument("--parts-dir", type=Path, default=None, help="per-sample records (default: <out stem>_parts next to the table, i.e. data/closures_parts)")
    g.add_argument("--plot-only", action="store_true", help="only redraw the quick-look figure from --out")
    g.add_argument("--no-plot", action="store_true")
    g.add_argument("--figure-name", default=None, help="figure stem (default task5_closures_quicklook)")
    return p.parse_args(argv)


def _add_file_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("voxlat")
    if any(getattr(h, "_task5_file", False) for h in root.handlers):
        return
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    fh._task5_file = True  # type: ignore[attr-defined]
    root.addHandler(fh)


def main(argv: list[str] | None = None) -> int:
    a = _parse(argv)
    set_seed()
    if a.quick:
        a.n_sobol, a.no_boundary, a.n_coarse, a.n_fine = 4, True, 8, 16
        a.timing_samples = min(a.timing_samples, 1)
        stem = "closures_quick"
        a.figure_name = a.figure_name or "task5_closures_quicklook_quick"
    else:
        stem = "closures"
    out = a.out or data_dir() / f"{stem}.parquet"
    parts = a.parts_dir or out.parent / f"{out.stem}_parts"
    fields = data_dir() / "fields" if a.save_fields else None
    fig_name = a.figure_name or "task5_closures_quicklook"
    if not a.plot_only:
        _add_file_log(out.parent / f"{out.stem}_build.log")

    if a.plot_only:
        path = out if out.exists() else out.with_suffix(".csv")
        df = read_table(path)
        LOG.info("figure -> %s", plot_quicklook(df, fig_name))
        return 0

    cfg = load_config()
    settings = DatasetSettings(n_coarse=a.n_coarse, n_fine=a.n_fine, preconditioner=a.preconditioner,
                               save_fields=a.save_fields).resolved(cfg)
    space = DesignSpace.from_config(cfg)
    design = build_design(a.n_sobol, space=space, settings=settings, seed=a.seed, boundary=not a.no_boundary,
                          edge_points=a.edge_points, pure_line_points=a.pure_line_points)
    fam: dict[str, int] = {}
    for s in design:
        fam[s.family] = fam.get(s.family, 0) + 1
    LOG.info("=" * 78)
    LOG.info("VoxLat Task 5 closure dataset   %s", time.strftime("%Y-%m-%d %H:%M"))
    LOG.info("design: %d samples %s; w %s, rho* %s, a_z levels %.4f ... %.4f (step %.5f)", len(design), fam,
             space.w, space.rho, min(s.a_z for s in design), max(s.a_z for s in design), settings.quantum)
    LOG.info("grids R(%d, %d), metrics n = %d, k_f/k_s = 1/%.0f, E_s = %.0f GPa, nu = %.2f, settings %s",
             settings.n_coarse, settings.n_fine, settings.metrics_n, settings.k_s / settings.k_f,
             settings.E_s / 1e9, settings.nu_s, settings.fingerprint())
    LOG.info("output %s (parts in %s)%s", out, parts,
             "" if parquet_available() or out.suffix != ".parquet" else "  [pyarrow missing -> CSV fallback]")

    existing = load_existing(parts, out)
    try:
        todo = pending_samples(design, existing, settings, retry_errors=a.retry_errors, force=a.force)
    except SettingsMismatchError as exc:
        LOG.error("%s", exc)
        return 2
    LOG.info("%d already done, %d pending", len(design) - len(todo), len(todo))
    if a.limit is not None:
        todo = sorted(todo, key=lambda s: (-relative_cost(s), s.id))[: a.limit]
        LOG.info("--limit %d: computing %d of them", a.limit, len(todo))

    interrupted = False
    try:
        if todo and not a.skip_timing and a.timing_samples > 0:
            timed_s = pick_timing_samples(todo, a.timing_samples)
            LOG.info("timing %d sample(s) on one core: %s", len(timed_s), ", ".join(
                f"{s.id} (w={s.w:.2f}, rho*={s.rho:.2f}, a_z={s.a_z:.3f})" for s in timed_s))
            res = run_samples(timed_s, settings, parts_dir=parts, n_jobs=1, fields_dir=fields)
            by_id = {s.id: s for s in timed_s}
            timed = [(by_id[r["id"]], r["wall_time_s"]) for r in res]
            todo = [s for s in todo if s.id not in by_id]
            rss = max(r["peak_rss_mb"] for r in res)
            est = estimate_runtime(timed, todo, a.n_jobs, peak_rss_mb=rss)
            LOG.info("runtime estimate for this machine:\n%s", est.summary())
        if a.estimate_only:
            LOG.info("--estimate-only: stopping. Timed samples are saved and will be skipped by the full run.")
        elif todo:
            LOG.info("running %d samples on %d worker(s), longest first ...", len(todo), a.n_jobs)
            run_samples(todo, settings, parts_dir=parts, n_jobs=a.n_jobs, fields_dir=fields)
    except KeyboardInterrupt:
        interrupted = True
        LOG.warning("interrupted - finished samples are saved; rerun the same command to resume")

    df, path = consolidate(parts, out, samples=design)
    if len(df):
        counts = df["status"].value_counts().to_dict()
        LOG.info("table: %d/%d samples %s -> %s", len(df), len(design), counts, path)
        bad = df[df["status"] != "ok"]
        for _, r in bad.iterrows():
            LOG.warning("  %s %s: %s", r["id"], r["status"], r.get("error", ""))
        if len(df) and not a.no_plot:
            LOG.info("figure -> %s", plot_quicklook(df, fig_name))
    missing = len(design) - int((df["status"] == "ok").sum()) if len(df) else len(design)
    if missing and not a.estimate_only:
        LOG.info("%d sample(s) not (successfully) computed yet%s", missing,
                 " - rerun to resume" if interrupted or a.limit else "")
    return 130 if interrupted else 0


if __name__ == "__main__":  # required on Windows (joblib spawns worker processes)
    sys.exit(main())
