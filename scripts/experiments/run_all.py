"""Reproduce the VoxLat paper figures and tables.

    python scripts/experiments/run_all.py --list             # catalog: items, producer steps, runtime estimates
    python scripts/experiments/run_all.py --quick            # plumbing run of every step (~10-20 min on a laptop)
    python scripts/experiments/run_all.py --replot           # figures/tables from the committed data (~15 min)
    python scripts/experiments/run_all.py                    # full recomputation (days; see --list / --estimate)
    python scripts/experiments/run_all.py --only F20 T07     # only what these items need
    python scripts/experiments/run_all.py --estimate         # print the full-run time budget and stop

Modes
  --quick   every producer at tiny resolution/budget, run in a throw-away COPY of the repo (default
            <tmp>/voxlat_quick; --sandbox DIR to choose, --in-place to write into the repo instead), so the
            committed paper results are never overwritten. Proves the code paths run end to end.
  --replot  re-plot / re-tabulate from data/, results/, models/ (no long solves). Writes into the repo.
  (full)    the exact commands behind the numbers in STATUS.md. Writes into the repo. Long steps resume
            where supported (Task 5 dataset, Task 7 strips, Task 11 runs).

Every mode ends with the paper tables (tables.py), the figure gallery (make_gallery.py) and the consistency audit
(check_consistency.py), unless --no-post. A JSON log with per-step wall times goes to results/reproduce/.
Steps needing the Windows PicoGK setup (``manual``) are never run here; their command is printed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from catalog import ITEMS, STEPS, Item, Step, item, step  # noqa: E402

COPY_IGNORE = shutil.ignore_patterns(".git", "__pycache__", "*.egg-info", ".pytest_cache", "results_quick*",
                                     "export_ci", "*.stl", "*.bin", ".venv", "venv")


def _fmt(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def _missing(mods: tuple[str, ...]) -> list[str]:
    return [m for m in mods if importlib.util.find_spec(m) is None]


def steps_for(items: list[Item], with_deps: bool = True) -> list[Step]:
    """Producer steps of ``items`` in catalog (= dependency) order, plus their upstream steps if ``with_deps``."""
    wanted = {i.step for i in items}
    if with_deps:
        todo = list(wanted)
        while todo:
            for d in step(todo.pop()).deps:
                if d not in wanted:
                    wanted.add(d)
                    todo.append(d)
    return [s for s in STEPS if s.id in wanted]


def args_for(s: Step, mode: str) -> tuple[str, ...] | None:
    return {"full": s.full, "quick": s.quick, "replot": s.replot}[mode]


def make_sandbox(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    shutil.copytree(ROOT, path, ignore=COPY_IGNORE)
    return path


def run_step(s: Step, mode: str, root: Path, log_dir: Path, *, dry: bool = False) -> dict:
    rec = dict(step=s.id, script=s.script, mode=mode, status="", seconds=0.0, command="")
    a = args_for(s, mode)
    if s.manual:
        rec.update(status="manual", command=" ".join(["python", s.script, *(s.full or ())]), note=s.note)
        return rec
    if a is None:
        rec.update(status="n/a", note=f"no {mode} mode; committed outputs used. {s.note}".strip())
        return rec
    miss = _missing(s.replot_needs if mode == "replot" and s.replot_needs is not None else s.needs)
    if miss:
        rec.update(status="skipped", note="missing modules: " + ", ".join(miss) + " (pip install -e .[ml,opt])")
        return rec
    cmd = [sys.executable, str(root / s.script), *a]
    rec["command"] = " ".join(["python", s.script, *a])
    if dry:
        rec["status"] = "dry-run"
        return rec
    env = dict(os.environ, VOXLAT_ROOT=str(root), MPLBACKEND="Agg", PYTHONHASHSEED="0")
    if root != ROOT:   # sandbox: import the sandbox copy of the package (editable install points at the repo)
        env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    log = log_dir / f"{s.id}.log"
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, cwd=root, env=env, stdout=fh, stderr=subprocess.STDOUT)
    rec.update(seconds=round(time.perf_counter() - t0, 1), returncode=p.returncode, log=str(log),
               status="ok" if p.returncode == 0 else "FAILED")
    if p.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        rec["tail"] = tail
    return rec


def post(root: Path, mode: str, log_dir: Path, items: list[Item]) -> list[dict]:
    recs = []
    env = dict(os.environ, VOXLAT_ROOT=str(root), MPLBACKEND="Agg")
    jobs = [("tables", [str(root / "scripts/experiments/tables.py"), "all"]),
            ("gallery", [str(root / "scripts/experiments/make_gallery.py")]
             + (["--out", str(root / "results/reproduce/gallery_quick.pdf"), "--dpi", "60"] if mode == "quick" else [])),
            ("consistency", [str(root / "scripts/experiments/check_consistency.py")])]
    for name, cmd in jobs:
        t0 = time.perf_counter()
        log = log_dir / f"post_{name}.log"
        with open(log, "w", encoding="utf-8") as fh:
            p = subprocess.run([sys.executable, *cmd], cwd=root, env=env, stdout=fh, stderr=subprocess.STDOUT)
        if p.returncode == 0:
            status = "ok"
        elif name == "consistency":   # flags are findings, not failures; in quick mode the data are quick data
            status = "ran (quick data)" if mode == "quick" else "FLAGS"
        else:
            status = "FAILED"
        recs.append(dict(step=name, status=status,
                         seconds=round(time.perf_counter() - t0, 1), log=str(log), returncode=p.returncode))
    return recs


def check_outputs(items: list[Item], root: Path, since: float) -> list[dict]:
    out = []
    for it in items:
        p = root / it.path
        out.append(dict(item=it.id, path=it.path, exists=p.exists(),
                        fresh=p.exists() and p.stat().st_mtime >= since))
    return out


def print_catalog() -> None:
    by_step: dict[str, list[str]] = {}
    for it in ITEMS:
        by_step.setdefault(it.step, []).append(it.id)
    print(f"{'step':16s} {'items':24s} {'full (laptop)':45s} quick replot  script")
    for s in STEPS:
        flags = ("yes" if s.quick is not None else " - ", "yes" if s.replot is not None else " - ")
        tag = " [manual]" if s.manual else ""
        print(f"{s.id:16s} {','.join(by_step.get(s.id, ['(supp.)'])):24s} {s.est_full[:44]:45s} "
              f"{flags[0]:5s} {flags[1]:6s} {s.script}{tag}")
    print()
    for it in ITEMS:
        print(f"{it.id}  {it.path:58s} <- {it.step}")


def print_estimate() -> None:
    total = sum(s.est_full_s for s in STEPS if not s.manual)
    seq = sum(s.est_full_s for s in STEPS if not s.manual and s.id != "task11_opt")
    print("Full from-scratch reproduction (laptop estimates from STATUS.md):")
    for s in sorted(STEPS, key=lambda s: -s.est_full_s):
        print(f"  {s.id:16s} {_fmt(s.est_full_s):>9s}  {s.est_full}")
    print(f"  everything except the Task 11 optimization study: {_fmt(seq)}")
    print(f"  Task 11 study on the laptop (4 runs in parallel):  {_fmt(step('task11_opt').est_full_s)} "
          "(or 2 h 35 min on GitHub Actions)")
    print(f"  total sequential on one laptop: {_fmt(total)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="plumbing run in a sandbox copy")
    g.add_argument("--replot", action="store_true", help="figures/tables from committed data")
    ap.add_argument("--only", nargs="+", default=None, help="item ids (F01 ... T09) or step ids")
    ap.add_argument("--skip", nargs="+", default=[], help="step ids to skip")
    ap.add_argument("--no-deps", action="store_true",
                    help="with --only: run just the producer, not its upstream steps (reuse committed inputs)")
    ap.add_argument("--sandbox", default=None, help="quick-mode copy of the repo (default <tmp>/voxlat_quick)")
    ap.add_argument("--in-place", action="store_true", help="quick mode: write into the repo, not a copy")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed step (default for --quick)")
    ap.add_argument("--no-post", action="store_true", help="skip tables, gallery and consistency audit")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    a = ap.parse_args(argv)

    if a.list:
        print_catalog()
        return 0
    if a.estimate:
        print_estimate()
        return 0
    mode = "quick" if a.quick else "replot" if a.replot else "full"
    keep_going = a.keep_going or mode == "quick"

    if a.only:
        ids = {x.upper() if x[0].lower() in "ft" and x[1:].isdigit() else x for x in a.only}
        items = [i for i in ITEMS if i.id in ids or i.step in ids]
        if not items:
            ap.error(f"no item/step matches {a.only}")
    else:
        items = list(ITEMS)
    with_deps = mode != "replot" and not a.no_deps
    steps = [s for s in steps_for(items, with_deps) if s.id not in a.skip and s.id != "tables"]
    if mode == "quick":
        steps = [s for s in steps if not s.quick_last] + [s for s in steps if s.quick_last]

    if mode == "quick" and not a.in_place and not a.dry_run:
        root = Path(a.sandbox) if a.sandbox else Path(tempfile.gettempdir()) / "voxlat_quick"
        print(f"[quick] copying the repo to {root} (committed results stay untouched) ...", flush=True)
        make_sandbox(root)
    else:
        root = ROOT
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = root / "results" / "reproduce" / f"{mode}_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    t0 = time.perf_counter()
    recs = []
    for s in steps:
        print(f"[{mode}] {s.id:16s} ...", end=" ", flush=True)
        r = run_step(s, mode, root, log_dir, dry=a.dry_run)
        recs.append(r)
        print(f"{r['status']:8s} {_fmt(r['seconds']) if r['seconds'] else '':>8s}  {r.get('note', '')}", flush=True)
        if r["status"] == "FAILED":
            print("    " + "\n    ".join(r.get("tail", [])))
            if not keep_going:
                break
    if not a.no_post and not a.dry_run:
        for r in post(root, mode, log_dir, items):
            recs.append(r)
            print(f"[{mode}] {r['step']:16s} ... {r['status']:8s} {_fmt(r['seconds']):>8s}", flush=True)
    wall = time.perf_counter() - t0

    outputs = check_outputs(items, root, t_start) if not a.dry_run else []
    summary = dict(mode=mode, root=str(root), started_utc=stamp, wall_s=round(wall, 1),
                   python=sys.version.split()[0], platform=platform.platform(), steps=recs, outputs=outputs)
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    failed = [r["step"] for r in recs if r["status"] == "FAILED"]
    skipped = [r["step"] for r in recs if r["status"] == "skipped"]
    manual = [r for r in recs if r["status"] == "manual"]
    print(f"\n{mode}: {len(recs)} steps in {_fmt(wall)}; failed: {failed or 'none'}; skipped (deps): "
          f"{skipped or 'none'}")
    if outputs:
        stale = [o["item"] for o in outputs if not o["fresh"]]
        print(f"items regenerated in this run: {len(outputs) - len(stale)}/{len(outputs)}"
              + (f" (committed version kept: {', '.join(stale)})" if stale else ""))
    for r in manual:
        print(f"manual step {r['step']} (Windows + PicoGK): {r['command']}")
    print(f"log: {log_dir / 'summary.json'}")
    return 1 if failed else 0


def run_item(item_id: str, argv: list[str] | None = None) -> int:
    """Entry point of the per-item scripts ``scripts/experiments/<id>_<name>.py``.

    Same flags as run_all (``--quick``, ``--replot``, ``--dry-run``; default = full recomputation of the producer).
    Tables are pure post-processing of committed results and are rebuilt directly.
    """
    it = item(item_id)
    argv = list(sys.argv[1:] if argv is None else argv)
    print(f"{it.id} {it.name}: {it.path}\n  {it.caption}\n")
    if it.step == "tables":
        if "--dry-run" in argv:
            print(f"python scripts/experiments/tables.py {it.id}")
            return 0
        sys.path.insert(0, str(HERE))
        import tables
        return tables.main([it.id])
    return main(["--only", it.id, "--no-post", *argv])


if __name__ == "__main__":
    raise SystemExit(main())
