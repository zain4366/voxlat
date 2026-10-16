"""Task 11: run the multi-objective optimization studies (RQ2, RQ3).

Each (layout, method, seed) writes results/task11/<layout>/<method>_s<seed>.csv.gz + .json
(checkpoint every ~2 min; ``--skip-done`` resumes an interrupted run from it). Runs are independent;
``--parallel N`` runs N of them side by side (one core each).

Examples (Windows, from the repo root, venv active)
    # full study, both layouts, 5 seeds (see STATUS.md Task 11 for runtimes)
    python scripts/task11_optimize.py --layout slots end_rings --method nsga2 qnehvi sobol --seeds 0 1 2 3 4 --parallel 4
    # one quick check (~2 min)
    python scripts/task11_optimize.py --quick
Then: python scripts/task11_analysis.py
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
import warnings


def _job(args: tuple) -> str:
    layout, method, seed, kw = args
    warnings.simplefilter("ignore")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from voxlat.optimize.study import run_one

    t = time.perf_counter()
    p = run_one(layout, method, seed, **kw)
    return f"{layout}/{method} seed {seed}: done in {time.perf_counter() - t:.0f} s -> {p}.csv.gz"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", nargs="+", default=["slots", "end_rings"], choices=["slots", "end_rings"])
    ap.add_argument("--method", nargs="+", default=["nsga2", "qnehvi", "sobol"], choices=["nsga2", "qnehvi", "sobol"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--pop-size", type=int, default=100, help="NSGA-II population (default 100)")
    ap.add_argument("--n-gen", type=int, default=100, help="NSGA-II generations (default 100 -> 10 000 proposals)")
    ap.add_argument("--bo-budget", type=int, default=200, help="qNEHVI expensive evaluations (default 200)")
    ap.add_argument("--n-init", type=int, default=32, help="qNEHVI initial Sobol evaluations (default 32)")
    ap.add_argument("--q", type=int, default=4, help="qNEHVI batch size (default 4)")
    ap.add_argument("--backend", default="auto", choices=["auto", "botorch", "numpy"], help="qNEHVI backend")
    ap.add_argument("--sobol-budget", type=int, default=1000, help="random-search expensive evaluations (default 1000)")
    ap.add_argument("--parallel", type=int, default=1, help="independent runs side by side (processes)")
    ap.add_argument("--root", default=None, help="output root (default results/task11)")
    ap.add_argument("--no-anchors", action="store_true", help="start every method without the 9 uniform anchors")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip runs already finished with the same budget settings and resume interrupted ones from "
                         "their last checkpoint (runs with other settings, e.g. the shipped sandbox runs, are redone)")
    ap.add_argument("--retries", type=int, default=3,
                    help="restart the unfinished runs this many times if a worker process dies (default 3)")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (slots, 1 seed, few evaluations)")
    a = ap.parse_args(argv)
    if a.quick:
        a.layout, a.seeds, a.pop_size, a.n_gen, a.bo_budget, a.n_init, a.sobol_budget = (["slots"], [0], 12, 3, 16, 8, 16)
        a.root = a.root or os.path.join("results", "task11_quick")
    jobs = []
    for layout, method, seed in itertools.product(a.layout, a.method, a.seeds):
        kw = dict(root=a.root, anchors=not a.no_anchors)
        if method == "nsga2":
            kw.update(pop_size=a.pop_size, n_gen=a.n_gen)
        elif method == "qnehvi":
            kw.update(budget=a.bo_budget, n_init=a.n_init, q=a.q, backend=a.backend)
        else:
            kw.update(budget=a.sobol_budget)
        if a.skip_done:
            from voxlat.optimize.study import checkpoint_status, run_is_complete

            # done AND same budget settings (stale results, e.g. the sandbox runs shipped with the repo, are redone)
            if run_is_complete(layout, method, seed, **kw):
                print(f"skip {layout}/{method} seed {seed} (done)")
                continue
            # interrupted with the same settings -> continue from its checkpoint
            prev = checkpoint_status(layout, method, seed, **kw)
            if prev is not None:
                print(f"resume {layout}/{method} seed {seed} from {prev.get('n_expensive')} evaluations")
            kw["resume"] = True
        jobs.append((layout, method, seed, kw))
    print(f"{len(jobs)} runs, {a.parallel} in parallel", flush=True)
    failed = []
    if a.parallel > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool

        todo = list(jobs)
        for attempt in range(1 + a.retries):
            if not todo:
                break
            done_ids: set[int] = set()
            broken = False
            with ProcessPoolExecutor(a.parallel) as ex:
                futs = {ex.submit(_job, j): k for k, j in enumerate(todo)}
                for f in as_completed(futs):
                    k = futs[f]
                    try:
                        print(f.result(), flush=True)
                        done_ids.add(k)
                    except BrokenProcessPool:
                        broken = True
                    except Exception as exc:  # one failing run must not stop the others
                        lay, meth, sd, _ = todo[k]
                        print(f"FAILED {lay}/{meth} seed {sd}: {type(exc).__name__}: {exc}", flush=True)
                        failed.append(todo[k])
                        done_ids.add(k)
            todo = [j for k, j in enumerate(todo) if k not in done_ids]
            for j in todo:
                j[3]["resume"] = True  # restarted runs continue from their last checkpoint (~2 min lost at most)
            if broken and todo:
                print(f"a worker process died (out of memory, sleep or a forced restart?); restarting the "
                      f"{len(todo)} unfinished run(s) ({attempt + 1}/{a.retries} retries used)", flush=True)
        failed += todo
    else:
        for j in jobs:
            try:
                print(_job(j), flush=True)
            except Exception as exc:
                print(f"FAILED {j[0]}/{j[1]} seed {j[2]}: {type(exc).__name__}: {exc}", flush=True)
                failed.append(j)
    if failed:
        print(f"{len(failed)} run(s) did not finish: " + ", ".join(f"{l}/{m} s{s}" for l, m, s, _ in failed)
              + "  -> rerun the same command with --skip-done", flush=True)
        return 1
    print("all runs finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
