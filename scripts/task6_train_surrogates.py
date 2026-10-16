"""Task 6: train the production closure surrogates on the full Task 5 table and save them to models/.

    python scripts/task6_train_surrogates.py                 # GP + deep ensemble, CV-recalibrated (~7 min)
    python scripts/task6_train_surrogates.py --cnn           # + 3D CNN for K*, E* (+15 min)
    python scripts/task6_train_surrogates.py --table data/closures.csv --n-jobs 4

Writes models/closure_gp.npz, models/closure_ensemble.npz (and models/closure_cnn.npz),
then prints a sanity check: predictions at the pure gyroid/diamond points against the
Task 2-4 values (STATUS.md) and the train-set fit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from voxlat.surrogates import ClosureModel, CNNClosureModel, latent_from_table, load_training_table
from voxlat.surrogates.evaluation import point_metrics
from voxlat.surrogates.targets import QUANTITY_INFO, derived_quantities
from voxlat.utils import get_logger, set_seed
from voxlat.utils.paths import results_dir

LOG = get_logger("task6.train")

#: hyper-parameters used for the production models AND the evaluation (scripts/task6_evaluate.py)
GP_SETTINGS = dict(n_restarts=2, linear_mean=True)
ENSEMBLE_SETTINGS = dict(n_members=5, hidden=(64, 64, 64), epochs=3000, warmup=0.3, lr=3e-3, weight_decay=1e-4)
CNN_SETTINGS = dict(n_members=3, epochs=120, batch=16, lr=2e-3, weight_decay=1e-4, warmup=0.25, channels=(16, 32, 32),
                    pool="avg")


def _sanity(cm: ClosureModel, df) -> list[dict]:
    """Train-set fit + pure-line check (the points of the Task 2-4 tables)."""
    Y, cl = latent_from_table(df)
    truth = derived_quantities(Y, None, cl)
    pred = cm.predict_dimensionless(df[["w", "rho_target", "a_z"]].to_numpy(float))
    rows = []
    for q in ("K_mean", "keff_mean", "E_axial", "C44", "a_sf_L", "loc_uniaxial_z_p99"):
        m = point_metrics(truth[q][0], pred[q][0], pred[q][1], QUANTITY_INFO[q][1])
        rows.append({"quantity": q, "train_R2": m["R2"], "train_MAPE": m["MAPE"]})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default=None, help="closure table (default data/closures.parquet)")
    ap.add_argument("--backends", nargs="*", default=["gp", "ensemble"], choices=["gp", "ensemble"],
                    help="parameter-based models to train (none: --backends with no value)")
    ap.add_argument("--cnn", action="store_true", help="also train the 3D CNN (K*, E*)")
    ap.add_argument("--n-jobs", type=int, default=2, help="parallel GP targets / ensemble members / CNN members")
    ap.add_argument("--quick", action="store_true", help="tiny settings (plumbing test, not for use)")
    ap.add_argument("--no-recalibrate", action="store_true", help="skip the 5-fold CV std recalibration")
    ap.add_argument("--out-dir", default=None, help="models directory (default models/)")
    args = ap.parse_args(argv)

    set_seed(2026)
    df = load_training_table(args.table)
    LOG.info("closure table: %d rows", len(df))
    from pathlib import Path

    out_dir = Path(args.out_dir) if args.out_dir else None
    report = {}
    for backend in args.backends:
        t0 = time.perf_counter()
        kw = dict(GP_SETTINGS) if backend == "gp" else dict(ENSEMBLE_SETTINGS)
        if args.quick:
            kw.update({"n_restarts": 0} if backend == "gp" else {"n_members": 2, "epochs": 300})
        kw["n_jobs"] = args.n_jobs
        cm = ClosureModel.fit(df, backend, recalibrate=not args.no_recalibrate, cv_folds=2 if args.quick else 5, **kw)
        path = cm.save(None if out_dir is None else out_dir / f"closure_{backend}.npz")
        dt = time.perf_counter() - t0
        LOG.info("%s: trained in %.1f s -> %s", backend, dt, path)
        report[backend] = {"train_time_s": dt, "path": str(path), "sanity": _sanity(cm, df),
                           "recalibration": cm.meta.get("recalibration")}
        if cm.std_scale is not None:
            LOG.info("%s std multipliers (median by group): %s", backend,
                     {g: round(v, 2) for g, v in cm.meta["recalibration"]["by_group"].items()})
        pure = df[(df["a_z"] == 1.0) & df["w"].isin([0.0, 1.0])].sort_values(["w", "rho_target"])
        pred = cm.predict_dimensionless(pure[["w", "rho_target", "a_z"]].to_numpy(float))
        print(f"\n{backend}: pure gyroid / diamond (a_z = 1), prediction vs Task 5 value")
        print(f"{'w':>3} {'rho*':>5} | {'K*/L^2':>18} | {'k*/k_s':>16} | {'E*/E_s':>16} | {'loc_z p99':>14}")
        for (_, r), *vals in zip(pure.iterrows(), *(zip(*pred[q]) for q in ("K_mean", "keff_mean", "E_axial", "loc_uniaxial_z_p99"))):
            (km, _), (kk, _), (ee, _), (ll, _) = vals
            print(f"{r.w:3.0f} {r.rho_target:5.3f} | {km:.4e} ({km / r.K_mean - 1:+.1%}) | {kk:.4f} ({kk / r.keff_mean - 1:+.1%})"
                  f" | {ee:.4f} ({ee / r.E_axial - 1:+.1%}) | {ll:5.1f} ({ll / r.loc_uniaxial_z_p99 - 1:+.1%})")
    if args.cnn:
        t0 = time.perf_counter()
        kw = dict(CNN_SETTINGS)
        if args.quick:
            kw.update(n_members=1, epochs=3)
        kw["n_jobs"] = args.n_jobs
        cnn = CNNClosureModel.fit(df, **kw)
        path = cnn.save(None if out_dir is None else out_dir / "closure_cnn.npz")
        dt = time.perf_counter() - t0
        LOG.info("cnn: trained in %.1f s -> %s", dt, path)
        report["cnn"] = {"train_time_s": dt, "path": str(path)}
    rp = results_dir() / "task6_train_report.json"
    if not args.quick:
        old = json.loads(rp.read_text()) if rp.is_file() else {}
        old.update(report)  # keep entries of models not retrained in this run
        rp.write_text(json.dumps(old, indent=2, default=float))
        LOG.info("report -> %s", rp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
