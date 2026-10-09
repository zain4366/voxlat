"""Task 6: accuracy and calibration of the GP and deep-ensemble closure surrogates.

    python scripts/task6_evaluate.py              # 5-fold CV + band and sparse splits (~10 min, 2 cores)
    python scripts/task6_evaluate.py --replot     # figures/tables from results/task6_predictions.csv.gz
    python scripts/task6_evaluate.py --quick      # 2 folds, tiny models (plumbing test)

Outputs: results/task6_predictions.csv.gz (every test prediction, long format),
results/task6_metrics.csv (per quantity), results/task6_metrics_groups.csv (pooled
groups = the STATUS table); figures task6_parity_cv.png, task6_parity_band.png,
task6_calibration.png. Models "gp+cal" / "ensemble+cal" are the same predictions with
CV-recalibrated std's (cross-fitted in cv5; full-CV multiplier on the other splits).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task6_train_surrogates import ENSEMBLE_SETTINGS, GP_SETTINGS  # noqa: E402

from voxlat.surrogates import latent_from_table, load_training_table  # noqa: E402
from voxlat.surrogates.evaluation import (  # noqa: E402
    QUANTITY_GROUPS,
    SPLITS,
    cross_validate,
    plot_calibration,
    plot_parity,
    predict_split,
    summarize,
    summarize_groups,
)
from voxlat.surrogates.targets import DesignBox  # noqa: E402
from voxlat.utils import get_logger, save_figure, set_seed  # noqa: E402
from voxlat.utils.paths import results_dir  # noqa: E402

LOG = get_logger("task6.evaluate")
MODELS = ("gp", "ensemble")
CAL_MODELS = ("gp", "gp+cal", "ensemble", "ensemble+cal")
CAL_GROUPS = {
    "K principal": QUANTITY_GROUPS["K principal values"],
    "k_eff diag": QUANTITY_GROUPS["k_eff diagonal"],
    "C normal": QUANTITY_GROUPS["C normal (C11, C22, C33)"],
    "C shear": QUANTITY_GROUPS["C shear (C44, C55, C66)"],
    "E*": ("E_axial",),
    "a_sf L": ("a_sf_L",),
    "loc p99": QUANTITY_GROUPS["loc p99, uniaxial"] + QUANTITY_GROUPS["loc p99, shear"],
}


def run(df: pd.DataFrame, *, quick: bool, n_jobs: int) -> pd.DataFrame:
    settings = {"gp": dict(GP_SETTINGS), "ensemble": dict(ENSEMBLE_SETTINGS)}
    if quick:
        settings["gp"].update(n_restarts=0)
        settings["ensemble"].update(n_members=2, epochs=300)
    for s in settings.values():
        s["n_jobs"] = n_jobs
    rows: list[dict] = []
    Y, classes = latent_from_table(df)
    box = DesignBox.from_table(df)
    for model in MODELS:
        t0 = time.perf_counter()
        cv_rows, scale = cross_validate(df, model, k=2 if quick else 5, progress=LOG.info, **settings[model])
        rows += cv_rows
        LOG.info("%s cv done in %.0f s; std multipliers median %.2f (max %.2f)", model, time.perf_counter() - t0,
                 float(np.median(scale)), float(scale.max()))
        for split, (desc, fn) in SPLITS.items():
            tr, te = fn(df["w"].to_numpy(float))
            rows += predict_split(df, tr, te, model, split=split, box=box, Y=Y, classes=classes, std_scale=scale,
                                  **settings[model])
            LOG.info("%s %s split (%d train / %d test): %s", model, split, len(tr), len(te), desc)
    return pd.DataFrame(rows)


def report(pred: pd.DataFrame) -> None:
    out = results_dir()
    per_q = summarize(pred)
    per_q.to_csv(out / "task6_metrics.csv", index=False, float_format="%.6g")
    groups = summarize_groups(pred)
    groups.to_csv(out / "task6_metrics_groups.csv", index=False, float_format="%.6g")
    LOG.info("metrics -> %s, %s", out / "task6_metrics.csv", out / "task6_metrics_groups.csv")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        for split in ("cv5", "band", "sparse"):
            g = groups[groups["split"] == split]
            if g.empty:
                continue
            tab = g.pivot_table(index="group", columns="model", values=["R2", "MAPE", "NMAE", "cov95"], sort=False)
            print(f"\n=== {split} ===")
            print(tab.round(4))
    cal = [m for m in ("gp+cal", "ensemble+cal") if m in set(pred["model"])] or list(MODELS)
    fig = plot_parity(pred, split="cv5", models=cal,
                      title="Task 6: 5-fold cross-validation (bars = recalibrated 95 % predictive interval)")
    save_figure(fig, "task6_parity_cv")
    fig = plot_parity(pred, split="band", models=cal,
                      title="Task 6: extrapolation to held-out blends 0.25 <= w <= 0.75 (trained on w < 0.25 or > 0.75)")
    save_figure(fig, "task6_parity_band")
    fig = plot_calibration(pred, groups=CAL_GROUPS, splits=["cv5", "band", "sparse"],
                           models=[m for m in CAL_MODELS if m in set(pred["model"])],
                           title="Task 6: calibration of central predictive intervals (raw and CV-recalibrated)")
    save_figure(fig, "task6_calibration")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default=None, help="closure table (default data/closures.parquet)")
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true", help="only redo tables and figures")
    args = ap.parse_args(argv)
    set_seed(2026)
    path = results_dir() / ("task6_predictions_quick.csv.gz" if args.quick else "task6_predictions.csv.gz")
    if args.replot:
        pred = pd.read_csv(path)
    else:
        df = load_training_table(args.table)
        t0 = time.perf_counter()
        pred = run(df, quick=args.quick, n_jobs=args.n_jobs)
        pred.to_csv(path, index=False, float_format="%.9g")
        LOG.info("predictions (%d rows) -> %s in %.0f s", len(pred), path, time.perf_counter() - t0)
    if not args.quick:
        report(pred)
    else:
        print(summarize_groups(pred).pivot_table(index="group", columns=["model", "split"], values="R2").round(3))
    return 0


if __name__ == "__main__":
    np.seterr(over="ignore")
    sys.exit(main())
