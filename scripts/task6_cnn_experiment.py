"""Task 6c: does geometry-based learning (3D CNN on voxels) generalize better than parameter-based (GP, ensemble)?

    python scripts/task6_cnn_experiment.py               # cv5 + band + sparse, 3 CNN members (~80 min, 2 cores)
    python scripts/task6_cnn_experiment.py --splits band sparse
    python scripts/task6_cnn_experiment.py --replot
    python scripts/task6_cnn_experiment.py --quick       # plumbing test

All three models learn the same two targets, log K* = log(tr K / 3 / L^2) and
log E* = log(E_axial / E_s), on the same rows:
  * GP and deep ensemble see theta = (w, rho*, a_z);
  * the CNN sees the 32^3 voxel cell (normalized coordinates) + a_z.
Splits: cv5 (interpolation), band (held-out blends 0.25 <= w <= 0.75), sparse
(pure G/D + 25 % of blends -> other 75 %).
Outputs: results/task6_cnn_predictions.csv, results/task6_cnn_metrics.csv,
figure results/figures/task6_cnn_extrapolation.png.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task6_train_surrogates import CNN_SETTINGS, ENSEMBLE_SETTINGS, GP_SETTINGS  # noqa: E402

from voxlat.surrogates import (  # noqa: E402
    CNNEnsemble,
    GaussianProcessSurrogate,
    MLPEnsemble,
    cnn_voxels,
    design_features,
    load_training_table,
    scalar_targets_from_table,
)
from voxlat.surrogates.evaluation import SPLITS, kfold_indices, point_metrics  # noqa: E402
from voxlat.surrogates.targets import DesignBox  # noqa: E402
from voxlat.utils import get_logger, save_figure, set_seed  # noqa: E402
from voxlat.utils.paths import results_dir  # noqa: E402

LOG = get_logger("task6.cnn")
TARGETS = (("K_star", "K* / L^2"), ("E_star", "E* / E_s"))
MODELS = ("gp", "ensemble", "cnn")


def _splits(df: pd.DataFrame, names: list[str], quick: bool) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    out = []
    w = df["w"].to_numpy(float)
    for name in names:
        if name == "cv5":
            for f, (tr, te) in enumerate(kfold_indices(len(df), 2 if quick else 5)):
                out.append(("cv5", f, tr, te))
        else:
            tr, te = SPLITS[name][1](w)
            out.append((name, 0, tr, te))
    return out


def run(df: pd.DataFrame, splits: list[str], *, quick: bool, n_jobs: int, members: int, epochs: int) -> pd.DataFrame:
    th = df[["w", "rho_target", "a_z"]].to_numpy(float)
    box = DesignBox.from_table(df)
    u = design_features(th, box)
    Y = scalar_targets_from_table(df)
    t0 = time.perf_counter()
    vox = cnn_voxels(th)
    LOG.info("voxelized %d cells at 32^3 in %.1f s", len(df), time.perf_counter() - t0)
    side = CNNEnsemble.side_features(u)
    gp_kw = dict(GP_SETTINGS, n_jobs=1)
    ens_kw = dict(ENSEMBLE_SETTINGS, n_jobs=n_jobs)
    cnn_kw = dict(CNN_SETTINGS, n_jobs=n_jobs, n_members=members, epochs=epochs)
    if quick:
        gp_kw.update(n_restarts=0)
        ens_kw.update(n_members=2, epochs=300)
        cnn_kw.update(n_members=1, epochs=2)
    rows = []
    for split, fold, tr, te in _splits(df, splits, quick):
        preds = {}
        t1 = time.perf_counter()
        preds["gp"] = GaussianProcessSurrogate(**gp_kw).fit(u[tr], Y[tr]).predict(u[te])
        preds["ensemble"] = MLPEnsemble(**ens_kw).fit(u[tr], Y[tr]).predict(u[te])
        t2 = time.perf_counter()
        cnn = CNNEnsemble(**cnn_kw).fit(vox[tr], side[tr], Y[tr])
        preds["cnn"] = cnn.predict(vox[te], side[te])
        LOG.info("%s fold %d: %d train / %d test; gp+ens %.0f s, cnn %.0f s", split, fold, len(tr), len(te), t2 - t1,
                 time.perf_counter() - t2)
        for model, (mu, sd) in preds.items():
            for t, (name, _) in enumerate(TARGETS):
                v, s = np.exp(mu[:, t]), np.exp(mu[:, t]) * sd[:, t]
                for k, i in enumerate(te):
                    rows.append({"model": model, "split": split, "fold": fold, "id": df["id"].iat[i], "w": th[i, 0],
                                 "rho": th[i, 1], "a_z": th[i, 2], "quantity": name, "true": float(np.exp(Y[i, t])),
                                 "pred": float(v[k]), "std": float(s[k])})
    return pd.DataFrame(rows)


def metrics(pred: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (split, model, q), g in pred.groupby(["split", "model", "quantity"], sort=False):
        out.append({"split": split, "model": model, "quantity": q, **point_metrics(g["true"], g["pred"], g["std"], True)})
    return pd.DataFrame(out)


def plot(pred: pd.DataFrame, splits: list[str]):
    import matplotlib.pyplot as plt

    have = [s for s in splits if s in set(pred["split"])]
    fig, axes = plt.subplots(len(TARGETS), len(have) + 1, figsize=(3.2 * (len(have) + 1), 3.0 * len(TARGETS)),
                             squeeze=False, constrained_layout=True)
    colors = {"gp": "tab:blue", "ensemble": "tab:orange", "cnn": "tab:green"}
    marks = {"gp": "o", "ensemble": "s", "cnn": "^"}
    for r, (q, label) in enumerate(TARGETS):
        for c, split in enumerate(have):
            ax = axes[r, c]
            g = pred[(pred["split"] == split) & (pred["quantity"] == q)]
            lo, hi = g[["true", "pred"]].min().min(), g[["true", "pred"]].max().max()
            for model in MODELS:
                sub = g[g["model"] == model]
                m = point_metrics(sub["true"], sub["pred"], sub["std"], True)
                ax.scatter(sub["true"], sub["pred"], s=7, marker=marks[model], color=colors[model], alpha=0.7,
                           linewidths=0, label=f"{model}: MAPE {100 * m['MAPE']:.1f} %, cov95 {100 * m['cov95']:.0f} %")
            ax.plot([lo, hi], [lo, hi], "k-", lw=0.7)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{label} - {split}", fontsize=9)
            ax.set_xlabel("computed (Task 5)", fontsize=8)
            ax.set_ylabel("predicted", fontsize=8)
            ax.legend(fontsize=6, loc="upper left")
            ax.tick_params(labelsize=7)
        # error vs w on the band split (or the last split)
        ax = axes[r, -1]
        sp = "band" if "band" in have else have[-1]
        g = pred[(pred["split"] == sp) & (pred["quantity"] == q)]
        for model in MODELS:
            sub = g[g["model"] == model].sort_values("w")
            ax.scatter(sub["w"], 100 * (sub["pred"] / sub["true"] - 1), s=7, marker=marks[model], color=colors[model],
                       alpha=0.7, linewidths=0, label=model)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel("w", fontsize=8)
        ax.set_ylabel("error [%]", fontsize=8)
        ax.set_title(f"{label} error vs w ({sp})", fontsize=9)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)
    fig.suptitle("Task 6c: geometry-based (CNN on 32³ voxels) vs parameter-based (GP, ensemble) learning", fontsize=10)
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default=None)
    ap.add_argument("--splits", nargs="+", default=["band", "sparse", "cv5"], choices=["cv5", "band", "sparse"])
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--members", type=int, default=CNN_SETTINGS["n_members"])
    ap.add_argument("--epochs", type=int, default=CNN_SETTINGS["epochs"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args(argv)
    set_seed(2026)
    out = results_dir()
    path = out / ("task6_cnn_predictions_quick.csv" if args.quick else "task6_cnn_predictions.csv")
    if args.replot:
        pred = pd.read_csv(path)
    else:
        df = load_training_table(args.table)
        pred = run(df, args.splits, quick=args.quick, n_jobs=args.n_jobs, members=args.members, epochs=args.epochs)
        if path.exists() and not args.quick:  # keep splits computed earlier
            old = pd.read_csv(path)
            pred = pd.concat([old[~old["split"].isin(args.splits)], pred], ignore_index=True)
        pred.to_csv(path, index=False, float_format="%.9g")
    m = metrics(pred)
    if not args.quick:
        m.to_csv(out / "task6_cnn_metrics.csv", index=False, float_format="%.6g")
    with pd.option_context("display.width", 200):
        print(m.pivot_table(index=["split", "quantity"], columns="model", values=["MAPE", "R2", "cov95"]).round(3))
    if not args.quick:
        save_figure(plot(pred, ["cv5", "band", "sparse"]), "task6_cnn_extrapolation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
