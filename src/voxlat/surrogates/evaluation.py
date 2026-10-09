"""Evaluation of the closure surrogates: 5-fold CV, the blend-extrapolation split, metrics, plots.

Metrics (per derived quantity, ``targets.QUANTITY_INFO``):

* **R^2** on the modelling scale: log values for positive quantities (K spans one
  decade, E* two), linear for signed tensor components (off-diagonals).
* **MAPE** = mean |pred/true - 1| (positive quantities only).
* **NMAE** for signed components: mean |pred - true| / mean |reference diagonal|
  (k_eff off-diagonals relative to k*, C off-class terms relative to C11..C33).
* **cov95**: fraction of truths inside the central 95 % predictive interval
  (log-normal for positive quantities: |ln(true/pred)| <= 1.96 sd/pred).
* calibration curve: observed vs nominal coverage of central intervals.

**Recalibration.** GP and ensemble std's are too small by a factor 1.2-3 (CV z-score
spread). ``std_scale_from_oof`` turns out-of-fold latent z-scores into one std
multiplier per latent target (95th percentile of |z| / 1.96 over the triclinic rows,
where every latent direction is active; clipped to [1, 20]). In CV it is applied *cross-fitted* - the
multiplier for fold f comes from the other folds only - and reported as model
"<backend>+cal"; the production ``ClosureModel`` stores the multiplier from a 5-fold
CV of the full table (``std_scale``).

"Truth" is the symmetry-projected closure (``latent_from_table``), i.e. the Task 5
value with voxel symmetry-breaking artifacts removed (<= 1 %, STATUS Task 6).

Splits:

* ``kfold_indices``: 5 shuffled folds (seeded) over all rows ("cv5").
* ``extrapolation_split`` ("band"): train on w in [0, 0.25) U (0.75, 1] - pure gyroid /
  diamond plus the blends near them - and test on the held-out blends
  0.25 <= w <= 0.75: the pinch-neck topologies of Tasks 1-4 never appear in training.
* ``sparse_blend_split`` ("sparse"): train on every pure G/D row plus a random 25 % of
  the blends, test on the other 75 %: generalization in w from few blend examples.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from voxlat.surrogates.symmetry import LOAD_CASES
from voxlat.surrogates.targets import (
    COMP3,
    QUANTITY_INFO,
    DesignBox,
    derived_quantities,
    design_features,
    latent_from_table,
)

__all__ = [
    "kfold_indices",
    "extrapolation_split",
    "sparse_blend_split",
    "SPLITS",
    "point_metrics",
    "QUANTITY_GROUPS",
    "REFERENCE_SCALE",
    "predict_split",
    "latent_cv",
    "std_scale_from_oof",
    "cross_validate",
    "summarize",
    "summarize_groups",
    "calibration_curve",
    "plot_parity",
    "plot_calibration",
]

#: pooled groups for the compact accuracy table
QUANTITY_GROUPS: dict[str, tuple[str, ...]] = {
    "K principal values": ("K_eig1", "K_eig2", "K_eig3"),
    "K* = tr K/3": ("K_mean",),
    "k_eff diagonal": ("keff_xx", "keff_yy", "keff_zz"),
    "k_eff off-diagonal": ("keff_yz", "keff_xz", "keff_xy"),
    "C normal (C11, C22, C33)": ("C11", "C22", "C33"),
    "C coupling (C12, C13, C23)": ("C12", "C13", "C23"),
    "C shear (C44, C55, C66)": ("C44", "C55", "C66"),
    "C off-class (12 others)": ("C14", "C15", "C16", "C24", "C25", "C26", "C34", "C35", "C36", "C45", "C46", "C56"),
    "E* = mean(E_x, E_y, E_z)": ("E_axial",),
    "E_z": ("E_z",),
    "a_sf L": ("a_sf_L",),
    "loc p99, uniaxial": tuple(f"loc_{c}_p99" for c in LOAD_CASES[:3]),
    "loc p99, shear": tuple(f"loc_{c}_p99" for c in LOAD_CASES[3:]),
}

#: reference quantity for the NMAE of signed components
REFERENCE_SCALE: dict[str, str] = {
    **{f"K_{c}": "K_mean" for c in COMP3},
    **{f"keff_{c}": "keff_mean" for c in COMP3},
    **{q: "C_normal" for q in QUANTITY_INFO if q.startswith("C")},
}


# =============================================================================
# Splits
# =============================================================================
def kfold_indices(n: int, k: int = 5, seed: int = 2026) -> list[tuple[np.ndarray, np.ndarray]]:
    """k shuffled folds: list of (train_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    return [(np.sort(np.concatenate([f for j, f in enumerate(folds) if j != i])), np.sort(folds[i])) for i in range(k)]


def extrapolation_split(w: np.ndarray, band: tuple[float, float] = (0.25, 0.75)) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx, test_idx): test = held-out blends band[0] <= w <= band[1]."""
    w = np.asarray(w, dtype=float)
    test = (w >= band[0]) & (w <= band[1])
    return np.flatnonzero(~test), np.flatnonzero(test)


def sparse_blend_split(w: np.ndarray, fraction: float = 0.25, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx, test_idx): train = every pure G/D row + a random ``fraction`` of the blends,
    test = the other blends (sample-efficiency in w rather than extrapolation across a gap)."""
    w = np.asarray(w, dtype=float)
    blend = np.flatnonzero((w > 0.0) & (w < 1.0))
    rng = np.random.default_rng(seed)
    keep = rng.choice(blend, size=int(round(fraction * len(blend))), replace=False)
    test = np.setdiff1d(blend, keep)
    return np.setdiff1d(np.arange(len(w)), test), test


#: named splits used by the Task 6 scripts: name -> (description, function of w)
SPLITS: dict[str, tuple[str, Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]]] = {
    "band": ("train w < 0.25 or w > 0.75, test held-out blends 0.25 <= w <= 0.75", extrapolation_split),
    "sparse": ("train pure G/D + 25 % of the blends, test the other 75 % of the blends", sparse_blend_split),
}


# =============================================================================
# Metrics
# =============================================================================
def point_metrics(true: np.ndarray, pred: np.ndarray, std: np.ndarray, positive: bool,
                  scale: np.ndarray | None = None) -> dict[str, float]:
    """R^2, MAPE / NMAE, RMSE, cov95 and z-score moments for one (pooled) quantity."""
    t = np.asarray(true, float).ravel()
    p = np.asarray(pred, float).ravel()
    s = np.maximum(np.asarray(std, float).ravel(), 1e-300)
    if positive:
        lt, lp = np.log(t), np.log(p)
        z = (lt - lp) / (s / p)
        res = lt - lp
        ss_tot = np.sum((lt - lt.mean()) ** 2)
        mape = float(np.mean(np.abs(p / t - 1.0)))
        nmae = float("nan")
        rmse = float(np.sqrt(np.mean(res**2)))
    else:
        z = (t - p) / s
        res = t - p
        ss_tot = np.sum((t - t.mean()) ** 2)
        mape = float("nan")
        sc = np.abs(np.asarray(scale, float).ravel()) if scale is not None else np.abs(t)
        nmae = float(np.mean(np.abs(res)) / max(np.mean(sc), 1e-300))
        rmse = float(np.sqrt(np.mean(res**2)))
    r2 = float(1.0 - np.sum(res**2) / ss_tot) if ss_tot > 0 else float("nan")
    return {"n": int(t.size), "R2": r2, "MAPE": mape, "NMAE": nmae, "RMSE": rmse,
            "max_APE": float(np.max(np.abs(p / t - 1.0))) if positive else float("nan"),
            "cov95": float(np.mean(np.abs(z) <= 1.959964)), "cov68": float(np.mean(np.abs(z) <= 1.0)),
            "z_mean": float(np.mean(z)), "z_std": float(np.std(z))}


# =============================================================================
# Predictions on splits
# =============================================================================
def _fit_predict_latent(backend: str, X_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray,
                        **kw: Any) -> tuple[np.ndarray, np.ndarray, Any]:
    from voxlat.surrogates.ensemble import MLPEnsemble
    from voxlat.surrogates.gp import GaussianProcessSurrogate

    if backend == "gp":
        m: Any = GaussianProcessSurrogate(**kw).fit(X_tr, Y_tr)
    elif backend == "ensemble":
        m = MLPEnsemble(**kw).fit(X_tr, Y_tr)
    else:
        raise ValueError(backend)
    mu, sd = m.predict(X_te)
    return mu, sd, m


def _long_rows(df: Any, idx: np.ndarray, truth: Mapping[str, tuple[np.ndarray, np.ndarray]],
               pred: Mapping[str, tuple[np.ndarray, np.ndarray]], model: str, split: str, fold: int) -> list[dict[str, Any]]:
    rows = []
    ids = df["id"].to_numpy()[idx] if "id" in df else idx.astype(str)
    w = df["w"].to_numpy(float)[idx]
    r = df["rho_target"].to_numpy(float)[idx]
    a = df["a_z"].to_numpy(float)[idx]
    for q, (pv, ps) in pred.items():
        tv = truth[q][0][idx]
        for k in range(len(idx)):
            rows.append({"model": model, "split": split, "fold": fold, "id": ids[k], "w": w[k], "rho": r[k], "a_z": a[k],
                         "quantity": q, "true": tv[k], "pred": pv[k], "std": ps[k]})
    return rows


def latent_cv(X: np.ndarray, Y: np.ndarray, backend: str, *, k: int = 5, seed: int = 2026,
              folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
              progress: Callable[[str], None] | None = None, **kw: Any
              ) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Out-of-fold latent mean / std (N, T) of ``backend`` and the folds used."""
    folds = folds if folds is not None else kfold_indices(len(X), k, seed)
    mu = np.empty_like(np.asarray(Y, dtype=float))
    sd = np.empty_like(mu)
    for f, (tr, te) in enumerate(folds):
        mu[te], sd[te], _ = _fit_predict_latent(backend, X[tr], Y[tr], X[te], **kw)
        if progress:
            progress(f"{backend}: fold {f + 1}/{len(folds)} done")
    return mu, sd, folds


def std_scale_from_oof(Y: np.ndarray, mu: np.ndarray, sd: np.ndarray, classes: np.ndarray,
                       level: float = 0.95, clip: tuple[float, float] = (1.0, 20.0), min_rows: int = 10) -> np.ndarray:
    """Per-latent-target std multiplier from out-of-fold z-scores (triclinic rows), clipped.

    multiplier = quantile(|z|, level) / z_level (z_0.95 = 1.96): the central ``level``
    interval then covers ``level`` of the out-of-fold points. A quantile rather than
    the RMS of z because the errors are heavy-tailed (pinch-neck blends): matching the
    variance would over-widen the bulk and still under-cover the tail.
    """
    from scipy.stats import norm

    classes = np.asarray(classes)
    rows = classes == "triclinic"
    if rows.sum() < min_rows:
        rows = np.ones(len(classes), dtype=bool)
    z = (np.asarray(Y)[rows] - np.asarray(mu)[rows]) / np.maximum(np.asarray(sd)[rows], 1e-300)
    return np.clip(np.quantile(np.abs(z), level, axis=0) / norm.ppf(0.5 + level / 2), *clip)


def predict_split(df: Any, train_idx: np.ndarray, test_idx: np.ndarray, backend: str, *, split: str = "split",
                  fold: int = 0, box: DesignBox | None = None, Y: np.ndarray | None = None,
                  classes: np.ndarray | None = None, std_scale: np.ndarray | None = None,
                  **kw: Any) -> list[dict[str, Any]]:
    """Fit ``backend`` on train rows, predict test rows; long-format rows of every derived quantity.

    With ``std_scale`` (per latent target) the rows are repeated with the scaled std as
    model "<backend>+cal".
    """
    box = box or DesignBox.from_table(df)
    if Y is None or classes is None:
        Y, classes = latent_from_table(df)
    X = design_features(df[["w", "rho_target", "a_z"]].to_numpy(float), box)
    mu, sd, _ = _fit_predict_latent(backend, X[train_idx], Y[train_idx], X[test_idx], **kw)
    truth = derived_quantities(Y, None, classes)
    rows = _long_rows(df, test_idx, truth, derived_quantities(mu, sd, classes[test_idx]), backend, split, fold)
    if std_scale is not None:
        rows += _long_rows(df, test_idx, truth, derived_quantities(mu, sd * std_scale, classes[test_idx]),
                           f"{backend}+cal", split, fold)
    return rows


def cross_validate(df: Any, backend: str, *, k: int = 5, seed: int = 2026, recalibrate: bool = True,
                   progress: Callable[[str], None] | None = None, **kw: Any) -> tuple[list[dict[str, Any]], np.ndarray]:
    """k-fold CV rows (long format) for one backend, plus the full-CV std multiplier.

    With ``recalibrate`` the rows are repeated as "<backend>+cal" with a cross-fitted
    multiplier (fold f scaled by ``std_scale_from_oof`` of the other folds).
    """
    Y, classes = latent_from_table(df)
    box = DesignBox.from_table(df)
    X = design_features(df[["w", "rho_target", "a_z"]].to_numpy(float), box)
    mu, sd, folds = latent_cv(X, Y, backend, k=k, seed=seed, progress=progress, **kw)
    truth = derived_quantities(Y, None, classes)
    rows: list[dict[str, Any]] = []
    for f, (_, te) in enumerate(folds):
        rows += _long_rows(df, te, truth, derived_quantities(mu[te], sd[te], classes[te]), backend, "cv5", f)
        if recalibrate:
            other = np.setdiff1d(np.arange(len(df)), te)
            sc = std_scale_from_oof(Y[other], mu[other], sd[other], classes[other])
            rows += _long_rows(df, te, truth, derived_quantities(mu[te], sd[te] * sc, classes[te]),
                               f"{backend}+cal", "cv5", f)
    return rows, std_scale_from_oof(Y, mu, sd, classes)


def _reference_lookup(pred_df: Any) -> dict[str, dict[str, float]]:
    """Per-sample reference scales for NMAE: K_mean, keff_mean, mean(C11, C22, C33) (true values)."""
    ref: dict[str, dict[str, float]] = {}
    base = pred_df.drop_duplicates(["id", "quantity"])
    for name, qs in (("K_mean", ("K_mean",)), ("keff_mean", ("keff_mean",)), ("C_normal", ("C11", "C22", "C33"))):
        sub = base[base["quantity"].isin(qs)]
        ref[name] = sub.groupby("id")["true"].mean().to_dict()
    return ref


def summarize(pred_df: Any, quantities: Iterable[str] | None = None) -> Any:
    """Metrics per (model, split, quantity)."""
    import pandas as pd

    ref = _reference_lookup(pred_df)
    out = []
    qs = list(quantities) if quantities is not None else list(dict.fromkeys(pred_df["quantity"]))
    for (model, split), g in pred_df.groupby(["model", "split"], sort=False):
        for q in qs:
            sub = g[g["quantity"] == q]
            if sub.empty:
                continue
            pos = QUANTITY_INFO.get(q, ("", True, ""))[1]
            scale = None
            if not pos and q in REFERENCE_SCALE:
                scale = sub["id"].map(ref[REFERENCE_SCALE[q]]).to_numpy(float)
            out.append({"model": model, "split": split, "quantity": q, "positive": pos,
                        **point_metrics(sub["true"], sub["pred"], sub["std"], pos, scale)})
    return pd.DataFrame(out)


def summarize_groups(pred_df: Any, groups: Mapping[str, Sequence[str]] = QUANTITY_GROUPS) -> Any:
    """Pooled metrics per (model, split, group) - the compact STATUS table."""
    import pandas as pd

    ref = _reference_lookup(pred_df)
    out = []
    for (model, split), g in pred_df.groupby(["model", "split"], sort=False):
        for name, qs in groups.items():
            sub = g[g["quantity"].isin(qs)]
            if sub.empty:
                continue
            pos = all(QUANTITY_INFO.get(q, ("", True, ""))[1] for q in qs)
            scale = None
            if not pos:
                scale = np.array([ref[REFERENCE_SCALE[q]].get(i, np.nan) for q, i in zip(sub["quantity"], sub["id"])])
            out.append({"model": model, "split": split, "group": name, "positive": pos,
                        **point_metrics(sub["true"], sub["pred"], sub["std"], pos, scale)})
    return pd.DataFrame(out)


def calibration_curve(pred_df: Any, quantities: Sequence[str] | None = None,
                      levels: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(nominal, observed) coverage of central Gaussian intervals pooled over quantities."""
    from scipy.stats import norm

    levels = np.linspace(0.05, 0.99, 30) if levels is None else np.asarray(levels)
    sub = pred_df if quantities is None else pred_df[pred_df["quantity"].isin(quantities)]
    t, p, s = sub["true"].to_numpy(float), sub["pred"].to_numpy(float), np.maximum(sub["std"].to_numpy(float), 1e-300)
    pos = np.array([QUANTITY_INFO.get(q, ("", True, ""))[1] for q in sub["quantity"]])
    z = np.where(pos, (np.log(np.abs(t) + 1e-300) - np.log(np.abs(p) + 1e-300)) / (s / np.abs(p)), (t - p) / s)
    obs = np.array([np.mean(np.abs(z) <= norm.ppf(0.5 + lv / 2)) for lv in levels])
    return levels, obs


# =============================================================================
# Plots
# =============================================================================
_PANELS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("K principal values / L^2", ("K_eig1", "K_eig2", "K_eig3"), "log"),
    ("k_eff diagonal / k_s", ("keff_xx", "keff_yy", "keff_zz"), "log"),
    ("C11, C22, C33 / E_s", ("C11", "C22", "C33"), "log"),
    ("C44, C55, C66 / E_s", ("C44", "C55", "C66"), "log"),
    ("E* / E_s", ("E_axial",), "log"),
    ("a_sf L", ("a_sf_L",), "linear"),
    ("loc p99 (uniaxial)", tuple(f"loc_{c}_p99" for c in LOAD_CASES[:3]), "log"),
    ("k_eff off-diagonal / k_s", ("keff_yz", "keff_xz", "keff_xy"), "linear"),
)


def plot_parity(pred_df: Any, *, split: str, models: Sequence[str], title: str = "",
                panels: Sequence[tuple[str, tuple[str, ...], str]] = _PANELS, color_by: str = "w",
                ncols: int = 4) -> Any:
    """Parity plots (pred vs true, 95 % error bars), coloured by w; panels wrap at ``ncols``,
    one block of rows per model."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    nblk = int(np.ceil(len(panels) / ncols))
    nrow = nblk * len(models)
    fig, axes = plt.subplots(nrow, ncols, figsize=(2.9 * ncols, 2.75 * nrow), squeeze=False, constrained_layout=True)
    sc = None
    for mi, model in enumerate(models):
        g = pred_df[(pred_df["model"] == model) & (pred_df["split"] == split)]
        for k, (label, qs, scale) in enumerate(panels):
            r, c = mi * nblk + k // ncols, k % ncols
            ax = axes[r, c]
            sub = g[g["quantity"].isin(qs)]
            if sub.empty:
                ax.set_axis_off()
                continue
            t, p, s = sub["true"].to_numpy(), sub["pred"].to_numpy(), sub["std"].to_numpy()
            ax.errorbar(t, p, yerr=1.96 * s, fmt="none", ecolor="0.75", elinewidth=0.5, zorder=1)
            sc = ax.scatter(t, p, c=sub[color_by], cmap="viridis", vmin=0, vmax=1, s=6, zorder=2, linewidths=0)
            lo, hi = np.nanmin(np.r_[t, p]), np.nanmax(np.r_[t, p])
            if scale == "log" and lo > 0:
                ax.set_xscale("log")
                ax.set_yscale("log")
                pad = (hi / lo) ** 0.05
                lo, hi = lo / pad, hi * pad
                for axis in (ax.xaxis, ax.yaxis):
                    axis.set_minor_formatter(NullFormatter())
            else:
                pad = 0.05 * (hi - lo + 1e-12)
                lo, hi = lo - pad, hi + pad
            ax.plot([lo, hi], [lo, hi], "k-", lw=0.7, zorder=3)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            m = point_metrics(t, p, s, scale == "log" and np.all(t > 0))
            txt = f"R²={m['R2']:.3f}\n" + (f"MAPE={100 * m['MAPE']:.1f}%" if np.isfinite(m["MAPE"]) else f"RMSE={m['RMSE']:.2g}")
            txt += f"\ncov95={100 * m['cov95']:.0f}%"
            ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", fontsize=7,
                    bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))
            ax.tick_params(labelsize=7)
            ax.set_title(f"{model}: {label}", fontsize=8)
            if c == 0:
                ax.set_ylabel("predicted", fontsize=8)
            ax.set_xlabel("computed (Task 5)", fontsize=8)
        for k in range(len(panels), nblk * ncols):
            axes[mi * nblk + k // ncols, k % ncols].set_axis_off()
    if sc is not None:
        fig.colorbar(sc, ax=axes[:, -1], shrink=0.5, label="w (0 = gyroid, 1 = diamond)")
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def plot_calibration(pred_df: Any, *, groups: Mapping[str, Sequence[str]], splits: Sequence[str],
                     models: Sequence[str], title: str = "") -> Any:
    """Observed vs nominal coverage per model, one panel per split, one line per quantity group."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(models), len(splits), figsize=(3.4 * len(splits), 3.2 * len(models)), squeeze=False,
                             constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for r, model in enumerate(models):
        for c, split in enumerate(splits):
            ax = axes[r, c]
            g = pred_df[(pred_df["model"] == model) & (pred_df["split"] == split)]
            for k, (name, qs) in enumerate(groups.items()):
                sub = g[g["quantity"].isin(qs)]
                if sub.empty:
                    continue
                lv, obs = calibration_curve(sub)
                ax.plot(lv, obs, color=cmap(k % 10), lw=1.2, label=name)
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{model} - {split}", fontsize=9)
            ax.set_xlabel("nominal coverage", fontsize=8)
            ax.set_ylabel("observed coverage", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0, -1].legend(fontsize=6, loc="lower right")
    if title:
        fig.suptitle(title, fontsize=10)
    return fig
