"""Grid-convergence helpers shared by the homogenization tasks (2, 3, 4, 7).

Model: a quantity computed on a cell with n voxels per cell length behaves as

    f(n) = f_inf + C n^(-p) + (noise),

where p is the order of the discretization error (p = 1 for staircase voxel
interfaces, p = 2 for smooth fields) and the "noise" is the non-monotone
dependence of a voxelized geometry on how the surface happens to cut the grid.

* ``richardson``: classic two-grid extrapolation for a known order p.
* ``observed_order``: three grids with a constant refinement ratio.
* ``fit_convergence``: least-squares fit over any set of resolutions, with p
  fixed or free; returns f_inf, its standard error and the relative error of
  each grid. With noisy staircase data this is more robust than three-point
  Richardson.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import curve_fit

__all__ = ["richardson", "observed_order", "ConvergenceFit", "fit_convergence"]


def richardson(n1: float, f1: float, n2: float, f2: float, p: float = 1.0) -> float:
    """Two-grid Richardson estimate of f_inf from f(n1), f(n2) for error order p.

    f_inf = (n2^p f2 - n1^p f1) / (n2^p - n1^p).
    """
    if n1 == n2:
        raise ValueError("need two different resolutions")
    a, b = float(n1) ** p, float(n2) ** p
    return (b * f2 - a * f1) / (b - a)


def observed_order(n: Sequence[float], f: Sequence[float]) -> float:
    """Observed order from three grids n0 < n1 < n2 with a constant ratio r = n1/n0 = n2/n1.

    p = ln((f1 - f0)/(f2 - f1)) / ln r. Returns nan if the differences change sign.
    """
    n0, n1, n2 = (float(v) for v in n)
    f0, f1, f2 = (float(v) for v in f)
    r = n1 / n0
    if not np.isclose(n2 / n1, r, rtol=1e-6):
        raise ValueError("observed_order needs a constant refinement ratio")
    d1, d2 = f1 - f0, f2 - f1
    if d1 * d2 <= 0:
        return float("nan")
    return float(np.log(d1 / d2) / np.log(r))


@dataclass(frozen=True)
class ConvergenceFit:
    """Result of ``fit_convergence``: f(n) = f_inf + C n^-p."""

    f_inf: float
    C: float
    p: float
    f_inf_std: float  # standard error of f_inf from the fit covariance
    residual_rms: float  # rms of (data - model), absolute
    p_fixed: bool

    def predict(self, n: float | np.ndarray) -> np.ndarray | float:
        return self.f_inf + self.C * np.asarray(n, dtype=float) ** (-self.p)

    def rel_error(self, n: float | np.ndarray) -> np.ndarray | float:
        """Model relative error (f(n) - f_inf) / f_inf of a grid with n voxels per cell."""
        return self.C * np.asarray(n, dtype=float) ** (-self.p) / self.f_inf


def fit_convergence(
    n: Sequence[float], f: Sequence[float], p: float | None = 1.0
) -> ConvergenceFit:
    """Least-squares fit of f(n) = f_inf + C n^-p.

    p given -> linear least squares in (f_inf, C). p None -> nonlinear fit of
    (f_inf, C, p) with 0.25 <= p <= 4 (needs >= 4 points).
    """
    n = np.asarray(n, dtype=float)
    f = np.asarray(f, dtype=float)
    if n.shape != f.shape or n.size < 2:
        raise ValueError("need matching arrays with at least two resolutions")
    if p is not None:
        A = np.c_[np.ones_like(n), n ** (-p)]
        coef, *_ = np.linalg.lstsq(A, f, rcond=None)
        res = f - A @ coef
        dof = max(n.size - 2, 1)
        s2 = float(res @ res) / dof
        cov = s2 * np.linalg.inv(A.T @ A)
        return ConvergenceFit(
            float(coef[0]), float(coef[1]), float(p), float(np.sqrt(cov[0, 0])),
            float(np.sqrt(np.mean(res**2))), True,
        )
    if n.size < 4:
        raise ValueError("a free-order fit needs at least 4 resolutions")
    # Fit in normalized units: curve_fit's default tolerances are absolute-ish and it
    # stalls at p0 when |f| << 1 (e.g. permeabilities K/L^2 ~ 1e-3; found in Task 4).
    scale = float(np.max(np.abs(f))) or 1.0
    g = f / scale
    lin = fit_convergence(n, g, p=1.0)

    def model(x: np.ndarray, a: float, c: float, q: float) -> np.ndarray:
        return a + c * x ** (-q)

    popt, pcov = curve_fit(
        model, n, g, p0=(lin.f_inf, lin.C, 1.0),
        bounds=([-np.inf, -np.inf, 0.25], [np.inf, np.inf, 4.0]), maxfev=20000,
    )
    res = g - model(n, *popt)
    return ConvergenceFit(
        float(popt[0] * scale), float(popt[1] * scale), float(popt[2]),
        float(np.sqrt(abs(pcov[0, 0])) * scale), float(np.sqrt(np.mean(res**2)) * scale), False,
    )
