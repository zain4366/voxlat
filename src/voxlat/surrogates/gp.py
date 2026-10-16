"""Gaussian-process surrogate: one GP per latent target, ARD Matern-5/2 kernel.

Model (per target y, inputs u in [0, 1]^3 from ``targets.design_features``)::

    y(u) = b^T [1, u] + s_y * f(u),     f ~ GP(0, s_f^2 k_5/2(r) + s_n^2 delta),
    k_5/2(r) = (1 + sqrt5 r + 5 r^2 / 3) exp(-sqrt5 r),   r^2 = sum_d (u_d - u'_d)^2 / l_d^2

* linear trend b fitted by least squares first (power laws are linear in log rho*,
  so outside the data the GP reverts to the physics-like trend, not to a constant);
* residuals standardized (s_y) and fitted by scikit-learn's
  ``GaussianProcessRegressor`` (marginal-likelihood optimization with restarts) -
  ARD length scales l_d (one per input), signal s_f^2, white noise s_n^2 (the
  closures carry ~0.1-1 % voxel-alignment noise, Tasks 2-4);
* the fitted state (training inputs, standardized residuals, hyperparameters) is
  exported to plain numpy arrays; the Cholesky factor and alpha are recomputed on
  load (40 x 290^2, milliseconds), so model files stay ~100 kB and do not depend
  on the scikit-learn version.

Predictive std includes the white-noise term (it is the uncertainty of a new
*computed* closure value, which is what the evaluation compares against).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.linalg import cho_solve, solve_triangular

__all__ = ["matern52", "GPTarget", "GaussianProcessSurrogate"]

_SQRT5 = np.sqrt(5.0)


def matern52(A: np.ndarray, B: np.ndarray, length_scale: np.ndarray) -> np.ndarray:
    """ARD Matern-5/2 correlation matrix between rows of A (n, d) and B (m, d)."""
    a = A / length_scale
    b = B / length_scale
    d2 = np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T, 0.0)
    r = np.sqrt(d2)
    return (1.0 + _SQRT5 * r + (5.0 / 3.0) * d2) * np.exp(-_SQRT5 * r)


@dataclass
class GPTarget:
    """Exported state of one fitted GP (all numpy)."""

    length_scale: np.ndarray  # (d,)
    signal_var: float
    noise_var: float
    jitter: float
    trend: np.ndarray  # (d+1,) least-squares coefficients of [1, u]
    y_scale: float
    alpha: np.ndarray  # (n,)
    chol: np.ndarray  # (n, n) lower Cholesky of signal_var*K + (noise_var+jitter) I
    z: np.ndarray  # (n,) standardized training residuals
    log_marginal_likelihood: float = float("nan")

    @classmethod
    def from_hyperparameters(cls, X: np.ndarray, z: np.ndarray, length_scale: np.ndarray, signal_var: float,
                             noise_var: float, jitter: float, trend: np.ndarray, y_scale: float,
                             lml: float = float("nan")) -> "GPTarget":
        K = signal_var * matern52(X, X, length_scale) + (noise_var + jitter) * np.eye(X.shape[0])
        L = np.linalg.cholesky(K)
        alpha = cho_solve((L, True), z, check_finite=False)
        return cls(length_scale=np.asarray(length_scale, float), signal_var=float(signal_var), noise_var=float(noise_var),
                   jitter=float(jitter), trend=np.asarray(trend, float), y_scale=float(y_scale), alpha=alpha, chol=L,
                   z=np.asarray(z, float), log_marginal_likelihood=float(lml))

    def predict(self, Xtrain: np.ndarray, X: np.ndarray, return_std: bool = True) -> tuple[np.ndarray, np.ndarray]:
        Ks = self.signal_var * matern52(X, Xtrain, self.length_scale)  # (m, n)
        trend = self.trend[0] + X @ self.trend[1:]
        mean = trend + self.y_scale * (Ks @ self.alpha)
        if not return_std:
            return mean, np.zeros_like(mean)
        v = solve_triangular(self.chol, Ks.T, lower=True, check_finite=False)
        var = self.signal_var + self.noise_var - np.einsum("ij,ij->j", v, v)
        return mean, self.y_scale * np.sqrt(np.maximum(var, 1e-12 * self.signal_var))


def _fit_one(X: np.ndarray, y: np.ndarray, *, n_restarts: int, seed: int, linear_mean: bool,
             noise_bounds: tuple[float, float], length_bounds: tuple[float, float]) -> GPTarget:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    n, d = X.shape
    A = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0] if linear_mean else np.array([y.mean()] + [0.0] * d)
    r = y - A @ beta
    scale = float(np.std(r)) or 1.0
    z = r / scale
    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.full(d, 0.5), length_scale_bounds=length_bounds, nu=2.5)
              + WhiteKernel(1e-3, noise_bounds))
    jitter = 1e-10
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=jitter, normalize_y=False, n_restarts_optimizer=n_restarts,
                                   random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gpr.fit(X, z)
    k = gpr.kernel_
    sv = float(k.k1.k1.constant_value)
    ls = np.atleast_1d(np.asarray(k.k1.k2.length_scale, dtype=float))
    if ls.size == 1:
        ls = np.full(d, float(ls[0]))
    nv = float(k.k2.noise_level)
    return GPTarget.from_hyperparameters(X, z, ls, sv, nv, jitter, beta, scale, float(gpr.log_marginal_likelihood_value_))


@dataclass
class GaussianProcessSurrogate:
    """Independent ARD Matern-5/2 GPs, one per target column."""

    n_restarts: int = 2
    seed: int = 2026
    linear_mean: bool = True
    noise_bounds: tuple[float, float] = (1e-8, 1e-1)
    length_bounds: tuple[float, float] = (1e-2, 1e2)
    n_jobs: int = 1
    X: np.ndarray | None = None
    targets: list[GPTarget] = field(default_factory=list)
    names: tuple[str, ...] = ()

    def fit(self, X: np.ndarray, Y: np.ndarray, names: Sequence[str] | None = None) -> "GaussianProcessSurrogate":
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        kw = dict(n_restarts=self.n_restarts, seed=self.seed, linear_mean=self.linear_mean,
                  noise_bounds=self.noise_bounds, length_bounds=self.length_bounds)
        if self.n_jobs != 1 and Y.shape[1] > 1:
            from joblib import Parallel, delayed

            self.targets = list(Parallel(n_jobs=self.n_jobs)(delayed(_fit_one)(X, Y[:, t], **kw) for t in range(Y.shape[1])))
        else:
            self.targets = [_fit_one(X, Y[:, t], **kw) for t in range(Y.shape[1])]
        self.X = X
        self.names = tuple(names) if names is not None else tuple(f"y{t}" for t in range(Y.shape[1]))
        return self

    def predict(self, X: np.ndarray, return_std: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if self.X is None:
            raise RuntimeError("GP not fitted")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        mu = np.empty((X.shape[0], len(self.targets)))
        sd = np.empty_like(mu)
        for t, g in enumerate(self.targets):
            mu[:, t], sd[:, t] = g.predict(self.X, X, return_std)
        return mu, sd

    def hyperparameters(self) -> list[dict[str, Any]]:
        return [{"name": n, "length_scale": g.length_scale.tolist(), "signal_var": g.signal_var, "noise_var": g.noise_var,
                 "y_scale": g.y_scale, "noise_std_rel": float(np.sqrt(g.noise_var / (g.signal_var + g.noise_var))),
                 "lml": g.log_marginal_likelihood} for n, g in zip(self.names, self.targets)]

    # ------------------------------------------------------------------ persistence
    def to_arrays(self, prefix: str = "gp.") -> dict[str, np.ndarray]:
        if self.X is None:
            raise RuntimeError("GP not fitted")
        out: dict[str, np.ndarray] = {f"{prefix}X": self.X}
        out[f"{prefix}length_scale"] = np.stack([g.length_scale for g in self.targets])
        out[f"{prefix}signal_var"] = np.array([g.signal_var for g in self.targets])
        out[f"{prefix}noise_var"] = np.array([g.noise_var for g in self.targets])
        out[f"{prefix}jitter"] = np.array([g.jitter for g in self.targets])
        out[f"{prefix}trend"] = np.stack([g.trend for g in self.targets])
        out[f"{prefix}y_scale"] = np.array([g.y_scale for g in self.targets])
        out[f"{prefix}z"] = np.stack([g.z for g in self.targets])
        out[f"{prefix}lml"] = np.array([g.log_marginal_likelihood for g in self.targets])
        out[f"{prefix}names"] = np.array(self.names)
        return out

    @classmethod
    def from_arrays(cls, a: dict[str, np.ndarray], prefix: str = "gp.") -> "GaussianProcessSurrogate":
        m = cls()
        m.X = np.asarray(a[f"{prefix}X"], dtype=float)
        T = len(a[f"{prefix}signal_var"])
        m.targets = [
            GPTarget.from_hyperparameters(
                m.X, np.asarray(a[f"{prefix}z"][t], float), np.asarray(a[f"{prefix}length_scale"][t], float),
                float(a[f"{prefix}signal_var"][t]), float(a[f"{prefix}noise_var"][t]), float(a[f"{prefix}jitter"][t]),
                np.asarray(a[f"{prefix}trend"][t], float), float(a[f"{prefix}y_scale"][t]), float(a[f"{prefix}lml"][t]))
            for t in range(T)
        ]
        m.names = tuple(str(s) for s in a[f"{prefix}names"])
        return m
