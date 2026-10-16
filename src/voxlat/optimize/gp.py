"""Minimal exact Gaussian process for the BoTorch-free Bayesian-optimization fallback (Task 11).

Model (per outcome, inputs in the unit cube, outcomes standardized):
    y = m + f(x) + eps,   f ~ GP(0, s2 k(x, x')),   eps ~ N(0, s2_n),
    k = Matern-5/2 with ARD lengthscales l_k:  r^2 = sum_k ((x_k - x'_k)/l_k)^2,
        k(r) = (1 + sqrt5 r + 5 r^2/3) exp(-sqrt5 r).
Hyperparameters (log l, log s2, log s2_n) by maximum a posteriori with the priors of BoTorch's
current default single-task GP (Hvarfner, Hellsten & Nardi 2024, "Vanilla Bayesian
optimization performs great in high dimensions"):
    log l_k ~ N(sqrt2 + 0.5 log d, sqrt3^2)  (dimension-scaled),   log s2_n ~ N(-4, 1),
    s2_n >= 1e-6,  and a weak N(0, 2^2) prior on log s2.
L-BFGS-B on the analytic gradient of the log posterior, multi-start (prior median + previous
optimum). ``condition_on`` adds observations with fixed hyperparameters (kriging-believer
fantasies for batch selection).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

__all__ = ["GaussianProcess"]

SQ5 = math.sqrt(5.0)
NOISE_MIN = 1e-6


def _matern52(X1: np.ndarray, X2: np.ndarray, ls: np.ndarray) -> np.ndarray:
    A = X1 / ls
    B = X2 / ls
    r2 = np.maximum((A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2.0 * A @ B.T, 0.0)
    r = np.sqrt(r2)
    return (1.0 + SQ5 * r + 5.0 / 3.0 * r2) * np.exp(-SQ5 * r)


@dataclass
class _Fit:
    log_ls: np.ndarray
    log_s2: float
    log_noise: float


class GaussianProcess:
    """Exact GP regression with Matern-5/2 ARD kernel; ``fit(X, y)`` then ``predict(Xc)``."""

    def __init__(self, n_restarts: int = 2, maxiter: int = 200, seed: int = 0):
        self.n_restarts = n_restarts
        self.maxiter = maxiter
        self.rng = np.random.default_rng(seed)
        self.params: _Fit | None = None

    # ------------------------------------------------------------------ priors
    @staticmethod
    def _ls_prior(d: int) -> tuple[float, float]:
        return math.sqrt(2.0) + 0.5 * math.log(d), math.sqrt(3.0)

    def _neg_log_post(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray, D2: np.ndarray) -> tuple[float, np.ndarray]:
        n, d = X.shape
        log_ls, log_s2, log_nz = theta[:d], theta[d], theta[d + 1]
        ls = np.exp(log_ls)
        s2 = math.exp(log_s2)
        nz = math.exp(log_nz) + NOISE_MIN
        S = D2 / ls[:, None, None] ** 2  # (d, n, n) scaled squared distances
        r2 = S.sum(0)
        r = np.sqrt(r2)
        e = np.exp(-SQ5 * r)
        Kc = (1.0 + SQ5 * r + 5.0 / 3.0 * r2) * e
        K = s2 * Kc + nz * np.eye(n)
        try:
            c, low = cho_factor(K, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(theta)
        alpha = cho_solve((c, low), y, check_finite=False)
        logdet = 2.0 * np.log(np.diag(c)).sum()
        nll = 0.5 * y @ alpha + 0.5 * logdet + 0.5 * n * math.log(2 * math.pi)
        Kinv = cho_solve((c, low), np.eye(n), check_finite=False)
        W = np.outer(alpha, alpha) - Kinv  # dLML = 0.5 tr(W dK)
        # dK/dlog l_k = s2 (5/3)(1 + sqrt5 r) e * S_k
        common = s2 * (5.0 / 3.0) * (1.0 + SQ5 * r) * e
        g_ls = -0.5 * np.einsum("ij,kij->k", W * common, S)
        g_s2 = -0.5 * np.sum(W * (s2 * Kc))
        g_nz = -0.5 * np.trace(W) * math.exp(log_nz)
        # priors
        mu_l, sd_l = self._ls_prior(d)
        nlp = 0.5 * np.sum((log_ls - mu_l) ** 2) / sd_l**2 + 0.5 * (log_nz + 4.0) ** 2 + 0.5 * log_s2**2 / 4.0
        g_ls = g_ls + (log_ls - mu_l) / sd_l**2
        g_nz = g_nz + (log_nz + 4.0)
        g_s2 = g_s2 + log_s2 / 4.0
        return float(nll + nlp), np.concatenate([g_ls, [g_s2, g_nz]])

    # ------------------------------------------------------------------ fit / predict
    def fit(self, X: Any, y: Any) -> "GaussianProcess":
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        n, d = X.shape
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) if y.std() > 0 else 1.0
        ys = (y - self.y_mean) / self.y_std
        D2 = (X[None, :, :] - X[:, None, :]).transpose(2, 0, 1) ** 2  # (d, n, n)
        mu_l, sd_l = self._ls_prior(d)
        starts = [np.concatenate([np.full(d, mu_l - sd_l**2 / 2.0), [0.0, -4.0]])]
        if self.params is not None and len(self.params.log_ls) == d:
            starts.append(np.concatenate([self.params.log_ls, [self.params.log_s2, self.params.log_noise]]))
        for _ in range(max(0, self.n_restarts - len(starts))):
            starts.append(np.concatenate([mu_l - sd_l**2 / 2.0 + 0.5 * self.rng.standard_normal(d), [0.0, -4.0]]))
        bounds = [(math.log(1e-2), math.log(1e3))] * d + [(math.log(1e-2), math.log(1e2)), (math.log(1e-8), math.log(1.0))]
        best = None
        for th0 in starts:
            res = minimize(self._neg_log_post, th0, args=(X, ys, D2), jac=True, method="L-BFGS-B", bounds=bounds,
                           options=dict(maxiter=self.maxiter))
            if best is None or res.fun < best.fun:
                best = res
        th = best.x
        self.params = _Fit(th[:d].copy(), float(th[d]), float(th[d + 1]))
        self._set_data(X, ys)
        return self

    def _set_data(self, X: np.ndarray, ys: np.ndarray) -> None:
        p = self.params
        ls = np.exp(p.log_ls)
        s2 = math.exp(p.log_s2)
        nz = math.exp(p.log_noise) + NOISE_MIN
        K = s2 * _matern52(X, X, ls) + nz * np.eye(len(X))
        self._L = np.linalg.cholesky(K)
        self._alpha = cho_solve((self._L, True), ys)
        self._X = X
        self._ys = ys

    def condition_on(self, Xn: Any, yn: Any) -> "GaussianProcess":
        """Copy conditioned on extra observations (original units), same hyperparameters."""
        new = GaussianProcess(self.n_restarts, self.maxiter)
        new.params = self.params
        new.y_mean, new.y_std = self.y_mean, self.y_std
        Xn = np.atleast_2d(np.asarray(Xn, float))
        ysn = (np.asarray(yn, float).ravel() - self.y_mean) / self.y_std
        new._set_data(np.vstack([self._X, Xn]), np.concatenate([self._ys, ysn]))
        return new

    def predict(self, Xc: Any, return_std: bool = True) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Posterior mean and std of the latent f (original units)."""
        Xc = np.atleast_2d(np.asarray(Xc, float))
        p = self.params
        ls = np.exp(p.log_ls)
        s2 = math.exp(p.log_s2)
        Ks = s2 * _matern52(Xc, self._X, ls)
        mean = Ks @ self._alpha
        mu = self.y_mean + self.y_std * mean
        if not return_std:
            return mu
        v = solve_triangular(self._L, Ks.T, lower=True, check_finite=False)
        var = np.maximum(s2 - np.sum(v**2, axis=0), 1e-12)
        return mu, self.y_std * np.sqrt(var)

    @property
    def lengthscales(self) -> np.ndarray:
        return np.exp(self.params.log_ls)
