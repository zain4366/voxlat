"""Deep ensemble of heteroscedastic MLPs (Lakshminarayanan, Pritzel & Blundell 2017).

Each member maps the scaled inputs u in [0, 1]^3 (mapped to [-1, 1]) to a mean and
a variance for every (standardized) latent target:
``3 -> H -> H -> H -> 2T``, SiLU, He-uniform init, AdamW, full batch (<= 290 rows).
Training: a mean-squared-error warm-up (variance heads frozen at their init) and
then the Gaussian negative log-likelihood, cosine learning-rate decay. Members differ
only in their random initialisation (as in the original paper; bootstrapping hurts
on small data). The ensemble is a uniform Gaussian mixture:

    mean = (1/M) sum_m mu_m,
    var  = (1/M) sum_m (sigma_m^2 + mu_m^2) - mean^2      (aleatoric + epistemic).

All weights are numpy arrays (``nn``), saved in ``.npz``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from voxlat.surrogates.nn import MIN_VARIANCE, Adam, Dense, Sequential, SiLU, gaussian_nll, mse_loss, softplus

__all__ = ["MLPMember", "MLPEnsemble"]


def _build(n_in: int, n_out: int, hidden: Sequence[int], rng: np.random.Generator) -> Sequential:
    layers: list[Any] = []
    d = n_in
    for h in hidden:
        layers += [Dense(d, h, rng), SiLU()]
        d = h
    head = Dense(d, 2 * n_out, rng, gain=0.5)
    head.params["b"][n_out:] = np.log(np.expm1(0.1))  # initial variance 0.1 (standardized units)
    layers.append(head)
    return Sequential(layers)


@dataclass
class MLPMember:
    net: Sequential
    n_out: int

    def forward(self, x: np.ndarray, train: bool = False) -> tuple[np.ndarray, np.ndarray]:
        out = self.net.forward(x, train)
        return out[:, : self.n_out], out[:, self.n_out:]


def _train_member(X: np.ndarray, Y: np.ndarray, *, hidden: Sequence[int], epochs: int, warmup: float, lr: float,
                  weight_decay: float, seed: int) -> tuple[dict[str, np.ndarray], list[float]]:
    rng = np.random.default_rng(seed)
    T = Y.shape[1]
    net = _build(X.shape[1], T, hidden, rng)
    opt = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    n_warm = int(warmup * epochs)
    hist = []
    for ep in range(epochs):
        lr_t = 1e-5 + 0.5 * (lr - 1e-5) * (1.0 + np.cos(np.pi * ep / epochs))
        out = net.forward(X, train=True)
        mu, raw = out[:, :T], out[:, T:]
        if ep < n_warm:
            loss, dmu = mse_loss(mu, Y)
            draw = np.zeros_like(raw)
        else:
            loss, dmu, draw = gaussian_nll(mu, raw, Y)
        net.backward(np.concatenate([dmu, draw], axis=1))
        opt.step(lr_t)
        if ep % 50 == 0 or ep == epochs - 1:
            hist.append(loss)
    return net.state_dict(), hist


@dataclass
class MLPEnsemble:
    """Deep ensemble surrogate for a block of targets (inputs in [0, 1]^d)."""

    n_members: int = 5
    hidden: tuple[int, ...] = (64, 64, 64)
    epochs: int = 4000
    warmup: float = 0.3
    lr: float = 3e-3
    weight_decay: float = 1e-4
    seed: int = 2026
    n_jobs: int = 1
    members: list[MLPMember] = field(default_factory=list)
    y_mean: np.ndarray | None = None
    y_std: np.ndarray | None = None
    names: tuple[str, ...] = ()
    loss_history: list[list[float]] = field(default_factory=list)

    @staticmethod
    def _x(X: np.ndarray) -> np.ndarray:
        return 2.0 * np.asarray(X, dtype=float) - 1.0

    def fit(self, X: np.ndarray, Y: np.ndarray, names: Sequence[str] | None = None) -> "MLPEnsemble":
        X = self._x(X)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        self.y_mean = Y.mean(0)
        self.y_std = np.where(Y.std(0) > 0, Y.std(0), 1.0)
        Z = (Y - self.y_mean) / self.y_std
        kw = dict(hidden=self.hidden, epochs=self.epochs, warmup=self.warmup, lr=self.lr, weight_decay=self.weight_decay)
        seeds = [self.seed + 1000 * m for m in range(self.n_members)]
        if self.n_jobs != 1:
            from joblib import Parallel, delayed

            res = Parallel(n_jobs=self.n_jobs)(delayed(_train_member)(X, Z, seed=s, **kw) for s in seeds)
        else:
            res = [_train_member(X, Z, seed=s, **kw) for s in seeds]
        T = Y.shape[1]
        self.members = []
        self.loss_history = []
        for state, hist in res:
            net = _build(X.shape[1], T, self.hidden, np.random.default_rng(0))
            net.load_state_dict(state)
            self.members.append(MLPMember(net, T))
            self.loss_history.append(hist)
        self.names = tuple(names) if names is not None else tuple(f"y{t}" for t in range(T))
        return self

    def predict_members(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-member (M, N, T) means and stds in target units."""
        if not self.members or self.y_mean is None or self.y_std is None:
            raise RuntimeError("ensemble not fitted")
        x = self._x(np.atleast_2d(X))
        mus, sds = [], []
        for m in self.members:
            mu, raw = m.forward(x)
            mus.append(self.y_mean + self.y_std * mu)
            sds.append(self.y_std * np.sqrt(softplus(raw) + MIN_VARIANCE))
        return np.array(mus), np.array(sds)

    def predict(self, X: np.ndarray, return_std: bool = True) -> tuple[np.ndarray, np.ndarray]:
        mus, sds = self.predict_members(X)
        mean = mus.mean(0)
        var = (sds**2 + mus**2).mean(0) - mean**2
        return mean, np.sqrt(np.maximum(var, 0.0))

    # ------------------------------------------------------------------ persistence
    def to_arrays(self, prefix: str = "ens.") -> dict[str, np.ndarray]:
        if self.y_mean is None or self.y_std is None:
            raise RuntimeError("ensemble not fitted")
        out = {f"{prefix}y_mean": self.y_mean, f"{prefix}y_std": self.y_std, f"{prefix}names": np.array(self.names),
               f"{prefix}hidden": np.array(self.hidden), f"{prefix}n_members": np.array(len(self.members))}
        for i, m in enumerate(self.members):
            out.update(m.net.state_dict(prefix=f"{prefix}m{i}."))
        return out

    @classmethod
    def from_arrays(cls, a: dict[str, np.ndarray], prefix: str = "ens.") -> "MLPEnsemble":
        e = cls(hidden=tuple(int(h) for h in a[f"{prefix}hidden"]))
        e.y_mean = np.asarray(a[f"{prefix}y_mean"], float)
        e.y_std = np.asarray(a[f"{prefix}y_std"], float)
        e.names = tuple(str(s) for s in a[f"{prefix}names"])
        T = len(e.y_mean)
        n_in = np.asarray(a[f"{prefix}m0.0.W"]).shape[0]
        e.members = []
        for i in range(int(a[f"{prefix}n_members"])):
            net = _build(n_in, T, e.hidden, np.random.default_rng(0))
            net.load_state_dict(a, prefix=f"{prefix}m{i}.")
            e.members.append(MLPMember(net, T))
        e.n_members = len(e.members)
        return e
