"""Small CPU-trainable 3D CNN on 32^3 voxels: geometry-based prediction of K* and E*.

Question it answers (plan, Task 6c): trained on pure gyroid/diamond plus a subset
of blends, does a model that *sees the geometry* extrapolate to held-out blends
better than models that only see the parameters theta?

Input. The unit cell in normalized coordinates, ``voxelize(TPMSParams(w, rho*), 32)``
(solid = +1, fluid = -1), i.e. the topology at the target density; plus the stretch
as a side input (log a_z scaled to [-1, 1]) because a z-stretch does not change the
normalized voxel pattern. The density is not given - the network sees it.

Network (~45k parameters, float32)::

    32^3x1 --space-to-depth--> 16^3x8 --conv3 16, SiLU, avgpool--> 8^3x16
           --conv3 32, SiLU, avgpool--> 4^3x32 --conv3 32, SiLU--> global mean (32)
           [+ global max (32) with pool="avgmax"]
           --concat a_z--> dense 32, SiLU --> dense 2T (mean, variance per target)

All convolutions are periodic (circular padding), so the trunk is equivariant to
periodic translations; the global mean makes the output invariant to shifts by
multiples of 8 voxels and nearly invariant to the rest. Training augments with random
periodic shifts and the 16 operations of the cell's D4h symmetry (flips of x, y, z
and the x <-> y swap: they keep z the stretch axis, and K* = tr K / 3 and E* are
invariant under them); prediction averages the 16 operations (test-time augmentation).

Targets: log K* = log(tr K / 3 / L^2) and log E* = log(E_axial / E_s) (standardized),
Gaussian NLL after an MSE warm-up, AdamW, cosine schedule; an ensemble of M members
gives the mean and std as in ``ensemble``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from voxlat.surrogates.nn import (
    MIN_VARIANCE,
    Adam,
    AvgPool3d,
    Conv3dPeriodic,
    Dense,
    GlobalAvgMaxPool3d,
    GlobalAvgPool3d,
    Sequential,
    SiLU,
    SpaceToDepth3d,
    gaussian_nll,
    mse_loss,
    softplus,
)

__all__ = ["CNN_GRID", "cnn_voxels", "d4h_transforms", "apply_transform", "CNNEnsemble"]

#: voxels per cell edge of the CNN input
CNN_GRID = 32


def cnn_voxels(theta: np.ndarray, n: int = CNN_GRID) -> np.ndarray:
    """(N, n, n, n) bool normalized-coordinate cells of theta rows (w, rho*, a_z); a_z ignored."""
    from voxlat.geometry.tpms import TPMSParams, voxelize

    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    out = np.empty((theta.shape[0], n, n, n), dtype=bool)
    for i, (w, rho, _a) in enumerate(theta):
        out[i] = voxelize(TPMSParams(w=float(w), rho=float(rho)), n)
    return out


def d4h_transforms() -> list[tuple[bool, bool, bool, bool]]:
    """The 16 (flip_x, flip_y, flip_z, swap_xy) operations that keep z the stretch axis."""
    return list(itertools.product((False, True), repeat=4))


def apply_transform(v: np.ndarray, t: tuple[bool, bool, bool, bool], shift: Sequence[int] = (0, 0, 0)) -> np.ndarray:
    """Apply flips / x<->y swap / periodic shift to a batch (B, X, Y, Z) of cells."""
    fx, fy, fz, sw = t
    out = v
    if any(shift):
        out = np.roll(out, tuple(int(s) for s in shift), axis=(1, 2, 3))
    if fx:
        out = out[:, ::-1]
    if fy:
        out = out[:, :, ::-1]
    if fz:
        out = out[:, :, :, ::-1]
    if sw:
        out = out.transpose(0, 2, 1, 3)
    return out


def _to_input(v: np.ndarray) -> np.ndarray:
    return (2.0 * v.astype(np.float32) - 1.0)[..., None]


@dataclass
class _CNNNet:
    trunk: Sequential
    head: Sequential
    n_out: int

    @classmethod
    def build(cls, n_out: int, rng: np.random.Generator, channels: tuple[int, int, int] = (16, 32, 32),
              n_side: int = 1, hidden: int = 32, pool: str = "avg") -> "_CNNNet":
        c1, c2, c3 = channels
        if pool not in ("avg", "avgmax"):
            raise ValueError(f"pool must be 'avg' or 'avgmax', got {pool!r}")
        trunk = Sequential([
            SpaceToDepth3d(),
            Conv3dPeriodic(8, c1, rng, input_grad=False), SiLU(), AvgPool3d(),
            Conv3dPeriodic(c1, c2, rng), SiLU(), AvgPool3d(),
            Conv3dPeriodic(c2, c3, rng), SiLU(),
            GlobalAvgPool3d() if pool == "avg" else GlobalAvgMaxPool3d(),
        ])
        n_feat = c3 if pool == "avg" else 2 * c3
        last = Dense(hidden, 2 * n_out, rng, dtype=np.float32, gain=0.5)
        last.params["b"][n_out:] = np.float32(np.log(np.expm1(0.1)))
        head = Sequential([Dense(n_feat + n_side, hidden, rng, dtype=np.float32), SiLU(), last])
        return cls(trunk, head, n_out)

    def forward(self, x: np.ndarray, side: np.ndarray, train: bool = False) -> tuple[np.ndarray, np.ndarray]:
        f = self.trunk.forward(x, train)
        self._nf = f.shape[1]
        out = self.head.forward(np.concatenate([f, side.astype(np.float32)], axis=1), train)
        return out[:, : self.n_out], out[:, self.n_out:]

    def backward(self, dmu: np.ndarray, draw: np.ndarray) -> None:
        d = self.head.backward(np.concatenate([dmu, draw], axis=1).astype(np.float32))
        self.trunk.backward(d[:, : self._nf])

    def parameters(self) -> list[Any]:
        return self.trunk.parameters() + self.head.parameters()

    def state_dict(self, prefix: str = "") -> dict[str, np.ndarray]:
        return {**self.trunk.state_dict(prefix + "trunk."), **self.head.state_dict(prefix + "head.")}

    def load_state_dict(self, state: dict[str, np.ndarray], prefix: str = "") -> None:
        self.trunk.load_state_dict(state, prefix + "trunk.")
        self.head.load_state_dict(state, prefix + "head.")


def _train_cnn_member(vox: np.ndarray, side: np.ndarray, Z: np.ndarray, *, epochs: int, batch: int, lr: float,
                      weight_decay: float, warmup: float, seed: int, channels: tuple[int, int, int],
                      pool: str = "avg", threads: int | None = 1) -> tuple[dict[str, np.ndarray], list[float]]:
    try:  # one BLAS thread per worker: members run in parallel processes
        from threadpoolctl import threadpool_limits

        ctx: Any = threadpool_limits(limits=threads) if threads else None
    except ImportError:  # pragma: no cover
        ctx = None
    try:
        rng = np.random.default_rng(seed)
        N, n = vox.shape[0], vox.shape[1]
        T = Z.shape[1]
        net = _CNNNet.build(T, rng, channels, pool=pool)
        opt = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        ops = d4h_transforms()
        n_warm = int(warmup * epochs)
        Zf = Z.astype(np.float32)
        hist: list[float] = []
        for ep in range(epochs):
            lr_t = 1e-5 + 0.5 * (lr - 1e-5) * (1.0 + np.cos(np.pi * ep / epochs))
            perm = rng.permutation(N)
            tot = 0.0
            for b0 in range(0, N, batch):
                idx = perm[b0:b0 + batch]
                xb = np.empty((len(idx), n, n, n), dtype=bool)
                for j, i in enumerate(idx):
                    t = ops[rng.integers(len(ops))]
                    xb[j] = apply_transform(vox[i:i + 1], t, rng.integers(0, n, size=3))[0]
                mu, raw = net.forward(_to_input(xb), side[idx], train=True)
                if ep < n_warm:
                    loss, dmu = mse_loss(mu, Zf[idx])
                    draw = np.zeros_like(raw)
                else:
                    loss, dmu, draw = gaussian_nll(mu, raw, Zf[idx])
                net.backward(dmu, draw)
                opt.step(lr_t)
                tot += loss * len(idx)
            hist.append(tot / N)
        return net.state_dict(), hist
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)


@dataclass
class CNNEnsemble:
    """Ensemble of 3D CNNs: voxels (N, 32, 32, 32) + side features (N, 1) -> targets (N, T)."""

    n_members: int = 3
    epochs: int = 100
    batch: int = 16
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup: float = 0.25
    channels: tuple[int, int, int] = (16, 32, 32)
    pool: str = "avg"
    seed: int = 2026
    n_jobs: int = 1
    tta: bool = True
    members: list[_CNNNet] = field(default_factory=list)
    y_mean: np.ndarray | None = None
    y_std: np.ndarray | None = None
    names: tuple[str, ...] = ()
    loss_history: list[list[float]] = field(default_factory=list)

    @staticmethod
    def side_features(u: np.ndarray) -> np.ndarray:
        """Side input from the scaled design features u (N, 3): the stretch column, mapped to [-1, 1]."""
        return (2.0 * np.asarray(u, dtype=float)[:, 2:3] - 1.0).astype(np.float32)

    def fit(self, vox: np.ndarray, side: np.ndarray, Y: np.ndarray, names: Sequence[str] | None = None) -> "CNNEnsemble":
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        self.y_mean = Y.mean(0)
        self.y_std = np.where(Y.std(0) > 0, Y.std(0), 1.0)
        Z = (Y - self.y_mean) / self.y_std
        kw = dict(epochs=self.epochs, batch=self.batch, lr=self.lr, weight_decay=self.weight_decay, warmup=self.warmup,
                  channels=self.channels, pool=self.pool)
        seeds = [self.seed + 7919 * m for m in range(self.n_members)]
        vox = np.ascontiguousarray(vox, dtype=bool)
        side = np.asarray(side, dtype=np.float32)
        if self.n_jobs != 1:
            from joblib import Parallel, delayed

            res = Parallel(n_jobs=self.n_jobs)(delayed(_train_cnn_member)(vox, side, Z, seed=s, **kw) for s in seeds)
        else:
            res = [_train_cnn_member(vox, side, Z, seed=s, threads=None, **kw) for s in seeds]
        T = Y.shape[1]
        self.members, self.loss_history = [], []
        for state, hist in res:
            net = _CNNNet.build(T, np.random.default_rng(0), self.channels, pool=self.pool)
            net.load_state_dict(state)
            self.members.append(net)
            self.loss_history.append(hist)
        self.names = tuple(names) if names is not None else tuple(f"y{t}" for t in range(T))
        return self

    def predict_members(self, vox: np.ndarray, side: np.ndarray, batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
        if not self.members or self.y_mean is None or self.y_std is None:
            raise RuntimeError("CNN ensemble not fitted")
        vox = np.asarray(vox, dtype=bool)
        side = np.asarray(side, dtype=np.float32)
        ops = d4h_transforms() if self.tta else [(False, False, False, False)]
        mus, sds = [], []
        for net in self.members:
            mu_acc = np.zeros((vox.shape[0], net.n_out))
            var_acc = np.zeros_like(mu_acc)
            for t in ops:
                for b0 in range(0, vox.shape[0], batch):
                    sl = slice(b0, b0 + batch)
                    mu, raw = net.forward(_to_input(apply_transform(vox[sl], t)), side[sl])
                    mu_acc[sl] += mu
                    var_acc[sl] += softplus(raw.astype(float)) + MIN_VARIANCE
            mus.append(self.y_mean + self.y_std * mu_acc / len(ops))
            sds.append(self.y_std * np.sqrt(var_acc / len(ops)))
        return np.array(mus), np.array(sds)

    def predict(self, vox: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mus, sds = self.predict_members(vox, side)
        mean = mus.mean(0)
        var = (sds**2 + mus**2).mean(0) - mean**2
        return mean, np.sqrt(np.maximum(var, 0.0))

    # ------------------------------------------------------------------ persistence
    def to_arrays(self, prefix: str = "cnn.") -> dict[str, np.ndarray]:
        if self.y_mean is None or self.y_std is None:
            raise RuntimeError("CNN ensemble not fitted")
        out = {f"{prefix}y_mean": self.y_mean, f"{prefix}y_std": self.y_std, f"{prefix}names": np.array(self.names),
               f"{prefix}channels": np.array(self.channels), f"{prefix}pool": np.array(self.pool),
               f"{prefix}n_members": np.array(len(self.members))}
        for i, m in enumerate(self.members):
            out.update(m.state_dict(prefix=f"{prefix}m{i}."))
        return out

    @classmethod
    def from_arrays(cls, a: dict[str, np.ndarray], prefix: str = "cnn.") -> "CNNEnsemble":
        e = cls(channels=tuple(int(c) for c in a[f"{prefix}channels"]),  # type: ignore[arg-type]
                pool=str(a[f"{prefix}pool"]) if f"{prefix}pool" in a else "avg")
        e.y_mean = np.asarray(a[f"{prefix}y_mean"], float)
        e.y_std = np.asarray(a[f"{prefix}y_std"], float)
        e.names = tuple(str(s) for s in a[f"{prefix}names"])
        T = len(e.y_mean)
        e.members = []
        for i in range(int(a[f"{prefix}n_members"])):
            net = _CNNNet.build(T, np.random.default_rng(0), e.channels, pool=e.pool)
            net.load_state_dict(a, prefix=f"{prefix}m{i}.")
            e.members.append(net)
        e.n_members = len(e.members)
        return e
