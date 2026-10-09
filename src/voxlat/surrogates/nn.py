"""A minimal numpy neural-network toolkit for the Task 6 surrogates (CPU, no torch).

Why numpy and not torch: the networks here are tiny (an MLP on 3 inputs and a
~50k-parameter 3D CNN on 32^3 voxels), the training sets have <= 290 samples, and a
dependency-free implementation trains in minutes on any laptop core, runs in CI
and in sandboxes where torch cannot be installed, and stores weights as plain
``.npz`` (no pickle, no version lock-in). Every layer has an explicit backward
pass that is checked against finite differences in ``tests/test_surrogates.py``.

Layout: dense layers take (B, F); 3D layers take channels-last (B, X, Y, Z, C).

* ``Dense``            y = x W + b (He-uniform init)
* ``Conv3dPeriodic``   3x3x3 convolution with **circular padding**, stride 1. A
  periodic unit cell has no boundary, so circular padding is the exact choice:
  the layer is equivariant to periodic translations (a rolled input gives the
  rolled output, tested).
* ``AvgPool3d``        2x2x2 mean, stride 2 (keeps periodicity when sizes are even).
* ``GlobalAvgPool3d``  mean over the cell -> (B, C); with the periodic layers
  above the network is exactly invariant to translations by multiples of the
  total pooling stride, approximately to all others.
* ``SiLU``, ``ReLU``, ``Sequential``, ``Adam`` (decoupled weight decay, AdamW).
* ``gaussian_nll``     heteroscedastic loss for (mean, log-variance) heads.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "Layer",
    "Dense",
    "Conv3dPeriodic",
    "AvgPool3d",
    "SpaceToDepth3d",
    "GlobalAvgPool3d",
    "GlobalAvgMaxPool3d",
    "SiLU",
    "ReLU",
    "Sequential",
    "Adam",
    "gaussian_nll",
    "mse_loss",
    "softplus",
    "MIN_VARIANCE",
]

#: floor of the predicted variance in standardized units (keeps the NLL bounded)
MIN_VARIANCE = 1e-6


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class Layer:
    """Base class: ``params`` / ``grads`` dicts of arrays; ``forward`` caches what ``backward`` needs."""

    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def backward(self, dy: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def parameters(self) -> list[tuple["Layer", str]]:
        return [(self, k) for k in self.params]


class Dense(Layer):
    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, dtype: Any = np.float64,
                 gain: float = 1.0) -> None:
        super().__init__()
        lim = gain * np.sqrt(6.0 / n_in)
        self.params["W"] = rng.uniform(-lim, lim, size=(n_in, n_out)).astype(dtype)
        self.params["b"] = np.zeros(n_out, dtype=dtype)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        self._x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dy: np.ndarray) -> np.ndarray:
        x = self._x
        assert x is not None
        self.grads["W"] = x.reshape(-1, x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        self.grads["b"] = dy.reshape(-1, dy.shape[-1]).sum(0)
        return dy @ self.params["W"].T


class SiLU(Layer):
    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        self._x = x
        self._s = _sigmoid(x)
        return x * self._s

    def backward(self, dy: np.ndarray) -> np.ndarray:
        s = self._s
        return dy * (s * (1.0 + self._x * (1.0 - s)))


class ReLU(Layer):
    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        self._m = x > 0
        return x * self._m

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self._m


_OFFSETS = tuple(itertools.product((0, 1, 2), repeat=3))


def _pad_wrap(x: np.ndarray) -> np.ndarray:
    return np.pad(x, ((0, 0), (1, 1), (1, 1), (1, 1), (0, 0)), mode="wrap")


def _fold_wrap(dxp: np.ndarray) -> np.ndarray:
    """Adjoint of ``_pad_wrap``: add the halo back onto the opposite faces."""
    out = dxp
    for ax in (1, 2, 3):
        d = np.moveaxis(out, ax, 0)
        core = d[1:-1].copy()
        core[0] += d[-1]   # right halo = periodic image of index 0
        core[-1] += d[0]   # left halo = periodic image of the last index
        out = np.moveaxis(core, 0, ax)
    return out


class Conv3dPeriodic(Layer):
    """3x3x3 convolution with circular padding (channels-last), via one im2col matmul.

    y[b, i, j, k, :] = sum_{o in {-1,0,1}^3} x[b, (i, j, k) + o mod n, :] @ W[o] + bias
    """

    def __init__(self, c_in: int, c_out: int, rng: np.random.Generator, dtype: Any = np.float32,
                 input_grad: bool = True) -> None:
        super().__init__()
        fan_in = 27 * c_in
        lim = np.sqrt(6.0 / fan_in)
        self.c_in, self.c_out = c_in, c_out
        self.input_grad = input_grad  # False for the first layer: skips the (unused) input gradient
        self.params["W"] = rng.uniform(-lim, lim, size=(27 * c_in, c_out)).astype(dtype)
        self.params["b"] = np.zeros(c_out, dtype=dtype)

    def _im2col(self, x: np.ndarray) -> np.ndarray:
        B, X, Y, Z, C = x.shape
        xp = _pad_wrap(x)
        cols = np.empty((B, X, Y, Z, 27, C), dtype=x.dtype)
        for o, (i, j, k) in enumerate(_OFFSETS):
            cols[..., o, :] = xp[:, i:i + X, j:j + Y, k:k + Z, :]
        return cols.reshape(B * X * Y * Z, 27 * C)

    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        if x.shape[-1] != self.c_in:
            raise ValueError(f"expected {self.c_in} input channels, got {x.shape[-1]}")
        self._shape = x.shape
        cols = self._im2col(x)
        self._cols = cols if train else None
        y = cols @ self.params["W"] + self.params["b"]
        return y.reshape(x.shape[:4] + (self.c_out,))

    def backward(self, dy: np.ndarray) -> np.ndarray:
        B, X, Y, Z, C = self._shape
        cols = self._cols
        if cols is None:
            raise RuntimeError("backward needs forward(train=True)")
        dyf = dy.reshape(-1, self.c_out)
        self.grads["W"] = cols.T @ dyf
        self.grads["b"] = dyf.sum(0)
        self._cols = None
        if not self.input_grad:
            return np.zeros(0, dtype=dy.dtype)
        dcols = (dyf @ self.params["W"].T).reshape(B, X, Y, Z, 27, C)
        dxp = np.zeros((B, X + 2, Y + 2, Z + 2, C), dtype=dy.dtype)
        for o, (i, j, k) in enumerate(_OFFSETS):
            dxp[:, i:i + X, j:j + Y, k:k + Z, :] += dcols[..., o, :]
        return _fold_wrap(dxp)


class SpaceToDepth3d(Layer):
    """Lossless (B, X, Y, Z, C) -> (B, X/2, Y/2, Z/2, 8C): each 2x2x2 block becomes channels.

    Keeps all 32^3 voxel information while running the first convolution on a 16^3
    grid (8x fewer positions); periodicity is preserved for even sizes.
    """

    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        B, X, Y, Z, C = x.shape
        if X % 2 or Y % 2 or Z % 2:
            raise ValueError(f"SpaceToDepth3d needs even sizes, got {x.shape[1:4]}")
        self._shape = x.shape
        y = x.reshape(B, X // 2, 2, Y // 2, 2, Z // 2, 2, C).transpose(0, 1, 3, 5, 2, 4, 6, 7)
        return y.reshape(B, X // 2, Y // 2, Z // 2, 8 * C)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        B, X, Y, Z, C = self._shape
        d = dy.reshape(B, X // 2, Y // 2, Z // 2, 2, 2, 2, C).transpose(0, 1, 4, 2, 5, 3, 6, 7)
        return d.reshape(self._shape)


class AvgPool3d(Layer):
    """2x2x2 average pooling, stride 2 (all spatial sizes must be even)."""

    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        B, X, Y, Z, C = x.shape
        if X % 2 or Y % 2 or Z % 2:
            raise ValueError(f"AvgPool3d needs even sizes, got {x.shape[1:4]}")
        self._shape = x.shape
        return x.reshape(B, X // 2, 2, Y // 2, 2, Z // 2, 2, C).mean(axis=(2, 4, 6))

    def backward(self, dy: np.ndarray) -> np.ndarray:
        B, X, Y, Z, C = self._shape
        d = dy[:, :, None, :, None, :, None, :] / 8.0
        return np.broadcast_to(d, (B, X // 2, 2, Y // 2, 2, Z // 2, 2, C)).reshape(self._shape)


class GlobalAvgPool3d(Layer):
    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        self._shape = x.shape
        return x.mean(axis=(1, 2, 3))

    def backward(self, dy: np.ndarray) -> np.ndarray:
        B, X, Y, Z, C = self._shape
        return np.broadcast_to(dy[:, None, None, None, :] / (X * Y * Z), self._shape).copy()


class GlobalAvgMaxPool3d(Layer):
    """Concatenate the cell mean and the cell maximum of every channel -> (B, 2C).

    The maximum lets a channel that detects a local feature (e.g. a thin neck) report
    its strongest occurrence instead of its volume fraction.
    """

    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        B, X, Y, Z, C = x.shape
        self._shape = x.shape
        flat = x.reshape(B, -1, C)
        self._arg = flat.argmax(axis=1)  # (B, C)
        return np.concatenate([flat.mean(axis=1), np.take_along_axis(flat, self._arg[:, None, :], 1)[:, 0]], axis=1)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        B, X, Y, Z, C = self._shape
        n = X * Y * Z
        d = np.broadcast_to((dy[:, :C] / n)[:, None, :], (B, n, C)).copy()
        np.add.at(d, (np.arange(B)[:, None], self._arg, np.arange(C)[None, :]), dy[:, C:])
        return d.reshape(self._shape)


class Sequential(Layer):
    def __init__(self, layers: Sequence[Layer]) -> None:
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, train)
        return x

    def backward(self, dy: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
            if dy.size == 0:  # a layer with input_grad=False: nothing further upstream needs gradients
                break
        return dy

    def parameters(self) -> list[tuple[Layer, str]]:
        return [p for layer in self.layers for p in layer.parameters()]

    def state_dict(self, prefix: str = "") -> dict[str, np.ndarray]:
        return {f"{prefix}{i}.{k}": layer.params[k] for i, layer in enumerate(self.layers) for k in layer.params}

    def load_state_dict(self, state: dict[str, np.ndarray], prefix: str = "") -> None:
        for i, layer in enumerate(self.layers):
            for k in layer.params:
                arr = np.asarray(state[f"{prefix}{i}.{k}"])
                if arr.shape != layer.params[k].shape:
                    raise ValueError(f"shape mismatch for {prefix}{i}.{k}: {arr.shape} vs {layer.params[k].shape}")
                layer.params[k] = arr.astype(layer.params[k].dtype)


class Adam:
    """Adam with decoupled weight decay (AdamW) on weights only (not biases)."""

    def __init__(self, params: Iterable[tuple[Layer, str]], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.m = [np.zeros_like(layer.params[k]) for layer, k in self.params]
        self.v = [np.zeros_like(layer.params[k]) for layer, k in self.params]
        self.t = 0

    def step(self, lr: float | None = None) -> None:
        lr = self.lr if lr is None else lr
        self.t += 1
        c1 = 1.0 - self.b1**self.t
        c2 = 1.0 - self.b2**self.t
        for i, (layer, k) in enumerate(self.params):
            g = layer.grads[k]
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * g * g
            p = layer.params[k]
            if self.wd and k == "W":
                p -= (lr * self.wd) * p
            p -= (lr * (self.m[i] / c1) / (np.sqrt(self.v[i] / c2) + self.eps)).astype(p.dtype)


def gaussian_nll(mu: np.ndarray, raw_var: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None
                 ) -> tuple[float, np.ndarray, np.ndarray]:
    """Mean Gaussian negative log-likelihood with var = softplus(raw) + MIN_VARIANCE.

    Returns (loss, d loss / d mu, d loss / d raw_var); constant log(2 pi)/2 omitted.
    ``weight`` (same shape as y, or broadcastable) masks/weights entries.
    """
    var = softplus(raw_var) + MIN_VARIANCE
    r = mu - y
    w = np.ones_like(y) if weight is None else np.broadcast_to(weight, y.shape)
    n = max(float(w.sum()), 1.0)
    loss = float(np.sum(w * 0.5 * (np.log(var) + r * r / var)) / n)
    dmu = w * (r / var) / n
    dvar = w * 0.5 * (1.0 / var - r * r / (var * var)) / n
    draw = dvar * _sigmoid(raw_var)
    return loss, dmu, draw


def mse_loss(mu: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Mean squared error / 2 and its gradient."""
    r = mu - y
    w = np.ones_like(y) if weight is None else np.broadcast_to(weight, y.shape)
    n = max(float(w.sum()), 1.0)
    return float(np.sum(w * 0.5 * r * r) / n), w * r / n
