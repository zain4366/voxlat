"""Exact hypervolume, box decomposition and (expected) hypervolume improvement - Task 11.

All functions MINIMIZE. ``ref`` is the reference point; points that do not strictly
dominate it contribute nothing.

* ``hypervolume(F, ref)``: exact, 1-3 objectives (3-D by slicing along the last objective,
  each slice an exact 2-D sweep; O(n^2 log n), < 0.1 s for n = 1000).
* ``box_decomposition(P, ref)``: disjoint axis-aligned boxes whose union is the part of the
  box (-inf, ref] NOT dominated by the front P (2-D or 3-D). 3-D "column" decomposition: the
  (x, y) plane is cut at every front coordinate; in column (i, j), the non-dominated part is
  z < min(ref_z, min{p_z : p_x <= x_i, p_y <= y_j}). (n_x + 1)(n_y + 1) boxes.
* ``hv_improvement(Y, boxes)``: HV(P u {y}) - HV(P) = sum_boxes prod_k (u_k - max(l_k, y_k))_+.
* ``expected_hvi(mu, sigma, boxes)``: analytic EHVI for independent Gaussian outcomes
  (Emmerich et al. 2006/2011 box form; independence makes each box term a product of 1-D
  expectations E[(u - max(l, Y))_+] = (u - l) Phi(b_l) + (u - mu)(Phi(b_u) - Phi(b_l))
  + sigma (phi(b_u) - phi(b_l)), b = (. - mu)/sigma; sigma = 0 -> deterministic).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import ndtr

__all__ = [
    "pareto_mask",
    "hypervolume",
    "hypervolume_history",
    "box_decomposition",
    "hv_improvement",
    "expected_hvi",
]

_SQ2PI = np.sqrt(2.0 * np.pi)


def pareto_mask(F: Any) -> np.ndarray:
    """Non-dominated rows (minimization). Duplicates: all copies are kept."""
    F = np.asarray(F, float)
    n = len(F)
    if n == 0:
        return np.zeros(0, bool)
    order = np.lexsort(F.T[::-1])  # sort by first objective, ties by the next
    mask = np.ones(n, bool)
    Fs = F[order]
    keep = np.ones(n, bool)
    for a in range(n):
        if not keep[a]:
            continue
        dom = np.all(Fs[a] <= Fs[a + 1:], axis=1) & np.any(Fs[a] < Fs[a + 1:], axis=1)
        keep[a + 1:] &= ~dom
    mask[order] = keep
    return mask


def _clean(F: Any, ref: Any) -> tuple[np.ndarray, np.ndarray]:
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    if F.ndim != 2 or F.shape[1] != ref.size:
        raise ValueError("F must be (n, d) with d = len(ref)")
    F = F[np.all(np.isfinite(F), axis=1)]
    F = F[np.all(F < ref, axis=1)]
    if len(F):
        F = F[pareto_mask(F)]
    return F, ref


def _hv2(F: np.ndarray, ref: np.ndarray) -> float:
    if len(F) == 0:
        return 0.0
    o = np.argsort(F[:, 0], kind="stable")
    x, y = F[o, 0], F[o, 1]
    ymin = np.minimum.accumulate(y)
    xs = np.append(x[1:], ref[0])
    return float(np.sum((xs - x) * (ref[1] - ymin)))


def hypervolume(F: Any, ref: Any) -> float:
    """Exact hypervolume dominated by F and bounded by ref (minimization, d <= 3)."""
    F, ref = _clean(F, ref)
    d = ref.size
    if len(F) == 0:
        return 0.0
    if d == 1:
        return float(ref[0] - F[:, 0].min())
    if d == 2:
        return _hv2(F, ref)
    if d != 3:
        raise NotImplementedError("exact hypervolume implemented for d <= 3")
    o = np.argsort(F[:, 2], kind="stable")
    F = F[o]
    z = np.append(F[:, 2], ref[2])
    hv = 0.0
    for k in range(len(F)):
        dz = z[k + 1] - z[k]
        if dz > 0:
            hv += dz * _hv2(F[: k + 1, :2], ref[:2])
    return float(hv)


def hypervolume_history(F: Any, ref: Any, feasible: Any = None, checkpoints: Any = None) -> np.ndarray:
    """HV of the feasible non-dominated set of the first k rows, for each k in ``checkpoints``."""
    F = np.asarray(F, float)
    n = len(F)
    ok = np.ones(n, bool) if feasible is None else np.asarray(feasible, bool)
    ks = np.arange(1, n + 1) if checkpoints is None else np.asarray(checkpoints, int)
    out = np.zeros(len(ks))
    for a, k in enumerate(ks):
        sel = ok[:k]
        out[a] = hypervolume(F[:k][sel], ref) if sel.any() else 0.0
    return out


def box_decomposition(P: Any, ref: Any) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) arrays (m, d) of disjoint boxes covering the non-dominated region below ref.

    Lower bounds may be -inf. d = 2 or 3. An empty front gives the single box (-inf, ref).
    """
    P, ref = _clean(P, ref)
    d = ref.size
    if d not in (2, 3):
        raise NotImplementedError("box decomposition implemented for d = 2, 3")
    if len(P) == 0:
        return np.full((1, d), -np.inf), ref[None, :].copy()
    xs = np.unique(P[:, 0])
    X = np.concatenate([[-np.inf], xs, [ref[0]]])
    rx = np.searchsorted(xs, P[:, 0])  # 0-based x rank
    if d == 2:
        M = np.full(len(xs), np.inf)
        np.minimum.at(M, rx, P[:, 1])
        thr = np.concatenate([[np.inf], np.minimum.accumulate(M)])  # column i: points with x <= X[i]
        up_y = np.minimum(thr, ref[1])
        lo = np.column_stack([X[:-1], np.full(len(thr), -np.inf)])
        hi = np.column_stack([X[1:], up_y])
    else:
        ys = np.unique(P[:, 1])
        Y = np.concatenate([[-np.inf], ys, [ref[1]]])
        ry = np.searchsorted(ys, P[:, 1])
        M = np.full((len(xs), len(ys)), np.inf)
        np.minimum.at(M, (rx, ry), P[:, 2])
        CM = np.minimum.accumulate(np.minimum.accumulate(M, axis=0), axis=1)
        thr = np.full((len(xs) + 1, len(ys) + 1), np.inf)
        thr[1:, 1:] = CM
        up_z = np.minimum(thr, ref[2])
        I, J = np.meshgrid(np.arange(len(X) - 1), np.arange(len(Y) - 1), indexing="ij")
        lo = np.column_stack([X[I.ravel()], Y[J.ravel()], np.full(I.size, -np.inf)])
        hi = np.column_stack([X[I.ravel() + 1], Y[J.ravel() + 1], up_z.ravel()])
    keep = np.all(hi > lo, axis=1)
    return lo[keep], hi[keep]


def hv_improvement(Y: Any, boxes: tuple[np.ndarray, np.ndarray], chunk: int = 4096) -> np.ndarray:
    """HVI of each row of Y (n, d) with respect to the front behind ``boxes``."""
    lo, hi = boxes
    Y = np.atleast_2d(np.asarray(Y, float))
    out = np.empty(len(Y))
    for a in range(0, len(Y), chunk):
        y = Y[a:a + chunk, None, :]
        ext = np.clip(hi[None] - np.maximum(lo[None], y), 0.0, None)
        out[a:a + chunk] = np.prod(ext, axis=2).sum(axis=1)
    return out


def _e_term(lo: np.ndarray, hi: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """E[(hi - max(lo, Y))_+], Y ~ N(mu, sd^2); broadcasting; lo may be -inf; sd may be 0."""
    det = sd <= 0
    sd_ = np.where(det, 1.0, sd)
    bu = (hi - mu) / sd_
    with np.errstate(invalid="ignore"):
        bl = np.where(np.isinf(lo), -np.inf, (lo - mu) / sd_)
    Pu, Pl = ndtr(bu), ndtr(bl)
    pu = np.exp(-0.5 * bu**2) / _SQ2PI
    pl = np.where(np.isinf(bl), 0.0, np.exp(-0.5 * np.where(np.isinf(bl), 0.0, bl) ** 2) / _SQ2PI)
    with np.errstate(invalid="ignore"):
        first = np.where(np.isinf(lo), 0.0, (hi - lo) * Pl)
    val = first + (hi - mu) * (Pu - Pl) + sd_ * (pu - pl)
    deterministic = np.clip(hi - np.maximum(lo, mu), 0.0, None)
    return np.where(det, deterministic, np.clip(val, 0.0, None))


def expected_hvi(mu: Any, sigma: Any, boxes: tuple[np.ndarray, np.ndarray], chunk: int = 512) -> np.ndarray:
    """Analytic EHVI for independent Gaussian outcomes (mu, sigma: (n, d)); sigma 0 = known outcome."""
    lo, hi = boxes
    mu = np.atleast_2d(np.asarray(mu, float))
    sd = np.atleast_2d(np.asarray(sigma, float))
    out = np.empty(len(mu))
    for a in range(0, len(mu), chunk):
        m = mu[a:a + chunk, None, :]
        s = sd[a:a + chunk, None, :]
        terms = _e_term(lo[None], hi[None], m, s)
        out[a:a + chunk] = np.prod(terms, axis=2).sum(axis=1)
    return out
