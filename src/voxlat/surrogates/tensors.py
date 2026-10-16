"""Log-Euclidean tensor maps used by the closure surrogates (Task 6).

Positive-definite tensors (permeability K, conductivity k_eff, stiffness C in
Mandel form) are learned through their **matrix logarithm** X = log(T) and
mapped back with T = exp(X). For symmetric X, exp(X) is symmetric positive
definite for *any* prediction, so symmetry and positivity hold by construction;
the eigenvalues of X are the log principal values of T and its eigenvectors the
principal axes. The map is rotation-covariant, log(R T R^T) = R log(T) R^T, so a
material-symmetry projection can be applied to X instead of T (``symmetry``).

Conventions
-----------
* Second-rank symmetric tensors <-> 6-vectors of their *plain entries* in Voigt
  order ``(xx, yy, zz, yz, xz, xy)`` (``sym3_to_vec6`` / ``vec6_to_sym3``).
* Stiffness: Voigt matrix with engineering shear strain (Task 3) <-> Mandel matrix
  ``W C W``, W = diag(1, 1, 1, sqrt2, sqrt2, sqrt2) (orthonormal basis, so
  rotations act as Q C Q^T with orthogonal Q). Symmetric 6x6 <-> 21-vectors of
  the upper triangle, row-major (``sym6_to_vec21``).
* Everything is batched over leading axes.

The derivative of the matrix exponential of a symmetric matrix X = V diag(l) V^T
in direction E is the Daleckii-Krein formula
``Dexp_X[E] = V (G o (V^T E V)) V^T``, ``G_ij = (e^li - e^lj) / (li - lj)``
(``G_ii = e^li``); it gives the first-order (delta-method) standard deviations
of every derived quantity.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "VOIGT_PAIRS",
    "MANDEL_W",
    "TRIU6",
    "sym3_to_vec6",
    "vec6_to_sym3",
    "sym6_to_vec21",
    "vec21_to_sym6",
    "sym3_basis",
    "sym6_basis",
    "voigt_to_mandel",
    "mandel_to_voigt",
    "mandel_rotation",
    "sym_eigh",
    "sym_logm",
    "sym_expm",
    "exp_divided_differences",
    "expm_frechet_sym",
]

#: Voigt order of the index pairs: xx, yy, zz, yz, xz, xy
VOIGT_PAIRS: tuple[tuple[int, int], ...] = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
#: Mandel weights (orthonormal basis of symmetric 3x3 tensors)
MANDEL_W = np.array([1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)])
#: upper-triangle (i <= j) index pairs of a 6x6 matrix, row-major
TRIU6: tuple[tuple[int, int], ...] = tuple((i, j) for i in range(6) for j in range(i, 6))

_I3 = np.array([p[0] for p in VOIGT_PAIRS])
_J3 = np.array([p[1] for p in VOIGT_PAIRS])
_I6 = np.array([p[0] for p in TRIU6])
_J6 = np.array([p[1] for p in TRIU6])


# =============================================================================
# Vectorization
# =============================================================================
def sym3_to_vec6(T: np.ndarray) -> np.ndarray:
    """(..., 3, 3) symmetric -> (..., 6) plain entries (xx, yy, zz, yz, xz, xy)."""
    T = np.asarray(T, dtype=float)
    return T[..., _I3, _J3]


def vec6_to_sym3(v: np.ndarray) -> np.ndarray:
    """Inverse of ``sym3_to_vec6``."""
    v = np.asarray(v, dtype=float)
    T = np.zeros(v.shape[:-1] + (3, 3))
    T[..., _I3, _J3] = v
    T[..., _J3, _I3] = v
    return T


def sym6_to_vec21(M: np.ndarray) -> np.ndarray:
    """(..., 6, 6) symmetric -> (..., 21) upper triangle, row-major (11, 12, ..., 16, 22, ..., 66)."""
    M = np.asarray(M, dtype=float)
    return M[..., _I6, _J6]


def vec21_to_sym6(v: np.ndarray) -> np.ndarray:
    """Inverse of ``sym6_to_vec21``."""
    v = np.asarray(v, dtype=float)
    M = np.zeros(v.shape[:-1] + (6, 6))
    M[..., _I6, _J6] = v
    M[..., _J6, _I6] = v
    return M


def sym3_basis() -> np.ndarray:
    """(6, 3, 3): B_k with sym3_to_vec6(sum_k z_k B_k) = z (B = E_ii, or E_ij + E_ji)."""
    return vec6_to_sym3(np.eye(6))


def sym6_basis() -> np.ndarray:
    """(21, 6, 6): basis matrices dual to ``sym6_to_vec21``."""
    return vec21_to_sym6(np.eye(21))


# =============================================================================
# Voigt / Mandel
# =============================================================================
def voigt_to_mandel(C: np.ndarray) -> np.ndarray:
    """Voigt stiffness (engineering shear) -> Mandel matrix W C W (batched)."""
    return MANDEL_W[:, None] * np.asarray(C, dtype=float) * MANDEL_W[None, :]


def mandel_to_voigt(Cm: np.ndarray) -> np.ndarray:
    """Inverse of ``voigt_to_mandel`` (batched)."""
    return np.asarray(Cm, dtype=float) / (MANDEL_W[:, None] * MANDEL_W[None, :])


def mandel_rotation(R: np.ndarray) -> np.ndarray:
    """6x6 orthogonal Q with mandel(R a R^T) = Q mandel(a) for symmetric 3x3 a.

    Built column by column from the orthonormal Mandel basis, so Q is exactly
    orthogonal for orthogonal R (proper or improper).
    """
    R = np.asarray(R, dtype=float)
    Q = np.empty((6, 6))
    for J, (k, l) in enumerate(VOIGT_PAIRS):
        e = np.zeros((3, 3))
        if k == l:
            e[k, k] = 1.0
        else:
            e[k, l] = e[l, k] = 1.0 / np.sqrt(2.0)
        r = R @ e @ R.T
        Q[:, J] = sym3_to_vec6(r) * MANDEL_W
    return Q


# =============================================================================
# Matrix functions of symmetric matrices (batched, via eigh)
# =============================================================================
def sym_eigh(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of (the symmetric part of) a batch of symmetric matrices."""
    A = np.asarray(A, dtype=float)
    return np.linalg.eigh(0.5 * (A + np.swapaxes(A, -1, -2)))


def sym_logm(A: np.ndarray) -> np.ndarray:
    """Matrix logarithm of SPD matrices; raises if an eigenvalue is <= 0."""
    w, V = sym_eigh(A)
    if np.any(w <= 0.0):
        raise ValueError(f"matrix logarithm needs SPD input (min eigenvalue {w.min():.3e})")
    return np.einsum("...ij,...j,...kj->...ik", V, np.log(w), V)


def sym_expm(X: np.ndarray) -> np.ndarray:
    """Matrix exponential of symmetric matrices (always SPD)."""
    w, V = sym_eigh(X)
    return np.einsum("...ij,...j,...kj->...ik", V, np.exp(w), V)


def exp_divided_differences(w: np.ndarray) -> np.ndarray:
    """G_ij = (e^wi - e^wj)/(wi - wj), G_ii = e^wi; stable for close eigenvalues."""
    w = np.asarray(w, dtype=float)
    d = w[..., :, None] - w[..., None, :]
    ej = np.exp(w)[..., None, :]
    small = np.abs(d) < 1e-8
    safe = np.where(small, 1.0, d)
    ratio = np.where(small, 1.0 + 0.5 * d, np.expm1(d) / safe)  # expm1(d)/d -> 1 + d/2
    return ej * ratio


def expm_frechet_sym(
    X: np.ndarray,
    E: np.ndarray,
    *,
    directions: bool = False,
    eig: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Directional derivative of exp at symmetric X along symmetric E (Daleckii-Krein).

    ``X``: (..., n, n). ``directions=False``: E is (..., n, n), one direction per
    matrix. ``directions=True``: E is (K, n, n) or (..., K, n, n), K directions
    (broadcast over the batch); the result is (..., K, n, n).
    """
    w, V = eig if eig is not None else sym_eigh(X)
    G = exp_divided_differences(w)
    E = np.asarray(E, dtype=float)
    if not directions:
        Et = np.einsum("...ji,...jk,...kl->...il", V, E, V)
        return np.einsum("...ij,...jk,...lk->...il", V, G * Et, V)
    # direction axis at -3
    Et = np.einsum("...ji,...djk,...kl->...dil", V, E, V)
    return np.einsum("...ij,...djk,...lk->...dil", V, G[..., None, :, :] * Et, V)


def as_rotation_list(R: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Stack rotation matrices into a (G, 3, 3) array."""
    return np.asarray(R, dtype=float).reshape(-1, 3, 3)
