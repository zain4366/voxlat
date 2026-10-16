"""Two-level (Jacobi + aggregation coarse space) preconditioners for the voxel solvers (Task 7).

Why
---
With a Jacobi (diagonal) preconditioner the PCG/MINRES iteration count of the
voxel solvers grows linearly with the longest grid dimension. Task 7's
finite-gap strips are up to 8 cells long across the gap but one cell wide in
the other two directions, so Jacobi needs ~3x more iterations than for a cubic
cell with the same number of unknowns (gyroid strip N = 8, n = 32: 1100-1350
elasticity iterations). pyamg removes this growth but is not always installed
(it is not in the build sandbox). This module provides a dependency-free
two-level additive preconditioner that does most of the same job:

    M^-1 r = D^-1 r + P (P^T A P + eps I)^-1 P^T r

* D = diag(A) (Jacobi smoother for the high-frequency error),
* P = unsmoothed aggregation basis: the grid is cut into boxes of
  ``block``^3 voxels (aggregates) and every aggregate carries the null-space
  modes of the operator restricted to it --
  constants for the scalar Laplacians (conduction, each Stokes velocity
  component) and the 6 rigid-body modes (3 translations + 3 infinitesimal
  rotations about the aggregate centroid) for elasticity,
* eps = 1e-8 x mean diagonal of the coarse matrix, which makes the coarse
  matrix invertible when the fine operator is only semi-definite (periodic
  translations, floating islands); the residuals of the consistent fine systems
  are orthogonal to those null vectors, so eps does not change the solution.

M^-1 is symmetric positive definite, so it can be used inside CG and as the
velocity block of the MINRES block preconditioner. The coarse matrix is sparse
(27-neighbour aggregate stencil) and factorized once with SuperLU.

Measured (build sandbox, 1 core, gyroid rho* = 0.35, n = 32, tol 1e-8):

    | problem                                  | Jacobi       | two-level (block 8; Stokes 4) |
    |------------------------------------------|--------------|-------------------------------|
    | elasticity, strip N = 4, 3 load cases    | 835 it, 70 s | 125 it, 12 s                  |
    | Stokes, strip N = 8, one direction       | 730 it, 27 s | 135 it, 7 s                   |

Results agree with the Jacobi solves to the solver tolerance (tested in
tests/test_finite_gap.py and the solver test files).

The aggregation idea is classic (algebraic multigrid by aggregation, e.g.
Vanek, Mandel & Brezina 1996, Computing 56:179; two-level additive Schwarz,
Toselli & Widlund 2005, *Domain Decomposition Methods*, ch. 3); nothing here is
new except its packaging for voxel grids.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, splu

__all__ = [
    "DEFAULT_COARSE_BLOCK",
    "box_aggregates",
    "scalar_coarse_basis",
    "rigid_body_coarse_basis",
    "TwoLevelPreconditioner",
]

#: default aggregate edge (voxels) for conduction and elasticity
DEFAULT_COARSE_BLOCK = 8


def box_aggregates(grid_index: np.ndarray, shape: Sequence[int], block: int) -> np.ndarray:
    """Aggregate label (0 ... m-1, compact) of every unknown from its grid position.

    ``grid_index``: (n, 3) integer (i, j, k) voxel/node indices of the unknowns;
    the grid of ``shape`` is cut into boxes of ``block`` voxels per edge (the
    last box along an axis may be smaller). Empty boxes get no label.
    """
    if block < 1:
        raise ValueError("block must be >= 1")
    gi = np.asarray(grid_index, dtype=np.int64)
    nb = [(int(s) + block - 1) // block for s in shape]
    key = ((gi[:, 0] // block) * nb[1] + gi[:, 1] // block) * nb[2] + gi[:, 2] // block
    _, labels = np.unique(key, return_inverse=True)
    return labels.astype(np.int64).ravel()


def scalar_coarse_basis(labels: np.ndarray) -> sp.csr_matrix:
    """Piecewise-constant aggregation basis P (n x m): P[i, labels[i]] = 1."""
    labels = np.asarray(labels, dtype=np.int64)
    n = labels.size
    m = int(labels.max()) + 1 if n else 0
    return sp.csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, m))


def rigid_body_coarse_basis(
    node_index: np.ndarray, labels: np.ndarray, block: int
) -> sp.csr_matrix:
    """Rigid-body aggregation basis for 3-D elasticity, P (3 n_nodes x 6 m).

    Unknown ordering: dof 3 a + i = displacement component i of node a (as in
    ``voxlat.homogenization.elasticity``). Aggregate g carries columns
    6 g + i (translation e_i) and 6 g + 3 + a (rotation about axis a,
    u = e_a x (x - x_g) / block, x_g = centroid of the aggregate's nodes).
    Aggregates are boxes of the node grid that do not wrap around the periodic
    boundary, so plain index differences are valid lever arms.
    """
    xi = np.asarray(node_index, dtype=float)
    labels = np.asarray(labels, dtype=np.int64)
    n_nodes = labels.size
    m = int(labels.max()) + 1 if n_nodes else 0
    cnt = np.bincount(labels, minlength=m).astype(float)
    cen = np.stack([np.bincount(labels, weights=xi[:, d], minlength=m) / cnt for d in range(3)], axis=1)
    r = (xi - cen[labels]) / float(block)
    base = 3 * np.arange(n_nodes)
    rows, cols, vals = [], [], []
    for i in range(3):
        rows.append(base + i)
        cols.append(6 * labels + i)
        vals.append(np.ones(n_nodes))
    eye = np.eye(3)
    for a in range(3):
        u = np.cross(eye[a][None, :], r)  # (n_nodes, 3)
        for i in range(3):
            if a == i:
                continue  # (e_a x r)_a = 0
            rows.append(base + i)
            cols.append(6 * labels + 3 + a)
            vals.append(u[:, i])
    P = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * n_nodes, 6 * m)
    )
    P.eliminate_zeros()
    return P


class TwoLevelPreconditioner(LinearOperator):
    """Additive two-level preconditioner M^-1 = D^-1 + P (P^T A P + eps I)^-1 P^T.

    Parameters
    ----------
    A:
        Sparse SPD or SPSD matrix (CSR), the operator to precondition.
    P:
        Sparse coarse basis (n x m), e.g. from ``scalar_coarse_basis`` or
        ``rigid_body_coarse_basis``.
    diag:
        Optional diagonal of A (default ``A.diagonal()``); entries <= 0 get a zero
        Jacobi weight.
    eps:
        Relative coarse regularization (times the mean coarse diagonal).

    Applies to vectors (n,) and to column blocks (n, k) (lock-step PCG).
    """

    def __init__(self, A: sp.spmatrix, P: sp.spmatrix, diag: np.ndarray | None = None, eps: float = 1e-8):
        A = sp.csr_matrix(A)
        n = A.shape[0]
        super().__init__(dtype=np.float64, shape=(n, n))
        d = A.diagonal() if diag is None else np.asarray(diag, dtype=float)
        self.dinv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
        self.P = sp.csr_matrix(P)
        self.PT = self.P.T.tocsr()
        Ac = (self.PT @ (A @ self.P)).tocsc()
        Ac = 0.5 * (Ac + Ac.T)
        dc = Ac.diagonal()
        scale = float(dc[dc > 0].mean()) if (dc > 0).any() else 1.0
        # empty coarse columns (no unknown) cannot occur with compact labels; guard anyway
        self.n_coarse = int(Ac.shape[0])
        self.lu = splu((Ac + eps * scale * sp.identity(self.n_coarse, format="csc")).tocsc())

    def _apply(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            return self.dinv * X + self.P @ self.lu.solve(self.PT @ X)
        Y = self.dinv[:, None] * X
        C = np.asarray(self.PT @ X)
        return Y + self.P @ self.lu.solve(np.ascontiguousarray(C))

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self._apply(np.asarray(x).ravel())

    def _matmat(self, X: np.ndarray) -> np.ndarray:
        return self._apply(X)

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        return self._matvec(x)

    def __call__(self, X: Any) -> np.ndarray:
        return self._apply(X)
