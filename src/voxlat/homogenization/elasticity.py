"""Periodic effective stiffness C_eff (6x6) and stress localization of a voxel cell (Task 3).

Governing equations
-------------------
Linear elasticity, no body force, in a periodic cell Y with a piecewise-constant
isotropic material per voxel (Young's modulus E(x), common Poisson ratio nu):

    div sigma = 0,   sigma = C(x) : eps,   eps = sym grad u               in Y
    u(x) = Ebar . x + u~(x),   u~ Y-periodic                             (cell problem)

Ebar is the imposed macroscopic strain and u~ the periodic fluctuation. Periodicity
of u~ together with div sigma = 0 makes the traction anti-periodic on opposite faces
(equilibrium across the cell boundary). The effective tensor is defined by the
volume-averaged stress,

    <sigma> = C_eff : Ebar,

and, equivalently (Hill-Mandel), by the averaged strain energy,

    Ebar_i : C_eff : Ebar_j = < eps^(i) : C : eps^(j) >,   eps^(i) = field for Ebar = e_i.

Six unit macroscopic strains give the six columns of C_eff.

Voigt / Mandel convention (stated once, used everywhere)
--------------------------------------------------------
Component order 1..6 = xx, yy, zz, yz, xz, xy (standard Voigt order).

* **Voigt (the reported ``C_eff``)**: stress vector [s_xx, s_yy, s_zz, s_yz, s_xz, s_xy],
  strain vector with ENGINEERING shear strains [e_xx, e_yy, e_zz, g_yz, g_xz, g_xy],
  g_ij = 2 e_ij. Then sigma = C_eff eps and C_44 = C_55 = C_66 = mu for an isotropic
  solid. Load case j sets the j-th entry of this strain vector to 1 (case 4 is
  g_yz = 1, i.e. e_yz = e_zy = 1/2). The compliance S = C_eff^-1 has S_44 = 1/G_yz and
  1/S_ii = E_i (i <= 3).
* **Mandel** (``C_mandel``): both vectors use sqrt(2) x tensor shear components, so
  C_mandel = W C_eff W with W = diag(1, 1, 1, sqrt2, sqrt2, sqrt2). C_mandel is the
  matrix of the 4th-order tensor in an orthonormal basis of symmetric 2nd-order
  tensors: its eigenvalues are the true Kelvin moduli, used for the positive-
  definiteness check and any rotation of the tensor.

Discretization: 8-node trilinear hexahedra on voxels
-----------------------------------------------------
Every voxel is one cubic Q1 element (voxel edge h; in 3-D the element stiffness
scales as E h and the cell volume as h^3 while C_eff is h-independent, so the code
works in voxel units h = 1). Nodes sit on voxel corners and are periodic: node
(i, j, k) is shared by the voxels around it with indices taken mod (nx, ny, nz).
2 x 2 x 2 Gauss quadrature (exact for the Q1 stiffness of a cube) gives, for E = 1,

    K1 = int B^T D1 B dV  (24 x 24),     F1 = int B^T D1 dV  (24 x 6),

D1 the isotropic Voigt stiffness for E = 1. The discrete cell problem for load case j is

    K u~ = f_j,     K = sum_e E_e K1,     f_j = - sum_e E_e F1 Ebar_j

(assembled with the periodic node numbering; f_j is the self-equilibrated force of
the unit eigen-strain). C_eff is evaluated two ways, both exact for the discrete problem:

    flux (average stress):  C_flux[:, j] = (1/V) sum_e E_e (D1 Ebar_j + F1^T u_e^(j))
    energy:                 C_en[i, j]  = (1/V) sum_e E_e (u_e^(i) + Ebar_i)-energy
                                       = (1/V) sum_e E_e [Ebar_i.D1 Ebar_j + (F1^T u_e^(j))_i
                                                          + (F1^T u_e^(i))_j + u_e^(i).K1 u_e^(j)]

with V = number of voxels. They agree to the solver tolerance (discrete Hill-Mandel);
``effective_elasticity`` raises if they do not. The energy form is symmetric by
construction and is reported as ``C_eff``.

Void voxels: removed (E_void = 0), not a soft E_min material
------------------------------------------------------------
The coolant has no shear stiffness, so E_void = 0 is the physically exact limit and
the default. Elements with E = 0 are dropped and nodes that touch no solid voxel
carry no unknowns. Compared with a soft void E_min = 1e-6..1e-9 E_s on all voxels:

* 2.5x fewer unknowns and ~2.6x less time (gyroid rho* = 0.3, n = 24: 16 443 vs
  41 472 DOFs; only the nodes touching solid are active), and
* no bias: a soft void shifts C_eff by ~22 E_min/E_s relative (measured: 2.2e-3 at
  E_min = 1e-4 E_s, 2.2 % at 1e-3, 22 % at 1e-2 -- the lattice itself is only ~5 %
  as stiff as the solid, which amplifies the void's contribution). With Jacobi
  scaling the iteration count is not affected by the contrast either way.

The price is that K becomes singular in more ways than the 3 periodic translations:
floating solid islands (rigid-body modes) and voxels hanging on a single edge or
corner node (hinge mechanisms). All these null vectors w satisfy K w = 0 and
B w = 0 element-wise, so (i) the load f_j is orthogonal to them (the system is
consistent, CG converges), and (ii) they contribute nothing to C_eff or to element
stresses. The per-component translations are additionally projected out of f_j
(round-off) and out of the solution (cosmetic). A soft void is available for
comparison via ``void_modulus`` (tests show it converges to the removed-void result
linearly in E_min/E_s).

Solver
------
Preconditioned conjugate gradients (scipy ``cg``), relative residual tol 1e-8.
Preconditioner: pyamg smoothed aggregation (V-cycle, 3x3 nodal blocks, translation
near-null space) if pyamg is installed, else Jacobi (diagonal). Nodal 3x3
block-Jacobi is available but saves only ~4 % of the iterations and costs more per
iteration (Task 3 timings). The operator is
the assembled CSR matrix (vectorized, chunked COO assembly; required for AMG) or a
matrix-free element-by-element product (``operator="matrix_free"``; memory ~25x
smaller, used automatically for very large grids).

Stress localization
-------------------
Element stresses are evaluated at the element centroid, which for a cubic Q1 element
equals the element-average stress. For a macroscopic stress Sigma, the macroscopic
strain is Ebar = S Sigma (S = C_eff^-1) and the local stress field is the
superposition of the six unit cases. The localization factor of solid voxel e is

    K_e = sigma_vm(e) / Sigma_vm,

(von Mises of the local over the von Mises of the macroscopic stress), reported as
max and 99th percentile over the solid voxels. Two standard load cases: uniaxial
Sigma_zz = 1 (Sigma_vm = 1) and shear Sigma_xz = 1 (Sigma_vm = sqrt 3). Use:
sigma_vm,local,max ~ K_p99 x Sigma_vm,macro (the max is a voxel-corner singularity
that grows with resolution; see STATUS.md, Task 3).

Units: E in Pa (any consistent unit), C_eff in the same unit. Defaults come from the
reference config (AlSi10Mg: E = 70 GPa, nu = 0.33).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, cg

from voxlat.utils.log import get_logger

__all__ = [
    "VOIGT_ORDER",
    "MANDEL_W",
    "isotropic_stiffness",
    "lame_parameters",
    "voigt_to_mandel",
    "mandel_to_voigt",
    "rotate_stiffness",
    "von_mises",
    "hex8_element",
    "StressLocalization",
    "ElasticityResult",
    "effective_elasticity",
    "effective_elasticity_tpms",
    "ExtrapolatedElasticity",
    "extrapolated_elasticity_tpms",
    "laminate_stiffness",
    "hashin_shtrikman_porous",
    "voigt_reuss_hill",
    "youngs_extremes",
    "cubic_deviation",
    "pyamg_available",
    "averaged_elasticity_tpms",
    "AveragedElasticity",
    "grid_offsets",
    "MACRO_STRESS_CASES",
    "ALL_STRESS_CASES",
]

LOG = get_logger("homogenization.elasticity")

Preconditioner = Literal["auto", "amg", "block_jacobi", "jacobi", "none"]
Operator = Literal["auto", "assembled", "matrix_free"]

#: Voigt / Mandel component order
VOIGT_ORDER: tuple[str, ...] = ("xx", "yy", "zz", "yz", "xz", "xy")
#: Mandel weights: C_mandel = diag(W) C_voigt diag(W)
MANDEL_W = np.array([1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)])

#: Macroscopic stress cases for the localization factors (Voigt stress vectors, unit magnitude).
#: The default ``localization`` of ``effective_elasticity`` is ("uniaxial_z", "shear_xz");
#: the other four complete the six unit stresses (Task 5 dataset: stretched cells and blends
#: are not cubic, so e.g. uniaxial_x != uniaxial_z and shear_xy != shear_xz there).
MACRO_STRESS_CASES: dict[str, np.ndarray] = {
    "uniaxial_z": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    "shear_xz": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    "uniaxial_x": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "uniaxial_y": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    "shear_yz": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    "shear_xy": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
}
ALL_STRESS_CASES: tuple[str, ...] = (
    "uniaxial_x", "uniaxial_y", "uniaxial_z", "shear_yz", "shear_xz", "shear_xy",
)

# element corner a = cx + 2 cy + 4 cz  (x fastest)
_CORNERS = np.array([[a & 1, (a >> 1) & 1, (a >> 2) & 1] for a in range(8)])
# assembled-matrix size above which "auto" switches to the matrix-free operator
_AUTO_MATRIX_FREE_NNZ = 6.0e7


# =============================================================================
# Material and tensor helpers
# =============================================================================
def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    """(lambda, mu) of an isotropic solid."""
    if not -1.0 < nu < 0.5:
        raise ValueError(f"Poisson ratio must be in (-1, 0.5), got {nu}")
    return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), E / (2.0 * (1.0 + nu))


def isotropic_stiffness(E: float, nu: float) -> np.ndarray:
    """6x6 Voigt stiffness (engineering shear strains, order xx yy zz yz xz xy)."""
    lam, mu = lame_parameters(E, nu)
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[[0, 1, 2], [0, 1, 2]] += 2.0 * mu
    D[[3, 4, 5], [3, 4, 5]] = mu
    return D


def voigt_to_mandel(C: np.ndarray) -> np.ndarray:
    """Voigt stiffness (engineering shear) -> Mandel matrix W C W."""
    return MANDEL_W[:, None] * np.asarray(C) * MANDEL_W[None, :]


def mandel_to_voigt(Cm: np.ndarray) -> np.ndarray:
    """Inverse of ``voigt_to_mandel``."""
    return np.asarray(Cm) / (MANDEL_W[:, None] * MANDEL_W[None, :])


def _mandel_rotation(R: np.ndarray) -> np.ndarray:
    """6x6 orthogonal matrix Q with (R a R^T)_mandel = Q a_mandel."""
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    Q = np.empty((6, 6))
    # for k != l: basis tensor (e_k e_l + e_l e_k)/sqrt2 ; rows scaled for Mandel
    for I, (i, j) in enumerate(pairs):
        for J, (k, l) in enumerate(pairs):
            if k == l:
                v = R[i, k] * R[j, k]
            else:
                v = (R[i, k] * R[j, l] + R[i, l] * R[j, k]) / np.sqrt(2.0)
            Q[I, J] = v * (np.sqrt(2.0) if i != j else 1.0)
    return Q


def rotate_stiffness(C: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Voigt stiffness of the material rotated by the 3x3 rotation R (x' = R x)."""
    Q = _mandel_rotation(np.asarray(R, dtype=float))
    return mandel_to_voigt(Q @ voigt_to_mandel(C) @ Q.T)


def von_mises(s: np.ndarray) -> np.ndarray:
    """von Mises stress of Voigt stress vectors s[..., 6] (true shear stresses)."""
    s = np.asarray(s)
    a, b, c = s[..., 0], s[..., 1], s[..., 2]
    return np.sqrt(
        0.5 * ((a - b) ** 2 + (b - c) ** 2 + (c - a) ** 2)
        + 3.0 * (s[..., 3] ** 2 + s[..., 4] ** 2 + s[..., 5] ** 2)
    )


# =============================================================================
# Element
# =============================================================================
def _B(xi: np.ndarray) -> np.ndarray:
    """Strain-displacement matrix (6 x 24, engineering shear) of the unit cube at xi."""
    B = np.zeros((6, 24))
    for a, (cx, cy, cz) in enumerate(_CORNERS):
        f = [xi[d] if c else 1.0 - xi[d] for d, c in enumerate((cx, cy, cz))]
        s = [1.0 if c else -1.0 for c in (cx, cy, cz)]
        dx, dy, dz = s[0] * f[1] * f[2], s[1] * f[0] * f[2], s[2] * f[0] * f[1]
        c = 3 * a
        B[0, c] = dx
        B[1, c + 1] = dy
        B[2, c + 2] = dz
        B[3, c + 1], B[3, c + 2] = dz, dy
        B[4, c], B[4, c + 2] = dz, dx
        B[5, c], B[5, c + 1] = dy, dx
    return B


@lru_cache(maxsize=16)
def hex8_element(nu: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(K1, F1, B_centre, D1) of a unit-cube Q1 element with E = 1.

    K1 = int B^T D1 B (24x24), F1 = int B^T D1 (24x6), both by 2x2x2 Gauss
    quadrature (exact). B_centre is B at the centroid (= element-average B).
    DOF order: 3 * corner + component, corner a = cx + 2 cy + 4 cz.
    """
    D1 = isotropic_stiffness(1.0, nu)
    g = 0.5 + np.array([-1.0, 1.0]) / (2.0 * np.sqrt(3.0))
    K1 = np.zeros((24, 24))
    F1 = np.zeros((24, 6))
    for x in g:
        for y in g:
            for z in g:
                B = _B(np.array([x, y, z]))
                K1 += B.T @ D1 @ B / 8.0
                F1 += B.T @ D1 / 8.0
    K1 = 0.5 * (K1 + K1.T)
    for arr in (K1, F1, D1):
        arr.setflags(write=False)
    Bc = _B(np.array([0.5, 0.5, 0.5]))
    Bc.setflags(write=False)
    return K1, F1, Bc, D1


# =============================================================================
# Mesh (periodic voxel grid -> elements, active nodes)
# =============================================================================
@dataclass
class _Mesh:
    shape: tuple[int, int, int]
    E_el: np.ndarray  # (Ne,) modulus per active element
    el_voxel: np.ndarray  # (Ne,) flat voxel index of each element
    edof: np.ndarray  # (Ne, 24) int32/64 dof indices (active numbering)
    n_nodes: int  # active nodes
    comp: np.ndarray  # (n_nodes,) connected-component label
    n_comp: int


def _build_mesh(Efield: np.ndarray) -> _Mesh:
    shape = tuple(int(s) for s in Efield.shape)
    nx, ny, nz = shape
    vox = np.flatnonzero(Efield.ravel() > 0)
    i, j, k = np.unravel_index(vox, shape)
    nid = np.stack(
        [((i + cx) % nx * ny + (j + cy) % ny) * nz + (k + cz) % nz for cx, cy, cz in _CORNERS],
        axis=1,
    )
    used = np.zeros(nx * ny * nz, dtype=bool)
    used[nid.ravel()] = True
    remap = np.full(used.size, -1, dtype=np.int64)
    n_nodes = int(used.sum())
    remap[used] = np.arange(n_nodes)
    en = remap[nid]
    itype = np.int32 if 3 * n_nodes < 2**31 - 1 else np.int64
    edof = (3 * en[:, :, None] + np.arange(3)).reshape(-1, 24).astype(itype)
    # node connectivity graph: corner 0 to the other 7 corners of every element
    if n_nodes:
        r = np.repeat(en[:, 0], 7)
        c = en[:, 1:].ravel()
        Gm = sp.coo_matrix((np.ones(r.size, dtype=np.int8), (r, c)), shape=(n_nodes, n_nodes))
        n_comp, comp = connected_components(Gm, directed=False)
    else:
        n_comp, comp = 0, np.zeros(0, dtype=np.int64)
    return _Mesh(shape, Efield.ravel()[vox].astype(np.float64), vox, edof, n_nodes,
                 comp.astype(np.int64), int(n_comp))


def _assemble(mesh: _Mesh, K1: np.ndarray, chunk_entries: int = 8_000_000) -> sp.csr_matrix:
    """Global stiffness (CSR), vectorized COO assembly in element chunks (bounded memory)."""
    ndof = 3 * mesh.n_nodes
    Ne = mesh.E_el.size
    per = max(1, chunk_entries // 576)
    K = None
    k1 = K1.ravel()
    for s in range(0, Ne, per):
        ed = mesh.edof[s : s + per]
        rows = np.repeat(ed, 24, axis=1).ravel()
        cols = np.tile(ed, (1, 24)).ravel()
        vals = (mesh.E_el[s : s + per, None] * k1[None, :]).ravel()
        Kc = sp.csr_matrix((vals, (rows, cols)), shape=(ndof, ndof))
        Kc.sum_duplicates()
        K = Kc if K is None else K + Kc
    if K is None:
        K = sp.csr_matrix((ndof, ndof))
    K.eliminate_zeros()
    return K


class _MatrixFree(LinearOperator):
    """y = K x element by element (no assembled matrix)."""

    def __init__(self, mesh: _Mesh, K1: np.ndarray) -> None:
        ndof = 3 * mesh.n_nodes
        super().__init__(dtype=np.float64, shape=(ndof, ndof))
        self.mesh = mesh
        self.K1 = np.ascontiguousarray(K1)
        self.flat = mesh.edof.ravel()

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        xe = x.ravel()[self.mesh.edof]  # (Ne, 24)
        ye = (xe @ self.K1) * self.mesh.E_el[:, None]
        return np.bincount(self.flat, weights=ye.ravel(), minlength=self.shape[0])

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        return self._matvec(x)

    def _matmat(self, X: np.ndarray) -> np.ndarray:
        Xt = np.ascontiguousarray(np.asarray(X).T)
        # one GEMM-backed product per column beats a batched einsum
        return np.stack([self._matvec(x) for x in Xt], axis=1)


def _block_diagonal(mesh: _Mesh, K1: np.ndarray) -> np.ndarray:
    """3x3 nodal diagonal blocks of K, shape (n_nodes, 3, 3)."""
    nodes = mesh.edof[:, ::3] // 3  # (Ne, 8)
    blocks = np.zeros((mesh.n_nodes, 3, 3))
    for a in range(8):
        Ka = K1[3 * a : 3 * a + 3, 3 * a : 3 * a + 3]
        w = np.bincount(nodes[:, a], weights=mesh.E_el, minlength=mesh.n_nodes)
        blocks += w[:, None, None] * Ka[None]
    return blocks


def pyamg_available() -> bool:
    """True if pyamg can be imported (enables the AMG preconditioner)."""
    try:
        import pyamg  # noqa: F401
    except ImportError:
        return False
    return True


class _Precond(LinearOperator):
    """Preconditioner z = M r that also applies column-wise to (ndof, k) blocks."""

    def __init__(self, ndof: int, fn: Any) -> None:
        super().__init__(dtype=np.float64, shape=(ndof, ndof))
        self.fn = fn

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self.fn(np.asarray(x).ravel())

    def _matmat(self, X: np.ndarray) -> np.ndarray:
        return self.fn(np.asarray(X))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.fn(X)


def _pcg_block(
    A: Any, B: np.ndarray, M: Any, tol: float, maxiter: int
) -> tuple[np.ndarray, list[int], list[float], list[bool]]:
    """k independent preconditioned CG solves run in lock-step (A @ X for all columns at once).

    Mathematically identical to k separate PCG runs (separate step lengths per column,
    no block-Krylov coupling); a converged column is frozen. The single sparse
    matrix pass for k right-hand sides is ~2x faster than k passes (memory bound).
    Convergence: ||b - A x|| <= tol ||b|| per column (true residual re-checked at the end).
    """
    n, k = B.shape
    X = np.zeros((n, k))
    R = B.copy()
    bn = np.linalg.norm(B, axis=0)
    active = bn > 0
    iters = np.zeros(k, dtype=int)
    if not active.any():
        return X, iters.tolist(), [0.0] * k, [True] * k
    Mf = (lambda Y: Y) if M is None else M
    Z = Mf(R)
    P = Z.copy()
    rz = np.einsum("ij,ij->j", R, Z)
    for it in range(1, maxiter + 1):
        AP = A @ P
        pAp = np.einsum("ij,ij->j", P, AP)
        alpha = np.where(active & (pAp > 0), rz / np.where(pAp > 0, pAp, 1.0), 0.0)
        X += alpha * P
        R -= alpha * AP
        rn = np.linalg.norm(R, axis=0)
        done = active & (rn <= tol * bn)
        iters[done] = it
        active &= ~done
        if not active.any():
            break
        Z = Mf(R)
        rz_new = np.einsum("ij,ij->j", R, Z)
        beta = np.where(active, rz_new / np.where(rz != 0, rz, 1.0), 0.0)
        P = Z + beta * P
        rz = rz_new
    iters[active] = maxiter
    Rt = B - A @ X
    res = [float(np.linalg.norm(Rt[:, c]) / bn[c]) if bn[c] > 0 else 0.0 for c in range(k)]
    return X, iters.tolist(), res, (~active).tolist()


def _preconditioner(
    kind: str, mesh: _Mesh, K1: np.ndarray, K: sp.csr_matrix | None
) -> tuple[Any, str]:
    if kind == "auto":
        kind = "amg" if (pyamg_available() and K is not None) else "jacobi"
    ndof = 3 * mesh.n_nodes
    if kind == "none":
        return None, "none"
    if kind == "jacobi":
        d = np.einsum("nii->ni", _block_diagonal(mesh, K1)).ravel()
        dinv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
        return _Precond(ndof, lambda X: dinv.reshape(-1, *([1] * (X.ndim - 1))) * X), "jacobi"
    if kind == "block_jacobi":
        Bd = _block_diagonal(mesh, K1)
        # every active node belongs to >= 1 element -> its block is SPD
        Binv = np.linalg.inv(Bd)

        def mv(X: np.ndarray) -> np.ndarray:
            if X.ndim == 1:
                return np.einsum("nij,nj->ni", Binv, X.reshape(-1, 3)).ravel()
            return np.einsum("nij,njc->nic", Binv, X.reshape(-1, 3, X.shape[1])).reshape(X.shape)

        return _Precond(ndof, mv), "block_jacobi"
    if kind == "amg":
        if K is None:
            raise ValueError("preconditioner='amg' needs operator='assembled'")
        try:
            import pyamg
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("preconditioner='amg' needs pyamg (pip install pyamg)") from exc
        Bns = np.kron(np.ones((mesh.n_nodes, 1)), np.eye(3))  # 3 translations
        ml = pyamg.smoothed_aggregation_solver(
            K.tobsr(blocksize=(3, 3)), B=Bns, symmetry="hermitian", max_coarse=500
        )
        Mamg = ml.aspreconditioner(cycle="V")

        def mv_amg(X: np.ndarray) -> np.ndarray:
            if X.ndim == 1:
                return Mamg @ X
            return np.stack([Mamg @ X[:, c] for c in range(X.shape[1])], axis=1)

        return _Precond(ndof, mv_amg), "amg"
    raise ValueError(f"unknown preconditioner {kind!r}")


# =============================================================================
# Results
# =============================================================================
@dataclass
class StressLocalization:
    """Von Mises localization statistics for one macroscopic stress state.

    ``factor`` arrays are sigma_vm(voxel) / Sigma_vm(macro) over solid voxels.
    """

    name: str
    macro_stress: np.ndarray  # Voigt, unit magnitude
    macro_von_mises: float
    max: float
    p99: float
    p999: float
    mean: float  # volume average over solid of sigma_vm / Sigma_vm
    argmax_voxel: tuple[int, int, int]
    factor: np.ndarray | None = field(default=None, repr=False)  # per solid voxel (float32)


@dataclass
class ElasticityResult:
    """Output of ``effective_elasticity``.

    Attributes
    ----------
    C_eff:
        6x6 effective stiffness, Voigt notation (engineering shear strains, order
        xx yy zz yz xz xy), energy form (symmetric). Same unit as E.
    C_eff_flux:
        Same from the volume-averaged stress (column j = response to unit strain j).
    agreement:
        max|C_flux - C_energy| / max|C_energy| (~ solver tol).
    E_s, nu_s, solid_fraction, shape:
        Inputs (E_s is None for a general modulus field).
    localization:
        name -> StressLocalization (see ``MACRO_STRESS_CASES``).
    iterations, residuals, case_times, solve_time:
        PCG iterations and final true relative residuals per load case; wall time
        per load case (equal shares of ``solve_time`` for the lock-step solver).
    preconditioner, operator, wall_time, assembly_time:
        Solver details; ``wall_time`` is total (mesh, assembly, 6 solves, post-processing).
    n_dofs, n_elements, n_components:
        Unknowns, active (solid) elements, connected node components (floating islands > 1).
    unit_stresses:
        (6, Ne, 6) float32 element-average stresses for the six unit strains, if
        ``return_fields`` (rows of axis 0 = load case, last axis = Voigt stress).
    element_voxel, element_solid:
        flat voxel index of each element and whether it is solid (not soft void).
    """

    C_eff: np.ndarray
    C_eff_flux: np.ndarray
    agreement: float
    E_s: float | None
    nu_s: float
    solid_fraction: float | None
    shape: tuple[int, int, int]
    localization: dict[str, StressLocalization]
    iterations: list[int]
    residuals: list[float]
    case_times: list[float]
    solve_time: float
    preconditioner: str
    operator: str
    wall_time: float
    assembly_time: float
    n_dofs: int
    n_elements: int
    n_components: int
    element_voxel: np.ndarray = field(repr=False)
    element_solid: np.ndarray = field(repr=False)
    unit_stresses: np.ndarray | None = field(default=None, repr=False)

    # ---- tensor views ----------------------------------------------------------
    @property
    def C_mandel(self) -> np.ndarray:
        return voigt_to_mandel(self.C_eff)

    @property
    def S(self) -> np.ndarray:
        """Voigt compliance (engineering shear): eps = S sigma."""
        return np.linalg.inv(self.C_eff)

    @property
    def eigenvalues(self) -> np.ndarray:
        """Kelvin moduli: eigenvalues of the Mandel matrix (ascending)."""
        return np.linalg.eigvalsh(self.C_mandel)

    @property
    def is_spd(self) -> bool:
        ev = self.eigenvalues
        sym = np.allclose(self.C_eff, self.C_eff.T, rtol=0, atol=1e-12 * max(abs(ev[-1]), 1e-300))
        return bool(sym and ev[0] > 1e-12 * ev[-1])

    def relative(self, E_ref: float | None = None) -> np.ndarray:
        ref = self.E_s if E_ref is None else E_ref
        if not ref:
            raise ValueError("need a reference modulus")
        return self.C_eff / ref

    # ---- engineering constants -----------------------------------------------
    @property
    def youngs_moduli(self) -> np.ndarray:
        """(E_x, E_y, E_z) = 1/S_11, 1/S_22, 1/S_33."""
        return 1.0 / np.diag(self.S)[:3]

    @property
    def shear_moduli(self) -> np.ndarray:
        """(G_yz, G_xz, G_xy) = 1/S_44, 1/S_55, 1/S_66."""
        return 1.0 / np.diag(self.S)[3:]

    @property
    def E_axial(self) -> float:
        """Mean of the three axial Young's moduli."""
        return float(np.mean(self.youngs_moduli))

    def youngs_modulus(self, direction: Sequence[float]) -> float:
        """Young's modulus along a unit direction d: 1/E = v^T S v, v = (d1^2, d2^2, d3^2, d2d3, d1d3, d1d2)."""
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        v = np.array([d[0] ** 2, d[1] ** 2, d[2] ** 2, d[1] * d[2], d[0] * d[2], d[0] * d[1]])
        return float(1.0 / (v @ self.S @ v))

    def youngs_extremes(self, n_dirs: int = 2000) -> tuple[float, float]:
        """(min, max) directional Young's modulus over a Fibonacci sphere + the <100>/<111> axes."""
        return youngs_extremes(self.C_eff, n_dirs)

    @property
    def cubic_constants(self) -> tuple[float, float, float]:
        """(C11, C12, C44) as averages of the cubic-equivalent entries."""
        C = self.C_eff
        return (float(np.mean(np.diag(C)[:3])), float(np.mean([C[0, 1], C[0, 2], C[1, 2]])),
                float(np.mean(np.diag(C)[3:])))

    @property
    def cubic_deviation(self) -> float:
        """||C - C_cubic|| / ||C|| (Frobenius, Mandel): 0 for exact cubic symmetry about x, y, z."""
        return cubic_deviation(self.C_eff)

    @property
    def zener_ratio(self) -> float:
        """A = 2 C44 / (C11 - C12) from the cubic averages (1 = isotropic)."""
        c11, c12, c44 = self.cubic_constants
        return float(2.0 * c44 / (c11 - c12))

    @property
    def universal_anisotropy(self) -> float:
        """A^U = 5 G_V/G_R + K_V/K_R - 6 (0 = isotropic); valid for any symmetry, incl. blends."""
        return float(voigt_reuss_hill(self.C_eff)["A_U"])

    @property
    def bulk_modulus(self) -> float:
        """Effective bulk modulus 1 / sum_{i,j<=3} S_ij (exact for hydrostatic stress)."""
        return float(1.0 / np.sum(self.S[:3, :3]))

    def as_row(self, prefix: str = "C", E_ref: float | None = None) -> dict[str, float]:
        """Flat dict for tables (moduli divided by E_ref, default E_s)."""
        ref = E_ref if E_ref is not None else (self.E_s or 1.0)
        C = self.C_eff / ref
        row: dict[str, float] = {}
        for i in range(6):
            for j in range(i, 6):
                row[f"{prefix}{i + 1}{j + 1}"] = float(C[i, j])
        Ex, Ey, Ez = self.youngs_moduli / ref
        Gyz, Gxz, Gxy = self.shear_moduli / ref
        c11, c12, c44 = (v / ref for v in self.cubic_constants)
        emin, emax = self.youngs_extremes()
        row.update(
            {
                "E_x": Ex, "E_y": Ey, "E_z": Ez, "E_axial": self.E_axial / ref,
                "G_yz": Gyz, "G_xz": Gxz, "G_xy": Gxy,
                "E_min": emin / ref, "E_max": emax / ref,
                "K_bulk": self.bulk_modulus / ref,
                "C11c": c11, "C12c": c12, "C44c": c44,
                "zener": self.zener_ratio, "cubic_dev": self.cubic_deviation,
                "A_U": self.universal_anisotropy, "E_hill": voigt_reuss_hill(self.C_eff)["E_H"] / ref,
                "eig_min": float(self.eigenvalues[0] / ref),
                f"{prefix}_agreement": self.agreement,
                f"{prefix}_wall_time_s": self.wall_time,
                f"{prefix}_iterations_max": int(max(self.iterations)),
            }
        )
        for name, loc in self.localization.items():
            row[f"loc_{name}_max"] = loc.max
            row[f"loc_{name}_p99"] = loc.p99
            row[f"loc_{name}_p999"] = loc.p999
            row[f"loc_{name}_mean"] = loc.mean
        return row

    def localize(self, macro_stress: Sequence[float], name: str = "custom",
                 keep_field: bool = False) -> StressLocalization:
        """Localization statistics for any macroscopic Voigt stress (needs ``return_fields``)."""
        if self.unit_stresses is None:
            raise ValueError("run effective_elasticity(..., return_fields=True) to keep unit stresses")
        return _localize(self.unit_stresses, self.S, np.asarray(macro_stress, float), name,
                         self.shape, self.element_voxel, self.element_solid, keep_field)


def cubic_deviation(C: np.ndarray) -> float:
    """Relative Frobenius distance (Mandel) to the closest cubic tensor with axes x, y, z."""
    C = np.asarray(C)
    c11 = np.mean(np.diag(C)[:3])
    c12 = np.mean([C[0, 1], C[0, 2], C[1, 2]])
    c44 = np.mean(np.diag(C)[3:])
    Cc = np.zeros((6, 6))
    Cc[:3, :3] = c12
    Cc[[0, 1, 2], [0, 1, 2]] = c11
    Cc[[3, 4, 5], [3, 4, 5]] = c44
    M = voigt_to_mandel(C)
    return float(np.linalg.norm(M - voigt_to_mandel(Cc)) / np.linalg.norm(M))


def voigt_reuss_hill(C: np.ndarray) -> dict[str, float]:
    """Voigt, Reuss and Hill isotropic moduli of a Voigt stiffness C, and the universal
    anisotropy index A^U = 5 G_V/G_R + K_V/K_R - 6 (Ranganathan & Ostoja-Starzewski 2008,
    Phys. Rev. Lett. 101, 055504; 0 for isotropy, any symmetry class)."""
    C = np.asarray(C, float)
    S = np.linalg.inv(C)
    a, b, c = np.trace(C[:3, :3]), C[0, 1] + C[1, 2] + C[0, 2], np.trace(C[3:, 3:])
    KV, GV = (a + 2 * b) / 9.0, (a - b + 3 * c) / 15.0
    sa, sb, sc = np.trace(S[:3, :3]), S[0, 1] + S[1, 2] + S[0, 2], np.trace(S[3:, 3:])
    KR, GR = 1.0 / (sa + 2 * sb), 15.0 / (4 * sa - 4 * sb + 3 * sc)
    KH, GH = 0.5 * (KV + KR), 0.5 * (GV + GR)
    return {"K_V": KV, "G_V": GV, "K_R": KR, "G_R": GR, "K_H": KH, "G_H": GH,
            "E_H": 9 * KH * GH / (3 * KH + GH), "A_U": 5 * GV / GR + KV / KR - 6.0}


def youngs_extremes(C: np.ndarray, n_dirs: int = 2000) -> tuple[float, float]:
    """(min, max) directional Young's modulus of the Voigt stiffness C."""
    S = np.linalg.inv(C)
    k = np.arange(n_dirs) + 0.5
    z = 1.0 - 2.0 * k / n_dirs
    r = np.sqrt(1.0 - z * z)
    ph = np.pi * (1.0 + 5**0.5) * k
    d = np.c_[r * np.cos(ph), r * np.sin(ph), z]
    s3 = 1 / np.sqrt(3.0)
    d = np.vstack([d, np.eye(3), [[s3, s3, s3], [s3, s3, -s3], [s3, -s3, s3], [-s3, s3, s3]]])
    v = np.c_[d[:, 0] ** 2, d[:, 1] ** 2, d[:, 2] ** 2, d[:, 1] * d[:, 2], d[:, 0] * d[:, 2], d[:, 0] * d[:, 1]]
    E = 1.0 / np.einsum("ni,ij,nj->n", v, S, v)
    return float(E.min()), float(E.max())


def _localize(
    unit_stresses: np.ndarray,
    S: np.ndarray,
    Sigma: np.ndarray,
    name: str,
    shape: tuple[int, int, int],
    el_voxel: np.ndarray,
    solid: np.ndarray,
    keep_field: bool,
) -> StressLocalization:
    ebar = S @ Sigma  # macroscopic Voigt strain
    sig = np.einsum("j,jek->ek", ebar, unit_stresses, dtype=np.float64)
    vm = von_mises(sig)[solid]
    svm = float(von_mises(Sigma))
    if svm <= 0:
        raise ValueError("macroscopic stress must be non-zero")
    f = vm / svm
    imax = int(np.argmax(f))
    vox = np.unravel_index(int(el_voxel[solid][imax]), shape)
    return StressLocalization(
        name=name, macro_stress=Sigma.copy(), macro_von_mises=svm,
        max=float(f.max()), p99=float(np.percentile(f, 99.0)), p999=float(np.percentile(f, 99.9)),
        mean=float(f.mean()), argmax_voxel=tuple(int(v) for v in vox),
        factor=f.astype(np.float32) if keep_field else None,
    )


# =============================================================================
# Main solver
# =============================================================================
def effective_elasticity(
    cell: np.ndarray,
    E: float | None = None,
    nu: float | None = None,
    *,
    void_modulus: float = 0.0,
    tol: float = 1e-8,
    preconditioner: Preconditioner = "auto",
    operator: Operator = "auto",
    maxiter: int | None = None,
    block_solve: bool = True,
    localization: Sequence[str] | None = ("uniaxial_z", "shear_xz"),
    keep_localization_fields: bool = False,
    return_fields: bool = False,
    check: bool = True,
    agreement_tol: float = 1e-5,
) -> ElasticityResult:
    """Periodic effective stiffness tensor (6x6 Voigt) of a voxel cell by Q1 voxel FEA.

    Parameters
    ----------
    cell:
        bool array (True = solid, needs ``E`` and ``nu``) or float array of voxel
        Young's moduli (common ``nu``). Index order (x, y, z), periodic in x, y, z,
        cubic voxels (anisotropic TPMS cells have different voxel counts per axis).
    E, nu:
        Solid Young's modulus and Poisson ratio. For a float cell ``E`` is ignored.
    void_modulus:
        Soft-void option: void voxels of a bool cell get ``void_modulus * E``
        (default 0 = void removed, see module docstring).
    tol:
        PCG relative residual tolerance.
    preconditioner:
        "auto" (AMG if pyamg is installed and the matrix is assembled, else
        Jacobi), "amg", "block_jacobi", "jacobi" or "none".
    operator:
        "assembled" (CSR), "matrix_free", or "auto" (assembled unless the matrix
        would exceed ~6e7 non-zeros, i.e. ~0.7 GB).
    maxiter:
        PCG iteration cap per load case (default max(5000, 100 n_max)).
    block_solve:
        Run the six PCG solves in lock-step with one sparse product per iteration
        for all six right-hand sides (default; ~1.3-1.8x faster with the assembled
        matrix, identical iterates). False, or ``operator="matrix_free"``, runs
        scipy ``cg`` six times.
    localization:
        Names from ``MACRO_STRESS_CASES`` to evaluate (None/() to skip).
    keep_localization_fields:
        Keep the per-voxel localization factors in each ``StressLocalization``.
    return_fields:
        Keep the element-average unit-strain stresses (for ``ElasticityResult.localize``).
    check:
        Raise ``RuntimeError`` if PCG fails or flux/energy C_eff disagree by more
        than ``agreement_tol`` (relative to max|C|).

    Returns
    -------
    ElasticityResult (``.C_eff`` = symmetric 6x6 Voigt stiffness).

    Boundary conditions: periodic fluctuation u~ on all faces (anti-periodic
    tractions), six unit macroscopic Voigt strains. Equations: module docstring.
    """
    t0 = time.perf_counter()
    cell = np.asarray(cell)
    if cell.ndim != 3:
        raise ValueError(f"cell must be 3-D, got shape {cell.shape}")
    if nu is None:
        raise ValueError("nu is required")
    if cell.dtype == bool:
        if E is None:
            raise ValueError("a bool cell needs E")
        if void_modulus < 0:
            raise ValueError("void_modulus must be >= 0")
        Efield = np.where(cell, float(E), float(void_modulus) * float(E))
        solid_fraction = float(cell.mean())
        E_s: float | None = float(E)
        solid_flat = cell.ravel()
    else:
        Efield = np.asarray(cell, dtype=np.float64)
        if (Efield < 0).any():
            raise ValueError("moduli must be >= 0")
        solid_fraction = None
        E_s = None
        solid_flat = None
    shape = tuple(int(s) for s in Efield.shape)
    N = Efield.size
    if not (Efield > 0).any():
        raise ValueError("cell has no stiff voxel")

    K1, F1, Bc, D1 = hex8_element(float(nu))
    mesh = _build_mesh(Efield)
    ndof = 3 * mesh.n_nodes
    Ne = mesh.E_el.size

    if operator == "auto":
        est_nnz = 3.0 * mesh.n_nodes * 3 * 27 * 0.85
        operator = "matrix_free" if (est_nnz > _AUTO_MATRIX_FREE_NNZ and preconditioner != "amg") else "assembled"
    ta = time.perf_counter()
    if operator == "assembled":
        K = _assemble(mesh, K1)
        A: Any = K
    elif operator == "matrix_free":
        K = None
        A = _MatrixFree(mesh, K1)
    else:
        raise ValueError(f"unknown operator {operator!r}")
    M, pc_name = _preconditioner(preconditioner, mesh, K1, K)
    t_asm = time.perf_counter() - ta
    if maxiter is None:
        maxiter = max(5000, 100 * max(shape))

    # per-component, per-direction translation projector
    comp3 = np.repeat(mesh.comp, 3) * 3 + np.tile(np.arange(3), mesh.n_nodes)
    cnt = np.bincount(comp3, minlength=3 * mesh.n_comp).astype(float)

    def _proj(v: np.ndarray) -> np.ndarray:
        return v - (np.bincount(comp3, weights=v, minlength=cnt.size) / cnt)[comp3]

    flat = mesh.edof.ravel()
    Ee = mesh.E_el
    F = np.empty((ndof, 6))
    for j in range(6):
        fe = -(Ee[:, None] * F1[:, j][None, :])  # (Ne, 24) unit eigen-strain load
        F[:, j] = _proj(np.bincount(flat, weights=fe.ravel(), minlength=ndof))
    # forces of a unit eigenstrain cancel exactly in a homogeneous region; treat a load
    # that is round-off relative to the uncancelled element forces as exactly zero
    fscale = np.sqrt(float(np.sum(Ee**2))) * np.linalg.norm(F1, axis=0)
    fnorm = np.linalg.norm(F, axis=0)
    F[:, fnorm <= 1e-12 * fscale] = 0.0
    fnorm = np.linalg.norm(F, axis=0)
    ts = time.perf_counter()
    if block_solve and operator == "assembled":  # the matrix-free product gains nothing
        Xs, iters, resid, conv = _pcg_block(A, F, M, tol, maxiter)
        U = [_proj(Xs[:, j]) for j in range(6)]
        tcase = [(time.perf_counter() - ts) / 6.0] * 6
    else:
        U, iters, resid, conv, tcase = [], [], [], [], []
        for j in range(6):
            tc = time.perf_counter()
            if fnorm[j] == 0.0:
                U.append(np.zeros(ndof))
                iters.append(0)
                resid.append(0.0)
                conv.append(True)
                tcase.append(time.perf_counter() - tc)
                continue
            it = [0]

            def _cb(_xk: np.ndarray, it: list[int] = it) -> None:
                it[0] += 1

            x, info = cg(A, F[:, j], rtol=tol, atol=0.0, maxiter=maxiter, M=M, callback=_cb)
            U.append(_proj(x))
            iters.append(it[0])
            resid.append(float(np.linalg.norm(F[:, j] - A @ x) / fnorm[j]))
            conv.append(info == 0)
            tcase.append(time.perf_counter() - tc)
    t_solve = time.perf_counter() - ts
    if check:
        for j in range(6):
            if not conv[j] and resid[j] > 10 * tol:
                raise RuntimeError(
                    f"PCG did not converge for load case {VOIGT_ORDER[j]}: relative residual "
                    f"{resid[j]:.2e} after {iters[j]} iterations"
                )

    # ---- effective tensor: flux and energy forms ------------------------------
    Vs = float(Ee.sum())
    C_flux = np.empty((6, 6))
    C_en = np.empty((6, 6))
    P = []  # (Ne, 6): F1^T u_e^(j)
    for j in range(6):
        ue = U[j][mesh.edof]
        P.append(ue @ F1)
    for j in range(6):
        C_flux[:, j] = (Vs * D1[:, j] + (Ee[:, None] * P[j]).sum(axis=0)) / N
    for i in range(6):
        ui = U[i][mesh.edof]
        for j in range(i, 6):
            uj = U[j][mesh.edof] if j != i else ui
            q = float(np.sum(Ee * np.einsum("ek,ek->e", ui @ K1, uj)))
            v = Vs * D1[i, j] + float(Ee @ P[j][:, i]) + float(Ee @ P[i][:, j]) + q
            C_en[i, j] = C_en[j, i] = v / N
    scale = float(np.max(np.abs(C_en)))
    agreement = float(np.max(np.abs(C_flux - C_en)) / scale) if scale > 0 else 0.0
    if check and agreement > agreement_tol:
        raise RuntimeError(f"flux- and energy-based C_eff disagree: rel. diff {agreement:.2e}")

    # ---- element-average stresses for the unit strains, localization ----------
    need_stress = return_fields or bool(localization)
    unit = None
    if need_stress:
        unit = np.empty((6, Ne, 6), dtype=np.float32)
        for j in range(6):
            eps = U[j][mesh.edof] @ Bc.T  # (Ne, 6)
            eps[:, j] += 1.0
            unit[j] = (Ee[:, None] * (eps @ D1.T)).astype(np.float32)
    if solid_flat is not None:
        solid_el = solid_flat[mesh.el_voxel]
    else:
        solid_el = Ee >= 0.5 * Ee.max()
    S = np.linalg.inv(C_en) if need_stress else None
    loc: dict[str, StressLocalization] = {}
    if localization:
        for name in localization:
            if name not in MACRO_STRESS_CASES:
                raise ValueError(f"unknown localization case {name!r}; choose from {list(MACRO_STRESS_CASES)}")
            loc[name] = _localize(unit, S, MACRO_STRESS_CASES[name], name, shape,
                                  mesh.el_voxel, solid_el, keep_localization_fields)

    wall = time.perf_counter() - t0
    LOG.debug("C_eff shape=%s dofs=%d elems=%d pc=%s op=%s iters=%s agree=%.1e time=%.2fs",
              shape, ndof, Ne, pc_name, operator, iters, agreement, wall)
    res = ElasticityResult(
        C_eff=C_en, C_eff_flux=C_flux, agreement=agreement, E_s=E_s, nu_s=float(nu),
        solid_fraction=solid_fraction, shape=shape, localization=loc, iterations=iters,
        residuals=resid, case_times=tcase, solve_time=t_solve, preconditioner=pc_name, operator=operator,
        wall_time=wall, assembly_time=t_asm, n_dofs=ndof, n_elements=Ne,
        n_components=mesh.n_comp, element_voxel=mesh.el_voxel, element_solid=solid_el,
        unit_stresses=unit if return_fields else None,
    )
    return res


def _material(E: float | None, nu: float | None, cfg: Any) -> tuple[float, float]:
    if E is None or nu is None:
        if cfg is None:
            from voxlat.utils.config import load_config

            cfg = load_config()
        E = cfg.material.youngs_modulus if E is None else E
        nu = cfg.material.poisson_ratio if nu is None else nu
    return float(E), float(nu)


def effective_elasticity_tpms(
    params: Any,
    n: int = 40,
    E: float | None = None,
    nu: float | None = None,
    *,
    cfg: Any = None,
    offset: Sequence[float] | None = None,
    **kwargs: Any,
) -> ElasticityResult:
    """C_eff of a TPMS cell (``voxlat.geometry.TPMSParams``) voxelized with n voxels per L.

    E, nu default to the reference config (AlSi10Mg 70 GPa, 0.33). ``offset`` shifts
    the TPMS against the grid. Extra keyword arguments go to ``effective_elasticity``.
    A single grid carries an O(1/n) staircase error (see STATUS.md, Task 3); use
    ``extrapolated_elasticity_tpms`` for production numbers.
    """
    from voxlat.geometry.tpms import voxelize

    E, nu = _material(E, nu, cfg)
    solid = voxelize(params, n, offset=offset)
    return effective_elasticity(solid, E, nu, **kwargs)


@dataclass
class ExtrapolatedElasticity:
    """Two-grid Richardson estimate C_inf = (n2^p C(n2) - n1^p C(n1)) / (n2^p - n1^p).

    ``correction`` = max|C_inf - C(n2)| / max|C_inf|. Localization factors are NOT
    extrapolated (the max does not converge); use ``fine.localization``.
    """

    C_eff: np.ndarray
    n: tuple[int, int]
    p: float
    coarse: ElasticityResult
    fine: ElasticityResult
    correction: float
    wall_time: float

    @property
    def S(self) -> np.ndarray:
        return np.linalg.inv(self.C_eff)

    @property
    def youngs_moduli(self) -> np.ndarray:
        return 1.0 / np.diag(self.S)[:3]

    @property
    def E_axial(self) -> float:
        return float(np.mean(self.youngs_moduli))

    @property
    def zener_ratio(self) -> float:
        C = self.C_eff
        c11 = np.mean(np.diag(C)[:3])
        c12 = np.mean([C[0, 1], C[0, 2], C[1, 2]])
        return float(2.0 * np.mean(np.diag(C)[3:]) / (c11 - c12))

    @property
    def eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvalsh(voigt_to_mandel(self.C_eff))


def extrapolated_elasticity_tpms(
    params: Any,
    n: tuple[int, int] = (32, 64),
    E: float | None = None,
    nu: float | None = None,
    *,
    p: float = 1.0,
    cfg: Any = None,
    offset: Sequence[float] | None = None,
    **kwargs: Any,
) -> ExtrapolatedElasticity:
    """Production C_eff of a TPMS cell: two resolutions + Richardson of order p (component-wise).

    Default R(32, 64), p = 1 (Task 3 convergence study, STATUS.md): offset-averaged
    stiffness converges from below at first order (staircase), and R(32, 64) on the
    default grid is within ~3 % of the reference for E*, C11, C12, C44, K and the
    Zener ratio over G/D at rho* = 0.2-0.5 (a single n = 48 grid: E* -2 %, C12 down
    to -6 %). Cost ~1.1x one n = 64 solve. Localization comes from the fine grid.
    """
    t0 = time.perf_counter()
    n1, n2 = (int(v) for v in n)
    if not n1 < n2:
        raise ValueError("need n1 < n2")
    E, nu = _material(E, nu, cfg)
    kwargs.setdefault("localization", ("uniaxial_z", "shear_xz"))
    loc1 = kwargs.pop("localization")
    r1 = effective_elasticity_tpms(params, n1, E, nu, offset=offset, localization=(), **kwargs)
    r2 = effective_elasticity_tpms(params, n2, E, nu, offset=offset, localization=loc1, **kwargs)
    a, b = float(n1) ** p, float(n2) ** p
    C = (b * r2.C_eff - a * r1.C_eff) / (b - a)
    C = 0.5 * (C + C.T)
    sc = float(np.max(np.abs(C)))
    corr = float(np.max(np.abs(C - r2.C_eff)) / sc) if sc > 0 else 0.0
    return ExtrapolatedElasticity(C, (n1, n2), p, r1, r2, corr, time.perf_counter() - t0)


@dataclass
class AveragedElasticity:
    """C_eff averaged over rigid shifts of the TPMS against the voxel grid.

    ``C_eff`` is the mean of the per-offset tensors (symmetric), ``C_std`` the
    component-wise standard deviation over offsets (alignment noise), ``localization``
    the per-offset means of each statistic (name -> {"max", "p99", "p999", "mean"}).
    """

    C_eff: np.ndarray
    C_std: np.ndarray
    n: int
    offsets: np.ndarray
    results: list[ElasticityResult] = field(repr=False)
    localization: dict[str, dict[str, float]]
    wall_time: float

    @property
    def S(self) -> np.ndarray:
        return np.linalg.inv(self.C_eff)

    @property
    def youngs_moduli(self) -> np.ndarray:
        return 1.0 / np.diag(self.S)[:3]

    @property
    def E_axial(self) -> float:
        return float(np.mean(self.youngs_moduli))

    @property
    def zener_ratio(self) -> float:
        C = self.C_eff
        c11 = np.mean(np.diag(C)[:3])
        c12 = np.mean([C[0, 1], C[0, 2], C[1, 2]])
        return float(2.0 * np.mean(np.diag(C)[3:]) / (c11 - c12))

    @property
    def eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvalsh(voigt_to_mandel(self.C_eff))


def grid_offsets(k: int, seed: int = 2026) -> np.ndarray:
    """k rigid shifts (in cell units): offset 0 plus k - 1 seeded uniform random shifts."""
    rng = np.random.default_rng(seed)
    off = rng.random((k, 3))
    off[0] = 0.0
    return off


def averaged_elasticity_tpms(
    params: Any,
    n: int = 40,
    n_offsets: int = 3,
    E: float | None = None,
    nu: float | None = None,
    *,
    seed: int = 2026,
    cfg: Any = None,
    **kwargs: Any,
) -> AveragedElasticity:
    """C_eff of a TPMS cell averaged over ``n_offsets`` grid shifts at resolution n.

    For voxel FEA the alignment noise of a single grid (+-1-3 % at n = 24-48) is as
    large as the systematic staircase error, so averaging shifts is the cheapest way
    to reduce the error (see STATUS.md, Task 3, for the recommended (n, n_offsets)).
    """
    t0 = time.perf_counter()
    E, nu = _material(E, nu, cfg)
    offs = grid_offsets(int(n_offsets), seed)
    res = [effective_elasticity_tpms(params, n, E, nu, offset=tuple(o), **kwargs) for o in offs]
    Cs = np.stack([r.C_eff for r in res])
    C = Cs.mean(axis=0)
    C = 0.5 * (C + C.T)
    loc: dict[str, dict[str, float]] = {}
    for name in res[0].localization:
        loc[name] = {s: float(np.mean([getattr(r.localization[name], s) for r in res]))
                     for s in ("max", "p99", "p999", "mean")}
    return AveragedElasticity(C, Cs.std(axis=0, ddof=1) if len(res) > 1 else np.zeros((6, 6)),
                              int(n), offs, res, loc, time.perf_counter() - t0)


# =============================================================================
# Analytic references
# =============================================================================
def laminate_stiffness(
    fractions: Sequence[float], E: Sequence[float], nu: Sequence[float], axis: int = 0
) -> np.ndarray:
    """Exact effective stiffness (6x6 Voigt) of a laminate of isotropic layers normal to ``axis``.

    Backus (1962) averaging, written for the layer normal along local "1": with
    continuity of t = (s11, s12, s13) and of the in-plane strains (e22, e33, g23),

        C11 = < 1/(lam+2mu) >^-1,               C12 = C13 = C11 < lam/(lam+2mu) >,
        C22 = C33 = < 4mu(lam+mu)/(lam+2mu) > + C12^2 / C11,
        C23 = < 2 lam mu/(lam+2mu) > + C12^2 / C11,
        C44 (in-plane shear 23) = < mu >,       C55 = C66 = < 1/mu >^-1  (Reuss).

    A zero-stiffness (void) layer gives the limits C11 = C12 = C55 = C66 = 0 and the
    plane-stress in-plane terms of the remaining plates. The result is rotated so
    that the layer normal is ``axis`` (0 = x, 1 = y, 2 = z).
    """
    f = np.asarray(fractions, float)
    f = f / f.sum()
    lam, mu = np.zeros(f.size), np.zeros(f.size)
    for i, (e, v) in enumerate(zip(E, nu)):
        if e > 0:
            lam[i], mu[i] = lame_parameters(float(e), float(v))
    P = lam + 2 * mu
    has_void = np.any(P == 0)
    ok = P > 0
    if has_void:
        C11 = 0.0
        C12 = 0.0
        C55 = 0.0
    else:
        C11 = 1.0 / np.sum(f / P)
        C12 = C11 * np.sum(f * lam / P)
        C55 = 1.0 / np.sum(f / mu)
    a = np.sum(f[ok] * 4 * mu[ok] * (lam[ok] + mu[ok]) / P[ok])
    b = np.sum(f[ok] * 2 * lam[ok] * mu[ok] / P[ok])
    C22 = a + (C12**2 / C11 if C11 > 0 else 0.0)
    C23 = b + (C12**2 / C11 if C11 > 0 else 0.0)
    C44 = np.sum(f * mu)
    C = np.zeros((6, 6))
    C[0, 0] = C11
    C[0, 1] = C[1, 0] = C[0, 2] = C[2, 0] = C12
    C[1, 1] = C[2, 2] = C22
    C[1, 2] = C[2, 1] = C23
    C[3, 3] = C44
    C[4, 4] = C[5, 5] = C55
    if axis == 0:
        return C
    perm = [axis, (axis + 1) % 3, (axis + 2) % 3]  # local 1,2,3 -> global axes
    R = np.zeros((3, 3))
    for loc, glob in enumerate(perm):
        R[glob, loc] = 1.0
    return rotate_stiffness(C, R)


def hashin_shtrikman_porous(phi_s: Any, E: float, nu: float) -> dict[str, np.ndarray]:
    """Hashin-Shtrikman upper bounds for an isotropic porous solid (void second phase).

        K+ = 4 phi K mu / (3 K (1 - phi) + 4 mu)
        G+ = mu + (1 - phi) / ( -1/mu + 6 phi (K + 2mu) / (5 mu (3K + 4mu)) )
        E+ = 9 K+ G+ / (3 K+ + G+)

    (Hashin & Shtrikman, J. Mech. Phys. Solids 11, 127, 1963.) Bounds apply to the
    isotropic part of any statistically isotropic microstructure; for cubic cells
    they bound the Voigt-Reuss-Hill isotropic averages, not every single component.
    """
    phi = np.asarray(phi_s, float)
    lam, mu = lame_parameters(E, nu)
    K = lam + 2.0 * mu / 3.0
    Kp = 4.0 * phi * K * mu / (3.0 * K * (1.0 - phi) + 4.0 * mu)
    with np.errstate(divide="ignore", invalid="ignore"):
        Gp = mu + (1.0 - phi) / (-1.0 / mu + 6.0 * phi * (K + 2.0 * mu) / (5.0 * mu * (3.0 * K + 4.0 * mu)))
        Ep = np.where(3.0 * Kp + Gp > 0, 9.0 * Kp * Gp / (3.0 * Kp + Gp), 0.0)
    return {"K": Kp, "G": Gp, "E": Ep}
