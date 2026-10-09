"""Periodic effective thermal conductivity k_eff (3x3) of a two-phase voxel cell (Task 2).

Governing equations
-------------------
Steady conduction without sources in a periodic cell Y (one TPMS unit cell or any
periodic voxel block), local conductivity k(x) piecewise constant per voxel:

    div q = 0,     q = -k(x) grad T                                   in Y
    T(x) = G . x + theta(x),   theta  Y-periodic                      (cell problem)

G is the imposed macroscopic temperature gradient and theta the periodic
fluctuation. The effective (homogenized) tensor is defined by the volume-averaged
flux,

    <q> = -k_eff G,

and, equivalently (Hill-Mandel), by the averaged dissipation,

    G_i k_eff,ij G_j = < grad T^(i) . k grad T^(j) >,   T^(i) = solution for G = e_i.

Three unit gradients G = e_x, e_y, e_z give the three columns of k_eff.

Discretization: cell-centred finite volumes
-------------------------------------------
One unknown theta_p per voxel p (voxel edge h; h cancels out of k_eff, so the
solver works in voxel units h = 1). Between voxel p and its +d neighbour
q = p + e_d (periodic wrap) the face conductance is the HARMONIC mean

    k_pq = 2 k_p k_q / (k_p + k_q)          (0 if either side is 0),

which is the exact series resistance of two half voxels. It makes 1-D laminates
exact and handles any contrast, including the insulating-fluid limit k_f = 0.
Face temperature drop and flux (voxel units):

    dT_pq = theta_q - theta_p + G_d,         f_pq = -k_pq dT_pq.

Flux balance in every voxel, sum_q k_pq dT_pq = 0, gives the linear system

    A theta = b,   (A theta)_p = sum_q k_pq (theta_p - theta_q),
                   b_p = sum_d G_d (k_{p,p+d} - k_{p,p-d}).

A is the weighted graph Laplacian of the voxel grid: symmetric positive
SEMI-definite, with one constant null vector per conducting component (theta is
defined up to a constant). b sums to zero over every component, so the system is
consistent and CG converges; b is additionally projected onto range(A) to remove
round-off. Voxels with no conducting face (all of the fluid when k_f = 0) are
removed from the system.

Effective tensor, two ways (both exact for the discrete problem):

    flux:    k_eff^flux[d, j]  = (1/N) sum_{faces normal to d} k_pq dT_pq^(j)
    energy:  k_eff^energy[i, j] = (1/N) sum_{all faces} k_pq dT_pq^(i) dT_pq^(j)

(N = number of voxels). They agree to the solver tolerance because
theta^(i)T A theta^(j) - theta^(i)T b^(j) = 0 at the solution (discrete Hill-Mandel);
``effective_conductivity`` checks that and raises if they disagree. The energy
form is symmetric by construction and is the value reported as ``k_eff``.

Solver: preconditioned conjugate gradients (scipy.sparse.linalg.cg), relative
residual tol 1e-8 by default. Preconditioner: algebraic multigrid
(pyamg smoothed aggregation, V-cycle) if pyamg is installed, else Jacobi
(diagonal). Jacobi is enough for n <= 64 (see STATUS.md, Task 2, timings).
``preconditioner="two_level"`` (Task 7) adds a piecewise-constant aggregation
coarse space to Jacobi (``voxlat.homogenization.coarse``); it removes the
iteration growth on long grids (finite-gap strips) without pyamg.
``directions`` (Task 7) restricts the solve to some macroscopic gradients; the
unsolved columns/rows of k_eff are NaN.

Units: k_s, k_f in W/(m K) (or any consistent unit); k_eff has the same unit.
Defaults come from the reference config (AlSi10Mg 130, water-glycol 0.40 W/(m K)).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, cg

from voxlat.utils.log import get_logger

__all__ = [
    "ConductivityResult",
    "phase_conductivity",
    "face_conductances",
    "assemble_conduction_system",
    "effective_conductivity",
    "effective_conductivity_tpms",
    "ExtrapolatedConductivity",
    "extrapolated_conductivity_tpms",
    "hashin_shtrikman_bounds",
    "wiener_bounds",
    "rayleigh_sc_spheres",
    "pyamg_available",
]

LOG = get_logger("homogenization.conduction")

Preconditioner = Literal["auto", "amg", "jacobi", "two_level", "none"]


# =============================================================================
# Building blocks
# =============================================================================
def pyamg_available() -> bool:
    """True if pyamg can be imported (enables the AMG preconditioner)."""
    try:
        import pyamg  # noqa: F401
    except ImportError:
        return False
    return True


def phase_conductivity(solid: np.ndarray, k_s: float, k_f: float) -> np.ndarray:
    """Voxel conductivity field: k_s where ``solid`` is True, k_f elsewhere (float64)."""
    solid = np.asarray(solid, dtype=bool)
    if k_s < 0 or k_f < 0:
        raise ValueError(f"conductivities must be >= 0, got k_s={k_s}, k_f={k_f}")
    return np.where(solid, float(k_s), float(k_f))


def face_conductances(k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Harmonic-mean conductances of the +x, +y, +z faces of every voxel (periodic).

    ``kf[d][p]`` is the conductance of the face between voxel p and p + e_d
    (wrapped), ``2 k_p k_q / (k_p + k_q)``, 0 where either side is 0.
    """
    k = np.asarray(k, dtype=np.float64)
    if k.ndim != 3:
        raise ValueError(f"expected a 3-D voxel field, got shape {k.shape}")
    out = []
    for d in range(3):
        kq = np.roll(k, -1, axis=d)
        s = k + kq
        with np.errstate(divide="ignore", invalid="ignore"):
            kf = np.where(s > 0, 2.0 * k * kq / np.where(s > 0, s, 1.0), 0.0)
        out.append(kf)
    return tuple(out)  # type: ignore[return-value]


def assemble_conduction_system(
    kf: Sequence[np.ndarray],
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Graph-Laplacian matrix A (periodic, all voxels) and the active-voxel mask.

    A is returned restricted to *active* voxels (non-zero diagonal, i.e. at least
    one conducting face); ``active`` is the boolean mask over the flattened grid
    (C order). Duplicate couplings (grids with 1 or 2 voxels along an axis) are
    summed correctly by the COO -> CSR conversion.
    """
    shape = kf[0].shape
    N = int(np.prod(shape))
    idx = np.arange(N).reshape(shape)
    rows, cols, vals = [], [], []
    diag = np.zeros(N)
    for d in range(3):
        k_d = kf[d].ravel()
        nz = k_d > 0
        p = idx.ravel()[nz]
        q = np.roll(idx, -1, axis=d).ravel()[nz]
        w = k_d[nz]
        np.add.at(diag, p, w)
        np.add.at(diag, q, w)
        rows += [p, q]
        cols += [q, p]
        vals += [-w, -w]
    rows.append(np.arange(N))
    cols.append(np.arange(N))
    vals.append(diag)
    A = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)
    ).tocsr()
    A.sum_duplicates()
    A.eliminate_zeros()
    active = A.diagonal() > 0
    if not active.all():
        A = A[active][:, active].tocsr()
    return A, active


def _rhs(kf: Sequence[np.ndarray], G: Sequence[float]) -> np.ndarray:
    """b_p = sum_d G_d (k_{p,p+d} - k_{p,p-d})  (flattened, all voxels)."""
    b = np.zeros(kf[0].shape)
    for d in range(3):
        if G[d] != 0.0:
            b += G[d] * (kf[d] - np.roll(kf[d], 1, axis=d))
    return b.ravel()


def _face_drops(theta: np.ndarray, G: Sequence[float]) -> list[np.ndarray]:
    """dT across every +d face: theta(p + e_d) - theta(p) + G_d."""
    return [np.roll(theta, -1, axis=d) - theta + G[d] for d in range(3)]


def _make_preconditioner(
    A: sp.csr_matrix,
    kind: Preconditioner,
    active: np.ndarray | None = None,
    shape: Sequence[int] | None = None,
    coarse_block: int | None = None,
) -> tuple[Any, str]:
    if kind == "auto":
        kind = "amg" if pyamg_available() else "jacobi"
    if kind == "none":
        return None, "none"
    if kind == "jacobi":
        dinv = 1.0 / A.diagonal()
        return LinearOperator(A.shape, matvec=lambda x: dinv * x, dtype=np.float64), "jacobi"
    if kind == "two_level":
        from voxlat.homogenization.coarse import (
            DEFAULT_COARSE_BLOCK,
            TwoLevelPreconditioner,
            box_aggregates,
            scalar_coarse_basis,
        )

        if active is None or shape is None:
            raise ValueError("two_level needs the active mask and the grid shape")
        block = int(coarse_block or DEFAULT_COARSE_BLOCK)
        gi = np.stack(np.unravel_index(np.flatnonzero(active), tuple(shape)), axis=1)
        P = scalar_coarse_basis(box_aggregates(gi, shape, block))
        return TwoLevelPreconditioner(A, P), "two_level"
    if kind == "amg":
        try:
            import pyamg
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("preconditioner='amg' needs pyamg (pip install pyamg)") from exc
        ml = pyamg.smoothed_aggregation_solver(A, symmetry="hermitian", max_coarse=500)
        return ml.aspreconditioner(cycle="V"), "amg"
    raise ValueError(f"unknown preconditioner {kind!r}")


# =============================================================================
# Result
# =============================================================================
@dataclass
class ConductivityResult:
    """Output of ``effective_conductivity``.

    Attributes
    ----------
    k_eff:
        3x3 effective tensor (energy form, symmetric), same unit as k_s, k_f.
    k_eff_flux:
        3x3 tensor from the averaged flux (column j = response to G = e_j).
    k_eff_energy:
        Same as ``k_eff``.
    agreement:
        max|k_flux - k_energy| / max|k_energy| (should be ~ solver tol).
    k_s, k_f, solid_fraction, shape:
        Inputs (k_s/k_f are None if a general conductivity field was given).
    iterations, residuals:
        PCG iterations and final relative residuals for the three solves.
    preconditioner, wall_time:
        Preconditioner actually used, and total wall time in s (assembly + 3 solves).
    n_active:
        Unknowns actually solved for (voxels with at least one conducting face).
    theta:
        Fluctuation fields (3, nx, ny, nz) if ``return_fields`` (voxel units, float32).
    """

    k_eff: np.ndarray
    k_eff_flux: np.ndarray
    k_eff_energy: np.ndarray
    agreement: float
    k_s: float | None
    k_f: float | None
    solid_fraction: float | None
    shape: tuple[int, int, int]
    iterations: list[int]
    residuals: list[float]
    preconditioner: str
    wall_time: float
    n_active: int
    theta: np.ndarray | None = field(default=None, repr=False)

    # ---- derived quantities -------------------------------------------------
    @property
    def eigenvalues(self) -> np.ndarray:
        """Principal conductivities (ascending)."""
        return np.linalg.eigvalsh(self.k_eff)

    @property
    def is_spd(self) -> bool:
        """Symmetric and all eigenvalues > 0 (relative to the largest)."""
        ev = self.eigenvalues
        sym = np.allclose(self.k_eff, self.k_eff.T, rtol=0, atol=1e-12 * max(ev[-1], 1e-300))
        return bool(sym and ev[0] > 1e-12 * ev[-1])

    @property
    def mean(self) -> float:
        """Isotropic part tr(k_eff)/3."""
        return float(np.trace(self.k_eff) / 3.0)

    @property
    def anisotropy(self) -> float:
        """(lambda_max - lambda_min) / mean eigenvalue; 0 for an isotropic tensor."""
        ev = self.eigenvalues
        return float((ev[-1] - ev[0]) / ev.mean()) if ev.mean() > 0 else float("nan")

    def relative(self, k_ref: float | None = None) -> np.ndarray:
        """k_eff / k_ref (default k_s)."""
        ref = self.k_s if k_ref is None else k_ref
        if not ref:
            raise ValueError("need a reference conductivity")
        return self.k_eff / ref

    def as_row(self, prefix: str = "k") -> dict[str, float]:
        """Flat dict for tables: 6 independent components, mean, anisotropy, timing."""
        K = self.k_eff
        names = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2), "xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
        row = {f"{prefix}_{s}": float(K[i, j]) for s, (i, j) in names.items()}
        row.update(
            {
                f"{prefix}_mean": self.mean,
                f"{prefix}_anisotropy": self.anisotropy,
                f"{prefix}_flux_energy_agreement": self.agreement,
                f"{prefix}_wall_time_s": self.wall_time,
                f"{prefix}_iterations_max": int(max(self.iterations)),
            }
        )
        return row


# =============================================================================
# Main solver
# =============================================================================
def effective_conductivity(
    cell: np.ndarray,
    k_s: float | None = None,
    k_f: float | None = None,
    *,
    tol: float = 1e-8,
    preconditioner: Preconditioner = "auto",
    maxiter: int | None = None,
    return_fields: bool = False,
    check: bool = True,
    agreement_tol: float = 1e-5,
    directions: Sequence[int] | None = None,
    coarse_block: int | None = None,
) -> ConductivityResult:
    """Periodic effective conductivity tensor of a voxel cell.

    Parameters
    ----------
    cell:
        Either a bool array (True = solid; needs ``k_s`` and ``k_f``) or a float
        array of voxel conductivities. Index order (x, y, z); the cell is
        periodic in all three directions; voxels are cubes (anisotropic TPMS
        cells simply have different voxel counts per axis).
    k_s, k_f:
        Solid and fluid conductivities for a bool cell. k_f = 0 (or k_s = 0) is
        allowed: the non-conducting phase is removed from the system.
    tol:
        PCG relative residual tolerance ||b - A theta|| / ||b||.
    preconditioner:
        "auto" (AMG if pyamg is installed, else Jacobi), "amg", "jacobi",
        "two_level" (Jacobi + aggregation coarse space, no pyamg needed; best for
        long grids such as finite-gap strips) or "none".
    maxiter:
        PCG iteration cap per solve (default 20 * total voxels^(1/3) * 10, >= 2000).
    return_fields:
        Keep the three fluctuation fields theta (float32) in the result.
    check:
        Raise ``RuntimeError`` if flux- and energy-based k_eff differ by more than
        ``agreement_tol`` (relative to max|k_eff|), or if PCG did not converge.
    directions:
        Macroscopic gradients to solve (subset of (0, 1, 2); default all three).
        Column j of ``k_eff_flux`` and the (i, j) entries of the energy form exist
        only for solved i, j; the rest is NaN (then ``eigenvalues`` etc. are NaN).
    coarse_block:
        Aggregate edge in voxels for ``preconditioner="two_level"`` (default 8).

    Returns
    -------
    ConductivityResult (``.k_eff`` is the symmetric 3x3 tensor).

    Boundary conditions: periodic theta on all six faces; macroscopic gradient
    G = e_x, e_y, e_z. See the module docstring for the equations.
    """
    t0 = time.perf_counter()
    cell = np.asarray(cell)
    if cell.ndim != 3:
        raise ValueError(f"cell must be 3-D, got shape {cell.shape}")
    if cell.dtype == bool:
        if k_s is None or k_f is None:
            raise ValueError("a bool cell needs k_s and k_f")
        k = phase_conductivity(cell, k_s, k_f)
        solid_fraction = float(cell.mean())
    else:
        k = np.asarray(cell, dtype=np.float64)
        if (k < 0).any():
            raise ValueError("conductivities must be >= 0")
        solid_fraction = None
        k_s = k_f = None
    shape = tuple(int(s) for s in k.shape)
    N = k.size
    if not (k > 0).any():
        raise ValueError("cell has no conducting voxel")

    kf = face_conductances(k)
    A, active = assemble_conduction_system(kf)
    n_act = int(active.sum())
    if maxiter is None:
        maxiter = max(2000, int(200 * round(N ** (1 / 3))))

    # one null vector (constant) per conducting component -> project b
    n_comp, labels = connected_components(A, directed=False)
    counts = np.bincount(labels, minlength=n_comp).astype(float)

    if directions is None:
        dirs = (0, 1, 2)
    else:
        dirs = tuple(sorted({int(d) for d in directions}))
        if not dirs or any(d not in (0, 1, 2) for d in dirs):
            raise ValueError(f"directions must be a subset of (0, 1, 2), got {directions}")

    M, pc_name = (
        _make_preconditioner(A, preconditioner, active, shape, coarse_block) if n_act > 0 else (None, "none")
    )

    drops_all: dict[int, list[np.ndarray]] = {}
    iters: list[int] = []
    resid: list[float] = []
    thetas = np.zeros((3, *shape), dtype=np.float32) if return_fields else None
    for j in dirs:
        G = [0.0, 0.0, 0.0]
        G[j] = 1.0
        b_full = _rhs(kf, G)
        b = b_full[active]
        b = b - (np.bincount(labels, weights=b, minlength=n_comp) / counts)[labels]
        theta_full = np.zeros(N)
        bnorm = float(np.linalg.norm(b))
        if bnorm > 0.0:
            it = [0]

            def _count(_xk: np.ndarray, it: list[int] = it) -> None:
                it[0] += 1

            x, info = cg(A, b, rtol=tol, atol=0.0, maxiter=maxiter, M=M, callback=_count)
            r = float(np.linalg.norm(b - A @ x) / bnorm)
            if info != 0 and check and r > 10 * tol:
                raise RuntimeError(
                    f"PCG did not converge for G = e_{'xyz'[j]}: info={info}, "
                    f"relative residual {r:.2e} after {it[0]} iterations"
                )
            # remove the arbitrary per-component constant (cosmetic; k_eff is unaffected)
            x = x - (np.bincount(labels, weights=x, minlength=n_comp) / counts)[labels]
            theta_full[active] = x
            iters.append(it[0])
            resid.append(r)
        else:
            iters.append(0)
            resid.append(0.0)
        theta = theta_full.reshape(shape)
        if thetas is not None:
            thetas[j] = theta
        drops_all[j] = _face_drops(theta, G)

    K_flux = np.full((3, 3), np.nan)
    K_en = np.full((3, 3), np.nan)
    for j in dirs:
        for d in range(3):
            K_flux[d, j] = float(np.sum(kf[d] * drops_all[j][d])) / N
    for a, i in enumerate(dirs):
        for j in dirs[a:]:
            e = sum(float(np.sum(kf[d] * drops_all[i][d] * drops_all[j][d])) for d in range(3)) / N
            K_en[i, j] = K_en[j, i] = e

    sel = np.ix_(dirs, dirs)
    scale = float(np.max(np.abs(K_en[sel])))
    agreement = float(np.max(np.abs(K_flux[sel] - K_en[sel])) / scale) if scale > 0 else 0.0
    if check and agreement > agreement_tol:
        raise RuntimeError(
            f"flux- and energy-based k_eff disagree: rel. diff {agreement:.2e} > {agreement_tol:.0e}"
        )
    wall = time.perf_counter() - t0
    LOG.debug(
        "k_eff shape=%s active=%d pc=%s iters=%s agreement=%.1e time=%.2fs",
        shape, n_act, pc_name, iters, agreement, wall,
    )
    return ConductivityResult(
        k_eff=K_en.copy(),
        k_eff_flux=K_flux,
        k_eff_energy=K_en,
        agreement=agreement,
        k_s=None if k_s is None else float(k_s),
        k_f=None if k_f is None else float(k_f),
        solid_fraction=solid_fraction,
        shape=shape,  # type: ignore[arg-type]
        iterations=iters,
        residuals=resid,
        preconditioner=pc_name,
        wall_time=wall,
        n_active=n_act,
        theta=thetas,
    )


def effective_conductivity_tpms(
    params: Any,
    n: int = 48,
    k_s: float | None = None,
    k_f: float | None = None,
    *,
    cfg: Any = None,
    offset: Sequence[float] | None = None,
    **kwargs: Any,
) -> ConductivityResult:
    """k_eff of a TPMS cell (``voxlat.geometry.TPMSParams``) voxelized at n voxels per L.

    k_s and k_f default to the reference config (material / coolant conductivity).
    ``offset`` shifts the TPMS relative to the grid (``voxlat.geometry.voxelize``).
    Extra keyword arguments go to ``effective_conductivity``.

    Note: a single resolution carries an O(1/n) staircase error (~6 % low at
    n = 48 for gyroid/diamond, see STATUS.md); use ``extrapolated_conductivity_tpms``
    for production numbers.
    """
    from voxlat.geometry.tpms import voxelize

    if k_s is None or k_f is None:
        if cfg is None:
            from voxlat.utils.config import load_config

            cfg = load_config()
        k_s = cfg.material.conductivity if k_s is None else k_s
        k_f = cfg.coolant.conductivity if k_f is None else k_f
    solid = voxelize(params, n, offset=offset)
    return effective_conductivity(solid, k_s, k_f, **kwargs)


@dataclass
class ExtrapolatedConductivity:
    """Two-grid Richardson (p = 1) estimate of k_eff for a TPMS cell.

    ``k_eff = (n2 k(n2) - n1 k(n1)) / (n2 - n1)``, component-wise; ``coarse``/``fine``
    hold the two raw solutions. ``correction`` = max|k_eff - k(n2)| / max|k_eff| is
    the size of the extrapolation step (an upper-bound style error indicator for
    the fine grid alone). Accuracy of the estimate itself: see STATUS.md, Task 2.
    """

    k_eff: np.ndarray
    n: tuple[int, int]
    coarse: ConductivityResult
    fine: ConductivityResult
    correction: float
    wall_time: float

    @property
    def mean(self) -> float:
        return float(np.trace(self.k_eff) / 3.0)

    @property
    def eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvalsh(self.k_eff)


def extrapolated_conductivity_tpms(
    params: Any,
    n: tuple[int, int] = (32, 64),
    k_s: float | None = None,
    k_f: float | None = None,
    *,
    cfg: Any = None,
    offset: Sequence[float] | None = None,
    **kwargs: Any,
) -> ExtrapolatedConductivity:
    """Production k_eff of a TPMS cell: two resolutions + first-order Richardson.

    The voxel staircase makes k_eff(n) converge as k_inf - C/n (observed order
    p = 1.0, Task 2), so k_inf ~ (n2 k(n2) - n1 k(n1)) / (n2 - n1). The default pair
    (32, 64) costs ~1.3x a single n = 64 solve. Both grids use the same ``offset``.
    """
    t0 = time.perf_counter()
    n1, n2 = (int(v) for v in n)
    if not n1 < n2:
        raise ValueError("need n1 < n2")
    r1 = effective_conductivity_tpms(params, n1, k_s, k_f, cfg=cfg, offset=offset, **kwargs)
    r2 = effective_conductivity_tpms(params, n2, k_s, k_f, cfg=cfg, offset=offset, **kwargs)
    K = (n2 * r2.k_eff - n1 * r1.k_eff) / (n2 - n1)
    K = 0.5 * (K + K.T)
    scale = float(np.max(np.abs(K)))
    corr = float(np.max(np.abs(K - r2.k_eff)) / scale) if scale > 0 else 0.0
    return ExtrapolatedConductivity(K, (n1, n2), r1, r2, corr, time.perf_counter() - t0)


# =============================================================================
# Analytic references
# =============================================================================
def hashin_shtrikman_bounds(phi_s: Any, k_s: float, k_f: float) -> tuple[np.ndarray, np.ndarray]:
    """Hashin-Shtrikman (1962) bounds for a statistically isotropic 3-D two-phase medium.

    With phi_f = 1 - phi_s and k_s >= k_f:

        upper = k_s + phi_f / ( 1/(k_f - k_s) + phi_s/(3 k_s) )
        lower = k_f + phi_s / ( 1/(k_s - k_f) + phi_f/(3 k_f) )      (lower = 0 if k_f = 0)

    (Hashin & Shtrikman, J. Appl. Phys. 33, 3125, 1962.) Valid for any isotropic
    k_eff, hence for cubic-symmetric cells such as gyroid and diamond, whose
    second-order k_eff is isotropic. Returns (lower, upper); phases are sorted
    internally, so k_s < k_f is also accepted.
    """
    phi_s = np.asarray(phi_s, dtype=float)
    k_hi, k_lo = (k_s, k_f) if k_s >= k_f else (k_f, k_s)
    phi_hi = phi_s if k_s >= k_f else 1.0 - phi_s
    phi_lo = 1.0 - phi_hi
    if k_hi == k_lo:
        v = np.full_like(phi_s, k_hi)
        return v, v.copy()
    # upper (Maxwell-Garnett with the better conductor as matrix), written stably
    upper = k_hi * (1.0 - 3.0 * phi_lo * (k_hi - k_lo) / (3.0 * k_hi - phi_hi * (k_hi - k_lo)))
    if k_lo == 0.0:
        lower = np.zeros_like(phi_s)
    else:
        lower = k_lo + phi_hi / (1.0 / (k_hi - k_lo) + phi_lo / (3.0 * k_lo))
    return lower, upper


def wiener_bounds(phi_s: Any, k_s: float, k_f: float) -> tuple[np.ndarray, np.ndarray]:
    """Wiener bounds: harmonic (series, lower) and arithmetic (parallel, upper) means."""
    phi_s = np.asarray(phi_s, dtype=float)
    upper = phi_s * k_s + (1.0 - phi_s) * k_f
    if k_s == 0.0 or k_f == 0.0:
        lower = np.zeros_like(phi_s)
    else:
        lower = 1.0 / (phi_s / k_s + (1.0 - phi_s) / k_f)
    return lower, upper


def rayleigh_sc_spheres(phi: Any, k_p: float, k_m: float) -> np.ndarray:
    """Rayleigh (1892) formula for a simple-cubic array of spheres (volume fraction phi).

        k_eff / k_m = 1 + 3 phi / ( 1/beta - phi - 0.525 (alpha - 1)/(alpha + 4/3) phi^(10/3) ),
        alpha = k_p / k_m,  beta = (alpha - 1)/(alpha + 2).

    Accurate to ~1 % for phi <= 0.3 (cf. Perrins, McKenzie & McPhedran 1979,
    Proc. R. Soc. A 369, 207). Used only as a verification reference.
    """
    phi = np.asarray(phi, dtype=float)
    alpha = k_p / k_m
    beta = (alpha - 1.0) / (alpha + 2.0)
    denom = 1.0 / beta - phi - 0.525 * (alpha - 1.0) / (alpha + 4.0 / 3.0) * phi ** (10.0 / 3.0)
    return k_m * (1.0 + 3.0 * phi / denom)
