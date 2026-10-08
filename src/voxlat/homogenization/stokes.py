"""Periodic creeping-flow (Stokes) permeability tensor K of a voxel cell (Task 4).

Governing equations
-------------------
Steady, incompressible creeping flow of a Newtonian fluid (viscosity mu) in the
fluid voxels of a periodic cell Y, driven by a uniform body force f per unit
volume (equivalent to a macroscopic pressure gradient grad P = -f):

    -mu lap u + grad p = f          in the fluid
            div u      = 0          in the fluid
                u      = 0          on the fluid-solid interface (no slip, no penetration)
    u, p Y-periodic.

The superficial (Darcy) velocity U = <u> (average over the WHOLE cell, u = 0 in
the solid) is linear in f, which defines the permeability tensor

    U = (1/mu) K f,          K_ij = mu <u_i^(j)> / |f|,   u^(j) = solution for f = e_j.

K is symmetric positive semi-definite (Onsager); it is positive definite when the
fluid percolates in all three directions. Its energy form (Hill-Mandel analogue)

    K_ij = (mu / |f|^2) < grad u^(i) : grad u^(j) >

is a Gram matrix and hence symmetric PSD by construction.

Discretization: staggered (MAC) finite volumes on the voxel grid
----------------------------------------------------------------
Voxel edge h; the solver works in voxel units (h = 1, mu = 1, |f| = 1) and
scales K by h^2 at the end. Pressure p_c lives at fluid-cell centres; the
velocity component u_d lives on the +d face of cell c (between c and c + e_d,
periodic wrap). The solid-fluid interface is the set of voxel faces between a
solid and a fluid voxel, exactly the geometry used by Tasks 2 and 3.

* Face (c, d) is ACTIVE (an unknown) iff both c and c + e_d are fluid. All other
  faces carry u_d = 0 exactly: wall-normal velocity at a wall face is the
  no-penetration condition, imposed exactly at the wall position.
* Continuity in fluid cell c: sum_d (u_d[c] - u_d[c - e_d]) = 0.
* x_d-momentum on active face (c, d): 7-point Laplacian of u_d plus
  (p[c + e_d] - p[c]) = f_d. Neighbour (c +- e_e, d):
    - active                      -> coupling -1 (standard central difference);
    - e = d, inactive             -> it is a wall-normal face, u = 0 at distance h
                                     (diagonal +1);
    - e != d, both cells of the neighbour face solid
                                  -> a flat wall lies h/2 away; mirror ghost
                                     u_ghost = -u (diagonal +2), the standard
                                     second-order MAC no-slip (e.g. Manwart et al.
                                     2002, Phys. Rev. E 66, 016702);
    - e != d, exactly one of them solid (staircase step)
                                  -> that neighbour face is itself a fluid-solid
                                     face with u = 0 at distance h (diagonal +1).
  The resulting A = -lap_h is symmetric positive definite whenever the cell
  contains any solid voxel; the three velocity components decouple in A.

Saddle-point system (symmetric, indefinite):

    [  A   -D^T ] [u]   [f]          D = discrete divergence (fluid cells x active faces),
    [ -D    0   ] [p] = [0]          -D^T = discrete pressure gradient.

The pressure is defined up to one constant per connected fluid component;
fluid cells without any active face are dropped. The right-hand side is
orthogonal to these null vectors, so the system is consistent.

Solver: MINRES (scipy) with the block-diagonal SPD preconditioner
diag(A_hat, I): A_hat = diag(A) (Jacobi) or one smoothed-aggregation AMG
V-cycle (pyamg, if installed) for the velocity block, and the pressure mass
matrix (the identity in voxel units, mu = 1) for the Schur block -- the
classic optimal block preconditioner for Stokes (Silvester & Wathen 1994,
SIAM J. Numer. Anal. 31:1352; Elman, Silvester & Wathen 2014, ch. 4).
Iterations grow ~ n with Jacobi (n = 48: 350-900, n = 64: 500-1200).

K is computed from the same solves in two ways:

    flux:    K_flux[i, j]   = (h^2 / N) sum_{faces normal to i} u_i^(j)    (= <u_i^(j)>)
    energy:  K_energy[i, j] = (h^2 / N) u^(i)T A u^(j)

At the discrete solution u^(i)T A u^(j) = u^(i)T f^(j) because D u^(i) = 0, so
K_energy = K_flux^T; ``permeability`` checks the agreement. The reported ``K``
is the symmetrized flux form: its error is quadratic in the solver residual
(it is the compliance functional of a symmetric system), the energy form's only
linear (measured: 3e-8 vs 4e-7 relative at tol 1e-8, gyroid n = 48).

Staircase error
---------------
* Walls aligned with the grid (plane Poiseuille, square duct): second order.
  Plane channel of H fluid voxels: K = (H^2 + 2)/12 h^2 exactly (continuum
  H^2/12), i.e. +2/H^2 relative error.
* A 45-degree wall (symmetric staircase) is still second order (inclined slit:
  exactly -4/N^2 along the slit, +8/N^2 along z, for porosity 0.5).
* Generic inclined and curved walls (slopes 1:2, 1:3, spheres, TPMS): the voxel
  staircase moves the effective wall by ~0.05 h into the fluid -> first-order
  error, K low by ~0.2-0.26 h/H for a slit of H voxels, plus a non-monotone
  "alignment" noise from how the surface cuts the grid. TPMS at n = 48: 0.1-0.6 %
  low (offset-averaged), diamond rho* = 0.5 1.7 %; far smaller than the 3-17 %
  of k_eff (Task 2). Sizes and the recommended two-grid Richardson estimator
  R(32, 64): STATUS.md, Task 4.

Why not FFT-Brinkman penalization: the penalized wall is smeared over the
penalty boundary layer (error O(sqrt(eps))), the spectral discretization rings
at the discontinuous solid indicator, and the conditioning grows as 1/eps.
Uzawa / augmented Lagrangian needs nested inner velocity solves. MAC + MINRES
has an exact no-penetration wall at the same voxel faces as Tasks 2 and 3, no
tuning parameter, and a cheap optimal-form preconditioner.

Derived outputs: porosity phi (fluid fraction), mean interstitial velocity
<u>_f = U / phi, and hydraulic tortuosity T = <|u|> / |<u>| (cell-centred
velocities; Duda, Koza & Matyka 2011, Phys. Rev. E 84, 036319).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, minres

from voxlat.utils.log import get_logger

__all__ = [
    "StokesSystem",
    "assemble_stokes_system",
    "solve_stokes",
    "PermeabilityResult",
    "permeability",
    "permeability_tpms",
    "tpms_symmetry",
    "ExtrapolatedPermeability",
    "extrapolated_permeability_tpms",
    "AveragedPermeability",
    "averaged_permeability_tpms",
    "kozeny_constant",
    "slit_permeability",
    "mac_slit_permeability",
    "rectangular_duct_mean_velocity",
    "ZICK_HOMSY_SC",
    "sangani_acrivos_sc_drag",
    "zick_homsy_sc_drag",
    "sc_sphere_permeability",
    "sphere_array_cell",
    "inclined_slit_cell",
    "inclined_slit_reference",
    "pyamg_available",
]

LOG = get_logger("homogenization.stokes")

Preconditioner = Literal["auto", "amg", "jacobi"]
Symmetry = Literal["none", "cubic", "tetragonal_z"]

#: Zick & Homsy (1982), J. Fluid Mech. 115:13-26, Table 2, simple-cubic array:
#: solid volume fraction c -> dimensionless drag K* = F / (6 pi mu a U), with F the
#: force per sphere balancing the mean pressure gradient over the whole cell
#: (F = |grad P| V_cell) and U the superficial velocity.
ZICK_HOMSY_SC: dict[float, float] = {
    0.027: 2.008,
    0.064: 2.810,
    0.125: 4.292,
    0.216: 7.442,
    0.343: 15.4,
    0.45: 28.1,
    0.5236: 42.1,
}


def pyamg_available() -> bool:
    """True if pyamg can be imported (enables the AMG velocity preconditioner)."""
    try:
        import pyamg  # noqa: F401
    except ImportError:
        return False
    return True


# =============================================================================
# Assembly
# =============================================================================
@dataclass
class StokesSystem:
    """Assembled MAC Stokes operator of a periodic voxel cell (voxel units, mu = 1).

    Attributes
    ----------
    solid:
        The input cell (bool, True = solid).
    active:
        Three bool masks; ``active[d][c]`` = face between c and c + e_d is an unknown.
    face_id:
        Three int arrays; global velocity-unknown index of face (c, d), -1 if inactive.
    offsets:
        Velocity unknowns of component d are ``offsets[d]:offsets[d+1]``.
    pressure_cells:
        Bool mask of fluid cells that carry a pressure unknown.
    pressure_id:
        Int array, pressure-unknown index per cell (-1 if none).
    A, D:
        Velocity Laplacian (n_u x n_u, SPD) and divergence (n_p x n_u).
    saddle:
        [[A, -D^T], [-D, 0]] (CSR).
    n_pressure_components:
        Connected fluid components (= dimension of the pressure null space).
    """

    solid: np.ndarray
    active: tuple[np.ndarray, np.ndarray, np.ndarray]
    face_id: tuple[np.ndarray, np.ndarray, np.ndarray]
    offsets: np.ndarray
    pressure_cells: np.ndarray
    pressure_id: np.ndarray
    A: sp.csr_matrix
    D: sp.csr_matrix
    saddle: sp.csr_matrix
    n_pressure_components: int
    assembly_time: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.solid.shape)  # type: ignore[return-value]

    @property
    def n_u(self) -> int:
        return int(self.offsets[-1])

    @property
    def n_p(self) -> int:
        return int(self.D.shape[0])

    @property
    def n_dofs(self) -> int:
        return self.n_u + self.n_p

    def force_vector(self, force: Sequence[float]) -> np.ndarray:
        """Right-hand side [f; 0] for a uniform body force (voxel units)."""
        b = np.zeros(self.n_dofs)
        for d in range(3):
            b[self.offsets[d]:self.offsets[d + 1]] = float(force[d])
        return b

    def faces_to_grid(self, u: np.ndarray) -> np.ndarray:
        """Velocity vector (n_u,) -> staggered grid (3, nx, ny, nz); 0 on inactive faces."""
        out = np.zeros((3, *self.shape))
        for d in range(3):
            out[d][self.active[d]] = u[self.offsets[d]:self.offsets[d + 1]]
        return out

    def pressure_to_grid(self, p: np.ndarray) -> np.ndarray:
        """Pressure vector (n_p,) -> cell array; NaN where there is no pressure unknown."""
        out = np.full(self.shape, np.nan)
        out[self.pressure_cells] = p
        return out


def assemble_stokes_system(solid: np.ndarray) -> StokesSystem:
    """Assemble the periodic MAC Stokes saddle-point system of a voxel cell.

    ``solid``: bool array (True = solid), index order (x, y, z), periodic in all
    three directions; any size >= 1 per axis. Equations and wall treatment: see
    the module docstring. Raises ``ValueError`` for a cell without solid
    (the periodic Stokes problem then has no bounded solution).
    """
    t0 = time.perf_counter()
    solid = np.asarray(solid, dtype=bool)
    if solid.ndim != 3:
        raise ValueError(f"cell must be 3-D, got shape {solid.shape}")
    if not solid.any():
        raise ValueError("cell has no solid voxel: the periodic Stokes problem is unbounded")
    shape = solid.shape
    fluid = ~solid

    active = tuple(fluid & np.roll(fluid, -1, axis=d) for d in range(3))
    counts = [int(a.sum()) for a in active]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    n_u = int(offsets[-1])
    face_id = []
    for d in range(3):
        fid = np.full(shape, -1, dtype=np.int64)
        fid[active[d]] = np.arange(counts[d], dtype=np.int64) + offsets[d]
        face_id.append(fid)

    # ---- velocity Laplacian A (three decoupled scalar blocks) -----------------
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    diag = np.zeros(n_u)
    for d in range(3):
        a = active[d]
        if not a.any():
            continue
        me = face_id[d][a]
        for e in range(3):
            for s in (-1, 1):
                nb = np.roll(face_id[d], -s, axis=e)[a]  # face (c + s e_e, d)
                ok = nb >= 0
                rows.append(me[ok])
                cols.append(nb[ok])
                vals.append(np.full(int(ok.sum()), -1.0))
                if e == d:
                    wall = np.ones(me.size)
                else:
                    c1 = np.roll(solid, -s, axis=e)  # cell c + s e_e
                    both = (c1 & np.roll(c1, -1, axis=d))[a]  # and c + s e_e + e_d
                    wall = np.where(both, 2.0, 1.0)
                diag[me] += np.where(ok, 1.0, wall)  # me unique within one (e, s) pass
    rows.append(np.arange(n_u))
    cols.append(np.arange(n_u))
    vals.append(diag)
    A = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_u, n_u)
    ).tocsr()
    A.sum_duplicates()
    A.eliminate_zeros()

    # ---- divergence D (fluid cells x faces) ---------------------------------
    n_cells = solid.size
    cell_idx = np.arange(n_cells).reshape(shape)
    drows, dcols, dvals = [], [], []
    for d in range(3):
        a = active[d]
        f = face_id[d][a]
        drows += [cell_idx[a], np.roll(cell_idx, -1, axis=d)[a]]
        dcols += [f, f]
        dvals += [np.ones(f.size), -np.ones(f.size)]
    D_all = sp.coo_matrix(
        (np.concatenate(dvals), (np.concatenate(drows), np.concatenate(dcols))), shape=(n_cells, n_u)
    ).tocsr()
    D_all.sum_duplicates()
    D_all.eliminate_zeros()
    has_p = np.diff(D_all.indptr) > 0  # cells touching at least one active face (all fluid)
    D = D_all[has_p].tocsr()
    pressure_cells = has_p.reshape(shape)
    pressure_id = np.full(shape, -1, dtype=np.int64)
    pressure_id[pressure_cells] = np.arange(int(has_p.sum()))

    n_comp = 0
    if D.shape[0] > 0:
        n_comp, _ = connected_components(D @ D.T, directed=False)

    saddle = sp.bmat([[A, -D.T], [-D, None]], format="csr")
    return StokesSystem(
        solid=solid,
        active=active,  # type: ignore[arg-type]
        face_id=tuple(face_id),  # type: ignore[arg-type]
        offsets=offsets,
        pressure_cells=pressure_cells,
        pressure_id=pressure_id,
        A=A,
        D=D,
        saddle=saddle,
        n_pressure_components=int(n_comp),
        assembly_time=time.perf_counter() - t0,
    )


def _make_preconditioner(system: StokesSystem, kind: Preconditioner) -> tuple[LinearOperator, str]:
    """Block-diagonal SPD preconditioner diag(A_hat^-1, I) for MINRES."""
    if kind == "auto":
        kind = "amg" if pyamg_available() else "jacobi"
    n_u, n_p = system.n_u, system.n_p
    if kind == "jacobi":
        dinv = np.concatenate([1.0 / system.A.diagonal(), np.ones(n_p)])
        return LinearOperator(system.saddle.shape, matvec=lambda x: dinv * x, dtype=np.float64), "jacobi"
    if kind == "amg":
        try:
            import pyamg
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("preconditioner='amg' needs pyamg (pip install pyamg)") from exc
        # symmetric smoothers -> symmetric (SPD) V-cycle, as MINRES requires
        ml = pyamg.smoothed_aggregation_solver(
            system.A, symmetry="hermitian", max_coarse=500,
            presmoother=("gauss_seidel", {"sweep": "symmetric"}),
            postsmoother=("gauss_seidel", {"sweep": "symmetric"}),
        )
        Pu = ml.aspreconditioner(cycle="V")

        def mv(x: np.ndarray) -> np.ndarray:
            y = np.empty_like(x)
            y[:n_u] = Pu @ x[:n_u]
            y[n_u:] = x[n_u:]
            return y

        return LinearOperator(system.saddle.shape, matvec=mv, dtype=np.float64), "amg"
    raise ValueError(f"unknown preconditioner {kind!r}")


@dataclass
class StokesSolution:
    """One Stokes solve (voxel units): face velocities, pressure, solver diagnostics."""

    u: np.ndarray  # (n_u,)
    p: np.ndarray  # (n_p,)
    iterations: int
    residual: float  # ||b - K x|| / ||b||
    divergence: float  # max |D u| / max |u|
    info: int
    solve_time: float


def solve_stokes(
    system: StokesSystem,
    force: Sequence[float],
    *,
    tol: float = 1e-8,
    maxiter: int | None = None,
    preconditioner: Preconditioner | LinearOperator = "auto",
) -> StokesSolution:
    """Solve the MAC Stokes system for a uniform body force (voxel units, mu = 1).

    ``tol`` is scipy MINRES's relative stopping tolerance (a backward-error test,
    not ||r||/||b||). With the default 1e-8 the true relative residual is
    ~1e-4 and K is converged to ~1e-7 relative (K error ~ residual^2); see
    STATUS.md, Task 4. The true residual and the velocity divergence are reported.
    """
    t0 = time.perf_counter()
    b = system.force_vector(force)
    if maxiter is None:
        maxiter = max(5000, 60 * max(system.shape))
    if isinstance(preconditioner, str):
        M, _ = _make_preconditioner(system, preconditioner)
    else:
        M = preconditioner
    bnorm = float(np.linalg.norm(b))
    if bnorm == 0.0 or system.n_u == 0:
        return StokesSolution(np.zeros(system.n_u), np.zeros(system.n_p), 0, 0.0, 0.0, 0,
                              time.perf_counter() - t0)
    it = [0]

    def _count(_xk: np.ndarray) -> None:
        it[0] += 1

    x, info = minres(system.saddle, b, M=M, rtol=tol, maxiter=maxiter, callback=_count)
    r = float(np.linalg.norm(b - system.saddle @ x) / bnorm)
    u = x[: system.n_u]
    p = x[system.n_u:]
    umax = float(np.max(np.abs(u))) if u.size else 0.0
    div = float(np.max(np.abs(system.D @ u)) / umax) if umax > 0 and system.n_p else 0.0
    return StokesSolution(u, p, it[0], r, div, int(info), time.perf_counter() - t0)


# =============================================================================
# Result
# =============================================================================
@dataclass
class PermeabilityResult:
    """Output of ``permeability``.

    Units: lengths in units of ``voxel_size`` (default: the voxel edge h; the TPMS
    wrappers use the cell size L, so K is in L^2). Velocities are per unit
    body force and viscosity, i.e. in units of |f| * voxel_size^2 / mu.

    Attributes
    ----------
    K:
        3x3 permeability tensor (symmetrized flux form), units voxel_size^2.
        With ``symmetry`` != "none" the unsolved entries are filled by symmetry;
        with a ``directions`` subset they are NaN.
    K_flux, K_energy:
        Flux form (column j = <u> for f = e_j; error ~ solver residual^2) and
        energy form (Gram matrix, symmetric PSD by construction; error ~ residual),
        solved columns only (NaN elsewhere).
    agreement:
        max |K_flux - K_energy^T| / max |K| over solved columns (~ solver tol).
    directions:
        Body-force directions actually solved (0 = x, 1 = y, 2 = z).
    symmetry:
        Symmetry used to fill the tensor ("none", "cubic", "tetragonal_z").
    porosity:
        Fluid volume fraction phi of the cell.
    interstitial_velocity:
        3x3; column j = <u>_f (fluid average) for f = e_j (= K_flux[:, j] / phi).
    mean_speed:
        Per solved direction, <|u|>_f (fluid average of the cell-centred speed).
    tortuosity:
        Per solved direction, T = <|u|> / |<u>| (Duda, Koza & Matyka 2011).
    velocity:
        If ``return_fields``: (n_dirs, 3, nx, ny, nz) float32 staggered face
        velocities; ``velocity[k, d]`` at the +d face of every cell, for force
        direction ``directions[k]``.
    pressure:
        If ``return_fields``: (n_dirs, nx, ny, nz) float32 periodic pressure
        fluctuation (units |f| voxel_size), NaN in solid.
    """

    K: np.ndarray
    K_flux: np.ndarray
    K_energy: np.ndarray
    agreement: float
    directions: tuple[int, ...]
    symmetry: str
    porosity: float
    interstitial_velocity: np.ndarray
    mean_speed: dict[int, float]
    tortuosity: dict[int, float]
    voxel_size: float
    shape: tuple[int, int, int]
    iterations: list[int]
    residuals: list[float]
    divergence: list[float]
    preconditioner: str
    n_dofs: int
    n_pressure_components: int
    wall_time: float
    velocity: np.ndarray | None = field(default=None, repr=False)
    pressure: np.ndarray | None = field(default=None, repr=False)

    # ---- derived ------------------------------------------------------------
    @property
    def eigenvalues(self) -> np.ndarray:
        """Principal permeabilities (ascending)."""
        return np.linalg.eigvalsh(self.K)

    @property
    def mean(self) -> float:
        """Isotropic part tr(K)/3."""
        return float(np.trace(self.K) / 3.0)

    @property
    def is_spd(self) -> bool:
        """Symmetric and all eigenvalues > 1e-10 * largest."""
        ev = self.eigenvalues
        if ev[-1] <= 0:
            return False
        sym = np.allclose(self.K, self.K.T, rtol=0, atol=1e-12 * ev[-1])
        return bool(sym and ev[0] > 1e-10 * ev[-1])

    @property
    def anisotropy(self) -> float:
        """(lambda_max - lambda_min) / mean eigenvalue."""
        ev = self.eigenvalues
        return float((ev[-1] - ev[0]) / ev.mean()) if ev.mean() > 0 else float("nan")

    def cell_velocity(self, k: int = 0) -> np.ndarray:
        """Cell-centred velocity (3, nx, ny, nz) for the k-th solved direction.

        u_c,d = (u_d[c - e_d] + u_d[c]) / 2 (needs ``return_fields``).
        """
        if self.velocity is None:
            raise ValueError("run permeability(..., return_fields=True) to keep the velocity")
        v = self.velocity[k].astype(np.float64)
        return np.stack([0.5 * (v[d] + np.roll(v[d], 1, axis=d)) for d in range(3)])

    def as_row(self, prefix: str = "K") -> dict[str, float]:
        """Flat dict for tables: 6 components, mean, eigen-extremes, phi, tortuosity, timing."""
        K = self.K
        names = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2), "xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
        row: dict[str, float] = {f"{prefix}_{s}": float(K[i, j]) for s, (i, j) in names.items()}
        ev = self.eigenvalues
        row.update(
            {
                f"{prefix}_mean": self.mean,
                f"{prefix}_min": float(ev[0]),
                f"{prefix}_max": float(ev[-1]),
                f"{prefix}_anisotropy": self.anisotropy,
                "porosity": self.porosity,
                "tortuosity": float(np.nanmean(list(self.tortuosity.values())))
                if np.isfinite(list(self.tortuosity.values())).any() else float("nan"),
                f"{prefix}_flux_energy_agreement": self.agreement,
                f"{prefix}_wall_time_s": self.wall_time,
                f"{prefix}_iterations_max": int(max(self.iterations) if self.iterations else 0),
            }
        )
        return row


def _symmetry_directions(symmetry: Symmetry) -> tuple[int, ...]:
    return {"none": (0, 1, 2), "cubic": (0,), "tetragonal_z": (0, 2)}[symmetry]


# =============================================================================
# Main solver
# =============================================================================
def permeability(
    cell: np.ndarray,
    *,
    voxel_size: float = 1.0,
    symmetry: Symmetry = "none",
    directions: Sequence[int] | None = None,
    tol: float = 1e-8,
    preconditioner: Preconditioner = "auto",
    maxiter: int | None = None,
    return_fields: bool = False,
    check: bool = True,
    agreement_tol: float = 1e-3,
) -> PermeabilityResult:
    """Periodic permeability tensor of a voxel cell (MAC Stokes, see module docstring).

    Parameters
    ----------
    cell:
        Bool array (True = solid), index order (x, y, z), periodic in x, y, z.
        Voxels are cubes of edge ``voxel_size``.
    voxel_size:
        Voxel edge in the length unit wanted for K (K scales with voxel_size^2).
        1.0 -> voxel units; 1/n -> units of the cell size L for n voxels per L;
        L/n in metres -> m^2.
    symmetry:
        "none" (solve f = e_x, e_y, e_z; full tensor), "cubic" (solve e_x only;
        K = K_xx I -- valid for gyroid/diamond cells), "tetragonal_z" (solve
        e_x, e_z; K = diag(K_xx, K_xx, K_zz) -- cubic cells stretched along z).
    directions:
        Override the solved directions (then ``symmetry`` must be "none";
        unsolved entries of K are NaN).
    tol, preconditioner, maxiter:
        MINRES settings (see ``solve_stokes``); preconditioner "auto" = AMG
        velocity block if pyamg is installed, else Jacobi.
    return_fields:
        Keep velocity (float32, staggered) and pressure fields.
    check:
        Raise ``RuntimeError`` if MINRES fails or the flux and energy forms of K
        disagree by more than ``agreement_tol`` (relative to max |K|). The energy
        form's error is linear in the residual (the flux form's is quadratic), so at
        tol = 1e-8 they agree to ~1e-7 (G/D, n = 48) ... 1e-4 (blends, n = 64);
        a sloppy solve (tol = 1e-4) disagrees by ~1e-2.

    Boundary conditions: periodic u, p on all six faces; no slip and no
    penetration on every solid-fluid voxel face; uniform body force e_j.
    """
    t0 = time.perf_counter()
    solid = np.asarray(cell)
    if solid.dtype != bool:
        raise TypeError("cell must be a bool array (True = solid)")
    if solid.ndim != 3:
        raise ValueError(f"cell must be 3-D, got shape {solid.shape}")
    if directions is not None:
        if symmetry != "none":
            raise ValueError("give either directions or symmetry, not both")
        dirs = tuple(sorted({int(d) for d in directions}))
        if not dirs or any(d not in (0, 1, 2) for d in dirs):
            raise ValueError(f"directions must be a subset of (0, 1, 2), got {directions}")
    else:
        dirs = _symmetry_directions(symmetry)
    shape = tuple(int(s) for s in solid.shape)
    N = solid.size
    phi = float(1.0 - solid.mean())
    h2 = float(voxel_size) ** 2

    nan33 = np.full((3, 3), np.nan)
    if phi == 0.0:  # no fluid: K = 0
        Z = np.zeros((3, 3))
        return PermeabilityResult(
            K=Z, K_flux=Z.copy(), K_energy=Z.copy(), agreement=0.0, directions=dirs,
            symmetry=symmetry, porosity=0.0, interstitial_velocity=nan33.copy(),
            mean_speed={d: 0.0 for d in dirs}, tortuosity={d: float("nan") for d in dirs},
            voxel_size=float(voxel_size), shape=shape,  # type: ignore[arg-type]
            iterations=[0] * len(dirs), residuals=[0.0] * len(dirs), divergence=[0.0] * len(dirs),
            preconditioner="none", n_dofs=0, n_pressure_components=0,
            wall_time=time.perf_counter() - t0,
        )

    system = assemble_stokes_system(solid)
    M, pc_name = _make_preconditioner(system, preconditioner)

    U = np.zeros((system.n_u, len(dirs)))
    iters, resid, divs = [], [], []
    vel = np.zeros((len(dirs), 3, *shape), dtype=np.float32) if return_fields else None
    prs = np.zeros((len(dirs), *shape), dtype=np.float32) if return_fields else None
    mean_speed: dict[int, float] = {}
    tort: dict[int, float] = {}
    K_flux = nan33.copy()
    n_fluid = int((~solid).sum())
    mnorm: dict[int, float] = {}
    for k, j in enumerate(dirs):
        force = [0.0, 0.0, 0.0]
        force[j] = 1.0
        sol = solve_stokes(system, force, tol=tol, maxiter=maxiter, preconditioner=M)
        if check and sol.info != 0 and sol.residual > 1e-3:
            raise RuntimeError(
                f"MINRES did not converge for f = e_{'xyz'[j]}: info={sol.info}, "
                f"relative residual {sol.residual:.2e} after {sol.iterations} iterations"
            )
        U[:, k] = sol.u
        iters.append(sol.iterations)
        resid.append(sol.residual)
        divs.append(sol.divergence)
        ug = system.faces_to_grid(sol.u)
        K_flux[:, j] = ug.reshape(3, -1).sum(axis=1) / N
        # cell-centred velocity -> tortuosity (Duda et al. 2011)
        uc = np.stack([0.5 * (ug[d] + np.roll(ug[d], 1, axis=d)) for d in range(3)])
        speed_sum = float(np.sqrt((uc**2).sum(axis=0)).sum())
        mean_vec = uc.reshape(3, -1).sum(axis=1)
        mean_speed[j] = speed_sum / n_fluid * h2
        mnorm[j] = float(np.linalg.norm(mean_vec))
        tort[j] = speed_sum / mnorm[j] if mnorm[j] > 0 else float("nan")
        if vel is not None and prs is not None:
            vel[k] = (ug * h2).astype(np.float32)
            prs[k] = (system.pressure_to_grid(sol.p) * float(voxel_size)).astype(np.float32)

    # no net flow along a direction (non-percolating) -> tortuosity undefined
    # (voxel units: any open flow path has K >= ~1/12 * phi, so mean velocities below
    # 1e-14 are round-off; the relative test catches blocked directions next to open ones)
    mmax = max(mnorm.values())
    for j in dirs:
        if mnorm[j] <= 1e-9 * mmax or mnorm[j] / N <= 1e-14:
            tort[j] = float("nan")

    AU = system.A @ U
    G = (U.T @ AU) / N  # Gram matrix (len(dirs) x len(dirs))
    K_en = nan33.copy()
    for a, i in enumerate(dirs):
        for b, j in enumerate(dirs):
            K_en[i, j] = G[a, b]
    sub_f = K_flux[np.ix_(dirs, dirs)]
    sub_e = K_en[np.ix_(dirs, dirs)]
    # relative to max |K|, with an absolute floor (voxel units) for closed cells, K = 0
    scale = max(float(np.max(np.abs(sub_e))), 1e-12)
    agreement = float(np.max(np.abs(sub_f - sub_e.T)) / scale)
    if check and agreement > agreement_tol:
        raise RuntimeError(
            f"flux- and energy-based K disagree: rel. diff {agreement:.2e} > {agreement_tol:.0e}"
        )

    # reported tensor: symmetrized flux form (error ~ residual^2, vs ~ residual for the
    # energy form), unsolved entries filled by symmetry
    if symmetry == "cubic":
        K = K_flux[0, 0] * np.eye(3)
    elif symmetry == "tetragonal_z":
        K = np.diag([K_flux[0, 0], K_flux[0, 0], K_flux[2, 2]])
    else:
        K = nan33.copy()
        K[np.ix_(dirs, dirs)] = 0.5 * (sub_f + sub_f.T)
    K = K * h2
    K_flux = K_flux * h2
    K_en = K_en * h2
    vi = K_flux / phi
    wall = time.perf_counter() - t0
    LOG.debug(
        "K shape=%s dofs=%d pc=%s iters=%s res=%s agreement=%.1e time=%.2fs",
        shape, system.n_dofs, pc_name, iters, ["%.1e" % r for r in resid], agreement, wall,
    )
    return PermeabilityResult(
        K=K,
        K_flux=K_flux,
        K_energy=K_en,
        agreement=agreement,
        directions=dirs,
        symmetry=symmetry,
        porosity=phi,
        interstitial_velocity=vi,
        mean_speed=mean_speed,
        tortuosity=tort,
        voxel_size=float(voxel_size),
        shape=shape,  # type: ignore[arg-type]
        iterations=iters,
        residuals=resid,
        divergence=divs,
        preconditioner=pc_name,
        n_dofs=system.n_dofs,
        n_pressure_components=system.n_pressure_components,
        wall_time=wall,
        velocity=vel,
        pressure=prs,
    )


# =============================================================================
# TPMS wrappers
# =============================================================================
def tpms_symmetry(params: Any) -> Symmetry:
    """Symmetry of K for a TPMS cell: pure gyroid/diamond (w in {0, 1}) are cubic,
    "tetragonal_z" if stretched along z only, blends 0 < w < 1 are trigonal ("none")."""
    a = tuple(float(v) for v in params.a)
    pure = float(params.w) in (0.0, 1.0)
    if not pure:
        return "none"
    if a[0] == a[1] == a[2]:
        return "cubic"
    if a[0] == a[1]:
        return "tetragonal_z"
    return "none"


def permeability_tpms(
    params: Any,
    n: int = 48,
    *,
    offset: Sequence[float] | None = None,
    symmetry: Symmetry | Literal["auto"] = "auto",
    **kwargs: Any,
) -> PermeabilityResult:
    """K of a TPMS cell (``voxlat.geometry.TPMSParams``), n voxels per L, K in units of L^2.

    ``symmetry="auto"`` uses ``tpms_symmetry(params)``: one solve (f = e_x) for
    pure gyroid/diamond, three for blends. ``offset`` shifts the TPMS against the
    grid. A single resolution carries the staircase + alignment error quantified
    in STATUS.md, Task 4; use ``extrapolated_permeability_tpms`` for production.
    """
    from voxlat.geometry.tpms import voxelize

    sym = tpms_symmetry(params) if symmetry == "auto" else symmetry
    solid = voxelize(params, n, offset=offset)
    return permeability(solid, voxel_size=1.0 / n, symmetry=sym, **kwargs)  # type: ignore[arg-type]


@dataclass
class ExtrapolatedPermeability:
    """Two-grid Richardson estimate of K (units L^2) for a TPMS cell.

    ``K = (n2^p K(n2) - n1^p K(n1)) / (n2^p - n1^p)`` component-wise (p = 1 default);
    ``coarse``/``fine`` are the raw solutions; ``correction`` = max|K - K(n2)| / max|K|.
    Porosity and tortuosity are taken from the fine grid.
    """

    K: np.ndarray
    n: tuple[int, int]
    p: float
    coarse: PermeabilityResult
    fine: PermeabilityResult
    correction: float
    wall_time: float

    @property
    def mean(self) -> float:
        return float(np.trace(self.K) / 3.0)

    @property
    def eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvalsh(self.K)

    @property
    def porosity(self) -> float:
        return self.fine.porosity

    @property
    def tortuosity(self) -> dict[int, float]:
        return self.fine.tortuosity


def extrapolated_permeability_tpms(
    params: Any,
    n: tuple[int, int] = (32, 64),
    *,
    p: float = 1.0,
    offset: Sequence[float] | None = None,
    **kwargs: Any,
) -> ExtrapolatedPermeability:
    """Production K of a TPMS cell: two resolutions + Richardson of order p.

    Both grids use the same ``offset``; the recommended pair and order come from
    the Task 4 convergence study (STATUS.md).
    """
    t0 = time.perf_counter()
    n1, n2 = (int(v) for v in n)
    if not n1 < n2:
        raise ValueError("need n1 < n2")
    r1 = permeability_tpms(params, n1, offset=offset, **kwargs)
    r2 = permeability_tpms(params, n2, offset=offset, **kwargs)
    a, b = float(n1) ** p, float(n2) ** p
    K = (b * r2.K - a * r1.K) / (b - a)
    K = 0.5 * (K + K.T)
    scale = float(np.max(np.abs(K)))
    corr = float(np.max(np.abs(K - r2.K)) / scale) if scale > 0 else 0.0
    return ExtrapolatedPermeability(K, (n1, n2), float(p), r1, r2, corr, time.perf_counter() - t0)


@dataclass
class AveragedPermeability:
    """K (units L^2) averaged over rigid grid shifts at one resolution n."""

    K: np.ndarray
    K_std: np.ndarray
    n: int
    offsets: np.ndarray
    results: list[PermeabilityResult]
    wall_time: float

    @property
    def mean(self) -> float:
        return float(np.trace(self.K) / 3.0)


def averaged_permeability_tpms(
    params: Any, n: int = 48, n_offsets: int = 3, *, seed: int = 2026, **kwargs: Any
) -> AveragedPermeability:
    """K of a TPMS cell averaged over ``n_offsets`` grid shifts (offset 0 + seeded random)."""
    from voxlat.homogenization.elasticity import grid_offsets

    t0 = time.perf_counter()
    offs = grid_offsets(int(n_offsets), seed)
    res = [permeability_tpms(params, n, offset=tuple(o), **kwargs) for o in offs]
    Ks = np.stack([r.K for r in res])
    K = Ks.mean(axis=0)
    return AveragedPermeability(
        0.5 * (K + K.T), Ks.std(axis=0, ddof=1) if len(res) > 1 else np.zeros((3, 3)),
        int(n), offs, res, time.perf_counter() - t0,
    )


def kozeny_constant(K: Any, porosity: Any, a_sf: Any) -> np.ndarray:
    """Effective Kozeny constant c_K = phi^3 / (K a_sf^2).

    ``a_sf`` is the interface area per unit TOTAL volume (``TPMSMetrics.a_sf_L`` in
    1/L when K is in L^2). Kozeny-Carman K = phi^3 / (c_K a_sf^2) with c_K ~ 5 for
    packed beds (Carman 1937); c_K = 2 for a cylindrical capillary, 3 for a slit.
    """
    K = np.asarray(K, dtype=float)
    return np.asarray(porosity, dtype=float) ** 3 / (K * np.asarray(a_sf, dtype=float) ** 2)


# =============================================================================
# Analytic references and verification geometries
# =============================================================================
def slit_permeability(H: Any) -> np.ndarray:
    """Intrinsic permeability of a plane slit of width H (plane Poiseuille): H^2 / 12."""
    return np.asarray(H, dtype=float) ** 2 / 12.0


def mac_slit_permeability(H: Any) -> np.ndarray:
    """Exact DISCRETE (MAC, mirror-ghost walls) slit permeability for H fluid voxels:
    (H^2 + 2) / 12 in voxel units -> relative error +2/H^2 vs ``slit_permeability``."""
    H = np.asarray(H, dtype=float)
    return (H**2 + 2.0) / 12.0


def rectangular_duct_mean_velocity(a: float, b: float, n_terms: int = 200) -> float:
    """Mean velocity of fully developed laminar flow in an a x b rectangular duct,
    per unit pressure gradient and viscosity (series solution, e.g. White,
    *Viscous Fluid Flow*, 3rd ed., eq. 3-48; Shah & London 1978):

        u_mean = (b^2/12) [1 - (192 b / (pi^5 a)) sum_{k odd} tanh(k pi a / (2 b)) / k^5],  a >= b.

    Square duct: u_mean = 0.0351443 a^2.
    """
    a, b = (float(a), float(b)) if a >= b else (float(b), float(a))
    k = np.arange(1, 2 * n_terms, 2, dtype=float)
    s = float(np.sum(np.tanh(k * np.pi * a / (2.0 * b)) / k**5))
    return b**2 / 12.0 * (1.0 - 192.0 * b / (np.pi**5 * a) * s)


def sangani_acrivos_sc_drag(c: Any) -> np.ndarray:
    """Dilute-to-moderate series for the SC array drag K* = F/(6 pi mu a U)
    (Sangani & Acrivos 1982, Int. J. Multiphase Flow 8:343; leading term Hasimoto 1959):

        1/K* = 1 - 1.7601 chi + chi^3 - 1.5593 chi^6 + 3.9799 chi^8 - 3.0734 chi^10,  chi = c^(1/3).

    Within 1 % of Zick & Homsy (1982) for c <= 0.216.
    """
    chi = np.asarray(c, dtype=float) ** (1.0 / 3.0)
    inv = 1 - 1.7601 * chi + chi**3 - 1.5593 * chi**6 + 3.9799 * chi**8 - 3.0734 * chi**10
    return 1.0 / inv


def zick_homsy_sc_drag(c: Any) -> np.ndarray:
    """SC-array drag K*(c) at any solid fraction: monotone cubic (PCHIP) interpolation of
    ``ZICK_HOMSY_SC`` in (log c, log K*) for 0.027 <= c <= 0.5236, the Sangani-Acrivos
    series below 0.027. Used to compare voxel spheres at their *realized* volume
    fraction (the voxel shell structure misses the target c by up to a few %, and
    d ln k / d ln c ~ -1 ... -1.3). Interpolation error vs the series < 0.3 % for
    c <= 0.216 (tested)."""
    from scipy.interpolate import PchipInterpolator

    c = np.asarray(c, dtype=float)
    cs = np.array(sorted(ZICK_HOMSY_SC))
    ks = np.array([ZICK_HOMSY_SC[v] for v in cs])
    f = PchipInterpolator(np.log(cs), np.log(ks))
    if np.any(c > cs[-1] + 1e-12):
        raise ValueError("SC spheres overlap above c = pi/6")
    out = np.where(c >= cs[0], np.exp(f(np.log(np.clip(c, cs[0], cs[-1])))), sangani_acrivos_sc_drag(c))
    return out


def sc_sphere_permeability(c: Any, drag: Any) -> np.ndarray:
    """Permeability / l^2 of a simple-cubic sphere array (period l, solid fraction c)
    from the dimensionless drag K* (Zick & Homsy convention: F = |grad P| l^3,
    superficial U): k = l^3 / (6 pi a K*), a = l (3c / 4 pi)^(1/3)."""
    c = np.asarray(c, dtype=float)
    a = (3.0 * c / (4.0 * np.pi)) ** (1.0 / 3.0)
    return 1.0 / (6.0 * np.pi * a * np.asarray(drag, dtype=float))


def sphere_array_cell(n: int, c: float, *, match_volume: bool = True) -> np.ndarray:
    """One SC unit cell (n^3 voxels) with a centred sphere of solid fraction c.

    ``match_volume``: choose the voxel radius (a shell of equal centre distances
    at a time, so the cell keeps its cubic symmetry) whose voxel count is closest
    to c n^3 -- removes the volume-fraction error that otherwise dominates the
    comparison at moderate n. Otherwise the geometric radius is used.
    """
    x = np.arange(n) + 0.5 - n / 2.0
    r2 = x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2
    if not match_volume:
        a = (3.0 * c / (4.0 * np.pi)) ** (1.0 / 3.0) * n
        return r2 <= a * a
    target = c * n**3
    vals, counts = np.unique(r2, return_counts=True)
    cum = np.cumsum(counts)
    k = int(np.argmin(np.abs(cum - target)))
    return r2 <= vals[k]


def inclined_slit_cell(N: int, porosity: float = 0.5, slope: int = 1, nz: int = 1) -> np.ndarray:
    """Periodic inclined slit: N x N x nz cell, fluid where (i + slope j) mod N < round(porosity N).

    Slit normal n = (1, slope, 0)/sqrt(1 + slope^2), tangent t = (slope, -1, 0)/sqrt(1 + slope^2),
    period along the normal P = N h / sqrt(1 + slope^2). Every residue class holds exactly N
    voxels, so the fluid fraction is exactly round(porosity N)/N. Continuum superficial K
    (voxel units): phi^3 P^2 / 12 along t and z, 0 along n (``inclined_slit_reference``).
    slope = 1 is a 45-degree wall (staircase error happens to be 2nd order);
    slope = 2, 3 give the generic 1st-order staircase error of a non-aligned wall.
    """
    i = np.arange(N)
    s = (i[:, None] + int(slope) * i[None, :]) % N
    fluid2d = s < int(round(porosity * N))
    return np.repeat(~fluid2d[:, :, None], nz, axis=2)


def inclined_slit_reference(N: int, porosity: float = 0.5, slope: int = 1) -> tuple[float, np.ndarray, np.ndarray]:
    """(K_along, t, n) for ``inclined_slit_cell``: continuum superficial K along the slit
    (voxel units), unit tangent t (in the x-y plane) and unit normal n."""
    phi = int(round(porosity * N)) / N
    q = float(slope)
    P = N / np.sqrt(1.0 + q * q)
    t = np.array([q, -1.0, 0.0]) / np.sqrt(1.0 + q * q)
    nrm = np.array([1.0, q, 0.0]) / np.sqrt(1.0 + q * q)
    return phi**3 * P**2 / 12.0, t, nrm
