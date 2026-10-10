"""Finite-gap (scale-separation) study: TPMS lattice strips between solid walls (Task 7, RQ1).

Question
--------
The jacket lattice fills a gap of h = 6 mm with cells of L = 2-6 mm, i.e. only
N = h/L = 1-3 cells across the gap. Bulk (periodic) closures assume N -> infinity.
How wrong are they for finite N, and can a simple correction remove the error?

Frame and geometry
------------------
Jacket frame: x = r (gap normal), y = s (circumferential), z = axial. A *strip*
is a TPMS lattice of N cells across a gap H = N L bounded by solid walls,
periodic and ONE cell wide in y and z. The homogenization solvers of Tasks 2-4
are periodic in x as well, so the computational cell is the stack

    [ lattice, thickness H | solid wall, thickness t_w ]      repeated along x,

i.e. every lattice layer is bounded by the same wall on both sides (no-slip,
perfectly bonded, k_s). Both walls cut the lattice: the inner wall at the
*phase* phi_0 (cell units along x), the outer one at phi_0 + N; a fractional N
(1.5) therefore cuts the two walls at different phases. Symmetry-distinct cuts:
``default_phases`` and ``reduced_phase``.

Lattice voxels (n per L): the level set phi_w is sampled at
xi_x = (i + 1/2)/n + phi_0, i = 0 ... n_lat - 1, n_lat = round(N n), and in-plane
exactly like the bulk cell of ``voxlat.geometry.voxelize`` (stretch a_z along z).
Uniform strips use the bulk threshold c(w, rho*) of the SAME grid, so the strip
lattice is a bit-exact x-tiling of the bulk reference cell (``bulk_cell``): the
voxel staircase error (Tasks 2-4) is common to strip and bulk and cancels in
their ratio. Graded strips use rho*(x) = rho_0 + g (x/L - H/2L) with
g = |grad rho*| L (change of rho* per cell) and a threshold per voxel layer,
c(rho*(x_i)), from bisection curves on the same grid (``threshold_curve``).

Apparent lattice properties (the "plain-wall" homogenized interpretation)
------------------------------------------------------------------------
Each strip is solved with the unchanged periodic solvers; the walls are then
removed analytically, assuming the homogenized picture "homogeneous lattice
layer + plain walls" (exact for a laminate of homogeneous layers):

* through-gap conductivity (series):          T / k_stack = H / k_n + t_w / k_s
  (``effective_conductivity``, macroscopic gradient e_x only);
* in-plane permeability, flow along the gap (walls carry no flow):
      K_t = (T / H) K_stack[y z block]        2x2, superficial over the gap
  (``permeability``, body force e_y and e_z);
* stiffness: the full 6x6 apparent core tensor from the laminate *mixed form*
  (``laminate_mixed_form``): with the layer normal along x, the tractions
  s_n = (s_xx, s_xz, s_xy) and the in-plane strains e_t = (e_yy, e_zz, g_yz) are
  continuous across layers, and the map [s_n; e_t] -> [e_n; s_t] is the volume
  average of the layers' maps, so the core map follows by subtracting the wall
  share. Reported:
    C_nn = C_xx,xx  normal stiffness across the gap (constrained modulus),
    G_t  = (C_55 + C_66)/2 transverse ("in-plane sliding") shear: the walls slide
           parallel to themselves (G_rz = C_55, G_rs = C_66) - the shear that
           carries torque and thrust from the sleeve to the outer wall;
    G_sz = C_44     shear in the plane of the gap (parallel to the walls),
  (``effective_elasticity``, all six unit strains).

The bulk reference (``bulk_properties``) evaluates the same quantities on the
periodic cell (no wall): k_xx, the y-z block of K, and C.

Discrepancy
-----------
For every quantity q:   delta_q = q_strip / q_hom - 1,   q_hom = homogenized prediction:
the bulk closure for uniform strips; for graded strips the *local* bulk closure
q_b(rho*(x)) combined across the gap like a laminate (series for k_n, mixed
form for C, arithmetic mean for K_t) - exactly what a local-closure device model
does. The error of the homogenized prediction is e_q = q_hom/q_strip - 1 =
-delta/(1 + delta). Corrected closure: q = q_bulk (1 + delta_q).

Discrepancy models (``PhysicalDiscrepancyModel``, ``GPDiscrepancyModel``)
--------------------------------------------------------------------------
Physical form: each wall adds a boundary layer whose effect does not depend on
how far away the other wall is, so the wall effect scales with the wall area per
lattice volume, 2/H = 2/(N L):

    series quantities (k_n, C_nn, G_t):   q_bulk / q = 1 + a(theta) / N  (+ gradient term)
    parallel quantities (K_t, G_sz):      q / q_bulk = 1 - a(theta) / N  (+ gradient term)

a = 2 x (excess layer thickness)/L at one wall: a series quantity behaves as if
each wall added a thin layer of extra resistance, a parallel quantity as if a
layer of thickness a L / 2 at each wall carried nothing. a(theta) is a small
linear regression in (w, rho*, a_z); the phase phi_0 is a nuisance (scatter
reported as a standard deviation). A Gaussian process on the residuals
(``GPDiscrepancyModel``) tests whether anything systematic is left. Result
(STATUS.md, Task 7): the 1/N form is sufficient - the GP never improves the
leave-one-group-out CV error; what is left is cut-phase scatter.

Units: k in W/(m K), C in Pa, K in units of L^2 (multiply by L^2 for m^2). The
material constants default to the reference config.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

from voxlat.utils.log import get_logger

__all__ = [
    "QUANTITIES",
    "PRIMARY_QUANTITIES",
    "QUANTITY_TYPE",
    "NORMAL_IDX",
    "INPLANE_IDX",
    "GapSpec",
    "Strip",
    "GapProperties",
    "strip_level_set",
    "threshold_curve",
    "build_strip",
    "bulk_cell",
    "laminate_mixed_form",
    "from_mixed_form",
    "laminate_average",
    "laminate_core",
    "compute_properties",
    "strip_properties",
    "bulk_properties",
    "BulkInterpolator",
    "homogenized_properties",
    "discrepancy",
    "homogenized_error",
    "brinkman_channel_factor",
    "morphology_label",
    "default_phases",
    "reduced_phase",
    "distinct_cuts",
    "PhysicalDiscrepancyModel",
    "GPDiscrepancyModel",
    "FiniteGapCorrection",
    "grouped_cv",
    "FEATURE_NAMES",
    "QUANTITY_LABEL",
]

LOG = get_logger("homogenization.finite_gap")

Kind = Literal["network", "sheet"]

#: Voigt indices of the jumping strains / continuous tractions for layers normal to x
#: (xx, xz, xy) and of the continuous in-plane strains (yy, zz, yz).
NORMAL_IDX: tuple[int, int, int] = (0, 4, 5)
INPLANE_IDX: tuple[int, int, int] = (1, 2, 3)

#: Scalar quantities reported per strip (``GapProperties.scalars``).
QUANTITIES: tuple[str, ...] = (
    "k_n", "K_t", "K_s", "K_z", "C_nn", "G_t", "G_rs", "G_rz", "G_sz", "C_tt",
)
#: The four quantities of the plan (through-gap conductivity, in-plane permeability,
#: transverse shear, normal stiffness) plus in-plane shear.
PRIMARY_QUANTITIES: tuple[str, ...] = ("k_n", "K_t", "C_nn", "G_t", "G_sz")
#: How the wall effect combines: "series" (across the gap) or "parallel" (along the gap).
QUANTITY_TYPE: dict[str, str] = {
    "k_n": "series", "C_nn": "series", "G_t": "series", "G_rs": "series", "G_rz": "series",
    "K_t": "parallel", "K_s": "parallel", "K_z": "parallel", "G_sz": "parallel", "C_tt": "parallel",
}
QUANTITY_LABEL: dict[str, str] = {
    "k_n": "through-gap conductivity $k_{rr}$",
    "K_t": "in-plane permeability $K_t$ (flow along the gap)",
    "K_s": "permeability $K_{ss}$",
    "K_z": "permeability $K_{zz}$",
    "C_nn": "normal stiffness $C_{rrrr}$",
    "G_t": "transverse shear $G_t=(G_{rs}+G_{rz})/2$",
    "G_rs": "transverse shear $G_{rs}$",
    "G_rz": "transverse shear $G_{rz}$",
    "G_sz": "in-plane shear $G_{sz}$",
    "C_tt": "in-plane normal stiffness",
}


# =============================================================================
# Specification and geometry
# =============================================================================
def morphology_label(w: float) -> str:
    """'G' (w = 0), 'D' (w = 1), else 'B0.50' style."""
    if w == 0.0:
        return "G"
    if w == 1.0:
        return "D"
    return f"B{w:.2f}"


def default_phases(k: int = 3) -> tuple[float, ...]:
    """Cut phases phi_0 = j / k, j = 0 ... k-1 (cell units).

    Blends (trigonal, no symmetry along x) need cuts over a full period: {0, 1/3, 2/3}.
    Gyroid and diamond network solids have 4_1 screw axes along <100> and 2-fold axes
    normal to x (I4_132, Fd-3m), so phi ~ phi + 1/4 ~ -phi and the distinct cuts lie
    in [0, 1/8] (``reduced_phase``): the default 1/3 and 2/3 both reduce to 1/12. The
    study therefore adds the cut 1/8 for G and D (``finite_gap_study.PURE_EXTRA_PHASES``).
    """
    return tuple(j / k for j in range(k))


def reduced_phase(w: Any, phase: Any) -> np.ndarray:
    """Symmetry-reduced cut phase (cell units).

    Gyroid and diamond (w in {0, 1}): the 4_1 screw axes along x make phi and
    phi + 1/4 equivalent (strip rotated by 90 deg about x), and the 2-fold axes
    normal to x make phi and -phi equivalent (strip mirrored, walls swapped), so the
    distinct cuts are phi_r = min(phi mod 1/4, 1/4 - phi mod 1/4) in [0, 1/8]
    (verified: the cuts 1/3 and 2/3 give identical uniform-strip properties).
    Blends have no such symmetry: phi_r = phi mod 1. Graded and stretched strips
    break the mirror / screw, so this applies to uniform, unstretched strips only.
    """
    w = np.asarray(w, dtype=float)
    p = np.asarray(phase, dtype=float)
    q = np.mod(p, 0.25)
    pure = (w == 0.0) | (w == 1.0)
    out = np.where(pure, np.minimum(q, 0.25 - q), np.mod(p, 1.0))
    return np.round(out, 9)


def distinct_cuts(df: Any) -> Any:
    """Rows with one representative per symmetry-distinct cut (uniform, a_z = 1 strips are
    de-duplicated with ``reduced_phase``; other rows are kept). Adds column ``phase_r``."""
    import pandas as pd

    df = pd.DataFrame(df).copy()
    w = df["w"].to_numpy(float)
    uni = (df.get("gradient", 0.0) == 0.0) & (df.get("a_z", 1.0) == 1.0)
    df["phase_r"] = np.where(uni, reduced_phase(w, df["phase"]), df["phase"])
    keys = [c for c in ("w", "rho", "N", "gradient", "a_z", "wall", "n", "kind", "phase_r") if c in df]
    return df.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)


@dataclass(frozen=True)
class GapSpec:
    """One finite-gap strip.

    Parameters
    ----------
    w, rho:
        Blend weight and relative density (mid-gap value for graded strips).
    N:
        Cells across the gap, H = N L (fractional allowed; realized as round(N n) voxels).
    phase:
        Where the inner wall cuts the lattice (cell units along x; see ``default_phases``).
    gradient:
        g = d rho*/d(x/L): change of rho* per cell across the gap (0 = uniform).
    a_z:
        Cell stretch along z (axial, in-plane).
    wall:
        Wall thickness t_w / L of the (single, periodic) wall layer.
    n:
        Voxels per cell length L.
    kind:
        "network" (default) or "sheet".
    """

    w: float
    rho: float
    N: float
    phase: float = 0.0
    gradient: float = 0.0
    a_z: float = 1.0
    wall: float = 0.5
    n: int = 32
    kind: Kind = "network"

    def __post_init__(self) -> None:
        for name in ("w", "rho", "N", "phase", "gradient", "a_z", "wall"):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "n", int(self.n))
        if not 0.0 <= self.w <= 1.0:
            raise ValueError(f"w must be in [0, 1], got {self.w}")
        if not 0.0 < self.rho < 1.0:
            raise ValueError(f"rho must be in (0, 1), got {self.rho}")
        if self.N <= 0:
            raise ValueError("N must be > 0")
        if self.n < 8:
            raise ValueError("n must be >= 8")
        if self.wall < 0:
            raise ValueError("wall must be >= 0")
        if self.a_z <= 0:
            raise ValueError("a_z must be > 0")
        if self.n_lattice < 1:
            raise ValueError("the gap must contain at least one voxel layer")
        lo, hi = self.rho_range
        if not (0.0 < lo and hi < 1.0):
            raise ValueError(f"graded density leaves (0, 1): {lo:.3f} ... {hi:.3f}")

    # ---- derived --------------------------------------------------------------
    @property
    def n_lattice(self) -> int:
        """Lattice voxel layers across the gap, round(N n)."""
        return int(round(self.N * self.n))

    @property
    def n_wall(self) -> int:
        """Wall voxel layers, round(wall n) (>= 1 if wall > 0)."""
        if self.wall == 0.0:
            return 0
        return max(1, int(round(self.wall * self.n)))

    @property
    def nz(self) -> int:
        """Voxels along z of one cell, round(n a_z) (as ``voxlat.geometry.grid_shape``)."""
        return max(4, int(round(self.n * self.a_z)))

    @property
    def H(self) -> float:
        """Realized gap width in units of L."""
        return self.n_lattice / self.n

    @property
    def layer_x(self) -> np.ndarray:
        """Voxel-layer centres across the gap, x / L (0 = inner wall)."""
        return (np.arange(self.n_lattice) + 0.5) / self.n

    @property
    def layer_rho(self) -> np.ndarray:
        """Target relative density of every lattice voxel layer."""
        return self.rho + self.gradient * (self.layer_x - 0.5 * self.H)

    @property
    def rho_range(self) -> tuple[float, float]:
        r = self.layer_rho
        return float(r.min()), float(r.max())

    @property
    def graded(self) -> bool:
        return self.gradient != 0.0

    @property
    def morphology(self) -> str:
        return morphology_label(self.w)

    @property
    def case_id(self) -> str:
        """Readable unique id, e.g. 'G-r0.350-N1.5-p0.333-g0.000-az1.000-tw0.500-n32'."""
        return (f"{self.morphology}-r{self.rho:.3f}-N{self.N:g}-p{self.phase:.4f}-g{self.gradient:.3f}"
                f"-az{self.a_z:.4f}-tw{self.wall:.3f}-n{self.n}" + ("" if self.kind == "network" else "-sheet"))

    def bulk_key(self, rho: float | None = None) -> tuple:
        """Key of the bulk reference cell on this strip's grid (rho defaults to the spec's)."""
        r = self.rho if rho is None else float(rho)
        return (self.w, round(r, 6), self.n, round(self.phase % 1.0, 6), self.a_z, self.kind)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cell_field(w: float, n: int, phase: float, a_z: float) -> np.ndarray:
    """Level set of one bulk cell on the strip grid: shape (n, n, nz), x shifted by phase."""
    from voxlat.geometry.tpms import TPMSParams, grid_shape, sample_level_set

    shape = grid_shape(TPMSParams(w=w, rho=0.5, a=(1.0, 1.0, a_z)), n)
    return sample_level_set(shape, w, offset=(phase, 0.0, 0.0))


def strip_level_set(spec: GapSpec) -> np.ndarray:
    """phi_w on the lattice voxels of the strip, shape (n_lattice, n, nz) (an x-tiling of the cell)."""
    cell = _cell_field(spec.w, spec.n, spec.phase, spec.a_z)
    reps = int(math.ceil(spec.n_lattice / spec.n))
    return np.tile(cell, (reps, 1, 1))[: spec.n_lattice]


@lru_cache(maxsize=64)
def _threshold_curve_cached(w: float, n: int, phase: float, a_z: float, kind: str,
                            rho_grid: tuple[float, ...]) -> np.ndarray:
    from voxlat.geometry.tpms import indicator_field, threshold_for_density

    g = indicator_field(_cell_field(w, n, phase, a_z), kind)  # type: ignore[arg-type]
    return np.array([threshold_for_density(w, r, g.shape, kind, g=g) for r in rho_grid])  # type: ignore[arg-type]


def threshold_curve(w: float, n: int, phase: float = 0.0, a_z: float = 1.0, kind: Kind = "network",
                    rho_grid: Sequence[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Bisection thresholds c(rho*) on the bulk grid of the strip (for graded strips).

    Returns (rho_grid, c); default grid rho* = 0.02 ... 0.98 in steps of 0.01. c is
    non-decreasing in rho*; ``np.interp`` gives a threshold for any density.
    """
    grid = tuple(np.round(np.arange(0.02, 0.9801, 0.01), 6)) if rho_grid is None else tuple(float(r) for r in rho_grid)
    c = _threshold_curve_cached(float(w), int(n), float(phase), float(a_z), str(kind), grid)
    return np.asarray(grid), c


@dataclass
class Strip:
    """Voxelized strip: ``solid`` (bool, (n_lattice + n_wall, n, nz)), True = solid.

    ``solid[:n_lattice]`` is the lattice (x = 0 at the inner wall), ``solid[n_lattice:]``
    the wall layer (periodic: it is also the inner wall). ``thresholds`` holds the
    level per lattice layer, ``layer_density`` the realized planar solid fraction.
    """

    spec: GapSpec
    solid: np.ndarray
    thresholds: np.ndarray

    @property
    def n_lattice(self) -> int:
        return self.spec.n_lattice

    @property
    def n_wall(self) -> int:
        return self.spec.n_wall

    @property
    def lattice(self) -> np.ndarray:
        return self.solid[: self.n_lattice]

    @property
    def layer_density(self) -> np.ndarray:
        return self.lattice.mean(axis=(1, 2))

    @property
    def lattice_density(self) -> float:
        return float(self.lattice.mean())

    @property
    def H(self) -> float:
        return self.n_lattice / self.spec.n

    @property
    def T(self) -> float:
        return self.solid.shape[0] / self.spec.n

    @property
    def t_wall(self) -> float:
        return self.n_wall / self.spec.n


def build_strip(spec: GapSpec) -> Strip:
    """Voxelize a strip (see the module docstring for the geometry)."""
    from voxlat.geometry.tpms import indicator_field, threshold_for_density

    phi = strip_level_set(spec)
    g = indicator_field(phi, spec.kind)
    if spec.graded:
        grid, c = threshold_curve(spec.w, spec.n, spec.phase, spec.a_z, spec.kind)
        lo, hi = spec.rho_range
        if lo < grid[0] or hi > grid[-1]:
            raise ValueError(f"graded density {lo:.3f}...{hi:.3f} outside the threshold table")
        thr = np.interp(spec.layer_rho, grid, c)
    else:
        gc = indicator_field(_cell_field(spec.w, spec.n, spec.phase, spec.a_z), spec.kind)
        c0 = threshold_for_density(spec.w, spec.rho, gc.shape, spec.kind, g=gc)
        thr = np.full(spec.n_lattice, c0)
    lattice = g <= thr[:, None, None]
    wall = np.ones((spec.n_wall, lattice.shape[1], lattice.shape[2]), dtype=bool)
    solid = np.concatenate([lattice, wall], axis=0)
    return Strip(spec, solid, thr)


def bulk_cell(w: float, rho: float, n: int, phase: float = 0.0, a_z: float = 1.0,
              kind: Kind = "network") -> np.ndarray:
    """The periodic bulk reference cell on a strip's grid (bit-identical to one strip period)."""
    from voxlat.geometry.tpms import indicator_field, threshold_for_density

    g = indicator_field(_cell_field(w, n, phase, a_z), kind)
    c = threshold_for_density(w, rho, g.shape, kind, g=g)
    return g <= c


# =============================================================================
# Laminate algebra (layers normal to x, Voigt with engineering shear)
# =============================================================================
def laminate_mixed_form(C: np.ndarray) -> np.ndarray:
    """Mixed (partially inverted) form of a layer stiffness for layers normal to x.

    With n = (xx, xz, xy) (continuous tractions, jumping strains) and
    t = (yy, zz, yz) (continuous strains, jumping stresses), Voigt blocks
    C_nn, C_nt, C_tn, C_tt:

        [e_n; s_t] = M [s_n; e_t],
        M = [[ C_nn^-1,           -C_nn^-1 C_nt          ],
             [ C_tn C_nn^-1,  C_tt - C_tn C_nn^-1 C_nt   ]].

    M is volume-averaged exactly over a laminate of homogeneous layers.
    Returned in the ordering (n, t) = (0, 4, 5, 1, 2, 3).
    """
    C = np.asarray(C, dtype=float)
    n, t = list(NORMAL_IDX), list(INPLANE_IDX)
    Cnn, Cnt, Ctn, Ctt = C[np.ix_(n, n)], C[np.ix_(n, t)], C[np.ix_(t, n)], C[np.ix_(t, t)]
    A = np.linalg.inv(Cnn)
    M = np.empty((6, 6))
    M[:3, :3] = A
    M[:3, 3:] = -A @ Cnt
    M[3:, :3] = Ctn @ A
    M[3:, 3:] = Ctt - Ctn @ A @ Cnt
    return M


def from_mixed_form(M: np.ndarray) -> np.ndarray:
    """Inverse of ``laminate_mixed_form``: Voigt stiffness from the mixed form."""
    M = np.asarray(M, dtype=float)
    A, B, Cc, D = M[:3, :3], M[:3, 3:], M[3:, :3], M[3:, 3:]
    Cnn = np.linalg.inv(A)
    Cnt = -Cnn @ B
    Ctn = Cc @ Cnn
    Ctt = D - Cc @ Cnn @ B
    n, t = list(NORMAL_IDX), list(INPLANE_IDX)
    C = np.empty((6, 6))
    C[np.ix_(n, n)] = Cnn
    C[np.ix_(n, t)] = Cnt
    C[np.ix_(t, n)] = Ctn
    C[np.ix_(t, t)] = Ctt
    return 0.5 * (C + C.T)


def laminate_average(Cs: Sequence[np.ndarray], fractions: Sequence[float]) -> np.ndarray:
    """Exact stiffness of a laminate of homogeneous layers normal to x (any anisotropy)."""
    f = np.asarray(fractions, dtype=float)
    f = f / f.sum()
    M = sum(fi * laminate_mixed_form(Ci) for fi, Ci in zip(f, Cs))
    return from_mixed_form(M)


def laminate_core(C_stack: np.ndarray, C_wall: np.ndarray, wall_fraction: float) -> np.ndarray:
    """Apparent core stiffness from a two-layer stack (core + wall, wall volume fraction f_w).

    M_core = (M_stack - f_w M_wall) / (1 - f_w), then back to Voigt. Exact inverse of
    ``laminate_average([C_core, C_wall], [1 - f_w, f_w])``.
    """
    fw = float(wall_fraction)
    if not 0.0 <= fw < 1.0:
        raise ValueError("wall_fraction must be in [0, 1)")
    if fw == 0.0:
        return np.asarray(C_stack, dtype=float).copy()
    M = (laminate_mixed_form(C_stack) - fw * laminate_mixed_form(C_wall)) / (1.0 - fw)
    return from_mixed_form(M)


# =============================================================================
# Properties of a strip / bulk cell
# =============================================================================
@dataclass
class GapProperties:
    """Apparent properties of the lattice layer (strip) or of the bulk cell.

    Attributes
    ----------
    k_n:
        Through-gap (x) conductivity of the lattice layer, W/(m K).
    K_t:
        2x2 in-plane permeability (y, z), superficial over the lattice gap, units L^2.
    C:
        6x6 Voigt stiffness of the lattice layer (mixed-form extraction), Pa.
    k_stack, K_stack, C_stack:
        Raw periodic results of the whole stack (lattice + wall); equal to the
        above for the bulk cell.
    H, T, t_wall:
        Lattice, stack and wall thickness in units of L.
    lattice_density, porosity:
        Realized solid fraction and porosity of the lattice region.
    k_s, E_s, nu_s:
        Material constants used (for dimensionless output).
    wall_times, iterations:
        Per solver ("k", "K", "C").
    """

    k_n: float
    K_t: np.ndarray
    C: np.ndarray
    k_stack: float
    K_stack: np.ndarray
    C_stack: np.ndarray
    H: float
    T: float
    t_wall: float
    lattice_density: float
    porosity: float
    k_s: float
    k_f: float
    E_s: float
    nu_s: float
    wall_times: dict[str, float] = field(default_factory=dict)
    iterations: dict[str, list[int]] = field(default_factory=dict)
    preconditioner: str = ""
    velocity: np.ndarray | None = field(default=None, repr=False)  # (2, 3, nx, ny, nz) if kept

    def scalars(self) -> dict[str, float]:
        """The reported scalar quantities (``QUANTITIES``), in SI / L^2 units."""
        C = self.C
        K = self.K_t
        return {
            "k_n": float(self.k_n),
            "K_t": float(np.trace(K) / 2.0) if K is not None else float("nan"),
            "K_s": float(K[0, 0]) if K is not None else float("nan"),
            "K_z": float(K[1, 1]) if K is not None else float("nan"),
            "C_nn": float(C[0, 0]) if C is not None else float("nan"),
            "G_t": float(0.5 * (C[4, 4] + C[5, 5])) if C is not None else float("nan"),
            "G_rs": float(C[5, 5]) if C is not None else float("nan"),
            "G_rz": float(C[4, 4]) if C is not None else float("nan"),
            "G_sz": float(C[3, 3]) if C is not None else float("nan"),
            "C_tt": float(0.5 * (C[1, 1] + C[2, 2])) if C is not None else float("nan"),
        }

    def dimensionless(self) -> dict[str, float]:
        """k / k_s, K / L^2, C / E_s."""
        s = self.scalars()
        out = {}
        for q, v in s.items():
            if q == "k_n":
                out[q] = v / self.k_s
            elif q.startswith("K"):
                out[q] = v
            else:
                out[q] = v / self.E_s
        return out

    def as_row(self, prefix: str) -> dict[str, Any]:
        """Flat table columns: dimensionless scalars, full tensors, timings."""
        row: dict[str, Any] = {f"{prefix}_{q}": v for q, v in self.dimensionless().items()}
        if self.K_t is not None:
            row[f"{prefix}_K_sz"] = float(self.K_t[0, 1])
        if self.C is not None:
            row[f"{prefix}_C_json"] = json.dumps((self.C / self.E_s).round(10).tolist())
        row[f"{prefix}_lattice_density"] = self.lattice_density
        for k, v in self.wall_times.items():
            row[f"{prefix}_time_{k}_s"] = v
        for k, v in self.iterations.items():
            row[f"{prefix}_iters_{k}"] = int(max(v)) if v else 0
        return row


def _material(k_s: float | None, k_f: float | None, E: float | None, nu: float | None,
              cfg: Any = None) -> tuple[float, float, float, float]:
    if None in (k_s, k_f, E, nu):
        if cfg is None:
            from voxlat.utils.config import load_config

            cfg = load_config()
        k_s = cfg.material.conductivity if k_s is None else k_s
        k_f = cfg.coolant.conductivity if k_f is None else k_f
        E = cfg.material.youngs_modulus if E is None else E
        nu = cfg.material.poisson_ratio if nu is None else nu
    return float(k_s), float(k_f), float(E), float(nu)  # type: ignore[arg-type]


def compute_properties(
    solid: np.ndarray,
    n: int,
    n_lattice: int,
    n_wall: int,
    *,
    k_s: float | None = None,
    k_f: float | None = None,
    E: float | None = None,
    nu: float | None = None,
    cfg: Any = None,
    quantities: Iterable[str] = ("k", "K", "C"),
    preconditioner: str = "two_level",
    tol: float = 1e-8,
    keep_velocity: bool = False,
) -> GapProperties:
    """Run the Task 2-4 solvers on a stack (lattice + wall layer along x) and remove the wall.

    ``solid``: bool (n_lattice + n_wall, ny, nz), lattice first. n_wall = 0 means a
    periodic bulk cell (no extraction). ``quantities``: any of "k" (conduction, e_x),
    "K" (Stokes, e_y and e_z), "C" (elasticity, six unit strains). Unrequested
    results are NaN. ``preconditioner``: "two_level" (default; no pyamg needed),
    "auto", "amg" or "jacobi" - passed to all three solvers.
    """
    from voxlat.homogenization.conduction import effective_conductivity
    from voxlat.homogenization.elasticity import effective_elasticity, isotropic_stiffness
    from voxlat.homogenization.stokes import permeability

    solid = np.asarray(solid, dtype=bool)
    if solid.shape[0] != n_lattice + n_wall:
        raise ValueError(f"solid has {solid.shape[0]} x-layers, expected {n_lattice} + {n_wall}")
    if n_wall > 0 and not solid[n_lattice:].all():
        raise ValueError("the wall layers must be fully solid")
    k_s, k_f, E, nu = _material(k_s, k_f, E, nu, cfg)
    quantities = set(quantities)
    nx = solid.shape[0]
    H, T, tw = n_lattice / n, nx / n, n_wall / n
    lat = solid[:n_lattice]
    times: dict[str, float] = {}
    iters: dict[str, list[int]] = {}
    nan22 = np.full((2, 2), np.nan)
    nan66 = np.full((6, 6), np.nan)

    k_stack = k_n = float("nan")
    if "k" in quantities:
        rc = effective_conductivity(solid, k_s, k_f, directions=(0,), preconditioner=preconditioner, tol=tol)  # type: ignore[arg-type]
        k_stack = float(rc.k_eff[0, 0])
        k_n = k_stack if n_wall == 0 else H / (T / k_stack - tw / k_s)
        times["k"] = rc.wall_time
        iters["k"] = list(rc.iterations)

    K_stack, K_t, vel = nan22.copy(), nan22.copy(), None
    if "K" in quantities:
        rk = permeability(solid, voxel_size=1.0 / n, directions=(1, 2), preconditioner=preconditioner,  # type: ignore[arg-type]
                          tol=tol, return_fields=keep_velocity)
        K_stack = rk.K[1:, 1:].copy()
        K_t = K_stack * (T / H) if n_wall > 0 else K_stack.copy()
        times["K"] = rk.wall_time
        iters["K"] = list(rk.iterations)
        vel = rk.velocity

    C_stack, C = nan66.copy(), nan66.copy()
    if "C" in quantities:
        re = effective_elasticity(solid, E, nu, localization=None, preconditioner=preconditioner,  # type: ignore[arg-type]
                                  tol=tol)
        C_stack = re.C_eff.copy()
        C = C_stack.copy() if n_wall == 0 else laminate_core(C_stack, isotropic_stiffness(E, nu), tw / T)
        times["C"] = re.wall_time
        iters["C"] = list(re.iterations)

    return GapProperties(
        k_n=float(k_n), K_t=K_t, C=C, k_stack=k_stack, K_stack=K_stack, C_stack=C_stack,
        H=H, T=T, t_wall=tw, lattice_density=float(lat.mean()), porosity=float(1.0 - lat.mean()),
        k_s=k_s, k_f=k_f, E_s=E, nu_s=nu, wall_times=times, iterations=iters,
        preconditioner=preconditioner, velocity=vel,
    )


def strip_properties(spec: GapSpec, **kwargs: Any) -> tuple[Strip, GapProperties]:
    """Build the strip of ``spec`` and compute its apparent lattice properties."""
    strip = build_strip(spec)
    return strip, compute_properties(strip.solid, spec.n, spec.n_lattice, spec.n_wall, **kwargs)


def bulk_properties(w: float, rho: float, n: int, phase: float = 0.0, a_z: float = 1.0,
                    kind: Kind = "network", **kwargs: Any) -> GapProperties:
    """Same quantities on the periodic bulk cell of the strip grid (the homogenized reference)."""
    cell = bulk_cell(w, rho, n, phase, a_z, kind)
    return compute_properties(cell, n, cell.shape[0], 0, **kwargs)


# =============================================================================
# Homogenized prediction and discrepancy
# =============================================================================
class BulkInterpolator:
    """Bulk closures as smooth functions of rho* (fixed w, a_z, grid, phase).

    Built from bulk results at several densities; interpolates log k, the matrix
    log of K_t and of the Mandel stiffness with monotone cubic (PCHIP) splines
    (Log-Euclidean interpolation keeps every tensor SPD). Used for the local-closure
    prediction of graded strips.
    """

    def __init__(self, rhos: Sequence[float], props: Sequence[GapProperties]):
        from scipy.interpolate import PchipInterpolator

        from voxlat.surrogates.tensors import sym_logm, voigt_to_mandel

        order = np.argsort(rhos)
        self.rho = np.asarray(rhos, dtype=float)[order]
        props = [props[i] for i in order]
        if len(self.rho) < 2:
            raise ValueError("need at least two densities")
        p0 = props[0]
        self.k_s, self.k_f, self.E_s, self.nu_s = p0.k_s, p0.k_f, p0.E_s, p0.nu_s
        logk = np.log([p.k_n for p in props])
        logK = np.stack([sym_logm(p.K_t) for p in props])
        logC = np.stack([sym_logm(voigt_to_mandel(p.C / p.E_s)) for p in props])
        self._k = PchipInterpolator(self.rho, logk)
        self._K = PchipInterpolator(self.rho, logK.reshape(len(props), -1), axis=0)
        self._C = PchipInterpolator(self.rho, logC.reshape(len(props), -1), axis=0)

    def __call__(self, rho: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(k (m,), K_t (m, 2, 2), C (m, 6, 6) Voigt, Pa) at densities ``rho``."""
        from voxlat.surrogates.tensors import mandel_to_voigt, sym_expm

        r = np.atleast_1d(np.asarray(rho, dtype=float))
        if r.min() < self.rho[0] - 1e-12 or r.max() > self.rho[-1] + 1e-12:
            raise ValueError(f"rho {r.min():.3f}...{r.max():.3f} outside {self.rho[0]:.3f}...{self.rho[-1]:.3f}")
        k = np.exp(self._k(r))
        X = self._K(r).reshape(-1, 2, 2)
        K = sym_expm(0.5 * (X + np.swapaxes(X, -1, -2)))
        Y = self._C(r).reshape(-1, 6, 6)
        C = mandel_to_voigt(sym_expm(0.5 * (Y + np.swapaxes(Y, -1, -2)))) * self.E_s
        return k, K, C


def homogenized_properties(
    spec: GapSpec,
    bulk: GapProperties | BulkInterpolator,
) -> GapProperties:
    """Homogenized prediction for a strip ("bulk closure + plain walls").

    Uniform strip: the bulk result itself (``bulk`` may be a ``GapProperties``).
    Graded strip: local closures at every voxel layer's target density,
    combined across the gap like a laminate: harmonic mean of k (series),
    arithmetic mean of K_t (parallel flow), laminate mixed-form average of C.
    """
    if not spec.graded:
        if isinstance(bulk, BulkInterpolator):
            k, K, C = bulk([spec.rho])
            return GapProperties(float(k[0]), K[0], C[0], float(k[0]), K[0], C[0], spec.H, spec.H, 0.0,
                                 spec.rho, 1 - spec.rho, bulk.k_s, bulk.k_f, bulk.E_s, bulk.nu_s)
        return bulk
    if not isinstance(bulk, BulkInterpolator):
        raise TypeError("a graded strip needs a BulkInterpolator")
    r = spec.layer_rho
    k, K, C = bulk(r)
    k_h = 1.0 / float(np.mean(1.0 / k))
    K_h = K.mean(axis=0)
    C_h = laminate_average(list(C), np.ones(len(r)))
    return GapProperties(k_h, K_h, C_h, k_h, K_h, C_h, spec.H, spec.H, 0.0, float(r.mean()),
                         float(1 - r.mean()), bulk.k_s, bulk.k_f, bulk.E_s, bulk.nu_s)


def discrepancy(strip: GapProperties, hom: GapProperties,
                quantities: Sequence[str] = QUANTITIES) -> dict[str, float]:
    """delta_q = q_strip / q_hom - 1 for every quantity."""
    s, h = strip.scalars(), hom.scalars()
    return {q: (s[q] / h[q] - 1.0) if h[q] not in (0.0,) and np.isfinite(h[q]) else float("nan")
            for q in quantities}


def homogenized_error(delta: Any) -> Any:
    """Relative error of the homogenized prediction, e = q_hom/q_strip - 1 = -delta/(1+delta)."""
    d = np.asarray(delta, dtype=float)
    return -d / (1.0 + d)


def brinkman_channel_factor(N: Any, K_over_L2: float, viscosity_ratio: float = 1.0) -> np.ndarray:
    """Brinkman reference for K_t/K_bulk in a channel of N cells with no-slip walls.

    Brinkman flow -mu_e u'' + mu u / K = f with u(0) = u(H) = 0 gives
    <u>/u_Darcy = 1 - (2 l / H) tanh(H / (2 l)),  l = sqrt(K mu_e / mu).
    Only a reference: it predicts a pure boundary-layer loss ~ 2 l / H.
    """
    N = np.asarray(N, dtype=float)
    ell = math.sqrt(K_over_L2 * viscosity_ratio)
    return 1.0 - (2.0 * ell / N) * np.tanh(N / (2.0 * ell))


# =============================================================================
# Discrepancy models
# =============================================================================
def _features(w: Any, rho: Any, a_z: Any) -> np.ndarray:
    """Regression basis of the wall coefficient a(theta).

    [1, r, w, b, r w, r b, s] with r = (rho* - 0.35)/0.1, b = 4 w (1 - w)
    (bump: 0 for pure G/D, 1 at w = 0.5), s = log(a_z).
    """
    w, rho, a_z = np.broadcast_arrays(np.asarray(w, dtype=float), np.asarray(rho, dtype=float),
                                      np.asarray(a_z, dtype=float))  # mixed scalars / arrays (Task 9)
    r = (rho - 0.35) / 0.1
    b = 4.0 * w * (1.0 - w)
    s = np.log(a_z)
    one = np.ones_like(w)
    return np.stack([one, r, w, b, r * w, r * b, s], axis=-1)


FEATURE_NAMES: tuple[str, ...] = ("1", "r", "w", "b", "r*w", "r*b", "log a_z")


def _wall_coefficient(qtype: str, delta: Any, N: Any) -> np.ndarray:
    """Observed a from delta: series a = N (1/(1+delta) - 1); parallel a = -N delta."""
    d = np.asarray(delta, dtype=float)
    N = np.asarray(N, dtype=float)
    if qtype == "series":
        return N * (1.0 / (1.0 + d) - 1.0)
    return -N * d


def _delta_from_a(qtype: str, a: Any, N: Any) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    N = np.asarray(N, dtype=float)
    if qtype == "series":
        return 1.0 / (1.0 + a / N) - 1.0
    return -a / N


def _get(df: Any, name: str, default: float) -> np.ndarray:
    if name in df:
        return np.asarray(df[name], dtype=float)
    return np.full(len(df), default)


@dataclass
class _QModel:
    qtype: str
    beta: list[float]  # coefficients of FEATURE_NAMES (wall coefficient a)
    active: list[bool]  # which features were fitted
    c_g: float  # gradient term: factor (1 + c_g g^2)
    a_scatter: float  # std of the phase scatter of a (pooled within groups)
    rmse_a: float  # rms residual of a around the fit (all samples)


class PhysicalDiscrepancyModel:
    """delta_q(N, theta, g) from the wall-layer form (module docstring).

    series:    q/q_hom = (1 + c_g g^2) / (1 + a(theta)/N)
    parallel:  q/q_hom = (1 + c_g g^2) (1 - a(theta)/N)

    a(theta) = beta . features(w, rho*, a_z) (``FEATURE_NAMES``), fitted by least
    squares to the observed a = N(1/(1+delta) - 1) (series) or -N delta (parallel)
    of the uniform strips, rows weighted by 1/N (``weighting="delta"``: the fit
    minimizes the error in delta ~ a/N, i.e. it favours the narrow gaps N = 1-3 of
    the jacket); features without variation in the data are dropped.
    c_g is then fitted on the graded strips (least squares on
    (1+delta_g)/(1+delta_phys(g=0)) - 1 = c_g g^2), by default on gyroid/diamond
    strips only (``gradient_rows="pure"``). The phase scatter is a std of a
    (``a_scatter``), i.e. delta scatter ~ a_scatter / N.
    """

    def __init__(self, quantities: Sequence[str] = PRIMARY_QUANTITIES, weighting: str = "delta",
                 gradient_rows: str = "pure"):
        if weighting not in ("delta", "a"):
            raise ValueError("weighting must be 'delta' or 'a'")
        if gradient_rows not in ("pure", "all"):
            raise ValueError("gradient_rows must be 'pure' or 'all'")
        self.quantities = tuple(quantities)
        self.weighting = weighting
        self.gradient_rows = gradient_rows
        self.models: dict[str, _QModel] = {}

    # ---- fitting ----------------------------------------------------------------
    def fit(self, df: Any, N_min: float = 0.0) -> "PhysicalDiscrepancyModel":
        """Fit on a table with columns w, rho, N, gradient, a_z, phase, delta_<q>."""
        g = _get(df, "gradient", 0.0)
        uni = g == 0.0
        N = _get(df, "N", np.nan)
        sel = uni & (N >= N_min)
        X = _features(_get(df, "w", 0.0), _get(df, "rho", 0.35), _get(df, "a_z", 1.0))
        for q in self.quantities:
            qtype = QUANTITY_TYPE[q]
            d = np.asarray(df[f"delta_{q}"], dtype=float)
            ok = sel & np.isfinite(d)
            a = _wall_coefficient(qtype, d[ok], N[ok])
            Xq = X[ok]
            active = [bool(np.ptp(Xq[:, j]) > 1e-12) or j == 0 for j in range(Xq.shape[1])]
            # drop linearly dependent columns (e.g. w and b when only pure G/D are present)
            cols = []
            for j in range(Xq.shape[1]):
                if not active[j]:
                    continue
                trial = cols + [j]
                if np.linalg.matrix_rank(Xq[:, trial]) == len(trial):
                    cols.append(j)
            active = [j in cols for j in range(Xq.shape[1])]
            beta = np.zeros(Xq.shape[1])
            if cols:
                # weighting "delta": minimize the error in delta ~ a/N (rows scaled by 1/N), which
                # emphasizes the small gaps that matter; "a": every N counts equally
                sw = 1.0 / N[ok] if self.weighting == "delta" else np.ones(int(ok.sum()))
                sol, *_ = np.linalg.lstsq(Xq[:, cols] * sw[:, None], a * sw, rcond=None)
                beta[cols] = sol
            res = a - Xq @ beta
            # phase scatter: pooled within-group std (groups = same theta, N, g)
            key = _group_keys(df, ok, with_N=True)
            a_sc = _pooled_std(res, key)
            m = _QModel(qtype, beta.tolist(), active, 0.0, a_sc, float(np.sqrt(np.mean(res**2))))
            # gradient term from graded strips (pure G/D only by default: graded blends cross the
            # pinch-neck transition, so their discrepancy is dominated by the density dependence of
            # the wall effect itself, not by the gradient)
            wv = _get(df, "w", 0.0)
            pure_rows = (wv == 0.0) | (wv == 1.0)
            okg = (~uni) & np.isfinite(np.asarray(df[f"delta_{q}"], dtype=float))
            if self.gradient_rows == "pure" and (okg & pure_rows).any():
                okg = okg & pure_rows
            if okg.any():
                d0 = _delta_from_a(qtype, X[okg] @ beta, N[okg])
                y = (1.0 + np.asarray(df[f"delta_{q}"], dtype=float)[okg]) / (1.0 + d0) - 1.0
                g2 = g[okg] ** 2
                m.c_g = float(np.sum(g2 * y) / np.sum(g2 * g2))
            self.models[q] = m
        return self

    # ---- prediction ---------------------------------------------------------------
    def wall_coefficient(self, q: str, w: Any, rho: Any, a_z: Any = 1.0) -> np.ndarray:
        m = self.models[q]
        return _features(w, rho, a_z) @ np.asarray(m.beta)

    def predict(self, q: str, N: Any, w: Any, rho: Any, a_z: Any = 1.0, gradient: Any = 0.0) -> np.ndarray:
        """Mean discrepancy delta_q (phase-averaged)."""
        m = self.models[q]
        a = self.wall_coefficient(q, w, rho, a_z)
        d = _delta_from_a(m.qtype, a, N)
        return (1.0 + d) * (1.0 + m.c_g * np.asarray(gradient, dtype=float) ** 2) - 1.0

    def predict_std(self, q: str, N: Any) -> np.ndarray:
        """Phase-scatter std of delta (first order: a_scatter / N)."""
        return self.models[q].a_scatter / np.asarray(N, dtype=float)

    def predict_df(self, df: Any) -> dict[str, np.ndarray]:
        return {q: self.predict(q, _get(df, "N", np.nan), _get(df, "w", 0.0), _get(df, "rho", 0.35),
                                _get(df, "a_z", 1.0), _get(df, "gradient", 0.0)) for q in self.quantities}

    # ---- io -----------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"features": list(FEATURE_NAMES), "weighting": self.weighting, "gradient_rows": self.gradient_rows,
                "models": {q: asdict(m) for q, m in self.models.items()}}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PhysicalDiscrepancyModel":
        obj = cls(tuple(d["models"]), weighting=d.get("weighting", "delta"),
                  gradient_rows=d.get("gradient_rows", "pure"))
        obj.models = {q: _QModel(**m) for q, m in d["models"].items()}
        return obj


def _group_keys(df: Any, mask: np.ndarray, with_N: bool = True) -> np.ndarray:
    cols = ["w", "rho", "a_z", "gradient", "wall", "n"] + (["N"] if with_N else [])
    parts = [np.round(_get(df, c, 0.0)[mask], 6).astype(str) for c in cols]
    return np.array(["|".join(t) for t in zip(*parts)]) if parts else np.zeros(int(mask.sum()), str)


def _pooled_std(res: np.ndarray, key: np.ndarray) -> float:
    ss, dof = 0.0, 0
    for k in np.unique(key):
        r = res[key == k]
        if r.size > 1:
            ss += float(np.sum((r - r.mean()) ** 2))
            dof += r.size - 1
    return float(np.sqrt(ss / dof)) if dof > 0 else 0.0


class GPDiscrepancyModel:
    """Gaussian process on the residuals of the physical model (one GP per quantity).

    Inputs u = (1/N, w, rho*, g, log a_z) scaled to unit range; target
    r = delta - delta_phys; kernel C * Matern-5/2 (ARD) + white noise (the phase
    scatter). Prediction delta = delta_phys + mean_GP.
    """

    INPUTS: tuple[str, ...] = ("1/N", "w", "rho", "gradient", "log a_z")

    def __init__(self, physical: PhysicalDiscrepancyModel, restarts: int = 2, seed: int = 2026):
        self.physical = physical
        self.restarts = restarts
        self.seed = seed
        self.gps: dict[str, Any] = {}
        self.lo: np.ndarray | None = None
        self.span: np.ndarray | None = None

    @staticmethod
    def _raw_inputs(df: Any) -> np.ndarray:
        return np.stack([1.0 / _get(df, "N", np.nan), _get(df, "w", 0.0), _get(df, "rho", 0.35),
                         _get(df, "gradient", 0.0), np.log(_get(df, "a_z", 1.0))], axis=1)

    def _scale(self, U: np.ndarray) -> np.ndarray:
        assert self.lo is not None and self.span is not None
        return (U - self.lo) / self.span

    def fit(self, df: Any) -> "GPDiscrepancyModel":
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

        U = self._raw_inputs(df)
        self.lo = U.min(axis=0)
        span = np.ptp(U, axis=0)
        self.span = np.where(span > 0, span, 1.0)
        Us = self._scale(U)
        phys = self.physical.predict_df(df)
        for q in self.physical.quantities:
            d = np.asarray(df[f"delta_{q}"], dtype=float)
            ok = np.isfinite(d)
            r = d[ok] - phys[q][ok]
            sd = float(np.std(r)) or 1e-6
            kern = (ConstantKernel(sd**2, (1e-8, 1e2)) *
                    Matern(length_scale=np.ones(U.shape[1]), length_scale_bounds=(1e-2, 1e3), nu=2.5)
                    + WhiteKernel((0.3 * sd) ** 2, (1e-10, 1e1)))
            gp = GaussianProcessRegressor(kern, normalize_y=False, n_restarts_optimizer=self.restarts,
                                          random_state=self.seed)
            with warnings.catch_warnings():
                # length scales at their bounds just mean "input irrelevant" / "noise only"
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(Us[ok], r)
            self.gps[q] = gp
        return self

    def predict(self, df: Any, return_std: bool = False) -> dict[str, Any]:
        Us = self._scale(self._raw_inputs(df))
        phys = self.physical.predict_df(df)
        out: dict[str, Any] = {}
        for q, gp in self.gps.items():
            if return_std:
                m, s = gp.predict(Us, return_std=True)
                out[q] = (phys[q] + m, s)
            else:
                out[q] = phys[q] + gp.predict(Us)
        return out


def grouped_cv(df: Any, quantities: Sequence[str] = PRIMARY_QUANTITIES, *, groups: str = "theta",
               use_gp: bool = True, restarts: int = 1,
               strata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Leave-one-group-out CV of the discrepancy models.

    ``groups="theta"``: hold out every (w, rho*, a_z) combination in turn (all N,
    phases, gradients); ``"N"``: hold out one N at a time (extrapolation in N).
    Returns per quantity the rms error (in delta) of: no correction ("none" =
    delta itself), the physical model, physical + GP; the pooled phase scatter of
    delta ("phase_floor") and the number of rows. With ``strata`` (name -> boolean
    row mask) the statistics are computed per stratum: {stratum: {quantity: row}}
    (the models are still trained on all training rows).
    """
    import pandas as pd

    df = pd.DataFrame(df).reset_index(drop=True)
    if groups == "theta":
        key = (df["w"].round(6).astype(str) + "|" + df["rho"].round(6).astype(str) + "|"
               + df.get("a_z", pd.Series(1.0, index=df.index)).round(6).astype(str))
    elif groups == "N":
        key = df["N"].round(6).astype(str)
    else:
        raise ValueError("groups must be 'theta' or 'N'")
    pred_p = {q: np.full(len(df), np.nan) for q in quantities}
    pred_g = {q: np.full(len(df), np.nan) for q in quantities}
    for k in key.unique():
        test = (key == k).to_numpy()
        train = df[~test]
        if len(train) < 5:
            continue
        pm = PhysicalDiscrepancyModel(quantities).fit(train)
        pp = pm.predict_df(df[test])
        for q in quantities:
            pred_p[q][test] = pp[q]
        if use_gp:
            gm = GPDiscrepancyModel(pm, restarts=restarts).fit(train)
            pg = gm.predict(df[test])
            for q in quantities:
                pred_g[q][test] = pg[q]
    gkey = (df["w"].round(6).astype(str) + "|" + df["rho"].round(6).astype(str) + "|" + df["N"].round(6).astype(str)
            + "|" + df.get("gradient", pd.Series(0.0, index=df.index)).round(6).astype(str)
            + "|" + df.get("a_z", pd.Series(1.0, index=df.index)).round(6).astype(str)).to_numpy()

    def _stats(mask: np.ndarray) -> dict[str, dict[str, float]]:
        res: dict[str, dict[str, float]] = {}
        for q in quantities:
            d = df[f"delta_{q}"].to_numpy(dtype=float)
            ok = mask & np.isfinite(d) & np.isfinite(pred_p[q])
            if not ok.any():
                continue
            row = {
                "none": float(np.sqrt(np.mean(d[ok] ** 2))),
                "physical": float(np.sqrt(np.mean((d[ok] - pred_p[q][ok]) ** 2))),
                "phase_floor": _pooled_std(d[ok], gkey[ok]),
                "n": int(ok.sum()),
            }
            if use_gp:
                okg = ok & np.isfinite(pred_g[q])
                row["physical+gp"] = float(np.sqrt(np.mean((d[okg] - pred_g[q][okg]) ** 2)))
            res[q] = row
        return res

    if strata is None:
        return _stats(np.ones(len(df), dtype=bool))
    return {name: _stats(np.asarray(m, dtype=bool)) for name, m in strata.items()}


# =============================================================================
# Correction API for the device model (Task 9)
# =============================================================================
class FiniteGapCorrection:
    """Finite-gap correction factors for bulk closures (load from ``models/finite_gap_correction.json``).

    >>> corr = FiniteGapCorrection.load()                                  # doctest: +SKIP
    >>> K_eff = K_bulk * corr.factor("K_t", N=6e-3 / L, w=0.0, rho=0.35)   # doctest: +SKIP

    ``factor(q, N, w, rho, a_z=1, gradient=0)`` = 1 + delta_q (phase-averaged);
    ``factor_std`` = phase-scatter std. Quantities: ``PRIMARY_QUANTITIES``.
    """

    DEFAULT_FILE = "finite_gap_correction.json"

    def __init__(self, physical: PhysicalDiscrepancyModel, meta: Mapping[str, Any] | None = None):
        self.physical = physical
        self.meta = dict(meta or {})

    def factor(self, q: str, N: Any, w: Any, rho: Any, a_z: Any = 1.0, gradient: Any = 0.0) -> np.ndarray:
        return 1.0 + self.physical.predict(q, N, w, rho, a_z, gradient)

    def factor_std(self, q: str, N: Any) -> np.ndarray:
        return self.physical.predict_std(q, N)

    def to_dict(self) -> dict[str, Any]:
        return {"meta": self.meta, "physical": self.physical.to_dict()}

    def save(self, path: str | Path | None = None) -> Path:
        from voxlat.utils.paths import models_dir

        p = Path(path) if path is not None else models_dir() / self.DEFAULT_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path | None = None) -> "FiniteGapCorrection":
        from voxlat.utils.paths import models_dir

        p = Path(path) if path is not None else models_dir() / cls.DEFAULT_FILE
        d = json.loads(Path(p).read_text())
        return cls(PhysicalDiscrepancyModel.from_dict(d["physical"]), d.get("meta"))
