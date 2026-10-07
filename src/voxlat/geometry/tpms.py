"""TPMS unit cells: gyroid / Schwarz-diamond level sets, blends, periodic voxelization.

Level sets
----------
On normalized cell coordinates xi in [0,1)^3 let X = 2 pi xi_x, Y = 2 pi xi_y,
Z = 2 pi xi_z. The classical nodal approximations are

    gyroid   G = sin X cos Y + sin Y cos Z + sin Z cos X                  (|G| <= 3/2)
    diamond  D = sin X sin Y sin Z + sin X cos Y cos Z
               + cos X sin Y cos Z + cos X cos Y sin Z                    (|D| <= sqrt 2)

(Schoen 1970 for the gyroid; Schwarz 1890 for D; nodal forms e.g. von Schnering &
Nesper 1991, Z. Phys. B 83, 407-412). With period 1 in xi, ``G = 0`` and ``D = 0``
approximate the exact minimal surfaces in their *conventional cubic cells*; their
areas per cell (3.0917 and 3.838 at fine resolution, see tests) match the exact
minimal-surface values A/a^2 = 3.0915 (G) and 3.8377 (D) to < 0.05 %.

Amplitude normalization. G and D have different ranges, so a plain blend would be
dominated by the gyroid. We divide each by its maximum,

    phi_G = G / (3/2),   phi_D = D / sqrt 2   ->   both span exactly [-1, 1],

and blend  phi_w = (1 - w) phi_G + w phi_D,  w in [0, 1]. Because G only has
Fourier modes of type (1,1,0) and D only of type (1,1,1), the two fields are
orthogonal; their RMS values are 0.577 (G) and 0.500 (D), i.e. comparable. Any
convex blend still satisfies |phi_w| <= 1, so thresholds always live in [-1, 1].
All three fields are odd (phi(-xi) = -phi(xi)), hence phi <= 0 fills exactly half
the cell: c(w, rho = 0.5) = 0 for every w.

Network vs sheet (why "network" is the default)
-----------------------------------------------
* ``network`` (skeletal) solid: solid where phi <= c. The solid is one thickened
  labyrinth (struts); the fluid, phi > c, is the *other* labyrinth plus the space
  between, which forms ONE connected domain. A single coolant enters at one
  manifold and must reach the other, so the jacket needs exactly that: one
  connected fluid, one connected solid. It also gives open pores for powder
  removal and lower pressure drop at a given density.
* ``sheet`` (shell) solid: solid where |phi| <= c, a thickened wall around the
  surface. It splits the fluid into TWO disjoint, interpenetrating labyrinths;
  that suits two-fluid heat exchangers, not our single-coolant jacket (the
  homogenized model would need two coupled fluid continua). Supported so the
  difference can be shown and tested (``periodic_components`` finds 2 fluid
  components).

Anisotropic cells (decision)
----------------------------
A cell is stretched by (a_x, a_y, a_z): physical size L*a_x x L*a_y x L*a_z
(the reference problem varies only a_z; L is the in-plane cell size).
We keep **cubic voxels** of edge h = L / n and give the stretched axis more or
fewer voxels: ``n_i = round(n * a_i)``. The stretch actually represented is the
snapped value ``a_i = n_i / n`` (``effective_stretch``), and the level set is
evaluated at xi_i = (k + 1/2) / n_i, so the grid is exactly periodic for any a.
Reasons: (1) every downstream solver (finite volumes, trilinear hex FEA, MAC
Stokes) stays isotropic and simple - one element stiffness, one face
conductance; (2) staircase error is the same in every direction; (3) density is
affine-invariant, so c(w, rho) needs no a-dependence. Cost: the snapping changes
a_z by at most 1/(2n) (0.5/48 ~ 1 %), and it is always reported.

Exact periodicity
-----------------
Sampling at voxel centres xi = (k + 1/2)/n_i with k = 0 ... n_i - 1 means voxel k
and voxel k + n_i of a tiled crystal are the same point of the periodic
function, so ``np.tile(voxelize(p, n), 2)`` is the voxelization of a 2x2x2
block. There is no duplicated face layer (the classic bug of linspace(0, 1, n)).

Threshold for a target density
------------------------------
``threshold_for_density`` bisects on c until the voxel solid fraction is within
half a voxel of the target, on the actual grid that will be used, so each voxel
cell has the requested density to O(1/n^3). (Bisection on a step function is the
same as taking the rho-quantile of the sampled field; it is written as bisection
because the C# exporter of Task 12 uses the same idea.)
``threshold_from_table`` instead interpolates a precomputed continuum table
c(w, rho) (sampled at n_ref = 128 voxels, i.e. converged to ~1e-4 in rho). It is
vectorized and is what graded designs (Tasks 9, 11, 12) use point-wise.

Units: everything in this module is dimensionless (cell units) unless a cell
size L in metres is passed; metric outputs then carry SI units (m, 1/m).
"""

from __future__ import annotations

import functools
import math
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from voxlat.geometry.voxel_tools import (
    PeriodicComponents,
    ThicknessStats,
    periodic_components,
    periodic_surface_area,
    periodic_surface_points,
    thickness_stats,
    voxel_face_area,
)

__all__ = [
    "TPMSParams",
    "GYROID_AMPLITUDE",
    "DIAMOND_AMPLITUDE",
    "GYROID_AREA_EXACT",
    "DIAMOND_AREA_EXACT",
    "gyroid",
    "diamond",
    "level_set",
    "sample_level_set",
    "indicator_field",
    "grid_shape",
    "effective_stretch",
    "threshold_for_density",
    "threshold_from_table",
    "ThresholdTable",
    "build_threshold_table",
    "load_threshold_table",
    "voxelize",
    "TPMSMetrics",
    "compute_metrics",
    "specific_surface_area",
    "ManufacturabilityReport",
    "check_manufacturable",
    "MorphologyTable",
    "build_morphology_table",
    "load_morphology_table",
    "min_cell_size",
    "FeasibleRegion",
    "feasible_region",
    "save_threshold_tables",
    "THRESHOLD_TABLE_FILE",
    "MORPHOLOGY_TABLE_FILE",
    "DATA_DIR",
]

Kind = Literal["network", "sheet"]

GYROID_AMPLITUDE = 1.5  # max |G|
DIAMOND_AMPLITUDE = math.sqrt(2.0)  # max |D|

# Exact minimal-surface areas per conventional cubic cell of edge 1 (A / a^2).
# Schroeder-Turk, Fogden & Hyde (2006), Eur. Phys. J. B 54, 509-524 (also Hyde et
# al., "The Language of Shape", 1997). Verify before citing in the manuscript.
GYROID_AREA_EXACT = 3.0915
DIAMOND_AREA_EXACT = 3.8377

DATA_DIR = Path(__file__).resolve().parent / "data"
THRESHOLD_TABLE_FILE = DATA_DIR / "tpms_threshold_table.npz"


# =============================================================================
# Parameters
# =============================================================================
@dataclass(frozen=True)
class TPMSParams:
    """Morphology of one TPMS unit cell (dimensionless; the cell size L is separate).

    Parameters
    ----------
    w:
        Blend weight, 0 = gyroid, 1 = Schwarz diamond.
    rho:
        Target relative density (solid volume fraction), 0 < rho < 1.
    a:
        Stretch factors (a_x, a_y, a_z); the cell is L*a_x x L*a_y x L*a_z.
    kind:
        "network" (default, solid where phi <= c) or "sheet" (solid where |phi| <= c).
    """

    w: float = 0.0
    rho: float = 0.3
    a: tuple[float, float, float] = (1.0, 1.0, 1.0)
    kind: Kind = "network"

    def __post_init__(self) -> None:
        object.__setattr__(self, "w", float(self.w))
        object.__setattr__(self, "rho", float(self.rho))
        object.__setattr__(self, "a", tuple(float(v) for v in self.a))
        if not 0.0 <= self.w <= 1.0:
            raise ValueError(f"w must be in [0, 1], got {self.w}")
        if not 0.0 < self.rho < 1.0:
            raise ValueError(f"rho must be in (0, 1), got {self.rho}")
        if len(self.a) != 3 or any(v <= 0 for v in self.a):
            raise ValueError(f"a must be three positive stretch factors, got {self.a}")
        if self.kind not in ("network", "sheet"):
            raise ValueError(f"kind must be 'network' or 'sheet', got {self.kind!r}")

    @classmethod
    def from_design(cls, w: float, rho: float, a_z: float = 1.0, kind: Kind = "network") -> "TPMSParams":
        """Parameters of the reference problem's design space (w, rho*, a_z)."""
        return cls(w=w, rho=rho, a=(1.0, 1.0, a_z), kind=kind)

    def with_(self, **changes: Any) -> "TPMSParams":
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================================
# Level sets
# =============================================================================
def gyroid(X: Any, Y: Any, Z: Any) -> np.ndarray:
    """Raw gyroid nodal function of angles X, Y, Z (radians); |G| <= 3/2."""
    return np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)


def diamond(X: Any, Y: Any, Z: Any) -> np.ndarray:
    """Raw Schwarz-D nodal function of angles X, Y, Z (radians); |D| <= sqrt 2."""
    sx, sy, sz = np.sin(X), np.sin(Y), np.sin(Z)
    cx, cy, cz = np.cos(X), np.cos(Y), np.cos(Z)
    return sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz


def level_set(xi_x: Any, xi_y: Any, xi_z: Any, w: float = 0.0) -> np.ndarray:
    """Normalized blended level set phi_w at normalized coordinates (period 1 in each).

    phi_w = (1 - w) * G / (3/2) + w * D / sqrt(2). Inputs broadcast like numpy
    arrays; works for points anywhere in space (periodic). Used by later tasks
    (finite-gap strips, exporter cross-checks) for arbitrary sample points.
    """
    X = 2.0 * np.pi * np.asarray(xi_x, dtype=float)
    Y = 2.0 * np.pi * np.asarray(xi_y, dtype=float)
    Z = 2.0 * np.pi * np.asarray(xi_z, dtype=float)
    out = 0.0
    if w < 1.0:
        out = out + (1.0 - w) / GYROID_AMPLITUDE * gyroid(X, Y, Z)
    if w > 0.0:
        out = out + w / DIAMOND_AMPLITUDE * diamond(X, Y, Z)
    return np.asarray(out, dtype=float)


def _centres(n: int) -> np.ndarray:
    """Voxel-centre angles 2 pi (k + 1/2) / n, k = 0 ... n-1 (exactly periodic)."""
    return 2.0 * np.pi * (np.arange(n) + 0.5) / n


def sample_level_set(shape: Sequence[int], w: float = 0.0) -> np.ndarray:
    """phi_w sampled at the voxel centres of one cell discretised as ``shape`` voxels.

    Separable evaluation: sin/cos are computed on 1-D axes and combined by
    broadcasting, so n = 128 (2.1 M voxels) takes ~0.1 s.
    """
    nx, ny, nz = (int(s) for s in shape)
    tx, ty, tz = _centres(nx), _centres(ny), _centres(nz)
    sx, cx = np.sin(tx)[:, None, None], np.cos(tx)[:, None, None]
    sy, cy = np.sin(ty)[None, :, None], np.cos(ty)[None, :, None]
    sz, cz = np.sin(tz)[None, None, :], np.cos(tz)[None, None, :]
    phi = np.zeros((nx, ny, nz))
    if w < 1.0:
        phi += ((1.0 - w) / GYROID_AMPLITUDE) * (sx * cy + sy * cz + sz * cx)
    if w > 0.0:
        phi += (w / DIAMOND_AMPLITUDE) * (sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz)
    return phi


def indicator_field(phi: np.ndarray, kind: Kind = "network") -> np.ndarray:
    """Field g whose sub-level set {g <= c} is the solid: phi (network) or |phi| (sheet)."""
    if kind == "network":
        return phi
    if kind == "sheet":
        return np.abs(phi)
    raise ValueError(f"unknown kind {kind!r}")


def grid_shape(params: TPMSParams | Sequence[float], n: int) -> tuple[int, int, int]:
    """Voxel grid of one cell: n_i = round(n * a_i) (cubic voxels of edge L / n).

    ``n`` is the number of voxels per reference cell length L. Accepts a
    TPMSParams or a plain stretch triple.
    """
    a = params.a if isinstance(params, TPMSParams) else tuple(params)
    if n < 4:
        raise ValueError("n must be >= 4")
    return tuple(max(4, int(round(n * ai))) for ai in a)  # type: ignore[return-value]


def effective_stretch(params: TPMSParams | Sequence[float], n: int) -> tuple[float, float, float]:
    """Stretch actually represented on the grid: a_i = n_i / n."""
    return tuple(s / n for s in grid_shape(params, n))  # type: ignore[return-value]


# =============================================================================
# Density thresholds
# =============================================================================
def _solid_fraction(g: np.ndarray, c: float) -> float:
    return float(np.count_nonzero(g <= c)) / g.size


def threshold_for_density(
    w: float,
    rho: float,
    n: int | Sequence[int] = 48,
    kind: Kind = "network",
    *,
    g: np.ndarray | None = None,
    max_iter: int = 200,
) -> float:
    """Level c such that the solid fraction of the n-voxel cell equals rho (bisection).

    The solid fraction f(c) = #(g <= c) / N is a non-decreasing step function of c.
    Bisection on the bracket [-1, 1] (network) or [0, 1] (sheet) stops when
    |f(c) - rho| <= 0.5 / N (the best a voxel grid can do) or when the bracket
    collapses; then the closer side is returned. Cubic symmetry makes many
    voxels share one field value, so the closest achievable fraction can be a
    few voxels away from rho (still ~1e-4 at n = 24). ``n`` may be an int (cubic cell) or a grid shape;
    ``g`` lets the caller pass a precomputed indicator field.
    """
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must be in (0, 1)")
    if g is None:
        shape = (n, n, n) if np.isscalar(n) else tuple(n)  # type: ignore[arg-type]
        g = indicator_field(sample_level_set(shape, w), kind)
    tol = 0.5 / g.size
    lo, hi = (-1.0 - 1e-12, 1.0 + 1e-12) if kind == "network" else (-1e-12, 1.0 + 1e-12)
    f_lo, f_hi = 0.0, 1.0  # f(lo) = 0 < rho < 1 = f(hi)
    for _ in range(max_iter):
        c = 0.5 * (lo + hi)
        if c == lo or c == hi:  # adjacent floats: the bracket holds exactly one step of f
            break
        f = _solid_fraction(g, c)
        if abs(f - rho) <= tol:
            return c
        if f < rho:
            lo, f_lo = c, f
        else:
            hi, f_hi = c, f
    # Symmetric cells contain many voxels with *identical* field values, so f can
    # jump by several voxels at one c; return whichever side is closer to rho.
    return lo if abs(f_lo - rho) <= abs(f_hi - rho) else hi


@dataclass(frozen=True)
class ThresholdTable:
    """Continuum threshold table c(w, rho) for one TPMS kind (bilinear interpolation)."""

    kind: str
    w: np.ndarray
    rho: np.ndarray
    c: np.ndarray  # shape (len(w), len(rho))
    n_ref: int
    meta: dict = field(default_factory=dict)

    def __call__(self, w: Any, rho: Any) -> np.ndarray | float:
        """c(w, rho), vectorized (broadcasting); raises outside the tabulated range."""
        w_arr, r_arr = np.broadcast_arrays(np.asarray(w, float), np.asarray(rho, float))
        if np.any((r_arr < self.rho[0] - 1e-12) | (r_arr > self.rho[-1] + 1e-12)):
            raise ValueError(f"rho outside table range [{self.rho[0]}, {self.rho[-1]}]")
        if np.any((w_arr < -1e-12) | (w_arr > 1 + 1e-12)):
            raise ValueError("w outside [0, 1]")
        from scipy.interpolate import RegularGridInterpolator

        f = RegularGridInterpolator((self.w, self.rho), self.c, method="linear")
        pts = np.stack([np.clip(w_arr, 0, 1).ravel(), np.clip(r_arr, self.rho[0], self.rho[-1]).ravel()], -1)
        out = f(pts).reshape(w_arr.shape)
        return float(out) if out.ndim == 0 else out


def build_threshold_table(
    kind: Kind = "network",
    *,
    n_ref: int = 128,
    w_grid: np.ndarray | None = None,
    rho_grid: np.ndarray | None = None,
) -> ThresholdTable:
    """Tabulate the continuum threshold c(w, rho).

    For each w the field is sampled at n_ref^3 voxel centres (midpoint rule, error
    O(n_ref^-2) in volume fraction) and c(rho) is read off as the rho-quantile
    of the indicator field - the converged limit of the bisection above.
    Default grid: w = 0, 0.05, ..., 1 (21) x rho = 0.01, 0.02, ..., 0.99 (99).
    """
    w_grid = np.round(np.linspace(0.0, 1.0, 21), 10) if w_grid is None else np.asarray(w_grid, float)
    rho_grid = np.round(np.arange(1, 100) / 100.0, 10) if rho_grid is None else np.asarray(rho_grid, float)
    c = np.empty((len(w_grid), len(rho_grid)))
    for i, wi in enumerate(w_grid):
        g = indicator_field(sample_level_set((n_ref,) * 3, float(wi)), kind).ravel()
        c[i] = np.quantile(g, rho_grid, method="linear")
    return ThresholdTable(kind=kind, w=w_grid, rho=rho_grid, c=c, n_ref=n_ref)


def save_threshold_tables(path: Path = THRESHOLD_TABLE_FILE, n_ref: int = 128) -> Path:
    """Build network + sheet tables and store them as one .npz (package data)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tn = build_threshold_table("network", n_ref=n_ref)
    ts = build_threshold_table("sheet", n_ref=n_ref)
    np.savez_compressed(
        path, w=tn.w, rho=tn.rho, c_network=tn.c, c_sheet=ts.c, n_ref=np.int64(n_ref),
        note=np.array("VoxLat Task 1: continuum TPMS thresholds c(w, rho); "
                      "phi_w=(1-w)G/1.5+w D/sqrt2; network solid phi<=c, sheet solid |phi|<=c"),
    )
    return path


@functools.lru_cache(maxsize=4)
def load_threshold_table(kind: Kind = "network") -> ThresholdTable:
    """Load the shipped table (``geometry/data/tpms_threshold_table.npz``).

    If the file is missing (e.g. a non-editable install without package data), the
    table is rebuilt in memory (a few seconds) with a warning.
    """
    if THRESHOLD_TABLE_FILE.is_file():
        with np.load(THRESHOLD_TABLE_FILE) as d:
            return ThresholdTable(kind=kind, w=d["w"], rho=d["rho"], c=d[f"c_{kind}"], n_ref=int(d["n_ref"]))
    warnings.warn(f"{THRESHOLD_TABLE_FILE.name} not found; rebuilding threshold table in memory", stacklevel=2)
    return build_threshold_table(kind)


def threshold_from_table(w: Any, rho: Any, kind: Kind = "network") -> np.ndarray | float:
    """Fast, vectorized continuum threshold c(w, rho) from the precomputed table."""
    return load_threshold_table(kind)(w, rho)


# =============================================================================
# Voxelization
# =============================================================================
def voxelize(
    params: TPMSParams,
    n: int = 48,
    *,
    threshold: float | None = None,
    method: Literal["exact", "table"] = "exact",
    return_field: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, float]:
    """Boolean voxel cell (True = solid), shape ``grid_shape(params, n)``, exactly periodic.

    Parameters
    ----------
    params:
        Morphology (w, rho, stretch, kind).
    n:
        Voxels per reference cell length L (voxel edge h = L / n).
    threshold:
        Use this level c instead of deriving it from rho.
    method:
        "exact" (default): bisection on this grid -> density within half a voxel.
        "table": continuum c(w, rho) from the table -> density within O(n^-2).
    return_field:
        Also return the sampled level set phi and the threshold c.
    """
    shape = grid_shape(params, n)
    phi = sample_level_set(shape, params.w)
    g = indicator_field(phi, params.kind)
    if threshold is not None:
        c = float(threshold)
    elif method == "exact":
        c = threshold_for_density(params.w, params.rho, shape, params.kind, g=g)
    elif method == "table":
        c = float(threshold_from_table(params.w, params.rho, params.kind))
    else:
        raise ValueError(f"unknown method {method!r}")
    solid = g <= c
    return (solid, phi, c) if return_field else solid


# =============================================================================
# Metrics
# =============================================================================
def specific_surface_area(
    phi: np.ndarray, c: float, kind: Kind = "network", *, n: int | None = None
) -> float:
    """Dimensionless specific surface area a_sf * L of a sampled periodic cell.

    a_sf = (solid-fluid interface area) / (cell volume). The interface is the
    level set phi = c (network) or phi = +-c (sheet: two walls), triangulated by
    periodic marching cubes. With cubic voxels of edge h = L / n:
    area = A_vox * h^2, volume = N_vox * h^3  ->  a_sf * L = n * A_vox / N_vox.
    ``n`` (voxels per L) defaults to phi.shape[0] (correct when a_x = 1).
    """
    n = phi.shape[0] if n is None else n
    if kind == "network":
        A = periodic_surface_area(phi, c)
    else:
        A = periodic_surface_area(phi, c) + periodic_surface_area(phi, -c)
    return n * A / phi.size


@dataclass(frozen=True)
class TPMSMetrics:
    """Geometric metrics of one voxelized TPMS cell.

    Dimensionless values are in units of the cell size L ("hat" quantities);
    ``*_m`` fields are in metres (or 1/m) for the given L. Thickness statistics
    are local-thickness statistics (see ``voxel_tools.local_thickness``).
    """

    params: dict
    n: int
    shape: tuple[int, int, int]
    a_effective: tuple[float, float, float]
    threshold: float
    relative_density: float
    a_sf_L: float  # specific surface area * L (marching cubes on the level set)
    a_sf_voxel_L: float  # staircase (voxel-face) area * L, diagnostic only
    wall: ThicknessStats  # solid local thickness / L
    pore: ThicknessStats  # fluid local thickness / L
    solid_components: int
    fluid_components: int
    solid_percolates: tuple[bool, bool, bool]
    fluid_percolates: tuple[bool, bool, bool]
    solid_largest_fraction: float
    fluid_largest_fraction: float
    cell_size_m: float | None = None

    @property
    def solid_connected(self) -> bool:
        """One solid component, infinite in x, y, z (no floating islands)."""
        return self.solid_components == 1 and all(self.solid_percolates)

    @property
    def fluid_connected(self) -> bool:
        """One fluid component, infinite in x, y, z (no trapped pockets, flow in every direction)."""
        return self.fluid_components == 1 and all(self.fluid_percolates)

    @property
    def voxel_size(self) -> float:
        """Voxel edge h / L."""
        return 1.0 / self.n

    def a_sf(self) -> float:
        """Specific surface area in 1/m (needs cell_size_m)."""
        return self.a_sf_L / self._L()

    def wall_m(self) -> ThicknessStats:
        return self.wall.scaled(self._L())

    def pore_m(self) -> ThicknessStats:
        return self.pore.scaled(self._L())

    def _L(self) -> float:
        if self.cell_size_m is None:
            raise ValueError("cell_size_m not set; pass L to compute_metrics")
        return self.cell_size_m

    def as_row(self) -> dict[str, Any]:
        """Flat dict (dimensionless + SI if L known) for data tables."""
        row: dict[str, Any] = {
            "w": self.params["w"], "rho_target": self.params["rho"],
            "a_x": self.a_effective[0], "a_y": self.a_effective[1], "a_z": self.a_effective[2],
            "kind": self.params["kind"], "n": self.n, "threshold": self.threshold,
            "rho": self.relative_density, "a_sf_L": self.a_sf_L, "a_sf_voxel_L": self.a_sf_voxel_L,
            "solid_components": self.solid_components, "fluid_components": self.fluid_components,
            "solid_connected": self.solid_connected, "fluid_connected": self.fluid_connected,
        }
        for name, st in (("wall", self.wall), ("pore", self.pore)):
            for k, v in asdict(st).items():
                row[f"{name}_{k}_L"] = v
        if self.cell_size_m is not None:
            L = self.cell_size_m
            row["L_m"] = L
            row["a_sf_per_m"] = self.a_sf_L / L
            for name, st in (("wall", self.wall), ("pore", self.pore)):
                row[f"{name}_min_m"] = st.min * L
                row[f"{name}_p5_m"] = st.p5 * L
        return row


def compute_metrics(
    params: TPMSParams,
    n: int = 48,
    L: float | None = None,
    *,
    thickness: bool = True,
    connectivity: bool = True,
) -> TPMSMetrics:
    """All Task-1 geometry metrics of one cell.

    * relative density - solid voxel fraction;
    * a_sf * L - periodic marching cubes on the sampled level set at the grid's
      own threshold (O(h^2) convergent, see tests);
    * wall / pore local-thickness statistics (min, p1, p5, mean, max) from a
      periodic Euclidean distance transform, in units of L;
    * periodic 6-connectivity of solid and fluid (components, percolation).

    ``L`` (metres) is optional; dimensionless results do not depend on it.
    Typical cost at n = 48: ~0.1 s without thickness, ~2 s with it.
    """
    solid, phi, c = voxelize(params, n, return_field=True)  # type: ignore[misc]
    shape = solid.shape
    a_sf_L = specific_surface_area(phi, c, params.kind, n=n)
    a_vox_L = n * voxel_face_area(solid) / solid.size
    if thickness:
        levels = [c] if params.kind == "network" else [c, -c]
        pts = periodic_surface_points(phi, levels)
        wall = thickness_stats(solid, spacing=1.0 / n, surface_points=pts)
        pore = thickness_stats(~solid, spacing=1.0 / n, surface_points=pts)
    else:
        nan = ThicknessStats(*(float("nan"),) * 6)
        wall = pore = nan
    if connectivity:
        cs: PeriodicComponents = periodic_components(solid)
        cf: PeriodicComponents = periodic_components(~solid)
        sc, fc = cs.n_components, cf.n_components
        sp = cs.percolates[0] if sc else (False, False, False)
        fp = cf.percolates[0] if fc else (False, False, False)
        sl = cs.volume_fractions[0] if sc else 0.0
        fl = cf.volume_fractions[0] if fc else 0.0
    else:
        sc = fc = -1
        sp = fp = (False, False, False)
        sl = fl = float("nan")
    return TPMSMetrics(
        params=params.to_dict(), n=n, shape=shape, a_effective=effective_stretch(params, n),
        threshold=c, relative_density=float(solid.mean()), a_sf_L=a_sf_L, a_sf_voxel_L=a_vox_L,
        wall=wall, pore=pore, solid_components=sc, fluid_components=fc,
        solid_percolates=sp, fluid_percolates=fp,
        solid_largest_fraction=sl, fluid_largest_fraction=fl, cell_size_m=L,
    )


# =============================================================================
# Manufacturability
# =============================================================================
MORPHOLOGY_TABLE_FILE = DATA_DIR / "tpms_morphology_table.npz"
MORPHOLOGY_FIELDS = (
    "rho", "a_sf_L", "wall_min", "wall_p1", "wall_p5", "wall_mean", "wall_throat",
    "pore_min", "pore_p1", "pore_p5", "pore_mean", "pore_throat",
    "solid_connected", "fluid_connected",
)
WallStatistic = Literal["min", "p1", "p5", "throat"]


def _stat(st: ThicknessStats, name: str) -> float:
    return float(getattr(st, name))


@dataclass(frozen=True)
class ManufacturabilityReport:
    """Result of ``check_manufacturable`` (lengths in metres).

    ``ok`` requires: wall >= min_wall_thickness, pore >= min_pore_size, one
    solid component infinite in x, y, z (no floating islands, continuous load
    path) and one fluid component infinite in x, y, z (no trapped powder or
    coolant pockets, flow possible in every direction).
    ``resolution_limited`` flags a wall or pore value of only ~4 voxels or less:
    usually a pinch point of the level set (a neck whose true thickness tends to
    zero), so the number shrinks with refinement - treat as "thinner than this".
    """

    ok: bool
    wall: float
    pore: float
    wall_statistic: str
    pore_statistic: str
    min_wall: float
    min_pore: float
    wall_ok: bool
    pore_ok: bool
    solid_connected: bool
    fluid_connected: bool
    in_design_bounds: bool
    resolution_limited: bool
    n: int
    reasons: tuple[str, ...] = ()


def check_manufacturable(
    params: TPMSParams,
    L: float,
    *,
    n: int = 48,
    cfg: Any = None,
    wall_statistic: WallStatistic = "min",
    pore_statistic: WallStatistic = "throat",
    metrics: TPMSMetrics | None = None,
) -> ManufacturabilityReport:
    """Check one cell of size L (m) against the manufacturing limits of the config.

    Default criteria (``configs/reference.yaml`` -> ``manufacturing``):

    * wall: the *minimum* local thickness of the solid (thinnest strut / neck
      anywhere) >= min_wall_thickness (0.35 mm). Strict on purpose: a neck that
      is too thin anywhere cannot be printed or fails in fatigue.
    * pore: the fluid *throat* - largest sphere that can pass through the pore
      network in x, y and z - >= min_pore_size (0.8 mm). That is the criterion
      for removing unfused powder and for not clogging; dead-end crevices are
      reported in ``metrics.pore.min`` but do not fail the check by default.
    * solid and fluid each one percolating component (periodic, 6-connected).

    ``in_design_bounds`` (rho, L, a_z inside ``design_bounds``) is reported but is
    not part of ``ok``. Pass ``metrics`` to reuse an earlier ``compute_metrics``.
    """
    if cfg is None:
        from voxlat.utils.config import load_config

        cfg = load_config()
    mf, b = cfg.manufacturing, cfg.design_bounds
    m = metrics if metrics is not None else compute_metrics(params, n, L)
    wall = _stat(m.wall, wall_statistic) * L
    pore = _stat(m.pore, pore_statistic) * L
    h = L / m.n
    wall_ok = wall >= mf.min_wall_thickness
    pore_ok = pore >= mf.min_pore_size
    in_bounds = bool(
        b.relative_density.contains(params.rho)
        and b.cell_size.contains(L)
        and b.blend_w.contains(params.w)
        and b.axial_stretch.contains(params.a[2])
    )
    reasons = []
    if not wall_ok:
        reasons.append(f"wall {wall_statistic} {wall * 1e3:.3f} mm < {mf.min_wall_thickness * 1e3:.3f} mm")
    if not pore_ok:
        reasons.append(f"pore {pore_statistic} {pore * 1e3:.3f} mm < {mf.min_pore_size * 1e3:.3f} mm")
    if not m.solid_connected:
        reasons.append("solid is not a single percolating component")
    if not m.fluid_connected:
        reasons.append("fluid is not a single percolating component")
    return ManufacturabilityReport(
        ok=bool(wall_ok and pore_ok and m.solid_connected and m.fluid_connected),
        wall=wall, pore=pore, wall_statistic=wall_statistic, pore_statistic=pore_statistic,
        min_wall=mf.min_wall_thickness, min_pore=mf.min_pore_size,
        wall_ok=bool(wall_ok), pore_ok=bool(pore_ok),
        solid_connected=m.solid_connected, fluid_connected=m.fluid_connected,
        in_design_bounds=in_bounds,
        resolution_limited=bool(min(wall, pore) <= 4.0 * h),
        n=m.n, reasons=tuple(reasons),
    )


# -----------------------------------------------------------------------------
# Morphology table: thickness statistics (in units of L) over (w, rho), a = 1
# -----------------------------------------------------------------------------
def morphology_row(w: float, rho: float, n: int = 48, kind: Kind = "network") -> dict[str, float]:
    """Dimensionless morphology of one (w, rho) cell for the table (lengths / L)."""
    m = compute_metrics(TPMSParams(w=w, rho=rho, kind=kind), n)
    row = {"rho": m.relative_density, "a_sf_L": m.a_sf_L,
           "solid_connected": float(m.solid_connected), "fluid_connected": float(m.fluid_connected)}
    for name, st in (("wall", m.wall), ("pore", m.pore)):
        for k in ("min", "p1", "p5", "mean", "throat"):
            row[f"{name}_{k}"] = _stat(st, k)
    return row


def build_morphology_table(
    *,
    n: int = 48,
    w_grid: Sequence[float] | None = None,
    rho_grid: Sequence[float] | None = None,
    n_jobs: int = -1,
    path: Path | None = MORPHOLOGY_TABLE_FILE,
    progress: bool = True,
) -> "MorphologyTable":
    """Compute the (w, rho) morphology table (network cells, a = 1) and save it.

    Default grid: w = 0, 0.125, ..., 1 (9) x rho = 0.15, 0.175, ..., 0.55 (17);
    ~3.5 s per point at n = 48 -> ~9 min on one core (joblib-parallel).
    """
    from joblib import Parallel, delayed

    w_grid = np.round(np.linspace(0, 1, 9), 10) if w_grid is None else np.asarray(w_grid, float)
    rho_grid = np.round(np.arange(0.15, 0.5501, 0.025), 10) if rho_grid is None else np.asarray(rho_grid, float)
    jobs = [(float(w), float(r)) for w in w_grid for r in rho_grid]
    it = jobs
    if progress:
        try:
            from tqdm import tqdm

            it = tqdm(jobs, desc="morphology table")
        except ImportError:  # pragma: no cover
            pass
    rows = Parallel(n_jobs=n_jobs)(delayed(morphology_row)(w, r, n) for w, r in it)
    data = {k: np.array([row[k] for row in rows]).reshape(len(w_grid), len(rho_grid)) for k in MORPHOLOGY_FIELDS}
    table = MorphologyTable(w=np.asarray(w_grid), rho=np.asarray(rho_grid), n=n, data=data)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, w=table.w, rho_target=table.rho, n=np.int64(n), **{f"f_{k}": v for k, v in data.items()},
            note=np.array("VoxLat Task 1: network TPMS morphology vs (w, rho), a=(1,1,1); "
                          "lengths in units of the cell size L; local thickness + throat"),
        )
    return table


@dataclass(frozen=True)
class MorphologyTable:
    """Tabulated dimensionless morphology over (w, rho) with bilinear interpolation."""

    w: np.ndarray
    rho: np.ndarray
    n: int
    data: dict

    def __call__(self, name: str, w: Any, rho: Any) -> np.ndarray | float:
        from scipy.interpolate import RegularGridInterpolator

        w_arr, r_arr = np.broadcast_arrays(np.asarray(w, float), np.asarray(rho, float))
        if np.any((r_arr < self.rho[0] - 1e-9) | (r_arr > self.rho[-1] + 1e-9)):
            raise ValueError(f"rho outside table range [{self.rho[0]}, {self.rho[-1]}]")
        f = RegularGridInterpolator((self.w, self.rho), self.data[name], method="linear")
        pts = np.stack([np.clip(w_arr, 0, 1).ravel(), np.clip(r_arr, self.rho[0], self.rho[-1]).ravel()], -1)
        out = f(pts).reshape(w_arr.shape)
        return float(out) if out.ndim == 0 else out


@functools.lru_cache(maxsize=1)
def load_morphology_table() -> MorphologyTable:
    """Load the shipped morphology table (build it with scripts/task1_build_tables.py)."""
    if not MORPHOLOGY_TABLE_FILE.is_file():
        raise FileNotFoundError(
            f"{MORPHOLOGY_TABLE_FILE} missing - run `python scripts/task1_build_tables.py` (~10 min)"
        )
    with np.load(MORPHOLOGY_TABLE_FILE) as d:
        data = {k[2:]: d[k] for k in d.files if k.startswith("f_")}
        return MorphologyTable(w=d["w"], rho=d["rho_target"], n=int(d["n"]), data=data)


def min_cell_size(
    w: Any,
    rho: Any,
    *,
    cfg: Any = None,
    wall_statistic: WallStatistic = "min",
    pore_statistic: WallStatistic = "throat",
    return_parts: bool = False,
) -> Any:
    """Smallest manufacturable cell size L_min(w, rho) in metres (vectorized; a = 1).

    Wall thickness and pore size both scale linearly with L at fixed (w, rho):
    t = t_hat(w, rho) L and p = p_hat(w, rho) L. Both limits are therefore
    *lower* bounds on L,

        L >= t_min / t_hat(w, rho)   (walls thicken with rho  -> binds at low rho)
        L >= p_min / p_hat(w, rho)   (pores shrink with rho   -> binds at high rho)

    and the feasible set is simply L >= L_min = max of the two. That is the
    optimisation constraint for Task 11: g = L_min(w, rho) / L - 1 <= 0.
    With ``return_parts`` returns (L_min, L_wall, L_pore).
    """
    if cfg is None:
        from voxlat.utils.config import load_config

        cfg = load_config()
    tab = load_morphology_table()
    t_hat = np.asarray(tab(f"wall_{wall_statistic}", w, rho))
    p_hat = np.asarray(tab(f"pore_{pore_statistic}", w, rho))
    with np.errstate(divide="ignore"):
        L_wall = np.where(t_hat > 0, cfg.manufacturing.min_wall_thickness / t_hat, np.inf)
        L_pore = np.where(p_hat > 0, cfg.manufacturing.min_pore_size / p_hat, np.inf)
    L_min = np.maximum(L_wall, L_pore)
    sq = (lambda x: float(x)) if L_min.ndim == 0 else (lambda x: x)
    return (sq(L_min), sq(L_wall), sq(L_pore)) if return_parts else sq(L_min)


@dataclass(frozen=True)
class FeasibleRegion:
    """Manufacturable (rho, L) region for one blend weight w (lengths in m)."""

    w: float
    rho: np.ndarray  # (k,)
    L_min: np.ndarray  # (k,) overall lower bound on L
    L_min_wall: np.ndarray
    L_min_pore: np.ndarray
    L: np.ndarray  # (m,) grid for the mask
    mask: np.ndarray  # (k, m) True = manufacturable AND inside design bounds
    rho_bounds: tuple[float, float]
    L_bounds: tuple[float, float]

    def feasible_fraction(self) -> float:
        """Fraction of the (rho, L) design box that is manufacturable."""
        return float(self.mask.mean())


def feasible_region(
    w: float,
    *,
    rho: Sequence[float] | None = None,
    L: Sequence[float] | None = None,
    cfg: Any = None,
    wall_statistic: WallStatistic = "min",
    pore_statistic: WallStatistic = "throat",
) -> FeasibleRegion:
    """Feasible (rho, L) region for blend weight w within the design bounds (a = 1).

    Returns the boundary L_min(rho) (and which limit sets it) plus a boolean mask
    on a (rho, L) grid. Defaults: 121 x 161 points spanning the design box.
    """
    if cfg is None:
        from voxlat.utils.config import load_config

        cfg = load_config()
    b = cfg.design_bounds
    rho = np.linspace(b.relative_density.lo, b.relative_density.hi, 121) if rho is None else np.asarray(rho, float)
    L = np.linspace(b.cell_size.lo, b.cell_size.hi, 161) if L is None else np.asarray(L, float)
    L_min, L_w, L_p = min_cell_size(np.full_like(rho, w), rho, cfg=cfg, wall_statistic=wall_statistic,
                                    pore_statistic=pore_statistic, return_parts=True)
    mask = (L[None, :] >= np.asarray(L_min)[:, None])
    mask &= b.relative_density.contains(rho)[:, None] & b.cell_size.contains(L)[None, :]
    return FeasibleRegion(
        w=float(w), rho=rho, L_min=np.asarray(L_min), L_min_wall=np.asarray(L_w), L_min_pore=np.asarray(L_p),
        L=L, mask=mask, rho_bounds=(b.relative_density.lo, b.relative_density.hi),
        L_bounds=(b.cell_size.lo, b.cell_size.hi),
    )
