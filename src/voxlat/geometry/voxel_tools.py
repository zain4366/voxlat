"""Generic tools for *periodic* voxel phases: surface area, local thickness, connectivity.

These functions know nothing about TPMS; they work on any periodic two-phase voxel
cell (``bool[nx, ny, nz]``, index order x, y, z) or on any periodic scalar field.
Tasks 2-4 and 7 reuse them (e.g. to check that the fluid percolates before running
a Stokes solve).

Why "periodic" matters
----------------------
A homogenization unit cell represents an infinite crystal. A strut that leaves
the cell through the face x = 1 re-enters through x = 0. Every measurement here
therefore treats the array as a 3-torus:

* surface area: the marching-cubes mesh is built on the field plus one wrapped
  layer, so cubes that straddle the cell boundary are counted exactly once;
* distance transform: the array is padded with ``mode="wrap"`` (copies of the
  neighbouring cells), so a voxel near a face "sees" material in the next cell;
* connectivity: components are labelled non-periodically, then glued across the
  six faces with a union-find that also tracks *which periodic image* a piece
  connects to. A component that connects to its own translate by a lattice
  vector v is infinite ("percolating") along v.

Conventions: distances are returned in voxel units unless a ``spacing`` (voxel
edge length, cubic voxels) is given; the caller converts to metres or mm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import ndimage

__all__ = [
    "periodic_surface_area",
    "voxel_face_area",
    "periodic_edt",
    "periodic_surface_points",
    "interface_distance",
    "local_thickness",
    "throat_diameter",
    "ThicknessStats",
    "thickness_stats",
    "PeriodicComponents",
    "periodic_components",
]


# =============================================================================
# Surface area
# =============================================================================
def periodic_surface_area(field_: np.ndarray, level: float, spacing: float = 1.0) -> float:
    """Area of the iso-surface ``field_ == level`` inside one periodic cell.

    The field is sampled at voxel centres of a periodic cell. One wrapped layer is
    appended on the high side of each axis (sample n == sample 0), giving n cubes
    per axis that tile exactly one period, so no face is lost or double-counted.
    Marching cubes (Lorensen & Cline 1987, as implemented in scikit-image) then
    triangulates the surface with linear interpolation along cube edges; the area
    error is O(h^2) for a smooth field (verified in the tests).

    Returns the area in units of ``spacing**2`` (0.0 if the level is not crossed).
    """
    from skimage.measure import marching_cubes, mesh_surface_area

    f = np.asarray(field_, dtype=float)
    if f.ndim != 3:
        raise ValueError("field must be 3-D")
    if not (f.min() < level < f.max()):
        return 0.0
    f = np.pad(f, ((0, 1), (0, 1), (0, 1)), mode="wrap")
    verts, faces, _n, _v = marching_cubes(f, level=level, spacing=(spacing,) * 3)
    return float(mesh_surface_area(verts, faces))


def voxel_face_area(phase: np.ndarray, spacing: float = 1.0) -> float:
    """Area of the staircase interface: number of solid/fluid voxel faces * h^2 (periodic).

    For a smooth, randomly oriented surface this over-estimates the true area by
    ~3/2 and does **not** converge with refinement (the staircase never gets
    flatter). It is reported only as a diagnostic: it is the interface the voxel
    solvers of Tasks 2-4 actually "see".
    """
    p = np.asarray(phase, dtype=bool)
    n_faces = sum(int(np.count_nonzero(p != np.roll(p, 1, axis=ax))) for ax in range(3))
    return n_faces * spacing**2


# =============================================================================
# Distance to the interface, local thickness, throat size
# =============================================================================
def _wrap_pad(a: np.ndarray, pad: int) -> np.ndarray:
    return np.pad(a, pad, mode="wrap")


def periodic_edt(phase: np.ndarray, *, pad: int | None = None) -> np.ndarray:
    """Periodic Euclidean distance transform of ``phase`` (voxel units).

    For every True voxel: distance from its centre to the nearest False voxel
    *centre*, taking periodic images into account (False voxels give 0).

    The cell is padded with wrapped copies (``pad`` voxels per side, adaptive by
    default) and the padding is grown until the largest distance found is safely
    smaller than the pad, so the result equals the infinite-crystal EDT.
    """
    p = np.asarray(phase, dtype=bool)
    if not p.any():
        return np.zeros(p.shape)
    if p.all():
        return np.full(p.shape, np.inf)
    nmax = max(p.shape)
    pad = max(4, nmax // 4) if pad is None else int(pad)
    while True:
        P = _wrap_pad(p, pad)
        d = ndimage.distance_transform_edt(P)
        core = d[pad:-pad, pad:-pad, pad:-pad]
        if core.max() + 1.0 < pad or pad >= 2 * nmax:
            return core
        pad = int(math.ceil(core.max())) + 4


def periodic_surface_points(field_: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    """Dense point cloud on the iso-surface(s) of a periodic field (voxel-index coordinates).

    Marching-cubes vertices (on cube edges, linearly interpolated -> O(h^2) accurate
    surface position) plus triangle centroids, wrapped into [0, n_i). Voxel centre k
    has coordinate k. Point spacing is <= ~0.7 voxel.
    """
    from skimage.measure import marching_cubes

    f = np.pad(np.asarray(field_, dtype=float), ((0, 1), (0, 1), (0, 1)), mode="wrap")
    shape = np.array(field_.shape, dtype=float)
    pts = []
    for lev in levels:
        if not (f.min() < lev < f.max()):
            continue
        v, faces, _n, _v = marching_cubes(f, level=lev)
        pts.append(v)
        pts.append(v[faces].mean(axis=1))
    if not pts:
        return np.empty((0, 3))
    P = np.mod(np.concatenate(pts), shape)
    P[P >= shape] = 0.0  # guard against round-off exactly at the period
    return P


def interface_distance(
    phase: np.ndarray, surface_points: np.ndarray | None = None
) -> np.ndarray:
    """Distance (voxel units) from each voxel centre of ``phase`` to the phase interface.

    * With ``surface_points`` (from ``periodic_surface_points`` of the level set
      that generated the voxels): exact Euclidean distance to the sub-voxel
      interface, periodic (``scipy.spatial.cKDTree`` with ``boxsize``). This is
      the accurate option - a voxel-only transform cannot tell whether the true
      surface is 0.05 or 0.85 voxel beyond the last voxel centre.
    * Without: periodic EDT minus 1/2 (the interface is assumed half-way between
      the last phase voxel centre and the first other-phase centre). Error up to
      about +-0.5 voxel; use for voxel arrays that have no level set behind them.

    Returns 0 outside the phase.
    """
    p = np.asarray(phase, dtype=bool)
    out = np.zeros(p.shape)
    if not p.any():
        return out
    if surface_points is None or len(surface_points) == 0:
        d = periodic_edt(p)
        out[p] = np.maximum(d[p] - 0.5, 0.0)
        return out
    from scipy.spatial import cKDTree

    tree = cKDTree(surface_points, boxsize=np.array(p.shape, dtype=float))
    idx = np.argwhere(p).astype(float)
    dist, _ = tree.query(idx, k=1, workers=-1)
    out[p] = dist
    return out


def local_thickness(
    phase: np.ndarray,
    *,
    distance: np.ndarray | None = None,
    step: float = 0.25,
    slack: float = 0.5,
) -> np.ndarray:
    """Periodic local thickness (Hildebrand & Rueegsegger 1997), in voxel units.

    Definition: the local thickness LT(p) of a point p of a phase is the diameter
    of the largest ball that (i) lies entirely inside the phase and (ii) contains p.
    This is the standard "wall thickness" / "pore size" measure of micro-CT
    analysis (BoneJ / ImageJ "Local Thickness"); unlike the raw distance
    transform, it is not fooled by points that merely lie close to the surface.

    Algorithm (granulometry by openings):

    1. D(q) = distance from voxel centre q to the interface (``interface_distance``;
       sub-voxel accurate when a level set is available). D(q) is the radius of the
       largest ball centred at q that stays inside the phase.
    2. For trial radii r = r_0, r_0 + step, ...: admissible centres E_r = {q : D(q) >= r};
       the union of the balls of radius r around E_r (the morphological opening by
       a ball of radius r) is O_r = {p : dist(p, E_r) <= r}, obtained with one
       Euclidean distance transform of the complement of E_r on a wrap-padded
       array (pad = r + 2, so balls from neighbouring cells are included).
    3. LT(p) = 2 * max{r : p in O_r}. Openings are nested, so we overwrite upward.

    Discretisation: radii are quantised to ``step`` (default 0.25 voxel) and ball
    centres sit on voxel centres, so a ball whose ideal centre lies between voxels
    is under-sized by up to ~0.5 voxel. Measured on analytic shapes (tests): slabs
    and cylinders within about -1 ... +0.5 voxel of the true diameter. Keep >= 8
    voxels across the thinnest feature and report the resolution with the number.

    Returns an array of the phase's shape: LT (voxels) on phase voxels, 0 elsewhere.
    """
    p = np.asarray(phase, dtype=bool)
    lt = np.zeros(p.shape)
    if not p.any():
        return lt
    if p.all():
        lt[:] = np.inf
        return lt
    D = interface_distance(p) if distance is None else np.where(p, distance, 0.0)
    Rad = np.where(p, D + slack, 0.0)  # ball radius attributed to each centre
    r_pos = Rad[p]
    r0 = max(step, float(np.floor(r_pos.min() / step) * step))
    r_max = float(r_pos.max())
    lt[p] = 2.0 * np.minimum(r_pos, r0)  # floor: every voxel lies in its own ball
    for r in np.arange(r0, r_max + 1e-9, step):
        pad = int(math.ceil(r)) + 2
        centres = _wrap_pad(Rad >= r - 1e-12, pad)
        if not centres.any():
            break
        dist_c = ndimage.distance_transform_edt(~centres)[pad:-pad, pad:-pad, pad:-pad]
        covered = (dist_c <= r + 1e-9) & p
        lt[covered] = 2.0 * r
    return lt


def throat_diameter(
    phase: np.ndarray,
    *,
    distance: np.ndarray | None = None,
    tol: float = 0.05,
    slack: float = 0.5,
) -> float:
    """Diameter (voxels) of the largest ball that can travel through the phase in x, y and z.

    "Throat" or bottleneck size: the largest r such that the admissible centres
    {q : D(q) >= r} still contain a component that is infinite in all three
    directions (``periodic_components``). It is the passable-sphere size of
    mercury-intrusion porosimetry. For the pore space it is the largest powder
    particle / bubble that can be flushed through the lattice; for the solid it
    is the neck diameter of the load path. For gyroid and diamond, where all
    necks are symmetry-equivalent, it equals the thinnest neck; for blends it is
    the bottleneck of the best-connected path (>= the thinnest strut).

    Centres move between 26-neighbours (a ball can slide diagonally; with
    6-neighbours a diagonal strut would be traversed by a staircase of off-axis
    centres and the throat under-estimated). As in ``local_thickness``, the
    off-grid position of the ideal centre is compensated by ``slack`` (0.5 voxel,
    calibrated on rod networks in the tests): result = 2 (r* + slack).
    Bisection on r to ``tol`` voxel. Returns 0 if the phase does not percolate.
    """
    p = np.asarray(phase, dtype=bool)
    if not p.any():
        return 0.0
    D = interface_distance(p) if distance is None else np.where(p, distance, 0.0)

    def ok(r: float) -> bool:
        c = periodic_components(p & (D >= r), connectivity=26)
        return any(all(pc) for pc in c.percolates)

    lo, hi = 0.0, float(D.max()) + 1e-9
    if not ok(1e-9):
        return 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return 2.0 * (lo + slack)


@dataclass(frozen=True)
class ThicknessStats:
    """Thickness statistics of one phase (units of the caller).

    ``min``, ``p1``, ``p5``, ``mean``, ``max``: local thickness, volume-weighted
    over the phase voxels. ``throat``: largest ball passable in x, y and z.
    """

    min: float
    p1: float
    p5: float
    mean: float
    max: float
    throat: float = float("nan")

    def scaled(self, factor: float) -> "ThicknessStats":
        return ThicknessStats(
            *(v * factor for v in (self.min, self.p1, self.p5, self.mean, self.max, self.throat))
        )


def thickness_stats(
    phase: np.ndarray,
    *,
    spacing: float = 1.0,
    surface_points: np.ndarray | None = None,
    step: float = 0.25,
    throat: bool = True,
) -> ThicknessStats:
    """Local-thickness and throat statistics of a periodic phase.

    ``spacing`` is the voxel edge length (results in the same unit);
    ``surface_points`` enables sub-voxel distances (strongly recommended).
    """
    p = np.asarray(phase, dtype=bool)
    if not p.any():
        return ThicknessStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    D = interface_distance(p, surface_points)
    lt = local_thickness(p, distance=D, step=step)[p]
    q1, q5 = np.percentile(lt, [1.0, 5.0])
    th = throat_diameter(p, distance=D) if throat else float("nan")
    return ThicknessStats(
        float(lt.min()), float(q1), float(q5), float(lt.mean()), float(lt.max()), th
    ).scaled(spacing)


# =============================================================================
# Periodic connectivity
# =============================================================================
@dataclass(frozen=True)
class PeriodicComponents:
    """Connected components of a phase on the 3-torus.

    Attributes
    ----------
    n_components:
        Number of distinct components of the infinite periodic phase (per cell).
    volume_fractions:
        Fraction of the *phase* volume in each component, largest first.
    percolates:
        For each component (same order), booleans (x, y, z): True if the component
        is infinite along that axis, i.e. it is connected to one of its own periodic
        images by a translation with a non-zero component along the axis.
    wrap_rank:
        For each component, rank (0-3) of the lattice of translations under which
        it is connected to itself: 0 = finite island, 1 = infinite "rods",
        2 = infinite "sheets", 3 = fully 3-D connected network.
    connectivity:
        6 (face neighbours, default) - the convention of the finite-volume / FE
        solvers, which exchange flux only through shared faces - or 26.
    """

    n_components: int
    volume_fractions: tuple[float, ...]
    percolates: tuple[tuple[bool, bool, bool], ...]
    wrap_rank: tuple[int, ...]
    labels: np.ndarray = field(repr=False, compare=False)
    connectivity: int = 6

    @property
    def single_percolating(self) -> bool:
        """Exactly one component, and it percolates in x, y and z."""
        return self.n_components == 1 and all(self.percolates[0])

    @property
    def largest_percolates_all(self) -> bool:
        return self.n_components > 0 and all(self.percolates[0])


class _OffsetUnionFind:
    """Union-find over components with integer lattice offsets ("potentials").

    Node (x, cell z) is connected to (root(x), cell z + pot(x)). When a periodic
    link closes a loop with a non-zero net translation v, the cluster connects to
    its own image shifted by v; v is stored as a wrap vector of the root.
    """

    def __init__(self, n: int):
        self.parent = np.arange(n)
        self.pot = np.zeros((n, 3), dtype=np.int64)
        self.wraps: dict[int, list[np.ndarray]] = {}

    def find(self, x: int) -> tuple[int, np.ndarray]:
        path = []
        while self.parent[x] != x:
            path.append(x)
            x = self.parent[x]
        root = x
        # path compression, accumulating potentials from the root downwards
        acc = np.zeros(3, dtype=np.int64)
        for node in reversed(path):
            acc = acc + self.pot[node]
            self.pot[node] = acc.copy()
            self.parent[node] = root
        return root, (self.pot[path[0]].copy() if path else np.zeros(3, dtype=np.int64))

    def link(self, a: int, b: int, e: np.ndarray) -> None:
        """Record that (a, cell 0) touches (b, cell e)."""
        ra, pa = self.find(a)
        rb, pb = self.find(b)
        if ra == rb:
            v = e + pb - pa
            if np.any(v != 0):
                self.wraps.setdefault(ra, []).append(v)
            return
        # attach rb below ra: (rb, z) ~ (ra, z + q)
        q = pa - e - pb
        self.parent[rb] = ra
        self.pot[rb] = q
        if rb in self.wraps:
            self.wraps.setdefault(ra, []).extend(self.wraps.pop(rb))


def _half_neighbour_offsets(connectivity: int) -> list[tuple[int, int, int]]:
    """One offset of each +-pair of neighbour offsets (3 for 6-conn, 13 for 26-conn)."""
    offs = []
    for o in np.ndindex(3, 3, 3):
        o = tuple(int(v) - 1 for v in o)
        if o == (0, 0, 0) or o <= (0, 0, 0):  # keep the lexicographically positive half
            continue
        if connectivity == 6 and sum(abs(v) for v in o) != 1:
            continue
        offs.append(o)
    return offs


def periodic_components(phase: np.ndarray, connectivity: int = 6) -> PeriodicComponents:
    """Label the components of a periodic phase and test percolation.

    Steps: (1) ``scipy.ndimage.label`` inside the cell with the chosen
    connectivity (6 = faces, 26 = faces + edges + corners); (2) for every
    neighbour offset o, each pair of phase voxels x and x + o that crosses the
    periodic boundary links label(x) in cell 0 with label(x + o mod n) in cell
    e = floor((x + o) / n); (3) the offset union-find merges labels and collects
    the net translations of closed loops (wrap vectors). The rank of the wrap
    vectors tells whether a component is an island (0), rods (1), sheets (2) or
    a 3-D network (3).

    Use 6 for transport (finite volumes / FE exchange flux only through faces);
    26 for the motion of a ball centre (``throat_diameter``).
    """
    if connectivity not in (6, 26):
        raise ValueError("connectivity must be 6 or 26")
    p = np.asarray(phase, dtype=bool)
    structure = ndimage.generate_binary_structure(3, 1 if connectivity == 6 else 3)
    labels, k = ndimage.label(p, structure=structure)
    if k == 0:
        return PeriodicComponents(0, (), (), (), labels, connectivity)
    shape = np.array(p.shape)
    uf = _OffsetUnionFind(k + 1)  # index 0 = background, unused
    idx = np.indices(p.shape)
    for o in _half_neighbour_offsets(connectivity):
        tgt = idx + np.array(o).reshape(3, 1, 1, 1)
        e = np.floor_divide(tgt, shape.reshape(3, 1, 1, 1))
        crossing = np.any(e != 0, axis=0) & (labels > 0)
        if not crossing.any():
            continue
        t = np.mod(tgt[:, crossing], shape.reshape(3, 1))
        lb = labels[t[0], t[1], t[2]]
        la = labels[crossing]
        ee = e[:, crossing].T
        m = lb > 0
        if not m.any():
            continue
        trip = np.unique(np.column_stack([la[m], lb[m], ee[m]]), axis=0)
        for a, b, ex, ey, ez in trip:
            uf.link(int(a), int(b), np.array([ex, ey, ez], dtype=np.int64))
    roots = np.array([uf.find(i)[0] for i in range(k + 1)])
    roots[0] = 0
    merged = roots[labels]
    counts = np.bincount(merged.ravel(), minlength=k + 1)
    counts[0] = 0
    order = [int(r) for r in np.argsort(-counts, kind="stable") if counts[r] > 0]
    total = counts.sum()
    vols, perc, ranks = [], [], []
    relabel = np.zeros(k + 1, dtype=np.int32)
    for new, r in enumerate(order, start=1):
        relabel[r] = new
        vols.append(float(counts[r] / total))
        W = np.unique(np.array(uf.wraps.get(r, []), dtype=np.int64).reshape(-1, 3), axis=0)
        rank = int(np.linalg.matrix_rank(W.astype(float))) if len(W) else 0
        ranks.append(rank)
        perc.append(tuple(bool(np.any(W[:, i] != 0)) for i in range(3)) if len(W) else (False, False, False))
    return PeriodicComponents(
        n_components=len(order),
        volume_fractions=tuple(vols),
        percolates=tuple(perc),
        wrap_rank=tuple(ranks),
        labels=relabel[merged],
        connectivity=connectivity,
    )
