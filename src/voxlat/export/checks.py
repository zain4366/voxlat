"""Manufacturability checks on a voxelized jacket (sector or full) - Task 12.

Input: a ``VoxelMask`` (PicoGK slices written by the C# exporter, the numpy generator, or a
re-voxelized STL) and the ``JacketField`` of the same design (for the analytic regions: lattice gap,
plenums, port bores, open lattice faces, sector wedge). All distances in mm.

Checks (criteria of Task 1, applied to the real geometry instead of a periodic cell):

1. **minimum wall** = minimum local thickness of the lattice solid (largest inscribed ball containing
   the point; Hildebrand & Rueegsegger 1997; the criterion of Task 1). p has local thickness >= d iff p
   lies in the opening of the solid by a ball of diameter d, so the test at d is "solid minus opening
   is empty" (``thin_mask``, discretization in its docstring); pass/fail at min_wall, the value by
   bisection. Evaluated in the lattice region of the *nominal* sector, farther than
   ``band`` = min_wall + 2h from an open lattice face (slot sides, ring faces): clipping a lattice with
   a plane necessarily leaves sub-wall slivers there; they are reported separately
   (``interface_slivers``) and removed by the exporter's sliver opening. Housing walls (sleeve, outer
   wall, caps, bosses) are thick by construction and are not evaluated (their sharp convex edges have
   vanishing local thickness by definition).
2. **minimum pore** = the throat between inlet and outlet: the largest sphere that can travel through
   the fluid from the inlet terminal to the outlet terminal (Task 1 "throat"; powder removal /
   clogging criterion). Sphere centres are fluid voxels with EDT-to-solid >= d/2 - h/4, connected
   26-wise (as Task 1: face connectivity under-reads the diagonal diamond channels); bisection on d.
   Calibration: square ducts within [-0.25 h, +0.5 h]; periodic G/D cells at n = 48 within ~1 voxel of
   the Task 1 table (unit-tested).
3. **inlet -> outlet connectivity**: inlet and outlet terminals in one 6-connected fluid component.
   Terminals: port bores and plenums inside the rendered region; where the sector does not contain one,
   the sector cut face(s) through which the coolant path leaves (fluid within 2 voxels of the rendered
   wedge's side planes).
4. **trapped pockets**: fluid components (inside the jacket envelope and the rendered wedge) that touch
   no terminal and no cut face. Pockets with an equivalent-sphere diameter below 3 voxels are listed
   separately as voxel-scale (not resolvable; rerun finer around them) and do not fail the check.
5. **loose solid**: solid components not connected to the housing (sleeve / outer wall / caps) and not
   touching a cut face (floating debris after clipping or the sliver opening).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage

from voxlat.export.field import JacketField
from voxlat.export.maskio import VoxelMask
from voxlat.export.sector import capped_edt

def _rss_gb() -> float:
    """Resident memory of this process in GB (Linux /proc; 0 elsewhere) - for the verbose log."""
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * 4096 / 1e9
    except OSError:
        pass
    try:  # Windows: working set of this process
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        h = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
        if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):  # type: ignore[attr-defined]
            return pmc.WorkingSetSize / 1e9
    except (AttributeError, OSError):
        pass
    return 0.0


def _trim() -> None:
    """Return freed heap memory to the OS (glibc only; no-op elsewhere). Many slice-sized temporaries
    otherwise fragment the heap and inflate the resident size between stages."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def available_ram_bytes() -> float | None:
    """Physical memory currently available (Linux /proc/meminfo, Windows GlobalMemoryStatusEx); None if unknown."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024.0
    except OSError:
        pass
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        ms = _MS()
        ms.dwLength = ctypes.sizeof(_MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):  # type: ignore[attr-defined]
            return float(ms.ullAvailPhys)
    except (AttributeError, OSError):
        pass
    return None


def _label_centres(lab: np.ndarray, labels) -> dict[int, list[float]]:
    """Mean voxel index of a few labelled components (memory: only their voxels' coordinates)."""
    labels = [int(v) for v in labels]
    if not labels:
        return {}
    idx = np.nonzero(np.isin(lab, labels))
    vals = lab[idx]
    out = {}
    for lbl in labels:
        sel = vals == lbl
        out[lbl] = [float(np.mean(i[sel])) for i in idx]
    return out


__all__ = ["RegionGrid", "region_grid", "min_wall", "throat", "check_mask", "thin_mask"]

_S6 = ndimage.generate_binary_structure(3, 1)
_S26 = np.ones((3, 3, 3), bool)
THROAT_SLACK = -0.25  # voxels: a sphere of diameter d is centred where EDT >= d/2 - 0.25 h


class RegionGrid:
    """Analytic region masks of the jacket on the voxel grid of a mask (computed once, slice by slice).

    Stored compactly as one uint8 code per voxel (bit flags) plus 2-D (x, y) masks for the sector wedge;
    the boolean views (``envelope``, ``lattice``, ...) are materialized on access. Peak memory of the
    checks is then ~10 bytes per voxel (solid, codes, float32 distance, int32 labels).
    """

    ENVELOPE, LATTICE, TERM_IN, TERM_OUT, HOUSING, IFACE, FACE = 1, 2, 4, 8, 16, 32, 64

    def __init__(self, field: JacketField, mask: VoxelMask, band: float):
        nx, ny, nz = mask.shape
        x, y, _ = mask.centres()
        X, Y = np.broadcast_arrays(x[:, :, 0], y[:, :, 0])
        z = mask.axis(2)
        h = mask.voxel
        self.code = np.zeros(mask.shape, np.uint8)
        frame = mask.frame or {}
        self.sector = field.mode == "sector"
        if self.sector:
            half = frame.get("half_angle_deg")
            self.half_wide = math.radians(float(half)) if half is not None else field.sector_half
            self.half_nominal = field.sector_half
            wedge_xy = field.wedge(X, Y, self.half_wide)
            self.nominal_xy = np.abs(np.arctan2(Y, X)) <= self.half_nominal
            near = wedge_xy > -2.0 * h
            self.face_lo_xy = near & (Y < 0)
            self.face_hi_xy = near & (Y > 0)
            self.inside_xy = wedge_xy <= 0.0
        else:
            self.nominal_xy = np.ones_like(X, bool)
            self.face_lo_xy = np.zeros_like(X, bool)
            self.face_hi_xy = np.zeros_like(X, bool)
            self.inside_xy = np.ones_like(X, bool)
        face_xy = (self.face_lo_xy | self.face_hi_xy).astype(np.uint8) * self.FACE
        inside_xy = self.inside_xy
        for k in range(nz):
            zk = np.full_like(X, z[k])
            reg = field.regions(X, Y, zk, inset=h)
            env = reg["envelope"] & inside_xy
            r = reg["r"]
            hous = ((r < field.r_li - field.delta - h) | (r > field.r_lo + field.delta + h)
                    | (zk < field.z_lo + field.t_cap) | (zk > field.z_hi - field.t_cap))
            if field.layout == "slots":
                hous |= (zk < -field.delta - h) | (zk > field.L_ax + field.delta + h)
            c = env.astype(np.uint8) * self.ENVELOPE
            c |= (reg["lattice"] & inside_xy).astype(np.uint8) * self.LATTICE
            c |= ((reg["plenum_in"] | reg["bore_in"]) & inside_xy).astype(np.uint8) * self.TERM_IN
            c |= ((reg["plenum_out"] | reg["bore_out"]) & inside_xy).astype(np.uint8) * self.TERM_OUT
            c |= (hous & env).astype(np.uint8) * self.HOUSING
            c |= (field.interface_distance(X, Y, zk) <= band).astype(np.uint8) * self.IFACE
            c |= face_xy * env
            self.code[:, :, k] = c

    def _flag(self, bit: int) -> np.ndarray:
        return (self.code & bit).astype(bool)

    envelope = property(lambda self: self._flag(self.ENVELOPE))
    lattice = property(lambda self: self._flag(self.LATTICE))
    term_in = property(lambda self: self._flag(self.TERM_IN))
    term_out = property(lambda self: self._flag(self.TERM_OUT))
    housing = property(lambda self: self._flag(self.HOUSING))
    iface = property(lambda self: self._flag(self.IFACE))
    nominal = property(lambda self: np.broadcast_to(self.nominal_xy[:, :, None], self.code.shape))
    inside = property(lambda self: np.broadcast_to(self.inside_xy[:, :, None], self.code.shape))
    face_lo = property(lambda self: self._flag(self.FACE) & self.face_lo_xy[:, :, None])
    face_hi = property(lambda self: self._flag(self.FACE) & self.face_hi_xy[:, :, None])

    def terminal(self, key: str) -> np.ndarray:
        """Boolean terminal region: 'inlet', 'outlet', 'cut_face_lo', 'cut_face_hi' (materialized on demand)."""
        if key == "inlet":
            return self._flag(self.TERM_IN)
        if key == "outlet":
            return self._flag(self.TERM_OUT)
        if key == "cut_face_lo":
            return self.face_lo
        if key == "cut_face_hi":
            return self.face_hi
        raise KeyError(key)

    def pairs(self) -> list[tuple[str, str]]:
        """Terminal pairs (names, see ``terminal``) that must be connected through the fluid."""
        has_in = bool(np.any(self.code & self.TERM_IN))
        has_out = bool(np.any(self.code & self.TERM_OUT))
        if not self.sector or (has_in and has_out):
            return [("inlet", "outlet")]
        if has_in:
            return [("inlet", "cut_face_lo"), ("inlet", "cut_face_hi")]
        if has_out:
            return [("cut_face_lo", "outlet"), ("cut_face_hi", "outlet")]
        return [("cut_face_lo", "cut_face_hi")]


def region_grid(field: JacketField, mask: VoxelMask, band: float) -> RegionGrid:
    return RegionGrid(field, mask, band)


def _near_cut(R: RegionGrid, solid: np.ndarray) -> np.ndarray:
    """Solid voxels within 2 voxels of the rendered wedge's side planes (they continue outside)."""
    if not R.sector:
        return np.zeros((1, 1, 1), bool)
    return (R.face_lo_xy | R.face_hi_xy)[:, :, None] & solid


# =============================================================================
# Wall thickness
# =============================================================================
ERODE_SLACK = 0.25  # voxels: ball centres need EDT >= d/2 + 0.25 h
DILATE_SLACK = 1.25  # voxels: covered if within d/2 + 1.25 h of a ball centre (gap of 1 h keeps the test monotone)


def thin_mask(solid: np.ndarray, voxel: float, d: float, D: np.ndarray | None = None) -> np.ndarray:
    """Solid voxels with local thickness below d: solid minus its opening by a ball of diameter d.

    Discrete opening on voxel centres: erosion EDT >= d/2 + 0.25 h, dilation <= d/2 + 1.25 h. The 1-voxel
    gap between the two slacks keeps the test monotone in d on plates, rods and balls (unit-tested);
    the threshold of the discrete test lies within [-0.5 h, +0.75 h] of the true thickness (plates, rods);
    the bisection in ``min_wall`` adds up to -h/4.
    """
    rho = 0.5 * d
    h = voxel
    if D is None:
        D = capped_edt(solid, h, rho + 2 * h)
    eroded = D >= rho + ERODE_SLACK * h
    if not eroded.any():
        return solid.copy()
    E = capped_edt(~eroded, h, rho + (DILATE_SLACK + 1) * h)
    return solid & ~(E <= rho + DILATE_SLACK * h)


def _crop(*arrays: np.ndarray, sel: np.ndarray, pad: int) -> tuple[tuple[slice, ...], list[np.ndarray]]:
    idx = np.argwhere(sel.any(axis=2))
    kz = np.flatnonzero(sel.any(axis=(0, 1)))
    lo = [max(0, int(idx[:, 0].min()) - pad), max(0, int(idx[:, 1].min()) - pad), max(0, int(kz.min()) - pad)]
    hi = [min(sel.shape[0], int(idx[:, 0].max()) + pad + 1), min(sel.shape[1], int(idx[:, 1].max()) + pad + 1),
          min(sel.shape[2], int(kz.max()) + pad + 1)]
    sl = tuple(slice(lo[k], hi[k]) for k in range(3))
    return sl, [a[sl] for a in arrays]


def min_wall(solid: np.ndarray, voxel: float, evaluate: np.ndarray, t_limit: float, d_max: float = 1.0,
             tol: float | None = None) -> dict[str, Any]:
    """Minimum local thickness (mm) over the solid voxels in ``evaluate``: pass/fail at ``t_limit`` and the
    value by bisection (to h/4) on ``thin_mask``. Works on the bounding box of ``evaluate`` (+ d_max)."""
    h = voxel
    tol = 0.25 * h if tol is None else tol
    sel = solid & evaluate
    if not sel.any():
        return {"min_wall_mm": None, "evaluated_voxels": 0, "thin_voxels_at_limit": 0}
    pad = int(math.ceil(d_max / h)) + 3
    sl, (S, Ev) = _crop(solid, evaluate, sel=sel, pad=pad)
    Ev = Ev & S
    D = capped_edt(S, h, 0.5 * d_max + 2 * h)

    def fails(d: float) -> np.ndarray:
        return thin_mask(S, h, d, D) & Ev

    bad = fails(t_limit)
    out: dict[str, Any] = {"evaluated_voxels": int(Ev.sum()), "resolution_mm": [-0.75 * h, 0.75 * h],
                           "thin_voxels_at_limit": int(bad.sum())}
    if bad.any():
        lo, hi = 0.0, t_limit
        lab, n = ndimage.label(bad, structure=np.ones((3, 3, 3), bool))
        sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        top = [int(v) for v in np.argsort(sizes)[::-1][:5] + 1]
        cen = _label_centres(lab, top)
        del lab
        out["thin_regions"] = int(n)
        out["largest_thin_regions_index"] = [[cen[t][k] + sl[k].start for k in range(3)] for t in top]
        out["largest_thin_regions_voxels"] = [int(sizes[t - 1]) for t in top]
    else:
        lo, hi = t_limit, d_max
        if not fails(hi).any():
            out["min_wall_mm"] = float(d_max)
            out["min_wall_is_lower_bound"] = True
            return out
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if fails(mid).any():
            hi = mid
        else:
            lo = mid
    out["min_wall_mm"] = float(lo)
    return out


# =============================================================================
# Pores (throat between terminals)
# =============================================================================
def _connected(passable: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    ia = np.flatnonzero(a)  # terminals are small; index lists avoid full-size temporaries
    ib = np.flatnonzero(b)
    pa, pb = passable.ravel()[ia], passable.ravel()[ib]
    if not pa.any() or not pb.any():
        return False
    lab, _ = ndimage.label(passable, structure=_S26)
    la = np.unique(lab.ravel()[ia[pa]])
    lb = np.unique(lab.ravel()[ib[pb]])
    la = la[la > 0]
    lb = lb[lb > 0]
    return bool(np.intersect1d(la, lb).size)


def throat(fluid: np.ndarray, Df: np.ndarray, voxel: float, a: np.ndarray, b: np.ndarray, d_max: float = 3.0,
           tol: float | None = None) -> float:
    """Largest sphere diameter (mm) that passes from terminal a to terminal b through ``fluid``."""
    h = voxel
    tol = 0.25 * h if tol is None else tol

    def passes(d: float) -> bool:
        p = Df >= np.float32(0.5 * d + THROAT_SLACK * h)
        p &= fluid
        return _connected(p, a, b)

    if not passes(0.0):
        return 0.0
    lo, hi = 0.0, d_max
    if passes(hi):
        return d_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if passes(mid):
            lo = mid
        else:
            hi = mid
    return float(lo)


# =============================================================================
# Everything
# =============================================================================
def check_mask(mask: VoxelMask, field: JacketField, *, min_wall_mm: float | None = None,
               min_pore_mm: float | None = None, band_mm: float | None = None, pocket_resolution_voxels: float = 3.0,
               d_max_wall: float = 1.0, d_max_pore: float = 3.0, verbose: bool = False) -> dict[str, Any]:
    """Run checks 1-5 (module docstring) and return a JSON-serializable report."""
    import time

    t0 = time.perf_counter()
    mfg = field.config["manufacturing"]
    t_min = float(mfg["min_wall_mm"]) if min_wall_mm is None else float(min_wall_mm)
    p_min = float(mfg["min_pore_mm"]) if min_pore_mm is None else float(min_pore_mm)
    h = mask.voxel
    band = t_min + 2 * h if band_mm is None else float(band_mm)
    solid = mask.solid
    def log(m: str) -> None:
        _trim()
        if verbose:
            print(f"  [checks {time.perf_counter() - t0:6.1f}s, {_rss_gb():.2f} GB] {m}", flush=True)
    log(f"regions on {mask.shape} voxels of {h} mm")
    R = RegionGrid(field, mask, band)
    report: dict[str, Any] = {"voxel_mm": h, "shape": list(mask.shape), "source": mask.source, "kind": mask.kind,
                              "limits": {"min_wall_mm": t_min, "min_pore_mm": p_min, "interface_band_mm": band}}
    resolved = h <= t_min / 4.0
    report["wall_resolved"] = bool(resolved)

    # ---- 1. walls
    log("wall thickness")
    lattice = R.lattice & R.nominal
    iface = R.iface
    wall = min_wall(solid, h, lattice & ~iface, t_min, d_max=d_max_wall)
    if wall.get("largest_thin_regions_index"):
        wall["largest_thin_regions_mm"] = [[mask.origin[k] + h * p[k] for k in range(3)]
                                           for p in wall.pop("largest_thin_regions_index")]
    report["wall"] = wall
    log(f"  min wall {wall.get('min_wall_mm')}")
    sl = min_wall(solid, h, lattice & iface, t_min, d_max=d_max_wall)
    del lattice, iface
    report["interface_slivers"] = {"min_wall_mm": sl.get("min_wall_mm"), "thin_voxels": sl.get("thin_voxels_at_limit", 0),
                                   "thin_volume_mm3": sl.get("thin_voxels_at_limit", 0) * h**3, "band_mm": band}
    wall_ok = wall["min_wall_mm"] is not None and wall["min_wall_mm"] >= t_min - 0.5 * h

    # ---- 3. connectivity (6-connected fluid components)
    fluid = ~solid & R.envelope
    pairs = R.pairs()
    lab, nlab = ndimage.label(fluid, structure=_S6)
    log(f"fluid components: {nlab}")
    conn_list = []
    connected_all = True

    def labels_in(key: str) -> np.ndarray:
        m = R.terminal(key)
        m &= fluid
        u = np.unique(lab[m])
        return u[u > 0]

    for na, nb in pairs:
        la, lb = labels_in(na), labels_in(nb)
        conn = bool(np.intersect1d(la, lb).size)
        conn_list.append(conn)
        connected_all &= conn and bool(la.size) and bool(lb.size)

    # ---- 4. pockets
    term = (R.code & (R.TERM_IN | R.TERM_OUT | R.FACE)).astype(bool)
    good = np.unique(lab[fluid & term])
    del term
    good = good[good > 0]
    sizes = np.bincount(lab.ravel(), minlength=nlab + 1)
    bad_labels = np.setdiff1d(np.arange(1, nlab + 1), good)
    bad_sizes = sizes[bad_labels]
    # pockets whose equivalent-sphere diameter is below ~3 voxels cannot be resolved (a closed or open
    # contact between a strut and a wall at voxel scale): listed as voxel-scale, verified at a finer voxel
    n_res = math.pi / 6.0 * pocket_resolution_voxels**3
    pockets = bad_labels[bad_sizes >= n_res]
    noise = bad_labels[bad_sizes < n_res]
    pocket_list = []
    if pockets.size:
        order = pockets[np.argsort(sizes[pockets])[::-1][:10]]
        cen = _label_centres(lab, order)
        for lbl, c in ((o, cen[int(o)]) for o in order):
            pocket_list.append({"voxels": int(sizes[lbl]), "volume_mm3": float(sizes[lbl] * h**3),
                                "centre_mm": [float(mask.origin[k] + h * c[k]) for k in range(3)]})
    voxel_scale = []
    if noise.size:
        order = noise[np.argsort(sizes[noise])[::-1][:10]]
        cen = _label_centres(lab, order)
        for lbl, c in ((o, cen[int(o)]) for o in order):
            voxel_scale.append({"voxels": int(sizes[lbl]),
                                "centre_mm": [float(mask.origin[k] + h * c[k]) for k in range(3)]})
    report["trapped_pockets"] = {"count": int(pockets.size), "volume_mm3": float(sizes[pockets].sum() * h**3),
                                 "largest": pocket_list,
                                 "voxel_scale_count": int(noise.size),
                                 "voxel_scale_volume_mm3": float(sizes[noise].sum() * h**3),
                                 "voxel_scale_largest": voxel_scale,
                                 "resolution_diameter_mm": pocket_resolution_voxels * h}
    pockets_ok = pockets.size == 0
    del lab
    log(f"pockets: {int(pockets.size)}")

    # ---- 2. throat per terminal pair (largest passable sphere)
    log("fluid distance")
    # distance to the nearest solid voxel; outside the rendered wedge counts as open (the coolant continues)
    Df = capped_edt(~solid, h, 0.5 * d_max_pore + 2 * h)
    pair_rep = []
    throat_min = math.inf
    for (na, nb), conn in zip(pairs, conn_list):
        A = R.terminal(na)
        A &= fluid
        B = R.terminal(nb)
        B &= fluid
        thr = throat(fluid, Df, h, A, B, d_max=d_max_pore) if conn else 0.0
        throat_min = min(throat_min, thr)
        pair_rep.append({"from": na, "to": nb, "connected": conn, "throat_mm": thr,
                         "terminal_voxels": [int(A.sum()), int(B.sum())]})
        del A, B
        log(f"  {na} -> {nb}: connected={conn}, throat={thr:.3f} mm")
    del Df, pairs, fluid
    report["paths"] = pair_rep
    report["connected"] = bool(connected_all)
    report["min_throat_mm"] = float(throat_min) if pair_rep else None
    pore_ok = connected_all and throat_min >= p_min - 0.5 * h

    # ---- 5. loose solid
    log("solid components")
    sol = solid & R.inside
    slab, ns = ndimage.label(sol, structure=_S6)
    anchored = np.unique(slab[sol & ((R.code & R.HOUSING).astype(bool) | _near_cut(R, solid))])
    anchored = anchored[anchored > 0]
    ssz = np.bincount(slab.ravel(), minlength=ns + 1)
    loose = np.setdiff1d(np.arange(1, ns + 1), anchored)
    report["loose_solid"] = {"count": int(loose.size), "volume_mm3": float(ssz[loose].sum() * h**3),
                             "largest_voxels": int(ssz[loose].max()) if loose.size else 0, "solid_components": int(ns)}
    loose_ok = loose.size == 0
    del slab

    report["pass"] = {"min_wall": bool(wall_ok) if resolved else None, "min_pore": bool(pore_ok),
                      "connected": bool(connected_all), "no_trapped_pockets": bool(pockets_ok),
                      "no_loose_solid": bool(loose_ok)}
    report["all_pass"] = all(v for v in report["pass"].values() if v is not None)
    report["wall_time_s"] = time.perf_counter() - t0
    return report
