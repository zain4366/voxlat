"""Tensor-product B-spline grading fields on the unwrapped jacket (s, z) - Task 11.

Each design field f in {rho*, w, L} (optionally a_z) is

    f(s, z) = lo_f + (hi_f - lo_f) * sum_ij x_ij  B_i^s(s)  B_j^z(z),      x_ij in [0, 1],

with the design bounds [lo_f, hi_f] of ``configs/reference.yaml``. Because both bases are
non-negative partitions of unity, f stays inside its bounds for every x in the unit cube
(convex-hull property), so the bound constraints never have to be checked.

* s (circumference, periodic): uniform cubic B-spline with n_s control points; control i is
  centred at theta_i = i 360/n_s deg (theta = s / r_m, measured from theta = 0, the inlet
  slot centre of the reference layout). B_i^s(s) = sum_k B3(n_s s / C - i - k n_s) with the
  centred cardinal cubic B3(t) = (4 - 6t^2 + 3|t|^3)/6 for |t| < 1, (2 - |t|)^3/6 for
  1 <= |t| < 2, 0 otherwise. Periodic, C2.
* z (axial, open): clamped uniform B-spline of degree p = min(3, n_z - 1) on [0, L_ax]
  (open uniform knot vector with p + 1 repeated end knots). n_z = 4 -> cubic Bezier,
  n_z = 2 -> linear, n_z = 1 -> constant. Cells outside [0, L_ax] (end-ring plenum rows)
  take the end value.

Default control grids (plan): 8 x 4 for rho* and w, 4 x 2 for L, a_z fixed at 1
(72 design variables). Variable order: rho (row-major (i_s, j_z)), w, L, [a_z].

Manufacturing constraints (Task 1, Task 7; evaluated exactly as ``JacketModel`` does on the
model grid, without running the solver):

* grading gradient: max over lattice-lattice faces of |rho_r - rho_l| / d * L_face <= limit
  (config ``max_density_change_per_cell`` = 0.05);
* cell size: L >= L_min(w, rho*) in every lattice cell (Task 1 morphology table, strict
  min-wall statistic, a_z = 1 screening): g_cell = 1 - min(L / L_min) <= 0.

Feasibility repair (``repair``; applied identically by every optimizer)
-----------------------------------------------------------------------
With the strict wall criterion the cell-size constraint is very restrictive for blends
(L_min reaches 4.8-7.6 mm for 0.3 < w < 0.75 at some densities, and > 6 mm - infeasible at any
allowed L - for w ~ 0.59-0.75, rho* ~ 0.22-0.34; Task 1 open issue 1). B-spline fields of random
control nets are weighted averages of several controls, so random designs are mostly mid-range
blends and almost never satisfy the constraint in all ~7500 cells (measured: 8 feasible out of
40 000 Sobol points). ``repair`` therefore maps a design to a nearby feasible one:
1. blend polarization (up to ``w_passes`` passes): every w control whose support contains a
   cell with L < L_min moves halfway toward the design's majority pure type among those
   controls (0 gyroid / 1 diamond; the strict-wall-feasible morphologies, L_min <= 2.2 mm over
   the density range). One common target, because a gyroid-diamond transition itself crosses
   the infeasible blend band;
2. cell size (afterwards, if still needed): each L control k is raised by the largest deficit
   L_min,c - L_c over the violating cells in its support (unit-scaled, capped at the upper
   bound); since the basis is a non-negative partition of unity every violating cell rises by
   at least its deficit;
3. gradient: the rho control net is shrunk toward its mean by alpha = limit / G_max - the
   field is linear in the controls, so the gradient scales exactly by alpha (mean density kept);
4. repeat (each step changes the others' inputs) until all hold (typically 2-5 passes).
A design that stays infeasible is a cheap reject. Repair never touches feasible designs.
(A first version with step 2 only pushed nearly every random design to L ~ 5-6 mm, which
starved NSGA-II's initial population of diversity; polarization keeps the cell size.)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "GradingSpec",
    "GradingParameterization",
    "periodic_cubic_basis",
    "clamped_bspline_basis",
    "grading_gradient_max",
    "FIELD_NAMES",
    "DEFAULT_CONSTRAINTS",
]

FIELD_NAMES = ("rho", "w", "L", "a_z")

# Cell-size constraint used inside the optimization loop (Task 11, after the first full run):
#   table "fine": Task 11 fine-w morphology table (w step 0.025) instead of the Task 1 table (0.125);
#   cell_margin: L >= (1 + margin) L_min - covers interpolation between table nodes;
#   valley_wall: reject cells in a table quad touching a pinch-neck valley (wall_min / L below this at a corner).
# {"table": "coarse", "cell_margin": 0.0, "valley_wall": None} reproduces the first study.
DEFAULT_CONSTRAINTS = {"table": "fine", "cell_margin": 0.01, "valley_wall": 0.09}


# =============================================================================
# Bases
# =============================================================================
def _b3(t: np.ndarray) -> np.ndarray:
    a = np.abs(t)
    out = np.zeros_like(a)
    m1 = a < 1.0
    m2 = (a >= 1.0) & (a < 2.0)
    out[m1] = (4.0 - 6.0 * a[m1] ** 2 + 3.0 * a[m1] ** 3) / 6.0
    out[m2] = (2.0 - a[m2]) ** 3 / 6.0
    return out


def periodic_cubic_basis(u: Any, n: int) -> np.ndarray:
    """Periodic uniform cubic B-spline basis (len(u), n); u in units of the period [0, 1).

    Control i is centred at u = i / n. Rows sum to 1 (partition of unity) for every n >= 1.
    """
    u = np.asarray(u, float).ravel()
    if n < 1:
        raise ValueError("need at least one control point")
    if n == 1:
        return np.ones((len(u), 1))
    t = np.mod(u * n, n)  # in [0, n)
    B = np.zeros((len(u), n))
    for i in range(n):
        d = t - i
        for k in (-2, -1, 0, 1, 2):  # wrapped copies (n may be < 4)
            B[:, i] += _b3(d - k * n)
    return B


def clamped_bspline_basis(t: Any, n: int, degree: int | None = None) -> np.ndarray:
    """Clamped (open uniform) B-spline basis (len(t), n) on t in [0, 1]; degree min(3, n - 1)."""
    t = np.clip(np.asarray(t, float).ravel(), 0.0, 1.0)
    if n < 1:
        raise ValueError("need at least one control point")
    if n == 1:
        return np.ones((len(t), 1))
    p = min(3, n - 1) if degree is None else int(degree)
    if not 1 <= p <= n - 1:
        raise ValueError(f"degree {p} impossible with {n} control points")
    inner = np.linspace(0.0, 1.0, n - p + 1)
    knots = np.concatenate([np.zeros(p), inner, np.ones(p)])
    # Cox-de Boor, right end handled by assigning t = 1 to the last non-empty span
    B = np.zeros((len(t), len(knots) - 1))
    last = len(knots) - p - 2
    for i in range(len(knots) - 1):
        if knots[i] < knots[i + 1]:
            B[:, i] = (t >= knots[i]) & (t < knots[i + 1])
    B[t >= 1.0, :] = 0.0
    B[t >= 1.0, last] = 1.0
    for k in range(1, p + 1):
        Bn = np.zeros((len(t), len(knots) - 1 - k))
        for i in range(len(knots) - 1 - k):
            d1 = knots[i + k] - knots[i]
            d2 = knots[i + k + 1] - knots[i + 1]
            a = (t - knots[i]) / d1 * B[:, i] if d1 > 0 else 0.0
            b = (knots[i + k + 1] - t) / d2 * B[:, i + 1] if d2 > 0 else 0.0
            Bn[:, i] = a + b
        B = Bn
    return B


# =============================================================================
# Specification
# =============================================================================
@dataclass(frozen=True)
class GradingSpec:
    """Control grids (n_s, n_z) per field; (0, 0) = field fixed at ``fixed[name]``."""

    rho: tuple[int, int] = (8, 4)
    w: tuple[int, int] = (8, 4)
    L: tuple[int, int] = (4, 2)
    a_z: tuple[int, int] = (0, 0)
    fixed: Mapping[str, float] = field(default_factory=lambda: {"a_z": 1.0})

    def shape(self, name: str) -> tuple[int, int]:
        return tuple(getattr(self, name))  # type: ignore[return-value]

    def active(self) -> list[str]:
        return [n for n in FIELD_NAMES if np.prod(self.shape(n)) > 0]

    @property
    def n_var(self) -> int:
        return int(sum(np.prod(self.shape(n)) for n in self.active()))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fixed"] = dict(self.fixed)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GradingSpec":
        kw = {k: tuple(v) for k, v in d.items() if k in FIELD_NAMES}
        return cls(**kw, fixed=dict(d.get("fixed", {"a_z": 1.0})))


def _bounds(cfg: Any, name: str) -> tuple[float, float]:
    b = cfg.design_bounds
    r = {"rho": b.relative_density, "w": b.blend_w, "L": b.cell_size, "a_z": b.axial_stretch}[name]
    return float(r.lo), float(r.hi)


class FastMinCellSize:
    """Vectorized L_min(w, rho) - the same bilinear interpolation of the Task 1 morphology table as
    ``geometry.min_cell_size`` (uniform table grid), ~20x faster than RegularGridInterpolator."""

    def __init__(self, cfg: Any, fine: bool = False):
        from voxlat.geometry.tpms import load_morphology_table

        tab = load_morphology_table(fine)
        self.w, self.r = np.asarray(tab.w, float), np.asarray(tab.rho, float)
        if not (np.allclose(np.diff(self.w), self.w[1] - self.w[0]) and np.allclose(np.diff(self.r), self.r[1] - self.r[0])):
            raise ValueError("morphology table grid is not uniform")
        self.t = np.asarray(tab.data["wall_min"], float)
        self.p = np.asarray(tab.data["pore_throat"], float)
        self.t_min = cfg.manufacturing.min_wall_thickness
        self.p_min = cfg.manufacturing.min_pore_size

    def quad_min_wall(self, w: Any, rho: Any) -> np.ndarray:
        """Smallest table wall_min/L at the 4 corners of the table quad containing each (w, rho) - the
        valley screen: pinch necks are narrow curves in (w, rho) on which the wall vanishes, so a cell whose quad
        touches one cannot be trusted to the interpolated value."""
        w = np.clip(np.asarray(w, float), self.w[0], self.w[-1])
        r = np.clip(np.asarray(rho, float), self.r[0], self.r[-1])
        i = np.minimum(((w - self.w[0]) / (self.w[1] - self.w[0])).astype(np.intp), len(self.w) - 2)
        j = np.minimum(((r - self.r[0]) / (self.r[1] - self.r[0])).astype(np.intp), len(self.r) - 2)
        T = self.t
        return np.minimum(np.minimum(T[i, j], T[i + 1, j]), np.minimum(T[i, j + 1], T[i + 1, j + 1]))

    def __call__(self, w: Any, rho: Any) -> np.ndarray:
        w = np.clip(np.asarray(w, float), self.w[0], self.w[-1])
        r = np.clip(np.asarray(rho, float), self.r[0], self.r[-1])
        fw = (w - self.w[0]) / (self.w[1] - self.w[0])
        fr = (r - self.r[0]) / (self.r[1] - self.r[0])
        i = np.minimum(fw.astype(np.intp), len(self.w) - 2)  # fw >= 0 after clipping
        j = np.minimum(fr.astype(np.intp), len(self.r) - 2)
        a, b = fw - i, fr - j
        k = i * len(self.r) + j
        nr = len(self.r)
        c00, c10, c01, c11 = (1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b
        T, Pt = self.t.ravel(), self.p.ravel()
        t = c00 * T[k] + c10 * T[k + nr] + c01 * T[k + 1] + c11 * T[k + nr + 1]
        p = c00 * Pt[k] + c10 * Pt[k + nr] + c01 * Pt[k + 1] + c11 * Pt[k + nr + 1]
        with np.errstate(divide="ignore"):
            Lw = np.where(t > 0, self.t_min / t, np.inf)
            Lp = np.where(p > 0, self.p_min / p, np.inf)
        return np.maximum(Lw, Lp)


def grading_gradient_max(model: Any, rho: np.ndarray, L: np.ndarray) -> float:
    """max |grad rho*| L over lattice-lattice faces, exactly as ``JacketModel._manufacturability``."""
    vals = []
    for key in ("s", "z"):
        f = model._faces[key]
        m = f["LL"]
        r_l = rho.ravel()[f["l"][m]]
        r_r = rho.ravel()[f["r"][m]]
        Lf = 0.5 * (L.ravel()[f["l"][m]] + L.ravel()[f["r"][m]])
        vals.append(np.abs(r_r - r_l) / (f["dl"][m] + f["dr"][m]) * Lf)
    v = np.concatenate(vals)
    return float(np.max(v)) if v.size else 0.0


# =============================================================================
# Parameterization on a model grid
# =============================================================================
class GradingParameterization:
    """Maps unit design vectors x in [0, 1]^n_var to cell fields on ``model.grid``."""

    def __init__(self, model: Any, spec: GradingSpec | None = None, constraints: Mapping[str, Any] | None = None):
        self.model = model
        self.constraints = dict(DEFAULT_CONSTRAINTS if constraints is None else constraints)
        self.cfg = model.cfg
        self.spec = spec or GradingSpec()
        g = model.grid
        self.grid = g
        self.active = self.spec.active()
        self.bounds = {n: _bounds(self.cfg, n) for n in FIELD_NAMES}
        self._slices: dict[str, slice] = {}
        k = 0
        for n in self.active:
            m = int(np.prod(self.spec.shape(n)))
            self._slices[n] = slice(k, k + m)
            k += m
        self.n_var = k
        u = g.s_centres / g.circumference
        t = np.clip(g.z_centres / g.axial_length, 0.0, 1.0)
        self._Bs = {n: periodic_cubic_basis(u, self.spec.shape(n)[0]) for n in self.active}
        self._Bz = {n: clamped_bspline_basis(t, self.spec.shape(n)[1]) for n in self.active}
        self.limit = float(self.cfg.manufacturing.max_density_change_per_cell
                           if model.options.gradient_limit is None else model.options.gradient_limit)
        self._lat = g.lattice
        self._supp: dict[str, np.ndarray] = {}
        c = self.constraints
        self._lmin_table = FastMinCellSize(self.cfg, fine=c["table"] == "fine")
        self._margin = float(c["cell_margin"])
        self._valley = None if c.get("valley_wall") is None else float(c["valley_wall"])
        self._lat_flat = np.flatnonzero(g.lattice.ravel())
        self._fa: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------ names
    def variable_names(self) -> list[str]:
        out = []
        for n in self.active:
            ns, nz = self.spec.shape(n)
            out += [f"{n}_{i}_{j}" for i in range(ns) for j in range(nz)]
        return out

    # ------------------------------------------------------------------ maps
    def unit_nets(self, x: Any) -> dict[str, np.ndarray]:
        x = np.asarray(x, float).ravel()
        if x.size != self.n_var:
            raise ValueError(f"expected {self.n_var} variables, got {x.size}")
        return {n: x[self._slices[n]].reshape(self.spec.shape(n)) for n in self.active}

    def control_nets(self, x: Any) -> dict[str, np.ndarray]:
        """Physical control values (SI) per active field, shape (n_s, n_z)."""
        out = {}
        for n, u in self.unit_nets(x).items():
            lo, hi = self.bounds[n]
            out[n] = lo + u * (hi - lo)
        return out

    def x_from_control_nets(self, nets: Mapping[str, Any]) -> np.ndarray:
        x = np.zeros(self.n_var)
        for n in self.active:
            lo, hi = self.bounds[n]
            x[self._slices[n]] = ((np.asarray(nets[n], float) - lo) / (hi - lo)).ravel()
        return x

    def fields(self, x: Any) -> dict[str, np.ndarray]:
        """Cell-centre fields (n_s, n_z) on the model grid: rho, w, L, a_z (fixed ones as arrays)."""
        nets = self.control_nets(x)
        out = {}
        for n in FIELD_NAMES:
            if n in nets:
                out[n] = self._Bs[n] @ nets[n] @ self._Bz[n].T
            else:
                out[n] = np.full(self.grid.shape, float(self.spec.fixed.get(n, 1.0 if n == "a_z" else np.nan)))
        return out

    def field_function(self, x: Any, name: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """f(s, z) at arbitrary points (s = theta r_m in m, z in m) - for export (Task 12) and plots."""
        nets = self.control_nets(x)
        g = self.grid
        if name not in nets:
            v = float(self.spec.fixed.get(name, 1.0))
            return lambda s, z: np.full(np.broadcast(np.asarray(s), np.asarray(z)).shape, v)
        P = nets[name]
        ns, nz = P.shape

        def f(s: Any, z: Any) -> np.ndarray:
            s, z = np.broadcast_arrays(np.asarray(s, float), np.asarray(z, float))
            Bs = periodic_cubic_basis(s.ravel() / g.circumference, ns)
            Bz = clamped_bspline_basis(z.ravel() / g.axial_length, nz)
            return np.einsum("ki,ij,kj->k", Bs, P, Bz).reshape(s.shape)

        return f

    # ------------------------------------------------------------------ constraints (batched)
    def fields_batch(self, X: Any, names: Sequence[str] = ("rho", "w", "L")) -> dict[str, np.ndarray]:
        """Fields of many designs, flattened cells: {name: (b, n_s * n_z)}."""
        X = np.atleast_2d(np.asarray(X, float))
        b = len(X)
        out = {}
        for n in names:
            if n in self.active:
                lo, hi = self.bounds[n]
                P = lo + (hi - lo) * X[:, self._slices[n]].reshape((b,) + self.spec.shape(n))
                out[n] = np.matmul(self._Bs[n], np.matmul(P, self._Bz[n].T)).reshape(b, -1)
            else:
                out[n] = np.full((b, self.grid.n_cells), float(self.spec.fixed.get(n, 1.0)))
        return out

    def _face_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._fa is None:
            ls, rs, ds = [], [], []
            for key in ("s", "z"):
                f = self.model._faces[key]
                m = f["LL"]
                ls.append(f["l"][m])
                rs.append(f["r"][m])
                ds.append(f["dl"][m] + f["dr"][m])
            self._fa = (np.concatenate(ls), np.concatenate(rs), np.concatenate(ds))
        return self._fa

    def _gradient_batch(self, rho: np.ndarray, L: np.ndarray) -> np.ndarray:
        """max |grad rho*| L per design (b,), same face formula as the device model."""
        l, r, d = self._face_arrays()
        if l.size == 0:
            return np.zeros(len(rho))
        return np.max(np.abs(rho[:, r] - rho[:, l]) / d * (0.5 * (L[:, l] + L[:, r])), axis=1)

    def _lmin(self, w: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """Effective cell-size limit (1 + cell_margin) L_min(w, rho) of the configured table."""
        return (1.0 + self._margin) * self._lmin_table(w, rho)

    def _valley_mask(self, w: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """True where a cell lies in a table quad touching a pinch-neck valley (False everywhere if disabled)."""
        if self._valley is None:
            return np.zeros(np.shape(w), bool)
        return self._lmin_table.quad_min_wall(w, rho) < self._valley

    def _lmin_batch(self, w: np.ndarray, rho: np.ndarray) -> np.ndarray:
        idx = self._lat_flat
        return self._lmin(np.clip(w[:, idx], 0.0, 1.0), np.clip(rho[:, idx], 0.15, 0.55))

    def _valley_batch(self, w: np.ndarray, rho: np.ndarray) -> np.ndarray:
        idx = self._lat_flat
        return self._valley_mask(np.clip(w[:, idx], 0.0, 1.0), np.clip(rho[:, idx], 0.15, 0.55))

    def gradient_max(self, x: Any, F: Mapping[str, np.ndarray] | None = None) -> float:
        if F is not None:
            return grading_gradient_max(self.model, F["rho"], F["L"])
        Fb = self.fields_batch(x, ("rho", "L"))
        return float(self._gradient_batch(Fb["rho"], Fb["L"])[0])

    def _support(self, name: str) -> np.ndarray:
        """(n_lattice_cells, n_ctrl) boolean: control k influences cell c."""
        if name not in self._supp:
            W = np.einsum("ia,jb->ijab", self._Bs[name], self._Bz[name])
            W = W.reshape(self.grid.n_s * self.grid.n_z, -1)[self._lat_flat]
            self._supp[name] = W > 1e-12
        return self._supp[name]

    def _L_min(self, F: Mapping[str, np.ndarray]) -> np.ndarray:
        """L_min on the lattice cells - identical to ``geometry.min_cell_size`` (bilinear on the same table)."""
        lat = self._lat
        return self._lmin(np.clip(F["w"][lat], 0.0, 1.0), np.clip(F["rho"][lat], 0.15, 0.55))

    def repair(self, x: Any, safety: float = 1e-6, max_pass: int = 10) -> np.ndarray:
        """Feasibility repair of one design (see ``repair_batch``)."""
        return self.repair_batch(np.asarray(x, float).ravel()[None], safety, max_pass)[0]

    def repair_batch(self, X: Any, safety: float = 1e-6, max_pass: int = 10, chunk: int = 128,
                     w_passes: int = 6) -> np.ndarray:
        """Feasibility repair (module docstring), vectorized over designs: raise the L controls over
        cells with L < L_min, shrink the rho net until |grad rho*| L <= limit, repeat. Feasible designs
        are returned unchanged; designs whose deficit persists with the L controls at their bound stop
        after two such passes (cheap rejects)."""
        X = np.clip(np.array(np.atleast_2d(X), float), 0.0, 1.0)
        lim = self.limit * (1.0 - safety)  # trigger; the targets below are slightly stricter, so a
        # repaired design is a fixed point of the repair (idempotent)
        for a0 in range(0, len(X), chunk):
            Xc = X[a0:a0 + chunk]
            live = np.ones(len(Xc), bool)
            stuck = np.zeros(len(Xc), int)
            wpass = np.zeros(len(Xc), int)
            for _ in range(max_pass):
                idx = np.flatnonzero(live)
                if idx.size == 0:
                    break
                changed = np.zeros(idx.size, bool)
                if "L" in self.active:
                    F = self.fields_batch(Xc[idx], ("rho", "w", "L"))
                    Lm = self._lmin_batch(F["w"], F["rho"])
                    Lc = F["L"][:, self._lat_flat]
                    vmask = self._valley_batch(F["w"], F["rho"])  # cells on a pinch-neck valley
                    has = np.any(Lm * (1.0 + safety) > Lc, axis=1)
                    deficit = Lm * (1.0 + 2.0 * safety) - Lc
                    bad_cells = (deficit > 0) | vmask
                    has_v = vmask.any(axis=1)
                    pol = (has | has_v) & (wpass[idx] < w_passes) if "w" in self.active else np.zeros_like(has)
                    if pol.any():  # step 1: polarize the blend controls over violating / valley cells
                        rows = np.flatnonzero(pol)
                        sw = self._slices["w"]
                        touch = (np.where(bad_cells[rows], 1.0, 0.0) @ self._support("w")) > 0  # (r, n_w)
                        xw = Xc[idx[rows], sw]
                        # common target per design: the majority type of the touched controls (a G-D
                        # transition itself crosses the infeasible blend band, so polarizing each control
                        # to its own nearer type would just create new transitions)
                        tw = np.where(touch, xw, np.nan)
                        tgt = np.where(np.nanmean(np.where(touch.any(1, keepdims=True), tw, 0.5), axis=1) < 0.5, 0.0, 1.0)
                        Xc[idx[rows], sw] = np.where(touch, xw + 0.5 * (tgt[:, None] - xw), xw)
                        wpass[idx[rows]] += 1
                        changed[rows] = True
                        has = has & ~pol
                    if has.any():  # step 2: raise the cell size
                        lo, hi = self.bounds["L"]
                        sl = self._slices["L"]
                        supp = self._support("L")
                        rows = np.flatnonzero(has)
                        d = np.clip(deficit[rows], 0.0, None)  # (r, n_lat)
                        need = np.max(np.where(supp[None], d[:, :, None], 0.0), axis=1) / (hi - lo)  # (r, n_ctrl)
                        xL = Xc[idx[rows], sl]
                        at_cap = np.all((xL >= 1.0) | (need <= 0), axis=1)
                        stuck[idx[rows]] += at_cap
                        Xc[idx[rows], sl] = np.minimum(xL + need, 1.0)
                        changed[rows] = True
                if "rho" in self.active:
                    F = self.fields_batch(Xc[idx], ("rho", "L"))
                    G = self._gradient_batch(F["rho"], F["L"])
                    over = G > lim
                    if over.any():
                        sl = self._slices["rho"]
                        rows = idx[over]
                        alpha = (self.limit * (1.0 - 2.0 * safety) / G[over])[:, None]
                        m = Xc[rows, sl].mean(axis=1, keepdims=True)
                        Xc[rows, sl] = m + alpha * (Xc[rows, sl] - m)
                        changed |= over
                live[idx] = changed
                live &= stuck < 2
            X[a0:a0 + chunk] = Xc
        return X

    def cheap_constraints_batch(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        """(g_gradient, g_cell_size) per design, g <= 0 feasible.

        g_cell_size = max(1 - min L / ((1 + margin) L_min), 1e-3 x number of valley cells): positive if any
        cell is too small OR lies on a pinch-neck valley."""
        F = self.fields_batch(X, ("rho", "w", "L"))
        ratio = np.min(F["L"][:, self._lat_flat] / self._lmin_batch(F["w"], F["rho"]), axis=1)
        n_valley = self._valley_batch(F["w"], F["rho"]).sum(axis=1)
        G = self._gradient_batch(F["rho"], F["L"])
        return G / self.limit - 1.0, np.maximum(1.0 - ratio, 1e-3 * n_valley)

    def cheap_feasible_batch(self, X: Any, chunk: int = 256) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        out = np.zeros(len(X), bool)
        for a0 in range(0, len(X), chunk):
            gg, gc = self.cheap_constraints_batch(X[a0:a0 + chunk])
            out[a0:a0 + chunk] = (gg <= 1e-9) & (gc <= 0.0)
        return out

    def cheap_constraints(self, x: Any, F: Mapping[str, np.ndarray] | None = None) -> dict[str, float]:
        """g <= 0 feasible: 'gradient' = G/limit - 1, 'cell_size' = 1 - min(L / L_min) (lattice cells)."""
        if F is not None:
            lat = self._lat
            ratio = float(np.min(F["L"][lat] / self._L_min(F)))
            n_valley = int(self._valley_mask(np.clip(F["w"][lat], 0, 1), np.clip(F["rho"][lat], 0.15, 0.55)).sum())
            G = grading_gradient_max(self.model, F["rho"], F["L"])
            return {"gradient": G / self.limit - 1.0, "cell_size": max(1.0 - ratio, 1e-3 * n_valley)}
        gg, gc = self.cheap_constraints_batch(np.asarray(x, float).ravel()[None])
        return {"gradient": float(gg[0]), "cell_size": float(gc[0])}

    def mass(self, F: Mapping[str, np.ndarray]) -> float:
        """Exact jacket mass (the model's formula; cheap - used as a known objective by the BO)."""
        cfg, g = self.cfg, self.grid
        jac, mat = cfg.jacket, cfg.material
        lat = self._lat
        m_lat = mat.density * g.gap * float(np.sum(F["rho"][lat] * g.area[lat]))
        r_s, t_sl = jac.stator_radius, jac.sleeve_thickness
        r_o, t_o = jac.outer_radius, jac.outer_wall_thickness
        L_ax = jac.axial_length
        m_w = mat.density * math.pi * (((r_s + t_sl) ** 2 - r_s**2) + (r_o**2 - (r_o - t_o) ** 2)) * L_ax
        return m_lat + m_w

    def describe(self) -> dict[str, Any]:
        """JSON description of the basis (enough to rebuild the fields without VoxLat)."""
        g = self.grid
        return {
            "spec": self.spec.to_dict(),
            "bounds": {n: list(self.bounds[n]) for n in FIELD_NAMES},
            "s_basis": "periodic uniform cubic B-spline; control i centred at theta_i = i*360/n_s deg "
                       "(theta = s / r_m from theta = 0); B3(t) = (4-6t^2+3|t|^3)/6 (|t|<1), (2-|t|)^3/6 (1<=|t|<2)",
            "z_basis": "clamped uniform B-spline of degree min(3, n_z-1) on z in [0, L_ax] (n_z=4: cubic Bezier, "
                       "n_z=2: linear, n_z=1: constant)",
            "value": "f = lo + (hi - lo) * sum_ij x_ij Bs_i(s) Bz_j(z), x_ij in [0, 1]",
            "r_mean_m": g.r_mean, "circumference_m": g.circumference, "axial_length_m": g.axial_length,
            "gradient_limit": self.limit,
        }
