"""Homogenized 2-D model of the lattice cooling jacket (Task 9).

The jacket annulus (lattice gap h between the stator sleeve and the outer wall) is
unwrapped to the plane (s, z), s = r_m theta the arc length at the lattice mean radius
r_m and z the axial coordinate, over the full circumference (periodic in s), and every
field is averaged over the gap depth h. Curvature is neglected (h / r_m = 0.08); the
planform area 2 pi r_m L_ax times h is exactly the annulus volume.

Lattice frame. The TPMS cell axes are (x, y, z) = (r, s, z) (Task 7 convention: x across
the gap, a_z stretches the axial direction), so the closure tensors of Task 6 are already
expressed in (r, s, z); blends (principal axes along the cube diagonals) keep their
off-diagonal components.

1. Flow - depth-averaged Darcy-Forchheimer (superficial velocity U = (U_s, U_z))
--------------------------------------------------------------------------------
    -grad p = mu (f_K K_t)^-1 U + rho C_F |U| U / sqrt(K_e),     div U = 0,

* K_t = K_tt - K_tr K_rr^-1 K_rt: the in-plane (s, z) permeability of the bulk 3x3 tensor
  under the constraint of zero net radial flow (Schur complement; = the (s, z) block for
  G/D, exact constraint for blends).
* f_K = 1 - a_K(theta)/N, N = h/L: the Task 7 finite-gap factor, applied to the viscous
  (Darcy) term only (STATUS Task 8, note 1).
* C_F: Gajetti et al. (2025) power laws (Task 8), K_e = 1 / (e . K_t^-1 . e) the BULK
  permeability along the local flow direction e = U/|U|.
Boundary conditions: inlet and outlet manifold slots (config ``manifolds``: centred at
theta_in, theta_out, arc width at r_m, centred axial extent) are plenums at uniform
pressure p_in (unknown) and 0; all other boundaries (axial ends z = 0, L_ax) are
no-flow. The total flow rate Q fixes p_in.
Discretization: cell-centred finite volumes on a tensor-product grid whose lines pass
through every manifold edge; two-point flux with harmonic mean of the face-normal
mobility, plus the cross term M_sz dp/dz at faces (implicit, central) for anisotropic
cells. The inertial term is linearized by Picard iteration on |U| and e (each step a
linear solve; p_in rescaled to the exact Q of the frozen system).

2. Heat - two-equation (local thermal non-equilibrium) model, depth-averaged
------------------------------------------------------------------------------
Coolant (T_f, per unit planform area):
    rho c_p h U . grad T_f - div(h phi k_f grad T_f) = U_eff (T_w - T_f)
Sleeve + lattice solid (T_w = sleeve temperature at the lattice face):
    -div(k_w t_w grad T_w) - div(h k_t grad Tbar_s) + U_eff (T_w - T_f) = q_in(z)
* The solid-phase equation of the two-equation model is solved analytically across the
  gap: with exchange H_v = h_sf a_sf, through-gap conductivity k_n = f_k k_eff,rr, bonded
  sleeve at T_w and an adiabatic outer wall, the lattice is a 1-D fin,
      T_s(r) - T_f = (T_w - T_f) cosh(m (h - r)) / cosh(m h),   m = sqrt(H_v / k_n),
  so the sleeve loses U_fin = sqrt(k_n H_v) tanh(m h) per unit area into the lattice and
  the depth-averaged solid temperature is Tbar_s = T_f + eta (T_w - T_f),
  eta = tanh(m h)/(m h) (fin efficiency). Exposed sleeve fraction phi (Task 7: planar
  porosity at a cut = bulk porosity) convects directly with h_sf (assumption):
      U_eff = sqrt(k_n H_v) tanh(m h) + phi h_sf.
  ``fin_model="isothermal"`` (eta = 1, U = H_v h + phi h_sf) is the perfect-fin bound.
* h_sf: Savoldi et al. (2026) modified Reynolds analogy, variant 1, times Pr^(1/3) (Task 8),
  with our bulk K_e, a_sf and C_F; k_eff from the Task 6 surrogate (two-phase value; the
  coolant adds 1-4 %, Task 2).
* In-plane conduction: sleeve k_s t_sl; lattice solid k_t = (k_ss, k_zz) h acting on
  Tbar_s (off-diagonal k ignored); coolant phi k_f h (stagnant, no dispersion).
* q_in(z) = q''(z) r_s / r_m: the config flux (defined on the stator radius r_s) per unit
  planform area at r_m; integrated over cells exactly, so sum(q_in A) = Q.
* Manifold slots: no lattice. The sleeve under a slot exchanges h_m (T_w - T_plenum) with
  a well-mixed plenum; h_m defaults to fully developed laminar flow between parallel plates
  with one wall heated, Nu_Dh = 5.385, D_h = 2 h (~180 W/m^2K, conservative). Plenum
  energy balances close the system; inflow enters at T_in.
* Axial ends adiabatic; first-order upwind advection; no diffusive flux across plenum faces.
Reported wall temperature: stator side of the sleeve, T_wall = T_w + q''(z) t_sl / k_s;
thermal resistance R = (max T_wall - T_in) / Q.

3. Structure - homogenized cylindrical sandwich with local C_eff
----------------------------------------------------------------
Sleeve and outer wall act as rigid faces; the lattice core carries torque and thrust in
shear with a uniform shear strain, so the local stress follows the local stiffness:
    tau_rs = T G_rs / (r_m sum(G_rs A)) (r_m/r_i)^2,   tau_rz = F G_rz / sum(G_rz A) (r_m/r_i),
G_rs = f_G C66, G_rz = f_G C55 (Voigt, (x,y,z) = (r,s,z)), evaluated at the sleeve radius
r_i (conservative; uniform core -> T/(2 pi r_i^2 L_ax)). Coolant pressure p (gauge + lattice
drop) is tied by the lattice in radial tension: sigma'_rr = p k_lat/(k_lat + k_hoop),
k_lat = f_C C11 / h, k_hoop = E t_o / R_o^2. Local p99 von Mises (Task 3 localization,
linear superposition, an upper bound):
    sigma_vm = K_ux sigma'_rr + sqrt(3) (K_sxy tau_rs + K_sxz tau_rz);  margin = sigma_allow / sigma_vm.
Outer wall under coolant pressure: hoop share (1 - share) p R/t, strip bending over the
manifold slot p b^2/(2 t^2), clamped-plate bending over a pore of radius L/2,
0.75 p (L/2t)^2. ``min_structural_margin`` = min over lattice and outer wall (>= 1 feasible).

4. Mass and manufacturability
-----------------------------
Mass = AlSi10Mg lattice (rho* h A over lattice cells) + sleeve + outer wall (exact annuli).
Flags: design bounds; L >= L_min(w, rho*) (Task 1 morphology table, a_z = 1 screening);
in-plane grading |grad rho*| L <= ``max_density_change_per_cell``; N = h/L >= 1 (finite-gap
model validity); fraction of lattice area in the pinch-neck region 0.3 < w < 0.7, rho* < 0.4
(surrogate errors ~10 %, Task 6).

Example
-------
>>> from voxlat.device import JacketModel
>>> model = JacketModel()                                          # doctest: +SKIP
>>> out = model.evaluate(rho=0.35, w=0.0, L=4e-3)                  # uniform gyroid, 3 L/min
>>> out["thermal_resistance"], out["delta_p"], out["pump_power"]   # doctest: +SKIP

Fields may be scalars, callables f(s, z) (s in [0, 2 pi r_m) measured from theta = 0 at
r_m, z in [0, L_ax], metres), arrays of the grid shape (n_s, n_z) in grid order, or any
other 2-D array, read as cell-centre values of a uniform grid over [0, 2 pi r_m) x [0, L_ax]
and interpolated (periodic in s).

References: Gajetti et al. 2025 (C_F), Savoldi et al. 2026 (Nu), see STATUS Task 8; fin
theory e.g. Incropera & DeWitt, Fundamentals of Heat and Mass Transfer, Ch. 3; the laminar
one-side-heated plate channel Nu = 5.385: Shah & London 1978; sandwich-core shear: Allen 1969.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from voxlat.utils.log import get_logger

__all__ = [
    "JacketGrid",
    "build_grid",
    "LocalClosures",
    "SurrogateClosures",
    "UniformClosures",
    "JacketOptions",
    "JacketFields",
    "JacketModel",
    "evaluate",
    "default_model",
    "fin_conductance",
    "manifold_htc_default",
    "cell_average_heat_flux",
    "in_plane_permeability",
    "directional_permeability",
    "uniform_flow_reference",
    "ntu_reference",
]

LOG = get_logger("device.jacket2d")

LATTICE, INLET, OUTLET = 0, 1, 2
NU_PLATE_ONE_SIDE = 5.385  # laminar, fully developed, one wall at uniform flux, other adiabatic (Shah & London 1978)
LOC_CASES = ("uniaxial_x", "shear_xy", "shear_xz")  # sigma_rr, tau_rs, tau_rz with (x, y, z) = (r, s, z)


# =============================================================================
# Grid
# =============================================================================
@dataclass(frozen=True)
class JacketGrid:
    """Tensor-product finite-volume grid on the unwrapped jacket (periodic in s).

    ``s_faces`` run over [0, C] in a shifted coordinate s' that starts at the left edge
    of the inlet slot; ``s_centres`` are the physical arc lengths (theta r_m, wrapped to
    [0, C)). ``kind[i, j]``: 0 lattice, 1 inlet plenum, 2 outlet plenum.
    """

    r_mean: float
    circumference: float
    axial_length: float
    gap: float
    s_shift: float  # physical s of s' = 0
    s_faces: np.ndarray  # (n_s + 1,) shifted coordinate
    z_faces: np.ndarray  # (n_z + 1,)
    kind: np.ndarray  # (n_s, n_z) int

    @property
    def n_s(self) -> int:
        return len(self.s_faces) - 1

    @property
    def n_z(self) -> int:
        return len(self.z_faces) - 1

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_s, self.n_z)

    @property
    def n_cells(self) -> int:
        return self.n_s * self.n_z

    @property
    def ds(self) -> np.ndarray:
        return np.diff(self.s_faces)

    @property
    def dz(self) -> np.ndarray:
        return np.diff(self.z_faces)

    @property
    def s_local(self) -> np.ndarray:
        """Cell centres in the shifted coordinate s' (monotone; for plotting)."""
        return 0.5 * (self.s_faces[1:] + self.s_faces[:-1])

    @property
    def s_centres(self) -> np.ndarray:
        """Physical arc length theta r_m of the cell centres, in [0, C)."""
        return np.mod(self.s_local + self.s_shift, self.circumference)

    @property
    def z_centres(self) -> np.ndarray:
        return 0.5 * (self.z_faces[1:] + self.z_faces[:-1])

    @property
    def area(self) -> np.ndarray:
        """Planform cell areas (n_s, n_z) [m^2] at r_m."""
        return np.outer(self.ds, self.dz)

    @property
    def lattice(self) -> np.ndarray:
        return self.kind == LATTICE

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """(S, Z) physical cell-centre coordinates, each (n_s, n_z)."""
        return np.meshgrid(self.s_centres, self.z_centres, indexing="ij")

    def theta_plot_deg(self) -> np.ndarray:
        """Cell-centre angle in degrees, continuous, inlet centre at its config angle."""
        return np.degrees((self.s_local + self.s_shift) / self.r_mean)

    def theta_faces_deg(self) -> np.ndarray:
        return np.degrees((self.s_faces + self.s_shift) / self.r_mean)


def _segment_faces(breaks: Sequence[float], spacing: float) -> np.ndarray:
    out = [breaks[0]]
    for a, b in zip(breaks[:-1], breaks[1:]):
        n = max(1, int(math.ceil((b - a) / spacing - 1e-9)))
        out.extend(list(np.linspace(a, b, n + 1)[1:]))
    return np.asarray(out, dtype=float)


def build_grid(cfg: Any, spacing: float = 2.5e-3, *, spacing_z: float | None = None) -> JacketGrid:
    """Grid with lines through every manifold edge; cell sizes <= ``spacing`` (s) / ``spacing_z`` (z)."""
    jac, man = cfg.jacket, cfg.manifolds
    r_m = jac.lattice_mean_radius
    C = 2.0 * math.pi * r_m
    L_ax = jac.axial_length
    width = man.width
    if not 0 < width < C / 2:
        raise ValueError("manifold width must be positive and below half the circumference")
    s_in = man.inlet_angle * r_m
    s_shift = s_in - width / 2.0
    s_out_local = np.mod(man.outlet_angle * r_m - s_shift, C)
    a, b = s_out_local - width / 2.0, s_out_local + width / 2.0
    if a < width - 1e-12 or b > C + 1e-12:
        raise ValueError("inlet and outlet manifold slots overlap")
    s_breaks = sorted({0.0, width, a, b, C})
    s_faces = _segment_faces(s_breaks, spacing)
    ext = min(man.axial_extent, L_ax)
    z0, z1 = (L_ax - ext) / 2.0, (L_ax + ext) / 2.0
    z_breaks = sorted({0.0, z0, z1, L_ax})
    z_faces = _segment_faces(z_breaks, spacing_z or spacing)
    sc = 0.5 * (s_faces[1:] + s_faces[:-1])
    zc = 0.5 * (z_faces[1:] + z_faces[:-1])
    in_z = (zc > z0) & (zc < z1)
    kind = np.zeros((len(sc), len(zc)), dtype=np.int8)
    kind[np.ix_(sc < width, in_z)] = INLET
    kind[np.ix_((sc > a) & (sc < b), in_z)] = OUTLET
    return JacketGrid(r_mean=r_m, circumference=C, axial_length=L_ax, gap=jac.lattice_gap, s_shift=s_shift,
                      s_faces=s_faces, z_faces=z_faces, kind=kind)


def cell_average_heat_flux(cfg: Any, z_faces: np.ndarray) -> np.ndarray:
    """Exact cell average of q''(z) (config profile, W/m^2 on the stator radius) over z cells."""
    L_ax = cfg.jacket.axial_length
    if cfg.heat_flux.profile != "end_peaked_quadratic":
        raise ValueError(f"unsupported heat-flux profile {cfg.heat_flux.profile!r}")
    a = cfg.heat_flux.amplitude
    zeta = np.asarray(z_faces) / L_ax - 0.5
    mean_sq = (zeta[1:] ** 3 - zeta[:-1] ** 3) / (3.0 * (zeta[1:] - zeta[:-1]))
    return cfg.mean_heat_flux * (1.0 + a * (12.0 * mean_sq - 1.0))


# =============================================================================
# Closures
# =============================================================================
@dataclass
class LocalClosures:
    """Per-cell closures (n cells). Tensors in the cell frame (x, y, z) = (r, s, z), SI."""

    K: np.ndarray  # (n, 3, 3) bulk Darcy permeability [m^2]
    k_eff: np.ndarray  # (n, 3, 3) effective conductivity [W/mK]
    C_eff: np.ndarray  # (n, 6, 6) Voigt stiffness [Pa]
    a_sf: np.ndarray  # (n,) [1/m]
    loc: dict[str, np.ndarray]  # p99 localization factors for LOC_CASES
    f_K: np.ndarray  # (n,) finite-gap factors (1 = none)
    f_k: np.ndarray
    f_C: np.ndarray
    f_G: np.ndarray


class SurrogateClosures:
    """Task 6 ``ClosureModel`` (bulk closures) x Task 7 ``FiniteGapCorrection`` factors.

    ``finite_gap``: quantities whose correction is applied (subset of K_t, k_n, C_nn, G_t;
    True = all four, False = none). Evaluations are deduplicated over identical cells.
    """

    QUANTITIES = ("K_t", "k_n", "C_nn", "G_t")

    def __init__(self, backend: str = "ensemble", *, finite_gap: bool | Sequence[str] = True,
                 check_bounds: str = "warn", model: Any = None, correction: Any = None):
        from voxlat.homogenization.finite_gap import FiniteGapCorrection
        from voxlat.surrogates import ClosureModel

        self.model = model if model is not None else ClosureModel.load(backend)
        if finite_gap is True:
            fg: tuple[str, ...] = self.QUANTITIES
        elif finite_gap is False:
            fg = ()
        else:
            fg = tuple(finite_gap)
            bad = set(fg) - set(self.QUANTITIES)
            if bad:
                raise ValueError(f"unknown finite-gap quantities {bad}")
        self.finite_gap = fg
        self.correction = correction if correction is not None or not fg else FiniteGapCorrection.load()
        self.check_bounds = check_bounds

    @property
    def material(self) -> dict[str, float]:
        return dict(self.model.material)

    def __call__(self, w: np.ndarray, rho: np.ndarray, a_z: np.ndarray, L: np.ndarray, gap: float) -> LocalClosures:
        X = np.column_stack([w, rho, a_z, L])
        uniq, inv = np.unique(X, axis=0, return_inverse=True)
        inv = inv.ravel()
        out = self.model.predict(uniq[:, :3], uniq[:, 3], return_std=False, check_bounds=self.check_bounds)

        def g(name: str) -> np.ndarray:
            return np.asarray(out[name][0], dtype=float)[inv]  # (n_unique, ...) -> (n_cells, ...)

        facs = {}
        N = gap / uniq[:, 3]
        for q in self.QUANTITIES:
            if q in self.finite_gap:
                facs[q] = np.asarray(self.correction.factor(q, N, uniq[:, 0], uniq[:, 1], uniq[:, 2]), float)[inv]
            else:
                facs[q] = np.ones(len(inv))
        return LocalClosures(K=g("K"), k_eff=g("k_eff"), C_eff=g("C_eff"), a_sf=g("a_sf"),
                             loc={c: g(f"loc_{c}_p99") for c in LOC_CASES},
                             f_K=facs["K_t"], f_k=facs["k_n"], f_C=facs["C_nn"], f_G=facs["G_t"])


@dataclass
class UniformClosures:
    """Constant closures for verification (independent of theta and L)."""

    K: Any  # scalar (isotropic) or (3, 3)
    k_eff: Any  # scalar or (3, 3)
    a_sf: float
    C_eff: Any = None  # (6, 6); default: isotropic-like cubic placeholder from E = 7 GPa
    loc: Mapping[str, float] = field(default_factory=lambda: {c: 10.0 for c in LOC_CASES})
    f_K: float = 1.0
    f_k: float = 1.0
    f_C: float = 1.0
    f_G: float = 1.0
    material: dict[str, float] = field(default_factory=dict)

    def __call__(self, w: np.ndarray, rho: np.ndarray, a_z: np.ndarray, L: np.ndarray, gap: float) -> LocalClosures:
        n = len(w)

        def t3(x: Any) -> np.ndarray:
            x = np.asarray(x, float)
            return np.broadcast_to(x * np.eye(3) if x.ndim == 0 else x, (n, 3, 3)).copy()

        if self.C_eff is None:
            from voxlat.homogenization.elasticity import isotropic_stiffness

            C = isotropic_stiffness(7.0e9, 0.33)
        else:
            C = np.asarray(self.C_eff, float)
        one = np.ones(n)
        return LocalClosures(K=t3(self.K), k_eff=t3(self.k_eff), C_eff=np.broadcast_to(C, (n, 6, 6)).copy(),
                             a_sf=np.full(n, float(self.a_sf)), loc={c: np.full(n, float(self.loc[c])) for c in LOC_CASES},
                             f_K=self.f_K * one, f_k=self.f_k * one, f_C=self.f_C * one, f_G=self.f_G * one)


def in_plane_permeability(K: np.ndarray) -> np.ndarray:
    """(s, z) permeability under zero net radial flow: Schur complement K_tt - K_tr K_rr^-1 K_rt."""
    K = np.asarray(K, float)
    Ktt = K[..., 1:, 1:]
    Ktr = K[..., 1:, :1]
    return Ktt - Ktr @ np.swapaxes(Ktr, -1, -2) / K[..., :1, :1]


def directional_permeability(Kt: np.ndarray, e: np.ndarray) -> np.ndarray:
    """K_e = 1 / (e . Kt^-1 . e) for unit vectors e (..., 2)."""
    inv = np.linalg.inv(Kt)
    return 1.0 / np.einsum("...i,...ij,...j->...", e, inv, e)


def fin_conductance(k_n: Any, H_v: Any, gap: float, model: str = "fin") -> tuple[np.ndarray, np.ndarray]:
    """Wall-to-coolant conductance of the lattice per unit sleeve area and fin efficiency.

    1-D fin across the gap (base at the sleeve, adiabatic tip at the outer wall):
    U = sqrt(k_n H_v) tanh(m h), eta = tanh(m h)/(m h), m = sqrt(H_v/k_n).
    ``model="isothermal"``: U = H_v h, eta = 1 (perfect-fin bound).
    """
    k = np.asarray(k_n, float)
    H = np.asarray(H_v, float)
    if model == "isothermal":
        return H * gap, np.ones(np.broadcast(k, H).shape)
    if model != "fin":
        raise ValueError(f"fin model must be 'fin' or 'isothermal', got {model!r}")
    mh = np.sqrt(H / k) * gap
    with np.errstate(invalid="ignore", divide="ignore"):
        eta = np.where(mh > 1e-8, np.tanh(mh) / np.where(mh > 1e-8, mh, 1.0), 1.0 - mh**2 / 3.0)
    return H * gap * eta, eta


def manifold_htc_default(cfg: Any) -> float:
    """h_m = Nu k_f / D_h with Nu = 5.385 (laminar plate channel, one wall heated), D_h = 2 h."""
    return NU_PLATE_ONE_SIDE * cfg.coolant.conductivity / (2.0 * cfg.jacket.lattice_gap)


# =============================================================================
# Options and results
# =============================================================================
@dataclass(frozen=True)
class JacketOptions:
    """Numerical and modelling options (defaults = production settings)."""

    spacing: float = 1.25e-3  # target cell size in s [m] (grid study: STATUS Task 9)
    spacing_z: float | None = 2.5e-3  # target cell size in z [m] (None = spacing)
    backend: str = "ensemble"  # Task 6 surrogate
    finite_gap: bool | tuple[str, ...] = True  # Task 7 corrections (K_t, k_n, C_nn, G_t)
    check_bounds: str = "warn"  # surrogate training-box check: warn | raise | ignore
    cf_source: str = "gajetti2025"  # Task 8 Forchheimer source
    inertia: bool = True  # include the Forchheimer term
    nu_variant: int = 1  # Savoldi 2026 variant (2 = low-h bound)
    pr_exponent: float = 1.0 / 3.0
    fin_model: str = "fin"  # "fin" | "isothermal"
    wall_exposed_convection: bool = True  # phi h_sf on the exposed sleeve
    manifold_htc: float | None = None  # W/m^2K; None -> manifold_htc_default
    sleeve_conduction: bool = True
    lattice_conduction: bool = True  # in-plane solid conduction of the lattice
    fluid_conduction: bool = True  # in-plane stagnant coolant conduction
    wall_condition: str = "flux"  # "flux" (q''(z) from config) | "temperature" (verification)
    wall_temperature: float | None = None  # K, for wall_condition="temperature"
    picard_tol: float = 1e-8  # relative change of dp
    picard_maxiter: int = 200
    picard_relaxation: float = 0.7  # under-relaxation of |U| and e (0.7: fewest iterations on graded designs)
    gradient_limit: float | None = None  # None -> cfg.manufacturing.max_density_change_per_cell


@dataclass
class JacketFields:
    """Cell fields of one evaluation (arrays (n_s, n_z); NaN outside the lattice where undefined)."""

    grid: JacketGrid
    rho: np.ndarray
    w: np.ndarray
    L: np.ndarray
    a_z: np.ndarray
    pressure: np.ndarray  # Pa (plenums: their pressure)
    U_s: np.ndarray  # superficial velocity components [m/s]
    U_z: np.ndarray
    speed: np.ndarray
    T_f: np.ndarray  # coolant (plenum cells: plenum temperature) [K]
    T_w: np.ndarray  # sleeve at the lattice face [K]
    T_wall: np.ndarray  # sleeve, stator side [K]
    T_s_mean: np.ndarray  # depth-averaged lattice solid [K]
    q_in: np.ndarray  # heat input per planform area [W/m^2]
    U_eff: np.ndarray  # wall-to-coolant conductance [W/m^2K] (plenum cells: h_m)
    h_sf: np.ndarray
    eta: np.ndarray
    reynolds: np.ndarray
    forchheimer: np.ndarray  # Fo = inertial / viscous resistance
    K_inplane: np.ndarray  # (n_s, n_z, 2, 2) corrected in-plane permeability
    sigma_vm: np.ndarray  # p99 local von Mises [Pa]
    margin: np.ndarray  # sigma_allow / sigma_vm
    face_flow_s: np.ndarray  # (n_s, n_z) volumetric flow through the +s face of each cell [m^3/s]
    face_flow_z: np.ndarray  # (n_s, n_z - 1) through the +z face

    def to_dict(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "grid"}


# =============================================================================
# Field input
# =============================================================================
def _resolve_field(spec: Any, grid: JacketGrid, name: str) -> np.ndarray:
    shape = grid.shape
    if callable(spec):
        S, Z = grid.meshgrid()
        val = np.asarray(spec(S, Z), dtype=float)
        val = np.broadcast_to(val, shape).astype(float)
    else:
        arr = np.asarray(spec, dtype=float)
        if arr.ndim == 0:
            val = np.full(shape, float(arr))
        elif arr.shape == shape:
            val = arr.astype(float).copy()
        elif arr.ndim == 2:
            from scipy.interpolate import RegularGridInterpolator

            m_s, m_z = arr.shape
            C, L_ax = grid.circumference, grid.axial_length
            sc = (np.arange(m_s) + 0.5) * C / m_s
            zc = (np.arange(m_z) + 0.5) * L_ax / m_z
            sp_ = np.concatenate([[sc[-1] - C], sc, [sc[0] + C]])
            ap = np.concatenate([arr[-1:], arr, arr[:1]], axis=0)
            zp = np.concatenate([[0.0], zc, [L_ax]]) if m_z > 1 else np.array([0.0, L_ax])
            ap = np.concatenate([ap[:, :1], ap, ap[:, -1:]], axis=1) if m_z > 1 else np.concatenate([ap, ap], axis=1)
            itp = RegularGridInterpolator((sp_, zp), ap)
            S, Z = grid.meshgrid()
            val = itp(np.stack([S, Z], axis=-1))
        else:
            raise ValueError(f"field {name!r}: expected scalar, callable or 2-D array, got shape {arr.shape}")
    if not np.all(np.isfinite(val)):
        raise ValueError(f"field {name!r} has non-finite values")
    return val


# =============================================================================
# The model
# =============================================================================
class JacketModel:
    """Homogenized jacket device model. ``evaluate`` returns the Task 9 output dict."""

    def __init__(self, cfg: Any = None, options: JacketOptions | None = None, closures: Any = None):
        if cfg is None:
            from voxlat.utils.config import load_config

            cfg = load_config()
        self.cfg = cfg
        self.options = options or JacketOptions()
        o = self.options
        self.grid = build_grid(cfg, o.spacing, spacing_z=o.spacing_z)
        self.closures = closures if closures is not None else SurrogateClosures(
            o.backend, finite_gap=o.finite_gap, check_bounds=o.check_bounds)
        self._build_topology()
        self._q_cell = cell_average_heat_flux(cfg, self.grid.z_faces)  # W/m^2 on r_s, per z cell

    def with_options(self, **kw: Any) -> "JacketModel":
        """Copy with changed options (closures reused when the grid-independent settings allow)."""
        opts = replace(self.options, **kw)
        same_closures = all(getattr(opts, k) == getattr(self.options, k) for k in ("backend", "finite_gap", "check_bounds"))
        return JacketModel(self.cfg, opts, self.closures if same_closures else None)

    # ------------------------------------------------------------------ topology
    def _build_topology(self) -> None:
        g = self.grid
        ns, nz = g.shape
        idx = np.arange(g.n_cells).reshape(ns, nz)
        kind = g.kind.ravel()
        ds, dz = g.ds, g.dz
        h = g.gap
        # s-faces: (i, j) -> (i+1, j), periodic
        sl = idx.ravel()
        sr = np.roll(idx, -1, axis=0).ravel()
        s_dl = np.repeat(ds / 2.0, nz)
        s_dr = np.repeat(np.roll(ds, -1) / 2.0, nz)
        s_len = np.tile(dz, ns)
        # z-faces: (i, j) -> (i, j+1)
        zl = idx[:, :-1].ravel()
        zr = idx[:, 1:].ravel()
        z_dl = np.tile(dz[:-1] / 2.0, ns)
        z_dr = np.tile(dz[1:] / 2.0, ns)
        z_len = np.repeat(ds, nz - 1)
        self._faces = {
            "s": dict(l=sl, r=sr, dl=s_dl, dr=s_dr, length=s_len, comp=0),
            "z": dict(l=zl, r=zr, dl=z_dl, dr=z_dr, length=z_len, comp=1),
        }
        for f in self._faces.values():
            f["kl"] = kind[f["l"]]
            f["kr"] = kind[f["r"]]
            f["LL"] = (f["kl"] == LATTICE) & (f["kr"] == LATTICE)
            f["LP"] = (f["kl"] == LATTICE) & (f["kr"] != LATTICE)
            f["PL"] = (f["kl"] != LATTICE) & (f["kr"] == LATTICE)
            f["area"] = h * f["length"]
        self._n_sfaces = len(sl)
        self._lat_idx = np.flatnonzero(kind == LATTICE)
        self._lat_pos = -np.ones(g.n_cells, dtype=np.int64)
        self._lat_pos[self._lat_idx] = np.arange(len(self._lat_idx))
        self._kind = kind
        self._idx = idx
        # divergence operator D (cells x faces): +1 for the left cell, -1 for the right
        nf = len(sl) + len(zl)
        rows = np.concatenate([sl, sr, zl, zr])
        cols = np.concatenate([np.arange(len(sl)), np.arange(len(sl)), len(sl) + np.arange(len(zl)),
                               len(sl) + np.arange(len(zl))])
        vals = np.concatenate([np.ones(len(sl)), -np.ones(len(sl)), np.ones(len(zl)), -np.ones(len(zl))])
        self._D = sp.csr_matrix((vals, (rows, cols)), shape=(g.n_cells, nf))
        self._gradient_ops()

    def _gradient_ops(self) -> None:
        """Cell-centred tangential gradient operators Gs, Gz (rows of lattice cells).

        Neighbour: lattice -> its centre; plenum -> the shared face (uniform plenum pressure);
        no-flow boundary -> one-sided with the cell itself.
        """
        g = self.grid
        ns, nz = g.shape
        idx = self._idx
        kind2 = g.kind
        sc_local = g.s_local
        zc = g.z_centres
        ds, dz = g.ds, g.dz
        C = g.circumference

        def build(axis: str) -> sp.csr_matrix:
            rows, cols, vals = [], [], []
            I, J = np.meshgrid(np.arange(ns), np.arange(nz), indexing="ij")
            P = idx
            lat = kind2 == LATTICE
            if axis == "z":
                plus_exists = J + 1 < nz
                minus_exists = J - 1 >= 0
                Jp = np.minimum(J + 1, nz - 1)
                Jm = np.maximum(J - 1, 0)
                Np_, Nm_ = idx[I, Jp], idx[I, Jm]
                d_plus_c = np.where(plus_exists, zc[Jp] - zc[J], 0.0)
                d_minus_c = np.where(minus_exists, zc[J] - zc[Jm], 0.0)
                half = dz[J] / 2.0
                kp = kind2[I, Jp]
                km = kind2[I, Jm]
            else:
                Ip = (I + 1) % ns
                Im = (I - 1) % ns
                plus_exists = np.ones_like(lat)
                minus_exists = np.ones_like(lat)
                Np_, Nm_ = idx[Ip, J], idx[Im, J]
                d_plus_c = np.mod(sc_local[Ip] - sc_local[I], C)
                d_minus_c = np.mod(sc_local[I] - sc_local[Im], C)
                half = ds[I] / 2.0
                kp = kind2[Ip, J]
                km = kind2[Im, J]
            d_plus = np.where(~plus_exists, 0.0, np.where(kp == LATTICE, d_plus_c, half))
            d_minus = np.where(~minus_exists, 0.0, np.where(km == LATTICE, d_minus_c, half))
            col_plus = np.where(plus_exists, Np_, P)
            col_minus = np.where(minus_exists, Nm_, P)
            den = d_plus + d_minus
            ok = lat & (den > 0)
            inv = np.where(ok, 1.0 / np.where(den > 0, den, 1.0), 0.0)
            rows = np.concatenate([P[ok], P[ok]])
            cols = np.concatenate([col_plus[ok], col_minus[ok]])
            vals = np.concatenate([inv[ok], -inv[ok]])
            return sp.csr_matrix((vals, (rows, cols)), shape=(g.n_cells, g.n_cells))

        self._Gs = build("s")
        self._Gz = build("z")

    # ------------------------------------------------------------------ public API
    def evaluate(self, rho: Any = 0.35, w: Any = 0.0, L: Any = 4e-3, a_z: Any = 1.0, flow_rate: float | None = None,
                 *, return_fields: bool = False) -> dict[str, Any]:
        """Evaluate one design. Fields: scalar | callable(s, z) | array (see module docstring).

        Returns a dict of scalars (SI) - thermal_resistance [K/W], delta_p [Pa], pump_power [W],
        mass [kg], min_structural_margin [-], manufacturable [bool], ... - plus ``fields``
        (``JacketFields``) when ``return_fields``.
        """
        t0 = time.perf_counter()
        cfg, o, g = self.cfg, self.options, self.grid
        Q = float(cfg.operating.nominal_flow_rate if flow_rate is None else flow_rate)
        if Q <= 0:
            raise ValueError("flow rate must be positive")
        F = {n: _resolve_field(v, g, n) for n, v in (("rho", rho), ("w", w), ("L", L), ("a_z", a_z))}
        lat = self._lat_idx
        props = self._local_properties({k: v.ravel()[lat] for k, v in F.items()})
        t1 = time.perf_counter()
        flow = self._solve_flow(props, Q)
        t2 = time.perf_counter()
        heat = self._solve_heat(props, flow, Q)
        t3 = time.perf_counter()
        struct = self._structure(props, F, flow)
        mass = self._mass(F)
        manu = self._manufacturability(F)
        t4 = time.perf_counter()

        Qh = cfg.motor.heat_to_jacket
        T_in = cfg.operating.inlet_temperature
        out: dict[str, Any] = {
            "thermal_resistance": (heat["T_wall_max"] - T_in) / Qh,
            "T_wall_max": heat["T_wall_max"],
            "T_wall_max_lattice": heat["T_wall_max_lattice"],
            "T_wall_max_s": heat["argmax_s"],
            "T_wall_max_z": heat["argmax_z"],
            "T_wall_mean": heat["T_wall_mean"],
            "T_in": T_in,
            "T_out": heat["T_out"],
            "heat_input": heat["heat_input"],
            "delta_p": flow["dp"],
            "pump_power": flow["dp"] * Q / cfg.operating.pump_efficiency,
            "hydraulic_power": flow["dp"] * Q,
            "flow_rate": Q,
            "mass": mass["total"],
            "mass_lattice": mass["lattice"],
            "mass_walls": mass["walls"],
            "coolant_volume": mass["coolant_volume"],
            "min_structural_margin": min(struct["lattice_margin"], struct["outer_wall_margin"]),
            "lattice_margin": struct["lattice_margin"],
            "outer_wall_margin": struct["outer_wall_margin"],
            "structural": struct["summary"],
            "manufacturable": manu["manufacturable"],
            "manufacturability": manu,
            "mass_balance_error": flow["mass_balance_error"],
            "max_cell_divergence": flow["max_cell_divergence"],
            "energy_balance_error": heat["energy_balance_error"],
            "picard_iterations": flow["iterations"],
            "picard_converged": flow["converged"],
            "diagnostics": {
                "flow_split": flow["flow_split"],
                "reynolds_range": (float(np.min(heat["re"])), float(np.max(heat["re"]))),
                "reynolds_median": float(np.median(heat["re"])),
                "forchheimer_number_range": (float(np.min(flow["Fo"])), float(np.max(flow["Fo"]))),
                "h_sf_mean": float(np.mean(heat["h_sf"])),
                "U_eff_mean": float(np.average(heat["U_eff"], weights=g.area.ravel()[lat])),
                "fin_efficiency_mean": float(np.mean(heat["eta"])),
                "heat_to_manifold_plenums": heat["heat_manifolds"],
                "manifold_htc": heat["h_m"],
                "finite_gap_factor_K_mean": float(np.mean(props["f_K"])),
                "grid": (g.n_s, g.n_z),
                "timings": {"closures": t1 - t0, "flow": t2 - t1, "heat": t3 - t2, "rest": t4 - t3},
            },
            "wall_time": time.perf_counter() - t0,
        }
        if return_fields:
            out["fields"] = self._fields(F, props, flow, heat, struct)
        return out

    # ------------------------------------------------------------------ closures
    def _local_properties(self, Fl: Mapping[str, np.ndarray]) -> dict[str, Any]:
        cfg, o, g = self.cfg, self.options, self.grid
        cl: LocalClosures = self.closures(Fl["w"], Fl["rho"], Fl["a_z"], Fl["L"], g.gap)
        from voxlat.closures.empirical import forchheimer_coefficient

        phi = 1.0 - Fl["rho"]
        Kt = in_plane_permeability(cl.K)  # bulk (n, 2, 2)
        Kt_eff = Kt * cl.f_K[:, None, None]
        Rv = cfg.coolant.dynamic_viscosity * np.linalg.inv(Kt_eff)
        cf = forchheimer_coefficient(phi, np.clip(Fl["w"], 0.0, 1.0), source=o.cf_source, on_extrapolation="ignore") \
            if o.inertia else np.zeros_like(phi)
        return dict(cl=cl, phi=phi, w=Fl["w"], rho=Fl["rho"], L=Fl["L"], a_z=Fl["a_z"], Kt=Kt, Kt_eff=Kt_eff, Rv=Rv,
                    cf=np.asarray(cf, float), f_K=cl.f_K, k_n=cl.k_eff[:, 0, 0] * cl.f_k,
                    k_t=np.stack([cl.k_eff[:, 1, 1], cl.k_eff[:, 2, 2]], axis=-1), a_sf=cl.a_sf)

    # ------------------------------------------------------------------ flow
    def _flow_operator(self, M: np.ndarray) -> sp.csr_matrix:
        """Face-flux operator F (faces x cells): flux = F @ p [m^3/s], positive left -> right."""
        g = self.grid
        n = g.n_cells
        Mfull = np.zeros((n, 2, 2))
        Mfull[self._lat_idx] = M
        cross_any = np.any(np.abs(M[:, 0, 1]) > 1e-14 * np.abs(M[:, 0, 0]))
        blocks = []
        for key, Gt in (("s", self._Gz), ("z", self._Gs)):
            f = self._faces[key]
            c = f["comp"]
            l, r, dl, dr, A = f["l"], f["r"], f["dl"], f["dr"], f["area"]
            Ml, Mr = Mfull[l, c, c], Mfull[r, c, c]
            T = np.zeros(len(l))
            LL, LP, PL = f["LL"], f["LP"], f["PL"]
            T[LL] = A[LL] / (dl[LL] / Ml[LL] + dr[LL] / Mr[LL])
            T[LP] = A[LP] * Ml[LP] / dl[LP]
            T[PL] = A[PL] * Mr[PL] / dr[PL]
            nf = len(l)
            fi = np.arange(nf)
            Fk = sp.csr_matrix((np.concatenate([T, -T]), (np.concatenate([fi, fi]), np.concatenate([l, r]))),
                               shape=(nf, n))
            if cross_any:
                cx = np.zeros(nf)
                cx[LL] = A[LL] * 0.5 * (Mfull[l[LL], c, 1 - c] + Mfull[r[LL], c, 1 - c])
                Sl = sp.csr_matrix((np.ones(nf), (fi, l)), shape=(nf, n))
                Sr = sp.csr_matrix((np.ones(nf), (fi, r)), shape=(nf, n))
                Fk = Fk - sp.diags(0.5 * cx) @ ((Sl + Sr) @ Gt)
            blocks.append(Fk)
        return sp.vstack(blocks).tocsr()

    def _plenum_face_sign(self, plenum: int) -> np.ndarray:
        """+1 / -1 on faces that carry flow from the given plenum into the lattice (0 elsewhere)."""
        sgn = []
        for key in ("s", "z"):
            f = self._faces[key]
            sgn.append(np.where(f["kl"] == plenum, 1.0, 0.0) * f["PL"] - np.where(f["kr"] == plenum, 1.0, 0.0) * f["LP"])
        return np.concatenate(sgn)

    def _solve_flow(self, props: Mapping[str, Any], Q: float) -> dict[str, Any]:
        o, cfg, g = self.options, self.cfg, self.grid
        rho_f = cfg.coolant.density
        lat = self._lat_idx
        kind = self._kind
        n = g.n_cells
        Rv, Kt, cf = props["Rv"], props["Kt"], props["cf"]
        D = self._D
        p_known = np.where(kind == INLET, 1.0, 0.0)
        is_lat = kind == LATTICE
        sign_in = self._plenum_face_sign(INLET)
        sign_out = self._plenum_face_sign(OUTLET)
        beta = np.zeros(len(lat))
        speed = np.zeros(len(lat))
        e = np.zeros((len(lat), 2))
        e[:, 0] = 1.0
        dp_old = None
        converged = False
        it = 0
        for it in range(1, o.picard_maxiter + 1):
            R = Rv + beta[:, None, None] * np.eye(2)
            M = np.linalg.inv(R)
            Fop = self._flow_operator(M)
            A = (D @ Fop).tocsr()
            A_LL = A[lat][:, lat]
            rhs = -(A[lat][:, ~is_lat] @ p_known[~is_lat])
            pL = spla.spsolve(A_LL.tocsc(), rhs)
            p = p_known.copy()
            p[lat] = pL
            q = Fop @ p
            Q1 = float(sign_in @ q)
            scale = Q / Q1
            p *= scale
            q *= scale
            dp = scale
            # cell-centred superficial velocity
            Us, Uz = self._cell_velocity(q)
            u_new = np.hypot(Us[lat], Uz[lat])
            if not o.inertia or not np.any(cf > 0):
                converged = True
                speed = u_new
                break
            with np.errstate(invalid="ignore", divide="ignore"):
                e_new = np.where(u_new[:, None] > 0, np.column_stack([Us[lat], Uz[lat]]) / np.where(
                    u_new > 0, u_new, 1.0)[:, None], e)
            w_ = o.picard_relaxation if it > 1 else 1.0
            speed = speed + w_ * (u_new - speed)
            e = e + w_ * (e_new - e)
            e /= np.maximum(np.linalg.norm(e, axis=1), 1e-300)[:, None]
            K_e = directional_permeability(Kt, e)
            beta = rho_f * cf * speed / np.sqrt(K_e)
            if dp_old is not None and abs(dp - dp_old) <= o.picard_tol * abs(dp):
                converged = True
                break
            dp_old = dp
        if not converged:
            LOG.warning("Picard iteration did not converge in %d iterations (last rel. change %.2e)", it,
                        abs(dp - dp_old) / abs(dp) if dp_old else float("nan"))
        div = D @ q
        q_in = float(sign_in @ q)
        q_out = -float(sign_out @ q)
        K_e = directional_permeability(Kt, e)
        Rv_e = cfg.coolant.dynamic_viscosity / directional_permeability(props["Kt_eff"], e)
        Fo = rho_f * cf * speed / np.sqrt(K_e) / Rv_e
        # flow split between the two arcs inlet -> outlet: circumferential flow through the s-face
        # closest to the middle of each lattice arc (+s direction from the inlet, and -s direction)
        plus, minus = self._arc_flows(q)
        return dict(p=p, q=q, dp=dp, Us=Us, Uz=Uz, speed=speed, e=e, K_e=K_e, beta=beta, Fo=Fo, iterations=it,
                    converged=converged, mass_balance_error=abs(q_in - q_out) / Q,
                    max_cell_divergence=float(np.max(np.abs(div[lat]))) / Q,
                    flow_split=(plus / Q, minus / Q))

    def _arc_flows(self, q: np.ndarray) -> tuple[float, float]:
        """Flow along +s on the arc after the inlet and along -s on the arc before it [m^3/s]."""
        g = self.grid
        nz = g.n_z
        sf = g.s_faces
        in_s = np.flatnonzero(np.any(g.kind == INLET, axis=1))
        out_s = np.flatnonzero(np.any(g.kind == OUTLET, axis=1))
        a_end, b_start = sf[in_s.max() + 1], sf[out_s.min()]  # arc 1: inlet -> outlet along +s
        c_end, d_start = sf[out_s.max() + 1], sf[-1]  # arc 2: outlet -> (inlet + C)
        qs = q[: self._n_sfaces].reshape(g.n_s, nz)  # flux through the +s face of cell i
        i1 = int(np.argmin(np.abs(sf[1:] - 0.5 * (a_end + b_start))))
        i2 = int(np.argmin(np.abs(sf[1:] - 0.5 * (c_end + d_start))))
        return float(np.sum(qs[i1])), float(-np.sum(qs[i2]))

    def _cell_velocity(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        g = self.grid
        ns, nz = g.shape
        h = g.gap
        nsf = self._n_sfaces
        qs = q[:nsf].reshape(ns, nz)  # +s face of cell (i, j)
        qz = q[nsf:].reshape(ns, nz - 1)
        Us = 0.5 * (qs + np.roll(qs, 1, axis=0)) / (h * g.dz[None, :])
        qz_full = np.zeros((ns, nz + 1))
        qz_full[:, 1:-1] = qz
        Uz = 0.5 * (qz_full[:, 1:] + qz_full[:, :-1]) / (h * g.ds[:, None])
        return Us.ravel(), Uz.ravel()

    # ------------------------------------------------------------------ heat
    def _solve_heat(self, props: Mapping[str, Any], flow: Mapping[str, Any], Q: float) -> dict[str, Any]:
        from voxlat.closures.empirical import interstitial_heat_transfer

        o, cfg, g = self.options, self.cfg, self.grid
        cool, mat, jac = cfg.coolant, cfg.material, cfg.jacket
        rcp = cool.density * cool.specific_heat
        h = g.gap
        n = g.n_cells
        lat = self._lat_idx
        nL = len(lat)
        kind = self._kind
        area = g.area.ravel()
        # --- local exchange
        ht = interstitial_heat_transfer(
            flow["speed"], props["phi"], props["a_sf"], flow["K_e"], w=np.clip(props["w"], 0, 1),
            conductivity=cool.conductivity, kinematic_viscosity=cool.kinematic_viscosity, prandtl=cool.prandtl,
            cf=props["cf"] if o.inertia else None, cf_source=o.cf_source, variant=o.nu_variant,
            pr_exponent=o.pr_exponent, on_extrapolation="ignore")
        h_sf = np.asarray(ht.h_sf, float)
        H_v = h_sf * props["a_sf"]  # volumetric solid-fluid exchange [W/m^3K]
        U_fin, eta = fin_conductance(props["k_n"], H_v, h, o.fin_model)
        U_dir = props["phi"] * h_sf if o.wall_exposed_convection else np.zeros(nL)
        U_eff = U_fin + U_dir  # wall-to-coolant conductance without in-plane conduction
        G_sf = H_v * h  # solid -> fluid, per planform area
        # sleeve -> lattice solid, calibrated so that the uncoupled limit is the exact fin:
        # T_s - T_f = eta (T_w - T_f) and G_ws (T_w - T_s) = U_fin (T_w - T_f)
        G_ws = U_fin / np.maximum(1.0 - eta, 1e-9)
        h_m = manifold_htc_default(cfg) if o.manifold_htc is None else float(o.manifold_htc)
        # --- heat input per planform area at r_m
        q_on_rs = np.tile(self._q_cell, g.n_s)  # (n,) W/m^2 on the stator radius
        q_in = q_on_rs * jac.stator_radius / g.r_mean
        # --- unknowns: T_f (lattice) | T_s (lattice) | T_w (all cells) | plenums (INLET, OUTLET)
        iF = lambda c: self._lat_pos[c]  # noqa: E731
        iS = lambda c: nL + self._lat_pos[c]  # noqa: E731
        iW = lambda c: 2 * nL + np.asarray(c)  # noqa: E731
        plen_ids = (INLET, OUTLET)
        iP = {k: 2 * nL + n + m for m, k in enumerate(plen_ids)}
        N_unk = 2 * nL + n + len(plen_ids)
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        b = np.zeros(N_unk)

        def add(r: Any, c: Any, v: Any) -> None:
            r, c, v = np.broadcast_arrays(np.asarray(r), np.asarray(c), np.asarray(v, float))
            rows.append(r.ravel())
            cols.append(c.ravel())
            vals.append(v.ravel())

        def couple(ia: np.ndarray, ib: np.ndarray, G: np.ndarray) -> None:
            """Symmetric exchange G (T_a - T_b) in row a and G (T_b - T_a) in row b."""
            add(ia, ia, G); add(ia, ib, -G); add(ib, ib, G); add(ib, ia, -G)

        kt_full = np.zeros((n, 2))
        kt_full[lat] = props["k_t"]
        phi_full = np.zeros(n)
        phi_full[lat] = props["phi"]
        q = flow["q"]
        nsf = self._n_sfaces
        face_q = {"s": q[:nsf], "z": q[nsf:]}
        k_sl = mat.conductivity * jac.sleeve_thickness if o.sleeve_conduction else 0.0
        flux_bc = o.wall_condition == "flux"
        for key in ("s", "z"):
            f = self._faces[key]
            l, r, dl, dr, A, length = f["l"], f["r"], f["dl"], f["dr"], f["area"], f["length"]
            c = f["comp"]
            LL, LP, PL = f["LL"], f["LP"], f["PL"]
            dist = dl + dr
            # ---- advection, first-order upwind (fluid rows and plenum rows)
            F = rcp * face_q[key]
            pos = F > 0
            m = LL & pos  # l -> r
            add(iF(l[m]), iF(l[m]), F[m]); add(iF(r[m]), iF(l[m]), -F[m])
            m = LL & ~pos  # r -> l
            add(iF(r[m]), iF(r[m]), -F[m]); add(iF(l[m]), iF(r[m]), F[m])
            for pk in plen_ids:
                m = LP & (f["kr"] == pk) & pos  # lattice -> plenum
                add(iF(l[m]), iF(l[m]), F[m]); add(iP[pk], iF(l[m]), -F[m])
                m = LP & (f["kr"] == pk) & ~pos  # plenum -> lattice
                add(iF(l[m]), iP[pk], F[m]); add(iP[pk], iP[pk], -F[m])
                m = PL & (f["kl"] == pk) & pos  # plenum -> lattice
                add(iF(r[m]), iP[pk], -F[m]); add(iP[pk], iP[pk], F[m])
                m = PL & (f["kl"] == pk) & ~pos  # lattice -> plenum
                add(iF(r[m]), iF(r[m]), -F[m]); add(iP[pk], iF(r[m]), F[m])
            # ---- coolant conduction (lattice-lattice faces only)
            if o.fluid_conduction:
                kf = cool.conductivity
                G = A[LL] / (dl[LL] / (kf * phi_full[l[LL]]) + dr[LL] / (kf * phi_full[r[LL]]))
                couple(iF(l[LL]), iF(r[LL]), G)
            # ---- lattice solid in-plane conduction (lattice-lattice faces; zero flux at open slot faces)
            if o.lattice_conduction:
                G = A[LL] / (dl[LL] / kt_full[l[LL], c] + dr[LL] / kt_full[r[LL], c])
                couple(iS(l[LL]), iS(r[LL]), G)
            # ---- sleeve conduction (every face)
            if flux_bc and k_sl > 0:
                couple(iW(l), iW(r), k_sl * length / dist)
        # ---- local exchange: solid-fluid, sleeve-solid, sleeve-fluid (exposed fraction)
        Al = area[lat]
        couple(iS(lat), iF(lat), G_sf * Al)
        couple(iW(lat), iS(lat), G_ws * Al)
        couple(iW(lat), iF(lat), U_dir * Al)
        for pk in plen_ids:
            cells = np.flatnonzero(kind == pk)
            couple(iW(cells), np.full(len(cells), iP[pk]), h_m * area[cells])
        if flux_bc:
            b[iW(np.arange(n))] = q_in * area
        elif o.wall_condition == "temperature":
            if o.wall_temperature is None:
                raise ValueError("wall_condition='temperature' needs wall_temperature")
            # Dirichlet rows: drop every wall-row entry, then T_w = T_w0
            wall_rows = np.arange(2 * nL, 2 * nL + n)
            keep = [~np.isin(r_, wall_rows) for r_ in rows]
            rows = [r_[k_] for r_, k_ in zip(rows, keep)]
            cols = [c_[k_] for c_, k_ in zip(cols, keep)]
            vals = [v_[k_] for v_, k_ in zip(vals, keep)]
            add(wall_rows, wall_rows, np.ones(n))
            b[wall_rows] = o.wall_temperature
        else:
            raise ValueError(f"wall_condition must be 'flux' or 'temperature', got {o.wall_condition!r}")
        # ---- external flow through the plenums: inflow at T_in, outflow at the plenum temperature
        T_in = cfg.operating.inlet_temperature
        Q_ext_in = 0.0
        for pk in plen_ids:
            net = rcp * float(self._plenum_face_sign(pk) @ q)  # net enthalpy-carrying flow plenum -> lattice
            if net > 0:
                b[iP[pk]] += net * T_in
                Q_ext_in += net
            else:
                add(iP[pk], iP[pk], -net)
        Amat = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N_unk, N_unk))
        T = spla.spsolve(Amat.tocsc(), b)
        T_f = T[:nL]
        T_s = T[nL:2 * nL]
        T_w = T[2 * nL:2 * nL + n]
        T_pl = {pk: T[iP[pk]] for pk in plen_ids}
        T_wall = T_w + (q_on_rs * jac.sleeve_thickness / mat.conductivity if flux_bc else 0.0)
        k_max = int(np.argmax(T_wall))
        S, Z = g.meshgrid()
        wall_out = (G_ws * (T_w[lat] - T_s) + U_dir * (T_w[lat] - T_f)) * Al
        heat_input = float(np.sum(q_in * area)) if flux_bc else float(np.sum(wall_out)) + sum(
            float(np.sum(h_m * area[kind == pk] * (T_w[kind == pk] - T_pl[pk]))) for pk in plen_ids)
        removed = Q_ext_in * (T_pl[OUTLET] - T_in)
        heat_manifolds = {pk: float(np.sum(h_m * area[kind == pk] * (T_w[kind == pk] - T_pl[pk]))) for pk in plen_ids}
        return dict(T_f=T_f, T_s=T_s, T_w=T_w, T_wall=T_wall, T_out=float(T_pl[OUTLET]), T_plenum=T_pl,
                    T_wall_max=float(T_wall[k_max]),
                    T_wall_max_lattice=float(np.max(T_wall[lat])),
                    T_wall_mean=float(np.average(T_wall, weights=area)),
                    argmax_s=float(S.ravel()[k_max]), argmax_z=float(Z.ravel()[k_max]),
                    heat_input=heat_input, energy_balance_error=abs(removed - heat_input) / heat_input,
                    h_sf=h_sf, U_eff=U_eff, eta=eta, re=np.asarray(ht.reynolds, float), q_in=q_in, h_m=h_m,
                    heat_manifolds={("inlet" if k == INLET else "outlet"): v for k, v in heat_manifolds.items()})

    # ------------------------------------------------------------------ structure
    def _structure(self, props: Mapping[str, Any], F: Mapping[str, np.ndarray], flow: Mapping[str, Any]) -> dict[str, Any]:
        cfg, g = self.cfg, self.grid
        jac, mat, loads = cfg.jacket, cfg.material, cfg.loads
        cl: LocalClosures = props["cl"]
        lat = self._lat_idx
        A = g.area.ravel()[lat]
        r_m, r_i = g.r_mean, jac.lattice_inner_radius
        h = g.gap
        G_rs = cl.C_eff[:, 5, 5] * cl.f_G
        G_rz = cl.C_eff[:, 4, 4] * cl.f_G
        C_rr = cl.C_eff[:, 0, 0] * cl.f_C
        tau_rs = loads.torque * G_rs / (r_m * np.sum(G_rs * A)) * (r_m / r_i) ** 2
        tau_rz = loads.thrust * G_rz / np.sum(G_rz * A) * (r_m / r_i)
        p = loads.coolant_pressure_gauge + flow["dp"]
        t_o = jac.outer_wall_thickness
        R_o = jac.lattice_outer_radius + t_o / 2.0
        k_hoop = mat.youngs_modulus * t_o / R_o**2
        k_lat = C_rr / h
        share = k_lat / (k_lat + k_hoop)
        sig_rr = p * share
        sx, sxy, sxz = cl.loc["uniaxial_x"], cl.loc["shear_xy"], cl.loc["shear_xz"]
        s3 = math.sqrt(3.0)
        vm = sx * sig_rr + s3 * (sxy * np.abs(tau_rs) + sxz * np.abs(tau_rz))
        margin = mat.allowable_stress / vm
        k = int(np.argmin(margin))
        # outer wall
        b = cfg.manifolds.width * R_o / r_m
        sig_hoop = (1.0 - float(np.min(share))) * p * R_o / t_o
        sig_slot = p * b**2 / (2.0 * t_o**2)
        L_max = float(np.max(F["L"].ravel()[lat]))
        sig_pore = 0.75 * p * (L_max / 2.0 / t_o) ** 2
        sig_ow = max(sig_hoop, sig_slot, sig_pore)
        ow_margin = mat.allowable_stress / sig_ow
        S, Z = g.meshgrid()
        summary = {
            "sigma_vm_max": float(vm[k]), "argmin_s": float(S.ravel()[lat][k]), "argmin_z": float(Z.ravel()[lat][k]),
            "contrib_pressure": float(sx[k] * sig_rr[k]), "contrib_torque": float(s3 * sxy[k] * abs(tau_rs[k])),
            "contrib_thrust": float(s3 * sxz[k] * abs(tau_rz[k])),
            "tau_rs_max": float(np.max(np.abs(tau_rs))), "tau_rz_max": float(np.max(np.abs(tau_rz))),
            "sigma_rr_max": float(np.max(sig_rr)), "pressure": p, "lattice_pressure_share_min": float(np.min(share)),
            "outer_wall_hoop": sig_hoop, "outer_wall_slot_bending": sig_slot, "outer_wall_pore_bending": sig_pore,
        }
        return dict(lattice_margin=float(margin[k]), outer_wall_margin=float(ow_margin), vm=vm, margin=margin,
                    summary=summary)

    # ------------------------------------------------------------------ mass, manufacturability
    def _mass(self, F: Mapping[str, np.ndarray]) -> dict[str, float]:
        cfg, g = self.cfg, self.grid
        jac, mat = cfg.jacket, cfg.material
        lat = self.grid.lattice
        A = g.area
        m_lat = mat.density * g.gap * float(np.sum(F["rho"][lat] * A[lat]))
        L_ax = jac.axial_length
        r_s, t_sl = jac.stator_radius, jac.sleeve_thickness
        r_o, t_o = jac.outer_radius, jac.outer_wall_thickness
        m_sleeve = mat.density * math.pi * ((r_s + t_sl) ** 2 - r_s**2) * L_ax
        m_outer = mat.density * math.pi * (r_o**2 - (r_o - t_o) ** 2) * L_ax
        v_cool = g.gap * float(np.sum((1.0 - F["rho"][lat]) * A[lat]) + np.sum(A[~lat]))
        return dict(total=m_lat + m_sleeve + m_outer, lattice=m_lat, walls=m_sleeve + m_outer, coolant_volume=v_cool)

    def _manufacturability(self, F: Mapping[str, np.ndarray]) -> dict[str, Any]:
        from voxlat.geometry import min_cell_size

        cfg, g = self.cfg, self.grid
        lat = g.lattice
        rho, w, L, az = (F[k][lat] for k in ("rho", "w", "L", "a_z"))
        bnd = cfg.design_bounds
        tol = 1e-9
        in_bounds = {
            "rho": bool(np.all((rho >= bnd.relative_density.lo - tol) & (rho <= bnd.relative_density.hi + tol))),
            "w": bool(np.all((w >= bnd.blend_w.lo - tol) & (w <= bnd.blend_w.hi + tol))),
            "L": bool(np.all((L >= bnd.cell_size.lo - tol) & (L <= bnd.cell_size.hi + tol))),
            "a_z": bool(np.all((az >= bnd.axial_stretch.lo - tol) & (az <= bnd.axial_stretch.hi + tol))),
        }
        L_min = np.asarray(min_cell_size(np.clip(w, 0, 1), np.clip(rho, 0.15, 0.55), cfg=cfg), float)
        size_ratio = float(np.min(L / L_min))
        # in-plane grading |grad rho*| L on lattice-lattice faces
        grad = []
        for key in ("s", "z"):
            f = self._faces[key]
            m = f["LL"]
            r_l = F["rho"].ravel()[f["l"][m]]
            r_r = F["rho"].ravel()[f["r"][m]]
            Lf = 0.5 * (F["L"].ravel()[f["l"][m]] + F["L"].ravel()[f["r"][m]])
            grad.append(np.abs(r_r - r_l) / (f["dl"][m] + f["dr"][m]) * Lf)
        g_all = np.concatenate(grad) if grad else np.zeros(1)
        g_max = float(np.max(g_all)) if g_all.size else 0.0
        limit = cfg.manufacturing.max_density_change_per_cell if self.options.gradient_limit is None \
            else self.options.gradient_limit
        N_min = float(np.min(g.gap / L))
        neck = (w > 0.3) & (w < 0.7) & (rho < 0.4)
        A = g.area[lat]
        out = {
            "within_bounds": all(in_bounds.values()), "bounds": in_bounds,
            "cell_size_ok": size_ratio >= 1.0, "cell_size_ratio_min": size_ratio,
            "gradient_ok": g_max <= limit + 1e-12, "gradient_max": g_max, "gradient_limit": limit,
            "N_min": N_min, "finite_gap_valid": N_min >= 1.0 - 1e-9,
            "pinch_neck_area_fraction": float(np.sum(A[neck]) / np.sum(A)),
        }
        out["manufacturable"] = bool(out["within_bounds"] and out["cell_size_ok"] and out["gradient_ok"])
        return out

    # ------------------------------------------------------------------ fields
    def _fields(self, F, props, flow, heat, struct) -> JacketFields:
        g = self.grid
        shp = g.shape
        lat = self._lat_idx
        n = g.n_cells

        def full(v: np.ndarray, fill: float = np.nan) -> np.ndarray:
            out = np.full(n, fill)
            out[lat] = v
            return out.reshape(shp)

        T_f = np.full(n, np.nan)
        T_f[lat] = heat["T_f"]
        for pk, T in heat["T_plenum"].items():
            T_f[self._kind == pk] = T
        U_eff = full(heat["U_eff"])
        U_eff.ravel()[self._kind != LATTICE] = heat["h_m"]
        Ts = full(heat["T_s"])
        Kin = np.full((n, 2, 2), np.nan)
        Kin[lat] = props["Kt_eff"]
        nsf = self._n_sfaces
        return JacketFields(
            grid=g, rho=F["rho"], w=F["w"], L=F["L"], a_z=F["a_z"], pressure=flow["p"].reshape(shp),
            U_s=flow["Us"].reshape(shp), U_z=flow["Uz"].reshape(shp), speed=np.hypot(flow["Us"], flow["Uz"]).reshape(shp),
            T_f=T_f.reshape(shp), T_w=heat["T_w"].reshape(shp), T_wall=heat["T_wall"].reshape(shp), T_s_mean=Ts,
            q_in=heat["q_in"].reshape(shp), U_eff=U_eff, h_sf=full(heat["h_sf"]), eta=full(heat["eta"]),
            reynolds=full(heat["re"]), forchheimer=full(flow["Fo"]), K_inplane=Kin.reshape(shp + (2, 2)),
            sigma_vm=full(struct["vm"]), margin=full(struct["margin"]),
            face_flow_s=flow["q"][:nsf].reshape(shp), face_flow_z=flow["q"][nsf:].reshape(g.n_s, g.n_z - 1))

    # ------------------------------------------------------------------ helpers for verification
    def closures_at(self, rho: float, w: float, L: float, a_z: float = 1.0) -> dict[str, Any]:
        """Local properties of one uniform cell (as used by ``evaluate``)."""
        props = self._local_properties({k: np.array([v], float) for k, v in
                                        (("rho", rho), ("w", w), ("L", L), ("a_z", a_z))})
        return props


_DEFAULT: dict[str, JacketModel] = {}


def default_model(**option_overrides: Any) -> JacketModel:
    """Cached ``JacketModel`` with the reference config (one per distinct option set)."""
    key = repr(sorted(option_overrides.items()))
    if key not in _DEFAULT:
        _DEFAULT[key] = JacketModel(options=JacketOptions(**option_overrides))
    return _DEFAULT[key]


def evaluate(rho: Any = 0.35, w: Any = 0.0, L: Any = 4e-3, a_z: Any = 1.0, flow_rate: float | None = None,
             *, return_fields: bool = False, model: JacketModel | None = None) -> dict[str, Any]:
    """Convenience wrapper: ``(model or default_model()).evaluate(...)`` (the Task 10/11 interface)."""
    return (model or default_model()).evaluate(rho, w, L, a_z, flow_rate, return_fields=return_fields)


# =============================================================================
# Analytic references for a uniform jacket with full-length manifolds
# =============================================================================
def _path_geometry(model: JacketModel) -> tuple[float, float]:
    cfg = model.cfg
    g = model.grid
    if cfg.manifolds.axial_extent < cfg.jacket.axial_length:
        raise ValueError("the 1-D references need full-length manifolds")
    span = abs(math.remainder(cfg.manifolds.outlet_angle - cfg.manifolds.inlet_angle, 2 * math.pi))
    if abs(span - math.pi) > 1e-9:
        raise ValueError("the 1-D references need diametrically opposite manifolds")
    ell = math.pi * g.r_mean - cfg.manifolds.width  # length of each of the two paths
    return ell, cfg.jacket.axial_length


def uniform_flow_reference(model: JacketModel, rho: float, w: float, L: float, a_z: float = 1.0,
                           flow_rate: float | None = None) -> dict[str, float]:
    """Exact 1-D solution: two parallel paths, uniform U = Q/(2 h L_ax) along s.

    dp = ell (mu U / K_ss,eff + rho C_F U^2 / sqrt(K_ss)); exact for G/D-type cells (no (s, z)
    cross term), where the 2-D flow is uniform.
    """
    cfg = model.cfg
    Q = cfg.operating.nominal_flow_rate if flow_rate is None else flow_rate
    ell, L_ax = _path_geometry(model)
    pr = model.closures_at(rho, w, L, a_z)
    U = Q / (2.0 * model.grid.gap * L_ax)
    e = np.array([[1.0, 0.0]])
    K_eff = float(directional_permeability(pr["Kt_eff"], e)[0])
    K_b = float(directional_permeability(pr["Kt"], e)[0])
    cf = float(pr["cf"][0])
    mu, rho_f = cfg.coolant.dynamic_viscosity, cfg.coolant.density
    grad = mu * U / K_eff + rho_f * cf * U**2 / math.sqrt(K_b)
    return dict(U=U, path_length=ell, K_eff=K_eff, K_bulk=K_b, cf=cf, dp=ell * grad, dp_darcy=ell * mu * U / K_eff,
                forchheimer_number=rho_f * cf * U / math.sqrt(K_b) / (mu / K_eff))


def ntu_reference(model: JacketModel, rho: float, w: float, L: float, a_z: float = 1.0,
                  flow_rate: float | None = None) -> dict[str, float]:
    """epsilon-NTU solution of a uniform jacket at fixed wall temperature T_w0.

    Each path: m_dot = rho Q / 2, exchanger area ell L_ax, conductance U_eff (fin model of the
    uniform cell at the path velocity) -> NTU = U_eff ell L_ax / (m_dot c_p), eps = 1 - exp(-NTU),
    T_out = T_in + eps (T_w0 - T_in). Valid with manifold exchange, coolant and solid in-plane
    conduction switched off (``JacketOptions(manifold_htc=0, fluid_conduction=False, ...)``).
    """
    from voxlat.closures.empirical import interstitial_heat_transfer

    cfg, o = model.cfg, model.options
    Q = cfg.operating.nominal_flow_rate if flow_rate is None else flow_rate
    ell, L_ax = _path_geometry(model)
    ref = uniform_flow_reference(model, rho, w, L, a_z, Q)
    pr = model.closures_at(rho, w, L, a_z)
    cool = cfg.coolant
    ht = interstitial_heat_transfer(ref["U"], pr["phi"], pr["a_sf"], ref["K_bulk"], w=w, conductivity=cool.conductivity,
                                    kinematic_viscosity=cool.kinematic_viscosity, prandtl=cool.prandtl,
                                    cf=pr["cf"] if o.inertia else None, variant=o.nu_variant,
                                    pr_exponent=o.pr_exponent, on_extrapolation="ignore")
    h_sf = float(np.asarray(ht.h_sf).ravel()[0])
    U_fin, eta = fin_conductance(pr["k_n"][0], h_sf * pr["a_sf"][0], model.grid.gap, o.fin_model)
    U_eff = float(U_fin) + (float(pr["phi"][0]) * h_sf if o.wall_exposed_convection else 0.0)
    mcp = cool.density * cool.specific_heat * Q / 2.0
    ntu = U_eff * ell * L_ax / mcp
    eps = 1.0 - math.exp(-ntu)
    return dict(U_eff=U_eff, h_sf=h_sf, eta=float(eta), ntu=ntu, effectiveness=eps, path_length=ell)
