"""Baseline jackets B1-B3 for the RQ2 comparison (Task 10).

All baselines expose the Task 9 interface: ``evaluate(design, flow_rate=None,
return_fields=False) -> dict`` with the same SI output keys (``thermal_resistance``,
``delta_p``, ``pump_power``, ``mass``, ``min_structural_margin``, ``manufacturable``,
balances, ``diagnostics``, optional ``fields``), plus two keys shared by every design:
``T_wall_max_active`` / ``R_active`` (maximum over the cooled region, i.e. excluding the
sleeve under the manifold slots).

B1 - rectangular-channel jacket in the same 6 mm gap (``ChannelDesign``)
--------------------------------------------------------------------------
Ribs of thickness t_r (AlSi10Mg, full gap height h, bonded to sleeve and outer wall)
separate channels of axial width w_c; the axial period is b = w_c + t_r = L_ax / N_pass.

* ``layout="helical"`` (the B1 of the plan): n_starts parallel helical channels, pitch
  P = n_starts b, fed by ring manifolds at the axial ends (z = 0 inlet, z = L_ax outlet;
  the rings are assumed outside the heated length, so the whole sleeve is finned).
* ``layout="circumferential"`` (B1s, slot-matched): N_pass parallel circumferential
  channels fed by the SAME axial manifold slots as the lattice (Task 9: theta = 0 / 180 deg,
  15 mm, full length, two half-annulus paths), with the same sleeve/plenum treatment under
  the slots. This is the like-for-like comparison for Task 9 open issue 1.

Hydraulics (per channel, all channels identical, constant properties):
    D_h = 2 w_c h / (w_c + h),  Re = U D_h / nu,  De = Re sqrt(D_h / (2 R_c)),
    R_c = r_m (1 + tan^2 alpha) (helix curvature radius; r_m for circumferential channels),
    dp = (f ell / D_h + K_inc) rho U^2 / 2     (ideal manifolds, as in Task 9).
* Straight-duct laminar f: exact Shah & London (1978) series for rectangular ducts.
* Curved-duct friction (Schmidt 1967, recommended in the Heat Exchanger Design Handbook
  1983): laminar f_c = f_s [1 + 0.14 (d/D)^0.97 Re^(1 - 0.644 (d/D)^0.312)] for
  Re < Re_crit = 2300 [1 + 8.6 (d/D)^0.45]; turbulent f_c = f_Blasius [1 + 2.88e4/Re (d/D)^0.62]
  (Re < 2.2e4) and f_Blasius [1 + 0.0823 (1 + d/D)(d/D)^0.53 Re^0.25] above, d = D_h, D = 2 R_c.
  For the rectangular section the laminar straight value f_s = (f Re)_rect / Re is used.
* K_inc: incremental pressure-drop number of the hydrodynamic entrance (1.25, circular-duct
  value of Shah & London; flagged as an approximation).
Heat transfer (one coefficient h_c = Nu k_f / D_h on every wetted wall):
* Laminar coil (VDI Heat Atlas Gc/G3, after Schmidt 1967 and Gnielinski 1986):
  Nu = 3.66 + 0.08 [1 + 0.8 (d/D)^0.9] Re^m Pr^(1/3), m = 0.5 + 0.2903 (d/D)^0.194,
  floored by the fully developed straight rectangular duct Nu_H1(alpha) (Shah & London 1978,
  8.235 (1 - 2.0421 a + 3.0853 a^2 - 2.4765 a^3 + 1.0578 a^4 - 0.1861 a^5)).
* Turbulent coil (Gnielinski 1986): Nu = (xi/8) Re Pr / (1 + 12.7 sqrt(xi/8) (Pr^(2/3) - 1)),
  xi = 0.3164 Re^-0.25 + 0.03 (d/D)^0.5.
* Transition (Gnielinski 1986, VDI): Re_crit < Re < 2.2e4,
  Nu = g Nu_lam(Re_crit) + (1 - g) Nu_turb(2.2e4), g = (2.2e4 - Re)/(2.2e4 - Re_crit).
* Thermal entrance enhancement is NOT included (the hot spot sits at the outlet, where the
  flow is developed; ``nu_model="developing_mean"`` gives the optimistic length-mean bound
  with the Baehr-Stephan correlation).
Wall-to-coolant conductance per unit sleeve area (sleeve isothermal over one period; the
spreading efficiency is reported):
    U_eff = [h_c w_c + sqrt(2 h_c k t_r) tanh(m h)] / b,  m = sqrt(2 h_c / (k t_r))
(rib = straight fin, both faces wetted, adiabatic tip at the outer wall - the same
adiabatic-outer-wall assumption as the Task 9 lattice; ``outer_wall_fin=True`` adds the
outer wall as a fin attached to the rib tip).

Planform heat model (same grid, heat input, sleeve conduction, plenum treatment and
reported T_wall as Task 9): coolant T_f(s, z) advected by the homogenized channel flux
(helical: q_s = Q/P per unit axial width and q_z = Q/C per unit arc length; circumferential:
q_s = +-Q/(2 L_ax)), no transverse mixing (ribs), first-order upwind; sleeve T_w with
in-plane conduction k_s t_sl, coupled by U_eff; slot sleeve <-> plenum with h_m.

Structure (homogenized, as Task 9): torque -> in-plane rib shear tau_rs/f_r; thrust ->
transverse rib bending (rigid faces, fixed-guided strip: sigma_b = 3 V' h / t_r^2,
V' = tau_rz b) plus shear 1.5 V'/t_r; coolant pressure -> rib tension p b/t_r times the
rib share of the radial stiffness; sigma_vm = K_t sqrt((sigma_p + sigma_b)^2 + 3 (tau_T^2 +
tau_V^2)) with root stress-concentration factor K_t (default 2.5, assumption). Outer wall:
hoop share, strip bending over the channel width (and over the slots for B1s).

B2 / B3 - lattices evaluated with the Task 9 ``JacketModel``
------------------------------------------------------------
* ``UniformLatticeDesign(rho, L, w=0)`` - B2 is the best uniform gyroid (``tune_b2``).
* ``GradedDensityDesign(rho_path, amp_z, L, w=0)`` - B3, density-only grading (prior art):
  rho(s, z) = interp(xi, rho_path) + amp_z psi(z), xi in [0, 1] the position along each
  half-annulus path from the inlet slot edge to the outlet slot edge (both paths mirror-
  symmetric), psi = (12 (z/L_ax - 1/2)^2 - 1)/2 the shape of the axial heat flux;
  w = 0 and L fixed (``tune_b3`` fixes L at the B2 optimum).

Selection rule (identical for every family, ``select_best``): minimum thermal resistance at
the nominal flow rate subject to manufacturable, min structural margin >= 1 and pump power
<= ``pump_power_cap``. ``pareto_front`` gives each family's non-dominated set in
(R, pump power, mass) for the Task 11 overlay.

References (bibliographic details in STATUS.md, Task 10): Shah & London 1978; Schmidt 1967;
Gnielinski 1986 / VDI Heat Atlas; Schluender (ed.) Heat Exchanger Design Handbook 1983;
Baehr & Stephan 2011; Incropera & DeWitt (fins).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from voxlat.device.jacket2d import (
    INLET,
    LATTICE,
    OUTLET,
    JacketGrid,
    JacketModel,
    JacketOptions,
    _segment_faces,
    build_grid,
    cell_average_heat_flux,
    manifold_htc_default,
)
from voxlat.utils.log import get_logger

__all__ = [
    # correlations
    "rect_duct_fre",
    "rect_duct_nu_h1",
    "helical_re_crit",
    "curved_duct_friction",
    "coil_nusselt_laminar",
    "coil_nusselt_turbulent",
    "curved_duct_nusselt",
    "developing_nusselt_mean",
    "rib_conductance",
    "DUCT_REFERENCES",
    # designs and models
    "ChannelDesign",
    "ChannelOptions",
    "ChannelFields",
    "ChannelJacketModel",
    "channel_hydraulics",
    "UniformLatticeDesign",
    "GradedDensityDesign",
    "path_coordinate",
    "evaluate",
    "design_from_dict",
    # tuning
    "b1_candidates",
    "tune_b1",
    "tune_b2",
    "tune_b3",
    "select_best",
    "pareto_front",
    "TuningResult",
    "summarize",
]

LOG = get_logger("device.baselines")
RE_TURB = 2.2e4  # upper end of the coil transition range (Schmidt 1967 / Gnielinski 1986)

DUCT_REFERENCES = {
    "shah_london_1978": "Shah R.K., London A.L. (1978). Laminar Flow Forced Convection in Ducts. Adv. Heat Transfer, "
                        "Suppl. 1. Academic Press. (rectangular duct f Re series; Nu_H1 polynomial; K(inf) = 1.25 circular)",
    "schmidt_1967": "Schmidt E.F. (1967). Waermeuebergang und Druckverlust in Rohrschlangen. Chem. Ing. Tech. 39(13), "
                    "781-789. (Re_crit, laminar/turbulent friction, laminar Nu of coils)",
    "gnielinski_1986": "Gnielinski V. (1986). Heat transfer and pressure drop in helically coiled tubes. Proc. 8th Int. "
                       "Heat Transfer Conf., Vol. 6, 2847-2854. Hemisphere. (turbulent Nu with xi, transition interpolation; "
                       "VDI Heat Atlas chapter Gc/G3)",
    "hedh_1983": "Schluender E.U. (ed.) (1983). Heat Exchanger Design Handbook. Hemisphere. (Schmidt correlations "
                 "recommended as defaults)",
    "baehr_stephan": "Baehr H.D., Stephan K. (2011). Heat and Mass Transfer, 3rd ed. Springer. (length-mean Nu, "
                     "simultaneously developing laminar flow, constant wall temperature)",
}


# =============================================================================
# Duct correlations
# =============================================================================
def rect_duct_fre(alpha: Any, n_terms: int = 200) -> np.ndarray:
    """Fully developed laminar Darcy f Re of a rectangular duct, aspect ratio alpha = short/long in (0, 1].

    Exact series solution (Shah & London 1978), with D_h = 4A/P:
        (f Re)_Fanning = 24 / ((1 + a)^2 [1 - (192 a / pi^5) sum_{n odd} tanh(n pi / (2 a)) / n^5]),
    returned in the Darcy convention (x 4): 96 (parallel plates, a -> 0) ... 56.91 (square).
    """
    a = np.asarray(alpha, float)
    if np.any((a <= 0) | (a > 1)):
        raise ValueError("aspect ratio must be in (0, 1]")
    n = np.arange(1, 2 * n_terms, 2, dtype=float)
    s = np.sum(np.tanh(np.multiply.outer(np.pi / (2.0 * a), n)) / n**5, axis=-1)
    fre_fanning = 24.0 / ((1.0 + a) ** 2 * (1.0 - 192.0 * a / np.pi**5 * s))
    return 4.0 * fre_fanning


def rect_duct_nu_h1(alpha: Any) -> np.ndarray:
    """Fully developed laminar Nu_H1 (four walls, axially uniform flux), Shah & London (1978) polynomial."""
    a = np.asarray(alpha, float)
    return 8.235 * (1 - 2.0421 * a + 3.0853 * a**2 - 2.4765 * a**3 + 1.0578 * a**4 - 0.1861 * a**5)


def helical_re_crit(d_ratio: Any) -> np.ndarray:
    """Critical Reynolds number of a coil, Schmidt (1967): 2300 [1 + 8.6 (d/D)^0.45]; d/D = D_h / (2 R_c)."""
    return 2300.0 * (1.0 + 8.6 * np.asarray(d_ratio, float) ** 0.45)


def _blasius(Re: np.ndarray) -> np.ndarray:
    return 0.3164 * Re ** -0.25


def curved_duct_friction(Re: Any, d_ratio: Any, alpha: Any = None) -> np.ndarray:
    """Darcy friction factor of a curved (helical) duct after Schmidt (1967).

    Laminar (Re <= Re_crit): f = f_s [1 + 0.14 (d/D)^0.97 Re^(1 - 0.644 (d/D)^0.312)], f_s = (f Re)_s / Re
    with (f Re)_s = 64 (circular, ``alpha=None``) or the exact rectangular value.
    Turbulent: f = f_Blasius [1 + 2.88e4 / Re (d/D)^0.62]  (Re < 2.2e4),
               f = f_Blasius [1 + 0.0823 (1 + d/D) (d/D)^0.53 Re^0.25]  (Re >= 2.2e4).
    The two branches nearly coincide at Re_crit (the switch is the critical Re itself).
    """
    Re, dr = np.broadcast_arrays(np.asarray(Re, float), np.asarray(d_ratio, float))
    fre_s = 64.0 if alpha is None else rect_duct_fre(alpha)
    f_lam = fre_s / Re * (1.0 + 0.14 * dr**0.97 * Re ** (1.0 - 0.644 * dr**0.312))
    f_t1 = _blasius(Re) * (1.0 + 2.88e4 / Re * dr**0.62)
    f_t2 = _blasius(Re) * (1.0 + 0.0823 * (1.0 + dr) * dr**0.53 * Re**0.25)
    turb = np.where(Re < RE_TURB, f_t1, f_t2)
    return np.where(Re <= helical_re_crit(dr), f_lam, turb)


def coil_nusselt_laminar(Re: Any, Pr: Any, d_ratio: Any) -> np.ndarray:
    """VDI laminar coil Nusselt (Schmidt 1967; VDI Heat Atlas): 3.66 + 0.08 [1 + 0.8 (d/D)^0.9] Re^m Pr^(1/3)."""
    Re, Pr, dr = (np.asarray(x, float) for x in (Re, Pr, d_ratio))
    m = 0.5 + 0.2903 * dr**0.194
    return 3.66 + 0.08 * (1.0 + 0.8 * dr**0.9) * Re**m * Pr ** (1.0 / 3.0)


def coil_nusselt_turbulent(Re: Any, Pr: Any, d_ratio: Any) -> np.ndarray:
    """Gnielinski (1986) turbulent coil Nusselt with xi = 0.3164 Re^-0.25 + 0.03 (d/D)^0.5 (no wall-viscosity term)."""
    Re, Pr, dr = (np.asarray(x, float) for x in (Re, Pr, d_ratio))
    xi = _blasius(Re) + 0.03 * np.sqrt(dr)
    return (xi / 8.0) * Re * Pr / (1.0 + 12.7 * np.sqrt(xi / 8.0) * (Pr ** (2.0 / 3.0) - 1.0))


def curved_duct_nusselt(Re: Any, Pr: Any, d_ratio: Any, alpha: Any = None) -> np.ndarray:
    """Nusselt number (D_h based) of a curved duct over all regimes.

    Laminar: max(VDI laminar coil, fully developed straight Nu_H1(alpha)); transition (Re_crit < Re < 2.2e4):
    linear interpolation in Re between the laminar value at Re_crit and the turbulent value at 2.2e4
    (Gnielinski 1986); turbulent: Gnielinski coil. ``alpha=None`` uses the circular floor 4.364.
    """
    Re, Pr, dr = np.broadcast_arrays(*(np.asarray(x, float) for x in (Re, Pr, d_ratio)))
    floor = 4.364 if alpha is None else rect_duct_nu_h1(alpha)
    Rc = helical_re_crit(dr)
    nu_lam = np.maximum(coil_nusselt_laminar(Re, Pr, dr), floor)
    nu_lam_c = np.maximum(coil_nusselt_laminar(Rc, Pr, dr), floor)
    nu_t22 = coil_nusselt_turbulent(np.full_like(Re, RE_TURB), Pr, dr)
    g = np.clip((RE_TURB - Re) / (RE_TURB - Rc), 0.0, 1.0)
    nu_tr = g * nu_lam_c + (1.0 - g) * nu_t22
    nu_turb = coil_nusselt_turbulent(Re, Pr, dr)
    return np.where(Re <= Rc, nu_lam, np.where(Re < RE_TURB, nu_tr, nu_turb))


def developing_nusselt_mean(Re: Any, Pr: Any, D_h: Any, length: Any) -> np.ndarray:
    """Length-mean Nu of simultaneously developing laminar flow, Baehr & Stephan (constant wall T, circular).

    Nu = [3.657 / tanh(2.264 Gz^-1/3 + 1.7 Gz^-2/3) + 0.0499 Gz tanh(1/Gz)] / tanh(2.432 Pr^(1/6) Gz^(-1/6)),
    Gz = D_h Re Pr / length. Used only for the optimistic bound ``nu_model="developing_mean"``.
    """
    Re, Pr, D, Lg = (np.asarray(x, float) for x in (Re, Pr, D_h, length))
    Gz = D * Re * Pr / Lg
    return (3.657 / np.tanh(2.264 * Gz ** (-1 / 3) + 1.7 * Gz ** (-2 / 3)) + 0.0499 * Gz * np.tanh(1.0 / Gz)) \
        / np.tanh(2.432 * Pr ** (1 / 6) * Gz ** (-1 / 6))


def rib_conductance(h_c: Any, k: float, t_r: Any, height: float, tip_conductance: Any = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Heat flow per unit rib length and unit base excess temperature [W/(m K)], and the fin efficiency.

    Straight fin of thickness t_r and height ``height``, both faces at h_c, base at the sleeve:
    m = sqrt(2 h_c / (k t_r)), M = sqrt(2 h_c k t_r) (= k t_r m),
    G = M [sinh(m H) + beta cosh(m H)] / [cosh(m H) + beta sinh(m H)], beta = G_tip / M
    (G_tip = tip conductance per unit length; 0 = adiabatic tip, G = M tanh(m H)).
    Efficiency = G / (2 h_c H).
    """
    h, t, Gt = (np.asarray(x, float) for x in (h_c, t_r, tip_conductance))
    m = np.sqrt(2.0 * h / (k * t))
    M = k * t * m
    beta = Gt / M
    mH = m * height
    G = M * (np.sinh(mH) + beta * np.cosh(mH)) / (np.cosh(mH) + beta * np.sinh(mH))
    return G, G / (2.0 * h * height)


# =============================================================================
# B1 designs and options
# =============================================================================
@dataclass(frozen=True)
class ChannelDesign:
    """Rectangular-channel jacket (B1). Lengths in m; the channel depth is the lattice gap h.

    ``n_starts`` is used by the helical layout only (pitch = n_starts (w_c + t_r)).
    """

    channel_width: float
    rib_thickness: float
    n_starts: int = 1
    layout: str = "helical"  # "helical" | "circumferential"

    def __post_init__(self) -> None:
        if self.layout not in ("helical", "circumferential"):
            raise ValueError(f"layout must be 'helical' or 'circumferential', got {self.layout!r}")
        if self.channel_width <= 0 or self.rib_thickness <= 0 or self.n_starts < 1:
            raise ValueError("channel width, rib thickness and n_starts must be positive")

    @classmethod
    def from_passes(cls, n_pass: int, rib_thickness: float, n_starts: int = 1, layout: str = "helical",
                    axial_length: float = 0.050) -> "ChannelDesign":
        """Design with N_pass channel passes over the axial length: w_c = L_ax / N_pass - t_r."""
        return cls(axial_length / n_pass - rib_thickness, rib_thickness, n_starts, layout)

    @property
    def period(self) -> float:
        return self.channel_width + self.rib_thickness

    @property
    def pitch(self) -> float:
        """Axial advance per turn of one start (helical)."""
        return self.n_starts * self.period

    @property
    def rib_fraction(self) -> float:
        return self.rib_thickness / self.period

    def to_dict(self) -> dict[str, Any]:
        return {"family": "B1" if self.layout == "helical" else "B1s", "type": "channel", **asdict(self)}


@dataclass(frozen=True)
class ChannelOptions:
    """Options of the channel jacket model (defaults = the B1 production settings)."""

    spacing: float = 1.25e-3  # s cell size [m] (as Task 9)
    spacing_z: float | None = 2.5e-3
    nu_model: str = "nominal"  # "nominal" | "straight" (no curvature, fully developed) | "developing_mean" (optimistic)
    k_inc: float = 1.25  # entrance incremental pressure-drop number
    outer_wall_fin: bool = False  # True: outer wall as a fin at the rib tip (Task 9 lattice: adiabatic outer wall)
    root_stress_factor: float = 2.5  # K_t at the rib roots (assumption)
    manifold_htc: float | None = None  # W/m^2K under the slots (None -> Task 9 default, 180 W/m^2K)
    sleeve_conduction: bool = True
    wall_condition: str = "flux"  # "flux" | "temperature" (verification)
    wall_temperature: float | None = None


@dataclass
class ChannelFields:
    """Cell fields of a channel-jacket evaluation (n_s, n_z); NaN where undefined."""

    grid: JacketGrid
    T_f: np.ndarray  # coolant (plenum cells: plenum temperature) [K]
    T_w: np.ndarray  # sleeve at the coolant face [K]
    T_wall: np.ndarray  # sleeve, stator side [K]
    q_in: np.ndarray  # W/m^2 per planform area at r_m
    U_eff: np.ndarray  # W/m^2K (plenum cells: h_m)
    face_flow_s: np.ndarray  # through the +s face [m^3/s]
    face_flow_z: np.ndarray  # (n_s, n_z + 1) through z faces incl. boundaries [m^3/s]

    def to_dict(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "grid"}


def channel_hydraulics(design: ChannelDesign, cfg: Any, flow_rate: float, *, nu_model: str = "nominal",
                       k_inc: float = 1.25, path_length: float | None = None) -> dict[str, float]:
    """Per-channel hydraulics and heat-transfer coefficient (all channels identical).

    Returns D_h, aspect ratio, n_channels, channel flow, U, Re, De, d_ratio, Re_crit, regime, f, Nu, h_c,
    path length, helix angle, curvature radius, dp (= (f ell / D_h + K_inc) rho U^2 / 2).
    """
    jac, cool, man = cfg.jacket, cfg.coolant, cfg.manifolds
    h = jac.lattice_gap
    r_m = jac.lattice_mean_radius
    C = 2.0 * math.pi * r_m
    L_ax = jac.axial_length
    w = design.channel_width
    D_h = 2.0 * w * h / (w + h)
    alpha = min(w, h) / max(w, h)
    if design.layout == "helical":
        n_ch = design.n_starts
        P = design.pitch
        tan_a = P / C
        ell = (L_ax / P) * math.sqrt(C**2 + P**2)
        R_c = r_m * (1.0 + tan_a**2)
    else:
        n_ch = 2.0 * L_ax / design.period  # two paths, N_pass channels each (homogenized count)
        tan_a = 0.0
        ell = math.pi * r_m - man.width
        R_c = r_m
    if path_length is not None:
        ell = float(path_length)
    Q_ch = flow_rate / n_ch
    U = Q_ch / (w * h)
    nu = cool.kinematic_viscosity
    Re = U * D_h / nu
    d_ratio = D_h / (2.0 * R_c)
    De = Re * math.sqrt(d_ratio)
    Re_c = float(helical_re_crit(d_ratio))
    f = float(curved_duct_friction(Re, d_ratio, alpha))
    Pr = cool.prandtl
    if nu_model == "nominal":
        Nu = float(curved_duct_nusselt(Re, Pr, d_ratio, alpha))
    elif nu_model == "straight":
        Nu = float(rect_duct_nu_h1(alpha)) if Re <= 2300 else float(coil_nusselt_turbulent(max(Re, 2300.0), Pr, 0.0))
    elif nu_model == "developing_mean":
        Nu = max(float(curved_duct_nusselt(Re, Pr, d_ratio, alpha)),
                 float(developing_nusselt_mean(Re, Pr, D_h, ell)) - 3.657 + float(rect_duct_nu_h1(alpha)))
    else:
        raise ValueError(f"nu_model must be 'nominal', 'straight' or 'developing_mean', got {nu_model!r}")
    h_c = Nu * cool.conductivity / D_h
    dyn = 0.5 * cool.density * U**2
    dp = (f * ell / D_h + k_inc) * dyn
    regime = "laminar" if Re <= Re_c else ("transitional" if Re < RE_TURB else "turbulent")
    return dict(D_h=D_h, aspect_ratio=alpha, n_channels=n_ch, channel_flow=Q_ch, velocity=U, reynolds=Re, dean=De,
                d_ratio=d_ratio, re_crit=Re_c, regime=regime, friction=f, nusselt=Nu, h_c=h_c, path_length=ell,
                helix_angle=math.atan(tan_a), curvature_radius=R_c, delta_p=dp, dp_entrance=k_inc * dyn,
                graetz_inverse=ell / (D_h * Re * Pr))


# =============================================================================
# B1 model
# =============================================================================
def _helical_grid(cfg: Any, spacing: float, spacing_z: float | None) -> JacketGrid:
    jac = cfg.jacket
    r_m = jac.lattice_mean_radius
    C = 2.0 * math.pi * r_m
    L_ax = jac.axial_length
    s_faces = _segment_faces([0.0, C], spacing)
    z_faces = _segment_faces([0.0, L_ax], spacing_z or spacing)
    kind = np.zeros((len(s_faces) - 1, len(z_faces) - 1), dtype=np.int8)
    return JacketGrid(r_mean=r_m, circumference=C, axial_length=L_ax, gap=jac.lattice_gap, s_shift=0.0,
                      s_faces=s_faces, z_faces=z_faces, kind=kind)


class ChannelJacketModel:
    """B1 channel-jacket model; ``evaluate(design, flow_rate, return_fields)`` -> Task 9 output dict."""

    def __init__(self, cfg: Any = None, options: ChannelOptions | None = None):
        if cfg is None:
            from voxlat.utils.config import load_config

            cfg = load_config()
        self.cfg = cfg
        self.options = options or ChannelOptions()
        o = self.options
        self._grids = {"helical": _helical_grid(cfg, o.spacing, o.spacing_z)}
        if cfg.manifolds.axial_extent < cfg.jacket.axial_length - 1e-12:
            LOG.warning("manifold slots shorter than the jacket: the circumferential layout is unavailable")
        else:
            self._grids["circumferential"] = build_grid(cfg, o.spacing, spacing_z=o.spacing_z)
        self._q_cell = {k: cell_average_heat_flux(cfg, g.z_faces) for k, g in self._grids.items()}

    def with_options(self, **kw: Any) -> "ChannelJacketModel":
        return ChannelJacketModel(self.cfg, replace(self.options, **kw))

    def grid(self, layout: str) -> JacketGrid:
        if layout not in self._grids:
            raise ValueError(f"layout {layout!r} not available with this configuration")
        return self._grids[layout]

    # ------------------------------------------------------------------ public
    def evaluate(self, design: ChannelDesign, flow_rate: float | None = None, *,
                 return_fields: bool = False) -> dict[str, Any]:
        t0 = time.perf_counter()
        cfg, o = self.cfg, self.options
        Q = float(cfg.operating.nominal_flow_rate if flow_rate is None else flow_rate)
        if Q <= 0:
            raise ValueError("flow rate must be positive")
        g = self.grid(design.layout)
        hyd = channel_hydraulics(design, cfg, Q, nu_model=o.nu_model, k_inc=o.k_inc)
        U_eff, eta_rib, G_rib = self._conductance(design, hyd["h_c"])
        qs, qz = self._face_flows(design, g, Q)
        heat = self._solve_heat(g, design.layout, U_eff, qs, qz, Q)
        struct = self._structure(design, g, hyd)
        mass = self._mass(design, g)
        manu = self._manufacturability(design)
        Qh = cfg.motor.heat_to_jacket
        T_in = cfg.operating.inlet_temperature
        k_sl = cfg.material.conductivity * cfg.jacket.sleeve_thickness
        m_sl = math.sqrt(hyd["h_c"] / k_sl)
        x = m_sl * design.channel_width / 2.0
        div = self._divergence(g, qs, qz)
        lat = g.kind.ravel() == LATTICE
        if design.layout == "helical":
            mb = (abs(float(np.sum(qz[:, 0])) - Q) + abs(float(np.sum(qz[:, -1])) - Q)) / Q
        else:
            kk = g.kind.ravel()
            mb = (abs(float(np.sum(div[kk == INLET])) - Q) + abs(float(np.sum(div[kk == OUTLET])) + Q)) / Q
        out: dict[str, Any] = {
            "design": design.to_dict(),
            "thermal_resistance": (heat["T_wall_max"] - T_in) / Qh,
            "T_wall_max": heat["T_wall_max"],
            "T_wall_max_lattice": heat["T_wall_max_active"],
            "T_wall_max_active": heat["T_wall_max_active"],
            "R_active": (heat["T_wall_max_active"] - T_in) / Qh,
            "T_wall_max_s": heat["argmax_s"],
            "T_wall_max_z": heat["argmax_z"],
            "T_wall_mean": heat["T_wall_mean"],
            "T_in": T_in,
            "T_out": heat["T_out"],
            "heat_input": heat["heat_input"],
            "delta_p": hyd["delta_p"],
            "pump_power": hyd["delta_p"] * Q / cfg.operating.pump_efficiency,
            "hydraulic_power": hyd["delta_p"] * Q,
            "flow_rate": Q,
            "mass": mass["total"],
            "mass_lattice": mass["ribs"],
            "mass_ribs": mass["ribs"],
            "mass_walls": mass["walls"],
            "coolant_volume": mass["coolant_volume"],
            "min_structural_margin": min(struct["rib_margin"], struct["outer_wall_margin"]),
            "lattice_margin": struct["rib_margin"],
            "outer_wall_margin": struct["outer_wall_margin"],
            "structural": struct["summary"],
            "manufacturable": manu["manufacturable"],
            "manufacturability": manu,
            "mass_balance_error": mb,
            "max_cell_divergence": float(np.max(np.abs(div[lat]))) / Q if np.any(lat) else 0.0,
            "energy_balance_error": heat["energy_balance_error"],
            "picard_iterations": 0,
            "picard_converged": True,
            "diagnostics": {
                **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in hyd.items()},
                "U_eff": float(U_eff),
                "rib_efficiency": float(eta_rib),
                "rib_conductance_per_length": float(G_rib),
                "area_enhancement": float(U_eff / hyd["h_c"]),
                "sleeve_spreading_efficiency": float(math.tanh(x) / x) if x > 0 else 1.0,
                "heat_to_manifold_plenums": heat["heat_manifolds"],
                "manifold_htc": heat["h_m"],
                "grid": g.shape,
            },
            "wall_time": time.perf_counter() - t0,
        }
        if return_fields:
            out["fields"] = heat["fields"]
        return out

    # ------------------------------------------------------------------ pieces
    def _conductance(self, d: ChannelDesign, h_c: float) -> tuple[float, float, float]:
        cfg, o = self.cfg, self.options
        k = cfg.material.conductivity
        h = cfg.jacket.lattice_gap
        G_tip = 0.0
        if o.outer_wall_fin:
            t_o = cfg.jacket.outer_wall_thickness
            m_o = math.sqrt(h_c / (k * t_o))
            G_tip = 2.0 * k * t_o * m_o * math.tanh(m_o * d.channel_width / 2.0)
        G, eta = rib_conductance(h_c, k, d.rib_thickness, h, G_tip)
        U = (h_c * d.channel_width + float(G)) / d.period
        return U, float(eta), float(G)

    def _face_flows(self, d: ChannelDesign, g: JacketGrid, Q: float) -> tuple[np.ndarray, np.ndarray]:
        """Volumetric flow through +s faces (n_s, n_z) and z faces (n_s, n_z + 1) [m^3/s]."""
        ns, nz = g.shape
        qz = np.zeros((ns, nz + 1))
        if d.layout == "helical":
            qs = np.outer(np.ones(ns), Q * g.dz / d.pitch)  # Q/P per unit axial width, along +s
            qz[:] = (Q * g.ds / g.circumference)[:, None]  # Q/C per unit arc length, along +z
            return qs, qz
        # circumferential: +s on the arc after the inlet slot, -s on the arc before it
        kind = g.kind
        qs = np.zeros((ns, nz))
        per_z = Q * g.dz / (2.0 * g.axial_length)
        in_cols = np.flatnonzero(np.any(kind == INLET, axis=1))
        out_cols = np.flatnonzero(np.any(kind == OUTLET, axis=1))
        a, b = in_cols.max(), out_cols.min()  # +s arc: faces a .. b-1 (cells a+1 .. b-1 are lattice)
        c, e = out_cols.max(), ns  # -s arc: faces c .. ns-1
        qs[a:b] = per_z[None, :]
        qs[c:e] = -per_z[None, :]
        return qs, qz

    def _divergence(self, g: JacketGrid, qs: np.ndarray, qz: np.ndarray) -> np.ndarray:
        """Net outflow per cell (lattice cells only meaningful); plenums carry the external flow."""
        div = qs - np.roll(qs, 1, axis=0) + qz[:, 1:] - qz[:, :-1]
        return div.ravel()

    def _solve_heat(self, g: JacketGrid, layout: str, U_eff: float, qs: np.ndarray, qz: np.ndarray,
                    Q: float) -> dict[str, Any]:
        """Coolant T_f (channel cells) + sleeve T_w (all cells) + plenum temperatures; one sparse solve."""
        cfg, o = self.cfg, self.options
        cool, mat, jac = cfg.coolant, cfg.material, cfg.jacket
        rcp = cool.density * cool.specific_heat
        ns, nz = g.shape
        n = g.n_cells
        kind = g.kind.ravel()
        lat = np.flatnonzero(kind == LATTICE)
        nL = len(lat)
        pos = -np.ones(n, dtype=np.int64)
        pos[lat] = np.arange(nL)
        idx = np.arange(n).reshape(ns, nz)
        area = g.area.ravel()
        plen = [k for k in (INLET, OUTLET) if np.any(kind == k)]
        iP = {k: nL + n + m for m, k in enumerate(plen)}
        N = nL + n + len(plen)
        h_m = manifold_htc_default(cfg) if o.manifold_htc is None else float(o.manifold_htc)
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        b = np.zeros(N)

        def add(r: Any, c: Any, v: Any) -> None:
            r, c, v = np.broadcast_arrays(np.asarray(r), np.asarray(c), np.asarray(v, float))
            rows.append(r.ravel()); cols.append(c.ravel()); vals.append(v.ravel())

        def couple(ia: Any, ib: Any, G: Any) -> None:
            add(ia, ia, G); add(ia, ib, -G); add(ib, ib, G); add(ib, ia, -G)

        def unk(cell: np.ndarray) -> np.ndarray:
            """Coolant unknown of a cell: its T_f (channel) or its plenum temperature."""
            cell = np.asarray(cell)
            out = np.empty(cell.shape, dtype=np.int64)
            k = kind[cell]
            m = k == LATTICE
            out[m] = pos[cell[m]]
            for pk in plen:
                out[k == pk] = iP[pk]
            return out

        def iW(cell: Any) -> np.ndarray:
            return nL + np.asarray(cell)

        def advect(up: np.ndarray, dn: np.ndarray, F: np.ndarray) -> None:
            """Upwind flux F >= 0 from cell 'up' to cell 'dn' (between distinct unknowns)."""
            m = F > 0
            up, dn, F = up[m], dn[m], F[m]
            u_up, u_dn = unk(up), unk(dn)
            keep = u_up != u_dn  # flow inside one plenum is internal
            up, dn, F, u_up, u_dn = up[keep], dn[keep], F[keep], u_up[keep], u_dn[keep]
            add(u_up, u_up, F); add(u_dn, u_up, -F)

        # s faces (periodic)
        left = idx.ravel()
        right = np.roll(idx, -1, axis=0).ravel()
        Fs = rcp * qs.ravel()
        advect(left, right, np.maximum(Fs, 0.0))
        advect(right, left, np.maximum(-Fs, 0.0))
        # interior z faces
        Fz = rcp * qz[:, 1:-1].ravel()
        lo, hi = idx[:, :-1].ravel(), idx[:, 1:].ravel()
        advect(lo, hi, np.maximum(Fz, 0.0))
        advect(hi, lo, np.maximum(-Fz, 0.0))
        # boundary z faces: inflow at z = 0 (T_in), outflow at z = L_ax (helical ring manifolds)
        T_in = cfg.operating.inlet_temperature
        F0 = rcp * qz[:, 0]
        FL = rcp * qz[:, -1]
        c0, cL = idx[:, 0], idx[:, -1]
        if np.any(F0 < 0) or np.any(FL < 0):
            raise ValueError("boundary z faces must carry inflow at z = 0 and outflow at z = L_ax")
        np.add.at(b, unk(c0), F0 * T_in)  # inflow: upstream temperature T_in on the right-hand side
        add(unk(cL), unk(cL), FL)  # outflow leaves at the cell temperature
        # external flow through the plenums (circumferential layout): inflow at T_in, outflow at T_plenum
        Fext = rcp * Q
        if INLET in plen:
            b[iP[INLET]] += Fext * T_in
        if OUTLET in plen:
            add(iP[OUTLET], iP[OUTLET], Fext)
        # wall <-> coolant
        Al = area[lat]
        couple(iW(lat), pos[lat], U_eff * Al)
        for pk in plen:
            cells = np.flatnonzero(kind == pk)
            couple(iW(cells), np.full(len(cells), iP[pk]), h_m * area[cells])
        # sleeve conduction
        flux_bc = o.wall_condition == "flux"
        k_sl = mat.conductivity * jac.sleeve_thickness if (o.sleeve_conduction and flux_bc) else 0.0
        if k_sl > 0:
            ds, dz = g.ds, g.dz
            dist_s = 0.5 * (ds + np.roll(ds, -1))
            couple(iW(left), iW(right), k_sl * np.repeat(1.0 / dist_s, nz) * np.tile(dz, ns))
            dist_z = 0.5 * (dz[:-1] + dz[1:])
            couple(iW(lo), iW(hi), k_sl * np.tile(1.0 / dist_z, ns) * np.repeat(ds, nz - 1))
        q_on_rs = np.tile(self._q_cell[layout], ns)
        q_in = q_on_rs * jac.stator_radius / g.r_mean
        if flux_bc:
            b[iW(np.arange(n))] = q_in * area
            A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
        elif o.wall_condition == "temperature":
            if o.wall_temperature is None:
                raise ValueError("wall_condition='temperature' needs wall_temperature")
            r_ = np.concatenate(rows); c_ = np.concatenate(cols); v_ = np.concatenate(vals)
            wrows = iW(np.arange(n))
            keep = ~np.isin(r_, wrows)
            r_ = np.concatenate([r_[keep], wrows]); c_ = np.concatenate([c_[keep], wrows])
            v_ = np.concatenate([v_[keep], np.ones(n)])
            b[wrows] = o.wall_temperature
            A = sp.csr_matrix((v_, (r_, c_)), shape=(N, N))
        else:
            raise ValueError(f"wall_condition must be 'flux' or 'temperature', got {o.wall_condition!r}")
        T = spla.spsolve(A.tocsc(), b)
        T_f = T[:nL]
        T_w = T[nL:nL + n]
        T_pl = {pk: float(T[iP[pk]]) for pk in plen}
        if layout == "helical":
            T_out = float(np.sum(FL * T_f[pos[cL]]) / np.sum(FL))
        else:
            T_out = T_pl[OUTLET]
        T_wall = T_w + (q_on_rs * jac.sleeve_thickness / mat.conductivity if flux_bc else 0.0)
        k_max = int(np.argmax(T_wall))
        S, Z = g.meshgrid()
        wall_out = float(np.sum(U_eff * Al * (T_w[lat] - T_f)))
        heat_m = {("inlet" if pk == INLET else "outlet"): float(np.sum(h_m * area[kind == pk] * (T_w[kind == pk] - T_pl[pk])))
                  for pk in plen}
        heat_input = float(np.sum(q_in * area)) if flux_bc else wall_out + sum(heat_m.values())
        removed = rcp * Q * (T_out - T_in)
        Tf_full = np.full(n, np.nan)
        Tf_full[lat] = T_f
        for pk in plen:
            Tf_full[kind == pk] = T_pl[pk]
        Ueff_full = np.where(kind == LATTICE, U_eff, h_m)
        fields = ChannelFields(grid=g, T_f=Tf_full.reshape(ns, nz), T_w=T_w.reshape(ns, nz),
                               T_wall=T_wall.reshape(ns, nz), q_in=q_in.reshape(ns, nz),
                               U_eff=Ueff_full.reshape(ns, nz), face_flow_s=qs.copy(), face_flow_z=qz.copy())
        return dict(T_wall_max=float(T_wall[k_max]), T_wall_max_active=float(np.max(T_wall[lat])),
                    T_wall_mean=float(np.average(T_wall, weights=area)), argmax_s=float(S.ravel()[k_max]),
                    argmax_z=float(Z.ravel()[k_max]), T_out=T_out, heat_input=heat_input,
                    energy_balance_error=abs(removed - heat_input) / heat_input, heat_manifolds=heat_m, h_m=h_m,
                    fields=fields)

    def _active_area(self, g: JacketGrid) -> float:
        return float(np.sum(g.area[g.kind == LATTICE]))

    def _structure(self, d: ChannelDesign, g: JacketGrid, hyd: Mapping[str, float]) -> dict[str, Any]:
        cfg, o = self.cfg, self.options
        jac, mat, loads = cfg.jacket, cfg.material, cfg.loads
        h = jac.lattice_gap
        r_m, r_i = g.r_mean, jac.lattice_inner_radius
        A = self._active_area(g)
        f_r = d.rib_fraction
        tau_T = loads.torque / (r_m * A) * (r_m / r_i) ** 2 / f_r  # in-plane rib shear (Task 9 sandwich, uniform core)
        tau_rz = loads.thrust / A * (r_m / r_i)  # mean core shear (thrust)
        Vp = tau_rz * d.period  # transverse load per unit rib length
        sig_b = 3.0 * Vp * h / d.rib_thickness**2  # fixed-guided strip, rigid faces
        tau_V = 1.5 * Vp / d.rib_thickness
        p = loads.coolant_pressure_gauge + hyd["delta_p"]
        t_o = jac.outer_wall_thickness
        R_o = jac.lattice_outer_radius + t_o / 2.0
        k_hoop = mat.youngs_modulus * t_o / R_o**2
        k_rib = mat.youngs_modulus * f_r / h
        share = k_rib / (k_rib + k_hoop)
        sig_p = p * share / f_r
        Kt = o.root_stress_factor
        vm = Kt * math.sqrt((sig_p + sig_b) ** 2 + 3.0 * (tau_T**2 + tau_V**2))
        rib_margin = mat.allowable_stress / vm
        sig_hoop = (1.0 - share) * p * R_o / t_o
        sig_chan = p * d.channel_width**2 / (2.0 * t_o**2)
        sig_slot = p * (cfg.manifolds.width * R_o / r_m) ** 2 / (2.0 * t_o**2) if d.layout == "circumferential" else 0.0
        sig_ow = max(sig_hoop, sig_chan, sig_slot)
        summary = dict(sigma_vm_max=vm, tau_torque=tau_T, tau_thrust=tau_V, sigma_thrust_bending=sig_b,
                       sigma_pressure=sig_p, pressure=p, rib_pressure_share=share, root_stress_factor=Kt,
                       outer_wall_hoop=sig_hoop, outer_wall_channel_bending=sig_chan, outer_wall_slot_bending=sig_slot)
        return dict(rib_margin=rib_margin, outer_wall_margin=mat.allowable_stress / sig_ow, summary=summary)

    def _mass(self, d: ChannelDesign, g: JacketGrid) -> dict[str, float]:
        cfg = self.cfg
        jac, mat = cfg.jacket, cfg.material
        A = self._active_area(g)
        m_rib = mat.density * jac.lattice_gap * d.rib_fraction * A
        L_ax = jac.axial_length
        r_s, t_sl = jac.stator_radius, jac.sleeve_thickness
        r_o, t_o = jac.outer_radius, jac.outer_wall_thickness
        m_sl = mat.density * math.pi * ((r_s + t_sl) ** 2 - r_s**2) * L_ax
        m_ow = mat.density * math.pi * (r_o**2 - (r_o - t_o) ** 2) * L_ax
        v_cool = jac.lattice_gap * ((1.0 - d.rib_fraction) * A + float(np.sum(g.area[g.kind != LATTICE])))
        return dict(total=m_rib + m_sl + m_ow, ribs=m_rib, walls=m_sl + m_ow, coolant_volume=v_cool)

    def _manufacturability(self, d: ChannelDesign) -> dict[str, Any]:
        man = self.cfg.manufacturing
        rib_ok = d.rib_thickness >= man.min_wall_thickness - 1e-12
        ch_ok = d.channel_width >= man.min_pore_size - 1e-12
        return dict(within_bounds=bool(rib_ok and ch_ok), rib_ok=bool(rib_ok), channel_ok=bool(ch_ok),
                    rib_thickness=d.rib_thickness, channel_width=d.channel_width,
                    manufacturable=bool(rib_ok and ch_ok))


# =============================================================================
# B2 / B3 lattice designs (Task 9 JacketModel)
# =============================================================================
def path_coordinate(grid: JacketGrid, cfg: Any) -> np.ndarray:
    """xi(s) in [0, 1] along each half-annulus path, 0 at the inlet slot edge, 1 at the outlet slot edge.

    Both paths are mapped symmetrically (theta and -theta give the same xi); cells inside the
    slots get 0 (inlet) or 1 (outlet). Returns an (n_s,) array for the grid's s cells.
    """
    r_m = grid.r_mean
    half_w = cfg.manifolds.width / 2.0 / r_m
    th = np.mod(grid.theta_plot_deg() * math.pi / 180.0 - cfg.manifolds.inlet_angle + math.pi, 2 * math.pi) - math.pi
    span = abs(math.remainder(cfg.manifolds.outlet_angle - cfg.manifolds.inlet_angle, 2 * math.pi))
    return np.clip((np.abs(th) - half_w) / (span - 2.0 * half_w), 0.0, 1.0)


def _psi_z(z: np.ndarray, L_ax: float) -> np.ndarray:
    zeta = np.asarray(z) / L_ax
    return 0.5 * (12.0 * (zeta - 0.5) ** 2 - 1.0)


@dataclass(frozen=True)
class UniformLatticeDesign:
    """Uniform lattice (B2: w = 0). Lengths in m."""

    rho: float
    L: float
    w: float = 0.0
    a_z: float = 1.0

    def fields(self, model: JacketModel) -> dict[str, Any]:
        return dict(rho=self.rho, w=self.w, L=self.L, a_z=self.a_z)

    def to_dict(self) -> dict[str, Any]:
        return {"family": "B2", "type": "uniform_lattice", **asdict(self)}


@dataclass(frozen=True)
class GradedDensityDesign:
    """Density-only graded lattice (B3): rho(s, z) = interp(xi, rho_path) + amp_z psi(z); w and L fixed."""

    rho_path: tuple[float, ...]
    amp_z: float
    L: float
    w: float = 0.0
    a_z: float = 1.0

    def rho_field(self, model: JacketModel) -> np.ndarray:
        g, cfg = model.grid, model.cfg
        xi = path_coordinate(g, cfg)
        knots = np.linspace(0.0, 1.0, len(self.rho_path))
        r_s = np.interp(xi, knots, np.asarray(self.rho_path, float))
        return r_s[:, None] + self.amp_z * _psi_z(g.z_centres, g.axial_length)[None, :]

    def fields(self, model: JacketModel) -> dict[str, Any]:
        return dict(rho=self.rho_field(model), w=self.w, L=self.L, a_z=self.a_z)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rho_path"] = list(self.rho_path)
        return {"family": "B3", "type": "graded_density_lattice", **d}


def design_from_dict(d: Mapping[str, Any]) -> Any:
    """Inverse of ``design.to_dict()`` (JSON round trip)."""
    d = dict(d)
    typ = d.pop("type")
    d.pop("family", None)
    if typ == "channel":
        return ChannelDesign(**d)
    if typ == "uniform_lattice":
        return UniformLatticeDesign(**d)
    if typ == "graded_density_lattice":
        d["rho_path"] = tuple(d["rho_path"])
        return GradedDensityDesign(**d)
    raise ValueError(f"unknown design type {typ!r}")


_MODELS: dict[str, Any] = {}


def _lattice_model() -> JacketModel:
    if "lattice" not in _MODELS:
        _MODELS["lattice"] = JacketModel(options=JacketOptions(check_bounds="ignore"))
    return _MODELS["lattice"]


def _channel_model() -> ChannelJacketModel:
    if "channel" not in _MODELS:
        _MODELS["channel"] = ChannelJacketModel()
    return _MODELS["channel"]


def evaluate(design: Any, flow_rate: float | None = None, *, return_fields: bool = False,
             model: Any = None) -> dict[str, Any]:
    """Evaluate any baseline design with the Task 9 output dict (+ ``T_wall_max_active``, ``R_active``, ``design``)."""
    if isinstance(design, Mapping):
        design = design_from_dict(design)
    if isinstance(design, ChannelDesign):
        return (model or _channel_model()).evaluate(design, flow_rate, return_fields=return_fields)
    if isinstance(design, (UniformLatticeDesign, GradedDensityDesign)):
        m = model or _lattice_model()
        out = m.evaluate(**design.fields(m), flow_rate=flow_rate, return_fields=return_fields)
        out["design"] = design.to_dict()
        out["T_wall_max_active"] = out["T_wall_max_lattice"]
        out["R_active"] = (out["T_wall_max_lattice"] - out["T_in"]) / m.cfg.motor.heat_to_jacket
        return out
    raise TypeError(f"unsupported design {type(design).__name__}")


# =============================================================================
# Tuning
# =============================================================================
SUMMARY_KEYS = ("thermal_resistance", "R_active", "delta_p", "pump_power", "mass", "min_structural_margin",
                "manufacturable", "T_wall_max", "T_out", "energy_balance_error")


def summarize(out: Mapping[str, Any]) -> dict[str, Any]:
    """Flat row of the comparison quantities of one evaluation."""
    row = {k: out[k] for k in SUMMARY_KEYS}
    row["flow_lpm"] = out["flow_rate"] * 6e4
    row["lattice_margin"] = out["lattice_margin"]
    row["outer_wall_margin"] = out["outer_wall_margin"]
    d = out["diagnostics"]
    for k in ("reynolds", "dean", "regime", "h_c", "U_eff", "nusselt", "rib_efficiency"):
        if k in d:
            row[k] = d[k]
    if "reynolds_median" in d:
        row["reynolds"] = d["reynolds_median"]
        row["U_eff"] = d["U_eff_mean"]
    return row


@dataclass
class TuningResult:
    family: str
    best: Any  # design
    best_output: dict[str, Any]
    table: pd.DataFrame  # every candidate evaluated
    rule: dict[str, Any] = field(default_factory=dict)


def _feasible(df: pd.DataFrame) -> pd.Series:
    return df["manufacturable"].astype(bool) & (df["min_structural_margin"] >= 1.0)


def select_best(df: pd.DataFrame, pump_power_cap: float | None, objective: str = "thermal_resistance") -> int:
    """Index of the minimum ``objective`` among feasible rows with pump_power <= cap (None: no cap)."""
    ok = _feasible(df)
    if pump_power_cap is not None:
        ok &= df["pump_power"] <= pump_power_cap
    if not ok.any():
        raise ValueError("no feasible candidate satisfies the pump-power cap")
    return int(df.loc[ok, objective].idxmin())


def pareto_front(df: pd.DataFrame, objectives: Sequence[str] = ("thermal_resistance", "pump_power", "mass")) -> pd.Series:
    """Boolean mask of the feasible, non-dominated rows (all objectives minimized)."""
    ok = _feasible(df).to_numpy()
    F = df[list(objectives)].to_numpy(float)
    mask = np.zeros(len(df), bool)
    idx = np.flatnonzero(ok)
    for i in idx:
        dom = np.all(F[idx] <= F[i], axis=1) & np.any(F[idx] < F[i], axis=1)
        mask[i] = not np.any(dom)
    return pd.Series(mask, index=df.index)


B1_RIBS = (0.35e-3, 0.5e-3, 0.75e-3, 1.0e-3, 1.5e-3, 2.0e-3)
B1_PASSES = (3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 25, 30, 35, 40)
B1_STARTS = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 40)


def b1_candidates(cfg: Any, layout: str = "helical", *, rib_thickness: Iterable[float] = B1_RIBS,
                  n_pass: Iterable[int] = B1_PASSES, n_starts: Iterable[int] = B1_STARTS) -> list[ChannelDesign]:
    """Grid of B1 designs: N_pass passes over L_ax (w_c = L_ax/N_pass - t_r), rib thickness, starts.

    Rib thickness starts at the config minimum wall (0.35 mm, the same limit as the lattice walls); keeps
    w_c >= min pore size and (helical) n_starts <= N_pass (every start makes at least one full pass).
    """
    L_ax = cfg.jacket.axial_length
    n_pass = list(n_pass)
    out = []
    for N in n_pass:
        for t in rib_thickness:
            w = L_ax / N - t
            if w < cfg.manufacturing.min_pore_size - 1e-12:
                continue
            if layout == "helical":
                for n in n_starts:
                    if n <= N:
                        out.append(ChannelDesign(w, t, int(n), "helical"))
            else:
                out.append(ChannelDesign(w, t, 1, "circumferential"))
    return out


def _run(designs: Sequence[Any], flow_rate: float | None, model: Any = None) -> pd.DataFrame:
    rows = []
    for d in designs:
        out = evaluate(d, flow_rate, model=model)
        row = summarize(out)
        row["design"] = json.dumps(d.to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def tune_b1(cfg: Any = None, *, layout: str = "helical", pump_power_cap: float | None = None,
            flow_rate: float | None = None, model: ChannelJacketModel | None = None, **grid_kw: Any) -> TuningResult:
    """Grid search over (N_pass, t_r, n_starts); best = ``select_best`` rule."""
    model = model or (ChannelJacketModel(cfg) if cfg is not None else _channel_model())
    cfg = model.cfg
    designs = b1_candidates(cfg, layout, **grid_kw)
    df = _run(designs, flow_rate, model)
    for k in ("channel_width", "rib_thickness", "n_starts"):
        df[k] = [getattr(d, k) for d in designs]
    df["pitch"] = [d.pitch if d.layout == "helical" else np.nan for d in designs]
    df["n_pass"] = [round(cfg.jacket.axial_length / d.period) for d in designs]
    i = select_best(df, pump_power_cap)
    best = designs[i]
    return TuningResult("B1" if layout == "helical" else "B1s", best, evaluate(best, flow_rate, model=model), df,
                        dict(pump_power_cap=pump_power_cap, n_candidates=len(df)))


def tune_b2(cfg: Any = None, *, pump_power_cap: float | None = None, flow_rate: float | None = None,
            rho_values: Iterable[float] | None = None, L_values: Iterable[float] | None = None,
            w: float = 0.0, model: JacketModel | None = None) -> TuningResult:
    """Grid search of the uniform lattice (w fixed, default gyroid) over (rho*, L) inside the design bounds.

    Candidates with L < L_min(w, rho*) are evaluated too (flagged non-manufacturable by the model).
    """
    model = model or (JacketModel(cfg, JacketOptions(check_bounds="ignore")) if cfg is not None else _lattice_model())
    cfg = model.cfg
    bnd = cfg.design_bounds
    rho_values = list(rho_values) if rho_values is not None else list(np.round(np.arange(bnd.relative_density.lo,
                                                                                         bnd.relative_density.hi + 1e-9, 0.025), 4))
    L_values = list(L_values) if L_values is not None else list(np.round(np.arange(bnd.cell_size.lo,
                                                                                   bnd.cell_size.hi + 1e-12, 0.25e-3), 6))
    designs = [UniformLatticeDesign(float(r), float(L), w) for r in rho_values for L in L_values]
    df = _run(designs, flow_rate, model)
    df["rho"] = [d.rho for d in designs]
    df["L"] = [d.L for d in designs]
    i = select_best(df, pump_power_cap)
    best = designs[i]
    return TuningResult("B2", best, evaluate(best, flow_rate, model=model), df,
                        dict(pump_power_cap=pump_power_cap, n_candidates=len(df)))


def tune_b3(cfg: Any = None, *, L: float, pump_power_cap: float | None, flow_rate: float | None = None,
            n_knots: int = 4, start: GradedDensityDesign | None = None, n_random: int = 64, maxfev: int = 200,
            seed: int = 0, model: JacketModel | None = None) -> TuningResult:
    """Density-only grading with w = 0 and L fixed: Sobol screening + Nelder-Mead polish.

    Variables: rho at ``n_knots`` points along the path and the axial amplitude amp_z. Objective: R at the
    nominal flow; constraints (manufacturable incl. bounds and |grad rho*| L, margin >= 1, pump power <= cap)
    as penalties. Every evaluation is recorded in ``table``.
    """
    from scipy.optimize import minimize
    from scipy.stats import qmc

    model = model or (JacketModel(cfg, JacketOptions(check_bounds="ignore")) if cfg is not None else _lattice_model())
    cfg = model.cfg
    bnd = cfg.design_bounds.relative_density
    lim = cfg.manufacturing.max_density_change_per_cell
    a_max = lim * cfg.jacket.axial_length / (6.0 * L)  # |d psi/dz| <= 6/L_ax
    lo = np.r_[np.full(n_knots, bnd.lo), -a_max]
    hi = np.r_[np.full(n_knots, bnd.hi), a_max]
    rows: list[dict[str, Any]] = []
    cache: dict[tuple, float] = {}

    def make(x: np.ndarray) -> GradedDensityDesign:
        return GradedDensityDesign(tuple(float(v) for v in np.round(x[:n_knots], 6)), float(np.round(x[n_knots], 6)), L)

    def penalty(out: Mapping[str, Any]) -> float:
        man = out["manufacturability"]
        pen = 0.0
        if not man["within_bounds"]:
            pen += 1.0
        if not man["cell_size_ok"]:
            pen += 1.0 - man["cell_size_ratio_min"]
        pen += max(0.0, man["gradient_max"] / man["gradient_limit"] - 1.0)
        pen += max(0.0, 1.0 - out["min_structural_margin"])
        if pump_power_cap is not None:
            pen += max(0.0, out["pump_power"] / pump_power_cap - 1.0)
        return pen

    def f(x: np.ndarray) -> float:
        x = np.clip(x, lo, hi)
        d = make(x)
        key = (d.rho_path, d.amp_z)
        if key in cache:
            return cache[key]
        out = evaluate(d, flow_rate, model=model)
        row = summarize(out)
        row.update(design=json.dumps(d.to_dict()), gradient_max=out["manufacturability"]["gradient_max"],
                   rho_min=float(np.min(d.rho_field(model)[model.grid.lattice])),
                   rho_max=float(np.max(d.rho_field(model)[model.grid.lattice])),
                   **{f"rho_{k}": v for k, v in enumerate(d.rho_path)}, amp_z=d.amp_z)
        # the field must stay inside the density bounds (the model flags it; record it explicitly)
        val = out["thermal_resistance"] * (1.0 + 10.0 * penalty(out))
        row["objective"] = val
        rows.append(row)
        cache[key] = val
        return val

    if start is not None and len(start.rho_path) == n_knots:
        x0 = np.r_[start.rho_path, start.amp_z]
    else:
        x0 = np.r_[np.full(n_knots, start.rho_path[0] if start is not None else 0.35), 0.0]
    best_x, best_v = x0, f(x0)
    if n_random > 0:
        S = qmc.Sobol(n_knots + 1, scramble=True, seed=seed).random(n_random)
        for s in S:
            x = lo + s * (hi - lo)
            v = f(x)
            if v < best_v:
                best_x, best_v = x, v
    scale = hi - lo
    res = minimize(lambda u: f(lo + u * scale), (best_x - lo) / scale, method="Nelder-Mead",
                   options=dict(maxfev=maxfev, xatol=1e-3, fatol=1e-7, initial_simplex=None))
    df = pd.DataFrame(rows)
    df["feasible_cap"] = _feasible(df) & ((df["pump_power"] <= pump_power_cap) if pump_power_cap is not None else True)
    i = select_best(df, pump_power_cap)
    best = design_from_dict(json.loads(df.loc[i, "design"]))
    return TuningResult("B3", best, evaluate(best, flow_rate, model=model), df,
                        dict(pump_power_cap=pump_power_cap, n_candidates=len(df), L=L, nelder_mead_nfev=int(res.nfev)))
