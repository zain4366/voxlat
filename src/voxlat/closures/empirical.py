"""Literature closures for the jacket model: interstitial heat transfer and inertial drag (Task 8).

What this module provides
-------------------------
VoxLat computes the creeping-flow permeability K, the effective conductivity and the
stiffness of every lattice itself (Tasks 2-7). Two closures of the two-equation
(Darcy-Forchheimer + local-thermal-non-equilibrium) jacket model of Task 9 cannot be
computed on a laptop and are taken from the literature:

1. the inertial (Forchheimer) drag coefficient C_F, and
2. the interstitial (solid-fluid) heat-transfer coefficient h_sf, through Nu(Re, Pr, phi).

Governing relations and definitions (SI units throughout)
--------------------------------------------------------
Momentum (Darcy-Forchheimer, Ward form with superficial velocity U_s):

    -grad p = (mu / K) U_s + (rho C_F / sqrt(K)) |U_s| U_s                         (1)

Geometry and dimensionless groups (as in Gajetti et al. 2025 and Savoldi et al. 2026):

    phi   = fluid volume fraction = 1 - rho*                (porosity)
    a_sf  = wetted area per TOTAL volume [1/m]              (Task 1, ``a_sf_L / L``)
    D_h   = 4 V_f / A_wet = 4 phi / a_sf                     (hydraulic diameter)
    U_b   = U_s / phi                                        (pore / interstitial velocity)
    Re    = Re_Dh = U_b D_h / nu = 4 U_s / (a_sf nu)          (pore-velocity, D_h based)
    Re_K  = U_s sqrt(K) / nu,      Fo = C_F Re_K              (Fo = inertial / viscous drag)
    f     = (dp/dx) D_h / (rho U_b^2 / 2)                     ("Darcy-type" friction factor = 4 x Fanning)

Inserting (1) into the definition of f gives the exact identity (Savoldi et al. 2026, Eq. 23)

    f Re = 2 phi D_h^2 / K  +  2 phi^2 C_F (D_h / sqrt(K)) Re                    (2)

Heat transfer: h_sf = Nu k_f / D_h, defined as the surface-averaged wall flux divided by
(T_wall - T_bulk) on the whole solid-fluid interface (Savoldi et al. 2026, Eqs. 24-28). The
volumetric exchange coefficient of the two-equation model is h_sf a_sf [W/(m^3 K)].

Recommended closures (STATUS.md, Task 8, gives the reasons)
-----------------------------------------------------------
* C_F: Gajetti, Boccardo, Savoldi & Marocco (2025), skeletal gyroid and diamond,
  C_F = b phi^m (``forchheimer_coefficient``, ``source="gajetti2025"``); blends 0 < w < 1 by
  log-linear interpolation in w (an assumption). Alternatives: ``"savoldi2026"`` (gyroid
  only, fitted to phi = 0.7) and ``"gajetti2025_table"`` (their Table 2, held constant
  outside phi = 0.3-0.6; brackets the diamond extrapolation, which is the largest C_F risk).
* Nu: Savoldi, Cammi, Gajetti & Marocco (2026) modified Reynolds analogy, correlation 1,
  St = 0.0267 phi^0.19 f (Pr = 1), with f from (2) evaluated with OUR K and a_sf and the
  recommended C_F, extended to Pr != 1 by the Chilton-Colburn factor Pr^(1/3):

      Nu = St(Pr = 1) Re Pr^(1/3) = 0.0267 phi^0.19 (f Re) Pr^(1/3)                  (3)

  (``nusselt_interstitial``, ``interstitial_heat_transfer``). Correlation 2 of the same paper
  (``variant=2``) is the low-h bound for sensitivity runs.

Which K to use: the literature correlations are bulk (periodic-cell) closures. Use the BULK
K (``ClosureModel``) in (2), in Fo and in the inertial term of (1); apply the Task 7 finite-gap
factor to the viscous (Darcy) term of the device model only. For anisotropic K use the
permeability along the local flow direction, 1 / (e . K^-1 . e).

Validity and extrapolation warnings
-----------------------------------
Every correlation carries its published validity box (``LITERATURE[key].validity``: Re,
porosity, Pr, topology). Evaluating outside it emits a ``CorrelationRangeWarning`` (or raises
``CorrelationRangeError`` with ``on_extrapolation="raise"``; ``"ignore"`` silences it).
``range_report`` returns the fraction of points outside each bound, for logging in Task 9.
For the VoxLat jacket (phi = 0.5-0.8, Re_Dh ~ 30-600, Pr = 20.7) the recommended closures are
used OUTSIDE their fitted Re, Pr and (for phi > 0.6-0.7) porosity ranges; these warnings are
expected and are the content of the manuscript's limitations paragraph.

References (full list in STATUS.md, Task 8)
-------------------------------------------
Gajetti E., Boccardo G., Savoldi L., Marocco L. (2025) Int. J. Heat Mass Transf. 252:127439,
    doi:10.1016/j.ijheatmasstransfer.2025.127439 (open access).
Savoldi L., Cammi A., Gajetti E., Marocco L. (2026) Int. J. Heat Fluid Flow 121:110631,
    doi:10.1016/j.ijheatfluidflow.2026.110631 (open access).
Gnielinski V. (1981) Int. Chem. Eng. 21(3); VDI Heat Atlas, 2nd ed. (2010) - packed beds.
Wakao N., Kaguei S. (1982) Heat and Mass Transfer in Packed Beds, Gordon & Breach.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np

__all__ = [
    "CorrelationRangeWarning",
    "CorrelationRangeError",
    "ValidityBox",
    "CorrelationInfo",
    "LITERATURE",
    "RECOMMENDED",
    "GAJETTI2025_POWER_LAWS",
    "GAJETTI2025_TABLE2",
    "SAVOLDI2026_HYDRAULICS",
    "SAVOLDI2026_STANTON",
    "RATHORE2023_CF",
    "literature_table",
    "check_range",
    "range_report",
    "hydraulic_diameter",
    "equivalent_particle_diameter",
    "pore_velocity",
    "reynolds_hydraulic",
    "reynolds_permeability",
    "forchheimer_number",
    "gajetti2025_permeability",
    "gajetti2025_forchheimer",
    "savoldi2026_hydraulic_diameter",
    "savoldi2026_permeability",
    "savoldi2026_forchheimer",
    "forchheimer_coefficient",
    "pressure_gradient",
    "friction_factor_re",
    "friction_factor",
    "stanton_savoldi2026",
    "nusselt_interstitial",
    "InterstitialHeatTransfer",
    "interstitial_heat_transfer",
    "nusselt_gnielinski_packed_bed",
    "nusselt_wakao_kaguei",
    "nusselt_reynolds2023",
]

OnExtrapolation = Literal["warn", "raise", "ignore"]


# =============================================================================
# Validity bookkeeping
# =============================================================================
class CorrelationRangeWarning(UserWarning):
    """A literature correlation was evaluated outside its published validity box."""


class CorrelationRangeError(ValueError):
    """Raised instead of the warning when ``on_extrapolation="raise"``."""


@dataclass(frozen=True)
class ValidityBox:
    """Published validity box of a correlation (None = not stated / not restricted).

    ``re`` refers to the Reynolds number in the correlation's OWN definition (``re_definition``
    in ``CorrelationInfo``); ``topologies`` lists the TPMS (or 'any') the data cover.
    """

    re: tuple[float, float] | None = None
    porosity: tuple[float, float] | None = None
    pr: tuple[float, float] | None = None
    topologies: tuple[str, ...] = ("any",)


@dataclass(frozen=True)
class CorrelationInfo:
    """One row of the Task 8 literature table."""

    key: str
    quantity: str  # "C_F", "K", "Nu", "St", "f", ...
    citation: str
    doi: str
    tpms: str  # e.g. "gyroid, diamond (skeletal)"
    method: str
    fluid: str
    re_definition: str
    formula: str
    validity: ValidityBox
    implemented: bool
    access: str  # how we verified the coefficients
    notes: str = ""

    def as_row(self) -> dict[str, Any]:
        v = self.validity

        def rng(r: tuple[float, float] | None) -> str:
            if r is None:
                return "not stated"
            return f"{r[0]:g}" if r[0] == r[1] else f"{r[0]:g}-{r[1]:g}"

        return {
            "key": self.key,
            "quantity": self.quantity,
            "source": self.citation,
            "doi": self.doi,
            "tpms": self.tpms,
            "method": self.method,
            "fluid": self.fluid,
            "Re_range": rng(v.re),
            "Re_definition": self.re_definition,
            "porosity_range": rng(v.porosity),
            "Pr_range": rng(v.pr),
            "formula": self.formula,
            "implemented": self.implemented,
            "access": self.access,
            "notes": self.notes,
        }


_TOL = 1e-9


def _as_array(x: Any) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=float))


def _outside(values: Any, bounds: tuple[float, float] | None) -> tuple[np.ndarray, float, float]:
    """Boolean mask of finite values outside ``bounds`` (+ min, max of the finite values)."""
    v = _as_array(values)
    v = v[np.isfinite(v)]
    if bounds is None or v.size == 0:
        return np.zeros(v.shape, bool), (float(v.min()) if v.size else math.nan), (
            float(v.max()) if v.size else math.nan
        )
    lo, hi = bounds
    scale = max(abs(lo), abs(hi), 1.0)
    mask = (v < lo - _TOL * scale) | (v > hi + _TOL * scale)
    return mask, float(v.min()), float(v.max())


def _topology_of(w: Any) -> list[str]:
    ws = _as_array(w)
    out = set()
    for x in ws[np.isfinite(ws)]:
        if x <= 1e-12:
            out.add("gyroid")
        elif x >= 1 - 1e-12:
            out.add("diamond")
        else:
            out.add("blend")
    return sorted(out)


def range_report(
    key: str,
    *,
    re: Any = None,
    porosity: Any = None,
    pr: Any = None,
    w: Any = None,
) -> dict[str, Any]:
    """Fraction of the given points outside each bound of ``LITERATURE[key].validity``.

    Returns ``{"re": {...}, "porosity": {...}, "pr": {...}, "topology": {...}}`` with, per
    dimension, ``bounds``, ``min``, ``max``, ``fraction_outside`` (absent when not given).
    """
    box = LITERATURE[key].validity
    report: dict[str, Any] = {}
    for name, values, bounds in (("re", re, box.re), ("porosity", porosity, box.porosity), ("pr", pr, box.pr)):
        if values is None:
            continue
        mask, vmin, vmax = _outside(values, bounds)
        report[name] = {
            "bounds": bounds,
            "min": vmin,
            "max": vmax,
            "fraction_outside": float(mask.mean()) if mask.size else 0.0,
        }
    if w is not None:
        topo = _topology_of(w)
        allowed = set(box.topologies)
        bad = [t for t in topo if "any" not in allowed and t not in allowed]
        report["topology"] = {"bounds": box.topologies, "present": topo, "outside": bad}
    return report


def check_range(
    key: str,
    *,
    re: Any = None,
    porosity: Any = None,
    pr: Any = None,
    w: Any = None,
    on_extrapolation: OnExtrapolation = "warn",
    stacklevel: int = 3,
) -> dict[str, Any]:
    """Warn (or raise) if any point lies outside the validity box of correlation ``key``.

    Returns the ``range_report``. Topology: blends (0 < w < 1) are outside every TPMS-specific
    box; the warning says which assumption is used instead.
    """
    if on_extrapolation not in ("warn", "raise", "ignore"):
        raise ValueError(f"on_extrapolation must be 'warn', 'raise' or 'ignore', got {on_extrapolation!r}")
    report = range_report(key, re=re, porosity=porosity, pr=pr, w=w)
    if on_extrapolation == "ignore":
        return report
    msgs = []
    for name in ("re", "porosity", "pr"):
        r = report.get(name)
        if r and r["fraction_outside"] > 0:
            lo, hi = r["bounds"]
            rng = f"{lo:g}" if lo == hi else f"[{lo:g}, {hi:g}]"
            msgs.append(
                f"{name} outside {rng} (data {r['min']:.4g}...{r['max']:.4g}, "
                f"{100 * r['fraction_outside']:.0f} % of points)"
            )
    topo = report.get("topology")
    if topo and topo["outside"]:
        msgs.append(f"topology {topo['outside']} not in {list(topo['bounds'])}")
    if msgs:
        text = f"{key} ({LITERATURE[key].citation}): extrapolation - " + "; ".join(msgs)
        if on_extrapolation == "raise":
            raise CorrelationRangeError(text)
        warnings.warn(text, CorrelationRangeWarning, stacklevel=stacklevel)
    return report


# =============================================================================
# Published coefficients (verified against the open-access full texts, see LITERATURE)
# =============================================================================
#: Gajetti et al. (2025), Table 3 (Eqs. 18-19): K/Lc^2 = a phi^n,  C_F = b phi^m,
#: valid 0.30 <= phi <= 0.60, 0.3 <= Re_Dh <= 100, skeletal (solid) TPMS, water at 20 C.
GAJETTI2025_POWER_LAWS: dict[str, dict[str, float]] = {
    "gyroid": {"a": 0.0156, "n": 2.78, "b": 0.0908, "m": -1.81},
    "diamond": {"a": 0.0106, "n": 2.89, "b": 0.0666, "m": -1.69},
    "split-p1": {"a": 0.00631, "n": 2.95, "b": 0.127, "m": -1.83},
    "split-p2": {"a": 0.00653, "n": 3.04, "b": 0.0941, "m": -2.39},
}

#: Gajetti et al. (2025), Table 2: CFD values (phi, K/Lc^2, C_F), Lc = 10 mm.
GAJETTI2025_TABLE2: dict[str, np.ndarray] = {
    "gyroid": np.array([[0.30, 5.6e-4, 0.81], [0.40, 1.2e-3, 0.53], [0.50, 2.2e-3, 0.31], [0.60, 3.9e-3, 0.20]]),
    "diamond": np.array([[0.30, 3.5e-4, 0.52], [0.40, 7.6e-4, 0.30], [0.50, 1.4e-3, 0.20], [0.60, 2.4e-3, 0.19]]),
    "split-p1": np.array([[0.30, 1.9e-4, 1.2], [0.40, 4.3e-4, 0.65], [0.50, 8.1e-4, 0.46], [0.60, 1.4e-3, 0.34]]),
    "split-p2": np.array([[0.30, 1.6e-4, 1.7], [0.40, 4.1e-4, 0.85], [0.50, 8.0e-4, 0.50], [0.60, 1.4e-3, 0.31]]),
}

#: Savoldi et al. (2026), Eqs. 34-36 (skeletal gyroid, 0.3 <= phi <= 0.7, 20 <= Re_Dh <= 100):
#: D_h/L = 4 phi / (c2 phi^2 + c1 phi + c0),  K/L^2 = a phi^n,  C_F = b phi^m.
SAVOLDI2026_HYDRAULICS: dict[str, float] = {
    "c2": -5.723,
    "c1": 5.729,
    "c0": 1.665,
    "a": 0.0157,
    "n": 2.72,
    "b": 0.0784,
    "m": -2.04,
}

#: Savoldi et al. (2026), Eqs. 37-40: St = A phi^n Re^(m0 + m1 phi) f  (Pr = 1).
#: variant 1 (Eq. 37): minimal, +-11 %; variant 2 (Eqs. 38-39): Re exponent 0.440 - phi, +-15 %;
#: variant 3 (Eq. 40): weak Re dependence, +-12 %.
SAVOLDI2026_STANTON: dict[int, dict[str, float]] = {
    1: {"A": 0.0267, "n": 0.19, "m0": 0.0, "m1": 0.0},
    2: {"A": 0.135, "n": 2.06, "m0": 0.440, "m1": -1.0},
    3: {"A": 0.034, "n": 0.20, "m0": -0.061, "m1": -0.0023},
}

#: Rathore, Mehta, Kumar & Asfer (2023), arXiv:2205.03591 Table (Type 1 = solid lattice):
#: (porosity, K [m^2], C_F) for a 4 x 4 x 4 mm channel of 1 mm cells (walls included).
RATHORE2023_CF: dict[str, tuple[float, float, float]] = {
    "gyroid": (0.32, 7.892e-10, 1.173),
    "diamond": (0.32, 4.335e-10, 0.525),
}


# =============================================================================
# Literature table (one row per correlation considered; implemented or not)
# =============================================================================
_GAJ = "Gajetti, Boccardo, Savoldi & Marocco (2025), Int. J. Heat Mass Transf. 252:127439"
_SAV = "Savoldi, Cammi, Gajetti & Marocco (2026), Int. J. Heat Fluid Flow 121:110631"

LITERATURE: dict[str, CorrelationInfo] = {
    "gajetti2025": CorrelationInfo(
        key="gajetti2025",
        quantity="K, C_F",
        citation=_GAJ,
        doi="10.1016/j.ijheatmasstransfer.2025.127439",
        tpms="gyroid, diamond, Split-P (skeletal/solid network)",
        method="pore-scale CFD (OpenFOAM, laminar, periodic unit cell Lc = 10 mm; pipe validation Lc = 5 mm)",
        fluid="water, 20 C, isothermal",
        re_definition="Re_Dh = U_s D_h / (phi nu), D_h = 4/S_V",
        formula="K/Lc^2 = a phi^n; C_F = b phi^m (G: a 0.0156, n 2.78, b 0.0908, m -1.81; "
        "D: a 0.0106, n 2.89, b 0.0666, m -1.69)",
        validity=ValidityBox(re=(0.3, 100.0), porosity=(0.30, 0.60), pr=None, topologies=("gyroid", "diamond")),
        implemented=True,
        access="open-access full text (Polimi / PoliTo repositories); Eqs. 17-19, Tables 2-3",
        notes="Darcy linear up to Re_Dh ~ 10; Ergun fails (>30-50 % error); errors < 10 % (G, D)",
    ),
    "savoldi2026_hydraulics": CorrelationInfo(
        key="savoldi2026_hydraulics",
        quantity="D_h, K, C_F, f",
        citation=_SAV,
        doi="10.1016/j.ijheatfluidflow.2026.110631",
        tpms="gyroid (skeletal/solid network)",
        method="pore-scale CFD (OpenFOAM v2212, single periodic cell, laminar)",
        fluid="constant-property, Pr = 1",
        re_definition="Re_Dh = U_b D_h / nu, U_b = U_s/phi",
        formula="D_h/L = 4phi/(-5.723phi^2 + 5.729phi + 1.665); K/L^2 = 0.0157 phi^2.72; "
        "C_F = 0.0784 phi^-2.04; f = 2phi(D_h,rel^2/(K_rel Re) + phi D_h,rel C_F/sqrt(K_rel))",
        validity=ValidityBox(re=(20.0, 100.0), porosity=(0.30, 0.70), pr=None, topologies=("gyroid",)),
        implemented=True,
        access="open-access full text (Polimi repository); Eqs. 19-23, 34-36, 41",
        notes="f correlation MAPE 5.8 % (45 regression points), max 18.5 %",
    ),
    "savoldi2026_st1": CorrelationInfo(
        key="savoldi2026_st1",
        quantity="Nu (St)",
        citation=_SAV + ", Eq. 37",
        doi="10.1016/j.ijheatfluidflow.2026.110631",
        tpms="gyroid (skeletal), whole solid-fluid interface (interstitial h)",
        method="pore-scale CFD, uniform wall T and uniform wall q (non-conjugate), periodic cell",
        fluid="Pr = 1 only",
        re_definition="Re_Dh = U_b D_h / nu; f Darcy-type, pore velocity",
        formula="St = Nu/(Re Pr) = 0.0267 phi^0.19 f",
        validity=ValidityBox(re=(20.0, 100.0), porosity=(0.30, 0.70), pr=(1.0, 1.0), topologies=("gyroid",)),
        implemented=True,
        access="open-access full text; Eq. 37",
        notes="+-11 % (1 sd); residual Re trend at high phi. RECOMMENDED (x Pr^1/3) - see STATUS",
    ),
    "savoldi2026_st2": CorrelationInfo(
        key="savoldi2026_st2",
        quantity="Nu (St)",
        citation=_SAV + ", Eqs. 38-39",
        doi="10.1016/j.ijheatfluidflow.2026.110631",
        tpms="gyroid (skeletal), interstitial h",
        method="as Eq. 37",
        fluid="Pr = 1 only",
        re_definition="Re_Dh = U_b D_h / nu",
        formula="St = 0.135 phi^2.06 Re^(0.440 - phi) f",
        validity=ValidityBox(re=(20.0, 100.0), porosity=(0.30, 0.70), pr=(1.0, 1.0), topologies=("gyroid",)),
        implemented=True,
        access="open-access full text; Eqs. 38-39 (Fig. 15 caption prints 0.134)",
        notes="+-15 %, symmetric residuals; non-monotone in Re outside its box (low-h bound)",
    ),
    "savoldi2026_st3": CorrelationInfo(
        key="savoldi2026_st3",
        quantity="Nu (St)",
        citation=_SAV + ", Eq. 40",
        doi="10.1016/j.ijheatfluidflow.2026.110631",
        tpms="gyroid (skeletal), interstitial h",
        method="as Eq. 37",
        fluid="Pr = 1 only",
        re_definition="Re_Dh = U_b D_h / nu",
        formula="St = 0.034 phi^0.20 Re^(-0.061 - 0.0023 phi) f",
        validity=ValidityBox(re=(20.0, 100.0), porosity=(0.30, 0.70), pr=(1.0, 1.0), topologies=("gyroid",)),
        implemented=True,
        access="open-access full text; Eq. 40 (Fig. 16 caption prints 0.0342)",
        notes="+-12 %, biased at phi = 0.3 and 0.7",
    ),
    "rathore2023": CorrelationInfo(
        key="rathore2023",
        quantity="K, C_F (data)",
        citation="Rathore, Mehta, Kumar & Asfer (2023), Transp. Porous Media 146:669-701 (arXiv:2205.03591)",
        doi="10.1007/s11242-022-01880-7",
        tpms="diamond, gyroid, I-WP, primitive; 1 mm cells, 4 mm channel",
        method="pore-scale DNS (laminar) incl. channel walls",
        fluid="water",
        re_definition="Re = U L / nu, channel length L",
        formula="data only: C_F = 1.173 (G), 0.525 (D) at phi = 0.32",
        validity=ValidityBox(re=(0.01, 100.0), porosity=(0.32, 0.32), pr=None, topologies=("gyroid", "diamond")),
        implemented=False,
        access="arXiv preprint (open)",
        notes="Darcy fails above Re ~ 10-20; single porosity, wall effects included",
    ),
    "cheng2021": CorrelationInfo(
        key="cheng2021",
        quantity="Nu (interstitial), flow resistance",
        citation="Cheng, Li, Xu & Jiang (2021), Int. Commun. Heat Mass Transf. 129:105713",
        doi="10.1016/j.icheatmasstransfer.2021.105713",
        tpms="I-WP, primitive, diamond, gyroid porous media",
        method="pore-scale CFD + strength tests",
        fluid="air (per Savoldi et al. 2026)",
        re_definition="Re_h",
        formula="correlations Nu(Re_h, eps) and resistance - coefficients NOT accessed (paywalled)",
        validity=ValidityBox(re=(10.0, 129.0), porosity=(0.2, 0.8), pr=(0.7, 0.7), topologies=("gyroid", "diamond")),
        implemented=False,
        access="abstract only",
        notes="closest scope to ours (porosity 0.2-0.8); worth adding when the full text is available",
    ),
    "iyer2022": CorrelationInfo(
        key="iyer2022",
        quantity="Nu, f",
        citation="Iyer, Moore, Nguyen, Roy & Stolaroff (2022), Appl. Therm. Eng. 209:118192 (details to verify)",
        doi="10.1016/j.applthermaleng.2022.118192 (to verify)",
        tpms="sheet TPMS and periodic nodal surfaces (two-fluid heat exchangers)",
        method="CFD",
        fluid="not accessed",
        re_definition="not accessed",
        formula="laminar correlations valid up to Re ~ 350 (per Brambati et al. 2024) - NOT accessed",
        validity=ValidityBox(re=None, porosity=None, pr=None, topologies=("gyroid", "diamond")),
        implemented=False,
        access="secondary mentions only",
        notes="sheet (two-fluid) topology",
    ),
    "reynolds2023": CorrelationInfo(
        key="reynolds2023",
        quantity="Nu",
        citation="Reynolds, Fee, Morison & Holland (2023), Int. J. Heat Mass Transf. (as quoted by Brambati et al. 2024, Eq. 21)",
        doi="(to verify)",
        tpms="gyroid heat exchanger",
        method="experiment",
        fluid="air (Pr ~ 0.7; the Pr^0.4 factor is assumed, not fitted)",
        re_definition="Re_Dh",
        formula="Nu_Dh = 0.49 Re_Dh^0.62 Pr^0.4",
        validity=ValidityBox(re=(100.0, 2500.0), porosity=None, pr=(0.7, 0.72), topologies=("gyroid",)),
        implemented=True,
        access="secondary (Brambati et al. 2024, open access)",
        notes="Brambati et al. found it ~2x above their turbulent CFD",
    ),
    "brambati2024": CorrelationInfo(
        key="brambati2024",
        quantity="Nu",
        citation="Brambati, Guilizzoni & Foletti (2024), Appl. Therm. Eng. 242:122492",
        doi="10.1016/j.applthermaleng.2024.122492",
        tpms="gyroid, Schwarz-P, Schwarz-D (sheet, two-fluid HX), porosity 70-90 %",
        method="conjugate RANS CFD (k-omega SST), Re 5000-50000",
        fluid="air, water (calibration), acetone (validation); Pr 0.7-7",
        re_definition="Re_Dp = W_p D_p / nu (pore diameter, pore velocity); Nu on D_h",
        formula="Nu_Dh = 0.0964 Re_Dp^0.7136 Pr^0.4 (+-20 %)",
        validity=ValidityBox(re=(5000.0, 50000.0), porosity=(0.70, 0.90), pr=(0.7, 7.0), topologies=("gyroid", "diamond")),
        implemented=False,
        access="open-access full text (Polimi repository)",
        notes="turbulent - not applicable; its fitted Pr exponent (0.39-0.40) is evidence for Pr scaling",
    ),
    "piandoro2026": CorrelationInfo(
        key="piandoro2026",
        quantity="Nu, f",
        citation="Piandoro et al. (2026), Lasers Manuf. Mater. Process. 13:555-581",
        doi="10.1007/s40516-026-00347-7",
        tpms="gyroid (sheet/offset), porosity 4-83 %",
        method="CFD + L-PBF samples, fully developed duct",
        fluid="ethylene glycol (Pr ~ 154 from their Table 4); EG/water Pr = 14 (validation)",
        re_definition="Re_por = rho C_F sqrt(K) u / mu; Nu on minimum channel diameter",
        formula="f ~ 1 + 1/Re_por; Nu = (23.44 + 7.5 phi) sqrt(Re_por)",
        validity=ValidityBox(re=None, porosity=(0.04, 0.83), pr=(14.0, 154.0), topologies=("gyroid",)),
        implemented=False,
        access="open-access full text",
        notes="only high-Pr TPMS data found; wall-to-bulk h of a filled duct, no Pr exponent",
    ),
    "ergun1952": CorrelationInfo(
        key="ergun1952",
        quantity="f (packed bed)",
        citation="Ergun (1952), Chem. Eng. Prog. 48:89-94 (assessed by Gajetti et al. 2025)",
        doi="-",
        tpms="random packed beds",
        method="experiment",
        fluid="gases, liquids",
        re_definition="Re* = U_s D_p / (nu (1 - phi)), D_p = 1.5 (1-phi)/phi D_h",
        formula="dp* = 150/Re* + 1.75",
        validity=ValidityBox(re=None, porosity=None, pr=None, topologies=("any",)),
        implemented=False,
        access="via Gajetti et al. 2025",
        notes="REJECTED for TPMS: 30-50 % errors (Gajetti et al. 2025)",
    ),
    "gnielinski_packed_bed": CorrelationInfo(
        key="gnielinski_packed_bed",
        quantity="Nu (fluid-particle)",
        citation="Gnielinski (1981) Int. Chem. Eng. 21(3); VDI Heat Atlas 2nd ed. (2010)",
        doi="-",
        tpms="packed spheres (generic reference, not TPMS)",
        method="correlation of experiments",
        fluid="gases and liquids",
        re_definition="Re_p = U_s d_p / (phi nu), d_p = 6 (1-phi)/a_sf",
        formula="Nu_p = f_a (2 + sqrt(Nu_lam^2 + Nu_turb^2)), Nu_lam = 0.664 Re^0.5 Pr^(1/3), f_a = 1 + 1.5(1-phi)",
        validity=ValidityBox(re=(0.1, 1000.0), porosity=None, pr=(0.4, 1000.0), topologies=("any",)),
        implemented=True,
        access="implementation of the `ht` library (C. Bell) and its documented example",
        notes="Pr-validated cross-check of the Pr^1/3 extrapolation",
    ),
    "wakao_kaguei": CorrelationInfo(
        key="wakao_kaguei",
        quantity="Nu (fluid-particle)",
        citation="Wakao & Kaguei (1982), Heat and Mass Transfer in Packed Beds, Gordon & Breach",
        doi="-",
        tpms="packed spheres (generic reference, not TPMS)",
        method="correlation of experiments",
        fluid="gases and liquids",
        re_definition="Re_p = U_s d_p / nu (superficial), d_p = 6 (1-phi)/a_sf",
        formula="Nu_p = 2 + 1.1 Re_p^0.6 Pr^(1/3)",
        validity=ValidityBox(re=(3.0, 3000.0), porosity=None, pr=None, topologies=("any",)),
        implemented=True,
        access="as documented by the `ht` library (C. Bell)",
        notes="no porosity dependence",
    ),
    "kuwahara2001": CorrelationInfo(
        key="kuwahara2001",
        quantity="Nu (interstitial)",
        citation="Kuwahara, Shirota & Nakayama (2001), Int. J. Heat Mass Transf. 44:1153-1159",
        doi="10.1016/S0017-9310(00)00166-6 (to verify)",
        tpms="regular square-rod arrays (not TPMS)",
        method="pore-scale CFD at Pr = 1",
        fluid="Pr = 1",
        re_definition="Re_D = U_s D / nu (rod size)",
        formula="h D/k = (porosity term) + (Re_D term) Pr^(1/3) - coefficients not verified",
        validity=ValidityBox(re=None, porosity=(0.2, 0.9), pr=(1.0, 1.0), topologies=("any",)),
        implemented=False,
        access="secondary sources with garbled equation text",
        notes="classic LTNE closure; Pr^1/3 also assumed there",
    ),
}

#: Recommended closures for Task 9 (keys into LITERATURE / function options).
RECOMMENDED: dict[str, Any] = {
    "forchheimer": {"source": "gajetti2025", "blend": "log-linear in w"},
    "nusselt": {"correlation": "savoldi2026_st1", "variant": 1, "pr_exponent": 1.0 / 3.0},
    "nusselt_low_bound": {"correlation": "savoldi2026_st2", "variant": 2, "pr_exponent": 1.0 / 3.0},
    "nusselt_cross_check": "gnielinski_packed_bed",
}


def literature_table(implemented_only: bool = False) -> Any:
    """The Task 8 literature table as a ``pandas.DataFrame`` (one row per correlation)."""
    import pandas as pd

    rows = [info.as_row() for info in LITERATURE.values() if info.implemented or not implemented_only]
    return pd.DataFrame(rows)


# =============================================================================
# Geometry and dimensionless groups
# =============================================================================
def hydraulic_diameter(porosity: Any, a_sf: Any) -> Any:
    """D_h = 4 phi / a_sf [m] (a_sf = wetted area per TOTAL volume, 1/m)."""
    return 4.0 * np.asarray(porosity, float) / np.asarray(a_sf, float)


def equivalent_particle_diameter(porosity: Any, a_sf: Any) -> Any:
    """Sphere diameter with the same specific surface, d_p = 6 (1 - phi) / a_sf [m].

    Equals Gajetti et al.'s D_p = 1.5 (1 - phi)/phi D_h; used by the packed-bed references.
    """
    return 6.0 * (1.0 - np.asarray(porosity, float)) / np.asarray(a_sf, float)


def pore_velocity(superficial_velocity: Any, porosity: Any) -> Any:
    """U_b = U_s / phi."""
    return np.asarray(superficial_velocity, float) / np.asarray(porosity, float)


def reynolds_hydraulic(superficial_velocity: Any, a_sf: Any, kinematic_viscosity: float) -> Any:
    """Re_Dh = U_b D_h / nu = 4 |U_s| / (a_sf nu)  (porosity cancels)."""
    return 4.0 * np.abs(np.asarray(superficial_velocity, float)) / (np.asarray(a_sf, float) * kinematic_viscosity)


def reynolds_permeability(superficial_velocity: Any, permeability: Any, kinematic_viscosity: float) -> Any:
    """Re_K = |U_s| sqrt(K) / nu."""
    return np.abs(np.asarray(superficial_velocity, float)) * np.sqrt(np.asarray(permeability, float)) / kinematic_viscosity


def forchheimer_number(cf: Any, re_k: Any) -> Any:
    """Fo = C_F Re_K = (inertial drag) / (viscous drag) in (1)."""
    return np.asarray(cf, float) * np.asarray(re_k, float)


# =============================================================================
# Inertial drag
# =============================================================================
def _norm_tpms(tpms: str) -> str:
    t = tpms.lower().replace("_", "-")
    aliases = {"g": "gyroid", "d": "diamond", "schwarz-d": "diamond", "splitp1": "split-p1", "splitp2": "split-p2"}
    t = aliases.get(t, t)
    if t not in GAJETTI2025_POWER_LAWS:
        raise ValueError(f"unknown TPMS {tpms!r}; expected one of {sorted(GAJETTI2025_POWER_LAWS)}")
    return t


def gajetti2025_permeability(porosity: Any, tpms: str = "gyroid") -> Any:
    """K / Lc^2 = a phi^n (Gajetti et al. 2025, Eq. 18). Dimensionless; no range check.

    Not used in VoxLat (we compute K ourselves); kept to compare with the Task 4 solver.
    """
    c = GAJETTI2025_POWER_LAWS[_norm_tpms(tpms)]
    return c["a"] * np.asarray(porosity, float) ** c["n"]


def gajetti2025_forchheimer(porosity: Any, tpms: str = "gyroid") -> Any:
    """C_F = b phi^m (Gajetti et al. 2025, Eq. 19). No range check (see ``forchheimer_coefficient``)."""
    c = GAJETTI2025_POWER_LAWS[_norm_tpms(tpms)]
    return c["b"] * np.asarray(porosity, float) ** c["m"]


def _gajetti2025_table_cf(porosity: Any, tpms: str) -> Any:
    """Table 2 of Gajetti et al. (2025): log-log interpolation in phi, held constant outside 0.3-0.6."""
    t = GAJETTI2025_TABLE2[_norm_tpms(tpms)]
    phi = np.clip(np.asarray(porosity, float), t[0, 0], t[-1, 0])
    return np.exp(np.interp(np.log(phi), np.log(t[:, 0]), np.log(t[:, 2])))


def savoldi2026_hydraulic_diameter(porosity: Any) -> Any:
    """D_h / L of the skeletal gyroid (Savoldi et al. 2026, Eq. 34)."""
    c = SAVOLDI2026_HYDRAULICS
    phi = np.asarray(porosity, float)
    return 4.0 * phi / (c["c2"] * phi**2 + c["c1"] * phi + c["c0"])


def savoldi2026_permeability(porosity: Any) -> Any:
    """K / L^2 = 0.0157 phi^2.72 (Savoldi et al. 2026, Eq. 35), skeletal gyroid."""
    c = SAVOLDI2026_HYDRAULICS
    return c["a"] * np.asarray(porosity, float) ** c["n"]


def savoldi2026_forchheimer(porosity: Any) -> Any:
    """C_F = 0.0784 phi^-2.04 (Savoldi et al. 2026, Eq. 36), skeletal gyroid."""
    c = SAVOLDI2026_HYDRAULICS
    return c["b"] * np.asarray(porosity, float) ** c["m"]


def forchheimer_coefficient(
    porosity: Any,
    w: Any = 0.0,
    *,
    source: Literal["gajetti2025", "gajetti2025_table", "savoldi2026"] = "gajetti2025",
    re: Any = None,
    on_extrapolation: OnExtrapolation = "warn",
) -> Any:
    """Dimensionless Forchheimer (inertial drag) coefficient C_F of Eq. (1).

    Parameters
    ----------
    porosity:
        phi = 1 - rho* (scalar or array).
    w:
        Gyroid -> diamond blend parameter (0 = gyroid, 1 = diamond), broadcast with ``porosity``.
        For 0 < w < 1: ln C_F = (1 - w) ln C_F,G + w ln C_F,D  - an ASSUMPTION (no blend data exist);
        flagged by the topology check.
    source:
        ``"gajetti2025"`` (recommended; power laws of Gajetti et al. 2025, G and D),
        ``"gajetti2025_table"`` (their Table 2, log-log interpolated, constant outside phi = 0.3-0.6:
        the plateau variant, an upper bracket for diamond at phi > 0.6), or
        ``"savoldi2026"`` (gyroid only, fitted up to phi = 0.7).
    re:
        Optional Re_Dh values, only for the validity check.
    on_extrapolation:
        ``"warn"`` | ``"raise"`` | ``"ignore"`` for points outside the source's validity box.
    """
    phi = np.asarray(porosity, float)
    ww = np.broadcast_to(np.asarray(w, float), np.broadcast(phi, np.asarray(w, float)).shape)
    phi_b = np.broadcast_to(phi, ww.shape)
    if np.any((ww < -1e-12) | (ww > 1 + 1e-12)):
        raise ValueError("w must lie in [0, 1]")
    if np.any((phi_b <= 0) | (phi_b >= 1)):
        raise ValueError("porosity must lie in (0, 1)")
    if source == "savoldi2026":
        if np.any(ww > 1e-12):
            raise ValueError("source='savoldi2026' covers the gyroid only (w = 0)")
        check_range("savoldi2026_hydraulics", re=re, porosity=phi_b, w=ww, on_extrapolation=on_extrapolation)
        out = savoldi2026_forchheimer(phi_b)
    elif source in ("gajetti2025", "gajetti2025_table"):
        check_range("gajetti2025", re=re, porosity=phi_b, w=ww, on_extrapolation=on_extrapolation)
        fun = gajetti2025_forchheimer if source == "gajetti2025" else _gajetti2025_table_cf
        log_g = np.log(fun(phi_b, "gyroid"))
        log_d = np.log(fun(phi_b, "diamond"))
        out = np.exp((1.0 - ww) * log_g + ww * log_d)
    else:
        raise ValueError(f"unknown source {source!r}")
    return out[()] if out.ndim == 0 else out


def pressure_gradient(
    superficial_velocity: Any,
    permeability: Any,
    cf: Any,
    *,
    density: float,
    dynamic_viscosity: float,
) -> Any:
    """|grad p| of the Darcy-Forchheimer law (1): mu U/K + rho C_F U^2/sqrt(K) [Pa/m] (U = |U_s|)."""
    u = np.abs(np.asarray(superficial_velocity, float))
    k = np.asarray(permeability, float)
    return dynamic_viscosity * u / k + density * np.asarray(cf, float) * u**2 / np.sqrt(k)


def friction_factor_re(
    re: Any,
    porosity: Any,
    hydraulic_diam: Any,
    permeability: Any,
    cf: Any,
) -> Any:
    """f Re of Eq. (2) - finite at Re = 0 (the Darcy limit 2 phi D_h^2 / K).

    ``hydraulic_diam`` [m] and ``permeability`` [m^2] in any consistent units (only D_h^2/K and
    D_h/sqrt(K) enter), e.g. D_h/L and K/L^2.
    """
    phi = np.asarray(porosity, float)
    dh = np.asarray(hydraulic_diam, float)
    k = np.asarray(permeability, float)
    return 2.0 * phi * dh**2 / k + 2.0 * phi**2 * np.asarray(cf, float) * dh / np.sqrt(k) * np.asarray(re, float)


def friction_factor(re: Any, porosity: Any, hydraulic_diam: Any, permeability: Any, cf: Any) -> Any:
    """Darcy-type friction factor f = (dp/dx) D_h / (rho U_b^2 / 2) from Eq. (2) (Re > 0)."""
    re_a = np.asarray(re, float)
    if np.any(re_a <= 0):
        raise ValueError("friction_factor needs Re > 0; use friction_factor_re for the Re -> 0 limit")
    return friction_factor_re(re_a, porosity, hydraulic_diam, permeability, cf) / re_a


# =============================================================================
# Interstitial heat transfer
# =============================================================================
def _stanton_re_factor(re: np.ndarray, porosity: np.ndarray, variant: int) -> np.ndarray:
    c = SAVOLDI2026_STANTON[variant]
    if c["m0"] == 0.0 and c["m1"] == 0.0:
        return np.ones(np.broadcast(re, porosity).shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        return re ** (c["m0"] + c["m1"] * porosity)


def stanton_savoldi2026(re: Any, porosity: Any, friction: Any, variant: int = 1) -> Any:
    """St = Nu/(Re Pr) at Pr = 1 from the modified Reynolds analogy (Savoldi et al. 2026, Eqs. 37-40).

    ``friction`` is the Darcy-type f of Eq. (2). No range check (``nusselt_interstitial`` checks).
    """
    if variant not in SAVOLDI2026_STANTON:
        raise ValueError(f"variant must be 1, 2 or 3, got {variant!r}")
    c = SAVOLDI2026_STANTON[variant]
    re_a = np.asarray(re, float)
    phi = np.asarray(porosity, float)
    return c["A"] * phi ** c["n"] * _stanton_re_factor(re_a, phi, variant) * np.asarray(friction, float)


def nusselt_interstitial(
    re: Any,
    porosity: Any,
    friction_re: Any,
    prandtl: Any,
    *,
    variant: int = 1,
    pr_exponent: float = 1.0 / 3.0,
    w: Any = 0.0,
    on_extrapolation: OnExtrapolation = "warn",
) -> Any:
    """Interstitial Nusselt number Nu = h_sf D_h / k_f (Eq. 3; recommended: variant 1).

        Nu = St_variant(Re, phi; Pr = 1) Re Pr^pr_exponent,   St Re = A phi^n Re^m(phi) (f Re)

    Parameters
    ----------
    re:
        Re_Dh = U_b D_h / nu (``reynolds_hydraulic``); Re = 0 allowed for variant 1 (Darcy limit).
    porosity:
        phi.
    friction_re:
        f Re from ``friction_factor_re`` (with the bulk K, a_sf and C_F of this cell).
    prandtl:
        Pr of the coolant. The data are at Pr = 1; the Chilton-Colburn factor Pr^(1/3) (j-factor
        analogy; boundary layers thin at Pe = Re Pr >> 1) carries them to other Pr. pr_exponent =
        0.4 is the turbulent TPMS fit of Brambati et al. (2024) - use it as the sensitivity bound.
    variant:
        1 (recommended), 2 (low-h bound), 3.
    w:
        Blend parameter, used only for the topology check (the analogy is applied to any
        morphology through its own f: an assumption outside the gyroid).
    """
    key = f"savoldi2026_st{variant}"
    if key not in LITERATURE:
        raise ValueError(f"variant must be 1, 2 or 3, got {variant!r}")
    re_a = np.asarray(re, float)
    phi = np.asarray(porosity, float)
    if np.any(re_a < 0):
        raise ValueError("Re must be >= 0")
    if variant != 1 and np.any(re_a <= 0):
        raise ValueError(f"variant {variant} is singular at Re = 0 (Re^(m) with m != 0); use variant 1")
    check_range(key, re=re_a, porosity=phi, pr=prandtl, w=w, on_extrapolation=on_extrapolation)
    c = SAVOLDI2026_STANTON[variant]
    nu1 = c["A"] * phi ** c["n"] * _stanton_re_factor(re_a, phi, variant) * np.asarray(friction_re, float)
    out = nu1 * np.asarray(prandtl, float) ** pr_exponent
    return out[()] if np.ndim(out) == 0 else out


@dataclass(frozen=True)
class InterstitialHeatTransfer:
    """Result of ``interstitial_heat_transfer`` (all SI; arrays broadcast like the inputs)."""

    h_sf: Any  # W/(m^2 K), on the wetted area
    h_sf_a_sf: Any  # W/(m^3 K), volumetric exchange coefficient of the two-equation model
    nusselt: Any  # Nu = h_sf D_h / k_f
    reynolds: Any  # Re_Dh
    friction: Any  # Darcy-type f (NaN at Re = 0)
    friction_re: Any  # f Re
    cf: Any  # C_F used
    hydraulic_diameter: Any  # m
    prandtl: float
    settings: Mapping[str, Any] = field(default_factory=dict)


def interstitial_heat_transfer(
    superficial_velocity: Any,
    porosity: Any,
    a_sf: Any,
    permeability: Any,
    *,
    w: Any = 0.0,
    conductivity: float,
    kinematic_viscosity: float,
    prandtl: float,
    cf: Any = None,
    cf_source: Literal["gajetti2025", "gajetti2025_table", "savoldi2026"] = "gajetti2025",
    variant: int = 1,
    pr_exponent: float = 1.0 / 3.0,
    on_extrapolation: OnExtrapolation = "warn",
) -> InterstitialHeatTransfer:
    """h_sf from the recommended chain: Re_Dh -> C_F -> f Re (Eq. 2) -> Nu (Eq. 3) -> h_sf.

    Parameters are SI: ``superficial_velocity`` |U_s| [m/s], ``a_sf`` [1/m], ``permeability`` the
    BULK K along the flow [m^2] (``ClosureModel``; not wall-corrected), coolant ``conductivity``
    [W/(m K)], ``kinematic_viscosity`` [m^2/s], ``prandtl``. ``cf`` overrides the literature C_F.

    >>> from voxlat.utils import load_config
    >>> cfg = load_config()
    >>> r = interstitial_heat_transfer(0.083, 0.65, 2.970 / 4e-3, 4.887e-3 * (4e-3) ** 2,
    ...     conductivity=cfg.coolant.conductivity, kinematic_viscosity=cfg.coolant.kinematic_viscosity,
    ...     prandtl=cfg.coolant.prandtl, on_extrapolation="ignore")
    >>> round(float(r.reynolds)), round(float(r.nusselt), 1), round(float(r.h_sf))
    (194, 41.3, 4718)
    """
    phi = np.asarray(porosity, float)
    asf = np.asarray(a_sf, float)
    re = reynolds_hydraulic(superficial_velocity, asf, kinematic_viscosity)
    dh = hydraulic_diameter(phi, asf)
    if cf is None:
        cf_val = forchheimer_coefficient(phi, w, source=cf_source, re=re, on_extrapolation=on_extrapolation)
    else:
        cf_val = np.asarray(cf, float)
    fre = friction_factor_re(re, phi, dh, permeability, cf_val)
    nu = nusselt_interstitial(
        re, phi, fre, prandtl, variant=variant, pr_exponent=pr_exponent, w=w, on_extrapolation=on_extrapolation
    )
    h = np.asarray(nu, float) * conductivity / dh
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(re > 0, fre / np.where(re > 0, re, 1.0), np.nan)
    return InterstitialHeatTransfer(
        h_sf=h,
        h_sf_a_sf=h * asf,
        nusselt=nu,
        reynolds=re,
        friction=f,
        friction_re=fre,
        cf=cf_val,
        hydraulic_diameter=dh,
        prandtl=float(prandtl),
        settings={"variant": variant, "pr_exponent": pr_exponent, "cf_source": cf_source if cf is None else "user"},
    )


# =============================================================================
# Generic references (not TPMS) and secondary TPMS correlations, for comparison only
# =============================================================================
def nusselt_gnielinski_packed_bed(re_p: Any, prandtl: Any, porosity: Any, *, on_extrapolation: OnExtrapolation = "warn") -> Any:
    """Gnielinski (VDI Heat Atlas) fluid-to-particle Nu_p = h d_p / k for packed spheres.

        Nu_p = f_a [2 + sqrt(Nu_lam^2 + Nu_turb^2)],   f_a = 1 + 1.5 (1 - phi)
        Nu_lam = 0.664 Re^0.5 Pr^(1/3),  Nu_turb = 0.037 Re^0.8 Pr / (1 + 2.443 Re^-0.1 (Pr^(2/3) - 1))

    with Re_p = U_s d_p / (phi nu) (interstitial velocity, particle diameter). Stated validity
    (spheres): 0.1 < Re < 1000, 0.4 < Pr < 1000. Used here only as a Pr-validated cross-check,
    with d_p = 6 (1 - phi)/a_sf (``equivalent_particle_diameter``); convert with Nu_Dh = Nu_p D_h/d_p.
    """
    check_range("gnielinski_packed_bed", re=re_p, pr=prandtl, on_extrapolation=on_extrapolation)
    re_a = np.asarray(re_p, float)
    pr = np.asarray(prandtl, float)
    lam = 0.664 * np.sqrt(re_a) * pr ** (1.0 / 3.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        turb = np.where(
            re_a > 0,
            0.037 * re_a**0.8 * pr / (1.0 + 2.443 * np.where(re_a > 0, re_a, 1.0) ** -0.1 * (pr ** (2.0 / 3.0) - 1.0)),
            0.0,
        )
    out = (1.0 + 1.5 * (1.0 - np.asarray(porosity, float))) * (2.0 + np.sqrt(lam**2 + turb**2))
    return out[()] if np.ndim(out) == 0 else out


def nusselt_wakao_kaguei(re_p: Any, prandtl: Any, *, on_extrapolation: OnExtrapolation = "warn") -> Any:
    """Wakao & Kaguei (1982) packed-bed Nu_p = 2 + 1.1 Re_p^0.6 Pr^(1/3) (superficial velocity, d_p)."""
    check_range("wakao_kaguei", re=re_p, pr=prandtl, on_extrapolation=on_extrapolation)
    out = 2.0 + 1.1 * np.asarray(re_p, float) ** 0.6 * np.asarray(prandtl, float) ** (1.0 / 3.0)
    return out[()] if np.ndim(out) == 0 else out


def nusselt_reynolds2023(re: Any, prandtl: Any, *, on_extrapolation: OnExtrapolation = "warn") -> Any:
    """Nu_Dh = 0.49 Re_Dh^0.62 Pr^0.4: gyroid HX experiments in air (Reynolds et al. 2023), as quoted by
    Brambati et al. (2024, Eq. 21). Secondary citation; for comparison plots only."""
    check_range("reynolds2023", re=re, pr=prandtl, w=0.0, on_extrapolation=on_extrapolation)
    out = 0.49 * np.asarray(re, float) ** 0.62 * np.asarray(prandtl, float) ** 0.4
    return out[()] if np.ndim(out) == 0 else out
