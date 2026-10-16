# Front matter of the VoxLat manuscript (Task 14).
HEADER = "Manuscript prepared for submission to the International Journal of Heat and Mass Transfer"
HEADER2 = "Original research article"

TITLE = ("Finite-gap corrections for homogenized TPMS lattices and the multi-objective design "
         "of a graded cooling jacket for an electric motor")

AUTHORS = [
    ("Zain ul Abiden", "a", True),
    ("Ahmed Mukhtar Bukhari", "a", False),
]
AFFILIATIONS = [
    ("a", "Department of Aerospace Engineering, School of Mechanical and Manufacturing Engineering (SMME), "
          "National University of Sciences and Technology (NUST), H-12, Islamabad 44000, Pakistan"),
]
CORRESPONDING = "Corresponding author. E-mail address: zabiden.ug23smme@student.nust.edu.pk (Z. ul Abiden)."

KEYWORDS = [
    "triply periodic minimal surface",
    "finite-size effect",
    "homogenization",
    "permeability",
    "electric motor cooling",
    "graded lattice",
    "multi-objective Bayesian optimization",
    "additive manufacturing",
]

ABSTRACT = (
    "Additively manufactured triply periodic minimal surface (TPMS) lattices are attractive for the liquid "
    "cooling jackets of electric machines because one printed structure can transfer heat, guide the coolant "
    "and carry load. Design studies of graded lattices rely on homogenized closures computed on periodic unit "
    "cells, yet a jacket gap of a few millimetres holds only one to three cells. We quantify the resulting error "
    "and use corrected closures to design a graded jacket for a 15 kW permanent-magnet motor (770 W into the "
    "jacket, 6 mm gap). Periodic voxel solvers for conductivity, stiffness and Stokes permeability, verified "
    "against exact and published solutions, are applied to lattice strips with N = 1–8 cells between bonded "
    "walls. Bulk closures over-predict the in-plane permeability by 23–40% at N = 1 and by 7–10% at N = 3; the "
    "deficit scales as 1/N with a wall coefficient of 0.19–0.28 for gyroid and diamond, whereas the through-gap "
    "conductivity changes by at most 2.3% and the walls stiffen the core. A physically based 1/N correction "
    "reduces the cross-validated permeability error from 12.9 to 0.4 percentage points. The corrected closures, "
    "literature correlations for inertia and interstitial heat transfer, and a three-field non-equilibrium model "
    "form a verified device model that runs in under one second. With axial-slot manifolds, optimizing the "
    "density, morphology and cell-size fields lowers the thermal resistance at equal pump power by 14.6% relative "
    "to circumferential channels, 10.1% relative to the best uniform gyroid and 3.3% relative to density-only "
    "grading; changing the manifolds to end rings matters more than any lattice grading. Batch Bayesian "
    "optimization (qNEHVI) needs about seven times fewer model evaluations than NSGA-II for the same hypervolume. "
    "The manufacturable optimum is exported as a printable geometry that passes minimum-wall, pore-throat and "
    "connectivity checks. All code and data are open source."
)

HIGHLIGHTS = [
    "Bulk TPMS closures over-predict in-plane permeability by up to 40% in one-cell gaps",
    "The permeability deficit follows 1/N with a wall coefficient of 0.19–0.28",
    "A verified sub-second jacket model couples Darcy–Forchheimer flow and LTNE heat",
    "Graded lattices cut thermal resistance by 10% vs uniform gyroid at equal pump power",
    "Batch Bayesian optimization needs about 7× fewer evaluations than NSGA-II",
]

GRAPHICAL_ABSTRACT_NOTE = ("Suggested graphical abstract: Fig. 1 (workflow) beside Fig. 14a (exported jacket render); "
                           "Elsevier asks for a separate image file of at least 531 × 1328 pixels (h × w).")

# Nomenclature: (symbol, description). Inline LaTeX allowed in $...$.
NOMENCLATURE = [
    ("Latin symbols", None),
    ("$a$", "wall coefficient of the finite-gap correction, Eqs. (@eq:gapser)–(@eq:gappar)"),
    ("$a_{sf}$", "specific solid–fluid surface area per total volume [m$^{-1}$]"),
    ("$a_z$", "axial stretch of the unit cell"),
    ("$C_{eff}$", "effective stiffness tensor (Voigt, engineering shear) [Pa]"),
    ("$C_F$", "Forchheimer (inertial) coefficient"),
    ("$C_{nn}$, $G_t$, $G_{sz}$", "normal stiffness across the gap, transverse and in-plane shear moduli of the core [Pa]"),
    ("$D_h$", "hydraulic diameter, $4\\phi/a_{sf}$ [m]"),
    ("$f$", "Darcy friction factor"),
    ("$f_K$", "finite-gap factor applied to the Darcy term, $1-a_K/N$"),
    ("$h$", "lattice gap height (6 mm) [m]"),
    ("$h_{sf}$", "interstitial heat-transfer coefficient [W m$^{-2}$ K$^{-1}$]"),
    ("$h_m$", "sleeve-to-plenum heat-transfer coefficient under a manifold slot [W m$^{-2}$ K$^{-1}$]"),
    ("$K$, $K_t$", "permeability tensor; in-plane permeability along the gap [m$^2$]"),
    ("$k_{eff}$, $k_n$", "effective conductivity tensor; through-gap conductivity [W m$^{-1}$ K$^{-1}$]"),
    ("$L$", "in-plane cell size [m]"),
    ("$L_{min}$", "smallest manufacturable cell size for given $(w,\\rho^*)$ [m]"),
    ("$N$", "number of cells across the gap, $h/L$"),
    ("$n$", "voxels per cell edge"),
    ("$P$", "pump power, $\\Delta p\\,\\dot V/\\eta_{pump}$ [W]"),
    ("$Pr$, $Re_{D_h}$", "Prandtl number; Reynolds number based on $D_h$ and pore velocity"),
    ("$Q$", "heat load into the jacket (770 W) [W]"),
    ("$R$", "thermal resistance, $(T_{wall,max}-T_{in})/Q$ [K W$^{-1}$]"),
    ("$R_{active}$", "thermal resistance of the cooled region, excluding the sleeve under the manifold slots"),
    ("$s$, $z$", "arc length at the mean lattice radius; axial coordinate [m]"),
    ("$U$", "superficial (Darcy) velocity [m s$^{-1}$]"),
    ("$U_{eff}$", "sleeve-to-coolant conductance per unit sleeve area [W m$^{-2}$ K$^{-1}$]"),
    ("$\\dot V$", "coolant volume flow rate (3 L/min nominal) [m$^3$ s$^{-1}$]"),
    ("$w$", "gyroid–diamond blend weight (0 = gyroid, 1 = diamond)"),
    ("Greek symbols", None),
    ("$\\delta_q$", "finite-gap discrepancy of quantity $q$, $q_{strip}/q_{hom}-1$"),
    ("$\\eta$", "fin efficiency of the lattice across the gap"),
    ("$\\theta$", "morphology parameters $(w,\\rho^*,a_z)$; or circumferential angle"),
    ("$\\rho^*$", "relative density (solid volume fraction)"),
    ("$\\phi$", "porosity, $1-\\rho^*$; also TPMS level-set value $\\phi_w$"),
    ("Abbreviations", None),
    ("B1, B1s, B2, B3", "baselines: helical channels, circumferential channels, best uniform gyroid, density-only grading"),
    ("G, D", "gyroid, Schwarz diamond"),
    ("HV", "hypervolume indicator"),
    ("LTNE", "local thermal non-equilibrium"),
    ("MAC", "marker-and-cell (staggered) finite volumes"),
    ("qNEHVI", "batch noisy expected hypervolume improvement"),
    ("TPMS", "triply periodic minimal surface"),
]
