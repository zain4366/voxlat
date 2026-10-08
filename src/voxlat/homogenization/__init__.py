"""Periodic homogenization solvers on voxel cells: conduction (k_eff, Task 2),
elasticity (C_eff + stress localization, Task 3), Stokes permeability (K, Task 4,
staggered MAC + MINRES) and finite-gap strips (Task 7).

>>> import numpy as np
>>> from voxlat.homogenization import effective_conductivity
>>> cell = np.zeros((8, 4, 4), bool); cell[:4] = True        # x-laminate, 50 % solid
>>> r = effective_conductivity(cell, k_s=10.0, k_f=1.0)
>>> round(float(r.k_eff[0, 0]), 6), round(float(r.k_eff[1, 1]), 6)   # harmonic, arithmetic
(1.818182, 5.5)
"""

from voxlat.homogenization.conduction import (
    ConductivityResult,
    ExtrapolatedConductivity,
    effective_conductivity,
    effective_conductivity_tpms,
    extrapolated_conductivity_tpms,
    hashin_shtrikman_bounds,
    wiener_bounds,
)
from voxlat.homogenization.elasticity import (
    ElasticityResult,
    ExtrapolatedElasticity,
    StressLocalization,
    effective_elasticity,
    effective_elasticity_tpms,
    extrapolated_elasticity_tpms,
    hashin_shtrikman_porous,
    isotropic_stiffness,
    laminate_stiffness,
)
from voxlat.homogenization.stokes import (
    ExtrapolatedPermeability,
    PermeabilityResult,
    assemble_stokes_system,
    extrapolated_permeability_tpms,
    kozeny_constant,
    permeability,
    permeability_tpms,
    solve_stokes,
)
from voxlat.homogenization.convergence import (
    ConvergenceFit,
    fit_convergence,
    observed_order,
    richardson,
)

__all__ = [
    "ConductivityResult",
    "ExtrapolatedConductivity",
    "effective_conductivity",
    "effective_conductivity_tpms",
    "extrapolated_conductivity_tpms",
    "hashin_shtrikman_bounds",
    "wiener_bounds",
    "ElasticityResult",
    "ExtrapolatedElasticity",
    "StressLocalization",
    "effective_elasticity",
    "effective_elasticity_tpms",
    "extrapolated_elasticity_tpms",
    "hashin_shtrikman_porous",
    "isotropic_stiffness",
    "laminate_stiffness",
    "ExtrapolatedPermeability",
    "PermeabilityResult",
    "assemble_stokes_system",
    "extrapolated_permeability_tpms",
    "kozeny_constant",
    "permeability",
    "permeability_tpms",
    "solve_stokes",
    "ConvergenceFit",
    "fit_convergence",
    "observed_order",
    "richardson",
]
