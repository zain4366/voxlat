"""Periodic homogenization solvers on voxel cells: conduction (k_eff, Task 2),
elasticity (C_eff + stress localization, Task 3), Stokes permeability (K, Task 4)
and finite-gap strips (Task 7).

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
    "ConvergenceFit",
    "fit_convergence",
    "observed_order",
    "richardson",
]
