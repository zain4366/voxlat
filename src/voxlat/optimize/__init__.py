"""Grading-field parameterization and multi-objective optimization of the jacket (Task 11).

* ``parameterization`` - tensor-product B-spline fields rho*(s, z), w(s, z), L(s, z) on the
  unwrapped jacket, mapped to the design bounds; exact grading-gradient / cell-size feasibility
  repair shared by every optimizer.
* ``problem`` - ``JacketProblem``: (R, pump power, mass) at 3 L/min from the Task 9 device
  model (slot or end-ring manifolds), constraints, archive, fixed objective normalization.
* ``hypervolume`` - exact 3-D hypervolume, box decomposition, HVI and analytic EHVI.
* ``gp`` - numpy GP (Matern-5/2 ARD, dimension-scaled priors) for the BoTorch-free fallback.
* ``algorithms`` - NSGA-II (pymoo), qNEHVI (BoTorch qLogNEHVI, numpy EHVI fallback),
  scrambled-Sobol random search.
* ``study`` - run/store/analyse the studies (results/task11/, results/designs/).
"""

from voxlat.optimize.algorithms import BOResult, anchor_designs, have_botorch, have_pymoo, run_nsga2, run_qnehvi, run_sobol
from voxlat.optimize.gp import GaussianProcess
from voxlat.optimize.hypervolume import (
    box_decomposition,
    expected_hvi,
    hv_improvement,
    hypervolume,
    hypervolume_history,
    pareto_mask,
)
from voxlat.optimize.parameterization import (
    FastMinCellSize,
    GradingParameterization,
    GradingSpec,
    clamped_bspline_basis,
    grading_gradient_max,
    periodic_cubic_basis,
)
from voxlat.optimize.problem import OBJECTIVES, JacketProblem, ObjectiveSpace, ProblemSettings, evaluate_design_fields

__all__ = [
    "BOResult", "anchor_designs", "have_botorch", "have_pymoo", "run_nsga2", "run_qnehvi", "run_sobol",
    "GaussianProcess",
    "box_decomposition", "expected_hvi", "hv_improvement", "hypervolume", "hypervolume_history", "pareto_mask",
    "FastMinCellSize", "GradingParameterization", "GradingSpec", "clamped_bspline_basis", "grading_gradient_max",
    "periodic_cubic_basis",
    "OBJECTIVES", "JacketProblem", "ObjectiveSpace", "ProblemSettings", "evaluate_design_fields",
]
