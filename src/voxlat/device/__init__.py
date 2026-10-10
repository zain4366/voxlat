"""Homogenized cooling-jacket device model (Task 9) and baselines B1-B3 (Task 10).

>>> from voxlat.device import JacketModel, evaluate
>>> out = evaluate(rho=0.35, w=0.0, L=4e-3)          # doctest: +SKIP   uniform gyroid, 3 L/min
>>> out["thermal_resistance"], out["pump_power"]       # doctest: +SKIP
"""

from voxlat.device.jacket2d import (
    JacketFields,
    JacketGrid,
    JacketModel,
    JacketOptions,
    LocalClosures,
    SurrogateClosures,
    UniformClosures,
    build_grid,
    cell_average_heat_flux,
    default_model,
    directional_permeability,
    evaluate,
    fin_conductance,
    in_plane_permeability,
    manifold_htc_default,
    ntu_reference,
    uniform_flow_reference,
)
from voxlat.device.baselines import (
    ChannelDesign,
    ChannelJacketModel,
    ChannelOptions,
    GradedDensityDesign,
    UniformLatticeDesign,
    design_from_dict,
    pareto_front,
    select_best,
    tune_b1,
    tune_b2,
    tune_b3,
)
from voxlat.device.baselines import evaluate as evaluate_baseline

__all__ = [
    "JacketFields",
    "JacketGrid",
    "JacketModel",
    "JacketOptions",
    "LocalClosures",
    "SurrogateClosures",
    "UniformClosures",
    "build_grid",
    "cell_average_heat_flux",
    "default_model",
    "directional_permeability",
    "evaluate",
    "fin_conductance",
    "in_plane_permeability",
    "manifold_htc_default",
    "ntu_reference",
    "uniform_flow_reference",
    # Task 10 baselines
    "ChannelDesign",
    "ChannelJacketModel",
    "ChannelOptions",
    "GradedDensityDesign",
    "UniformLatticeDesign",
    "design_from_dict",
    "evaluate_baseline",
    "pareto_front",
    "select_best",
    "tune_b1",
    "tune_b2",
    "tune_b3",
]
