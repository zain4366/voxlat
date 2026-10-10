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
]
