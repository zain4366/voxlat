"""Closure laws for the device model: dataset generation from the homogenization
solvers (Task 5, ``voxlat.closures.dataset``) and literature correlations for Nu and
Forchheimer (Task 8).

>>> from voxlat.closures import build_design
>>> design = build_design(256)          # 256 Sobol + 8 corners + 12 edges + 14 pure-line points
>>> len(design), design[0].id
(290, 'corner-0')

Literature closures (Task 8, ``voxlat.closures.empirical``):

>>> from voxlat.closures import forchheimer_coefficient
>>> round(float(forchheimer_coefficient(0.5, 0.0)), 4)     # gyroid, Gajetti et al. (2025)
0.3184
"""

from voxlat.closures.dataset import (
    DatasetSettings,
    DesignSpace,
    Sample,
    SettingsMismatchError,
    boundary_design,
    build_design,
    compute_sample,
    consolidate,
    estimate_runtime,
    plot_quicklook,
    read_table,
    run_samples,
    sobol_design,
)
from voxlat.closures.empirical import (
    LITERATURE,
    RECOMMENDED,
    CorrelationRangeError,
    CorrelationRangeWarning,
    InterstitialHeatTransfer,
    check_range,
    forchheimer_coefficient,
    friction_factor,
    friction_factor_re,
    hydraulic_diameter,
    interstitial_heat_transfer,
    literature_table,
    nusselt_interstitial,
    pressure_gradient,
    range_report,
    reynolds_hydraulic,
    reynolds_permeability,
)

__all__ = [
    # Task 8: literature closures
    "LITERATURE",
    "RECOMMENDED",
    "CorrelationRangeError",
    "CorrelationRangeWarning",
    "InterstitialHeatTransfer",
    "check_range",
    "forchheimer_coefficient",
    "friction_factor",
    "friction_factor_re",
    "hydraulic_diameter",
    "interstitial_heat_transfer",
    "literature_table",
    "nusselt_interstitial",
    "pressure_gradient",
    "range_report",
    "reynolds_hydraulic",
    "reynolds_permeability",
    # Task 5: dataset
    "DatasetSettings",
    "DesignSpace",
    "Sample",
    "SettingsMismatchError",
    "boundary_design",
    "build_design",
    "compute_sample",
    "consolidate",
    "estimate_runtime",
    "plot_quicklook",
    "read_table",
    "run_samples",
    "sobol_design",
]
