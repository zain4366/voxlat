"""Closure laws for the device model: dataset generation from the homogenization
solvers (Task 5, ``voxlat.closures.dataset``) and literature correlations for Nu and
Forchheimer (Task 8).

>>> from voxlat.closures import build_design
>>> design = build_design(256)          # 256 Sobol + 8 corners + 12 edges + 14 pure-line points
>>> len(design), design[0].id
(290, 'corner-0')
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

__all__ = [
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
