"""TPMS unit-cell geometry: level sets (gyroid, Schwarz diamond, blends), periodic
voxelization, density thresholds, geometric metrics and manufacturability (Task 1).

Main module: ``voxlat.geometry.tpms``. Generic periodic voxel tools (surface
area, local thickness, throat size, periodic connectivity) live in
``voxlat.geometry.voxel_tools`` and are reused by the homogenization tasks.

>>> from voxlat.geometry import TPMSParams, voxelize, compute_metrics
>>> cell = voxelize(TPMSParams(w=0.0, rho=0.3), n=32)   # bool[32, 32, 32], True = solid
>>> round(float(cell.mean()), 3)
0.3
"""

from voxlat.geometry.tpms import (
    FeasibleRegion,
    ManufacturabilityReport,
    TPMSMetrics,
    TPMSParams,
    check_manufacturable,
    compute_metrics,
    effective_stretch,
    feasible_region,
    grid_shape,
    level_set,
    min_cell_size,
    sample_level_set,
    threshold_for_density,
    threshold_from_table,
    voxelize,
)
from voxlat.geometry.voxel_tools import (
    local_thickness,
    periodic_components,
    periodic_edt,
    periodic_surface_area,
    throat_diameter,
)

__all__ = [
    "TPMSParams",
    "TPMSMetrics",
    "ManufacturabilityReport",
    "FeasibleRegion",
    "level_set",
    "sample_level_set",
    "grid_shape",
    "effective_stretch",
    "threshold_for_density",
    "threshold_from_table",
    "voxelize",
    "compute_metrics",
    "check_manufacturable",
    "min_cell_size",
    "feasible_region",
    "periodic_surface_area",
    "periodic_edt",
    "local_thickness",
    "throat_diameter",
    "periodic_components",
]
