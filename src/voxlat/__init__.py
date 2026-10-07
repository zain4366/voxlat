"""VoxLat — voxel-based design of graded TPMS lattice cooling jackets.

Pure-Python, CPU-only pipeline: TPMS geometry -> periodic homogenization
(conduction, elasticity, Stokes) -> finite-gap corrections -> surrogates ->
homogenized device model -> multi-objective optimization -> PicoGK export.
See ``claude/voxlat_task_plan.md`` (project doc) and ``STATUS.md``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("voxlat")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
