"""Geometry export of optimized designs through PicoGK (.NET) and a pure-numpy cross-check generator (Task 12).

Modules: ``config`` (export-config JSON), ``field`` (numpy mirror of the C# implicit), ``sector`` (numpy sector
generator, volume-fraction cross-check), ``checks`` (min wall / pore / connectivity / pockets / loose solid),
``maskio`` (voxel masks shared with C#), ``stl`` (STL IO + re-voxelizer), ``picogk`` (pythonnet driver).
C# sources: ``voxlat/export/dotnet``. CLI: ``scripts/task12_export.py``.
"""
