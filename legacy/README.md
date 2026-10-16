# legacy/ — the original Noyron 2.0 drone-boom prototype

These files are kept **unchanged** for history. They are not part of the `voxlat`
package, are not imported anywhere, and are excluded from pytest collection
(`norecursedirs` in `pyproject.toml`). They run only in the old `noyron_env`
environment (pythonnet + PicoGK on `D:\nyron 2`).

| File | What it was | Reused in VoxLat? |
|---|---|---|
| `noyron_core.py` | "Data factory": starts .NET via pythonnet, pre-loads `picogk.1.7.dll` with `NativeLibrary.Load`, runs `PicoGK.Library.Go(...)` with a Python `ThreadStart` task that builds 100 random hollow booms and writes STLs + `boom_dataset.csv`. | **Yes — Task 12.** The working PicoGK bootstrap (load order, `Library.Go` signature, `Single`/`Boolean`/`String` casts) is the template for the export driver. |
| `diagnostic.py` | Loads the native `picogk.1.7.dll` with `ctypes` to tell missing-native-dependency errors apart from .NET errors. | **Yes — Task 12 troubleshooting.** |
| `test_bridge.py` | Minimal pythonnet check that `PicoGK.dll` can be referenced. Note: points at `D:\nyron 2\PicoGK\bin\...`, while the working scripts use `PicoGK_Engine\bin\...`. | Task 12 troubleshooting only. Named `test_*.py`, so it must stay out of pytest collection. |
| `noyron_labeler.m` | MATLAB: labels each boom with a max deflection from an SDOF model (k = 3EI/L^3 with an ad-hoc 2 %/hole penalty, `lsim` forced response). | No. Replaced by voxel FEA homogenization (Task 3). |
| `noyron_dynamics.m` | MATLAB: SDOF harmonic-response demo on one STL. | No. |
| `noyron_ai.py`, `noyron_ai.ipynb` | MLP surrogate (2 inputs -> deflection) + grid search + PicoGK rebuild of the "optimum". | Concept only (surrogate -> optimizer -> geometry), redone properly in Tasks 6, 11, 12. |
| `nyron 2.sln` | Visual Studio solution for the PicoGK C# project. | No (Task 12 creates its own .NET project). |
| `tempCodeRunnerFile.py` | VS Code "Code Runner" scratch fragment — not real code. | No. Safe to delete. |

## Known issues (why these are not reused as-is)

- **Surrogate output is degenerate.** The notebook predicts 0.00–0.01 mm deflection
  for every design, so the 4.8 mm constraint is never active and the "optimum"
  is just the corner of the search box (thinnest wall, most holes).
- **Mass objective is a heuristic** (`thickness - 0.15*holes`), not a computed mass.
- **No held-out validation** in `noyron_ai.py` (trains on all data); the notebook
  splits but never evaluates the test set.
- `noyron_ai.py` loads the .NET runtime twice and repeats the import block.
- The notebook's second cell fails with `name 'PicoGK' is not defined` because the
  PicoGK bootstrap was never run in that kernel.
- The data from these scripts (`D:\nyron 2\noyron_workspace\*.stl, *.csv`) is **not**
  copied into this repository.
