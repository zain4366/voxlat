# VoxLat

**Vox**el-based design of graded TPMS **lat**tice cooling jackets.

VoxLat is a pure-Python, CPU-only computational pipeline. Its demonstrator is a
liquid-cooling jacket around the stator of a 15 kW inrunner electric motor, filled
with a spatially graded TPMS lattice (gyroid / Schwarz diamond / blends) that both
cools the motor and carries its loads.

Pipeline: TPMS geometry → periodic voxel homogenization (conduction, elasticity,
Stokes permeability) → finite-gap corrections → surrogate models → homogenized
device model → multi-objective optimization → PicoGK geometry export.

Project status and hand-off notes for every task are in **[STATUS.md](STATUS.md)**.
The reference problem (every number, SI units) lives in
**[configs/reference.yaml](configs/reference.yaml)**.

## Repository layout

```
voxlat/
├── src/voxlat/            # the package (src layout)
│   ├── geometry/          # TPMS level sets, voxelization, metrics        (Task 1)
│   ├── homogenization/    # k_eff, C_eff, K, finite-gap strips           (Tasks 2-4, 7)
│   ├── closures/          # dataset generation + literature correlations (Tasks 5, 8)
│   ├── surrogates/        # GP / ensemble / 3D-CNN closure models        (Task 6)
│   ├── device/            # homogenized jacket model + baselines         (Tasks 9, 10)
│   ├── optimize/          # grading fields + NSGA-II / qNEHVI            (Task 11)
│   ├── export/            # PicoGK export                                 (Task 12)
│   └── utils/             # config, seeding, logging, run records, figures
├── configs/reference.yaml # single source of truth for the reference problem
├── scripts/               # one script per figure / table / long run
├── tests/                 # pytest verification tests
├── data/                  # computed tables (*.parquet)
├── models/                # trained surrogate models
├── results/figures/       # 300-dpi PNG figures
├── legacy/                # original Noyron 2.0 prototype, unchanged (see legacy/README.md)
└── STATUS.md              # hand-off log, updated by every task
```

## Install (Windows, PowerShell)

Requires **Python 3.11 or 3.12** (64-bit) from python.org. Python 3.12 is the safest
choice for the optional PyTorch/BoTorch extras. Check with `py -0` which versions you have.

```powershell
cd D:\voxlat                      # wherever you cloned/extracted the repo
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1      # prompt now starts with (.venv)
python -m pip install --upgrade pip
pip install -e .                  # core only (enough for Tasks 0-5, 7-10)
pip install -e ".[ml,opt]"        # + torch, pymoo, botorch (needed from Task 6 / 11)
```

If PowerShell refuses to run `Activate.ps1` ("running scripts is disabled"), run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use `cmd.exe` and
`.venv\Scripts\activate.bat` instead.

PyTorch from PyPI on Windows is the CPU build, which is all VoxLat needs.

Keep this environment **separate** from the old `noyron_env` (pythonnet / PicoGK).
Task 12 explains how the two meet.

### Linux / macOS

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,opt]"
```

## Run the tests

```powershell
pytest                    # all tests (from the repo root)
pytest -q tests/test_config.py
pytest -m "not slow"      # skip long verification tests (added in later tasks)
```

`legacy/` is never collected by pytest.

## Using the config

```python
from voxlat.utils import load_config, set_seed, get_logger, run_record, save_figure

cfg = load_config()                       # frozen, typed dataclass (SI units)
cfg.jacket.lattice_gap                    # 0.006 m
cfg.mean_heat_flux                        # ~35 014 W/m^2
cfg.q_wall(z)                             # heat-flux profile q''(z), z in metres
cfg2 = load_config(overrides={"operating": {"nominal_flow_rate": 1e-4}})
```

Never hard-code a reference number in a module: read it from the config.
Set `VOXLAT_CONFIG=path\to\other.yaml` to use another file.

## Conventions

- SI units inside the code; unit cells on normalized coordinates [0,1)³ scaled by cell size L.
- Voxel arrays are `bool[nx, ny, nz]`, `True` = solid, index order (x, y, z).
- Every solver documents its governing equations and boundary conditions in its docstring.
- Every physics module ships pytest verification tests against analytic or published cases.
- Data rows carry `run_record()` metadata; figures go through `save_figure()` (300 dpi PNG),
  each produced by a script in `scripts/`.
- Seeds are deterministic (`set_seed`).

## Reproduce the paper

Added in Task 13.
