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
├── data/                  # computed tables (*.parquet; Task 7: finite_gap.csv)
├── models/                # trained surrogate models
├── results/figures/       # 300-dpi PNG figures
├── docs/                  # per-task explainers (docs/task1_explained.md, ...)
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
pip install -e ".[amg]"           # optional: pyamg preconditioner (faster homogenization solves)
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

## TPMS geometry (Task 1)

```python
from voxlat.geometry import TPMSParams, voxelize, compute_metrics, check_manufacturable, min_cell_size

p = TPMSParams(w=0.0, rho=0.3, a=(1, 1, 1.2))    # w: 0 gyroid ... 1 diamond; rho: solid fraction
cell = voxelize(p, n=48)                          # bool[48, 48, 58], True = solid, exactly periodic
m = compute_metrics(p, n=48, L=4e-3)              # a_sf, wall/pore thickness, connectivity
check_manufacturable(p, L=4e-3).ok                # min wall 0.35 mm, pore throat 0.8 mm, connected
min_cell_size(0.5, 0.35)                          # smallest manufacturable L [m] (table, vectorized)
```

Background and design decisions: [docs/task1_explained.md](docs/task1_explained.md).
The precomputed tables in `src/voxlat/geometry/data/` are rebuilt with
`python scripts/task1_build_tables.py` (only needed if the geometry definitions change).

## Homogenization (Tasks 2-4)

```python
from voxlat.geometry import TPMSParams
from voxlat.homogenization import (
    extrapolated_conductivity_tpms, extrapolated_elasticity_tpms, extrapolated_permeability_tpms,
)

p = TPMSParams(w=0.0, rho=0.35)
k = extrapolated_conductivity_tpms(p)        # k_eff 3x3 [W/(m K)], Richardson R(32, 64)
c = extrapolated_elasticity_tpms(p)          # C_eff 6x6 Voigt [Pa], Richardson R(32, 64), ~1 min
c.C_eff, c.youngs_moduli, c.zener_ratio
c.fine.localization["uniaxial_z"].p99        # von Mises stress concentration (n = 64 grid)
K = extrapolated_permeability_tpms(p)        # Stokes permeability 3x3 in units of L^2, ~15 s
K.K, K.porosity, K.tortuosity                # multiply K by L^2 [m^2] for physical units
```

Voigt order is xx, yy, zz, yz, xz, xy with engineering shear strains; see the
`voxlat.homogenization.elasticity` docstring. Figures: `python scripts/task3_stiffness_vs_density.py`,
convergence study: `python scripts/task3_convergence.py` (both take `--quick`).

Permeability (Task 4): staggered (MAC) finite volumes on the voxels, no slip on every
solid face, solved with block-preconditioned MINRES; the `voxlat.homogenization.stokes`
docstring has the equations. Any periodic bool cell works too:
`permeability(cell, voxel_size=h)` returns K in units of h². Gyroid/diamond need one solve
(cubic symmetry), blends three. Scripts (all take `--quick`):
`scripts/task4_verification.py` (Poiseuille, ducts, inclined slits, sphere arrays vs
Zick & Homsy), `scripts/task4_convergence.py`, `scripts/task4_permeability_vs_porosity.py`.

## Closure dataset (Task 5)

Every homogenized property over the morphology box w ∈ [0, 1], ρ* ∈ [0.20, 0.50],
a_z ∈ [0.7, 1.5]: 256 scrambled-Sobol points + 8 corners + 12 edge midpoints + 14
pure-gyroid/diamond points (a_z = 1), each with geometry metrics, k_eff/k_s, C_eff/E_s
+ stress localization (6 unit stresses) and K/L², all at Richardson R(32, 64).

```powershell
python scripts/build_dataset.py --estimate-only     # times 3 samples, prints the runtime estimate
python scripts/build_dataset.py --n-jobs 4          # overnight; Ctrl+C-safe, rerun the same command to resume
python scripts/build_dataset.py --plot-only         # redraw results/figures/task5_closures_quicklook.png
python scripts/build_dataset.py --quick             # 10-second plumbing test at R(8, 16)
```

Each finished sample is saved immediately to `data/closures_parts/<id>.json`; the run
ends by consolidating everything into `data/closures.parquet` (one row per sample,
~170 columns, dimensionless: k/k_s, C/E_s, K/L², lengths/L). Progress and ETA go to the
console and `data/closures_build.log`. Choose `--n-jobs` ≤ physical cores and keep
n_jobs × ~1.5 GB below ~70 % of your RAM.

```python
from voxlat.closures import read_table
df = read_table("data/closures.parquet")
ok = df[df.status == "ok"]                   # columns: w, rho_target, a_z, keff_*, C11..C66, K_*, loc_*, ...
```

## Closure surrogates (Task 6)

GP (ARD Matérn-5/2, one per target), 5-member MLP deep ensemble and a small 3D CNN,
trained on the Task 5 table. Pure numpy + scikit-learn: no torch needed, models are
plain `.npz` files in `models/`.

```powershell
python scripts/task6_train_surrogates.py            # models/closure_gp.npz + closure_ensemble.npz, CV-recalibrated (~7 min)
python scripts/task6_train_surrogates.py --cnn      # + models/closure_cnn.npz (3D CNN for K*, E*, +15 min)
python scripts/task6_evaluate.py                    # 5-fold CV + extrapolation splits, tables + figures (~10 min)
python scripts/task6_cnn_experiment.py              # CNN vs GP vs ensemble (~80 min; --splits band sparse ~20 min)
```

```python
from voxlat.surrogates import ClosureModel
cm = ClosureModel.load("gp")                        # or "ensemble"
out = cm.predict((0.0, 0.35, 1.0), L=4e-3)          # theta = (w, rho*, a_z); L in m
K, K_std = out["K"]                                  # (3, 3) m^2; also K_principal, K_star, k_eff [W/mK],
                                                     # C_eff [Pa, Voigt], E, E_star, a_sf [1/m], loc_<case>_p99
```

Every tensor is symmetric positive definite and has the exact symmetry class of theta
(cubic / tetragonal / trigonal / triclinic); std's are first-order predictive standard
deviations. Accuracy table: STATUS.md, Task 6.

## Finite-gap study (Task 7, RQ1)

Lattice strips of N = 1 ... 8 cells between bonded solid walls, solved with the Task 2-4
solvers and compared with the bulk ("plain-wall") homogenized prediction; graded strips,
resolution and wall-thickness checks; a physical 1/N correction plus a GP on its residuals.

```powershell
python scripts/task7_finite_gap.py --estimate-only --n 48         # time 3 strips, print the runtime
python scripts/task7_finite_gap.py --suite all --n-jobs 2         # n = 32, all suites (~1.5 h on 2 cores)
python scripts/task7_finite_gap.py --suite uniform graded stretch --n 48 --n-jobs 4   # overnight, n = 48 (~3-4 h)
python scripts/task7_fit_discrepancy.py                           # models/finite_gap_correction.json + tables
python scripts/task7_figures.py                                   # results/figures/task7_*.png
```

```python
from voxlat.homogenization import FiniteGapCorrection
corr = FiniteGapCorrection.load()                                 # models/finite_gap_correction.json
K_t = K_bulk * corr.factor("K_t", N=6e-3 / L, w=0.0, rho=0.35)    # also k_n, C_nn, G_t, G_sz
```

The solvers accept `preconditioner="two_level"` (Jacobi + aggregation coarse space, no pyamg
needed), `effective_conductivity(..., directions=)` and `effective_elasticity(..., load_cases=)`.
Results and coefficients: STATUS.md, Task 7.

## Literature closures (Task 8)

The inertial (Forchheimer) coefficient and the interstitial heat-transfer coefficient come from published
pore-scale CFD of skeletal gyroid/diamond cells (Gajetti et al. 2025; Savoldi et al. 2026), with validity
boxes that warn on extrapolation. Table, recommendations and risks: `docs/task8_literature_closures.md`.

```powershell
python scripts/task8_literature_closures.py      # results/task8_*.csv, results/figures/task8_*.png (~5 s)
```

```python
from voxlat.closures import forchheimer_coefficient, interstitial_heat_transfer
cf = forchheimer_coefficient(phi, w)                                   # C_F (blends: log-linear in w)
r = interstitial_heat_transfer(U_s, phi, a_sf, K_bulk, w=w, conductivity=0.40,
                               kinematic_viscosity=2.3e-6, prandtl=20.7) # r.h_sf, r.h_sf_a_sf, r.nusselt
```

## Jacket device model (Task 9)

The cooling jacket unwrapped to (s = r θ, z), depth-averaged over the 6 mm gap: Darcy-Forchheimer flow between
the manifold slots (Task 6 surrogates × Task 7 finite-gap factor, Task 8 C_F), a two-equation heat model
(coolant + lattice solid, h_sf a_sf from Task 8, fin-calibrated sleeve coupling) plus the sleeve wall with the
q''(z) profile, a homogenized sandwich stress check and manufacturability flags. ~0.2-0.7 s per evaluation.

```python
from voxlat.device import JacketModel, evaluate
out = evaluate(rho=0.35, w=0.0, L=4e-3)                       # uniform gyroid, 3 L/min (scalars, arrays or f(s, z))
out["thermal_resistance"], out["delta_p"], out["pump_power"], out["mass"], out["min_structural_margin"]
model = JacketModel()                                          # reuse for many evaluations (Task 11)
out = model.evaluate(rho_field, w_field, L_field, a_z=1.0, flow_rate=5e-5, return_fields=True)
```

```powershell
python scripts/task9_jacket_model.py     # verification, grid study, flow sweep, sensitivity, figures (~20 s)
```

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
