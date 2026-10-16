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

## Baselines B1-B3 (Task 10)

Same `evaluate()` outputs as Task 9 for every baseline, plus `R_active` (hottest point of the cooled region,
excluding the sleeve under the manifold slots):

- **B1** helical rectangular-channel jacket in the 6 mm gap (end-ring manifolds), laminar/transitional/turbulent
  curved-duct correlations (Shah & London, Schmidt, Gnielinski/VDI); **B1s** the same channels running
  circumferentially between the lattice's manifold slots (like-for-like manifold treatment).
- **B2** best uniform gyroid (ρ*, L); **B3** density-only graded gyroid (w = 0, L fixed).

```python
from voxlat.device import ChannelDesign, UniformLatticeDesign, GradedDensityDesign, evaluate_baseline
out = evaluate_baseline(ChannelDesign(channel_width=1.75e-3, rib_thickness=0.75e-3, n_starts=20))   # B1, 3 L/min
out = evaluate_baseline(GradedDensityDesign((0.25, 0.23, 0.37, 0.50), 0.0, 4e-3), flow_rate=5 / 6e4)
```

```powershell
python scripts/task10_baselines.py           # tuning, verification, 1/3/5 L/min table, figures (~3 min)
python scripts/task10_baselines.py --quick   # coarse grids (~1 min)
```
Selected designs are stored in `results/baselines/*.json` (`design_from_dict(json["design"])` rebuilds them).

## Multi-objective optimization (Task 11, RQ2/RQ3)

Graded jackets: B-spline fields ρ*(s, z) 8×4, w(s, z) 8×4, L(s, z) 4×2 (72 variables, periodic in s), objectives
(R, pump power, mass) at 3 L/min from the Task 9 model, constraints L ≥ L_min(w, ρ*), |∇ρ*| L ≤ 0.05,
structural margin ≥ 1 (a shared feasibility repair handles the first two). Two manifold layouts:
`slots` (reference config, compared with B1s/B2/B3) and `end_rings` (new `JacketOptions(manifold_layout="end_rings")`,
axial flow, compared with B1).

```powershell
pip install -e .[opt]                         # pymoo + botorch (torch); NSGA-II and qNEHVI need them
python scripts/task11_optimize.py --quick     # 1-2 min smoke run
# full study: 2 layouts x {NSGA-II 100x100, qNEHVI 200, Sobol 1000} x 5 seeds (~75 CPU-h: ~20 h with --parallel 4
# on the laptop; the final study ran in 2 h 35 min on 30 GitHub runners, .github/workflows/task11-study.yml)
python scripts/task11_optimize.py --layout slots end_rings --method nsga2 qnehvi sobol --seeds 0 1 2 3 4 --parallel 4
python scripts/task11_uniform_references.py   # uniform gyroid/diamond references, both layouts (~2 min)
python scripts/task11_analysis.py             # figures, tables, results/designs/*.json (~5 min)
```
Runs are independent; `--skip-done` skips finished runs and resumes interrupted ones from their last checkpoint, and a
crashed worker is restarted automatically (`--retries`). Each run writes
`results/task11/<layout>/<method>_s<seed>.csv.gz` (+ `.json` run record), checkpointed every ~2 min.

```python
from voxlat.optimize import JacketProblem, ProblemSettings, run_qnehvi
p = JacketProblem(ProblemSettings(layout="end_rings"))
run_qnehvi(p, budget=40, n_init=16, seed=0)          # BoTorch qLogNEHVI if installed, else numpy EHVI
p.table()                                             # every proposal: x, objectives, constraints
```

## PicoGK geometry export (Task 12)

`voxlat.export` turns a Task 11 design JSON into a printable jacket: a graded TPMS evaluated **in C#** as a PicoGK
implicit (phase-field grading in conformal cylindrical coordinates, integer cells around), clipped to the gap and
unioned with sleeve, outer wall, end caps and manifold ports. Python writes the export config, calls one C# entry
point through pythonnet, then checks the PicoGK voxel masks (min wall, min pore/throat, inlet→outlet connectivity,
trapped pockets, loose solid). `voxlat/export/field.py` is the numpy mirror of the C# implicit (the specification),
used for the self-test and a pure-numpy sector generator.

**One-time setup** (Windows; PicoGK ≥ 1.7.6 as in `legacy/noyron_core.py`; .NET 9 SDK from dotnet.microsoft.com):
```powershell
pip install -e .[export]                                    # pythonnet
$env:PICOGK = "D:\nyron 2\PicoGK_Engine\bin\Release\net9.0" # folder with PicoGK.dll + picogk.1.7.dll (+ tbb12.dll ...)
dotnet build src\voxlat\export\dotnet\VoxLat.Export\VoxLat.Export.csproj -c Release "-p:PicoGKDir=$env:PICOGK"
dotnet build src\voxlat\export\dotnet\VoxLat.Export.Probe\VoxLat.Export.Probe.csproj -c Release
python scripts\task12_export.py selftest        # C# implicit == numpy mirror (max |diff| < 1e-6 mm)
pytest tests\test_export_dotnet.py -v           # same, as tests
```

**Run**
```powershell
# 30-degree sector at 0.05 mm around the most critical cell (smallest L/L_min), viewer window, then checks
python scripts\task12_export.py build --design results\designs\task11_slots_best_at_cap.json --mode sector --picogk-dir $env:PICOGK
# full jacket (STL for assembly / visual; walls are not resolved at 0.2 mm, so no wall check)
python scripts\task12_export.py build --mode full --voxel 0.2 --headless --picogk-dir $env:PICOGK
python scripts\task12_export.py estimate --mode sector        # voxels, memory, STL size before a long run
python scripts\task12_export.py numpy --angle 30 --voxel 0.05 # same sector without .NET (numpy), checks only
python scripts\task12_export.py check --config results\export\<name>_config.json --stl results\export\<name>.stl --voxel 0.05
python scripts\task12_figures.py
```
Outputs in `results/export/`: `<name>.stl`, `<name>_mask(.json/.bin)` (cleaned voxels from PicoGK slices),
`<name>_mask_raw.*`, `<name>_report.json` (C#), `<name>_checks.json` (Python checks + numpy comparison),
`<name>_config.json`. Sector STLs are in the sector-local frame (sector centred on +x). `--headless` skips the
viewer; `--keep-viewer` keeps it open (close the window to continue to the checks).

## Conventions

- SI units inside the code; unit cells on normalized coordinates [0,1)³ scaled by cell size L.
- Voxel arrays are `bool[nx, ny, nz]`, `True` = solid, index order (x, y, z).
- Every solver documents its governing equations and boundary conditions in its docstring.
- Every physics module ships pytest verification tests against analytic or published cases.
- Data rows carry `run_record()` metadata; figures go through `save_figure()` (300 dpi PNG),
  each produced by a script in `scripts/`.
- Seeds are deterministic (`set_seed`).

## Reproduce the paper

Every paper figure (F01–F40) and table (T01–T09) has its own script in `scripts/experiments/`
(`F20_finite_gap_error_vs_N.py`, `T07_rq2_equal_pump_power.py`, ...). The catalog
`scripts/experiments/catalog.py` records, for each one, the output file, the caption, the producing
script in `scripts/`, its inputs and a laptop runtime estimate. The slot-manifold optimum used in the
paper is the manufacturable **`results/designs/task12_slots_best_at_cap_gyroid.json`** (Task 12).

**1. Environment.** Use the lock file for your platform (pinned versions of a working install, all
extras), then install VoxLat itself without dependencies:

```powershell
py -3.13 -m venv .venv; .\.venv\Scripts\Activate.ps1          # the laptop runs Python 3.13
python -m pip install -r requirements\lock-windows-py3.13.txt
python -m pip install -e . --no-deps
python -m pytest -m "not slow"                               # ~5 min
```
Linux: `requirements/lock-linux-py3.12.txt` (CPU-only torch; the file carries the PyTorch index URL).
The lock files are written by `.github/workflows/reproduce.yml`, which also runs the tests and the
quick and replot reproductions below on Windows and Linux.

**2. Three ways to reproduce**

| command | what it does | time (laptop) |
|---|---|---|
| `python scripts/experiments/run_all.py --quick` | every producer at plumbing size (tiny grids, budgets), in a throw-away copy of the repo (`<tmp>/voxlat_quick`); committed results are never touched | ~10–15 min |
| `python scripts/experiments/run_all.py --replot` | re-plots / re-tabulates from the committed `data/`, `results/`, `models/` (no long solves); rebuilds tables, gallery and audit | ~10 min |
| `python scripts/experiments/run_all.py` | full recomputation of every number in STATUS.md, in dependency order | ~14 h + the Task 11 study (~20 h with 4 parallel runs, or 2 h 35 min on GitHub Actions) |

```powershell
python scripts/experiments/run_all.py --list                # items, producers, which modes exist
python scripts/experiments/run_all.py --estimate            # full-run time budget per step
python scripts/experiments/run_all.py --only F20 T07        # one figure/table (+ its upstream steps)
python scripts/experiments/run_all.py --only F20 --no-deps  # just the producer, reuse committed inputs
python scripts/experiments/F33_hypervolume.py --replot      # same, through the per-item script
python scripts/experiments/run_all.py --dry-run             # print the commands only
```
Long steps resume where they stopped (Task 5 dataset, Task 7 strips, Task 11 runs with
`--skip-done`). Steps whose optional modules are missing are reported as `skipped`. The PicoGK
build (Task 12) is never run automatically: it needs the Windows .NET/PicoGK setup — see
"PicoGK geometry export (Task 12)"; `run_all` prints the command. Each run writes
`results/reproduce/<mode>_<time>/summary.json` with the wall time of every step.

Full-run budget (from STATUS.md; `--estimate`): Task 11 study ~75 CPU-h, Task 5 dataset 7.4 h
(27.8 CPU-h), Task 7 strips 1.5 h, Task 6 CNN 80 min, everything else ~4.5 h together.

**3. Outputs**
- `results/figures/*.png` (300 dpi) and `results/tables/T0x_*.csv/.md`
  (`python scripts/experiments/tables.py all`, seconds; pure post-processing of committed results).
- `results/figure_gallery.pdf` — every figure and the main tables with captions
  (`python scripts/experiments/make_gallery.py`).
- `results/reproduce/consistency_report.md` — audit of the figure/table data against the numbers
  quoted in STATUS.md (`python scripts/experiments/check_consistency.py`; exit code 1 if anything is
  flagged).

## Manuscript (Task 14)

The manuscript draft, its source and builder, the JOSS paper and the submission files are in [`docs/manuscript/`](docs/manuscript/). See STATUS.md, Task 14.

## License

MIT, see [LICENSE](LICENSE).
