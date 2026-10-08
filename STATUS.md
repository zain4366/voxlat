# VoxLat — STATUS (hand-off log)

Every task appends here: what was built, public API, key numbers (resolution,
timings, errors), open issues. **The next task reads this file first.**
Plan: project doc `claude/voxlat_task_plan.md`. Reference numbers: `configs/reference.yaml`.

| # | Task | State |
|---|---|---|
| 0 | Repo scaffold + config | **done** (2026-10-07) |
| 1 | TPMS unit-cell geometry | **done** (2026-10-08) |
| 2 | Conduction homogenization k_eff | **done** (2026-10-08) |
| 3 | Elasticity homogenization C_eff | **done** (2026-10-08) |
| 4 | Stokes permeability K | **done** (2026-10-08) |
| 5 | Closure dataset | not started |
| 6 | Closure surrogates | not started |
| 7 | Finite-gap study (RQ1) | not started |
| 8 | Literature closures (Nu, Forchheimer) | not started |
| 9 | Homogenized jacket device model | not started |
| 10 | Baselines B1–B3 | not started |
| 11 | Multi-objective optimization (RQ2, RQ3) | not started |
| 12 | PicoGK geometry export | not started |
| 13 | Reproducibility pass + figure set | not started |
| 14 | Manuscript draft (+ JOSS) | not started |

---

## Task 0 — Repo scaffold + config  ✅

**Built**
- src-layout package `voxlat` (v0.1.0) with subpackages `geometry`, `homogenization`,
  `closures`, `surrogates`, `device`, `optimize`, `export` (empty, docstring says which task
  fills them) and `utils` (implemented).
- `pyproject.toml`: Python >= 3.11; core deps numpy, scipy, scikit-image, scikit-learn,
  pandas, pyarrow, matplotlib, pyyaml, joblib, tqdm, pytest; extras `ml` = torch,
  `opt` = pymoo, botorch. pytest config: `testpaths = ["tests"]`, `legacy/` excluded,
  marker `slow` registered.
- `configs/reference.yaml`: every number of plan §2, SI units, comments give plan units.
- `scripts/task0_heat_flux_profile.py` → `results/figures/task0_heat_flux_profile.png`.
- `legacy/`: original Noyron files, unchanged, plus `legacy/README.md` (what each file was,
  what gets reused).
- `.github/workflows/tests.yml`: pytest on windows-latest + ubuntu-latest, Python 3.11/3.12.
- `.gitignore` (venvs, caches, `*.stl`, `data/fields/`, model binaries, .NET `bin/ obj/`).

**Public API (`voxlat.utils`)**
```python
load_config(path=None, overrides=None) -> ReferenceConfig   # frozen dataclass tree, SI
cfg.motor / jacket / heat_flux / coolant / operating / manifolds / material / loads
cfg.design_bounds.<relative_density|blend_w|cell_size|axial_stretch> -> Range(lo, hi)
cfg.manufacturing.<min_wall_thickness|min_pore_size|max_density_change_per_cell>
cfg.q_wall(z)             # q''(z) in W/m^2, z in m (scalar or array); raises outside [0, L_ax]
cfg.mean_heat_flux        # Q / (2 pi r_s L_ax)
cfg.heated_area, cfg.to_dict()
cfg.jacket.lattice_inner_radius / lattice_outer_radius / outer_radius / lattice_mean_radius / lattice_volume
cfg.coolant.dynamic_viscosity / prandtl ; cfg.material.shear_modulus ; cfg.motor.total_loss
heat_flux_profile(z, *, axial_length, mean_flux, amplitude=0.25, profile="end_peaked_quadratic")
set_seed(seed=DEFAULT_SEED) -> np.random.Generator   # random, numpy, torch (if installed), PYTHONHASHSEED
get_logger(name) -> logging.Logger                   # "voxlat.<name>", one handler, $VOXLAT_LOG_LEVEL
run_record(params, *, resolution=None, wall_time=None, **extra) -> dict
    # keys: params, params_json, voxlat_version, git_hash, resolution, wall_time_s,
    #       timestamp_utc, python, numpy, platform, extra   (all JSON-serializable)
Stopwatch(), stopwatch()  # context manager; .elapsed in s
save_figure(fig, name, directory=None, dpi=300) -> Path  # results/figures/<name>.png
repo_root(), data_dir(), models_dir(), results_dir(), figures_dir()
```
Env vars: `VOXLAT_CONFIG` (alternative YAML), `VOXLAT_ROOT` (repo root override),
`VOXLAT_LOG_LEVEL`.

Naming note: `cfg.heat_flux` is the config *section* (profile name + amplitude); the
profile *function* is `cfg.q_wall(z)`.

**Key numbers (computed from the config)**
| Quantity | Value |
|---|---|
| Lattice gap | r = 71.5 … 77.5 mm, h = 6.0 mm; jacket outer radius 79.5 mm |
| Mean heat flux q̄ | 35 014 W/m² (on r = 70 mm) |
| q''(0)/q''(L/2) | 2.000 (52.5 / 26.3 kW/m²) |
| ∫ q'' 2πr_s dz | 770.000 W (quad, rel. err < 1e-10); trapezoid n = 51 → 2e-4 rel. err |
| Coolant Pr | 20.72 |
| Total motor loss P(1/η − 1) | 957 W → Q = 770 W is an 80 % jacket share (assumption) |
| Nominal flow | 5.0e-5 m³/s (3 L/min); sweep 1–5 L/min stored in m³/s |

**Tests**: 34 passed (`tests/test_config.py`, `tests/test_utils.py`) in ~2 s.
Includes the two required smoke tests: config loads; q''(z) integrates to Q within 0.1 %.
Also checks frozen/typed config, unknown/missing keys rejected, overrides, env-var path,
seeding reproducibility, run_record JSON-serializability, 300-dpi PNG output size.

**Open issues / decisions for later tasks**
1. **Grading-gradient limit is a placeholder**: `manufacturing.max_density_change_per_cell = 0.05`
   (|∇ρ*|·L ≤ 0.05). The plan gives no number. Set it from the Task 7 graded-strip study
   before Task 11.
2. Q = 770 W vs total loss 957 W: the jacket is assumed to take ~80 % of the losses
   (rest via end windings / rotor / shaft). State this in the manuscript.
3. Manifold "width" is an arc length at the lattice (15 mm); Task 9 must decide whether it is
   measured at the mean radius (default assumption) or at r_inner.
4. Flow-rate sweep is stored in m³/s; use `cfg.operating.nominal_flow_rate_lpm` for labels.
5. `legacy/` files were recreated from the project docs; if the copies on `D:\nyron 2`
   differ, the D: versions are the originals.
6. Workspace note: PyPI was blocked in the build sandbox, so tests ran against
   pre-installed numpy 2.5 / scipy 1.18 / matplotlib 3.11 / pytest 9.1 with
   `pip install -e . --no-deps`; pyarrow, tqdm, torch were not exercised (not used yet).
   The CI workflow and the laptop install test the full dependency resolution.

---

## Task 1 — TPMS unit-cell geometry  ✅

**Built**
- `src/voxlat/geometry/tpms.py` — level sets, parameters, periodic voxelization, density
  thresholds (bisection + shipped table), metrics, manufacturability, feasible region.
- `src/voxlat/geometry/voxel_tools.py` — generic *periodic* voxel tools reused by Tasks 2–4, 7:
  marching-cubes surface area, periodic EDT, sub-voxel interface distance, local thickness,
  throat (passable-sphere) diameter, periodic connected components with wrap detection.
- `src/voxlat/geometry/data/tpms_threshold_table.npz` — c(w, ρ) for network + sheet,
  w = 0:0.05:1 × ρ = 0.01:0.01:0.99, sampled at n_ref = 128 (~29 kB, committed).
- `src/voxlat/geometry/data/tpms_morphology_table.npz` — wall/pore/throat/a_sf/connectivity
  over w = 0:0.125:1 × ρ = 0.15:0.025:0.55 at n = 48 (network, a = 1). Build: 6 min on 2 cores.
- `scripts/task1_build_tables.py`, `scripts/task1_tpms_figures.py`; `pyproject.toml` ships `data/*.npz`.
- Tests: `tests/test_tpms.py` (57), `tests/test_voxel_tools.py` (17).

**Decisions (each documented in the module docstring)**
1. *Level sets*: G = sinX cosY + sinY cosZ + sinZ cosX, D = sinX sinY sinZ + sinX cosY cosZ +
   cosX sinY cosZ + cosX cosY sinZ, X = 2π ξ_x. Normalized by their maxima (3/2, √2) so both
   span [−1, 1]; blend φ_w = (1−w)φ_G + wφ_D. G and D are orthogonal (disjoint Fourier modes);
   RMS 0.577 vs 0.500. All fields are odd → c(w, ρ = 0.5) = 0 exactly.
2. *Network default*: solid where φ ≤ c → one solid labyrinth + ONE connected fluid domain
   (single coolant, manifold to manifold). Sheet (|φ| ≤ c) splits the fluid into 2 disjoint
   labyrinths (verified by test) — wrong topology for this jacket; supported for comparison only.
3. *Anisotropy*: cubic voxels of edge h = L/n; stretched axis gets n_i = round(n·a_i) voxels;
   the represented stretch is snapped to a_i = n_i/n (`effective_stretch`, error ≤ 1/(2n)).
   Keeps every downstream solver isotropic (one element stiffness, one face conductance).
   L means the in-plane cell size; the cell is L × L × a_z L.
4. *Exact periodicity*: sampling at voxel centres ξ = (k + ½)/n_i; `np.tile` of a cell equals the
   supercell (tested). Never use linspace(0, 1, n) (duplicates the face layer).
5. *Thickness = local thickness* (Hildebrand & Rüegsegger 1997) on **sub-voxel** distances to the
   marching-cubes surface (cKDTree with periodic boxsize), ball radius D(q) + 0.5 voxel ("slack",
   compensates off-grid ideal centres; calibrated: mean bias ≈ 0, worst ±1 voxel on random
   slabs/cylinders). Voxel-only EDT under-read a 12-voxel cylinder as 9 — do not go back to it.
6. *Throat* = 2(r* + slack), r* = largest r with {D ≥ r} percolating in x, y, z (26-connected
   centres; 6-connected under-reads diagonal diamond struts). Calibrated on rod networks (±1 voxel).
7. *Manufacturability default*: wall = **min local thickness of solid** ≥ 0.35 mm (strict: any neck);
   pore = **fluid throat** ≥ 0.8 mm (powder removal / clogging); plus single percolating solid and
   fluid (6-connectivity). Both scale ∝ L → feasible set is simply **L ≥ L_min(w, ρ)**.

**Public API (`voxlat.geometry`, all in `voxlat.geometry.tpms`)**
```python
TPMSParams(w=0.0, rho=0.3, a=(1,1,1), kind="network"); TPMSParams.from_design(w, rho, a_z)
level_set(xi_x, xi_y, xi_z, w) -> phi          # any points, period 1, |phi| <= 1
sample_level_set(shape, w) -> phi[nx,ny,nz]   # voxel centres, separable (n=128: 0.1 s)
grid_shape(params, n), effective_stretch(params, n)
threshold_for_density(w, rho, n|shape, kind, g=None) -> c        # bisection on this grid
threshold_from_table(w, rho, kind) -> c        # vectorized continuum table (1e5 pts: 0.13 s)
voxelize(params, n=48, threshold=None, method="exact"|"table", return_field=False)
    -> bool[nx,ny,nz]  (True = solid)  | (solid, phi, c)
specific_surface_area(phi, c, kind, n=None) -> a_sf*L
compute_metrics(params, n=48, L=None, thickness=True, connectivity=True) -> TPMSMetrics
    .relative_density .a_sf_L .a_sf_voxel_L .wall/.pore: ThicknessStats(min,p1,p5,mean,max,throat) [/L]
    .solid_connected .fluid_connected .a_sf() .wall_m() .pore_m() .as_row()
check_manufacturable(params, L, n=48, cfg=None, wall_statistic="min", pore_statistic="throat")
    -> ManufacturabilityReport(ok, wall, pore, ..., resolution_limited, reasons)
min_cell_size(w, rho, cfg=None, return_parts=False) -> L_min [m]   # Task 11 constraint: L_min/L - 1 <= 0
feasible_region(w, rho=None, L=None, cfg=None) -> FeasibleRegion(rho, L_min, L_min_wall, L_min_pore, L, mask)
load_threshold_table(kind), load_morphology_table(), build_morphology_table(...)
# voxel_tools (generic, periodic):
periodic_surface_area, voxel_face_area, periodic_edt, periodic_surface_points, interface_distance,
local_thickness, throat_diameter, thickness_stats, periodic_components(phase, connectivity=6|26)
    -> PeriodicComponents(n_components, volume_fractions, percolates, wrap_rank, labels)
```

**Key numbers**
| Quantity | Value |
|---|---|
| Nodal G, c = 0: area per cell (Richardson, n → ∞) | 3.0917 vs exact minimal surface 3.0915 (+0.005 %) |
| Nodal D, c = 0 | 3.8381 vs 3.8377 (+0.009 %) |
| a_sf convergence (ρ = 0.3, fixed c) | order 2.0; max rel. error n = 24: 0.63 %, 32: 0.33 %, **48: 0.16 %**, 64: 0.09 % |
| a_sf·L at ρ = 0.3 (continuum) | G 2.862, D 3.541, w = 0.5 3.027 (staircase/voxel-face area ≈ 1.5× — never use it) |
| Density, exact bisection | closest achievable voxel fraction; < 0.1 % rel. at n = 32 (test: < 0.5 %) |
| Density, table threshold at n = 64 | max 0.42 % rel. error over 20 random (w, ρ) |
| Timings (laptop-class CPU, 1 core) | voxelize n = 48: 2 ms; metrics w/o thickness 0.05 s; **full metrics n = 32: 0.9 s, 48: 4 s, 64: 11 s** |
| Thickness, n = 48 vs 64 (G, D, ρ = 0.2–0.5) | wall min ≤ 0.016 L (≈5 %), throats ≤ 0.006 L (≈1–2 %) |
| Gyroid ρ = 0.2 / 0.35 / 0.5 (n = 48) | wall min 0.240 / 0.344 / 0.427 L; pore throat 0.625 / 0.528 / 0.440 L |
| Diamond ρ = 0.2 / 0.35 / 0.5 | wall min 0.198 / 0.292 / 0.365 L; pore throat 0.531 / 0.449 / 0.370 L |
| L_min (G) | 1.46 mm (ρ = 0.2) … 1.82 mm (ρ = 0.5) → 100 % of the 2–6 mm box feasible |
| L_min (D) | 1.77 … 2.16 mm → 99.5 % feasible (pore throat binds at ρ > 0.45, L ≈ 2 mm) |
| L_min (w = 0.5) | 1.98 … 4.80 mm, non-monotone → 79 % feasible (wall binds) |

**Recommended resolution**: n = 48 per cell for metrics/tables (a_sf 0.16 %, thickness ~1 voxel);
n = 32 is fine for quick connectivity checks. Tasks 2–4 will choose their own n by convergence.

**Figures** (`results/figures/`): `task1_tpms_cells.png` (renders + slices, ρ = 0.3),
`task1_asf_convergence.png` (+ `results/task1_asf_convergence.csv`),
`task1_morphology_vs_density.png`, `task1_feasible_region.png`.

**Open issues / notes for later tasks**
1. **Blends develop pinch-point necks (RQ2-relevant).** For 0 < w < 1 the min wall drops to
   0.04–0.1 L at some densities, non-monotonically (e.g. w = 0.5: 0.073 L at ρ = 0.375); at
   w = 0.5, ρ = 0.5 the neck is resolution-limited (≈3.5 voxels at every n → true thickness → 0,
   a singular point of the level set). w = 0.875, ρ = 0.15 even has a floating solid island.
   Pure G/D are smooth and monotone. Pointwise linear blending of G and D is therefore NOT a
   smooth morphological interpolation. Options for Task 11: keep the strict constraint (default),
   use `wall_statistic="throat"` (smooth, load-path neck only), or restrict w to {0, 1} plus
   narrow transition zones. Discuss in the paper.
2. The morphology table is coarse in w (0.125) while pinch points move continuously with w →
   for blends, treat `min_cell_size` as screening and re-check final designs with
   `check_manufacturable` (direct, n = 48).
3. With the reference limits the manufacturability constraint barely binds for pure G/D inside
   L = 2–6 mm. If the paper wants an active constraint, revisit `min_pore_size` / `cell_size` bounds.
4. Table and metrics are for a = (1,1,1). For a_z ≠ 1 call `compute_metrics` / `check_manufacturable`
   directly (stretching changes thicknesses non-trivially).
5. Citations to verify before the manuscript: exact areas G 3.0915 / D 3.8377 (attributed to
   Schröder-Turk, Fogden & Hyde 2006, Eur. Phys. J. B 54:509; our nodal values agree to 0.01 %);
   Schoen 1970 (NASA TN D-5541); von Schnering & Nesper 1991 (Z. Phys. B 83:407) for nodal forms;
   Hildebrand & Rüegsegger 1997 (J. Microsc. 185:67) for local thickness.
6. Workspace note: PyPI blocked again; tests ran on numpy 2.5.3 / scipy 1.18.1 / scikit-image 0.26
   / pytest 9.1.1, Python 3.13. scikit-image ≥ 0.22 emits harmless NumPy-2.5 deprecation warnings.

**Tests**: 108 passed (Task 0: 34, Task 1: 74 incl. 1 `slow`) in ~55 s; `pytest -m "not slow"` ~20 s.

## Task 2 — Conduction homogenization  ✅

**Built**
- `src/voxlat/homogenization/conduction.py` — periodic cell problem for k_eff (3×3):
  cell-centred finite volumes on voxels, harmonic-mean face conductances, periodic fluctuation
  θ, three unit macroscopic gradients, PCG (scipy `cg`, rtol 1e-8; pyamg smoothed-aggregation
  V-cycle if installed, else Jacobi). k_eff computed by **flux averaging and by energy**; the
  function raises if they differ by > 1e-5 (observed 1e-9 … 2e-9). Contrast 1e3 and the
  **k_f = 0 limit** work: non-conducting voxels are dropped, the singular-but-consistent
  per-component blocks (floating islands) are handled by projecting b onto range(A).
- `src/voxlat/homogenization/convergence.py` — `richardson`, `observed_order`,
  `fit_convergence` (shared by Tasks 3, 4, 7).
- `voxlat.geometry.voxelize(..., offset=)` / `sample_level_set(..., offset=)`: optional rigid shift
  of the TPMS against the grid (backward compatible; default = old behaviour). Used to separate
  grid-alignment noise from the systematic staircase error.
- Scripts: `scripts/task2_convergence.py` (~15 min, 2 cores), `scripts/task2_keff_vs_density.py`
  (~2 min). Both take `--quick`.
- `pyproject.toml`: new optional extra `amg = ["pyamg>=5.0"]`; README updated.
- Tests: `tests/test_conduction.py` (62 incl. 4 `slow`; the AMG test skips without pyamg),
  + 1 offset test in `tests/test_tpms.py`.

**Public API (`voxlat.homogenization`)**
```python
effective_conductivity(cell, k_s=None, k_f=None, *, tol=1e-8, preconditioner="auto"|"amg"|"jacobi"|"none",
                       maxiter=None, return_fields=False, check=True, agreement_tol=1e-5) -> ConductivityResult
    # cell: bool (True = solid, needs k_s, k_f) or float voxel conductivities; periodic in x, y, z
ConductivityResult: .k_eff (energy form, symmetric) .k_eff_flux .k_eff_energy .agreement .eigenvalues
    .mean (tr/3) .anisotropy ((lmax-lmin)/mean) .is_spd .relative() .as_row() .iterations .residuals
    .preconditioner .wall_time .n_active .theta (3,nx,ny,nz) if return_fields
effective_conductivity_tpms(params, n=48, k_s=None, k_f=None, *, cfg=None, offset=None, **kw)  # k from config
extrapolated_conductivity_tpms(params, n=(32, 64), k_s=None, k_f=None, ...) -> ExtrapolatedConductivity
    # .k_eff = (n2 K(n2) - n1 K(n1))/(n2 - n1), .coarse .fine .correction .mean .eigenvalues   <- PRODUCTION
hashin_shtrikman_bounds(phi_s, k_s, k_f) -> (lower, upper); wiener_bounds(...); rayleigh_sc_spheres(phi, k_p, k_m)
face_conductances(k), assemble_conduction_system(kf) -> (A, active)       # reusable for Task 7 strips
richardson(n1,f1,n2,f2,p=1), observed_order(n3, f3), fit_convergence(n, f, p=1|None) -> ConvergenceFit
```

**Verification (all pass)**
| Test | Result |
|---|---|
| Homogeneous cells (incl. 5×7×9, 1×4×3 grids) | exact to 1e-13, 0 iterations |
| Two-phase laminates along x, y, z (k_f/k_s = 0.1, 1e-3); 13-layer random laminate | harmonic across / arithmetic along to 1e-9 (exact for harmonic faces) |
| Laminate with k_f = 0 | k_across = 0, k_along = φ k_s exactly; fluid DOFs removed |
| Flux vs energy k_eff | agree to ~1e-9 (TPMS), assertion verified to fire on a sloppy solve |
| SPD, Wiener bounds | random 60 % cells at 3 contrasts |
| HS bounds | G and D, ρ = 0.2/0.35/0.5, k_f/k_s = 1/325, 1e-3, 0 — all eigenvalues inside |
| Near-isotropy G, D (n = 32, 48) | diagonal equal to 2e-4, off-diagonal < 1e-3, anisotropy < 2e-3 |
| Invariances | periodic roll → identical; axis permutation → permuted tensor; k_f → 0 continuous |
| SC sphere array vs Rayleigh (1892) | conducting spheres (α = 10): < 1 % at n = 48 (0.01–0.65 %); insulating: 0.4–2 %, halves with 2n |
| Observed order (offset-averaged 16/32/64) | p ≈ 1 for G and D (slow test) |

**Convergence (the important number).** The voxel staircase drops diagonal (edge/corner) contacts,
so k_eff converges **from below, first order** (k(n) = k_∞ − C/n): offset-averaged triples
24/48/96 give p_obs = 0.77–1.06 (8 cases). Reference k_ref = Richardson(48, 96, p = 1) on
3-offset averages. Errors vs k_ref (`results/task2_convergence_summary.csv`), G/D, ρ = 0.2–0.5:

| Estimator | max \|error\| over 8 cases | cost (n = 48 ≈ 1.6 s) |
|---|---|---|
| raw n = 32 | 5.8 – **17 %** low | 0.3 s |
| raw n = 48 | 4.0 – **12 %** low | 1.6 s |
| raw n = 64 | 3.0 – **8.3 %** low | 6 s |
| R(32, 48) | 1.8 % | 2 s |
| R(48, 64) | 2.3 % (noise amplified ×4) | 8 s |
| **R(32, 64)** | **0.7 %** | **~6.5 s** |

Error is largest at low ρ (more surface per solid volume), and larger for D than G.
Alignment noise of a single offset-0 grid ≈ 0.1–0.5 % of k (rms around the 1/n fit).
Uncertainty of k_ref itself (p = 1 vs observed p): ≤ 1.1 %.

**Recommended resolution: two-grid Richardson R(32, 64) = `extrapolated_conductivity_tpms`**
(error ≲ 1 % vs the reference; single grids are biased low by 3–17 %). If only one grid is
affordable, n = 64 with a −(3–8) % bias. Use R(32, 64) for the Task 5 dataset.

**Timings** (1 core of the build sandbox, Jacobi PCG, ρ = 0.35, full 3×3 tensor incl. assembly):
n = 32: 0.3–0.5 s · **n = 48: 1.6–2.9 s** (target < 30 s ✅) · n = 64: 5.6–9.4 s · n = 96: ~25 s.
Iterations ∝ n (G 179, D 178, blend 266 at n = 48). Blends are slowest. AMG not tested here (pyamg
not installable in the sandbox; code path guarded, test skips).

**Results — k_eff/k_s (R(32,64)), k_f/k_s = 1/325** (`results/task2_keff_vs_density.csv`)
| ρ | 0.20 | 0.30 | 0.35 | 0.40 | 0.50 |
|---|---|---|---|---|---|
| Gyroid | 0.104 | 0.168 | 0.208 | 0.246 | 0.335 |
| Diamond | 0.097 | 0.163 | 0.205 | 0.246 | 0.336 |
| Blend w = 0.5 (mean eig.) | 0.057 | 0.097 | 0.117 | 0.191 | 0.308 |
| HS upper | 0.145 | 0.225 | 0.267 | 0.311 | 0.401 |

- G and D are within 0.5–7 % of each other, at 67–83 % of the HS upper bound; fraction rises with ρ.
- Power-law fits over ρ = 0.2–0.5: G k/k_s ≈ 0.81 ρ^1.29, D ≈ 0.85 ρ^1.36 (config fluid);
  k_f = 0: G 0.81 ρ^1.31, D 0.86 ρ^1.38. The fluid (k_f/k_s = 1/325) adds only 1–4 % to k_eff.

**Open issues / notes for later tasks**
1. **Blends are not isotropic (RQ2-relevant).** G and D share only the trigonal sub-symmetry, so
   w = 0.5 gives a **uniaxial k_eff about [111]** (test-verified): at ρ = 0.3 eigenvalues
   0.066 / 0.066 / 0.159 k_s. Below ρ ≈ 0.375 the blend conducts ~40 % of HS upper and is
   strongly anisotropic; between ρ = 0.35 and 0.40 it jumps to ~60 % and becomes nearly
   isotropic — the same pinch-neck topology change found in Task 1 (open issue 1). Consequences:
   (a) Task 6 surrogates must predict the full tensor for 0 < w < 1 (the principal axes are
   the cube diagonals, not x/y/z, so diagonal-only closures are wrong for blends);
   (b) Task 9's 2-D jacket model must rotate/project this tensor into (r, s, z);
   (c) graded designs crossing ρ ≈ 0.375 at mid w see a property discontinuity.
2. **Staircase error will also hit Tasks 3 and 4** (voxel FEA stiffness and voxel Stokes are
   first-order in h too). Plan for the same two-grid Richardson + offset-averaging study there;
   `voxlat.homogenization.convergence` is ready. Tried and rejected: diagonal-only "laminate
   composite voxels" (Kabel et al. 2015 idea without the off-diagonal terms) — no improvement
   (G ρ = 0.3: 0.1582 vs 0.1596 binary at n = 48), because the dead staircase corners remain
   poorly connected in a 7-point stencil. A full-tensor (27-point / FE) composite-voxel scheme
   could fix it; not worth it while R(32, 64) costs 6 s.
3. Anisotropic cells (a_z ≠ 1) work (grid n × n × round(n a_z)); at ρ = 0.3, n = 32: a_z = 1.5 gives
   k_zz/k_xx = 1.53 (G) / 1.58 (D), a_z = 0.7 gives 0.68 / 0.58 — stretch matters as much as w.
   The Richardson pair then refers to the in-plane n.
4. Floating-point ties: G/D cyclic symmetry is exact mathematically but voxels exactly at the
   threshold can flip by round-off, so diagonals agree to ~1e-4, not machine precision.
5. Citations to verify: Hashin & Shtrikman 1962 (J. Appl. Phys. 33:3125); Rayleigh 1892
   (Phil. Mag. 34:481) and Perrins, McKenzie & McPhedran 1979 (Proc. R. Soc. A 369:207) for the
   sphere-array check; Kabel, Merkert & Schneider 2015 (CMAME 294:168) for composite voxels.
6. Workspace: PyPI blocked again (no pyamg); numpy 2.5.3, scipy 1.18.1, Python 3.13.
   Install `pip install -e ".[amg]"` on the laptop to get AMG; results must match Jacobi to 1e-6
   (`test_amg_matches_jacobi`).

**Figures** (`results/figures/`): `task2_keff_vs_density.png` (k_eff/k_s vs ρ for G, D, blend with HS
bounds, config and k_f = 0; panel b = fraction of HS upper), `task2_convergence.png`
(1/n plot of offset-0 grids; log-log offset-averaged error with slope −1).
Data: `results/task2_keff_vs_density.csv`, `results/task2_convergence.csv`,
`results/task2_convergence_summary.csv`, `results/task2_pytest_log.txt`.

**Tests**: 170 passed + 1 skipped (pyamg) in ~2.5 min (Tasks 0–2, incl. slow); `pytest -m "not slow"` ~1.5 min.

## Task 3 — Elasticity homogenization (voxel FEA)  ✅

**Built**
- `src/voxlat/homogenization/elasticity.py` — periodic Q1 voxel FEA: 8-node trilinear hexahedra
  (2×2×2 Gauss, exact for a cube), periodic fluctuation u~, six unit macroscopic strains → C_eff
  (6×6). Vectorized chunked COO→CSR assembly (no Python loop over elements) or matrix-free
  element-by-element product; PCG with pyamg smoothed aggregation (3×3 blocks, translation
  near-null space) if installed, else Jacobi; the six load cases run in **lock-step**
  (one CSR × 6-column product per iteration, identical iterates, ~1.3–1.8× faster).
  C_eff by **average stress (flux) and by energy**; raises if they differ > 1e-5 (observed 1e-9…2e-8).
  Element-centroid (= element-average) stresses → von Mises **stress localization**.
- `scripts/task3_convergence.py` (~14 min, 2 cores), `scripts/task3_n96_anchor.py` (~16 min, 1 core,
  ~1.4 GB), `scripts/task3_stiffness_vs_density.py` (~16 min, 2 cores). All but the anchor take `--quick`.
- README: "Homogenization (Tasks 2-3)" section.
- Tests: `tests/test_elasticity.py` (47 incl. 4 `slow`; AMG test skips without pyamg).

**Conventions (stated in the module docstring)**
- **Voigt** (reported `C_eff`): order xx, yy, zz, yz, xz, xy; strain vector uses **engineering shear**
  γ = 2ε, so σ = C ε and C44 = μ for an isotropic solid; load case 4 = γ_yz = 1. S = C⁻¹ gives
  E_i = 1/S_ii, G = 1/S_44….
- **Mandel** (`C_mandel` = W C W, W = diag(1,1,1,√2,√2,√2)): orthonormal basis; eigenvalues
  (Kelvin moduli) used for the SPD check and all rotations (`rotate_stiffness`).
- Localization factor K = σ_vm(voxel)/Σ_vm(macro); uniaxial Σ_zz (Σ_vm = 1) and shear Σ_xz
  (Σ_vm = √3). Use σ_vm,local ≈ K_p99 · Σ_vm,macro in Task 9.

**Void: removed, not soft E_min (justification, measured)**
Coolant has no shear stiffness, so E_void = 0 is the exact limit. Removing void elements/nodes gives
2.5× fewer DOFs and ~2.6× less time (G, ρ* = 0.3, n = 24: 16 443 vs 41 472 DOFs) and **no bias**; a soft
void biases C_eff by ≈ 22·E_min/E_s (2.2 % at 1e-3, 0.2 % at 1e-4; test-verified linear). The price is a
singular-but-consistent K (periodic translations, floating islands = rigid modes, corner/edge-hinged
voxels = mechanisms); all null vectors have B w = 0, so loads are orthogonal to them and they carry no
stress. Translations are projected per connected component. Edge-hinged voxels *do* carry the stretch
of the shared edge (Q1 physics) — tested.

**Public API (`voxlat.homogenization`, all in `.elasticity`)**
```python
effective_elasticity(cell, E=None, nu=None, *, void_modulus=0.0, tol=1e-8,
                     preconditioner="auto"|"amg"|"jacobi"|"block_jacobi"|"none",
                     operator="auto"|"assembled"|"matrix_free", block_solve=True,
                     localization=("uniaxial_z","shear_xz"), keep_localization_fields=False,
                     return_fields=False, check=True) -> ElasticityResult
    # cell: bool (True = solid; needs E, nu) or float field of voxel moduli (common nu)
ElasticityResult: .C_eff (Voigt, energy form) .C_eff_flux .agreement .C_mandel .S .eigenvalues .is_spd
    .youngs_moduli (Ex,Ey,Ez) .shear_moduli .E_axial .youngs_modulus(d) .youngs_extremes()
    .cubic_constants (C11,C12,C44) .cubic_deviation .zener_ratio .universal_anisotropy .bulk_modulus
    .localization[name] -> StressLocalization(max, p99, p999, mean, argmax_voxel, factor?)
    .localize(Sigma_voigt)  (needs return_fields)   .as_row()  .iterations .residuals .solve_time
    .n_dofs .n_elements .n_components .unit_stresses (6, Ne, 6) float32 if return_fields
effective_elasticity_tpms(params, n=40, E=None, nu=None, *, cfg=None, offset=None, **kw)   # E, nu from config
extrapolated_elasticity_tpms(params, n=(32, 64), ..., p=1.0) -> ExtrapolatedElasticity   <- PRODUCTION
    # .C_eff = (n2 C(n2) - n1 C(n1))/(n2 - n1) .coarse .fine (localization from fine) .correction
averaged_elasticity_tpms(params, n=40, n_offsets=3, ...) -> AveragedElasticity; grid_offsets(k, seed=2026)
isotropic_stiffness(E, nu), lame_parameters, voigt_to_mandel, mandel_to_voigt, rotate_stiffness(C, R),
von_mises(s), hex8_element(nu) -> (K1, F1, B_centre, D1), voigt_reuss_hill(C) -> {K_V..., E_H, A_U},
youngs_extremes(C), cubic_deviation(C), laminate_stiffness(fractions, E, nu, axis) (exact Backus),
hashin_shtrikman_porous(phi, E, nu) -> {K, G, E} upper bounds; MACRO_STRESS_CASES
```

**Verification (all pass)**
| Test | Result |
|---|---|
| Element: 6 rigid modes, constant-strain patch test, self-equilibrated F1 | exact (1e-14) |
| Homogeneous cells (6³, 5×7×9, 1×4×3; bool and float) | C = C_s to 1e-13, 0 iterations, K_loc ≡ 1 |
| Two-phase laminates along x, y, z (contrast 0.1, 1e-3), 13-layer 3-phase laminate | = exact Backus tensor to 1e-9; Reuss (normal/transverse shear) and Voigt (in-plane shear) components named and checked |
| Void laminates (disconnected plates, 2 components) | plane-stress in-plane terms exact (1e-12), K_loc = 1/φ exactly |
| Floating island / corner-hinged voxel | add exactly nothing (1e-9), hinged voxel stress-free; edge hinge converges |
| Random two-phase cell | Hill bounds C_Reuss ≤ C_eff ≤ C_Voigt (Loewner order), SPD |
| Flux vs energy; symmetric; SPD; Hill average ⟨σ⟩ = Σ | agree 1e-9; yes; yes; 1e-5 (float32 store) |
| Cubic symmetry G, D at ρ* = 0.2/0.35/0.5, n = 32 | C11=C22=C33 and C44s to < 5e-3, C12s < 1e-2, cubic deviation < 5e-3, 90° invariance |
| Blend w = 0.5 | invariant under cyclic x→y→z ([111] 3-fold axis) but cubic deviation 0.65 → **trigonal** |
| HS upper bounds (K, G, E) | respected |
| Invariances | periodic roll, axis permutation ↔ tensor rotation, soft void → removed void linearly |
| Lock-step PCG = scipy cg = matrix-free = block-Jacobi | to 1e-7 |
| E*/E_s = Cρ^m, ρ = 0.2–0.5, n = 32 | G 0.95 ρ^2.19, D 0.70 ρ^1.98 → bending-dominated, m ≈ 2 ✔ |

**Timings** (sandbox core, Jacobi, G ρ* = 0.35, all 6 cases incl. assembly + localization)
| n | DOFs | iterations | total | per load case |
|---|---|---|---|---|
| 24 | 18.6 k | 200–230 | 1.6 s | 0.3 s |
| 32 | 42 k | ~260 | 3.7 s | 0.6 s |
| **40** | **78 k** | **254–325** | **6.6–7.8 s** | **1.1–1.3 s** (target ≤ 60 s ✅, ~50× margin) |
| 48 | 133 k | ~370 | 15 s | 2.5 s |
| 64 | 304 k | ~490 | 45–56 s | 8–9 s |
| 96 | 1.0 M | 530–690 | 215–275 s (assembled, ~1.4 GB) | ~40 s |
Iterations ∝ n (Jacobi). Blends ~2× more iterations. `operator="auto"` switches to matrix-free above
~6e7 non-zeros (~0.7 GB); matrix-free is memory-light but ~3× slower — on a 16 GB laptop pass
`operator="assembled"` for n ≤ 96. AMG not tested (pyamg not installable here).

**Convergence (n = 16–64, 4 grid offsets each, G/D × ρ* = 0.2/0.35/0.5; `results/task3_convergence_summary.csv`)**
Reference = fit f_inf + C/n to offset-averaged n = 32–64. **Independent check at n = 96** (2 offsets,
ρ* = 0.35): the fit predicts the n = 96 values to ≤ 0.6 % for all six quantities → reference ±~0.5 %.
- Unlike k_eff (Task 2, 3–17 % low), the stiffness staircase error is **small**: offset-averaged C11, C12,
  C44, K converge **from below, first order**, −1.8…−4.3 % at n = 24, −1.4…−2.5 % (C12 −4 %) at n = 48.
  Q1 bending stiffening and staircase softening partly cancel.
- **Alignment noise is as large as the bias** for a single grid: std over offsets of E* ≈ 0.4–1.5 % (n = 32),
  0.2–0.8 % (n = 48); worst at low ρ*.
- Max |error| over the 6 cases, default grid (offset 0):

| Estimator | cost (2 parallel) | E* | all of E*, C11, C12, C44, K, Zener |
|---|---|---|---|
| raw n = 40 | 6 s | 5.8 % (low) | 5.8 % |
| raw n = 48 | 15 s | 2.1 % (low) | 6.4 % (C12, D ρ* = 0.2) |
| raw n = 64 | 45 s | 2.8 % | 3.0 % |
| 3 offsets n = 48 | 45 s | 3.0 % | 4.6 % |
| **R(32, 64), p = 1** | **47 s** | **2.6 %** | **2.9 %** |
| R(24, 48) on 3-offset averages | 47 s | 3.0 % | 3.0 % |

**Recommended: `extrapolated_elasticity_tpms(params, (32, 64))` (same pair as Task 2)** — ≤ 3 % on every
quantity incl. the Zener ratio, ~1 min/sample on a laptop core. Cheap alternative: raw n = 48 (E* ≤ 2 % low,
off-diagonal C12 up to 6 % low). Localization: **use p99 at n ≥ 48** — p99 is converged to ±3 % from n = 24
on (G ρ* = 0.35: 14.9 / 15.8 / 15.3 / 15.4 at n = 24/32/48/64); the **max is a voxel-corner singularity
that keeps growing** (G ρ* = 0.2: 35 → 48 from n = 16 to 64) — never use the max for design.

**Results (R(32, 64), AlSi10Mg; `results/task3_stiffness_vs_density.csv`, fits `results/task3_powerlaw_fits.csv`)**
| ρ* | 0.20 | 0.30 | 0.35 | 0.40 | 0.50 |
|---|---|---|---|---|---|
| Gyroid E*/E_s | 0.0300 | 0.0666 | 0.0998 | 0.1265 | 0.2074 |
| Diamond E*/E_s | 0.0307 | 0.0642 | 0.0864 | 0.1145 | 0.1789 |
| Blend w = 0.5 E*/E_s (x,y,z) | 0.0068 | 0.0169 | 0.0240 | 0.0756 | 0.1777 |
| Gyroid Zener A | 1.90 | 1.71 | 1.48 | 1.45 | 1.28 |
| Diamond Zener A | 2.29 | 2.11 | 2.04 | 1.92 | 1.72 |
| Gyroid K_p99 (uniaxial z / shear xz) | 36 / 17 | 19 / 9.1 | 15 / 7.0 | 12 / 5.8 | 8.3 / 4.1 |
| Diamond K_p99 | 23 / 17 | 12 / 9.3 | 9.7 / 7.4 | 8.2 / 6.1 | 6.2 / 4.4 |
| Blend K_p99 | 95 / 62 | 52 / 32 | 42 / 26 | 19 / 12 | 8.5 / 5.7 |

- **Power laws, ρ* = 0.2–0.5**: gyroid E*/E_s = **0.90 ρ*^2.13**, diamond **0.67 ρ*^1.93** (Hill-average E:
  0.86 ρ^1.87, 0.80 ρ^1.76). Exponent ≈ 2 = bending-dominated, matching Khaderi, Deshpande & Fleck 2014
  (gyroid lattice: E, G ∝ ρ², bulk ∝ ρ) and the bending classification of skeletal TPMS in Al-Ketan et al.
  2018. With m fixed at 2: C = 0.79 (G), 0.72 (D) vs **Maskery et al. 2018** experiments (PA2200, ρ* = 0.3,
  m assumed 2): C = 0.69 (G), 0.68 (D) → our ideal-geometry FE is 11 % (G) / 9 % (D) stiffer at ρ* = 0.3,
  plausible given SLS porosity/surface roughness. Bulk modulus scales more weakly (m ≈ 1.5–1.6, stretching
  under hydrostatic load, as Khaderi et al. predict).
- **G and D are not isotropic**: Zener 1.3–1.9 (G) and 1.7–2.3 (D) in the design range, decreasing with ρ*;
  E_max/E_min = 1.2–1.8 (G), 1.6–2.0 (D), stiffest along ⟨111⟩. Task 9 must not treat C_eff as isotropic.
- G is stiffer than D at ρ* ≥ 0.35 but has **~1.5× higher uniaxial stress concentration** (K_p99 15 vs 9.7 at
  0.35); in shear they are equal. Diamond is the better structural choice per unit E.

**Open issues / notes for later tasks**
1. **Blends are structurally bad below ρ* ≈ 0.375 (RQ2-relevant).** w = 0.5 is 3.4–4.5× softer than G/D
   (E ∝ ρ^3.7), strongly anisotropic (A^U ≈ 3.5 vs 0.2–0.9) and has 2.5–4× higher stress concentration
   (K_p99 up to 95 at ρ* = 0.2); at ρ* = 0.35→0.40 E jumps 3× and A^U drops 7× — the same pinch-neck topology
   change found in Tasks 1 and 2. Task 11 should either forbid mid-w at low ρ*, use the strict min-wall
   constraint, or carry the structural margin with K_p99 from these closures (which will penalize it).
2. Blend tensors are trigonal about [111] (cubic deviation 0.65): Task 6 must predict the full 6×6 (or
   its trigonal invariants in the [111] frame), and Task 9 must rotate C_eff into (r, s, z).
3. Localization factors are element-average (centroid) values on a staircase surface; p99 is
   resolution-stable, but absolute peak stress at a smooth printed surface needs a fatigue-notch argument
   in the paper (or a smoothed-boundary check in Task 12).
4. Anisotropic cells work (n × n × round(n a_z)); at ρ* = 0.35, n = 16, a_z = 1.5 gives E_z/E_x = 1.65 (G) and
   3.1 (D) — stretch changes stiffness far more than it changes k_eff (Task 2: 1.5–1.6).
   The Richardson pair refers to the in-plane n.
5. `extrapolated_elasticity_tpms` at (32, 64) ≈ 45–60 s per sample (G/D), ~100 s for blends → the Task 5
   dataset (N ≈ 270) needs ~3–4 h for elasticity alone on 2 cores; AMG on the laptop should cut that.
6. Citations to verify: Khaderi, Deshpande & Fleck 2014, Int. J. Solids Struct. 51(23–24):3866–3877;
   Maskery et al. 2018, Polymer 152:62–71 (doi 10.1016/j.polymer.2017.11.049; values from Table 3);
   Al-Ketan, Rowshan & Abu Al-Rub 2018, Addit. Manuf. 19:167–183; Lu et al. 2019, J. Mech. Behav. Biomed.
   Mater. 99:56–65 (cubic symmetry of G/D; they report G as the most anisotropic — our D is more
   anisotropic, check their geometry definition); Backus 1962, J. Geophys. Res. 67:4427 (laminates);
   Hashin & Shtrikman 1963, J. Mech. Phys. Solids 11:127; Ranganathan & Ostoja-Starzewski 2008, Phys. Rev.
   Lett. 101:055504 (A^U); Gibson & Ashby 1997, *Cellular Solids*, 2nd ed.
7. Workspace: PyPI blocked (no pyamg); numpy 2.5.3, scipy 1.18.1, Python 3.13; pytest from a uv tool env.

**Figures** (`results/figures/`): `task3_youngs_vs_density.png` (log-log E*/E_s, fits, HS bound, slope-2
guide, Maskery points, blend directional range), `task3_anisotropy_vs_density.png` (Zener; universal
anisotropy A^U incl. blend), `task3_localization_vs_density.png` (p99/max/mean, uniaxial z and shear xz),
`task3_convergence.png`. Data: `results/task3_*.csv`, `results/task3_pytest_log.txt`.

**Tests**: 216 passed + 2 skipped (pyamg) in 4.5 min (Tasks 0–3, incl. slow); Task 3 alone 46 + 1 skipped.

## Task 4 — Stokes permeability  ✅

**Built**
- `src/voxlat/homogenization/stokes.py` — periodic creeping-flow (Stokes) permeability K (3×3) of any
  bool voxel cell: staggered **MAC finite volumes** (p at fluid-cell centres, u_d on voxel faces), no slip /
  no penetration on every solid–fluid voxel face, unit body force e_j, **MINRES** on the symmetric
  saddle-point system with the block-diagonal preconditioner diag(A_hat, I) (A_hat = Jacobi, or one pyamg
  SA V-cycle if installed). Outputs K, staggered velocity and pressure fields, porosity, mean interstitial
  velocity, mean speed, hydraulic tortuosity. Flux and energy forms of K are both computed and checked.
  Verification geometries + analytic references live in the same module.
- `src/voxlat/homogenization/convergence.py`: the free-order fit (`fit_convergence(..., p=None)`) now fits
  in normalized units (bug fix, see open issue 6).
- Scripts: `scripts/task4_verification.py` (~2 min, 2 cores), `scripts/task4_convergence.py` (~11 min, 2 cores,
  incl. four n = 96 anchors, ~1.5 GB each), `scripts/task4_permeability_vs_porosity.py` (~9 min, 2 cores).
  `scripts/task4_benchmark_preconditioner.py` (Jacobi vs AMG timing and K agreement, ~3-5 min). All take `--quick`; the first two also `--replot`.
- `voxlat.homogenization` exports the Stokes API; README "Homogenization (Tasks 2-4)" updated.
- Tests: `tests/test_stokes.py` (50 incl. 6 `slow`; the AMG test skips without pyamg),
  + 3 scale-invariance regression tests in `tests/test_conduction.py`.

**Method choice (justification, also in the module docstring)**
- MAC: the wall sits exactly on the same voxel faces as in Tasks 2–3; wall-normal velocity is zero *at* the
  wall (exact no-penetration); tangential no-slip by the mirror ghost u_g = −u where a flat wall lies h/2 away
  (standard 2nd-order MAC, Manwart et al. 2002), u = 0 at distance h at staircase steps. No tuning parameter.
- MINRES + diag(Jacobi, pressure mass) is the classic block preconditioner for Stokes (Silvester & Wathen 1994);
  iterations grow ∝ n with Jacobi but the cost stays far inside the 2-min target (table below).
- Rejected: FFT-Brinkman (penalty boundary layer O(√ε), Gibbs ringing at the solid indicator, conditioning
  ∝ 1/ε); Uzawa / augmented Lagrangian (nested inner velocity solves); sparse LU (3-D fill: GBs at n = 48);
  lattice Boltzmann (O(n²) time steps to steady state in numpy).
- **Reported K = symmetrized flux form** (K_ij = ⟨u_i^(j)⟩): its error is quadratic in the solver residual
  (compliance functional), measured 3e-8 relative at tol 1e-8 (gyroid n = 48); the energy form
  u^(i)ᵀA u^(j)/N (Gram matrix, PSD by construction) is linear in the residual (4e-7 G/D … 1e-4 blends) and
  is used as the check: `agreement_tol = 1e-3` (a sloppy tol = 1e-4 solve disagrees by 1.4e-2 and raises).
- MINRES `tol` is scipy's backward-error test, not ‖r‖/‖b‖: tol 1e-8 → true relative residual ~1e-4,
  K converged to ~1e-7. Default tol 1e-8.

**Public API (`voxlat.homogenization.stokes`; main names also in `voxlat.homogenization`)**
```python
permeability(cell, *, voxel_size=1.0, symmetry="none"|"cubic"|"tetragonal_z", directions=None, tol=1e-8,
             preconditioner="auto"|"amg"|"jacobi", maxiter=None, return_fields=False, check=True,
             agreement_tol=1e-3) -> PermeabilityResult
    # cell: bool (True = solid), periodic; K in units of voxel_size^2 (1/n -> L^2; L/n in m -> m^2)
PermeabilityResult: .K (3x3) .K_flux .K_energy .agreement .porosity .interstitial_velocity (3x3, col j = <u>_f)
    .mean_speed{j} .tortuosity{j} (NaN if no net flow) .eigenvalues .mean .is_spd .anisotropy .as_row()
    .iterations .residuals (true ||r||/||b||) .divergence .n_dofs .n_pressure_components .wall_time
    .velocity (n_dirs,3,nx,ny,nz) float32 staggered (+d face) .pressure (n_dirs,nx,ny,nz)  [return_fields]
    .cell_velocity(k) -> (3,nx,ny,nz) cell-centred
permeability_tpms(params, n=48, *, offset=None, symmetry="auto", **kw)        # K in L^2; G/D: 1 solve, blends: 3
tpms_symmetry(params) -> "cubic" (w in {0,1}, a = 1) | "tetragonal_z" (a_x = a_y != a_z) | "none" (blends)
extrapolated_permeability_tpms(params, n=(32, 64), *, p=1.0, offset=None) -> ExtrapolatedPermeability  <- PRODUCTION
    # .K .coarse .fine .correction .porosity .tortuosity .mean .eigenvalues
averaged_permeability_tpms(params, n=48, n_offsets=3, seed=2026) -> AveragedPermeability(.K, .K_std, ...)
assemble_stokes_system(solid) -> StokesSystem(A, D, saddle, active, face_id, offsets, pressure_id, ...)
solve_stokes(system, force, *, tol, maxiter, preconditioner) -> StokesSolution(u, p, iterations, residual, divergence)
kozeny_constant(K, porosity, a_sf) = phi^3 / (K a_sf^2)
# references / verification geometries
slit_permeability(H) = H^2/12; mac_slit_permeability(H) = (H^2+2)/12; rectangular_duct_mean_velocity(a, b)
ZICK_HOMSY_SC (table), zick_homsy_sc_drag(c) (PCHIP, log-log), sangani_acrivos_sc_drag(c), sc_sphere_permeability(c, K*)
sphere_array_cell(n, c, match_volume=True); inclined_slit_cell(N, porosity, slope, nz); inclined_slit_reference(...)
```
`assemble_stokes_system`/`permeability` take any periodic bool cell, so Task 7's strips (lattice between
solid wall layers) need no new solver: the strip walls are grid-aligned, i.e. 2nd-order exact.

**Verification (all pass; `results/task4_verification_*.csv`, figure `task4_verification.png`)**
| Case | Result |
|---|---|
| (a) Plane Poiseuille, H = 1…64 fluid voxels, walls ⟂ x, y, z, thick walls, 1-voxel axes | K = (H²+2)/12·φ **exactly** (1e-15); profile u_j = (y_j(H−y_j)+¼)/2 exact; K across = 0; T = 1 |
| (a) vs continuum H²/12 | error **+2/H²** (H = 8: 3.1 %, 16: 0.78 %, 32: 0.20 %), observed order 2.000 |
| (b) Square duct vs series (0.0351443 a²G/μ) | +5.9 / 1.5 / 0.38 / 0.094 % at 8 / 16 / 32 / 64 voxels; order 2.0; R(16,32) p=2 → < 0.03 % |
| (b) 2:1 duct | +3.9 / 0.98 / 0.25 / 0.06 %, order 2.0 |
| (b) Inclined slit 45° (φ = 0.5) | exactly −4/N² along the slit, +8/N² along z → 2nd order (symmetric staircase) |
| (b) Inclined slit 1:2, 1:3 (generic wall) | **1st order**: K error ≈ −0.20 h/H (1:2), −0.26 h/H (1:3) (H = fluid-layer thickness) → effective wall shift ≈ 0.05 h into the fluid per wall; −1.1 % at H = 18 voxels |
| (c) SC sphere arrays vs Zick & Homsy (1982), c = 0.027, 0.064, 0.125, 0.216, 0.343, 0.45 | at the realized voxel fraction: **−0.3 … −0.9 % for every n = 32–64**, n = 24 within ±1.1 %; full tensor isotropic to 1e-6 |
| (c) same, vs the target c | ±1–4 % at n ≤ 32 — the voxel shell structure misses c by up to 7 % and d ln K/d ln c ≈ −1…−1.3, so always compare at the realized c |
| Sangani & Acrivos (1982) series vs Zick & Homsy | ≤ 0.8 % for c ≤ 0.216 (reference data cross-check) |
| Sealed cavities | add porosity, carry no flow (u ≤ 1e-9 max), K unchanged; fully closed cell → K = 0, T = NaN |
| Symmetry | G, D: full 3-direction K = k I to 1.4e-3 at n = 24 (voxel alignment) → `symmetry="cubic"` valid; blend w = 0.5: K = aI + b(J−I), non-degenerate axis [111] (trigonal); stretched a_z = 1.5: tetragonal, K_zz > 1.1 K_xx |
| Invariances | periodic roll and axis permutation ↔ tensor rotation to 1e-6 |
| Fields | staggered mean = K; discrete div u ≤ 1e-3 max|u| (true residual ~1e-4); u = 0 on all faces touching solid |
| (d) TPMS convergence (slow test) | offset-averaged G ρ* = 0.35, n = 16…64: every n ≥ 32 within 1.5 % of the fit, R(32,64) within 0.5 % |

**Convergence (n = 16–64, 4 grid offsets each; G/D × ρ* = 0.2/0.35/0.5 + blend w = 0.5, ρ* = 0.35;
`results/task4_convergence_summary.csv`)**
Reference = fit K_inf + C/n to offset-averaged n = 32–64. **Independent n = 96 check** (2 offsets, ρ* = 0.35):
the fit predicts n = 96 to **0.18 % (G) and 0.46 % (D)** → reference good to ~±0.5 %.
- Unlike k_eff (Task 2: 3–17 % low), the permeability staircase error is **small**: offset-averaged K is low
  by 0.0–0.8 % at n = 32 and 0.1–0.6 % at n = 48 (diamond ρ* = 0.5, the narrowest pores: −2.4 % / −1.7 %).
  Flow is not blocked by the staircase, the effective wall moves only ~0.05 h (inclined-slit result).
- Systematic error and alignment noise (0.1–0.7 % std at n = 32–48) are the same size, so the free order is
  not identifiable (normalized free fits hit the 0.25 / 4 bounds in 6 of 7 cases); p = 1 is used, as in Tasks 2–3.
- Tortuosity converges faster, from below: offset-averaged within 0.3 % from n = 32 on, 0.15 % at n = 48–64.
- Max |error| over the 7 cases on the default grid (offset 0):

| Estimator | cost per direction (G/D, 1 core) | max \|error\| | all but D ρ* = 0.5 |
|---|---|---|---|
| raw n = 32 | ~1 s | 3.2 % | 0.9 % |
| raw n = 48 | 3–4 s | 1.6 % | 0.9 % |
| raw n = 64 | 13–18 s | 1.1 % | 0.4 % |
| 3 offsets n = 48 | 8–13 s | 1.7 % | 0.6 % |
| R(24, 48) | 3–5 s | 1.2 % | 1.2 % |
| R(48, 64) | 16–22 s | 1.5 % | 1.5 % |
| **R(32, 64)** | **14–19 s** | **1.0 %** | **0.35 %** |

**Recommended: `extrapolated_permeability_tpms(params, (32, 64))`** — ≤ 1 % everywhere, same grid pair as
Tasks 2–3 (Task 5 can voxelize once). Cheap alternative: raw n = 48 (≤ 1.6 % low, mostly ≤ 0.9 %).

**Timings** (1 sandbox core, nothing else running, Jacobi, ρ* = 0.3, ONE body-force direction incl. assembly)
| n | DOFs | G | D | blend | iterations G / D / blend | peak RAM |
|---|---|---|---|---|---|---|
| 32 | 89 k | 1.0 s | 0.7 s | 1.4 s | 378 / 283 / 573 | 0.13 GB |
| **48** | **304 k** | **4.3 s** | **2.7 s** | **6.8 s** | 554 / 356 / 902 | 0.25 GB (**target ≤ 2 min ✅, 17–40× margin**) |
| 64 | 725 k | 18 s | 13 s | 26 s | 784 / 519 / 1206 | 0.5 GB |
| 96 | 2.46 M | 114 s | 72 s | – | 1252 / 800 | 1.5 GB |
Iterations ∝ n (Jacobi), cost ∝ n^4. Two parallel workers slow each ~1.3–1.5× (memory bandwidth). Production
R(32, 64): G ~19 s, D ~13 s per sample (one solve each, cubic symmetry), blends ~85 s (three directions).
pyamg should cut the iteration count (untested here).

**Results (R(32, 64); `results/task4_permeability_vs_porosity.csv`)**
| ρ* (φ) | 0.20 (0.80) | 0.30 (0.70) | 0.35 (0.65) | 0.40 (0.60) | 0.50 (0.50) |
|---|---|---|---|---|---|
| Gyroid K/L² | 1.030e-2 | 6.214e-3 | 4.887e-3 | 3.829e-3 | 2.224e-3 |
| Diamond K/L² | 6.651e-3 | 4.010e-3 | 3.103e-3 | 2.390e-3 | 1.410e-3 |
| Blend w = 0.5, mean (min–max) ×1e-3 | 10.49 (9.83–11.81) | 5.57 (5.26–6.20) | 4.04 (3.89–4.33) | 2.95 (2.89–3.06) | 1.39 (1.24–1.71) |
| Kozeny c_K G / D / blend | 7.76 / 7.95 / 7.72 | 6.72 / 6.81 / 6.72 | 6.37 / 6.53 / 6.39 | 6.12 / 6.35 / 6.14 | 5.87 / 6.01 / 6.66 |
| Tortuosity G / D / blend | 1.12 / 1.14 / 1.09 | 1.16 / 1.19 / 1.13 | 1.18 / 1.21 / 1.16 | 1.20 / 1.24 / 1.20 | 1.24 / 1.29 / 1.31 |

- Power laws over ρ* = 0.2–0.5: **G K/L² ≈ 0.0202 φ^3.22, D ≈ 0.0132 φ^3.29**. Diamond is ~35–37 % less
  permeable than gyroid at equal porosity (smaller pores and throats, Task 1).
- **Kozeny–Carman collapse**: c_K = φ³/(K a_sf²) is nearly topology-independent — G, D and the blend agree
  within ~4 % at equal φ for φ ≥ 0.5 (G vs D 7 % at φ = 0.45; blend +11 % at ρ* = 0.5), c_K falling from ~9
  (φ = 0.85) to ~6 (φ = 0.5), above Carman's 5 for packed beds. So K ≈ φ³/(c_K(φ) a_sf²) with the Task 1 a_sf is a good physics prior for
  Task 6 (or a check on its surrogates).
- Physical scale: K = (K/L²)·L²; e.g. gyroid ρ* = 0.35, L = 4 mm → K = 7.8e-8 m².

**Open issues / notes for later tasks**
1. **Blend K is trigonal and non-monotone in anisotropy (RQ2-relevant).** w = 0.5: (K_max−K_min)/K_mean ≈ 0.19
   at ρ* ≤ 0.25, a minimum 0.06 at ρ* = 0.4, then 0.34 / 0.55 at ρ* = 0.5 / 0.55 — the same pinch topology change
   seen in Tasks 1–3, here on the fluid side at high ρ*. Task 6 must predict the full tensor for 0 < w < 1 (axis
   [111]); Task 9 must rotate it into (r, s, z). Unlike stiffness, permeability does **not** collapse for blends at
   low ρ* (blend ≈ gyroid there), so blends are a flow-vs-structure trade-off, not simply worse.
2. Stretched cells: tetragonal K (`tpms_symmetry` → two solves). K_zz/K_xx for a_z ≠ 1 is not tabulated yet — Task 5
   samples it.
3. Creeping-flow only: K is the Darcy (Re → 0) permeability. Inertial (Forchheimer) corrections come from
   literature in Task 8; state this in the paper's limitations.
4. For Task 7: strip walls aligned with the grid are 2nd-order exact (Poiseuille / duct tests), so finite-gap
   errors measured there will be lattice physics, not discretization; keep the lattice at n ≥ 32 per cell.
5. AMG path (`preconditioner="amg"`, a symmetric-Gauss-Seidel SA V-cycle on the velocity block, SPD as MINRES
   requires; `"auto"` picks it whenever pyamg is installed) — **checked on the Windows laptop** (Python 3.13.7,
   pyamg installed): K matches Jacobi to 2e-6 of mean K. AMG aggregation is not symmetric under the cube's
   rotations and MINRES stops at a looser true residual for the same tol, so symmetric cells show off-diagonals
   ~1e-5 of K instead of round-off (Jacobi keeps them at 1e-8). Both are ~1000x below the discretization error.
   First laptop run failed 5 slow tests on tolerances tuned to Jacobi; fixed in the Task 4 follow-up commit
   (solver unchanged; the exact-symmetry check now runs with Jacobi explicitly). Speed on the laptop:
   run `python scripts/task4_benchmark_preconditioner.py` and record AMG vs Jacobi here before Task 5;
   if AMG is not faster, Task 5 should pass `preconditioner="jacobi"`.
6. **Bug fix in a shared helper:** `fit_convergence(..., p=None)` stalled at p0 = 1 when |f| ≪ 1 (curve_fit
   tolerances; K/L² ~ 1e-3). Now normalized internally; regression test added. This affected only the **`p_fit`
   diagnostic column of `results/task3_convergence_summary.csv`** (Task 3 references used fixed p = 1 and are
   unchanged); that column was recomputed from `results/task3_convergence.csv` (other columns identical).
   Corrected Task 3 free orders scatter 0.25–2.0, median 1.16 → noise-limited, consistent with Task 3's "first order".
7. Citations to verify: Zick & Homsy 1982, J. Fluid Mech. 115:13–26 (Table 2, SC: c = 0.027…0.5236 →
   K* = 2.008…42.1; drag F = |∇P|·V_cell, superficial U — convention confirmed against Basilisk's `spheres.c`
   test and the LBM drag-correlation paper arXiv:1401.2025, Sec. 2); Sangani & Acrivos 1982, Int. J. Multiphase Flow
   8:343 (SC series coefficients 1.7601, 1.5593, 3.9799, 3.0734); Hasimoto 1959, J. Fluid Mech. 5:317;
   Manwart et al. 2002, Phys. Rev. E 66:016702 (MAC on voxel images, mirror ghost); Silvester & Wathen 1994,
   SIAM J. Numer. Anal. 31:1352 (block preconditioner); Duda, Koza & Matyka 2011, Phys. Rev. E 84:036319
   (hydraulic tortuosity); Carman 1937, Trans. Inst. Chem. Eng. 15:150 (c_K = 5); duct series: Shah & London 1978
   / White, *Viscous Fluid Flow*.
8. Workspace: PyPI blocked (no pyamg); numpy 2.5.3, scipy 1.18.1, Python 3.13; pytest from a uv tool env.

**Figures** (`results/figures/`): `task4_verification.png` ((a) Poiseuille profile + 2/H² error, (b) ducts and
inclined slits vs h/D with slope-1/2 guides, (c) sphere arrays vs Zick & Homsy and the SA series, (d) sphere error
vs n), `task4_convergence.png` (K(n)/K_ref vs 1/n with n = 96 anchors; estimator errors; tortuosity),
`task4_permeability_vs_porosity.png` (K/L² vs φ with blend principal-value band; Kozeny constant; tortuosity).
Data: `results/task4_*.csv`, `results/task4_pytest_log.txt`.

**Tests**: 268 passed + 3 skipped (pyamg) in 7.9 min (Tasks 0–4, incl. slow; `results/task4_pytest_log.txt`).
Task 4 alone: 49 passed + 1 skipped; `pytest -m "not slow" tests/test_stokes.py` runs 44 tests in ~30 s.

## Task 5 — Closure dataset
_not started_

## Task 6 — Closure surrogates
_not started_

## Task 7 — Finite-gap study (RQ1)
_not started_

## Task 8 — Literature closures
_not started_

## Task 9 — Homogenized jacket device model
_not started_

## Task 10 — Baselines B1–B3
_not started_

## Task 11 — Multi-objective optimization
_not started_

## Task 12 — PicoGK geometry export
_not started_

## Task 13 — Reproducibility pass + figure set
_not started_

## Task 14 — Manuscript draft
_not started_
