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
| 5 | Closure dataset | **done** (2026-10-08, run on the laptop; record reconstructed in Task 6) |
| 6 | Closure surrogates | **done** (2026-10-09) |
| 7 | Finite-gap study (RQ1) | **done** (2026-10-09, n = 32 in the sandbox; n = 48 overnight run optional) |
| 8 | Literature closures (Nu, Forchheimer) | **done** (2026-10-10) |
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
1. ~~Grading-gradient limit is a placeholder~~ → confirmed at 0.05 by the Task 7 graded strips (see Task 7).
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

## Task 5 — Closure dataset  ✅

> The Task 5 session's own STATUS notes were not in the repo (this section said *not started*
> although commit 02b1054 holds the code and `data/closures.parquet`). The record below was
> reconstructed in Task 6 from the table, `voxlat.closures.dataset`, the README and the demo log.

**Built** — `src/voxlat/closures/dataset.py` (design, per-sample computation, resumable parts,
consolidation, quick-look plot), `scripts/build_dataset.py` (`--estimate-only`, `--n-jobs`,
`--plot-only`, `--quick`), `tests/test_dataset.py`, README section "Closure dataset (Task 5)".
API: `from voxlat.closures import build_design, compute_sample, run_samples, consolidate, read_table`.

**Dataset (`data/closures.parquet`, 290 rows x 170 columns, all `status == "ok"`)**
| Item | Value |
|---|---|
| Design | 256 scrambled Sobol (seed 20261008) + 8 corners + 12 edge midpoints + 14 pure-line points (w in {0, 1}, a_z = 1, rho* = 0.20:0.05:0.50) |
| Box | w in [0, 1], rho* in [0.20, 0.50], a_z on 27 exact levels l/32 = 0.6875 ... 1.5 |
| Estimators | k_eff, C_eff, K: Richardson R(32, 64); geometry metrics at n = 64; localization from the n = 64 grid |
| Settings hash | `64f53a501dd4` (k_s 130, k_f 0.4, E_s 70 GPa, nu 0.33, tol 1e-8) |
| Run | Windows 11, Python 3.13.7, numpy 2.5.3, git 677c30ae3ab7(+dirty); 2026-10-08 16:28-23:51 UTC (7.4 h wall), 27.8 CPU-h; AMG (3 timing rows Jacobi) |
| Cost per sample | median 312 s (pure 250 s, blends 322 s), max 628 s; peak RSS <= 2.0 GB |
| Topology | every cell: one solid and one fluid component, both percolating |
| Regression check | pure-line rows reproduce the Task 2-4 tables, e.g. gyroid rho* = 0.35: k/k_s 0.2079, K/L^2 4.887e-3, E*/E_s 0.0998, a_sf L 2.970 |

**Found in Task 6 (data quality)** — stretched pure cells carry *voxel symmetry-breaking*: for
diamond at a_z = 22/32 the 4_1 screw axis (translation c/4 = 5.5 voxels at n = 32) is off-grid, so
k_xy reaches 3 % of k_xx and C16 2 % of C11 (corner-4) although the true symmetry is tetragonal.
K is unaffected (solved with imposed symmetry). Task 6 removes these artifacts by symmetry
projection (max change of the latent logs: K 3e-4, k_eff 0.028, C 0.029, localization 0.005).

## Task 6 — Closure surrogates  ✅

**Built**
- `src/voxlat/surrogates/` (all numpy/scipy/scikit-learn; **no torch needed**):
  - `tensors.py` — Log-Euclidean maps (batched `sym_logm` / `sym_expm`), Daleckii-Krein derivative of
    exp, Voigt <-> Mandel, Mandel rotation (checked against `elasticity.rotate_stiffness`).
  - `symmetry.py` — symmetry classes of theta and their Reynolds projectors (2nd-rank, stiffness,
    load cases).
  - `targets.py` — closure table <-> 40 latent targets <-> derived quantities with delta-method std.
  - `gp.py` — ARD Matérn-5/2 GP per target (sklearn hyper-parameter fit, numpy prediction).
  - `nn.py` — minimal numpy NN layers (Dense, periodic Conv3d, pooling, space-to-depth, AdamW,
    Gaussian NLL), every backward pass gradient-checked.
  - `ensemble.py` — 5-member heteroscedastic MLP deep ensemble.
  - `cnn.py` — 3D CNN ensemble on 32³ voxels for K* and E*.
  - `model.py` — `ClosureModel` (GP or ensemble; the API for Tasks 9/11), `CNNClosureModel`, save/load.
  - `evaluation.py` — 5-fold CV, extrapolation splits, metrics, recalibration, parity/calibration plots.
- Scripts: `scripts/task6_train_surrogates.py` (production models -> `models/`),
  `scripts/task6_evaluate.py` (CV + splits -> tables, figures), `scripts/task6_cnn_experiment.py` (6c).
- `models/closure_gp.npz` (100 kB), `models/closure_ensemble.npz` (0.5 MB), `models/closure_cnn.npz`
  — plain `.npz` + JSON metadata (no pickle; independent of the scikit-learn version; the GP
  Cholesky factors are recomputed on load).
- Tests: `tests/test_surrogates.py`. README section "Closure surrogates (Task 6)".

**Decisions**
1. **Why numpy instead of torch.** PyPI is blocked in the build sandbox (no torch), and the models are
   tiny: the CNN has ~45 k parameters and trains in ~4 min per member on one core; numpy keeps it
   runnable everywhere and makes the weights plain arrays. The `ml` extra (torch) is still needed later
   by BoTorch (Task 11).
2. **Targets (40 latent, all smooth in theta):** matrix log of K/L² (6) and of k_eff/k_s (6), matrix log of
   the Mandel stiffness C/E_s (21), log a_sf L (1), log p99 localization of the six unit stresses (6).
   The eigenvalues of log(K/L²) **are** the log principal values of K, and its eigenvectors the
   principal axes ([111] for blends, Tasks 2-4) — so the device model gets the full tensor, not just
   three numbers. exp of any symmetric prediction is SPD: positivity and tensor symmetry hold by
   construction. The **max** localization is not modelled (voxel-corner singularity, Task 3).
3. **Material symmetry is imposed exactly** (projection of the latent logs; `symmetry.py`):

   | class | where | rows | independent K / C / load cases |
   |---|---|---|---|
   | cubic (432) | w in {0,1}, a_z = 1 | 14 | 1 / 3 / 2 |
   | tetragonal (422 about z) | w in {0,1}, a_z != 1 | 16 | 2 / 6 / 4 |
   | trigonal (3-fold about [111]) | 0 < w < 1, a_z = 1 | 10 | 2 / 7 / 2 |
   | triclinic | 0 < w < 1, a_z != 1 | 250 | 6 / 21 / 6 |

   Verified on the Task 5 data (trigonal C shows exactly the 7-constant cyclic pattern) and against
   the Task 2 solver in a test. Stretched blends have *no* symmetry: |k_xx - k_yy| reaches 9 % of k.
4. **L is analytic:** K = (K/L²) L², a_sf = (a_sf L)/L; k_eff, C, localization are scale-free.
5. **Inputs:** u = (w, log rho*, log a_z) scaled to [0, 1]³ (power laws become linear).
6. **GP:** least-squares linear trend + ARD Matérn-5/2 + white noise per target, 2 optimizer restarts.
   Fitted w length scales are short (~0.09) for k_eff and C: the blend pinch-neck transition (Tasks 1-4)
   sets a single stationary length scale.
7. **Deep ensemble:** 5 x MLP 3-64-64-64-80 (SiLU), MSE warm-up (30 %) then Gaussian NLL, AdamW, cosine
   schedule, 3000 full-batch epochs; uniform Gaussian mixture.
8. **Uncertainty:** latent std -> derived std by the exact first-order delta method (checked against
   Monte Carlo within ~2 %; sorted eigenvalues near a crossing are conservative). Raw std's are too
   small (raw CV 95 % coverage 74-92 %; only the ensemble's a_sf reaches 98 %), so the production models carry a **CV recalibration**: per latent
   target, multiplier = 95th percentile of the out-of-fold |z| / 1.96 (median by group — GP: K 1.02,
   k_eff 1.64, C 1.68, a_sf 1.26, loc 1.68; ensemble: K 1.09, k_eff 1.74, C 1.62, a_sf 1.00, loc 1.58).
   In the evaluation it is applied **cross-fitted** (fold f scaled with the other folds' multipliers).
   Errors are heavy-tailed (pinch-neck blends), so one multiplier cannot fix the whole calibration
   curve: intervals are conservative for most designs and still too narrow in the neck region.

**Public API (`voxlat.surrogates`)**
```python
cm = ClosureModel.load("ensemble")             # or "gp"; models/closure_<backend>.npz
out = cm.predict(theta, L)                     # theta: (3,) | (N,3) | dict | TPMSParams; L [m] scalar or (N,)
#   -> {name: (mean, std)} SI:  K (N,3,3) m^2, K_principal (N,3), K_star, k_eff (N,3,3) W/mK,
#      C_eff (N,6,6) Pa (Voigt, engineering shear), E (N,3) Pa, E_star, a_sf 1/m,
#      loc_<case>_p99 (6 cases), porosity (exact)
cm.predict(theta, L, return_std=False)         # means only (~0.12 s for 512 points)
cm.predict(theta, L, check_bounds="raise")     # default "warn" outside the training box
cm.predict_dimensionless(theta)                # K/L^2, k/k_s, C/E_s, ... (targets.QUANTITY_INFO names)
cm.predict_latent(theta) -> (mu, sd, classes); cm.symmetry_class(theta)
ClosureModel.fit(df, "gp"|"ensemble", recalibrate=True, **backend_kw); cm.save(path)
cnn = CNNClosureModel.load(); cnn.predict(theta, L) -> {"K_star", "E_star"}; cnn.predict_voxels(vox32, a_z, L)
load_training_table(path=None)                 # data/closures.parquet (csv fallback), status == ok
symmetry_class(w, a_z); rotation_group(name); projector_sym3/sym6/cases(name); independent_components(name)
latent_from_table(df) -> (Y, classes); derived_quantities(mu, sd, classes); tensors_from_latent(mu, classes)
GaussianProcessSurrogate, MLPEnsemble, CNNEnsemble, cnn_voxels(theta)
evaluation: kfold_indices, extrapolation_split, sparse_blend_split, cross_validate, predict_split,
            latent_cv, std_scale_from_oof, summarize, summarize_groups, plot_parity, plot_calibration
```

**Accuracy** (`results/task6_metrics_groups.csv`, per quantity `results/task6_metrics.csv`; every
prediction in `results/task6_predictions.csv.gz`). R² on log values for positive quantities; err =
MAPE, or NMAEᴺ = MAE / mean diagonal for signed components (off-diagonals, whose R² is not meaningful
because most are ~0); cov95 = coverage of the recalibrated 95 % interval (raw in brackets).

*5-fold CV (290 rows):*

| quantity | GP R² | GP err | GP cov95 | ens R² | ens err | ens cov95 |
|---|---|---|---|---|---|---|
| K principal values | 0.999 | 0.9 % | 93 % (92) | 0.999 | 0.9 % | 91 % (89) |
| K* = tr K/3 | 0.999 | 0.7 % | 91 % (89) | 0.999 | 0.7 % | 86 % (85) |
| k_eff diagonal | 0.995 | 1.9 % | 96 % (91) | 0.995 | 1.5 % | 95 % (89) |
| k_eff off-diagonal | 0.936 | 0.66 %ᴺ | 97 % (90) | 0.952 | 0.46 %ᴺ | 96 % (86) |
| C normal (C11, C22, C33) | 0.985 | 6.0 % | 93 % (89) | 0.988 | 4.3 % | 94 % (86) |
| C coupling (C12, C13, C23) | 0.980 | 2.18 %ᴺ | 92 % (86) | 0.992 | 1.01 %ᴺ | 93 % (86) |
| C shear (C44, C55, C66) | 0.982 | 6.4 % | 96 % (92) | 0.990 | 3.2 % | 97 % (88) |
| C off-class (12 others) | 0.752 | 0.45 %ᴺ | 97 % (92) | 0.891 | 0.25 %ᴺ | 96 % (88) |
| E* = mean(E_x, E_y, E_z) | 0.988 | 4.6 % | 92 % (88) | 0.988 | 3.9 % | 89 % (74) |
| E_z | 0.988 | 5.8 % | 96 % (91) | 0.989 | 4.1 % | 93 % (84) |
| a_sf L | 0.999 | 0.3 % | 95 % (92) | 0.999 | 0.2 % | 98 % (98) |
| loc p99, uniaxial | 0.988 | 4.3 % | 94 % (89) | 0.990 | 2.9 % | 94 % (88) |
| loc p99, shear | 0.983 | 5.0 % | 94 % (89) | 0.987 | 3.5 % | 93 % (86) |

Errors concentrate in the **pinch-neck region** (0.3 < w < 0.7, rho* < 0.4; 72 rows). CV MAPE, ensemble / GP:

| quantity | rest (218 rows) | neck (72 rows) |
|---|---|---|
| K* | 0.6 / 0.6 % | 1.3 / 1.2 % |
| k*/k_s | 0.6 / 1.1 % | 3.8 / 3.9 % |
| E* | 1.4 / 2.2 % (median 0.5 / 1.1 %) | 11.6 / 11.8 % |
| C44 | 1.4 / 4.3 % | 8.2 / 12.4 % |
| loc p99 uniaxial z | 1.5 / 2.8 % | 8.3 / 10.0 % |
| a_sf L | 0.2 / 0.2 % | 0.4 / 0.4 % |

Held-out pure G/D rows (30): E* 1.3 % (ensemble) / 1.8 % (GP), K* 0.7 / 0.9 %, k* 0.7 / 1.0 %.

*Extrapolation split "band": trained on w < 0.25 or w > 0.75 (158 rows), tested on the held-out blends
0.25 <= w <= 0.75 (132 rows):*

| quantity | GP R² | GP err | GP cov95 | ens R² | ens err | ens cov95 |
|---|---|---|---|---|---|---|
| K principal values | 0.974 | 6.4 % | 40 % (38) | 0.979 | 5.4 % | 85 % (85) |
| K* = tr K/3 | 0.979 | 5.6 % | 27 % (23) | 0.982 | 4.8 % | 80 % (80) |
| k_eff diagonal | 0.920 | 11.5 % | 84 % (72) | 0.916 | 12.4 % | 82 % (68) |
| k_eff off-diagonal | -0.678 | 5.27 %ᴺ | 69 % (55) | 0.033 | 3.77 %ᴺ | 60 % (32) |
| C normal (C11, C22, C33) | 0.770 | 59.2 % | 70 % (64) | 0.737 | 66.8 % | 72 % (53) |
| C coupling (C12, C13, C23) | 0.836 | 9.21 %ᴺ | 64 % (56) | 0.840 | 9.33 %ᴺ | 67 % (48) |
| C shear (C44, C55, C66) | 0.838 | 33.2 % | 82 % (75) | 0.817 | 36.9 % | 74 % (56) |
| C off-class (12 others) | -0.444 | 2.09 %ᴺ | 65 % (52) | 0.145 | 1.41 %ᴺ | 71 % (54) |
| E* = mean(E_x, E_y, E_z) | 0.822 | 43.1 % | 67 % (61) | 0.794 | 49.3 % | 62 % (39) |
| E_z | 0.837 | 42.1 % | 77 % (70) | 0.805 | 49.9 % | 76 % (57) |
| a_sf L | 0.945 | 2.3 % | 49 % (45) | 0.937 | 2.4 % | 86 % (86) |
| loc p99, uniaxial | 0.880 | 14.7 % | 72 % (62) | 0.856 | 16.8 % | 74 % (61) |
| loc p99, shear | 0.867 | 16.5 % | 72 % (59) | 0.848 | 17.8 % | 65 % (48) |

*Split "sparse": trained on all pure G/D + a random 25 % of the blends (95 rows), tested on the other
75 % of the blends (195 rows):*

| quantity | GP R² | GP err | GP cov95 | ens R² | ens err | ens cov95 |
|---|---|---|---|---|---|---|
| K principal values | 0.996 | 2.0 % | 90 % (89) | 0.994 | 2.4 % | 84 % (83) |
| K* = tr K/3 | 0.997 | 1.5 % | 83 % (82) | 0.995 | 2.1 % | 76 % (76) |
| k_eff diagonal | 0.981 | 4.0 % | 95 % (90) | 0.982 | 3.4 % | 91 % (82) |
| k_eff off-diagonal | 0.737 | 1.46 %ᴺ | 97 % (92) | 0.851 | 1.02 %ᴺ | 89 % (72) |
| C normal (C11, C22, C33) | 0.947 | 11.8 % | 91 % (85) | 0.947 | 11.8 % | 87 % (77) |
| C coupling (C12, C13, C23) | 0.868 | 5.85 %ᴺ | 85 % (73) | 0.958 | 3.10 %ᴺ | 78 % (68) |
| C shear (C44, C55, C66) | 0.942 | 9.9 % | 96 % (92) | 0.948 | 9.0 % | 93 % (81) |
| C off-class (12 others) | -2.400 | 1.62 %ᴺ | 98 % (91) | 0.585 | 0.61 %ᴺ | 92 % (79) |
| E* = mean(E_x, E_y, E_z) | 0.964 | 9.5 % | 89 % (84) | 0.959 | 9.7 % | 85 % (70) |
| E_z | 0.960 | 11.1 % | 93 % (88) | 0.965 | 10.0 % | 87 % (78) |
| a_sf L | 0.996 | 0.5 % | 87 % (82) | 0.994 | 0.6 % | 87 % (87) |
| loc p99, uniaxial | 0.977 | 6.6 % | 96 % (90) | 0.975 | 6.0 % | 90 % (79) |
| loc p99, shear | 0.974 | 7.0 % | 96 % (92) | 0.973 | 6.3 % | 90 % (80) |

**6c — Does geometry-based learning extrapolate better?** (`results/task6_cnn_metrics.csv`, figure
`task6_cnn_extrapolation.png`). Same targets (log K*, log E*) and rows for all three; GP/ensemble see
theta, the CNN sees the 32³ voxel cell + a_z; raw (not recalibrated) intervals; CNN = 3 members,
120 epochs.

| split | target | GP MAPE / cov95 | ensemble MAPE / cov95 | CNN MAPE / cov95 |
|---|---|---|---|---|
| cv5 | K_star | 0.7 % / 96 % | 0.8 % / 87 % | 1.3 % / 98 % |
| cv5 | E_star | 5.6 % / 91 % | 5.2 % / 85 % | 5.8 % / 95 % |
| sparse | K_star | 1.5 % / 88 % | 1.8 % / 81 % | 2.2 % / 93 % |
| sparse | E_star | 9.4 % / 89 % | 11.1 % / 70 % | 7.3 % / 92 % |
| band | K_star | 5.7 % / 35 % | 5.5 % / 47 % | 6.7 % / 45 % |
| band | E_star | 45.3 % / 66 % | 53.6 % / 33 % | 107.2 % / 20 % |

- **No, not across a gap of unseen topology.** On the band split the CNN is *worse* than the parameter
  models for E* (107 % vs 45-54 % MAPE) and slightly worse for K*. Blends with 0.375 < w < 0.625 lose
  72-79 % of the stiffness a log-linear G-D interpolation predicts at rho* <= 0.3 and ~50 % at
  rho* = 0.3-0.4 (pinch necks, Tasks 1-3); no training
  blend outside the band shows this (E*/interpolation >= 0.91), so the mechanism is absent from the
  training data and a voxel model has no basis to predict it — it meets unfamiliar neck features and
  over-predicts more than the smooth parameter models do. Its E* error even *grows* toward the band
  edge (up to ~500 % at w ~ 0.72): at rho* ~ 0.2-0.25, E* drops 3-8x between w ~ 0.76 (train) and
  w ~ 0.72 (test) while the controlling necks are 0.03-0.09 L thick = 1-3 voxels at 32³, i.e. the
  feature that sets the stiffness is at the input resolution and the two cells look almost the same
  to the network. Global max-pooling (a neck detector) did not help (95 % vs 110 %, 2-member trial);
  a 64³ input (8x cost) is the obvious next experiment.
- **Yes, when the topology change is sampled sparsely.** With 25 % of the blends in training (sparse)
  the CNN is the best E* model (7.3 % vs 9.4 % GP, 11.1 % ensemble) with the best-calibrated raw
  intervals (92 % vs 89 / 70 %); for K* the parameter models stay ahead (1.5-1.8 % vs 2.2 %).
- **Dense data (cv5): no advantage.** With 232 training rows the CNN is on par for E* (5.8 % vs
  5.2-5.6 %) and behind for K* (1.3 % vs 0.7-0.8 %), but again the best calibrated (95-98 % raw
  coverage vs 85-96 %). Geometry helps most when parameter-space data are scarce.
- Practical consequence: Task 9/11 use the parameter-based `ClosureModel` (full tensors, cheap); the CNN
  stays a research result (and a tool for geometries outside the (w, rho*, a_z) family).
- None of the models may be trusted in the pinch-neck region without examples there; the Task 5 table
  does sample it, so the production models do see it (CV errors above).

**Timings** (build sandbox, 2 cores; parallel jobs running, so upper bounds)
| Step | Time |
|---|---|
| GP fit, 40 targets (2 jobs) | ~50 s; production with 5-fold recalibration ~5 min |
| Ensemble fit, 5 members (2 jobs) | ~20 s; with recalibration ~2 min |
| `scripts/task6_evaluate.py` | ~6 min |
| CNN member, 290 rows x 120 epochs | ~6 min/member (1 core); production ensemble of 3 (2 jobs): 16.5 min |
| `predict`, 512 points, with std / mean only | GP 0.69 / 0.12 s; ensemble 0.33 / 0.14 s |
| CNN `predict`, 1 theta (voxelize + 16-fold TTA x 3 members) | ~1 s |

**Recommended for Task 9 / 11:** `ClosureModel.load("ensemble")` — best held-out accuracy for k_eff, C,
E* and localization on pure *and* blended cells, equal for K and a_sf, fastest with std. Cross-check
with `"gp"` (better calibrated for K: 91 vs 86 % coverage). Re-check final optimized designs with the
direct solvers when they sit in the neck region.

**Open issues / notes for later tasks**
1. **Pinch-neck region (0.3 < w < 0.7, rho* < 0.4):** stiffness/localization surrogate errors ~10 %
   (vs ~1-2 % elsewhere) and intervals too narrow there. Task 11 options: forbid the region (it is
   structurally poor anyway, Task 3), add Task 5 samples there (densify w in 0.3-0.7, rho* 0.2-0.4 and
   retrain — the scripts are ready), or verify final designs directly.
2. Ensemble K* coverage is 86 % in CV (K multipliers ~1.1): fine for design, not for K uncertainty
   studies — use the GP for those.
3. Off-diagonal / off-class tensor components (blends) are small (NMAE 0.3-0.7 % of the diagonal);
   their R² is low because most values are ~0 — Task 9 rotates full tensors, so these errors are
   negligible next to the diagonal ones.
4. k_eff/k_s is valid for the Task 5 contrast k_f/k_s = 1/325 only; C scales exactly with E_s at
   nu = 0.33. Retrain if the material changes.
5. `ClosureModel` warns outside the training box (rho* 0.2-0.5, a_z 0.6875-1.5); the optimizer
   bounds (plan §2) are inside it.
6. Task 5 STATUS notes were missing from the repo (see Task 5 above).
7. Workspace: PyPI blocked (no pyarrow, no torch); the parquet table was read with a throw-away
   pure-Python reader in the sandbox only (not shipped). Python 3.13, numpy 2.5.3, scipy 1.18.1,
   scikit-learn 1.9.1.
8. Citations to verify: Lakshminarayanan, Pritzel & Blundell 2017 (NeurIPS, deep ensembles); Arsigny,
   Fillard, Pennec & Ayache 2006 (Magn. Reson. Med. 56:411, Log-Euclidean metric); Rasmussen &
   Williams 2006 (*Gaussian Processes for Machine Learning*, Matérn/ARD); Daleckii & Krein 1965 /
   Higham 2008 (*Functions of Matrices*) for the Fréchet derivative; Kuleshov, Fenner & Ermon 2018
   (ICML, recalibration of regression uncertainty); Loshchilov & Hutter 2019 (AdamW).

**Figures** (`results/figures/`): `task6_parity_cv.png` (5-fold CV parity, GP and ensemble, recalibrated
95 % bars), `task6_parity_band.png` (band split), `task6_calibration.png` (observed vs nominal coverage,
raw and recalibrated, cv5 / band / sparse), `task6_cnn_extrapolation.png` (CNN vs GP vs ensemble, parity
per split + error vs w).

**Tests**: 336 passed + 4 skipped (pyamg x3, pyarrow x1) in 32.5 min — Tasks 0-6 incl. slow, run alongside
the CNN experiment (`results/task6_pytest_log.txt`). Task 6 alone (`tests/test_surrogates.py`): 46 passed
(1 `slow`: GP hold-out accuracy on the real table); `pytest -m "not slow" tests/test_surrogates.py` ~1 min.
Covered: vectorization / logm / expm / Fréchet derivative vs finite differences (incl. repeated
eigenvalues); Mandel rotation = `elasticity.rotate_stiffness`; group closure, projector rank and
idempotence per class, class structure of projected tensors (trigonal 7-constant pattern); declared
symmetry is physical (Task 2 solver: G cubic, w = 0.5 trigonal and *not* cubic, stretched G tetragonal);
latent round trip; delta-method std vs Monte Carlo; gradient checks of every NN layer and the NLL;
periodic-conv translation equivariance; Adam; GP = scikit-learn prediction, ARD switches off an
irrelevant input; ensemble uncertainty grows away from data; CNN exact shift invariance and exact
TTA invariance; `ClosureModel` shapes, SI and L scaling, SPD, exact symmetry classes, save/load,
bounds, recalibration; metrics and splits; real table (symmetry-projection size, hold-out accuracy);
trained models reproduce the Task 2-4 tables (rho* = 0.35: K*, k*, E* within 3 / 3 / 6 %).
Tests needing the Task 5 table read `data/closures.parquet` (pyarrow) or `$VOXLAT_CLOSURE_TABLE`; tests of
the trained models skip if `models/` is empty.

## Task 7 — Finite-gap study (RQ1)  ✅

**Built**
- `src/voxlat/homogenization/finite_gap.py` — strips (`GapSpec`, `build_strip`, `bulk_cell`), wall removal
  (`compute_properties`, laminate mixed form), homogenized prediction (`BulkInterpolator`,
  `homogenized_properties`), discrepancies, discrepancy models (`PhysicalDiscrepancyModel`,
  `GPDiscrepancyModel`, `grouped_cv`) and the Task 9 API `FiniteGapCorrection`.
- `src/voxlat/homogenization/finite_gap_study.py` — suite design, resumable workers (one JSON part per bulk
  cell / strip in `data/finite_gap/`, git-ignored), consolidation to **`data/finite_gap.csv`** (340 rows; +parquet
  when pyarrow exists).
- `src/voxlat/homogenization/coarse.py` — **two-level preconditioner** (Jacobi + unsmoothed aggregation coarse
  space on 8³ boxes: constants for conduction / each Stokes velocity component (4³ boxes), the 6 rigid-body modes
  for elasticity). Needed because Jacobi iterations grow with the long strip axis and pyamg is not installable here.
- Solver extensions (backward compatible; defaults and Task 2-6 numbers unchanged; Task 2-4 tests re-run):
  `preconditioner="two_level"` (+ `coarse_block`) in all three solvers; `effective_conductivity(..., directions=)`;
  `effective_elasticity(..., load_cases=)` (unsolved columns NaN; localization needs all six).
- Scripts: `scripts/task7_finite_gap.py` (run suites, `--estimate-only`, `--quick`, `--table-only`),
  `scripts/task7_fit_discrepancy.py` (models + tables), `scripts/task7_figures.py`.
- Model file `models/finite_gap_correction.json` (fitted coefficients + metadata, plain JSON).
- Tests: `tests/test_finite_gap.py` (26). `configs/reference.yaml`: gradient limit now evidence-based (below).

**Set-up and decisions**
1. *Strip* = lattice of N cells across the gap (x = r), periodic and one cell wide in y = s and z = axial, plus a
   solid wall layer t_w = 0.5 L; periodic in x too, so one wall bounds every lattice layer on both sides (no slip,
   perfect bond, k_s). N ∈ {1, 1.5, 2, 3, 4, 6, 8}; N = 1.5 cuts the two walls at different phases.
2. *Same grid for strip and bulk*: the strip lattice is a bit-exact x-tiling of the bulk reference cell (same n,
   same cut phase, same bisection threshold), so the voxel staircase error (Tasks 2-4: up to 17 % for k_eff) cancels
   in q_strip/q_bulk. Checked: n = 24/32/48 change δ by ≤ 0.4 pp (K_t), ≤ 0.6 pp (k_n), ≤ 1.4 pp (C_nn, G_t at N = 1).
3. *Walls removed analytically* ("bulk closure + plain walls" = laminate of homogeneous layers, exact for laminates,
   tested against Backus and the slit/series cases): k_n from T/k_stack = H/k_n + t_w/k_s; K_t = (T/H) K_stack[yz];
   C from the **laminate mixed form** (tractions s_xx, s_xz, s_xy and in-plane strains continuous; the map
   [s_n; e_t] → [e_n; s_t] is volume-additive) → full 6×6 apparent core tensor.
4. *Quantities*: k_n = through-gap conductivity; K_t = in-plane permeability (mean of K_ss, K_zz; flow along the gap);
   C_nn = C_rrrr normal stiffness across the gap; **G_t = (G_rs + G_rz)/2 = (C66 + C55)/2, the shear with the walls
   sliding parallel to each other — this is what carries torque (G_rs) and thrust (G_rz) from sleeve to outer wall**;
   G_sz = C44, shear in the plane of the gap. ("In-plane shear" in the plan was ambiguous; both are reported.)
5. *Cut phase*: default cuts φ0 ∈ {0, 1/3, 2/3}. **For gyroid and diamond the 4_1 screws along x and the 2-fold axes ⟂ x
   make φ ≡ φ + 1/4 ≡ −φ**, so the distinct cuts lie in [0, L/8]; 1/3 and 2/3 both reduce to L/12 (identical results,
   verified), so an extra cut L/8 was added for G/D → distinct cuts {0, L/12, L/8}. Blends (trigonal) have no such
   symmetry: 3 distinct cuts over a full period. Analyses use one row per distinct cut (`distinct_cuts`).
   Screw symmetry is also a test (K_ss ↔ K_zz, G_rs ↔ G_rz under φ → φ + 1/4).
6. *Discrepancy*: δ_q = q_strip/q_hom − 1 (corrected closure q = q_bulk (1 + δ)); the error of the homogenized
   prediction is −δ/(1+δ). Graded strips: q_hom = local closures at every voxel layer combined like a laminate
   (harmonic k, arithmetic K, mixed-form C) with log-Euclidean PCHIP interpolation of bulk cells at
   ρ* = 0.175…0.525 (step 0.025).

**Public API (`voxlat.homogenization.finite_gap`; main names also in `voxlat.homogenization`)**
```python
GapSpec(w, rho, N, phase=0, gradient=0, a_z=1, wall=0.5, n=32)  # .n_lattice .n_wall .layer_rho .case_id
build_strip(spec) -> Strip(.solid (x, y, z) bool, .thresholds, .layer_density); bulk_cell(w, rho, n, phase, a_z)
strip_properties(spec, preconditioner="two_level", k_s=, k_f=, E=, nu=) -> (Strip, GapProperties)
bulk_properties(w, rho, n, phase, a_z) -> GapProperties
GapProperties: .k_n [W/mK] .K_t (2x2, L^2) .C (6x6 Pa) + stack values; .scalars() -> k_n, K_t, K_s, K_z, C_nn,
    G_t, G_rs, G_rz, G_sz, C_tt; .dimensionless()
compute_properties(solid, n, n_lattice, n_wall, quantities=("k","K","C"))  # any stack, wall removed
laminate_mixed_form(C), from_mixed_form(M), laminate_average(Cs, f), laminate_core(C_stack, C_wall, f_wall)
BulkInterpolator(rhos, props); homogenized_properties(spec, bulk); discrepancy(strip, hom); homogenized_error(delta)
reduced_phase(w, phase); distinct_cuts(df); brinkman_channel_factor(N, K/L^2)
PhysicalDiscrepancyModel(quantities).fit(df) .predict(q, N, w, rho, a_z, gradient) .predict_std(q, N)
GPDiscrepancyModel(physical).fit(df).predict(df); grouped_cv(df, groups="theta"|"N", strata=)
corr = FiniteGapCorrection.load()          # models/finite_gap_correction.json
corr.factor(q, N, w, rho, a_z=1, gradient=0) -> 1 + delta (phase-averaged);  corr.factor_std(q, N)
# voxlat.homogenization.coarse: TwoLevelPreconditioner(A, P), box_aggregates, scalar_coarse_basis, rigid_body_coarse_basis
```

**Results (n = 32, wall 0.5 L; phase mean over distinct cuts; `results/task7_coefficients.txt`)**

δ [%] at N = 1 / 2 / 3 (the jacket has h = 6 mm, L = 2–6 mm → N = 1–3):

| | ρ* | k_n | K_t | C_nn | G_t | G_sz |
|---|---|---|---|---|---|---|
| Gyroid | 0.25 | −2.1/−1.0/−0.7 | **−28.4/−14.2/−9.4** | +7.3/+3.5/+2.3 | +7.3/+3.4/+2.2 | +11.4/+5.6/+3.7 |
| | 0.35 | −2.3/−1.1/−0.8 | **−26.6/−13.3/−8.9** | +4.1/+1.9/+1.3 | +3.3/+1.5/+1.0 | +4.2/+2.1/+1.4 |
| | 0.45 | −1.9/−1.0/−0.6 | **−25.4/−12.7/−8.5** | +1.9/+0.9/+0.6 | +1.6/+0.7/+0.5 | +1.9/+0.9/+0.6 |
| Diamond | 0.25 | −0.5/−0.3/−0.2 | **−22.4/−11.2/−7.5** | +4.4/+2.1/+1.4 | −2.0/−1.1/−0.8 | +11.2/+5.5/+3.6 |
| | 0.35 | −0.7/−0.4/−0.2 | **−20.3/−10.2/−6.8** | +4.4/+2.1/+1.4 | −3.7/−1.9/−1.3 | +7.7/+3.8/+2.5 |
| | 0.45 | −0.5/−0.3/−0.2 | **−18.7/−9.4/−6.2** | +4.4/+2.2/+1.4 | −3.9/−2.0/−1.4 | +5.5/+2.7/+1.8 |
| Blend 0.5 | 0.25 | +9.7/+4.3/+2.8 | −26.5/−13.4/−9.1 | +85/+18/+11 | +53/+18/+11 | +105/+52/+35 |
| | 0.35 | +13.9/+5.9/+3.7 | −23.9/−12.0/−8.1 | +86/+15/+9 | +51/+17/+10 | +84/+41/+27 |
| | 0.45 | +1.4/+0.7/+0.4 | −23.4/−11.7/−7.8 | +9.4/+4.3/+2.8 | +6.8/+3.1/+2.0 | +14/+7/+5 |

RQ1 answers:
1. **The dominant finite-gap error is the in-plane permeability: the bulk closure over-predicts K_t by 23–40 % at N = 1
   (δ = −19 … −28 %), 10–17 % at N = 2, 7–10 % at N = 3** — the whole jacket design range. It is cleanly ∝ 1/N
   (rms deviation 0.4 pp per (θ, cut); adding a 1/N² term changes nothing), almost independent of ρ* and of the cut
   (G/D phase std of a ≤ 0.011), i.e. a fixed wall layer of thickness a L/2 ≈ 0.13 L (G), 0.10 L (D), 0.12 L (blend)
   that carries no flow. **No near-wall channelling** (`task7_velocity_slices.png`): the planar porosity next to a cut
   TPMS equals the bulk's (unlike packed beds), the wall closes the pores it cuts and the flow deficit is confined to
   ≈ L/4 of each wall; the interior profile equals the bulk profile. Brinkman with μ_e = μ (2√K/L = 0.14 G, 0.11 D,
   0.13 blend at ρ* = 0.35) under-predicts the loss by ~1.9×, and a scales with L, not √K (a ≈ const in ρ*).
   Stretch a_z = 0.75 / 1 / 1.5: a_K = 0.24 / 0.27 / 0.30 (G), 0.18 / 0.20 / 0.25 (D).
2. **Through-gap conductivity: negligible for G/D** (|δ| ≤ 2.3 %, G −2 %/N, D ≈ 0, below the Task 6 k_eff surrogate
   error); the walls' constriction/spreading of strut contacts nearly balance. Blends at ρ* ≤ 0.35: +10–14 % at N = 1,
   strongly cut dependent (up to +38 % for one cut).
3. **Stiffness: the bonded walls stiffen the core.** G/D at N = 1 (∝ 1/N): C_nn +2…+7 %, G_t +1.6…+7 % (G) and −2…−4 % (D),
   G_sz +2…+11 %. Blends below ρ* ≈ 0.4: **+50…+190 % at N = 1** (cut through the pinch necks of Tasks 1-3: the wall
   replaces the weakest links; the bulk blend is only 0.013–0.027 E_s stiff there), not ∝ 1/N at N = 1 and dominated
   by the cut phase.

**Discrepancy model — fitted correction (`models/finite_gap_correction.json`)**
- series (k_n, C_nn, G_t): q/q_bulk = (1 + c_g g²) / (1 + a/N);  parallel (K_t, G_sz): q/q_bulk = (1 + c_g g²)(1 − a/N);
  a > 0 = loss, a < 0 = gain. a(θ) = β·[1, r, w, b, r w, r b, ln a_z], r = (ρ* − 0.35)/0.1, b = 4w(1−w); least
  squares on the observed a with rows weighted by 1/N (error in δ); c_g from the G/D graded strips.

| q | 1 | r | w | b | r·w | r·b | ln a_z | c_g | phase std of a |
|---|---|---|---|---|---|---|---|---|---|
| k_n | +0.0206 | −0.0011 | −0.0164 | −0.0729 | +0.0011 | +0.0248 | −0.0312 | +0.05 | 0.065 |
| **K_t** | **+0.2663** | −0.0147 | **−0.0616** | +0.0110 | −0.0035 | −0.0001 | +0.0891 | −1.29 | 0.039 |
| C_nn | −0.0385 | +0.0234 | −0.0034 | −0.2056 | −0.0239 | +0.1111 | −0.0563 | +0.84 | 0.065 |
| G_t | −0.0383 | +0.0261 | +0.0680 | −0.2169 | −0.0170 | +0.1032 | −0.0320 | +0.81 | 0.046 |
| G_sz | −0.0589 | +0.0473 | −0.0294 | −0.5978 | −0.0190 | +0.4105 | +0.0099 | −0.70 | 0.115 |

  Handy values: a_K = 0.266 (G), 0.205 (D), 0.251 (blend) at ρ* = 0.35, a_z = 1 → **K_eff = K_bulk (1 − 0.27/N)** for the
  gyroid. Observed a per morphology/density: `results/task7_wall_coefficients.csv`.
- **Which is sufficient?** Leave-one-(w, ρ*, a_z)-out CV, rms error of δ in pp (`results/task7_model_cv.csv`,
  `task7_discrepancy_models.png`; "floor" = pooled scatter over cut phases, which no (N, θ)-model can remove):

| | k_n | K_t | C_nn | G_t | G_sz |
|---|---|---|---|---|---|
| G/D: uncorrected / physical / +GP / floor | 1.15 / 0.85 / 0.85 / 0.96 | **12.9 / 0.40 / 0.40 / 0.28** | 2.79 / 1.17 / 1.16 / 1.31 | 2.23 / 0.98 / 1.04 / 0.82 | 4.26 / 1.77 / 1.95 / 0.67 |
| blend: uncorrected / physical / +GP / floor | 7.4 / 7.9 / 7.9 / 7.5 | **12.9 / 3.5 / 3.5 / 4.1** | 32 / 26 / 27 / 24 | 20 / 18 / 18 / 9.8 | 38 / 21 / 21 / 11 |

  Leave-one-N-out (extrapolation in N) gives the same picture (G/D K_t 13.4 → 0.29 pp, = floor).
  **The physical 1/N form is sufficient; the GP on its residuals never helps** (≤ 0.1 pp better, sometimes worse):
  what remains is cut-phase scatter (aleatoric for a homogenized model) or, for blends, the pinch-neck cut effect,
  neither of which is a smooth function of (N, θ, g). For blends the stiffness corrections are not predictable from
  (N, θ) at all (residual ≈ floor ≈ 10–25 pp) — another reason to keep mid-w out of the low-ρ* region in Task 11.

**Graded strips (`results/task7_graded.csv`, `task7_graded_error.png`)** — gradient effect relative to the uniform
strip with the same N, mid-gap ρ* and cut, Δ_g = (1+δ_graded)/(1+δ_uniform) − 1. G/D, worst single cut, any quantity:
**g = |∇ρ*| L = 0.025: 0.9 %, 0.05: 1.0 %, 0.075: 2.0 %, 0.15: 3.5 %** (largest for K_t, which drops: momentum
exchange between fast low-ρ* and slow high-ρ* layers that a local Darcy closure cannot see). → **`max_density_change_per_cell`
= 0.05 confirmed** (was a placeholder since Task 0; ≤ 1 % local-closure error for G/D). Blends: Δ_g = −8…−20 % (G_sz),
−4…+11 % (C_nn) already at small g, because their wall effect changes across the pinch transition (ρ* ≈ 0.375) as
the wall densities move; no gradient limit makes graded blends safe there.

**Checks (`task7_convergence.png`, `results/task7_convergence.csv`, `results/task7_wall_thickness.csv`)**
- Resolution n = 24 / 32 / 48 (G, D; ρ* = 0.35; N = 1, 2, 4): K_t to 0.4 pp, k_n 0.6 pp, G_t / G_sz ≤ 1 pp,
  C_nn ≤ 1.4 pp (N = 1). The n = 32 stiffness δ therefore carry ~±1.5 pp at N = 1.
- Wall thickness t_w/L = 1/8 … 1: K_t exactly independent; k_n and G_t converge by 0.5 L (within 0.2–0.4 pp of 1 L);
  thin walls (1/8 L) bias k_n up by ~2 pp (heat crosses a thin periodic wall between aligned strut contacts without
  spreading) and G_t down by ~3 pp (wall bending); C_nn varies ±1 pp non-monotonically. Real walls are 1.5–2 mm = 0.25–1 L.
- Exact cases (tests): laminate mixed form = Backus; empty gap = MAC slit (H²+2)/12; series laminate k; solid gap =
  solid; no-wall integer-N strip = bulk to 1e-6; screw symmetry; two-level = Jacobi to 1e-6.

**Timings (sandbox, 2 parallel workers, two-level, incl. extraction)**: median per strip at n = 32: N = 1: 11 s, N = 2: 18 s,
N = 4: 30 s, N = 8: 53 s (blends 13 / 22 / 50 / 96 s);
n = 48: N = 2 ~55–65 s, N = 8 blend 390 s (peak RSS 2.3 GB). Iterations: elasticity 113–344, Stokes 116–683, conduction
82–135 (Jacobi on the N = 8 gyroid strip: 1100–1350 elasticity iterations, 271 s). Full n = 32 study (336 strips +
157 bulk cells): 92 min on 2 cores. Fit + CV: 3.5 min. Figures: 1.5 min (velocity fields cached).

**Optional overnight run (n = 48, production numbers)**
```powershell
python scripts/task7_finite_gap.py --estimate-only --suite uniform graded stretch --n 48 --n-jobs 4
python scripts/task7_finite_gap.py --suite uniform graded stretch --n 48 --n-jobs 4   # ~3-4 h, <= 2.3 GB per worker
python scripts/task7_fit_discrepancy.py      # picks n = 48 automatically once its uniform suite (231 strips) is complete
python scripts/task7_figures.py
```
Sandbox estimate 8.5 CPU-h (the n = 32 estimate was ~25 % low); 4 workers need ≤ ~9 GB. Resumable: rerun the same
command. Use `--preconditioner auto` to try pyamg on the laptop (results agree to the solver tolerance).

**Open issues / notes for later tasks**
1. **Task 9 must apply the K correction**: K_t,eff = K_bulk (1 − a_K/N), N = h/L (local), ≈ −9…−28 % in the jacket
   range — larger than every other closure uncertainty. Apply it to the in-plane (s, z) permeability only; the
   through-gap (r) component is irrelevant in the depth-averaged model. `FiniteGapCorrection.factor("K_t", ...)`.
   k_n: skip for G/D (|δ| ≤ 2 %). Stiffness: the correction is conservative to ignore for G/D (walls stiffen) except
   diamond G_t (−2…−4 % at N = 1); use G_t (not G_sz) for torque/thrust transfer.
2. a is defined for the gap N = h/L with L the radial cell size; for graded designs use local L and ρ*.
   Valid: N ≥ 1, ρ* 0.25–0.45 fitted (0.2–0.5 graded), a_z 0.75–1.5, network TPMS with bonded walls.
3. **Cut phase is a free design choice (Task 12)**: for G/D its effect is small (K_t ±0.5 pp, stiffness ±1–2 pp at
   N = 1); for blends ±7 pp in K_t and up to a factor 2 in stiffness. Task 12 should fix the phase at the sleeve and
   report it; `factor_std` gives the phase scatter.
4. Blends with ρ* < 0.4: finite-gap stiffness corrections are cut-dependent and unpredictable (consistent with the
   Task 1-6 pinch-neck findings) — another argument to restrict mid-w at low ρ* in Task 11.
5. The stiffness extraction attributes all wall-lattice interaction energy to the lattice layer (the definition of an
   apparent core property); in-plane quantities (G_sz, C_tt) are wall-dominated in the stack (at N = 1 the wall carries
   ~3–10× the core's in-plane stiffness), so their δ are sensitive to small modelling details — report them as secondary.
6. Gradient study is radial (across the gap), the design (Task 11) grades in-plane (s, z). The radial result bounds the
   local-closure error per unit gradient for series (k_n, C_nn) and parallel (K_t) combinations alike; an in-plane
   graded check could be added in Task 9 if needed.
7. Numbers are at n = 32; the overnight n = 48 run would tighten stiffness δ by ~1 pp (K_t is already converged).
8. Citations to verify: Brinkman 1949 (Appl. Sci. Res. A1:27) / Neale & Nader 1974 (Can. J. Chem. Eng. 52:475) for
   the Brinkman channel; Backus 1962 and Postma 1955 (Geophysics 20:780) for the laminate mixed form; Vaněk, Mandel &
   Brezina 1996 (Computing 56:179) and Toselli & Widlund 2005 for aggregation / two-level preconditioning; space groups
   I4₁32 / Fd-3m of the gyroid / diamond networks (e.g. Schröder-Turk et al. 2006; Hyde et al. 1997) for the cut-phase
   symmetry; near-wall porosity/channelling in packed beds (e.g. Benenati & Brosilow 1962, AIChE J. 8:359) for the contrast.
9. Workspace: PyPI blocked (no pyamg, pyarrow, torch); numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, Python 3.13.

**Figures** (`results/figures/`): `task7_error_vs_N.png` (main figure: δ vs 1/N, 5 quantities, G/D top, blend bottom,
cut range bars, model lines, jacket band), `task7_discrepancy_models.png` (CV: uncorrected / physical / +GP / floor),
`task7_velocity_slices.png` (u_z slices and plane-averaged profiles, N = 2, n = 48), `task7_graded_error.png`,
`task7_convergence.png` (resolution and wall thickness). Tables: `results/task7_*.csv`, `results/task7_coefficients.txt`,
`results/task7_fit_log.txt`; data `data/finite_gap.csv`.

**Tests**: 361 passed + 6 skipped (pyamg ×3, pyarrow-dependent ×3) in 16.9 min, Tasks 0–7 incl. slow
(`results/task7_pytest_log.txt`). Task 7 alone (`tests/test_finite_gap.py`): 27 tests in ~25 s.

## Task 8 — Literature closures (Nu, Forchheimer)  ✅

**Built**
- `src/voxlat/closures/empirical.py` — C_F and interstitial Nu correlations with published validity boxes,
  `CorrelationRangeWarning` / `CorrelationRangeError`, a literature registry (`LITERATURE`, `literature_table()`),
  packed-bed references for cross-checks, and the convenience chain `interstitial_heat_transfer`. Main names
  re-exported from `voxlat.closures`.
- `tests/test_empirical.py` (26 tests, < 1 s): published coefficients, the papers' own tabulated CFD values, the
  Darcy–Forchheimer/friction identity, Pr scaling and the Re → 0 limit, validity warnings, reference values of the
  packed-bed correlations, and **independent checks of Tasks 1 and 4 against the published CFD** (below).
- `scripts/task8_literature_closures.py` (~5 s) → `results/task8_literature_table.csv`,
  `results/task8_operating_range.csv`, `results/task8_nusselt_comparison.csv`, `results/figures/task8_forchheimer.png`,
  `results/figures/task8_nusselt.png`.
- `docs/task8_literature_closures.md` — the full literature table, recommendations, risk table and the manuscript
  limitations paragraph.

**Definitions (module docstring)**: φ = 1 − ρ*, D_h = 4φ/a_sf (a_sf per total volume), U_b = U_s/φ,
**Re_Dh = U_b D_h/ν = 4U_s/(a_sf ν)**, Re_K = U_s√K/ν, Fo = C_F Re_K; −∇p = μU_s/K + ρC_F U_s²/√K;
f = (dp/dx) D_h/(½ρU_b²) (Darcy-type, 4× Fanning), so **f Re = 2φD_h²/K + 2φ²C_F (D_h/√K) Re** (Savoldi et al. 2026, Eq. 23);
Nu = h_sf D_h/k_f with h_sf the surface-averaged wall flux over (T_wall − T_bulk) on the whole solid–fluid interface.

**Public API (`voxlat.closures.empirical`)**
```python
forchheimer_coefficient(porosity, w=0, source="gajetti2025"|"gajetti2025_table"|"savoldi2026", re=None,
                        on_extrapolation="warn"|"raise"|"ignore") -> C_F        # blends: ln C_F linear in w
pressure_gradient(U_s, K, C_F, density=, dynamic_viscosity=) -> |grad p| [Pa/m]
friction_factor_re(Re, porosity, D_h, K, C_F) -> f Re  (finite at Re = 0);  friction_factor(...) -> f
nusselt_interstitial(Re, porosity, f_Re, Pr, variant=1, pr_exponent=1/3, w=0, on_extrapolation=) -> Nu_Dh
interstitial_heat_transfer(U_s, porosity, a_sf, K_bulk, w=0, conductivity=, kinematic_viscosity=, prandtl=,
                           cf=None, cf_source=, variant=1, pr_exponent=1/3) -> InterstitialHeatTransfer
    # .h_sf [W/m2K] .h_sf_a_sf [W/m3K] .nusselt .reynolds .friction .friction_re .cf .hydraulic_diameter
hydraulic_diameter, equivalent_particle_diameter (6(1-phi)/a_sf), pore_velocity, reynolds_hydraulic,
reynolds_permeability, forchheimer_number, stanton_savoldi2026(Re, phi, f, variant)
gajetti2025_permeability / gajetti2025_forchheimer (phi, tpms); savoldi2026_hydraulic_diameter / _permeability /
_forchheimer (phi); nusselt_gnielinski_packed_bed(Re_p, Pr, phi); nusselt_wakao_kaguei(Re_p, Pr);
nusselt_reynolds2023(Re, Pr)
check_range(key, re=, porosity=, pr=, w=, on_extrapolation=) ; range_report(...) -> fractions outside each bound
LITERATURE[key] -> CorrelationInfo(.citation .doi .tpms .method .fluid .re_definition .formula .validity ...)
literature_table(implemented_only=False) -> DataFrame ; RECOMMENDED (dict)
GAJETTI2025_POWER_LAWS, GAJETTI2025_TABLE2, SAVOLDI2026_HYDRAULICS, SAVOLDI2026_STANTON, RATHORE2023_CF
```

**Literature table** (coefficients read from the open-access full texts unless noted; full table with methods in
`docs/task8_literature_closures.md`, machine-readable in `results/task8_literature_table.csv`)

| Source | TPMS | Re; φ; Pr | Formula | Obtained by |
|---|---|---|---|---|
| **Gajetti et al. 2025** [1] | G, D, Split-P, skeletal | Re_Dh 0.3–100; φ 0.30–0.60 | K/L² = aφⁿ, C_F = bφᵐ (G: 0.0156, 2.78, 0.0908, −1.81; D: 0.0106, 2.89, 0.0666, −1.69) | OpenFOAM, periodic cell, water |
| Savoldi et al. 2026 [2] hydraulics | G, skeletal | Re_Dh 20–100; φ 0.3–0.7 | D_h/L = 4φ/(−5.723φ² + 5.729φ + 1.665); K/L² = 0.0157φ^2.72; C_F = 0.0784φ^−2.04 | same group |
| Rathore et al. 2023 [3] | D, G, I-WP, P | Re ≤ 100 (channel length); φ = 0.32 | data: C_F 1.173 (G), 0.525 (D) | DNS of a 4-cell channel |
| **Savoldi et al. 2026** [2] Eq. 37 | G, skeletal, interstitial | Re_Dh 20–100; φ 0.3–0.7; **Pr = 1** | St = 0.0267 φ^0.19 f (±11 %) | CFD, uniform wall T and q |
| Savoldi et al. 2026 [2] Eqs. 38–39 | as above | as above | St = 0.135 φ^2.06 Re^(0.440−φ) f (±15 %) | as above |
| Savoldi et al. 2026 [2] Eq. 40 | as above | as above | St = 0.034 φ^0.20 Re^(−0.061−0.0023φ) f (±12 %) | as above |
| Cheng et al. 2021 [4] | W, P, D, G porous media | Re_h 10–129; ε 0.2–0.8; air | Nu(Re_h, ε), flow resistance — **coefficients not accessed** | pore-scale CFD |
| Iyer et al. 2022 [5] | sheet TPMS / nodal surfaces | laminar, Re ≲ 350 | **not accessed** | CFD |
| Reynolds et al. 2023 [6] | gyroid HX | Re_Dh 100–2500; air | Nu = 0.49 Re^0.62 Pr^0.4 (quoted by [7]) | experiment |
| Brambati et al. 2024 [7] | G, P, D sheet, φ 0.7–0.9 | Re 5000–50000; Pr 0.7–7 | Nu_Dh = 0.0964 Re_Dp^0.7136 Pr^0.4 — turbulent, not applicable | conjugate RANS |
| Piandoro et al. 2026 [8] | gyroid sheet | EG (Pr ≈ 154), EG/water Pr 14 | Nu = (23.44 + 7.5φ)√Re_por, f ≈ 1 + 1/Re_por (wall-to-bulk h of a filled duct) | CFD + L-PBF |
| Gnielinski [9] | packed spheres | Re_p 0.1–1000; Pr 0.4–1000 | Nu_p = (1 + 1.5(1−φ))(2 + √(Nu_lam² + Nu_turb²)) | experiments |
| Wakao & Kaguei [10] | packed spheres | Re_p 3–3000 | Nu_p = 2 + 1.1 Re_p^0.6 Pr^(1/3) | experiments |
| Ergun [11] (via [1]) | packed beds | — | 150/Re* + 1.75 — rejected by [1] (30–50 % errors for TPMS) | experiments |
| Kuwahara et al. 2001 [12] | square-rod arrays | φ 0.2–0.9; Pr = 1 | (porosity term) + (Re term)Pr^(1/3) — coefficients not verified, not implemented | CFD |

**Recommended closures (`RECOMMENDED`)**
1. **C_F: Gajetti et al. 2025 power laws** (gyroid, diamond; blends ln C_F linear in w). Reasons: the only source with
   G *and* D from one method (consistent blend interpolation); skeletal cells like ours; Re range from the Darcy regime
   (0.3) separates K and C_F cleanly; the gyroid refit of Savoldi et al. 2026 (to φ = 0.7) stays within 10 % over
   φ = 0.5–0.8, which bounds the gyroid extrapolation. `source="gajetti2025_table"` (Table 2 held constant outside
   φ = 0.3–0.6) is the upper bracket for diamond.
2. **Nu: Savoldi et al. 2026, variant 1 (Eq. 37) × Pr^(1/3)**, with f from our bulk K, our a_sf and the Gajetti C_F:
   `Nu = 0.0267 φ^0.19 (f Re) Pr^(1/3)`. Reasons: the only laminar, skeletal, porosity-resolved interstitial
   correlation we could verify; variant 1 extrapolates physically (constant Nu as Re → 0, Forchheimer-controlled growth
   at high Re, monotone over the whole jacket box), whereas variant 2 makes Nu *fall* with Re outside its box
   (φ = 0.8, Pr = 1: 10.0 → 7.4 between Re 10 and 50) and is 0.33–0.94× variant 1 at Re = 600; with Pr^(1/3) variant 1
   stays within **0.63–1.37×** of the Pr-validated Gnielinski packed-bed correlation over the box (variant 2: down to 0.27×).
   Price: inside the fitted box variant 2 has more symmetric residuals and is up to 22 % lower at its high-φ/high-Re
   corner → **run variant 2 as the low-h bound in Task 9/11 sensitivity runs**.

**Independent verification of our solvers against the published CFD (tests)**
| Quantity | Literature | VoxLat (Task 1/4) | Difference |
|---|---|---|---|
| K/L², gyroid φ = 0.5 / 0.6 | 2.2e-3 / 3.9e-3 ([1] Table 2) | 2.224e-3 / 3.829e-3 | +1.1 % / −1.8 % |
| K/L², diamond φ = 0.5 / 0.6 | 1.4e-3 / 2.4e-3 | 1.410e-3 / 2.390e-3 | +0.7 % / −0.4 % |
| D_h/L, gyroid φ = 0.5–0.7 | fit of [2] | 4φ/(a_sf L) | ≤ 0.3 % |
| K/L², gyroid φ = 0.5–0.7 vs fit of [2] | | | −0.5 … +7 % inside; their power law is 17 % low at φ = 0.8 (outside its range) → use our K, never literature K fits |
Body-fitted OpenFOAM vs our voxel MAC solver agree to the 2-digit precision of their table: a citable validation of Task 4.

**Jacket operating box** (`results/task8_operating_range.csv`; w ∈ {0, 0.5, 1}, ρ* 0.20–0.50, L = 2/4/6 mm, 1–5 L/min,
mean superficial velocity U_s = Q/(2 h L_ax) of the two half-annulus paths)
| Quantity | Range (median) |
|---|---|
| Re_Dh | 25 – 577 (153); 70 % of points > 100 |
| Re_K | 0.9 – 37 (7.4) |
| C_F | 0.097 – 0.32 (0.165) |
| Fo = C_F Re_K | 0.19 – 5.4 → **inertia is 16 – 84 % of the pressure drop** (median 56 %) |
| Nu_Dh (variant 1, Pr = 20.7) | 15 – 105 (32) |
| h_sf | 1.7 – 11 kW/(m² K) (4.9); h_sf a_sf 0.7 – 20 MW/(m³ K) |
| Nominal (3 L/min, L = 4 mm, ρ* = 0.35) G / blend / D | Re_Dh 195 / 178 / 158; Fo 2.0 / 1.5 / 1.1; Nu 41 / 35 / 30 (variant 2: 31 / 27 / 23); h_sf 4.7 / 4.4 / 4.2 kW/(m² K) |

**Extrapolation risk (flag in the paper)**
1. **Pr**: the Nu data are at Pr = 1; Pr^(1/3) multiplies h_sf by 2.75 at Pr = 20.7. Exponent 0.4 (Brambati et al.'s turbulent
   TPMS fit 0.39–0.40; Reynolds et al.'s assumption) gives +22 %. Only TPMS data near our Pr: Piandoro et al. (sheet gyroid,
   Pr 14–154, wall-based h, no Pr exponent). Cross-check: Gnielinski (valid Pr 0.4–1000) 0.63–1.37×.
2. **Re**: both closures fitted to Re_Dh ≤ 100; 70 % of operating points lie above (max 577). C_F is assumed constant
   (strong-inertia form) beyond the weak-inertia range it was fitted in; variants 1 and 2 differ by up to 3× at Re = 600, φ = 0.8.
3. **Porosity**: C_F fitted to φ ≤ 0.6 (67 % of the operating box is above), Nu to φ ≤ 0.7 (38 % above). Diamond C_F at
   φ > 0.6: power law vs Table-2 plateau differ 1.2–2.0× → pump-power uncertainty of the same order on the inertial share.
4. **Topology**: Nu only calibrated for the gyroid; diamond and blends via the analogy with their own f. No C_F or Nu data
   for G–D blends at all (interpolation).
5. **Single source**: [1] and [2] come from one group with one CFD set-up; the only independent C_F (Rathore et al., φ = 0.32,
   4-cell channel with walls) is 1.15× (D) and 1.64× (G) of [1].
6. **Finite gap and roughness**: literature closures are bulk; the Task 7 wall correction is for creeping-flow K only. Savoldi
   et al. note systematically higher measured f for printed gyroids (roughness, finite size; Hirokawa & Miyata 2024).

**Notes for Task 9**
1. Use the **bulk** K (`ClosureModel`, not wall-corrected) in f Re, in Fo and in the inertial term ρC_F|U|U/√K; apply the
   Task 7 factor (1 − a_K/N) to the viscous Darcy term only. For anisotropic K use K_e = 1/(e·K⁻¹·e) along the local
   velocity direction e.
2. `interstitial_heat_transfer` handles U = 0 (Darcy-limit Nu, variant 1 only) and vectorized grids. Expect
   `CorrelationRangeWarning`s (Re, Pr, porosity, topology) on every evaluation: log `range_report(...)` once per run and
   then pass `on_extrapolation="ignore"`.
3. Sensitivity set for the paper: {variant 1, variant 2} × {Pr^(1/3), Pr^0.4} × {C_F power law, Table-2 plateau}.
4. h_sf is the fluid-side coefficient at a uniform-T/uniform-q surface (non-conjugate CFD); strut conduction (fin
   efficiency) must come from the solid-phase equation of the two-equation model with k_eff (Task 2/6).

**Manuscript limitations paragraph (~150 words)** — also in `docs/task8_literature_closures.md`:

> Two closures of the device model are taken from the literature. The inertial (Forchheimer) coefficient comes from
> pore-scale simulations of skeletal gyroid and diamond cells fitted for porosities of 0.3–0.6 and Re_Dh ≤ 100 (Gajetti
> et al., 2025). Our jacket operates at porosities of 0.5–0.8 and Re_Dh ≈ 25–580, where inertia carries 16–84 % of the
> pressure drop, so pumping-power predictions inherit this extrapolation; for diamond above φ = 0.6, plausible C_F values
> differ by up to a factor of two. The interstitial heat-transfer coefficient follows the modified Reynolds analogy of
> Savoldi et al. (2026), calibrated for the gyroid at Pr = 1, φ ≤ 0.7 and Re_Dh = 20–100. We extend it to the water–glycol
> coolant (Pr ≈ 21) with the Chilton–Colburn factor Pr^(1/3), and to diamond and blends through their own friction
> factors; an exponent of 0.4 would raise h_sf by 22 %. Removing these assumptions requires experiments or conjugate
> pore-resolved simulations at the operating Prandtl number.

**References** (open access = read in full; "to verify" = bibliographic details not confirmed from a primary page)
1. Gajetti E., Boccardo G., Savoldi L., Marocco L. (2025). Hydrodynamic characterization of Gyroid, Diamond and Split-P
   Triply Periodic Minimal Surfaces as porous medium. *Int. J. Heat Mass Transf.* 252, 127439.
   doi:10.1016/j.ijheatmasstransfer.2025.127439 (open access: re.public.polimi.it/handle/11311/1293551).
2. Savoldi L., Cammi A., Gajetti E., Marocco L. (2026). A modified Reynolds analogy for the Gyroid TPMS in laminar flow:
   Thermal-hydraulic correlations, numerical validation and multi-objective optimisation. *Int. J. Heat Fluid Flow* 121,
   110631. doi:10.1016/j.ijheatfluidflow.2026.110631 (open access: re.public.polimi.it/handle/11311/1324787).
   Text/caption mismatches in the paper: Eq. 38 coefficient 0.135 (Fig. 15: 0.134), Eq. 40 0.034 (Fig. 16: 0.0342); we use the equations.
3. Rathore S.S., Mehta B., Kumar P., Asfer M. (2023). Flow characterization in triply periodic minimal surface (TPMS)-based
   porous geometries: Part 1 — Hydrodynamics. *Transp. Porous Media* 146, 669–701. doi:10.1007/s11242-022-01880-7 (arXiv:2205.03591).
4. Cheng Z., Li X., Xu R., Jiang P. (2021). Investigations on porous media customized by triply periodic minimal surface:
   Heat transfer correlations and strength performance. *Int. Commun. Heat Mass Transf.* 129, 105713.
   doi:10.1016/j.icheatmasstransfer.2021.105713. Companion: Cheng Z., Xu R., Jiang P.-X. (2021) *Int. J. Heat Mass Transf.*
   170, 120902, doi:10.1016/j.ijheatmasstransfer.2021.120902.
5. Iyer J., Moore T., Nguyen D., Roy P., Stolaroff J. (2022). Heat transfer and pressure drop characteristics of heat
   exchangers based on triply periodic minimal and periodic nodal surfaces. *Appl. Therm. Eng.* 209, 118192 (to verify).
6. Reynolds B.W., Fee C.J., Morison K.R., Holland D.J. (2023). Characterisation of heat transfer within 3D printed TPMS
   heat exchangers. *Int. J. Heat Mass Transf.* (volume/article to verify; correlation taken from [7], Eq. 21).
7. Brambati G., Guilizzoni M., Foletti S. (2024). Convective heat transfer correlations for Triply Periodic Minimal
   Surfaces based heat exchangers. *Appl. Therm. Eng.* 242, 122492. doi:10.1016/j.applthermaleng.2024.122492 (open access).
8. Piandoro S., Azzini F., Francioso M., Zha D., Liverani E., Pulvirenti B., Fortunato A. (2026). Laser 3D-printed periodic
   porous structures for heat exchangers: a novel characterization approach under fully developed conditions.
   *Lasers Manuf. Mater. Process.* 13, 555–581. doi:10.1007/s40516-026-00347-7 (open access).
9. Gnielinski V. (1981). *Int. Chem. Eng.* 21(3); Gnielinski V. (1982) *Verfahrenstechnik* 16(1); VDI Heat Atlas, 2nd ed.
   (2010). Implementation and example value as in the `ht` Python library (C. Bell, `ht.conv_packed_bed`).
10. Wakao N., Kaguei S. (1982). *Heat and Mass Transfer in Packed Beds*. Gordon & Breach (Taylor & Francis).
11. Ergun S. (1952). Fluid flow through packed columns. *Chem. Eng. Prog.* 48, 89–94.
12. Kuwahara F., Shirota M., Nakayama A. (2001). A numerical study of interfacial convective heat transfer coefficient in
    two-energy equation model for convection in porous media. *Int. J. Heat Mass Transf.* 44, 1153–1159.
13. Hirokawa & Miyata (2024), cited by [2] for measured gyroid friction factors (details to verify).
14. Al-Safadi M., Ejaz F., Shuja S.Z., Zubair S.M. (2026). Quantitative synthesis and correlations of heat transfer and
    friction factor in TPMS structures. *Results Eng.* 29, 109402. doi:10.1016/j.rineng.2026.109402 — open-access
    meta-analysis of 57 studies with unified Nu/f correlations; **not read** (the publisher site blocked automated access).
    Worth reading before the manuscript; it may supersede rows of the table.

**Open issues**
1. Coefficients of Cheng et al. 2021 (closest scope: porous TPMS, ε 0.2–0.8, Re_h 10–129) and Iyer et al. 2022 were not
   accessible here; if you can download them (NUST library), add them to `LITERATURE` and the comparison figure.
2. Read Al-Safadi et al. 2026 [14] (open access) and check whether its unified laminar correlations change the recommendation.
3. A cheap way to retire the Pr risk later: a periodic-cell conjugate check with our own Stokes velocity field and an
   advection–diffusion solve at Pr = 1 and 20 (would measure the Pr exponent for our geometries; outside the Task 8 scope).
4. Workspace: PyPI blocked; tests ran on numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, Python 3.13 with pytest from a uv tool env.

**Figures** (`results/figures/`): `task8_forchheimer.png` ((a) C_F vs φ — Gajetti fits with Table 2 points, Savoldi gyroid
fit, Rathore points, diamond plateau, jacket band; (b) our Task 4 K vs the published K fits; (c) inertial share of the pressure
drop vs flow rate, ρ* = 0.35, L = 2/4/6 mm), `task8_nusselt.png` (Nu_Dh vs Re_Dh at Pr = 20.7 for G φ = 0.65, G φ = 0.80,
D φ = 0.65: variants 1–3 × Pr^(1/3), the Pr^0.4 band, Gnielinski and Wakao–Kaguei; fitted box and jacket range marked).

**Tests**: `tests/test_empirical.py` 26 passed in 0.3 s; full suite `-m "not slow"` 371 passed + 4 skipped (pyamg ×2,
pyarrow ×2) in 1.8 min (`results/task8_pytest_log.txt`). The 18 slow tests were not re-run: Task 8 adds no solver code
and only appends imports to `voxlat.closures`.

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
