# VoxLat — STATUS (hand-off log)

Every task appends here: what was built, public API, key numbers (resolution,
timings, errors), open issues. **The next task reads this file first.**
Plan: project doc `claude/voxlat_task_plan.md`. Reference numbers: `configs/reference.yaml`.

| # | Task | State |
|---|---|---|
| 0 | Repo scaffold + config | **done** (2026-10-07) |
| 1 | TPMS unit-cell geometry | **done** (2026-10-08) |
| 2 | Conduction homogenization k_eff | **done** (2026-10-08) |
| 3 | Elasticity homogenization C_eff | not started |
| 4 | Stokes permeability K | not started |
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

## Task 3 — Elasticity homogenization
_not started_

## Task 4 — Stokes permeability
_not started_

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
