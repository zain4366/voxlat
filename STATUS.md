# VoxLat — STATUS (hand-off log)

Every task appends here: what was built, public API, key numbers (resolution,
timings, errors), open issues. **The next task reads this file first.**
Plan: project doc `claude/voxlat_task_plan.md`. Reference numbers: `configs/reference.yaml`.

| # | Task | State |
|---|---|---|
| 0 | Repo scaffold + config | **done** (2026-10-07) |
| 1 | TPMS unit-cell geometry | **done** (2026-10-08) |
| 2 | Conduction homogenization k_eff | not started |
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

## Task 2 — Conduction homogenization
_not started_

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
