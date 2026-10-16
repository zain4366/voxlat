# VoxLat: Task Plan (laptop-only, code-only)

Working name **VoxLat** (voxel lattices).

---

## 1. Scope

**What the project is:** a pure-Python computational pipeline, runnable on a normal laptop CPU. There is no CFD cluster, no GPU requirement and no 3D printing.

**Demonstrator:** a liquid-cooling jacket around the stator of a 15 kW inrunner electric motor. The jacket is filled with a spatially graded TPMS lattice (gyroid / diamond / blends) that both cools the motor and carries its loads.

**What we compute ourselves (exactly, on voxels):**
- geometry metrics: density, surface area, wall/pore sizes, connectivity
- effective conductivity tensor k_eff (periodic voxel finite-volume)
- effective stiffness tensor C_eff and stress concentration (periodic voxel FEA)
- permeability tensor K (periodic voxel Stokes solver)
- **finite-gap corrections**: how these properties change when only 1–8 cells fit across the gap (the core research result)

**What we take from literature (stated as a limitation):**
- interstitial heat-transfer coefficient Nu(Re, Pr) for TPMS
- inertial (Forchheimer) coefficient

**Research questions:**
- **RQ1:** How large is the error of bulk homogenized properties when only N = 1–8 cells fit across a bounded gap? Can a simple learned correction remove it?
- **RQ2:** Under identical constraints, does multi-morphology grading (density + type + cell size) beat a helical channel, a uniform gyroid, and thickness-only grading?
- **RQ3:** How sample-efficient is Bayesian multi-objective optimization compared with NSGA-II?

---

## 2. Reference problem (single source of truth: `configs/reference.yaml`)

| Item | Value |
|---|---|
| Motor | 15 kW continuous inrunner PMSM, η = 0.94 |
| Heat into jacket | Q = 770 W |
| Stator / sleeve inner radius | 70 mm; sleeve thickness 1.5 mm |
| Lattice gap | h = 6 mm (r = 71.5 → 77.5 mm); outer wall 2 mm |
| Axial length | 50 mm |
| Heat-flux profile | q''(z) = q̄·[1 + 0.25·(12(ζ−½)² − 1)], ζ = z/50 mm, q̄ = Q/(2π·0.070·0.050) ≈ 35 kW/m² (ends 2× centre) |
| Coolant (50/50 water–glycol) | ρ = 1060 kg/m³, cp = 3400 J/kg·K, k = 0.40 W/m·K, ν = 2.3e-6 m²/s |
| Inlet conditions | 50 °C, nominal 3 L/min (sweep 1–5 L/min); pump efficiency 0.3 |
| Manifolds | inlet at θ = 0°, outlet at θ = 180° (two parallel paths), each 15 mm wide over full length (configurable) |
| Material (AlSi10Mg) | E = 70 GPa, ν = 0.33, ρ = 2670 kg/m³, k = 130 W/m·K, σ_allow = 80 MPa (fatigue-limited assumption) |
| Loads | torque 36 N·m, thrust 780 N, coolant pressure 2 bar(g) |
| Design bounds | ρ* (relative density) 0.20–0.50; w (gyroid→diamond blend) 0–1; cell size L 2–6 mm; axial stretch a_z 0.7–1.5 |
| Manufacturability | min wall 0.35 mm, min pore 0.8 mm, limit on grading gradient |

---

## 3. Conventions (every task follows these)

- **Environment:** Python ≥ 3.11, Windows-friendly, CPU only. Package `voxlat` in `src/` layout.
- **Core dependencies:** numpy, scipy, scikit-image, scikit-learn, pandas, pyarrow, matplotlib, pyyaml, joblib, tqdm, pytest.
- **Optional extras:** `ml` (torch), `opt` (pymoo, botorch).
- **Units:** SI inside the code. Unit cells use normalized coordinates [0,1)³, scaled by physical cell size L. Voxel arrays are `bool[nx, ny, nz]`, True = solid, index order (x, y, z).
- **Code style:** every solver function has a docstring stating the governing equations and boundary conditions.
- **Tests:** every physics module ships pytest verification tests against analytic or published cases. Write the tests before tuning.
- **Data:** tables in `data/*.parquet`; each record carries `run_record()` metadata (params, version, git hash, resolution, wall time).
- **Figures:** `results/figures/*.png` at 300 dpi, via `voxlat.utils.save_figure`. Each figure has a script in `scripts/`.
- **Reproducibility:** deterministic seeds everywhere.
- **`STATUS.md` in the repo root is the hand-off file.** Every task appends: what was built, public API, key numbers (chosen resolution, timings, errors), open issues.

---

## 4. Task overview

| # | Task | Depends on | Laptop time |
|---|---|---|---|
| 0 | Repo scaffold + config | – | minutes |
| 1 | TPMS unit-cell geometry | 0 | minutes |
| 2 | Conduction homogenization k_eff | 1 | minutes |
| 3 | Elasticity homogenization C_eff (voxel FEA) | 1 | ~10 min |
| 4 | Stokes permeability K | 1 | ~10 min |
| 5 | Closure dataset generation | 2, 3, 4 | overnight |
| 6 | Closure surrogates (GP / ensemble / 3D CNN) | 5 | ~1 h |
| 7 | **Finite-gap study (RQ1)** | 2, 3, 4 | a few hours |
| 8 | Literature closures for Nu and Forchheimer | – | – |
| 9 | Homogenized jacket device model | 6, 7, 8 | minutes |
| 10 | Baselines B1–B3 | 9 | minutes |
| 11 | Multi-objective optimization (RQ2, RQ3) | 9, 10 | a few hours |
| 12 | PicoGK geometry export of the optimum | 11 | ~30 min |
| 13 | Reproducibility pass + figure set | all | ~1 day of runs |
| 14 | Manuscript draft (+ JOSS paper) | 13 | – |
