# VoxLat — STATUS (hand-off log)

Every task appends here: what was built, public API, key numbers (resolution,
timings, errors), open issues. **The next task reads this file first.**
Plan: project doc `claude/voxlat_task_plan.md`. Reference numbers: `configs/reference.yaml`.

| # | Task | State |
|---|---|---|
| 0 | Repo scaffold + config | **done** (2026-10-07) |
| 1 | TPMS unit-cell geometry | not started |
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

## Task 1 — TPMS unit-cell geometry
_not started_

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
