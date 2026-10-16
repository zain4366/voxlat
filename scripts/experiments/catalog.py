"""Catalog of every paper figure and table: which script makes it, from what, how long it takes.

Two kinds of entries:

* ``Step``  - one producer script in ``scripts/`` (most produce several figures/tables at once), with the
              command for each reproduction mode and a laptop runtime estimate:
                full   - recompute everything from scratch (the numbers in STATUS.md)
                replot - regenerate figures/tables from the data committed in data/, results/, models/
                quick  - plumbing run: tiny resolutions/budgets, proves the code path works (minutes)
              ``None`` for a mode means "not available in that mode" (the committed output is used / skipped).
* ``Item``  - one paper figure (``Fxx``) or table (``Txx``): output file, caption, the step that makes it.

``scripts/experiments/<id>_<name>.py`` runs one item; ``scripts/experiments/run_all.py`` runs them all.
Runtimes are for the Windows laptop / a 2-4 core CPU (from STATUS.md); quick-mode times were measured in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Step", "Item", "STEPS", "ITEMS", "step", "item", "PAPER_DESIGN"]

#: manufacturable slots optimum used in the paper (Task 12 finding 1; user decision for Task 13)
PAPER_DESIGN = "results/designs/task12_slots_best_at_cap_gyroid.json"


@dataclass(frozen=True)
class Step:
    id: str
    script: str                      # path relative to the repo root
    full: tuple[str, ...] | None     # CLI args for a from-scratch run
    quick: tuple[str, ...] | None    # CLI args for the plumbing run
    replot: tuple[str, ...] | None   # CLI args that only re-plot / re-tabulate committed data
    est_full: str                    # human-readable laptop estimate
    est_full_s: float                # same in seconds (for the total)
    needs: tuple[str, ...] = ()      # importable modules required (extras: torch, pymoo, botorch, pyarrow, ...)
    note: str = ""
    manual: bool = False             # needs the Windows laptop (.NET + PicoGK); never run automatically
    replot_needs: tuple[str, ...] | None = None   # modules the replot mode needs (None = same as ``needs``)
    deps: tuple[str, ...] = ()       # upstream steps whose outputs this step reads (full/quick runs include them)
    quick_last: bool = False         # quick run overwrites package tables -> run it after everything else


@dataclass(frozen=True)
class Item:
    id: str
    name: str                        # short slug, also the experiment script name
    kind: str                        # "figure" | "table"
    path: str                        # output file relative to the repo root
    step: str                        # Step.id that writes it ("tables" = scripts/experiments/tables.py)
    section: str                     # paper section
    caption: str
    extra: tuple[str, ...] = field(default=())   # sibling files (panels, csv data)


PY = "scripts/"

STEPS: list[Step] = [
    Step("task0", PY + "task0_heat_flux_profile.py", (), (), (), "5 s", 5),
    Step("task1_tables", PY + "task1_build_tables.py", ("--jobs", "-1"), ("--n", "16", "--thresholds-only"), None,
         "~10 min (1 core)", 600, note="rebuilds the TPMS tables shipped in src/voxlat/geometry/data", quick_last=True),
    Step("task1_figs", PY + "task1_tpms_figures.py", (), (), (), "1-2 min", 90, deps=('task1_tables',)),
    Step("task2_conv", PY + "task2_convergence.py", (), ("--quick",), None, "~15 min (2 cores)", 900),
    Step("task2_keff", PY + "task2_keff_vs_density.py", (), ("--quick",), None, "~5 min (2 cores)", 300),
    Step("task3_conv", PY + "task3_convergence.py", (), ("--quick",), None, "~14 min (2 cores)", 840),
    Step("task3_anchor", PY + "task3_n96_anchor.py", (), None, None, "~16 min (1 core, 1.4 GB)", 960,
         note="n = 96 anchors of the convergence fit; no quick mode"),
    Step("task3_stiff", PY + "task3_stiffness_vs_density.py", (), ("--quick",), None, "~16 min (2 cores)", 960),
    Step("task4_verif", PY + "task4_verification.py", (), ("--quick",), ("--replot",), "~2 min (2 cores)", 120),
    Step("task4_conv", PY + "task4_convergence.py", (), ("--quick", "--no-anchor"), ("--replot",),
         "~11 min (2 cores, 1.5 GB)", 660),
    Step("task4_perm", PY + "task4_permeability_vs_porosity.py", (), ("--quick",), None, "~9 min (2 cores)", 540),
    Step("task4_bench", PY + "task4_benchmark_preconditioner.py", (), ("--quick",), None, "3-5 min", 240,
         note="Jacobi vs AMG timing (supplementary, not a paper figure)"),
    Step("task5_dataset", PY + "build_dataset.py", ("--skip-timing",),
         ("--quick", "--n-jobs", "1", "--skip-timing", "--no-plot"), ("--plot-only",),
         "~7.4 h (laptop, 4 workers; 27.8 CPU-h)", 7.4 * 3600, needs=("pyarrow",),
         note="Task 5 closure dataset data/closures.parquet (resumable)"),
    Step("task6_train", PY + "task6_train_surrogates.py", ("--cnn",), ("--quick", "--out-dir", "models_quick"), None,
         "~22 min (GP + ensemble 7 min, CNN 15 min)", 1320, needs=("pyarrow", "torch"), deps=('task5_dataset',)),
    Step("task6_eval", PY + "task6_evaluate.py", (), ("--quick",), ("--replot",), "~10 min (2 cores)", 600,
         needs=("pyarrow", "torch"), replot_needs=(), deps=('task5_dataset',)),
    Step("task6_cnn", PY + "task6_cnn_experiment.py", (), ("--quick",), ("--replot",), "~80 min (2 cores)", 4800,
         needs=("pyarrow", "torch"), replot_needs=(), deps=('task5_dataset',)),
    Step("task7_strips", PY + "task7_finite_gap.py", ("--suite", "all", "--n-jobs", "2"), ("--quick",), None,
         "~1.5 h (n = 32, 2 cores)", 5400, note="writes data/finite_gap/ and the table data/finite_gap.csv"),
    Step("task7_fit", PY + "task7_fit_discrepancy.py", (), ("--no-gp",), (), "~3 min", 180, deps=('task7_strips',)),
    Step("task7_figs", PY + "task7_figures.py", (), ("--velocity-n", "16"), (), "~4 min (velocity slices n = 48)",
         240, deps=('task7_fit',)),
    Step("task8", PY + "task8_literature_closures.py", (), (), (), "5 s", 5, deps=('task4_perm',)),
    Step("task9", PY + "task9_jacket_model.py", (), ("--quick",), (), "~20 s", 20, deps=('task6_train', 'task7_fit', 'task8')),
    Step("task10", PY + "task10_baselines.py", (), ("--quick",), (), "~2.5 min", 150, deps=('task9',)),
    Step("task11_table", PY + "task11_fine_morphology_table.py", (), None, None, "~30 min (2 cores)", 1800,
         note="fine-w TPMS morphology table used by the corrected cell-size rule; no quick mode"),
    Step("task11_opt", PY + "task11_optimize.py", ("--parallel", "4", "--skip-done"), ("--quick",), None,
         "~75 CPU-h: ~20 h on the laptop at 4 parallel runs (2 s/evaluation); 2 h 35 min on 30 GitHub "
         "runners (.github/workflows/task11-study.yml)", 20 * 3600, needs=("pymoo", "botorch"),
         note="30 runs = 2 layouts x {NSGA-II 10 000, qNEHVI 200, Sobol 1000} x 5 seeds", deps=('task11_table', 'task9')),
    Step("task11_refs", PY + "task11_uniform_references.py", (), (), (), "~2-6 min", 240, deps=('task11_table',)),
    Step("task11_analysis", PY + "task11_analysis.py", (), ("--no-designs",), (), "~5 min", 300,
         note="reads the committed runs in results/task11/", deps=('task11_opt', 'task11_refs', 'task10')),
    Step("task12_numpy", PY + "task12_export.py",
         ("numpy", "--design", PAPER_DESIGN, "--voxel", "0.06"),
         ("numpy", "--design", PAPER_DESIGN, "--voxel", "0.3", "--angle", "10", "--out", "results/export_quick"), None,
         "~15 min, 4-6 GB RAM (30 deg sector at 0.06 mm)", 900),
    Step("task12_figs", PY + "task12_figures.py", (), (), (), "~3 min", 180, deps=('task12_numpy',)),
    Step("task12_render", PY + "task12_render.py", (), ("--quick",), ("--quick",), "~2 min at 1800 px", 120,
         note="replot uses --quick (low resolution); the committed renders are full resolution"),
    Step("task12_picogk", PY + "task12_export.py",
         ("build", "--design", PAPER_DESIGN, "--mode", "sector"), None, None,
         "~15 min PicoGK + ~18 min checks (sector 0.05 mm, 16 GB RAM); full jacket 0.2 mm ~3 min",
         2000, manual=True,
         note="Windows + .NET 9 SDK + PicoGK 1.7; see README 'PicoGK geometry export (Task 12)'"),
    Step("tables", "scripts/experiments/tables.py", ("all",), ("all",), ("all",), "< 10 s", 10,
         note="paper tables T01-T09 from the committed result CSVs/JSONs"),
]

_STEP_INDEX = {s.id: s for s in STEPS}


def step(step_id: str) -> Step:
    return _STEP_INDEX[step_id]


F = "results/figures/"
T = "results/tables/"

ITEMS: list[Item] = [
    # ---------------------------------------------------------------- problem and geometry
    Item("F01", "heat_flux_profile", "figure", F + "task0_heat_flux_profile.png", "task0", "Reference problem",
         "Axial stator heat-flux profile q''(z) of the reference 15 kW motor: mean 35 kW/m^2 (Q = 770 W over the "
         "70 mm x 50 mm sleeve), parabolic with the ends at twice the centre value; integrates to Q within 0.1 %."),
    Item("F02", "tpms_cells", "figure", F + "task1_tpms_cells.png", "task1_figs", "Geometry",
         "Network TPMS unit cells at relative density 0.30: gyroid (w = 0), 50/50 blend (w = 0.5) and Schwarz "
         "diamond (w = 1), rendered and sliced; solid where the blended level set phi_w <= c(w, rho*)."),
    Item("F03", "asf_convergence", "figure", F + "task1_asf_convergence.png", "task1_figs", "Geometry",
         "Grid convergence of the specific surface area a_sf L (marching cubes on the periodic level set) for "
         "n = 24-64 voxels per cell; the gyroid sheet limit approaches the minimal-surface value ~3.09.",
         extra=("results/task1_asf_convergence.csv",)),
    Item("F04", "morphology_vs_density", "figure", F + "task1_morphology_vs_density.png", "task1_figs", "Geometry",
         "Minimum wall thickness, minimum pore size and surface area vs relative density for gyroid, diamond and "
         "blends (n = 48 periodic distance transform). Blends develop pinch-point necks (non-monotonic minimum wall)."),
    Item("F05", "feasible_region", "figure", F + "task1_feasible_region.png", "task1_figs", "Geometry",
         "Manufacturable (rho*, L) region per blend weight w for min wall 0.35 mm and min pore 0.8 mm: the "
         "cell-size floor L_min(w, rho*) used as the optimization constraint."),
    # ---------------------------------------------------------------- homogenization solvers
    Item("F06", "keff_vs_density", "figure", F + "task2_keff_vs_density.png", "task2_keff", "Homogenization",
         "Effective conductivity k_eff/k_s vs relative density for gyroid, diamond and a 50/50 blend "
         "(k_f/k_s = 1/325 and k_f = 0), with Hashin-Shtrikman bounds; panel (b) fraction of the HS upper bound. "
         "Two-grid Richardson estimator R(32, 64).", extra=("results/task2_keff_vs_density.csv",)),
    Item("F07", "keff_convergence", "figure", F + "task2_convergence.png", "task2_conv", "Homogenization",
         "Grid convergence of k_eff: offset-0 grids vs 1/n and offset-averaged error (log-log, slope -1); "
         "basis for the R(32, 64) production estimator.",
         extra=("results/task2_convergence.csv", "results/task2_convergence_summary.csv")),
    Item("F08", "youngs_vs_density", "figure", F + "task3_youngs_vs_density.png", "task3_stiff", "Homogenization",
         "Effective Young's modulus E*/E_s vs relative density (log-log) from periodic voxel FEA, with power-law "
         "fits E*/E_s = C rho*^m, the HS bound, a slope-2 guide and published gyroid/diamond data (Maskery et al.); "
         "blend shown as its directional range.", extra=("results/task3_stiffness_vs_density.csv",
                                                       "results/task3_powerlaw_fits.csv")),
    Item("F09", "anisotropy_vs_density", "figure", F + "task3_anisotropy_vs_density.png", "task3_stiff",
         "Homogenization", "Elastic anisotropy vs relative density: Zener ratio of the cubic gyroid and diamond and "
         "the universal anisotropy index A^U (including the trigonal blend)."),
    Item("F10", "localization_vs_density", "figure", F + "task3_localization_vs_density.png", "task3_stiff",
         "Homogenization", "Stress localization factors (99th percentile, maximum and mean von Mises over the "
         "macroscopic stress) under unit uniaxial-z and shear-xz stress vs relative density (n = 64)."),
    Item("F11", "stiffness_convergence", "figure", F + "task3_convergence.png", "task3_conv", "Homogenization",
         "Grid convergence of C_eff and stress localization (n = 16-64, 4 grid offsets each; n = 96 anchors): "
         "systematic staircase error vs alignment noise.",
         extra=("results/task3_convergence.csv", "results/task3_convergence_summary.csv",
                "results/task3_n96_anchor.csv")),
    Item("F12", "stokes_verification", "figure", F + "task4_verification.png", "task4_verif", "Homogenization",
         "Verification of the MAC Stokes solver: (a) plane Poiseuille profile and 2/H^2 error, (b) square / "
         "rectangular ducts and inclined slits vs the series solution, (c) simple-cubic sphere arrays vs Zick & "
         "Homsy (1982) and Sangani & Acrivos, (d) sphere error vs resolution.",
         extra=("results/task4_verification_channels.csv", "results/task4_verification_spheres.csv")),
    Item("F13", "permeability_convergence", "figure", F + "task4_convergence.png", "task4_conv", "Homogenization",
         "Grid convergence of the permeability K(n)/K_ref vs 1/n with n = 96 anchors, estimator errors and "
         "tortuosity.", extra=("results/task4_convergence.csv", "results/task4_convergence_summary.csv")),
    Item("F14", "permeability_vs_porosity", "figure", F + "task4_permeability_vs_porosity.png", "task4_perm",
         "Homogenization", "Permeability K/L^2 vs porosity for gyroid, diamond and the 50/50 blend (principal-value "
         "band), Kozeny constant and tortuosity; R(32, 64) estimator.",
         extra=("results/task4_permeability_vs_porosity.csv",)),
    # ---------------------------------------------------------------- closures and surrogates
    Item("F15", "closures_quicklook", "figure", F + "task5_closures_quicklook.png", "task5_dataset", "Closures",
         "Closure dataset (Task 5; 290 design points theta = (w, rho*, a_z), Sobol + boundary points): every "
         "closure vs relative density, coloured by blend weight w.", extra=("data/closures.parquet",)),
    Item("F16", "surrogate_parity_cv", "figure", F + "task6_parity_cv.png", "task6_eval", "Closures",
         "Five-fold cross-validated parity of the GP and deep-ensemble closure surrogates with recalibrated 95 % "
         "intervals.", extra=("results/task6_metrics.csv", "results/task6_predictions.csv.gz")),
    Item("F17", "surrogate_parity_band", "figure", F + "task6_parity_band.png", "task6_eval", "Closures",
         "Parity on the blend-band extrapolation split (training on near-pure morphologies, testing on blends)."),
    Item("F18", "surrogate_calibration", "figure", F + "task6_calibration.png", "task6_eval", "Closures",
         "Calibration of the surrogate uncertainty: observed vs nominal interval coverage, raw and recalibrated, "
         "for the cv5, band and sparse splits."),
    Item("F19", "cnn_extrapolation", "figure", F + "task6_cnn_extrapolation.png", "task6_cnn", "Closures",
         "Geometry-based (3D CNN on 32^3 voxels) vs parameter-based (GP, ensemble) learning of K* and E*: parity "
         "per split and error vs blend weight w.", extra=("results/task6_cnn_metrics.csv",)),
    # ---------------------------------------------------------------- RQ1 finite gap
    Item("F20", "finite_gap_error_vs_N", "figure", F + "task7_error_vs_N.png", "task7_figs", "RQ1 finite gap",
         "Main RQ1 result: relative error delta of the bulk-homogenized prediction vs 1/N (N cells across a "
         "wall-bounded gap) for through-gap conductivity k_n, in-plane permeability K_t, normal stiffness C_nn and "
         "shear moduli G_t, G_sz; gyroid/diamond (top), blend (bottom); bars = range over cut phases, lines = "
         "physical correction delta = a/N; shaded band = jacket design range.",
         extra=("data/finite_gap.csv", "results/task7_wall_coefficients.csv", "results/task7_one_over_N.csv")),
    Item("F21", "finite_gap_models", "figure", F + "task7_discrepancy_models.png", "task7_figs", "RQ1 finite gap",
         "Cross-validated error of the discrepancy models: uncorrected, physical a/N form, physical + GP on the "
         "residuals, and the cut-phase noise floor.", extra=("results/task7_model_cv.csv",)),
    Item("F22", "finite_gap_velocity", "figure", F + "task7_velocity_slices.png", "task7_figs", "RQ1 finite gap",
         "Near-wall channelling: streamwise velocity slices and plane-averaged profiles in strips with N = 2 cells "
         "across the gap (n = 48)."),
    Item("F23", "finite_gap_graded", "figure", F + "task7_graded_error.png", "task7_figs", "RQ1 finite gap",
         "Error of the homogenized prediction in strips with linear density grading vs the grading gradient "
         "|grad rho*| L.", extra=("results/task7_graded.csv",)),
    Item("F24", "finite_gap_convergence", "figure", F + "task7_convergence.png", "task7_figs", "RQ1 finite gap",
         "Resolution and wall-thickness sensitivity of the finite-gap discrepancy.",
         extra=("results/task7_convergence.csv", "results/task7_wall_thickness.csv")),
    Item("F25", "forchheimer", "figure", F + "task8_forchheimer.png", "task8", "Literature closures",
         "Inertial closure: (a) Forchheimer coefficient C_F vs porosity (Gajetti et al. 2025 fits and points, "
         "Savoldi gyroid fit, Rathore points), jacket band; (b) Task 4 permeability vs the published fits; "
         "(c) inertial share of the pressure drop vs flow rate (rho* = 0.35, L = 2/4/6 mm)."),
    Item("F26", "nusselt", "figure", F + "task8_nusselt.png", "task8", "Literature closures",
         "Interstitial Nusselt number Nu_Dh vs Re_Dh at Pr = 20.7 for the candidate TPMS correlations (Pr^(1/3) and "
         "Pr^0.4 scalings) with Gnielinski and Wakao-Kaguei references; fitted box and jacket operating range.",
         extra=("results/task8_nusselt_comparison.csv", "results/task8_operating_range.csv")),
    # ---------------------------------------------------------------- device model and baselines
    Item("F27", "jacket_fields", "figure", F + "task9_fields_uniform_gyroid.png", "task9", "Device model",
         "Homogenized jacket model, uniform gyroid reference (rho* = 0.35, L = 4 mm, 3 L/min): pressure, Darcy "
         "velocity, coolant and stator-side sleeve temperature on the unwrapped (s, z) plane; slots shaded, hot spot "
         "marked.", extra=("results/task9_reference_gyroid.json",)),
    Item("F28", "jacket_verification", "figure", F + "task9_verification.png", "task9", "Device model",
         "Device-model verification: epsilon-NTU convergence against the exact discrete solution, pressure drop vs "
         "the analytic value over 1-5 L/min, and grid convergence of the thermal resistance R.",
         extra=("results/task9_verification.csv", "results/task9_grid_convergence.csv")),
    Item("F29", "jacket_flow_sweep", "figure", F + "task9_flow_sweep.png", "task9", "Device model",
         "Thermal resistance, pressure drop and pump power vs flow rate for uniform gyroid, blend and diamond.",
         extra=("results/task9_flow_sweep.csv",)),
    Item("F30", "baseline_temperature_maps", "figure", F + "task10_temperature_maps.png", "task10", "Baselines",
         "Stator-side sleeve temperature of the baselines at 3 L/min on one colour scale: B1 helical channels "
         "(end-ring manifolds), B1s (slot manifolds), B2 best uniform gyroid, B3 density-only grading."),
    Item("F31", "baseline_fronts", "figure", F + "task10_fronts.png", "task10", "Baselines",
         "Baseline tuning: R and R_active vs pump power for all candidates, the per-family fronts and the selected "
         "designs (pump power <= 0.306 W of the reference gyroid).", extra=("results/task10_fronts.csv",)),
    Item("F32", "baseline_flow_comparison", "figure", F + "task10_flow_comparison.png", "task10", "Baselines",
         "Baselines B1-B3 at 1, 3 and 5 L/min: R, R_active and pump power.", extra=("results/task10_comparison.csv",)),
    # ---------------------------------------------------------------- RQ2 / RQ3 optimization
    Item("F33", "hypervolume", "figure", F + "task11_hypervolume.png", "task11_analysis", "RQ3 optimizers",
         "Sample efficiency (RQ3): normalized hypervolume vs evaluations for NSGA-II, qNEHVI and Sobol search, "
         "median and IQR over 5 seeds, slot and end-ring manifolds.",
         extra=("results/task11_hv_curves.csv", "results/task11_hv_summary.csv")),
    Item("F34", "pareto_fronts", "figure", F + "task11_pareto_fronts.png", "task11_analysis", "RQ2 designs",
         "Pareto fronts (R, pump power, mass at 3 L/min) of the graded lattices on both manifold layouts with "
         "baselines B1-B3 and uniform gyroid/diamond references overlaid.",
         extra=("results/task11_front_slots.csv", "results/task11_front_end_rings.csv")),
    Item("F35", "designs_slots", "figure", F + "task11_designs_slots.png", "task11_analysis", "RQ2 designs",
         "Representative slot-manifold designs (min-R, best at the 0.306 W cap, knee, min-mass): rho*, w and L "
         "fields on the unwrapped jacket. The best-at-cap design is exported with w = 0 "
         "(task12_slots_best_at_cap_gyroid) as the manufacturable optimum."),
    Item("F36", "designs_end_rings", "figure", F + "task11_designs_end_rings.png", "task11_analysis",
         "RQ2 designs", "Representative end-ring-manifold designs (min-R, knee, min-mass): rho*, w and L fields."),
    # ---------------------------------------------------------------- geometry export
    Item("F37", "export_renders", "figure", F + "task12_renders.png", "task12_render", "Geometry export",
         "The exported manufacturable optimum (task12_slots_best_at_cap_gyroid), ray-traced from the signed-"
         "distance field that the PicoGK export reproduces voxel by voxel (IoU 0.997): overview, lattice close-up "
         "and outlet-slot section.",
         extra=(F + "task12_render_overview.png", F + "task12_render_lattice_closeup.png",
                F + "task12_render_outlet_slot_section.png")),
    Item("F38", "export_phase_field", "figure", F + "task12_phase_field.png", "task12_figs", "Geometry export",
         "Exported lattice unwrapped at the mean gap radius for three Task 11 designs: conformal phase-field grading "
         "with an integer number of cells around, no torn cells."),
    Item("F39", "export_sector_slices", "figure", F + "task12_sector_slices.png", "task12_figs", "Geometry export",
         "30-degree sector of slots/best_at_cap around its most critical cell: axial, meridional and unwrapped cuts "
         "with sleeve, outer wall, end caps, slot and port (the blend version that fails the 0.35 mm wall check)."),
    Item("F40", "export_volume_fraction", "figure", F + "task12_volume_fraction.png", "task12_figs",
         "Geometry export", "Volume fraction of the exported sectors vs the design rho* and the phase-aware "
         "expectation, overall, core / near-wall and per patch."),
    # ---------------------------------------------------------------- tables
    Item("T01", "solver_verification", "table", T + "T01_solver_verification.csv", "tables", "Homogenization",
         "Production estimators of the cell solvers: convergence order, raw and R(32, 64) errors against the "
         "extrapolated reference (k_eff, C_eff, K) for gyroid and diamond."),
    Item("T02", "surrogate_accuracy", "table", T + "T02_surrogate_accuracy.csv", "tables", "Closures",
         "Closure surrogate accuracy (R^2, MAPE, 95 % coverage) for the GP and deep ensemble on the cv5, band and "
         "sparse splits."),
    Item("T03", "finite_gap_coefficients", "table", T + "T03_finite_gap_coefficients.csv", "tables",
         "RQ1 finite gap", "Wall coefficients a of the finite-gap correction delta = a/N per morphology and density "
         "(mean over cut phases, phase standard deviation)."),
    Item("T04", "literature_closures", "table", T + "T04_literature_closures.csv", "tables", "Literature closures",
         "TPMS heat-transfer and inertial correlations considered (source, TPMS type, method, validity ranges)."),
    Item("T05", "baselines", "table", T + "T05_baselines.csv", "tables", "Baselines",
         "Baselines B1, B1s, B2, B3 at 1, 3 and 5 L/min: R, R_active, pressure drop, pump power, mass, margin."),
    Item("T06", "rq3_hypervolume", "table", T + "T06_rq3_hypervolume.csv", "tables", "RQ3 optimizers",
         "Hypervolume (median, IQR over 5 seeds) at 50 / 100 / 200 evaluations and final budget; evaluations NSGA-II "
         "needs to match qNEHVI at 200."),
    Item("T07", "rq2_equal_pump_power", "table", T + "T07_rq2_equal_pump_power.csv", "tables", "RQ2 designs",
         "RQ2 at equal pump power (3 L/min): optimized R vs the best baseline of each family below the pump-power "
         "cap; slots row at the 0.306 W cap uses the manufacturable optimum task12_slots_best_at_cap_gyroid."),
    Item("T08", "representative_designs", "table", T + "T08_representative_designs.csv", "tables", "RQ2 designs",
         "Representative designs per layout (R, R_active, pump power, mass, margin, field ranges), including the "
         "manufacturable slot optimum best_at_cap_gyroid."),
    Item("T09", "export_checks", "table", T + "T09_export_checks.csv", "tables", "Geometry export",
         "Manufacturability checks of the exported geometry (numpy mirror and PicoGK laptop builds): min wall, "
         "throat, connectivity, pockets, loose solid, volume fraction, PicoGK-vs-numpy IoU."),
]

_ITEM_INDEX = {i.id: i for i in ITEMS}


def item(item_id: str) -> Item:
    return _ITEM_INDEX[item_id.upper()]
