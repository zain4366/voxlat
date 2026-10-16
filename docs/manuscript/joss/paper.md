---
title: 'VoxLat: verified voxel homogenization and multi-objective design of graded TPMS lattice cooling jackets on a laptop CPU'
tags:
  - Python
  - triply periodic minimal surfaces
  - homogenization
  - porous media
  - heat transfer
  - multi-objective optimization
  - additive manufacturing
authors:
  - name: Zain ul Abiden
    orcid: 0000-0000-0000-0000   # replace with the real ORCID
    corresponding: true
    affiliation: 1
  - name: Ahmed Mukhtar Bukhari
    affiliation: 1
affiliations:
  - name: Department of Aerospace Engineering, School of Mechanical and Manufacturing Engineering (SMME), National University of Sciences and Technology (NUST), Islamabad, Pakistan
    index: 1
date: 17 October 2026
bibliography: paper.bib
---

# Summary

Additively manufactured lattices based on triply periodic minimal surfaces (TPMS), such as the gyroid and the Schwarz diamond, can fill the coolant gap of a heat exchanger or an electric-motor jacket with one structure that transfers heat, guides the flow and carries load. Designing a spatially graded lattice for a whole device needs effective (homogenized) properties at every point of the design, because pore-resolved simulation of the full device is far too expensive for optimization. `VoxLat` is a pure-Python package that computes these properties on periodic voxel cells, corrects them for walls that bound only a few cells, assembles them into a fast device model, and optimizes and exports a printable graded lattice. Everything runs on a laptop CPU.

The package provides (i) TPMS level sets for gyroid, diamond and their blends with exact voxel periodicity, density thresholds, surface area, wall-thickness and pore-throat metrics and a manufacturable cell-size floor; (ii) periodic cell solvers for the effective conductivity tensor (finite volumes), the effective stiffness tensor and stress localization (trilinear voxel finite elements) and the Stokes permeability tensor (staggered marker-and-cell finite volumes with MINRES), each with a two-grid Richardson estimator; (iii) a closure dataset generator and surrogate models (Gaussian processes, deep ensembles and a small 3D CNN) that predict symmetric positive-definite tensors with exact material symmetry and calibrated uncertainty; (iv) a finite-gap module that solves wall-bounded lattice strips and fits a $1/N$ correction for $N$ cells across a gap; (v) a two-dimensional homogenized jacket model with Darcy–Forchheimer flow, a three-field local-thermal-non-equilibrium energy balance and a sandwich structural check; (vi) baselines (helical and circumferential channels, uniform and density-graded lattices); (vii) B-spline grading fields with a feasibility repair, wrappers for NSGA-II [@deb2002; @blank2020] and BoTorch qNEHVI [@daulton2021; @balandat2020], and exact hypervolume utilities; and (viii) an exporter that evaluates the graded lattice as an implicit function in C# inside the PicoGK geometry kernel [@picogk], with a numpy mirror and voxel-based manufacturability checks.

# Statement of need

Researchers who design graded lattices for thermal management usually take closures from published periodic-cell CFD [@gajetti2025; @savoldi2026] or compute them with commercial multiphysics codes, and they rarely check the closures against the few-cell geometries that real devices impose. A motor jacket, for example, holds only one to three cells across its gap. `VoxLat` targets researchers and students who need (a) verified effective properties of TPMS lattices without a commercial licence or a cluster, (b) a quantitative estimate of the error that bulk closures make in wall-bounded gaps, and (c) a complete, reproducible path from unit cell to optimized and checked geometry. With `VoxLat` we found that bulk closures over-predict the in-plane permeability of gyroid and diamond lattices by 23–40% when one cell spans the gap, and that a one-parameter $1/N$ correction removes this error to within the scatter caused by where the wall cuts the lattice [@bukhari2026].

# State of the field

General-purpose homogenization tools exist for elasticity and conduction (for example the procedure of @andreassen2014 and voxel-based FFT solvers), and pore-scale flow on images can be computed with lattice-Boltzmann or finite-volume codes such as OpenFOAM. TPMS geometry is available in design tools and in libraries that generate meshes for printing. To our knowledge, no open package combines TPMS-specific geometry and manufacturability metrics, verified cell solvers for all three closures on the same voxel grid, tensor-valued surrogates with symmetry constraints, finite-gap corrections, a device-level model and multi-objective optimization with geometry export. `VoxLat` contributes this integration and, more importantly, the verification that ties it together. Each solver computes its tensor in two independent ways (flux and energy) and is tested against exact solutions: laminates, plane Poiseuille and duct flow, Hashin–Shtrikman bounds and the Zick–Homsy sphere-array permeabilities [@zick1982]. Our gyroid and diamond permeabilities agree with independent body-fitted CFD to within 2% [@gajetti2025].

# Software design

`VoxLat` is a `src`-layout Python package (numpy, scipy, scikit-image, scikit-learn, pandas) with optional extras for BoTorch/PyTorch (`opt`, `ml`), pyamg (`amg`) and the PicoGK export through pythonnet (`export`). The design follows three rules. First, one voxel convention throughout: boolean arrays in $(x, y, z)$ order, solid = `True`, sampled at voxel centres so that tiled cells are exactly periodic. Every solver therefore sees the same staircase geometry, and the finite-gap strips are bit-exact tilings of the bulk cell, so discretization errors cancel in strip-to-bulk ratios. Second, verification first: every physics module ships `pytest` tests against analytic or published cases (527 fast tests pass on Windows and Linux in continuous integration; slow convergence tests are marked separately), and each solver docstring states its governing equations and boundary conditions. Third, reproducibility: one YAML file holds every number of the reference problem, every computed table carries a run record (parameters, version, git hash, resolution, wall time), seeds are fixed, and a catalog with `run_all.py` regenerates every figure and table (`--replot` from committed data in about ten minutes, `--quick` for every producer at reduced size). The device model has a single `evaluate(design) -> dict` interface shared by the lattice model and the baselines, so optimizers and comparisons do not depend on the cooling concept. Trained models are stored as plain `.npz` files with JSON metadata rather than pickles, which keeps them independent of library versions.

```python
from voxlat.geometry import TPMSParams, voxelize
from voxlat.homogenization import extrapolated_permeability_tpms
from voxlat.device import evaluate

K = extrapolated_permeability_tpms(TPMSParams(w=0.0, rho=0.35))   # gyroid, K / L^2
out = evaluate(rho=0.35, w=0.0, L=4e-3)                          # uniform jacket, 3 L/min
print(K.mean, out["thermal_resistance"], out["pump_power"], out["mass"])
```

# Research impact statement

`VoxLat` produced every result of the accompanying research article [@bukhari2026]: a 290-cell closure dataset (27.8 CPU-hours on a laptop), a 336-strip finite-gap study, tuned baselines, and a 30-run multi-objective study (NSGA-II, qNEHVI and Sobol search, five seeds each on two manifold layouts). In that study batch Bayesian optimization needed about seven times fewer device evaluations than NSGA-II for the same hypervolume. The exported optimum was built in PicoGK and agrees with the numpy specification at an intersection-over-union of 0.997. The data, trained models and design files are part of the repository, so others can reuse the closures, the finite-gap coefficients or the device model directly. [Add evidence of external use, such as citations, users or course adoption, before submission; JOSS asks for realized or concrete near-term impact.]

# AI usage disclosure

[Required by JOSS. To be completed by the authors before submission.]

# References
