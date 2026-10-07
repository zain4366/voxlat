# Task 1 explained — TPMS unit-cell geometry

A walk-through of what `voxlat.geometry` does, why each choice was made, and how
to check it yourself. Read it with `src/voxlat/geometry/tpms.py` open next to it.
Exact numbers are in `STATUS.md` (Task 1 section).

---

## Step 1 — The level set: one formula per lattice

A TPMS lattice is defined implicitly. You write a smooth periodic function φ(x, y, z);
the surface is where φ = c, and the solid is one side of it.

On cell coordinates ξ ∈ [0,1)³ (multiply by the cell size L to get metres),
with X = 2πξ_x etc.:

- **Gyroid** G = sin X cos Y + sin Y cos Z + sin Z cos X, ranging over [−1.5, 1.5]
- **Diamond** D = sin X sin Y sin Z + sin X cos Y cos Z + cos X sin Y cos Z + cos X cos Y sin Z, ranging over [−√2, √2]

**Why normalize.** To blend them you add them, φ_w = (1−w)φ_G + wφ_D. If you skipped
normalization, the gyroid's larger amplitude would dominate any blend. Dividing each by its
maximum makes both span exactly [−1, 1].

**Check it yourself:** `test_normalization_both_span_unit_amplitude`.
G and D share no Fourier modes, so they are orthogonal: `(g*d).mean() == 0`.

**Odd symmetry.** Every one of these fields satisfies φ(−ξ) = −φ(ξ), so exactly half the
cell has φ ≤ 0. That gives a free sanity check: the threshold for 50 % density is c = 0
for every w.

**Is the formula right?** The surface φ = 0 is a close approximation of the true minimal
surface. Its area per cell converges to 3.0917 (gyroid) and 3.8381 (diamond). The exact
minimal-surface values are 3.0915 and 3.8377, so the formula and the period are confirmed
to 0.01 %. A wrong factor (π instead of 2π, say) would be off by a factor of 4.

## Step 2 — Network vs sheet: which side is solid

- **Network** (default): solid where φ ≤ c. The solid is one strut labyrinth; the fluid is
  everything else and forms **one connected domain**.
- **Sheet**: solid where |φ| ≤ c, a thickened wall around the surface. The wall splits the
  fluid into **two separate labyrinths** that never meet.

The jacket has one coolant that must travel from the inlet manifold to the outlet manifold,
so it needs one connected fluid. That rules in network and rules out sheet. Sheet would suit a
two-fluid heat exchanger instead. The test `test_sheet_cell_splits_the_fluid_in_two` proves the
two-labyrinth claim on the voxels themselves.

## Step 3 — Voxelization: turning the formula into a 3-D image

`voxelize(params, n)` samples φ on an n×n×n grid and marks `True` (solid) where φ ≤ c.

**Sample at voxel centres, ξ = (k + ½)/n.** If you used `linspace(0, 1, n)`, the first and
last layers would be the same physical plane (ξ = 0 and ξ = 1 coincide). Every periodic
solver in Tasks 2–4 would then see a duplicated layer. Sampling at centres makes the cell
tile perfectly: `np.tile(cell, 2)` is exactly the 2×2×2 block
(`test_tiling_equals_supercell_sampling`).

**Separable evaluation.** sin and cos are computed once per axis on 1-D arrays and combined
by broadcasting. Even 128³ voxels takes about 0.1 s.

## Step 4 — Anisotropy: stretching the cell without stretching the voxels

A stretch a_z makes the cell L × L × a_z L. There are two ways to represent that:

1. Keep n voxels per axis and make the voxels non-cubic.
2. Keep cubic voxels of edge h = L/n and give z more voxels: n_z = round(n·a_z). **(Chosen.)**

Option 2 keeps every later solver simple: one hexahedral element stiffness, one face
conductance, and the same staircase error in every direction. The price is that a_z is
snapped to n_z/n. At n = 48 that changes a_z by at most about 1 %, and
`effective_stretch()` reports the snapped value so it is never hidden.

## Step 5 — Hitting a target density: bisection and the table

The solid fraction f(c) = #(φ ≤ c)/N increases with c, so **bisection** on c finds the level
that gives density ρ (`threshold_for_density`).

A subtlety the tests caught: because of cubic symmetry, many voxels have *identical* φ.
f(c) therefore jumps by several voxels at once, and an exact ρ can be impossible. The
function returns the closest achievable fraction — still under 0.1 % at n = 32.

**The table** `c(w, ρ)` (`threshold_from_table`) stores the continuum answer, computed once at
n = 128 and shipped in `geometry/data/`. It is vectorized, handling 100 000 points in 0.13 s.
Graded designs need it, because every point of the jacket has its own (w, ρ). Task 12's C#
exporter will use it in the same way.

## Step 6 — Specific surface area a_sf

a_sf = (solid–fluid interface area) / (cell volume). It is the "a" in the h·a·ΔT heat-transfer
term of the device model (Task 9).

- **Wrong way:** count voxel faces. A staircase over-estimates a curved surface by about
  **1.5×** and never improves with refinement. Look at `a_sf_voxel_L` in the metrics to see it.
- **Right way:** marching cubes on the smooth φ. It is second-order accurate (error ∝ h²), with
  0.16 % error at n = 48 (see `task1_asf_convergence.png`).

Periodic detail: one wrapped layer is appended before marching cubes, so the cubes that
straddle the cell boundary are counted exactly once.

## Step 7 — Wall thickness and pore size

What matters for manufacturing is the **thinnest strut** and the **narrowest channel**. These
are measured with *local thickness*: at each point, the diameter of the largest ball that fits
inside the phase and contains the point. It is the standard micro-CT measure, from
Hildebrand & Rüegsegger 1997, and the one implemented in BoneJ.

Two lessons from building it:

1. **Voxel-centre distances are not good enough.** A plain distance transform measures to the
   centre of the nearest *voxel*, but the true surface can sit anywhere from 0.05 to 0.85 voxel
   before that centre. The first version read a 12-voxel cylinder as 9 voxels. The fix measures
   distance to the **marching-cubes surface** with a periodic KD-tree, which is sub-voxel
   accurate.
2. **Ball centres sit on voxel centres.** The ideal centre usually lies between voxels, so balls
   come out slightly small. A calibrated half-voxel "slack" removes the average bias. On random
   cylinders and slabs the result is within ±1 voxel (`test_local_thickness_calibration`).

**Throat** = the largest ball that can *travel* through the phase in x, y and z. It is found by
growing a radius until the allowed ball centres stop percolating. For pores this is the powder
removal and clogging criterion. For the solid it is the neck of the load path.

## Step 8 — Connectivity on a torus

`periodic_components` first labels clusters inside the cell, then glues them across the six
faces with a union-find. While gluing, it also tracks *which neighbouring cell* each piece
connects to. If a cluster connects to a copy of itself shifted by one cell, it is infinite in
that direction (percolating). The rank of those shifts classifies each cluster:
0 = island, 1 = rod, 2 = sheet, 3 = full 3-D network.

Why care:

- A floating solid island cannot be printed.
- A trapped fluid pocket holds powder or air.
- A fluid that does not percolate in z has zero permeability K_zz in Task 4.

6-connectivity (faces only) is used for transport, because the solvers exchange flux through
faces. 26-connectivity is used for ball motion.

## Step 9 — Manufacturability and the feasible region

Wall and pore both scale with L: t = t̂(w, ρ)·L and p = p̂(w, ρ)·L. Both limits are therefore
*lower* bounds on L:

    L ≥ 0.35 mm / t̂(w, ρ)   (thin walls at low ρ)
    L ≥ 0.80 mm / p̂(w, ρ)   (small pores at high ρ)

The feasible region is simply **L ≥ L_min(w, ρ)**. That is one smooth inequality for the
optimizer in Task 11. `min_cell_size()` evaluates it from the precomputed morphology table, and
`check_manufacturable()` does the full direct check for a single design.

## Step 10 — What the results say

- Pure gyroid and diamond are manufacturable almost everywhere in the 2–6 mm box. L_min ranges
  from 1.5 to 2.2 mm, and only diamond's pore throat binds, at ρ > 0.45 and L ≈ 2 mm.
- **Blends are different.** Linear blending of G and D creates **pinch-point necks**: the
  minimum wall falls to 0.04–0.1 L at some densities, and at w = 0.5, ρ = 0.5 it tends to
  zero. For w = 0.5, only 79 % of the box is feasible. This is a real geometric property of the
  blended level set, not a voxel artefact; the thickness is resolution-independent where it
  converges. It bears directly on RQ2 and belongs in the paper.

---

## How to verify on your laptop

```powershell
pip install -e .                                  # picks up the new package data
pytest -q                                         # 108 tests, ~1 min (pytest -m "not slow": ~20 s)
python scripts/task1_tpms_figures.py              # regenerates the 4 figures (~5 s)
python scripts/task1_build_tables.py              # only if you change the geometry (~6-10 min)
```

Try it interactively:

```python
from voxlat.geometry import TPMSParams, voxelize, compute_metrics, check_manufacturable
p = TPMSParams(w=0.0, rho=0.3)                    # gyroid, 30 % solid
cell = voxelize(p, n=48)                          # bool[48,48,48]
m = compute_metrics(p, n=48, L=4e-3)              # 4 mm cell
m.a_sf(), m.wall_m().min, m.pore_m().throat       # 1/m, m, m
check_manufacturable(p, L=4e-3).ok
```
