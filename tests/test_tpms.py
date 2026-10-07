"""Task 1 verification of the TPMS unit-cell geometry (voxlat.geometry.tpms)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from voxlat.geometry.tpms import (
    DIAMOND_AREA_EXACT,
    GYROID_AREA_EXACT,
    MORPHOLOGY_TABLE_FILE,
    TPMSParams,
    build_threshold_table,
    check_manufacturable,
    compute_metrics,
    effective_stretch,
    feasible_region,
    grid_shape,
    level_set,
    load_threshold_table,
    min_cell_size,
    sample_level_set,
    specific_surface_area,
    threshold_for_density,
    threshold_from_table,
    voxelize,
)
from voxlat.geometry.voxel_tools import periodic_components
from voxlat.utils.config import load_config

needs_morph_table = pytest.mark.skipif(
    not MORPHOLOGY_TABLE_FILE.is_file(), reason="run scripts/task1_build_tables.py first"
)


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
def test_params_validation_and_design_constructor():
    p = TPMSParams.from_design(0.3, 0.25, 1.2)
    assert p.a == (1.0, 1.0, 1.2) and p.kind == "network"
    for bad in (dict(w=1.2), dict(rho=0.0), dict(rho=1.0), dict(a=(1, 1, 0)), dict(kind="foam")):
        with pytest.raises(ValueError):
            TPMSParams(**bad)


# -----------------------------------------------------------------------------
# Level sets
# -----------------------------------------------------------------------------
def test_normalization_both_span_unit_amplitude():
    for w in (0.0, 1.0):
        phi = sample_level_set((128,) * 3, w)
        assert 0.995 < np.abs(phi).max() <= 1.0 + 1e-12
    rng = np.random.default_rng(0)
    pts = rng.random((3, 20000))
    for w in rng.random(5):
        assert np.abs(level_set(*pts, w=w)).max() <= 1.0 + 1e-12


def test_rms_amplitudes_are_comparable():
    """Normalized G and D have RMS 1/sqrt(3) and 1/2 and are orthogonal (disjoint Fourier modes)."""
    g = sample_level_set((32,) * 3, 0.0)
    d = sample_level_set((32,) * 3, 1.0)
    assert np.sqrt((g**2).mean()) == pytest.approx(1 / math.sqrt(3), rel=1e-10)
    assert np.sqrt((d**2).mean()) == pytest.approx(0.5, rel=1e-10)
    assert abs((g * d).mean()) < 1e-12


@pytest.mark.parametrize("w", [0.0, 0.37, 1.0])
def test_level_set_is_odd_and_periodic(w):
    rng = np.random.default_rng(1)
    x, y, z = rng.uniform(-2, 2, (3, 1000))
    phi = level_set(x, y, z, w)
    assert np.allclose(level_set(-x, -y, -z, w), -phi, atol=1e-12)
    for shift in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -2, 3)]:
        assert np.allclose(level_set(x + shift[0], y + shift[1], z + shift[2], w), phi, atol=1e-12)


def test_sampled_field_matches_pointwise_level_set():
    n = (12, 10, 14)
    phi = sample_level_set(n, 0.4)
    idx = np.indices(n).reshape(3, -1)
    xi = [(idx[i] + 0.5) / n[i] for i in range(3)]
    assert np.allclose(phi.ravel(), level_set(*xi, w=0.4), atol=1e-12)


# -----------------------------------------------------------------------------
# Periodic voxelization
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("w,a", [(0.0, (1, 1, 1)), (1.0, (1, 1, 1)), (0.5, (1, 1, 1.5))])
def test_tiling_equals_supercell_sampling(w, a):
    """voxelize() tiles exactly: a 2x2x2 tiling equals sampling the 2x2x2 block directly."""
    p = TPMSParams(w=w, rho=0.3, a=a)
    n = 20
    cell, phi, c = voxelize(p, n, return_field=True)
    shape = cell.shape
    big = tuple(2 * s for s in shape)
    idx = np.indices(big).reshape(3, -1)
    xi = [(idx[i] + 0.5) / shape[i] for i in range(3)]  # coordinates in cell units, 0 ... 2
    phi_big = level_set(*xi, w=w).reshape(big)
    assert np.allclose(np.tile(phi, (2, 2, 2)), phi_big, atol=1e-12)
    ambiguous = np.abs(phi_big - c) < 1e-9
    assert np.array_equal(np.tile(cell, (2, 2, 2))[~ambiguous], (phi_big <= c)[~ambiguous])


def test_no_duplicated_boundary_layer():
    """First and last voxel layers differ (no linspace(0, 1, n) duplicate face)."""
    cell = voxelize(TPMSParams(w=0.0, rho=0.3), 32)
    assert not np.array_equal(cell[0], cell[-1])
    # but the field is continuous across the wrap: same statistics as any interior neighbour pair
    jump = np.mean(cell[0] != cell[-1])
    interior = np.mean([np.mean(cell[i] != cell[i + 1]) for i in range(31)])
    assert jump == pytest.approx(interior, abs=0.05)


def test_anisotropic_grid_and_stretch():
    p = TPMSParams(w=0.0, rho=0.3, a=(1.0, 1.0, 1.37))
    assert grid_shape(p, 48) == (48, 48, 66)
    assert effective_stretch(p, 48)[2] == pytest.approx(66 / 48)
    cell, phi, c = voxelize(p, 48, return_field=True)
    assert cell.shape == (48, 48, 66)
    # cubic voxels of edge L/n: voxel (i, j, k) sits at physical z = (k + 1/2) L / n,
    # i.e. cell coordinate xi_z = z / (a_z L)
    k = np.arange(66)
    xi_z = (k + 0.5) / 48 / effective_stretch(p, 48)[2]
    col = level_set((7 + 0.5) / 48, (11 + 0.5) / 48, xi_z, 0.0)
    assert np.allclose(phi[7, 11, :], col, atol=1e-12)


# -----------------------------------------------------------------------------
# Density thresholds
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["network", "sheet"])
@pytest.mark.parametrize("w", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("rho", [0.2, 0.35, 0.5])
def test_density_within_half_percent(kind, w, rho):
    for n, a in ((32, (1, 1, 1)), (40, (1, 1, 0.7))):
        cell = voxelize(TPMSParams(w=w, rho=rho, a=a, kind=kind), n)
        assert abs(cell.mean() - rho) / rho < 0.005


@pytest.mark.parametrize("rho", [0.27, 0.31, 0.444])
def test_bisection_hits_closest_achievable_density(rho):
    """The voxel fraction is a step function of c (with ties from cubic symmetry);
    bisection must land on the step closest to rho."""
    g = sample_level_set((24,) * 3, 0.3)
    c = threshold_for_density(0.3, rho, 24, g=g)
    achievable = np.searchsorted(np.sort(g.ravel()), np.unique(g), side="right") / g.size
    best = np.min(np.abs(achievable - rho))
    assert abs((g <= c).mean() - rho) == pytest.approx(best, abs=1e-15)


def test_odd_fields_give_zero_threshold_at_half_density():
    tab = load_threshold_table("network")
    j = int(np.argmin(np.abs(tab.rho - 0.5)))
    assert np.allclose(tab.c[:, j], 0.0, atol=2e-3)
    for w in (0.0, 0.5, 1.0):
        assert abs(threshold_for_density(w, 0.5, 32)) < 0.02


@pytest.mark.parametrize("kind", ["network", "sheet"])
def test_threshold_table_reproduces_density(kind):
    rng = np.random.default_rng(2)
    for w, rho in zip(rng.uniform(0, 1, 6), rng.uniform(0.18, 0.55, 6)):
        cell = voxelize(TPMSParams(w=w, rho=rho, kind=kind), 64, method="table")
        assert abs(cell.mean() - rho) / rho < 0.01


def test_threshold_table_vectorized_and_monotone():
    c = threshold_from_table(np.array([0.0, 0.5, 1.0])[:, None], np.linspace(0.1, 0.9, 9)[None, :])
    assert c.shape == (3, 9)
    assert np.all(np.diff(c, axis=1) > 0)
    with pytest.raises(ValueError):
        threshold_from_table(0.5, 0.995)


def test_shipped_table_matches_rebuild():
    tab = load_threshold_table("network")
    fresh = build_threshold_table("network", n_ref=64, w_grid=np.array([0.0, 0.5, 1.0]),
                                  rho_grid=np.array([0.2, 0.3, 0.4, 0.5]))
    stored = tab(fresh.w[:, None], fresh.rho[None, :])
    assert np.allclose(stored, fresh.c, atol=3e-3)


# -----------------------------------------------------------------------------
# Specific surface area
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("w,exact", [(0.0, GYROID_AREA_EXACT), (1.0, DIAMOND_AREA_EXACT)])
def test_minimal_surface_area_per_cell(w, exact):
    """phi = 0 nodal surfaces vs exact minimal-surface areas A/a^2 (G 3.0915, D 3.8377).

    The nodal approximations are not exactly minimal, but agree to < 0.1 %.
    """
    n = 96
    phi = sample_level_set((n,) * 3, w)
    # a_sf * L of a cubic cell of edge L is the area per cell in units of L^2
    assert specific_surface_area(phi, 0.0, n=n) == pytest.approx(exact, rel=1e-3)


def test_surface_area_convergence_second_order():
    """a_sf*L at fixed level c converges as O(h^2) over n = 24, 32, 48, 64."""
    c = threshold_from_table(0.0, 0.3)
    ns = np.array([24, 32, 48, 64])
    a = np.array([specific_surface_area(sample_level_set((n,) * 3, 0.0), c, n=n) for n in ns])
    # Richardson extrapolation assuming p = 2 from the two finest grids
    a_inf = a[-1] + (a[-1] - a[-2]) / ((64 / 48) ** 2 - 1)
    err = np.abs(a - a_inf)
    assert np.all(np.diff(err) < 0)
    p_obs = np.polyfit(np.log(1 / ns), np.log(err), 1)[0]
    assert 1.6 < p_obs < 2.4
    assert err[-1] / a_inf < 3e-3


def test_sheet_surface_is_two_walls():
    phi = sample_level_set((48,) * 3, 0.0)
    t = 0.2
    a_sheet = specific_surface_area(phi, t, "sheet", n=48)
    a_two = specific_surface_area(phi, t, n=48) + specific_surface_area(phi, -t, n=48)
    assert a_sheet == pytest.approx(a_two)


# -----------------------------------------------------------------------------
# Connectivity of typical cells
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("w", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("rho", [0.2, 0.35, 0.5])
def test_network_cells_have_one_solid_and_one_fluid(w, rho):
    for a in ((1, 1, 1), (1, 1, 1.5)):
        cell = voxelize(TPMSParams(w=w, rho=rho, a=a), 32)
        for phase in (cell, ~cell):
            comp = periodic_components(phase)
            assert comp.single_percolating, (w, rho, a, comp.n_components, comp.percolates)
            assert comp.wrap_rank[0] == 3


def test_sheet_cell_splits_the_fluid_in_two():
    cell = voxelize(TPMSParams(w=0.0, rho=0.3, kind="sheet"), 32)
    fluid = periodic_components(~cell)
    assert fluid.n_components == 2
    assert fluid.volume_fractions == pytest.approx((0.5, 0.5), abs=0.01)
    assert all(all(p) for p in fluid.percolates)
    assert periodic_components(cell).single_percolating


# -----------------------------------------------------------------------------
# Full metrics and manufacturability
# -----------------------------------------------------------------------------
def test_compute_metrics_gyroid_reference_values():
    m = compute_metrics(TPMSParams(w=0.0, rho=0.5), 48, L=4e-3)
    assert m.relative_density == pytest.approx(0.5, abs=1e-4)
    assert m.a_sf_L == pytest.approx(GYROID_AREA_EXACT, rel=3e-3)
    assert m.a_sf() == pytest.approx(m.a_sf_L / 4e-3)
    # staircase area is ~3/2 of the true area (and is not used as a_sf)
    assert 1.4 < m.a_sf_voxel_L / m.a_sf_L < 1.65
    # at rho = 0.5 the gyroid's solid and fluid are congruent -> identical statistics
    assert m.wall.mean == pytest.approx(m.pore.mean, rel=1e-6)
    assert m.wall.throat == pytest.approx(m.pore.throat, rel=1e-6)
    # sanity: neck/strut ~0.4-0.5 L thick, throat <= min local thickness + 1 voxel
    assert 0.38 < m.wall.min < 0.47
    assert m.wall.throat <= m.wall.min + 1.5 / 48
    assert m.solid_connected and m.fluid_connected
    row = m.as_row()
    assert row["wall_min_m"] == pytest.approx(m.wall.min * 4e-3)


def test_thickness_scales_with_density():
    lo = compute_metrics(TPMSParams(w=1.0, rho=0.2), 32, connectivity=False)
    hi = compute_metrics(TPMSParams(w=1.0, rho=0.45), 32, connectivity=False)
    assert hi.wall.p5 > lo.wall.p5 and hi.pore.throat < lo.pore.throat


def test_check_manufacturable_cases():
    cfg = load_config()
    good = check_manufacturable(TPMSParams(w=0.0, rho=0.35), 5e-3, n=40, cfg=cfg)
    assert good.ok and good.in_design_bounds and not good.reasons
    small_pores = check_manufacturable(TPMSParams(w=1.0, rho=0.5), 2e-3, n=40, cfg=cfg)
    assert not small_pores.ok and not small_pores.pore_ok and small_pores.wall_ok
    thin = check_manufacturable(TPMSParams(w=0.0, rho=0.12), 1.2e-3, n=40, cfg=cfg)
    assert not thin.wall_ok and not thin.in_design_bounds
    assert any("wall" in r for r in thin.reasons)


@needs_morph_table
def test_min_cell_size_structure():
    rho = np.linspace(0.2, 0.5, 13)
    for w in (0.0, 1.0):
        L, Lw, Lp = min_cell_size(np.full_like(rho, w), rho, return_parts=True)
        # walls thicken with density (min LT is quantised to 1/4 voxel -> allow flat steps)
        assert np.all(np.diff(Lw) <= 1e-12) and Lw[0] > 1.3 * Lw[-1]
        assert np.all(np.diff(Lp) > 0)  # pore throats shrink with density
        assert np.allclose(L, np.maximum(Lw, Lp))


@needs_morph_table
@pytest.mark.slow
def test_feasible_region_agrees_with_direct_check():
    """Table-based L_min must agree with a direct n = 48 check on both sides of the boundary.

    (rho, w) are table grid points, so this checks the plumbing exactly; L_min itself
    does not depend on the design bounds, so points below L = 2 mm are tested too.
    """
    cfg = load_config()
    checked = 0
    for w in (0.0, 0.5, 1.0):
        fr = feasible_region(w, cfg=cfg)
        assert fr.mask.shape == (len(fr.rho), len(fr.L))
        for rho in (0.25, 0.45):
            L_min = float(min_cell_size(w, rho, cfg=cfg))
            for L, expect in ((L_min * 1.05, True), (L_min * 0.95, False)):
                rep = check_manufacturable(TPMSParams(w=w, rho=rho), L, n=48, cfg=cfg)
                assert rep.ok == expect, (w, rho, L, rep)
                checked += 1
    assert checked == 12


def test_offset_by_whole_voxels_is_a_roll():
    """Task 2 addition: a grid offset of k/n cell lengths equals np.roll by -k voxels."""
    p = TPMSParams(w=0.3, rho=0.35)
    n = 24
    base = voxelize(p, n)
    shifted = voxelize(p, n, offset=(2 / n, 0.0, -1 / n))
    # identical up to voxels whose field value ties the threshold (round-off in sin/cos)
    mismatch = np.mean(shifted != np.roll(base, (-2, 1), axis=(0, 2)))
    assert mismatch < 2e-3, mismatch
    other = voxelize(p, n, offset=(0.37 / n, 0.11, 0.5))  # generic shift keeps the density
    assert abs(other.mean() - 0.35) < 1e-3
