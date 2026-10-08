"""Task 4 verification of the periodic Stokes permeability solver (voxlat.homogenization.stokes).

(a) plane Poiseuille flow: exact discrete MAC value and 2nd-order convergence to H^2/12;
(b) square / rectangular duct vs the series solution (2nd order);
    inclined slits: staircase error of non-aligned walls (1st order for slopes 1:2, 1:3);
(c) simple-cubic sphere arrays vs Zick & Homsy (1982) at 4 solid fractions;
(d) TPMS cells: convergence + Richardson, symmetry, SPD, invariances, flux = energy.
"""

from __future__ import annotations

import numpy as np
import pytest

from voxlat.geometry import TPMSParams, voxelize
from voxlat.homogenization.convergence import fit_convergence, richardson
from voxlat.homogenization.stokes import (
    ZICK_HOMSY_SC,
    assemble_stokes_system,
    averaged_permeability_tpms,
    extrapolated_permeability_tpms,
    inclined_slit_cell,
    inclined_slit_reference,
    kozeny_constant,
    mac_slit_permeability,
    permeability,
    permeability_tpms,
    pyamg_available,
    rectangular_duct_mean_velocity,
    sangani_acrivos_sc_drag,
    sc_sphere_permeability,
    slit_permeability,
    solve_stokes,
    sphere_array_cell,
    tpms_symmetry,
    zick_homsy_sc_drag,
)


def plates_cell(H: int, axis: int = 1, wall: int = 1, other: int = 2) -> np.ndarray:
    """Periodic channel: `wall` solid layers then H fluid layers along `axis`."""
    shape = [other, other, other]
    shape[axis] = H + wall
    cell = np.zeros(shape, bool)
    sl = [slice(None)] * 3
    sl[axis] = slice(0, wall)
    cell[tuple(sl)] = True
    return cell


def duct_cell(mx: int, my: int, nz: int = 1) -> np.ndarray:
    """Periodic rectangular duct along z: mx x my fluid voxels, one solid layer at x = 0 and y = 0."""
    cell = np.zeros((mx + 1, my + 1, nz), bool)
    cell[0] = True
    cell[:, 0] = True
    return cell


# =============================================================================
# (a) Plane Poiseuille
# =============================================================================
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("H", [1, 2, 4, 7, 16])
def test_plane_poiseuille_exact_discrete(axis, H):
    """MAC with mirror-ghost walls: K = (H^2 + 2)/12 * phi exactly; zero across the plates."""
    cell = plates_cell(H, axis)
    r = permeability(cell, tol=1e-12)
    phi = H / (H + 1)
    k_disc = float(mac_slit_permeability(H)) * phi
    along = [d for d in range(3) if d != axis]
    for d in along:
        assert r.K[d, d] == pytest.approx(k_disc, rel=1e-10)
        assert r.tortuosity[d] == pytest.approx(1.0, abs=1e-12)  # straight streamlines
        assert r.interstitial_velocity[d, d] == pytest.approx(k_disc / phi, rel=1e-10)
    assert abs(r.K[axis, axis]) < 1e-12 * k_disc  # no flow through the walls
    assert np.isnan(r.tortuosity[axis])
    np.testing.assert_allclose(r.K - np.diag(np.diag(r.K)), 0.0, atol=1e-12 * k_disc)
    assert r.porosity == pytest.approx(phi)
    assert r.n_pressure_components == 1
    assert r.agreement < 1e-8


def test_plane_poiseuille_second_order_to_exact():
    """Continuum K = H^2/12 * phi; relative error 2/H^2 -> observed order exactly 2."""
    Hs = np.array([4, 8, 16, 32])
    err = []
    for H in Hs:
        r = permeability(plates_cell(int(H), axis=1, wall=1, other=1), tol=1e-12)
        exact = float(slit_permeability(H)) * H / (H + 1)
        err.append(r.K[0, 0] / exact - 1.0)
    err = np.array(err)
    np.testing.assert_allclose(err, 2.0 / Hs**2, rtol=1e-8)
    p_obs = np.log2(err[:-1] / err[1:])
    np.testing.assert_allclose(p_obs, 2.0, atol=1e-6)


def test_thick_walls_and_unit_axis():
    """Thicker walls and a 1-voxel periodic axis give the same exact discrete slit value."""
    H = 6
    r = permeability(plates_cell(H, axis=2, wall=3, other=1), tol=1e-12)
    k_disc = float(mac_slit_permeability(H)) * H / (H + 3)
    assert r.K[0, 0] == pytest.approx(k_disc, rel=1e-10)
    assert r.K[1, 1] == pytest.approx(k_disc, rel=1e-10)


def test_voxel_size_scaling():
    cell = plates_cell(8, axis=1)
    r1 = permeability(cell)
    r2 = permeability(cell, voxel_size=0.25)
    np.testing.assert_allclose(r2.K, r1.K * 0.0625, rtol=1e-12, atol=1e-300)
    assert r2.mean_speed[0] == pytest.approx(r1.mean_speed[0] * 0.0625)


def test_poiseuille_velocity_profile():
    """Discrete profile u_j = (y_j (H - y_j) + 1/4)/2 (y_j = j - 1/2), u = 0 on wall faces."""
    H = 10
    cell = plates_cell(H, axis=1, other=3)
    r = permeability(cell, directions=[0], tol=1e-12, return_fields=True)
    u = r.velocity[0, 0][0, :, 0].astype(float)  # u_x along y, x = z = 0
    y = np.arange(1, H + 1) - 0.5
    np.testing.assert_allclose(u[1:], (y * (H - y) + 0.25) / 2.0, rtol=1e-6)
    assert u[0] == 0.0  # solid row
    assert np.max(np.abs(r.velocity[0, 1:])) < 1e-12 * u.max()  # no transverse flow
    assert np.isnan(r.K[1, 1]) and np.isnan(r.K[0, 1]) and r.K[0, 0] > 0  # only x solved


# =============================================================================
# (b) Ducts and inclined walls
# =============================================================================
def test_duct_series_reference_values():
    # square duct: u_mean = 0.0351443 a^2 G/mu (Poiseuille number f Re = 56.91)
    assert rectangular_duct_mean_velocity(1.0, 1.0) == pytest.approx(0.0351443, rel=2e-6)
    assert 32.0 / rectangular_duct_mean_velocity(1.0, 1.0) / 16.0 == pytest.approx(56.91, rel=1e-3)
    # wide duct -> plane Poiseuille with the 0.63 b/a end correction
    b, a = 1.0, 100.0
    assert rectangular_duct_mean_velocity(a, b) == pytest.approx(b * b / 12 * (1 - 0.630 * b / a), rel=1e-4)
    assert rectangular_duct_mean_velocity(2.0, 1.0) == rectangular_duct_mean_velocity(1.0, 2.0)


@pytest.mark.parametrize("aspect", [1, 2])
def test_duct_vs_series_second_order(aspect):
    ms = [8, 16, 32]
    errs = []
    for m in ms:
        mx, my = m, aspect * m
        cell = duct_cell(mx, my)
        r = permeability(cell, tol=1e-11)
        phi = mx * my / ((mx + 1) * (my + 1))
        exact = rectangular_duct_mean_velocity(mx, my) * phi
        errs.append(r.K[2, 2] / exact - 1.0)
        assert abs(r.K[0, 0]) < 1e-10 * r.K[2, 2] and abs(r.K[1, 1]) < 1e-10 * r.K[2, 2]
        assert r.tortuosity[2] == pytest.approx(1.0, abs=1e-12)
    errs = np.array(errs)
    assert np.all(errs > 0)  # mirror-ghost walls: slight over-prediction
    assert abs(errs[-1]) < 0.005  # 32 voxels across: < 0.5 %
    p_obs = np.log2(errs[:-1] / errs[1:])
    assert np.all((p_obs > 1.9) & (p_obs < 2.1))
    # Richardson (p = 2) removes the error almost entirely
    assert abs(richardson(16, errs[1], 32, errs[2], p=2)) < 3e-4


def test_inclined_slit_45deg_second_order():
    errs_t, errs_z = [], []
    for N in (10, 20, 40):
        cell = inclined_slit_cell(N, 0.5, slope=1)
        r = permeability(cell, tol=1e-11)
        k, t, nrm = inclined_slit_reference(N, 0.5, slope=1)
        errs_t.append(t @ r.K @ t / k - 1)
        errs_z.append(r.K[2, 2] / k - 1)
        assert abs(nrm @ r.K @ nrm) < 1e-9 * k  # no flow across the slit
    np.testing.assert_allclose(errs_t, [-4 / N**2 for N in (10, 20, 40)], rtol=1e-6)
    np.testing.assert_allclose(errs_z, [8 / N**2 for N in (10, 20, 40)], rtol=1e-6)


@pytest.mark.parametrize("slope", [2, 3])
def test_inclined_slit_first_order_staircase(slope):
    """A wall not aligned with the grid: K along the slit is under-predicted, error ~ 1/N."""
    Ns = (20, 40, 80)
    errs = []
    for N in Ns:
        cell = inclined_slit_cell(N, 0.5, slope=slope)
        r = permeability(cell, tol=1e-11)
        k, t, nrm = inclined_slit_reference(N, 0.5, slope=slope)
        errs.append(t @ r.K @ t / k - 1)
        assert abs(nrm @ r.K @ nrm) < 2e-3 * k
    errs = np.array(errs)
    assert np.all(errs < 0)
    assert abs(errs[-1]) < 0.025  # fluid layer ~ 18 (slope 2) / 13 (slope 3) voxels thick
    ratio = errs[1] / errs[2]
    if slope == 2:
        assert 1.8 < ratio < 2.2  # first order
        assert abs(richardson(40, errs[1], 80, errs[2], p=1)) < 1e-3
    # wall displacement per wall ~ -0.05 h: error ~ -0.2 h / H
    H = 0.5 * Ns[-1] / np.sqrt(1 + slope**2)
    assert -0.4 < errs[-1] * H < -0.1


# =============================================================================
# (c) Simple-cubic sphere arrays vs Zick & Homsy (1982)
# =============================================================================
def test_sphere_reference_data_consistent():
    """Sangani & Acrivos (1982) series agrees with Zick & Homsy (1982) for c <= 0.216."""
    for c in (0.027, 0.064, 0.125, 0.216):
        assert sangani_acrivos_sc_drag(c) == pytest.approx(ZICK_HOMSY_SC[c], rel=0.01)
    # dilute limit: Hasimoto's 1 / (1 - 1.7601 c^1/3)
    assert sangani_acrivos_sc_drag(1e-6) == pytest.approx(1 / (1 - 1.7601e-2), rel=1e-5)
    # k/l^2 = 1/(6 pi a K*) with a = (3c/4pi)^(1/3)
    a = (3 * 0.125 / (4 * np.pi)) ** (1 / 3)
    assert sc_sphere_permeability(0.125, 4.292) == pytest.approx(1 / (6 * np.pi * a * 4.292))
    # interpolated table: exact at the nodes, close to the series in between, monotone
    for c, k in ZICK_HOMSY_SC.items():
        assert zick_homsy_sc_drag(c) == pytest.approx(k, rel=1e-12)
    for c in (0.03, 0.05, 0.08, 0.1, 0.15):
        assert zick_homsy_sc_drag(c) == pytest.approx(sangani_acrivos_sc_drag(c), rel=4e-3)
    cs = np.linspace(0.01, 0.52, 200)
    assert np.all(np.diff(zick_homsy_sc_drag(cs)) > 0)
    with pytest.raises(ValueError):
        zick_homsy_sc_drag(0.6)


def sphere_error(n: int, c: float, **kw) -> tuple[float, object]:
    """Relative K error of a voxel sphere array vs Zick & Homsy at the REALIZED voxel fraction."""
    cell = sphere_array_cell(n, c)
    cv = float(cell.mean())
    r = permeability(cell, voxel_size=1 / n, **kw)
    k_ref = float(sc_sphere_permeability(cv, zick_homsy_sc_drag(cv)))
    return float(r.K[0, 0] / k_ref - 1.0), r


def test_sphere_cell_volume_matched():
    for n, c in ((16, 0.064), (32, 0.216), (48, 0.343)):
        s = sphere_array_cell(n, c)
        assert abs(s.mean() - c) < 0.07 * c  # shells of equal distance: a few % at small n
        # symmetric under the cube's axis permutations and mirrors
        assert np.array_equal(s, s.transpose(1, 2, 0)) and np.array_equal(s, s[::-1])
    assert abs(sphere_array_cell(64, 0.125).mean() / 0.125 - 1) < 1e-3


@pytest.mark.parametrize("c", [0.064, 0.125, 0.216, 0.343])
def test_sphere_array_vs_zick_homsy(c):
    """n = 48 voxels per period: within 1.2 % of Zick & Homsy (observed -0.7 ... -0.9 %)."""
    err, r = sphere_error(48, c, symmetry="cubic")
    assert -0.012 < err < 0.002
    assert r.is_spd and 1.0 < r.tortuosity[0] < 1.6


@pytest.mark.slow
@pytest.mark.parametrize("c", [0.027, 0.125, 0.216, 0.45])
def test_sphere_array_full_tensor_and_fine_grid(c):
    """Full 3-direction solve is isotropic (n = 32); n = 32 and 64 within 1 % of Zick & Homsy.

    With the default preconditioner (AMG when pyamg is installed) the off-diagonals show the
    solver error, ~1e-5 of K (AMG aggregation is not symmetric under the cube's rotations;
    seen on the Windows laptop). Jacobi preserves that symmetry, so there the discrete
    off-diagonals vanish to round-off -- checked separately to keep the exact-symmetry test.
    """
    err32, r = sphere_error(32, c)
    d = np.diag(r.K)
    assert np.ptp(d) < 1e-4 * d.mean()
    np.testing.assert_allclose(r.K - np.diag(d), 0.0, atol=1e-4 * d.mean())
    _, rj = sphere_error(32, c, preconditioner="jacobi")
    dj = np.diag(rj.K)
    assert np.ptp(dj) < 1e-6 * dj.mean()
    np.testing.assert_allclose(rj.K - np.diag(dj), 0.0, atol=1e-8 * dj.mean())
    err64, _ = sphere_error(64, c, symmetry="cubic")
    assert abs(err32) < 0.01 and abs(err64) < 0.01


# =============================================================================
# General properties, edge cases
# =============================================================================
def test_no_solid_raises_and_all_solid_is_zero():
    with pytest.raises(ValueError, match="no solid"):
        permeability(np.zeros((4, 4, 4), bool))
    r = permeability(np.ones((4, 4, 4), bool))
    assert np.all(r.K == 0.0) and r.porosity == 0.0
    with pytest.raises(TypeError):
        permeability(np.zeros((4, 4, 4)))
    with pytest.raises(ValueError):
        permeability(plates_cell(4), directions=[0], symmetry="cubic")


def test_sealed_cavity_carries_no_flow():
    """Isolated pockets add porosity but no superficial flow (u = 0, hydrostatic p)."""
    H = 6
    base = plates_cell(H, axis=1, wall=4, other=6)
    cav = base.copy()
    cav[2:4, 1:3, 2:4] = False  # pocket inside the 4-voxel wall
    r0 = permeability(base, tol=1e-12)
    r1 = permeability(cav, tol=1e-12, return_fields=True)
    np.testing.assert_allclose(r1.K, r0.K, rtol=1e-9, atol=1e-12 * r0.K.max())
    assert r1.porosity > r0.porosity
    assert r1.n_pressure_components == 2
    assert np.max(np.abs(r1.velocity[:, :, 2:4, 1:3, 2:4])) < 1e-9 * np.max(np.abs(r1.velocity))
    # fully closed cell: K = 0
    box = np.ones((6, 6, 6), bool)
    box[2:4, 2:4, 2:4] = False
    rb = permeability(box, tol=1e-12)
    np.testing.assert_allclose(rb.K, 0.0, atol=1e-14)
    assert all(np.isnan(t) for t in rb.tortuosity.values())


def test_system_structure():
    cell = voxelize(TPMSParams(w=0.0, rho=0.35), 16)
    sysm = assemble_stokes_system(cell)
    A = sysm.A
    assert abs(A - A.T).max() < 1e-14
    assert np.all(A.diagonal() >= 6) and np.all(A.diagonal() <= 10)
    S = sysm.saddle
    assert abs(S - S.T).max() < 1e-14
    # every face appears in exactly two cells with +1 / -1: D^T 1 = 0 (grad of a constant)
    np.testing.assert_array_equal(np.asarray(sysm.D.sum(axis=0)).ravel(), 0.0)
    assert sysm.n_pressure_components == 1
    # velocity divergence of a solution
    sol = solve_stokes(sysm, (1.0, 0.0, 0.0))
    assert sol.divergence < 1e-3 and sol.residual < 1e-3


def test_tolerance_convergence_of_K():
    cell = voxelize(TPMSParams(w=0.0, rho=0.3), 24)
    k_ref = permeability(cell, symmetry="cubic", tol=1e-11).K[0, 0]
    k8 = permeability(cell, symmetry="cubic", tol=1e-8).K[0, 0]
    k6 = permeability(cell, symmetry="cubic", tol=1e-6, check=False).K[0, 0]
    assert abs(k8 / k_ref - 1) < 1e-6
    assert abs(k6 / k_ref - 1) < 1e-2
    with pytest.raises(RuntimeError, match="disagree"):  # sloppy solve: ~1e-2 disagreement
        permeability(cell, symmetry="cubic", tol=1e-4)


def test_kozeny_constant_slit():
    """Slit: phi^3 / (K a_sf^2) = 3 exactly (K = phi H^2/12, a_sf = 2/P)."""
    H, P = 3.0, 5.0
    phi = H / P
    assert float(kozeny_constant(phi * H * H / 12, phi, 2 / P)) == pytest.approx(3.0)


# =============================================================================
# (d) TPMS cells
# =============================================================================
@pytest.mark.parametrize("w", [0.0, 1.0])
def test_tpms_cubic_symmetry(w):
    """Full 3-direction solve of gyroid / diamond: K = k I to alignment precision."""
    cell = voxelize(TPMSParams(w=w, rho=0.35), 24)
    r = permeability(cell, voxel_size=1 / 24)
    d = np.diag(r.K)
    # voxel alignment breaks the cubic symmetry at the 1e-3 level for n = 24
    assert np.ptp(d) < 4e-3 * d.mean()
    np.testing.assert_allclose(r.K - np.diag(d), 0.0, atol=4e-3 * d.mean())
    assert r.is_spd and r.agreement < 1e-4
    rc = permeability(cell, voxel_size=1 / 24, symmetry="cubic")
    assert rc.K[0, 0] == pytest.approx(r.K[0, 0], rel=1e-6)
    assert tpms_symmetry(TPMSParams(w=w, rho=0.35)) == "cubic"
    assert 1.0 < r.tortuosity[0] < 2.0


def test_tpms_blend_trigonal():
    """w = 0.5: invariant under x -> y -> z, so K = a I + b (J - I); principal axis [111]."""
    p = TPMSParams(w=0.5, rho=0.35)
    assert tpms_symmetry(p) == "none"
    r = permeability_tpms(p, 24)
    K = r.K
    d = np.diag(K)
    off = K[np.triu_indices(3, 1)]
    assert np.ptp(d) < 4e-3 * d.mean() and np.ptp(off) < 4e-3 * d.mean()
    assert r.is_spd
    _ev, vec = np.linalg.eigh(K)
    axis = np.ones(3) / np.sqrt(3)
    # the non-degenerate eigenvector is [111]
    i = int(np.argmax(np.abs(vec.T @ axis)))
    assert abs(vec[:, i] @ axis) > 0.999


def test_tpms_stretched_tetragonal():
    p = TPMSParams(w=0.0, rho=0.35, a=(1.0, 1.0, 1.5))
    assert tpms_symmetry(p) == "tetragonal_z"
    full = permeability_tpms(p, 16, symmetry="none")
    auto = permeability_tpms(p, 16)
    assert auto.directions == (0, 2)
    np.testing.assert_allclose(np.diag(auto.K), np.diag(full.K), rtol=2e-3)
    assert full.K[2, 2] > 1.1 * full.K[0, 0]  # stretched cells conduct flow better along z
    np.testing.assert_allclose(full.K - np.diag(np.diag(full.K)), 0.0, atol=5e-3 * full.mean)


def test_invariances():
    cell = voxelize(TPMSParams(w=0.5, rho=0.4), 16)
    r = permeability(cell)
    rr = permeability(np.roll(cell, (3, -5, 7), axis=(0, 1, 2)))
    np.testing.assert_allclose(rr.K, r.K, rtol=1e-6, atol=1e-6 * r.mean)
    perm = (2, 0, 1)
    rp = permeability(cell.transpose(perm))
    P = np.eye(3)[list(perm)]
    np.testing.assert_allclose(rp.K, P @ r.K @ P.T, rtol=1e-6, atol=1e-6 * r.mean)


def test_tpms_fields_consistency():
    """Staggered mean = K, discrete divergence ~ 0, u = 0 on all faces touching solid."""
    n = 20
    cell = voxelize(TPMSParams(w=1.0, rho=0.3), n)
    r = permeability(cell, voxel_size=1 / n, symmetry="cubic", return_fields=True)
    v = r.velocity[0].astype(float)
    assert v[0].mean() == pytest.approx(r.K[0, 0], rel=1e-5)
    div = sum(v[d] - np.roll(v[d], 1, axis=d) for d in range(3))
    assert np.max(np.abs(div[~cell])) < 1e-3 * np.max(np.abs(v)) and np.all(div[cell] == 0)
    for d in range(3):
        wall_face = cell | np.roll(cell, -1, axis=d)
        assert np.all(v[d][wall_face] == 0.0)
    uc = r.cell_velocity(0)
    assert uc.shape == (3, n, n, n)
    assert uc[0].mean() == pytest.approx(r.K[0, 0], rel=1e-5)
    assert np.isnan(r.pressure[0][cell]).all() and np.isfinite(r.pressure[0][~cell]).all()
    # interstitial velocity, tortuosity consistency
    assert r.interstitial_velocity[0, 0] == pytest.approx(r.K[0, 0] / r.porosity)
    speed = np.sqrt((uc**2).sum(axis=0))
    assert r.tortuosity[0] == pytest.approx(speed.sum() / abs(uc[0].sum()), rel=1e-4)  # float32 field


def test_density_ordering():
    """Denser lattices are less permeable; diamond is less permeable than gyroid at equal rho."""
    kG = [permeability_tpms(TPMSParams(w=0.0, rho=r), 24).K[0, 0] for r in (0.25, 0.35, 0.45)]
    kD = [permeability_tpms(TPMSParams(w=1.0, rho=r), 24).K[0, 0] for r in (0.25, 0.35, 0.45)]
    assert kG[0] > kG[1] > kG[2] and kD[0] > kD[1] > kD[2]
    assert all(g > d for g, d in zip(kG, kD))


def test_extrapolated_and_averaged_wrappers():
    p = TPMSParams(w=0.0, rho=0.35)
    e = extrapolated_permeability_tpms(p, (16, 24))
    k16, k24 = e.coarse.K[0, 0], e.fine.K[0, 0]
    assert e.K[0, 0] == pytest.approx(richardson(16, k16, 24, k24, p=1), rel=1e-12)
    assert e.correction < 0.05 and e.porosity == pytest.approx(0.65, abs=0.01)
    a = averaged_permeability_tpms(p, 16, n_offsets=3)
    assert len(a.results) == 3 and np.all(a.K_std[0, 0] < 0.05 * a.K[0, 0])
    with pytest.raises(ValueError):
        extrapolated_permeability_tpms(p, (24, 16))


@pytest.mark.slow
def test_tpms_convergence_richardson():
    """Offset-averaged gyroid rho* = 0.35, n = 16..64: K converges; R(32, 64) agrees with the
    free fit of all grids to < 0.5 %, and every grid from n = 32 on is within 1.5 %."""
    p = TPMSParams(w=0.0, rho=0.35)
    ns = [16, 24, 32, 48, 64]
    k = [averaged_permeability_tpms(p, n, n_offsets=3).K[0, 0] for n in ns]
    fit = fit_convergence(ns[1:], k[1:], p=1.0)
    for n, kn in zip(ns[2:], k[2:]):
        assert abs(kn / fit.f_inf - 1) < 0.015
    assert abs(richardson(32, k[2], 64, k[4]) / fit.f_inf - 1) < 0.005


@pytest.mark.slow
def test_amg_matches_jacobi():
    if not pyamg_available():
        pytest.skip("pyamg not installed")
    cell = voxelize(TPMSParams(w=0.5, rho=0.35), 24)
    rj = permeability(cell, preconditioner="jacobi")
    ra = permeability(cell, preconditioner="amg")
    # same tol, different preconditioned norm -> AMG stops at a looser true residual;
    # observed on the laptop: max |K_amg - K_jacobi| = 2e-6 of mean(K) (discretization ~1e-2)
    np.testing.assert_allclose(ra.K, rj.K, rtol=1e-5, atol=1e-5 * rj.mean)
    assert ra.preconditioner == "amg" and rj.preconditioner == "jacobi"
    # (speed/iterations are a performance question: scripts/task4_benchmark_preconditioner.py)
