"""Task 2 verification of the periodic conduction homogenization (voxlat.homogenization.conduction)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from voxlat.geometry import TPMSParams, voxelize
from voxlat.homogenization.conduction import (
    assemble_conduction_system,
    effective_conductivity,
    effective_conductivity_tpms,
    extrapolated_conductivity_tpms,
    face_conductances,
    hashin_shtrikman_bounds,
    pyamg_available,
    rayleigh_sc_spheres,
    wiener_bounds,
)
from voxlat.homogenization.convergence import (
    fit_convergence,
    observed_order,
    richardson,
)
from voxlat.utils.config import load_config

CFG = load_config()
K_S = CFG.material.conductivity  # 130 W/(m K)
K_F = CFG.coolant.conductivity  # 0.40 W/(m K)
CONTRASTS = [K_F / K_S, 1e-3, 0.0]  # config (1/325), 1e3 contrast, insulating fluid


# -----------------------------------------------------------------------------
# Exact cases
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(6, 6, 6), (5, 7, 9), (1, 4, 3)])
@pytest.mark.parametrize("all_solid", [True, False])
def test_homogeneous_is_exact(shape, all_solid):
    cell = np.full(shape, all_solid)
    r = effective_conductivity(cell, K_S, K_F)
    k = K_S if all_solid else K_F
    np.testing.assert_allclose(r.k_eff, k * np.eye(3), rtol=1e-13, atol=1e-13 * k)
    np.testing.assert_allclose(r.k_eff_flux, k * np.eye(3), rtol=1e-13, atol=1e-13 * k)
    assert r.iterations == [0, 0, 0]  # theta = 0 is the exact solution, b = 0


def test_homogeneous_float_field():
    r = effective_conductivity(np.full((4, 5, 6), 3.7))
    np.testing.assert_allclose(r.k_eff, 3.7 * np.eye(3), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("kf_ks", [0.1, 1e-3])
def test_two_phase_laminate_exact(axis, kf_ks):
    """Layers normal to `axis`: harmonic mean across, arithmetic mean along (exact)."""
    shape = [10, 8, 6]
    cell = np.zeros(shape, bool)
    sl = [slice(None)] * 3
    sl[axis] = slice(0, 3 * shape[axis] // 10 + 1)
    cell[tuple(sl)] = True
    phi = cell.mean()
    k_f = kf_ks * K_S
    r = effective_conductivity(cell, K_S, k_f, tol=1e-12)
    harm = 1.0 / (phi / K_S + (1 - phi) / k_f)
    arith = phi * K_S + (1 - phi) * k_f
    expected = np.full(3, arith)
    expected[axis] = harm
    np.testing.assert_allclose(np.diag(r.k_eff), expected, rtol=1e-9)
    np.testing.assert_allclose(r.k_eff - np.diag(np.diag(r.k_eff)), 0.0, atol=1e-10 * K_S)
    np.testing.assert_allclose(r.k_eff_flux, r.k_eff, rtol=0, atol=1e-9 * K_S)


def test_random_multilayer_laminate_exact():
    rng = np.random.default_rng(7)
    kz = rng.uniform(0.01, 100.0, size=13)  # 13 layers with random conductivity along z
    k = np.broadcast_to(kz[None, None, :], (3, 4, 13)).copy()
    r = effective_conductivity(k, tol=1e-12)
    np.testing.assert_allclose(r.k_eff[2, 2], 1.0 / np.mean(1.0 / kz), rtol=1e-9)
    np.testing.assert_allclose(r.k_eff[0, 0], np.mean(kz), rtol=1e-12)
    np.testing.assert_allclose(r.k_eff[1, 1], np.mean(kz), rtol=1e-12)


def test_laminate_insulating_fluid_limit():
    """k_f = 0: no conduction across the layers, phi*k_s along them."""
    cell = np.zeros((8, 5, 4), bool)
    cell[:3] = True
    r = effective_conductivity(cell, K_S, 0.0)
    assert abs(r.k_eff[0, 0]) < 1e-12 * K_S
    np.testing.assert_allclose(r.k_eff[1, 1], 3 / 8 * K_S, rtol=1e-12)
    np.testing.assert_allclose(r.k_eff[2, 2], 3 / 8 * K_S, rtol=1e-12)
    assert r.n_active == 3 * 5 * 4  # fluid voxels are removed from the system
    assert not r.is_spd  # zero eigenvalue across a non-percolating direction


def test_floating_island_insulating_fluid():
    """A detached solid island adds nothing when k_f = 0 (singular but consistent blocks)."""
    slab = np.zeros((12, 12, 12), bool)
    slab[:, :, :4] = True  # conducting slab in x-y
    island = slab.copy()
    island[4:8, 4:8, 6:10] = True  # floating cube in the insulating fluid
    r0 = effective_conductivity(slab, K_S, 0.0)
    r1 = effective_conductivity(island, K_S, 0.0)
    np.testing.assert_allclose(r1.k_eff, r0.k_eff, rtol=0, atol=1e-9 * K_S)


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------
def test_face_conductance_is_harmonic_and_periodic():
    k = np.array([2.0, 6.0, 0.0, 6.0]).reshape(4, 1, 1)
    kf = face_conductances(k)[0].ravel()
    # faces 0|1, 1|2, 2|3, 3|0 (wrap)
    np.testing.assert_allclose(kf, [3.0, 0.0, 0.0, 3.0])


def test_matrix_is_symmetric_laplacian():
    rng = np.random.default_rng(3)
    k = rng.uniform(0.1, 10.0, size=(5, 6, 7))
    A, active = assemble_conduction_system(face_conductances(k))
    assert active.all()
    assert abs(A - A.T).max() < 1e-14
    np.testing.assert_allclose(A @ np.ones(A.shape[0]), 0.0, atol=1e-12)  # constant null vector
    assert np.all(A.diagonal() > 0)


# -----------------------------------------------------------------------------
# Flux = energy, SPD, invariances
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("kf_ks", CONTRASTS)
def test_random_cell_spd_and_flux_energy_agree(kf_ks):
    rng = np.random.default_rng(11)
    cell = rng.random((12, 14, 10)) < 0.6  # percolating random solid
    r = effective_conductivity(cell, K_S, kf_ks * K_S)
    assert r.is_spd
    assert r.agreement < 1e-6
    np.testing.assert_allclose(r.k_eff, r.k_eff.T, rtol=0, atol=1e-14 * K_S)
    lo, hi = wiener_bounds(cell.mean(), K_S, kf_ks * K_S)
    assert np.all(r.eigenvalues >= lo * (1 - 1e-9)) and np.all(r.eigenvalues <= hi * (1 + 1e-9))


def test_disagreement_is_detected():
    """With a sloppy solve the internal flux/energy check must fire."""
    cell = voxelize(TPMSParams(w=0.0, rho=0.3), 24)
    with pytest.raises(RuntimeError, match="disagree"):
        effective_conductivity(cell, K_S, K_F, tol=1e-2, agreement_tol=1e-12, preconditioner="jacobi")


def test_translation_and_permutation_invariance():
    cell = voxelize(TPMSParams(w=1.0, rho=0.35, a=(1.0, 1.0, 1.25)), 24)
    r0 = effective_conductivity(cell, K_S, K_F, tol=1e-11)
    r1 = effective_conductivity(np.roll(cell, (5, -3, 7), axis=(0, 1, 2)), K_S, K_F, tol=1e-11)
    np.testing.assert_allclose(r1.k_eff, r0.k_eff, rtol=0, atol=1e-8 * K_S)
    perm = (2, 0, 1)
    r2 = effective_conductivity(np.transpose(cell, perm), K_S, K_F, tol=1e-11)
    np.testing.assert_allclose(r2.k_eff, r0.k_eff[np.ix_(perm, perm)], rtol=0, atol=1e-8 * K_S)


def test_continuity_to_insulating_limit():
    cell = voxelize(TPMSParams(w=0.0, rho=0.3), 24)
    r0 = effective_conductivity(cell, K_S, 0.0, tol=1e-10)
    r1 = effective_conductivity(cell, K_S, 1e-7 * K_S, tol=1e-10)
    np.testing.assert_allclose(r1.k_eff, r0.k_eff, rtol=1e-5, atol=1e-6 * K_S)


def test_anisotropic_cell():
    """a_z = 1.5 stretches the struts along z -> k_zz > k_xx = k_yy (cyclic G symmetry broken)."""
    r = effective_conductivity_tpms(TPMSParams(w=0.0, rho=0.3, a=(1, 1, 1.5)), 24, K_S, K_F)
    assert r.shape == (24, 24, 36)
    assert r.is_spd
    k = np.diag(r.k_eff)
    assert k[2] > 1.05 * k[0]
    # x and y are equivalent by the gyroid's 4_1 screw axis along z; voxels whose field value
    # ties the threshold can flip with the platform's sin/cos round-off (measured up to 7e-4)
    np.testing.assert_allclose(k[0], k[1], rtol=3e-3)


# -----------------------------------------------------------------------------
# TPMS: bounds, isotropy, analytic sphere array
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("w", [0.0, 1.0])
@pytest.mark.parametrize("rho", [0.2, 0.35, 0.5])
@pytest.mark.parametrize("kf_ks", CONTRASTS)
def test_hashin_shtrikman_bounds_gyroid_diamond(w, rho, kf_ks):
    cell = voxelize(TPMSParams(w=w, rho=rho), 32)
    r = effective_conductivity(cell, K_S, kf_ks * K_S)
    lo, hi = hashin_shtrikman_bounds(cell.mean(), K_S, kf_ks * K_S)
    ev = r.eigenvalues
    assert np.all(ev >= lo * (1 - 1e-6)), (ev, lo)
    assert np.all(ev <= hi * (1 + 1e-6)), (ev, hi)
    assert r.agreement < 1e-6
    assert r.is_spd


def test_blend_within_wiener_bounds():
    cell = voxelize(TPMSParams(w=0.5, rho=0.35), 32)
    r = effective_conductivity(cell, K_S, K_F)
    lo, hi = wiener_bounds(cell.mean(), K_S, K_F)
    assert lo < r.eigenvalues[0] and r.eigenvalues[-1] < hi


@pytest.mark.parametrize("w", [0.0, 1.0])
@pytest.mark.parametrize("n", [32, 48])
def test_near_isotropy_cubic_cells(w, n):
    r = effective_conductivity_tpms(TPMSParams(w=w, rho=0.3), n, K_S, K_F)
    d = np.diag(r.k_eff)
    # G, D are invariant under x->y->z; only voxels whose field value ties the threshold
    # (round-off, platform-dependent sin/cos) break it: diagonal equal to ~1e-4 ... 1e-3
    np.testing.assert_allclose(d, d.mean(), rtol=3e-3)
    off = r.k_eff - np.diag(d)
    assert np.max(np.abs(off)) < 3e-3 * d.mean()
    assert r.anisotropy < 5e-3


def test_blend_is_uniaxial_about_111():
    """G and D share only the 3-fold [111] axes (+ inversion-type ops) -> a 50/50 blend
    is trigonal: k_eff is uniaxial with its distinct axis along [111]."""
    r = effective_conductivity_tpms(TPMSParams(w=0.5, rho=0.3), 32, K_S, K_F)
    vals, vecs = np.linalg.eigh(r.k_eff)
    # two equal eigenvalues, one distinct
    i_distinct = int(np.argmax([abs(vals[i] - np.delete(vals, i).mean()) for i in range(3)]))
    others = np.delete(vals, i_distinct)
    assert abs(others[0] - others[1]) < 3e-3 * vals.max()  # round-off ties, see above
    assert abs(vals[i_distinct] - others.mean()) > 0.3 * vals.mean()  # strongly anisotropic
    axis = np.abs(vecs[:, i_distinct])
    np.testing.assert_allclose(axis, np.full(3, 1 / np.sqrt(3)), atol=1e-2)


@pytest.mark.parametrize("phi", [0.05, 0.15, 0.3])
def test_sc_sphere_array_vs_rayleigh(phi):
    """Simple-cubic array of spheres vs Rayleigh's formula (at the voxelized volume fraction)."""
    errs = {}
    for n in (24, 48):
        x = (np.arange(n) + 0.5) / n - 0.5
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        R = (3 * phi / (4 * np.pi)) ** (1 / 3)
        cell = X**2 + Y**2 + Z**2 <= R**2
        for kp in (10.0, 0.0):
            r = effective_conductivity(cell, kp, 1.0)
            np.testing.assert_allclose(np.diag(r.k_eff), r.mean, rtol=1e-6)
            errs[(n, kp)] = r.mean / rayleigh_sc_spheres(cell.mean(), kp, 1.0) - 1.0
    # conducting spheres: < 1 % already at n = 48
    assert abs(errs[(48, 10.0)]) < 0.01
    # insulating spheres: first-order staircase error, roughly halved by doubling n
    assert abs(errs[(48, 0.0)]) < 0.025
    if abs(errs[(24, 0.0)]) > 2e-3:
        assert 0.35 < errs[(48, 0.0)] / errs[(24, 0.0)] < 0.7


# -----------------------------------------------------------------------------
# Convergence, extrapolation, timing
# -----------------------------------------------------------------------------
def test_convergence_from_below_gyroid():
    """Staircase removes diagonal contacts -> k_eff(n) increases with n (G, D)."""
    ks = [effective_conductivity_tpms(TPMSParams(w=0.0, rho=0.3), n, K_S, K_F).mean for n in (16, 32, 48)]
    assert ks[0] < ks[1] < ks[2]


@pytest.mark.slow
@pytest.mark.parametrize("w", [0.0, 1.0])
def test_observed_order_is_one(w):
    """Offset-averaged triple n = 16, 32, 64 -> observed order ~1 (staircase)."""
    rng = np.random.default_rng(2026)
    offs = [(0.0, 0.0, 0.0)] + [tuple(o) for o in rng.random((2, 3))]
    f = []
    for n in (16, 32, 64):
        f.append(np.mean([
            effective_conductivity_tpms(TPMSParams(w=w, rho=0.3), n, K_S, K_F, offset=o).mean
            for o in offs
        ]))
    p = observed_order((16, 32, 64), f)
    assert 0.75 < p < 1.35, p


@pytest.mark.slow
def test_extrapolated_close_to_reference():
    """R(32, 64) vs the reference from the offset-averaged (48, 96) study (STATUS.md, Task 2)."""
    r = extrapolated_conductivity_tpms(TPMSParams(w=0.0, rho=0.3), (32, 64), K_S, K_F)
    k_ref = 0.1695 * K_S  # gyroid rho = 0.3, k_f/k_s = 1/325; Richardson(48, 96), 3 offsets
    assert abs(r.mean / k_ref - 1) < 0.015
    assert r.coarse.mean < r.fine.mean < r.mean


@pytest.mark.slow
def test_timing_n48_under_30s():
    t0 = time.perf_counter()
    r = effective_conductivity_tpms(TPMSParams(w=0.5, rho=0.35), 48, K_S, K_F)
    dt = time.perf_counter() - t0
    print(f"\nn=48 blend: {dt:.2f} s total, iterations {r.iterations}, pc={r.preconditioner}")
    assert dt < 30.0


@pytest.mark.skipif(not pyamg_available(), reason="pyamg not installed")
def test_amg_matches_jacobi():
    cell = voxelize(TPMSParams(w=1.0, rho=0.3), 32)
    a = effective_conductivity(cell, K_S, K_F, preconditioner="amg")
    j = effective_conductivity(cell, K_S, K_F, preconditioner="jacobi")
    assert a.preconditioner == "amg"
    np.testing.assert_allclose(a.k_eff, j.k_eff, rtol=1e-6, atol=1e-8 * K_S)


def test_bad_inputs():
    with pytest.raises(ValueError):
        effective_conductivity(np.ones((4, 4), bool), 1.0, 1.0)
    with pytest.raises(ValueError):
        effective_conductivity(np.ones((4, 4, 4), bool))
    with pytest.raises(ValueError):
        effective_conductivity(np.zeros((4, 4, 4)))
    with pytest.raises(ValueError):
        effective_conductivity(np.ones((4, 4, 4), bool), -1.0, 1.0)


def test_result_row_and_fields():
    r = effective_conductivity_tpms(TPMSParams(w=0.0, rho=0.3), 16, K_S, K_F, return_fields=True)
    assert r.theta.shape == (3, 16, 16, 16)
    row = r.as_row()
    assert set(row) >= {"k_xx", "k_yz", "k_mean", "k_anisotropy", "k_wall_time_s"}
    np.testing.assert_allclose(r.relative(), r.k_eff / K_S)


# -----------------------------------------------------------------------------
# Analytic bounds and convergence helpers
# -----------------------------------------------------------------------------
def test_hs_bounds_formulas():
    lo, hi = hashin_shtrikman_bounds(np.array([0.0, 1.0]), 10.0, 1.0)
    np.testing.assert_allclose(lo, [1.0, 10.0])
    np.testing.assert_allclose(hi, [1.0, 10.0])
    lo, hi = hashin_shtrikman_bounds(0.3, 1.0, 0.0)  # insulating fluid: 2 phi / (3 - phi)
    assert lo == 0.0
    np.testing.assert_allclose(hi, 2 * 0.3 / 2.7)
    phi = np.linspace(0.01, 0.99, 50)
    lo, hi = hashin_shtrikman_bounds(phi, 130.0, 0.4)
    wl, wh = wiener_bounds(phi, 130.0, 0.4)
    assert np.all(wl <= lo + 1e-12) and np.all(lo <= hi) and np.all(hi <= wh + 1e-12)
    # HS upper == Maxwell-Garnett with the conductor as matrix; symmetric in phase labels
    l2, h2 = hashin_shtrikman_bounds(1 - phi, 0.4, 130.0)
    np.testing.assert_allclose(l2, lo)
    np.testing.assert_allclose(h2, hi)


def test_convergence_helpers_recover_synthetic():
    n = np.array([16, 24, 32, 48, 64, 96.0])
    f = 2.0 - 3.0 * n**-1.0
    fit = fit_convergence(n, f, p=1.0)
    assert abs(fit.f_inf - 2.0) < 1e-12 and abs(fit.C + 3.0) < 1e-10
    free = fit_convergence(n, 2.0 - 3.0 * n**-1.6, p=None)
    assert abs(free.p - 1.6) < 1e-4 and abs(free.f_inf - 2.0) < 1e-6
    assert abs(richardson(32, 2 - 3 / 32, 64, 2 - 3 / 64) - 2.0) < 1e-12
    assert abs(observed_order((16, 32, 64), [2 - 3 / 16**2, 2 - 3 / 32**2, 2 - 3 / 64**2]) - 2.0) < 1e-12
    assert abs(fit.rel_error(48) - (-3 / 48) / 2) < 1e-12


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e9])
def test_free_order_fit_is_scale_invariant(scale):
    """Regression (found in Task 4): the free-order fit stalled at p0 = 1 for |f| ~ 1e-3."""
    n = np.array([16, 24, 32, 48, 64.0])
    free = fit_convergence(n, scale * (2.0 - 3.0 * n**-1.6), p=None)
    assert abs(free.p - 1.6) < 1e-4
    assert abs(free.f_inf / scale - 2.0) < 1e-6 and abs(free.C / scale + 3.0) < 1e-3
