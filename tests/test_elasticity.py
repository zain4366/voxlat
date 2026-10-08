"""Task 3 verification of the periodic voxel-FEA elasticity homogenization (voxlat.homogenization.elasticity)."""

from __future__ import annotations

import numpy as np
import pytest

from voxlat.geometry import TPMSParams, voxelize
from voxlat.homogenization.elasticity import (
    MACRO_STRESS_CASES,
    cubic_deviation,
    effective_elasticity,
    effective_elasticity_tpms,
    extrapolated_elasticity_tpms,
    hashin_shtrikman_porous,
    hex8_element,
    isotropic_stiffness,
    laminate_stiffness,
    lame_parameters,
    mandel_to_voigt,
    averaged_elasticity_tpms,
    grid_offsets,
    pyamg_available,
    rotate_stiffness,
    voigt_reuss_hill,
    von_mises,
    voigt_to_mandel,
)
from voxlat.utils.config import load_config

CFG = load_config()
E_S = CFG.material.youngs_modulus  # 70 GPa
NU = CFG.material.poisson_ratio  # 0.33
CYCLIC = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # x -> y -> z -> x


def _rel(A, B):
    return float(np.max(np.abs(np.asarray(A) - np.asarray(B))) / np.max(np.abs(B)))


# -----------------------------------------------------------------------------
# Element and tensor helpers
# -----------------------------------------------------------------------------
def test_hex8_element_rigid_modes_and_patch():
    K1, F1, Bc, D1 = hex8_element(NU)
    np.testing.assert_allclose(K1, K1.T, atol=1e-14)
    ev = np.linalg.eigvalsh(K1)
    assert np.all(np.abs(ev[:6]) < 1e-12) and ev[6] > 1e-3  # exactly 6 rigid-body modes
    # constant-strain patch test: u = eps . x at the nodes -> B u = eps everywhere
    corners = np.array([[a & 1, (a >> 1) & 1, (a >> 2) & 1] for a in range(8)], float)
    rng = np.random.default_rng(0)
    eps_t = rng.normal(size=(3, 3))
    eps_t = 0.5 * (eps_t + eps_t.T)
    u = (corners @ eps_t.T).ravel()
    ev_voigt = np.array([eps_t[0, 0], eps_t[1, 1], eps_t[2, 2], 2 * eps_t[1, 2], 2 * eps_t[0, 2], 2 * eps_t[0, 1]])
    np.testing.assert_allclose(Bc @ u, ev_voigt, atol=1e-14)
    np.testing.assert_allclose(K1 @ u, F1 @ ev_voigt, atol=1e-14)  # int B^T D B u = int B^T D eps
    # F1 is self-equilibrated: the nodal forces of any eigenstrain sum to zero per direction
    np.testing.assert_allclose(F1.reshape(8, 3, 6).sum(axis=0), 0.0, atol=1e-14)


def test_isotropic_stiffness_and_lame():
    lam, mu = lame_parameters(E_S, NU)
    C = isotropic_stiffness(E_S, NU)
    S = np.linalg.inv(C)
    np.testing.assert_allclose(1 / S[0, 0], E_S, rtol=1e-13)
    np.testing.assert_allclose(-S[0, 1] / S[0, 0], NU, rtol=1e-13)
    np.testing.assert_allclose(C[3, 3], mu, rtol=1e-13)
    np.testing.assert_allclose(C[0, 1], lam, rtol=1e-13)
    with pytest.raises(ValueError):
        lame_parameters(1.0, 0.5)


def test_voigt_mandel_and_rotation():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 6))
    C = A @ A.T + 6 * np.eye(6)
    np.testing.assert_allclose(mandel_to_voigt(voigt_to_mandel(C)), C, rtol=1e-14)
    # isotropic tensor is rotation invariant
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    Ci = isotropic_stiffness(E_S, NU)
    np.testing.assert_allclose(rotate_stiffness(Ci, R), Ci, rtol=1e-12, atol=1e-14 * E_S)
    # rotations compose and preserve the Kelvin moduli
    Cr = rotate_stiffness(C, R)
    np.testing.assert_allclose(np.linalg.eigvalsh(voigt_to_mandel(Cr)), np.linalg.eigvalsh(voigt_to_mandel(C)), rtol=1e-12)
    np.testing.assert_allclose(rotate_stiffness(Cr, R.T), C, rtol=1e-12, atol=1e-12)
    # a cubic tensor is invariant under the cyclic axis permutation and a 90 deg turn about z
    Cc = np.zeros((6, 6))
    Cc[:3, :3] = 0.4
    Cc[[0, 1, 2], [0, 1, 2]] = 1.0
    Cc[[3, 4, 5], [3, 4, 5]] = 0.2
    R90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(rotate_stiffness(Cc, CYCLIC), Cc, atol=1e-14)
    np.testing.assert_allclose(rotate_stiffness(Cc, R90), Cc, atol=1e-14)


def test_voigt_reuss_hill():
    lam, mu = lame_parameters(E_S, NU)
    v = voigt_reuss_hill(isotropic_stiffness(E_S, NU))
    for k in ("K_V", "K_R", "K_H"):
        assert v[k] == pytest.approx(lam + 2 * mu / 3, rel=1e-12)
    assert v["G_H"] == pytest.approx(mu, rel=1e-12) and v["E_H"] == pytest.approx(E_S, rel=1e-12)
    assert abs(v["A_U"]) < 1e-12
    # cubic: K_V = K_R and A^U = (6/5) (sqrt(A) - 1/sqrt(A))^2 (Ranganathan & Ostoja-Starzewski 2008)
    Cc = np.zeros((6, 6))
    Cc[:3, :3] = 0.4
    Cc[[0, 1, 2], [0, 1, 2]] = 1.0
    Cc[[3, 4, 5], [3, 4, 5]] = 0.5
    A = 2 * 0.5 / (1.0 - 0.4)
    v = voigt_reuss_hill(Cc)
    assert v["K_V"] == pytest.approx(v["K_R"], rel=1e-12)
    assert v["A_U"] == pytest.approx(1.2 * (np.sqrt(A) - 1 / np.sqrt(A)) ** 2, rel=1e-10)


def test_von_mises():
    np.testing.assert_allclose(von_mises(MACRO_STRESS_CASES["uniaxial_z"]), 1.0)
    np.testing.assert_allclose(von_mises(MACRO_STRESS_CASES["shear_xz"]), np.sqrt(3.0))
    np.testing.assert_allclose(von_mises(np.array([2.0, 2.0, 2.0, 0, 0, 0])), 0.0)  # hydrostatic


# -----------------------------------------------------------------------------
# Exact cases
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(6, 6, 6), (5, 7, 9), (1, 4, 3)])
def test_homogeneous_is_exact(shape):
    r = effective_elasticity(np.ones(shape, bool), E_S, NU)
    C = isotropic_stiffness(E_S, NU)
    np.testing.assert_allclose(r.C_eff, C, rtol=0, atol=1e-13 * E_S)
    np.testing.assert_allclose(r.C_eff_flux, C, rtol=0, atol=1e-13 * E_S)
    assert r.iterations == [0] * 6  # u~ = 0 is the exact solution (f = 0)
    for loc in r.localization.values():  # uniform stress -> factor 1 everywhere
        assert loc.max == pytest.approx(1.0, abs=1e-6) and loc.p99 == pytest.approx(1.0, abs=1e-6)


def test_homogeneous_float_field():
    r = effective_elasticity(np.full((4, 5, 6), 3.0e9), nu=0.25)
    np.testing.assert_allclose(r.C_eff, isotropic_stiffness(3.0e9, 0.25), atol=1e-13 * 3e9)
    assert r.E_s is None and r.solid_fraction is None


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("contrast", [0.1, 1e-3])
def test_two_phase_laminate_exact(axis, contrast):
    """Layers normal to `axis`: exact Backus stiffness, incl. Reuss (normal, transverse shear)
    and Voigt (in-plane shear) components."""
    shape = [10, 8, 6]
    cell = np.zeros(shape, bool)
    sl = [slice(None)] * 3
    sl[axis] = slice(0, shape[axis] // 2 - 1)
    cell[tuple(sl)] = True
    f = float(cell.mean())
    Efield = np.where(cell, E_S, contrast * E_S)
    r = effective_elasticity(Efield, nu=NU)
    Cref = laminate_stiffness([f, 1 - f], [E_S, contrast * E_S], [NU, NU], axis=axis)
    assert _rel(r.C_eff, Cref) < 1e-9
    assert r.agreement < 1e-9
    # named Voigt / Reuss checks
    mu1, mu2 = E_S / (2 * (1 + NU)), contrast * E_S / (2 * (1 + NU))
    reuss_mu = 1 / (f / mu1 + (1 - f) / mu2)
    voigt_mu = f * mu1 + (1 - f) * mu2
    # shear components that involve the normal direction are Reuss, the in-plane one Voigt
    idx = {0: (3, (4, 5)), 1: (4, (3, 5)), 2: (5, (3, 4))}[axis]
    np.testing.assert_allclose(r.C_eff[idx[0], idx[0]], voigt_mu, rtol=1e-9)
    for k in idx[1]:
        np.testing.assert_allclose(r.C_eff[k, k], reuss_mu, rtol=1e-9)


def test_three_layer_random_laminate_exact():
    rng = np.random.default_rng(3)
    layers = rng.choice([1.0, 0.3, 0.05], size=13)
    Efield = np.broadcast_to(layers[None, :, None] * E_S, (4, 13, 5)).copy()
    r = effective_elasticity(Efield, nu=NU)
    fr = [np.mean(layers == v) for v in (1.0, 0.3, 0.05)]
    Cref = laminate_stiffness(fr, [E_S, 0.3 * E_S, 0.05 * E_S], [NU] * 3, axis=1)
    assert _rel(r.C_eff, Cref) < 1e-9


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_void_laminate_exact(axis):
    """Solid plates separated by void layers: plane-stress in-plane terms, zero elsewhere.
    The plates are disconnected (2 components): checks void removal + null-space handling."""
    shape = [12, 8, 10]
    cell = np.zeros(shape, bool)
    sl = [slice(None)] * 3
    for a, b in ((0, 3), (5, 7)):  # plates separated by void layers incl. across the wrap
        sl[axis] = slice(a, b)
        cell[tuple(sl)] = True
    f = float(cell.mean())
    r = effective_elasticity(cell, E_S, NU)
    Cref = laminate_stiffness([f, 1 - f], [E_S, 0.0], [NU, NU], axis=axis)
    assert np.max(np.abs(r.C_eff - Cref)) < 1e-12 * E_S
    assert r.n_components == 2
    # plane-stress check of one in-plane term
    j = (axis + 1) % 3
    np.testing.assert_allclose(r.C_eff[j, j], f * E_S / (1 - NU**2), rtol=1e-12)
    # uniaxial in-plane stress -> uniform plate stress Sigma/f: factor exactly 1/f
    in_plane = np.zeros(6)
    in_plane[j] = 1.0
    rf = effective_elasticity(cell, E_S, NU, return_fields=True)
    loc = rf.localize(in_plane)
    for v in (loc.max, loc.p99, loc.mean):
        assert v == pytest.approx(1.0 / f, rel=1e-6)  # unit stresses are stored as float32


def test_floating_island_and_hinges_carry_nothing():
    """Singular-but-consistent systems: a floating cube and a voxel hanging on one corner node
    add no stiffness (rigid-body / ball-joint null modes); an edge-hinged voxel converges too."""
    base = np.zeros((10, 10, 10), bool)
    base[:, :, 0:3] = True  # percolating plate normal to z (top node layer z = 3)
    r0 = effective_elasticity(base, E_S, NU)
    island = base.copy()
    island[4:6, 4:6, 5:7] = True  # nodes z = 5..7: no contact with the plate
    r1 = effective_elasticity(island, E_S, NU)
    assert r1.n_components == 2
    assert _rel(r1.C_eff, r0.C_eff) < 1e-9
    post = base.copy()
    post[5, 5, 3] = True  # face-attached stub (nodes x,y in {5,6}, z in {3,4}): does carry strain
    r_post = effective_elasticity(post, E_S, NU)
    assert _rel(r_post.C_eff, r0.C_eff) > 1e-6
    corner = post.copy()
    corner[6, 6, 4] = True  # shares only node (6,6,4) with the stub: a ball joint
    rc = effective_elasticity(corner, E_S, NU, return_fields=True)
    assert max(rc.residuals) < 1e-7
    assert _rel(rc.C_eff, r_post.C_eff) < 1e-9  # adds exactly nothing
    k = int(np.flatnonzero(rc.element_voxel == np.ravel_multi_index((6, 6, 4), corner.shape))[0])
    assert np.abs(rc.unit_stresses[:, k, :]).max() < 1e-6 * E_S  # and is stress-free
    # one shared edge = a hinge (rotation mode -> singular K); the edge still transmits the
    # stretch of the stub's top edge, so the voxel is NOT stress-free, just mechanism-prone.
    edge = post.copy()
    edge[6, 5, 4] = True
    re = effective_elasticity(edge, E_S, NU)
    assert max(re.residuals) < 1e-7 and re.agreement < 1e-7
    assert _rel(re.C_eff, r_post.C_eff) < 1e-4


def test_random_two_phase_between_voigt_and_reuss():
    """Hill's bounds in the Loewner order: C_Reuss <= C_eff <= C_Voigt (Mandel matrices)."""
    rng = np.random.default_rng(7)
    cell = rng.random((8, 8, 8)) < 0.6
    E2 = 0.2 * E_S
    r = effective_elasticity(np.where(cell, E_S, E2), nu=NU)
    f = cell.mean()
    Cv = f * isotropic_stiffness(E_S, NU) + (1 - f) * isotropic_stiffness(E2, NU)
    Sr = f * np.linalg.inv(isotropic_stiffness(E_S, NU)) + (1 - f) * np.linalg.inv(isotropic_stiffness(E2, NU))
    Cr = np.linalg.inv(Sr)
    M = voigt_to_mandel(r.C_eff)
    assert np.linalg.eigvalsh(voigt_to_mandel(Cv) - M).min() > -1e-9 * E_S
    assert np.linalg.eigvalsh(M - voigt_to_mandel(Cr)).min() > -1e-9 * E_S
    assert r.is_spd and r.agreement < 1e-7


# -----------------------------------------------------------------------------
# Solver properties
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gyroid32():
    return effective_elasticity_tpms(TPMSParams(w=0.0, rho=0.3), 32, return_fields=True)


def test_flux_energy_agreement_and_spd(gyroid32):
    r = gyroid32
    assert r.agreement < 1e-6
    np.testing.assert_allclose(r.C_eff, r.C_eff.T, atol=1e-12 * E_S)
    assert r.is_spd
    assert max(r.residuals) < 1e-7
    assert r.n_dofs < 3 * np.prod(r.shape)  # void nodes removed


def test_agreement_check_fires():
    cell = voxelize(TPMSParams(w=0.0, rho=0.3), 16)
    with pytest.raises(RuntimeError):
        effective_elasticity(cell, E_S, NU, tol=1e-2, agreement_tol=1e-9)


def test_block_solver_matches_scipy_cg_and_matrix_free():
    cell = voxelize(TPMSParams(w=1.0, rho=0.35), 20)
    r_block = effective_elasticity(cell, E_S, NU)
    r_cg = effective_elasticity(cell, E_S, NU, block_solve=False)
    r_mf = effective_elasticity(cell, E_S, NU, operator="matrix_free")
    r_bj = effective_elasticity(cell, E_S, NU, preconditioner="block_jacobi")
    for other in (r_cg, r_mf, r_bj):
        assert _rel(other.C_eff, r_block.C_eff) < 1e-7
    # lock-step = same iterates up to the round-off of the stopping test
    assert max(abs(a - b) for a, b in zip(r_block.iterations, r_cg.iterations)) <= 2
    assert r_mf.operator == "matrix_free" and r_bj.preconditioner == "block_jacobi"


@pytest.mark.skipif(not pyamg_available(), reason="pyamg not installed")
def test_amg_matches_jacobi():
    cell = voxelize(TPMSParams(w=0.0, rho=0.35), 24)
    r_j = effective_elasticity(cell, E_S, NU, preconditioner="jacobi")
    r_a = effective_elasticity(cell, E_S, NU, preconditioner="amg")
    assert _rel(r_a.C_eff, r_j.C_eff) < 1e-6
    assert max(r_a.iterations) < max(r_j.iterations)


def test_soft_void_converges_to_removed_void():
    cell = voxelize(TPMSParams(w=0.0, rho=0.3), 16)
    r0 = effective_elasticity(cell, E_S, NU)
    errs = []
    for emin in (1e-3, 1e-4, 1e-5):
        r = effective_elasticity(cell, E_S, NU, void_modulus=emin)
        errs.append(_rel(r.C_eff, r0.C_eff))
        assert r.n_dofs == 3 * cell.size  # all nodes kept
    errs = np.array(errs)
    np.testing.assert_allclose(errs[:-1] / errs[1:], 10.0, rtol=0.05)  # linear in E_min
    assert errs[-1] < 5e-4


def test_periodic_shift_invariance():
    cell = voxelize(TPMSParams(w=0.0, rho=0.35), 16)
    r0 = effective_elasticity(cell, E_S, NU)
    r1 = effective_elasticity(np.roll(cell, (3, -5, 7), axis=(0, 1, 2)), E_S, NU)
    assert _rel(r1.C_eff, r0.C_eff) < 1e-7
    for k in r0.localization:
        assert r1.localization[k].max == pytest.approx(r0.localization[k].max, rel=1e-5)


def test_axis_permutation_permutes_tensor():
    rng = np.random.default_rng(11)
    cell = rng.random((6, 8, 10)) < 0.7
    Ef = np.where(cell, E_S, 0.1 * E_S)
    r = effective_elasticity(Ef, nu=NU)
    rp = effective_elasticity(np.transpose(Ef, (1, 2, 0)), nu=NU)  # new axes = (y, z, x)
    # new x = old y, new y = old z, new z = old x  -> rotation with x_new = R x_old
    R = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    assert _rel(rp.C_eff, rotate_stiffness(r.C_eff, R)) < 1e-7


def test_bad_inputs():
    with pytest.raises(ValueError):
        effective_elasticity(np.ones((4, 4), bool), E_S, NU)
    with pytest.raises(ValueError):
        effective_elasticity(np.ones((4, 4, 4), bool), None, NU)
    with pytest.raises(ValueError):
        effective_elasticity(np.zeros((4, 4, 4), bool), E_S, NU)
    with pytest.raises(ValueError):
        effective_elasticity(np.ones((4, 4, 4), bool), E_S, NU, localization=("bogus",))
    r = effective_elasticity(np.ones((3, 3, 3), bool), E_S, NU)
    with pytest.raises(ValueError):
        r.localize([0, 0, 1, 0, 0, 0])  # needs return_fields


# -----------------------------------------------------------------------------
# TPMS physics checks
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("w", [0.0, 1.0])
@pytest.mark.parametrize("rho", [0.2, 0.35, 0.5])
def test_cubic_symmetry_gyroid_diamond(w, rho):
    """Gyroid and diamond have cubic point symmetry: C11=C22=C33, C12=C13=C23, C44=C55=C66,
    all other entries 0. Voxel round-off at the threshold only breaks it at ~1e-3."""
    r = effective_elasticity_tpms(TPMSParams(w=w, rho=rho), 32)
    C = r.C_eff
    c11, c12, c44 = r.cubic_constants
    np.testing.assert_allclose(np.diag(C)[:3], c11, rtol=5e-3)
    np.testing.assert_allclose([C[0, 1], C[0, 2], C[1, 2]], c12, rtol=1e-2)
    np.testing.assert_allclose(np.diag(C)[3:], c44, rtol=5e-3)
    assert r.cubic_deviation < 5e-3
    assert r.is_spd
    # rotation by 90 degrees about z leaves a cubic tensor unchanged
    R90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert _rel(rotate_stiffness(C, R90), C) < 5e-3


def test_blend_is_trigonal_not_cubic():
    """w = 0.5 keeps only the 3-fold axis along [111] (cyclic x->y->z) shared by G and D:
    invariant under the cyclic permutation but far from cubic about x, y, z."""
    r = effective_elasticity_tpms(TPMSParams(w=0.5, rho=0.35), 24)
    assert _rel(rotate_stiffness(r.C_eff, CYCLIC), r.C_eff) < 5e-3
    assert r.cubic_deviation > 0.1
    assert r.is_spd


def test_hashin_shtrikman_upper_bound(gyroid32):
    """Isotropic (Voigt-average) moduli of a porous cell lie below the HS upper bounds."""
    hs = hashin_shtrikman_porous(gyroid32.solid_fraction, E_S, NU)
    c11, c12, c44 = gyroid32.cubic_constants
    K_v = (c11 + 2 * c12) / 3
    G_v = (c11 - c12 + 3 * c44) / 5
    assert K_v < hs["K"] and G_v < hs["G"]
    assert gyroid32.E_axial < hs["E"]
    # limits of the bound
    np.testing.assert_allclose(hashin_shtrikman_porous(1.0, E_S, NU)["E"], E_S, rtol=1e-12)
    np.testing.assert_allclose(hashin_shtrikman_porous(0.0, E_S, NU)["E"], 0.0, atol=1e-6)


def test_engineering_constants_consistency(gyroid32):
    r = gyroid32
    E3 = r.youngs_moduli
    np.testing.assert_allclose([r.youngs_modulus(e) for e in np.eye(3)], E3, rtol=1e-12)
    emin, emax = r.youngs_extremes()
    assert emin <= E3.min() * (1 + 1e-12) and emax >= E3.max() * (1 - 1e-12)
    c11, c12, c44 = r.cubic_constants
    # cubic: E_100 = (C11 - C12)(C11 + 2 C12)/(C11 + C12), bulk = (C11 + 2 C12)/3
    np.testing.assert_allclose(r.E_axial, (c11 - c12) * (c11 + 2 * c12) / (c11 + c12), rtol=1e-2)
    np.testing.assert_allclose(r.bulk_modulus, (c11 + 2 * c12) / 3, rtol=1e-2)
    np.testing.assert_allclose(r.zener_ratio, 2 * c44 / (c11 - c12))
    row = r.as_row()
    assert row["E_axial"] == pytest.approx(r.E_axial / E_S)
    assert {"loc_uniaxial_z_p99", "loc_shear_xz_max", "zener", "C66"} <= set(row)


def test_localization_sanity(gyroid32):
    r = gyroid32
    for name, loc in r.localization.items():
        assert loc.max >= loc.p999 >= loc.p99 >= loc.mean > 1.0  # porous: amplification
        assert np.isclose(loc.macro_von_mises, von_mises(MACRO_STRESS_CASES[name]))
    # localize() reproduces the built-in case, and is linear in the macroscopic stress
    l1 = r.localize(MACRO_STRESS_CASES["uniaxial_z"], keep_field=True)
    assert l1.max == pytest.approx(r.localization["uniaxial_z"].max, rel=1e-6)
    l2 = r.localize(3.0 * MACRO_STRESS_CASES["uniaxial_z"])
    assert l2.max == pytest.approx(l1.max, rel=1e-6)
    assert l1.factor is not None and l1.factor.size == int(voxelize(TPMSParams(w=0.0, rho=0.3), 32).sum())
    # average stress over the cell equals the macroscopic stress (Hill): mean_e sigma_e / rho = Sigma
    sig = np.einsum("j,jek->ek", r.S @ MACRO_STRESS_CASES["uniaxial_z"], r.unit_stresses.astype(float))
    np.testing.assert_allclose(sig.sum(axis=0) / np.prod(r.shape), MACRO_STRESS_CASES["uniaxial_z"], atol=1e-5)


def test_anisotropic_cell_stretch():
    """a_z = 1.5 stretches the cell along z: stiffer along z than across (aligned ligaments)."""
    r = effective_elasticity_tpms(TPMSParams.from_design(0.0, 0.35, a_z=1.5), 16)
    Ex, Ey, Ez = r.youngs_moduli
    assert r.shape[2] == 24
    assert Ez > 1.2 * Ex and Ex == pytest.approx(Ey, rel=2e-2)


def test_extrapolated_elasticity():
    ex = extrapolated_elasticity_tpms(TPMSParams(w=0.0, rho=0.35), (12, 24))
    np.testing.assert_allclose(ex.C_eff, (24 * ex.fine.C_eff - 12 * ex.coarse.C_eff) / 12, rtol=1e-12, atol=1e-6)
    assert ex.coarse.localization == {} and "uniaxial_z" in ex.fine.localization
    assert np.all(ex.eigenvalues > 0) and ex.correction < 0.2
    with pytest.raises(ValueError):
        extrapolated_elasticity_tpms(TPMSParams(), (24, 24))


def test_averaged_elasticity():
    offs = grid_offsets(3)
    assert np.all(offs[0] == 0) and offs.shape == (3, 3) and np.allclose(grid_offsets(3), offs)
    a = averaged_elasticity_tpms(TPMSParams(w=1.0, rho=0.35), 16, 3)
    Cs = np.stack([r.C_eff for r in a.results])
    np.testing.assert_allclose(a.C_eff, Cs.mean(axis=0), rtol=1e-12, atol=1e-6)
    assert a.C_std.max() > 0  # shifts do change the voxel geometry
    assert a.localization["uniaxial_z"]["p99"] == pytest.approx(
        np.mean([r.localization["uniaxial_z"].p99 for r in a.results]))
    assert np.all(a.eigenvalues > 0) and 1.0 < a.zener_ratio < 3.0


# -----------------------------------------------------------------------------
# Slow: timing target, convergence, density scaling
# -----------------------------------------------------------------------------
@pytest.mark.slow
def test_timing_n40_under_60s_per_load_case():
    r = effective_elasticity_tpms(TPMSParams(w=0.0, rho=0.35), 40)
    per_case = r.wall_time / 6.0
    print(f"n=40 gyroid rho=0.35: {r.wall_time:.1f} s total, {per_case:.1f} s per load case, "
          f"{r.n_dofs} dofs, iterations {r.iterations}")
    assert per_case < 60.0


@pytest.mark.slow
@pytest.mark.parametrize("w", [0.0, 1.0])
def test_convergence_24_to_48(w):
    """Offset-averaged E* changes by < 3 % from n = 24 to 48 and n = 32 vs 48 by < 2 %."""
    rng = np.random.default_rng(5)
    offs = [None, *rng.random((2, 3))]
    E = {}
    for n in (24, 32, 48):
        E[n] = np.mean([effective_elasticity_tpms(TPMSParams(w=w, rho=0.35), n, offset=o,
                                                  localization=()).E_axial for o in offs])
    assert abs(E[24] / E[48] - 1) < 0.03
    assert abs(E[32] / E[48] - 1) < 0.02


@pytest.mark.slow
@pytest.mark.parametrize("w", [0.0, 1.0])
def test_power_law_exponent_bending_dominated(w):
    """E*/E_s = C rho^m over rho = 0.2-0.5: network gyroid/diamond are bending-dominated,
    published exponent ~2 (Khaderi et al. 2014; Maskery et al. 2018). Thick ligaments at
    rho -> 0.5 lower the fitted exponent, so accept 1.5 <= m <= 2.6."""
    rho = np.array([0.2, 0.3, 0.4, 0.5])
    E = [effective_elasticity_tpms(TPMSParams(w=w, rho=p), 32, localization=()).E_axial / E_S for p in rho]
    m, logC = np.polyfit(np.log(rho), np.log(E), 1)
    print(f"w={w}: E*/E_s = {np.exp(logC):.3f} rho^{m:.3f}")
    assert 1.5 <= m <= 2.6
