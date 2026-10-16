"""Task 7: finite-gap strips, wall extraction, discrepancy models, two-level preconditioners."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voxlat.homogenization.conduction import effective_conductivity
from voxlat.homogenization.elasticity import (
    effective_elasticity,
    isotropic_stiffness,
    laminate_stiffness,
)
from voxlat.homogenization.finite_gap import (
    PRIMARY_QUANTITIES,
    QUANTITY_TYPE,
    BulkInterpolator,
    FiniteGapCorrection,
    GapProperties,
    GapSpec,
    GPDiscrepancyModel,
    PhysicalDiscrepancyModel,
    brinkman_channel_factor,
    build_strip,
    bulk_cell,
    bulk_properties,
    compute_properties,
    default_phases,
    discrepancy,
    distinct_cuts,
    reduced_phase,
    from_mixed_form,
    grouped_cv,
    homogenized_error,
    homogenized_properties,
    laminate_average,
    laminate_core,
    laminate_mixed_form,
    strip_level_set,
    strip_properties,
    threshold_curve,
)
from voxlat.homogenization.finite_gap_study import (
    StudySettings,
    bulk_requirements,
    props_from_json,
    props_to_json,
    study_design,
)
from voxlat.homogenization.stokes import mac_slit_permeability, permeability
from voxlat.geometry.tpms import TPMSParams, voxelize

KS, KF, E, NU = 130.0, 0.4, 70e9, 0.33


def _spd(rng: np.random.Generator, n: int = 6) -> np.ndarray:
    A = rng.normal(size=(n, n))
    return A @ A.T + n * np.eye(n)


# =============================================================================
# Laminate algebra
# =============================================================================
@pytest.mark.parametrize("fr", [(0.3, 0.7), (0.2, 0.5, 0.3)])
def test_laminate_average_equals_backus(fr):
    Es, nus = [10.0, 1.0, 4.0][: len(fr)], [0.3, 0.2, 0.45][: len(fr)]
    Cs = [isotropic_stiffness(e, v) for e, v in zip(Es, nus)]
    np.testing.assert_allclose(laminate_average(Cs, fr), laminate_stiffness(fr, Es, nus, axis=0),
                               rtol=1e-12, atol=1e-12)


def test_mixed_form_round_trip_and_core_extraction():
    rng = np.random.default_rng(7)
    C = _spd(rng)
    np.testing.assert_allclose(from_mixed_form(laminate_mixed_form(C)), C, rtol=1e-12, atol=1e-10)
    Cw = isotropic_stiffness(50.0, 0.33)
    for fw in (0.0, 0.1, 0.6):
        stack = laminate_average([C, Cw], [1 - fw, fw]) if fw > 0 else C
        np.testing.assert_allclose(laminate_core(stack, Cw, fw), C, rtol=1e-10, atol=1e-9)
    with pytest.raises(ValueError):
        laminate_core(C, Cw, 1.0)


def test_laminate_core_with_the_fe_solver():
    """Voxel FEA of an x-laminate (core E1 | wall E2) -> extraction recovers the core (conventions)."""
    cell = np.ones((10, 3, 3))
    cell[:7] = 3.0e9
    cell[7:] = 70e9
    r = effective_elasticity(cell, None, 0.33, localization=None, preconditioner="jacobi")
    core = laminate_core(r.C_eff, isotropic_stiffness(70e9, 0.33), 0.3)
    np.testing.assert_allclose(core, isotropic_stiffness(3.0e9, 0.33), rtol=1e-6, atol=1e-6 * 3e9)


# =============================================================================
# Specification and geometry
# =============================================================================
def test_spec_derived_quantities():
    s = GapSpec(0.5, 0.35, 1.5, 1 / 3, n=32)
    assert s.n_lattice == 48 and s.n_wall == 16 and s.nz == 32 and s.H == 1.5
    assert s.morphology == "B0.50"
    assert GapSpec(0.0, 0.3, 2, n=16).case_id != GapSpec(0.0, 0.3, 2, 0.5, n=16).case_id
    g = GapSpec(0.0, 0.35, 4, gradient=0.05, n=16)
    assert g.graded and np.isclose(g.layer_rho.mean(), 0.35)
    assert np.isclose(g.rho_range[1] - g.rho_range[0], 0.05 * (4 - 1 / 16))
    with pytest.raises(ValueError):
        GapSpec(0.0, 0.35, 4, gradient=0.2, n=16)  # leaves (0, 1)
    with pytest.raises(ValueError):
        GapSpec(1.5, 0.35, 1)
    assert default_phases(3) == (0.0, 1 / 3, 2 / 3)


@pytest.mark.parametrize("w,N,phase", [(0.0, 2.0, 0.0), (1.0, 1.5, 1 / 3), (0.5, 1.0, 0.25)])
def test_uniform_strip_is_a_tiling_of_the_bulk_cell(w, N, phase):
    n = 16
    spec = GapSpec(w, 0.35, N, phase, n=n)
    strip = build_strip(spec)
    cell = bulk_cell(w, 0.35, n, phase)
    reps = int(np.ceil(spec.n_lattice / n))
    np.testing.assert_array_equal(strip.lattice, np.tile(cell, (reps, 1, 1))[: spec.n_lattice])
    assert strip.solid[spec.n_lattice:].all()
    assert strip.solid.shape == (spec.n_lattice + spec.n_wall, n, n)
    # the bulk cell is the ordinary Task 1 voxelization, shifted by the phase
    np.testing.assert_array_equal(cell, voxelize(TPMSParams(w, 0.35), n, offset=(phase, 0, 0)))


def test_stretched_strip_grid():
    spec = GapSpec(0.0, 0.35, 1.0, a_z=1.5, n=16)
    assert build_strip(spec).solid.shape == (16 + 8, 16, 24)
    assert strip_level_set(spec).shape == (16, 16, 24)


def test_graded_strip_follows_the_target_density():
    spec = GapSpec(0.0, 0.35, 4.0, 0.0, gradient=0.075, n=24)
    strip = build_strip(spec)
    # per-cell density (one full period) follows the linear target
    dens = strip.layer_density.reshape(4, 24).mean(axis=1)
    target = spec.layer_rho.reshape(4, 24).mean(axis=1)
    np.testing.assert_allclose(dens, target, atol=0.01)
    assert np.all(np.diff(strip.thresholds) > 0)
    grid, c = threshold_curve(0.0, 24)
    assert np.all(np.diff(c) >= 0) and grid[0] == pytest.approx(0.02)


# =============================================================================
# Property extraction (exact cases)
# =============================================================================
def test_wall_extraction_conduction_series_exact():
    """Lattice = x-layers of solid/fluid: series formula is exact."""
    n, nw = 8, 4
    lat = np.zeros((8, 4, 4), bool)
    lat[:3] = True  # 3 solid + 5 fluid layers
    solid = np.concatenate([lat, np.ones((nw, 4, 4), bool)])
    p = compute_properties(solid, n, 8, nw, k_s=KS, k_f=KF, E=E, nu=NU, quantities=("k",),
                           preconditioner="jacobi")
    assert p.k_n == pytest.approx(1.0 / (3 / 8 / KS + 5 / 8 / KF), rel=1e-9)


def test_wall_extraction_slit_permeability_exact():
    """Empty gap = plane channel: K_t = MAC slit value (H^2 + 2)/12 voxel^2 exactly."""
    n, nl, nw = 8, 6, 2
    solid = np.concatenate([np.zeros((nl, 3, 3), bool), np.ones((nw, 3, 3), bool)])
    p = compute_properties(solid, n, nl, nw, k_s=KS, k_f=KF, E=E, nu=NU, quantities=("K",),
                           preconditioner="jacobi")
    ref = mac_slit_permeability(nl) / n**2  # units L^2, voxel = L/n
    np.testing.assert_allclose(np.diag(p.K_t), [ref, ref], rtol=1e-8)
    assert abs(p.K_t[0, 1]) < 1e-12


def test_solid_gap_returns_the_solid():
    n = 8
    solid = np.ones((12, 3, 3), bool)
    p = compute_properties(solid, n, 8, 4, k_s=KS, k_f=KF, E=E, nu=NU, quantities=("k", "C"),
                           preconditioner="jacobi")
    assert p.k_n == pytest.approx(KS, rel=1e-10)
    np.testing.assert_allclose(p.C, isotropic_stiffness(E, NU), rtol=1e-7, atol=1e-7 * E)


def test_no_wall_strip_equals_bulk():
    """wall = 0 and integer N: the strip is N periods of the bulk cell -> identical properties."""
    spec = GapSpec(0.0, 0.35, 2.0, 1 / 3, wall=0.0, n=12)
    _, sp = strip_properties(spec, k_s=KS, k_f=KF, E=E, nu=NU, preconditioner="jacobi")
    bp = bulk_properties(0.0, 0.35, 12, 1 / 3, k_s=KS, k_f=KF, E=E, nu=NU, preconditioner="jacobi")
    d = discrepancy(sp, bp)
    for q in PRIMARY_QUANTITIES:
        assert abs(d[q]) < 1e-6, (q, d[q])


def test_gyroid_screw_symmetry_of_the_cut_phase():
    """Shifting the cut by L/4 rotates a gyroid strip by 90 deg about x: K_s <-> K_z, G_rs <-> G_rz."""
    kw = dict(k_s=KS, k_f=KF, E=E, nu=NU, preconditioner="two_level")
    _, a = strip_properties(GapSpec(0.0, 0.35, 1.0, 0.1, n=16), **kw)
    _, b = strip_properties(GapSpec(0.0, 0.35, 1.0, 0.35, n=16), **kw)
    sa, sb = a.scalars(), b.scalars()
    for x, y in (("K_s", "K_z"), ("K_z", "K_s"), ("G_rs", "G_rz"), ("k_n", "k_n"), ("C_nn", "C_nn")):
        assert sa[x] == pytest.approx(sb[y], rel=2e-3), (x, y)


def test_finite_gap_physics_gyroid():
    """RQ1 sanity: no-slip walls cut K_t strongly, ~1/N; the bonded walls stiffen the core."""
    kw = dict(k_s=KS, k_f=KF, E=E, nu=NU, preconditioner="two_level")
    bp = bulk_properties(0.0, 0.35, 16, 0.0, **kw)
    d = {}
    for N in (1.0, 2.0):
        _, sp = strip_properties(GapSpec(0.0, 0.35, N, 0.0, n=16), **kw)
        d[N] = discrepancy(sp, bp)
    assert d[1.0]["K_t"] < -0.15
    assert d[2.0]["K_t"] / d[1.0]["K_t"] == pytest.approx(0.5, abs=0.06)
    assert d[1.0]["C_nn"] > 0.0 and d[1.0]["G_t"] > 0.0
    assert abs(d[1.0]["k_n"]) < 0.05


# =============================================================================
# Homogenized prediction, interpolation, small helpers
# =============================================================================
def _fake_props(k: float, K: float, Escale: float) -> GapProperties:
    C = isotropic_stiffness(Escale, 0.3)
    Kt = np.diag([K, 1.2 * K])
    return GapProperties(k, Kt, C, k, Kt, C, 1.0, 1.0, 0.0, 0.3, 0.7, KS, KF, E, NU)


def test_bulk_interpolator_nodes_spd_and_graded_prediction():
    rhos = [0.2, 0.3, 0.4, 0.5]
    props = [_fake_props(10 * r**1.3, 0.02 * (1 - r) ** 3, E * r**2) for r in rhos]
    bi = BulkInterpolator(rhos, props)
    k, K, C = bi(rhos)
    np.testing.assert_allclose(k, [p.k_n for p in props], rtol=1e-12)
    np.testing.assert_allclose(K, [p.K_t for p in props], rtol=1e-10)
    np.testing.assert_allclose(C, [p.C for p in props], rtol=1e-9, atol=1e-6)
    assert np.all(np.linalg.eigvalsh(bi([0.33])[2][0]) > 0)
    with pytest.raises(ValueError):
        bi([0.6])
    spec = GapSpec(0.0, 0.35, 4.0, gradient=0.05, n=16)
    hom = homogenized_properties(spec, bi)
    kk, KK, CC = bi(spec.layer_rho)
    assert hom.k_n == pytest.approx(1 / np.mean(1 / kk))  # series
    np.testing.assert_allclose(hom.K_t, KK.mean(axis=0))  # parallel
    # constant closures -> the graded prediction is that constant
    flat = BulkInterpolator([0.1, 0.6], [props[1], props[1]])
    h2 = homogenized_properties(spec, flat)
    assert h2.k_n == pytest.approx(props[1].k_n)
    np.testing.assert_allclose(h2.C, props[1].C, rtol=1e-9)
    assert homogenized_properties(GapSpec(0.0, 0.35, 2, n=16), props[1]) is props[1]


def test_reduced_phase_and_distinct_cuts():
    # G/D: phi ~ phi + 1/4 ~ -phi -> [0, 1/8]; blends: mod 1
    r = reduced_phase([0.0, 1.0, 0.0, 0.0, 0.5], [1 / 3, 2 / 3, 0.125, 0.3, 2 / 3])
    np.testing.assert_allclose(r, [1 / 12, 1 / 12, 0.125, 0.05, 2 / 3])
    df = pd.DataFrame(dict(w=[0.0, 0.0, 0.5, 0.5, 0.0], rho=0.35, N=1.0, phase=[1 / 3, 2 / 3, 1 / 3, 2 / 3, 1 / 3],
                           gradient=[0, 0, 0, 0, 0.05], a_z=1.0, wall=0.5, n=32))
    out = distinct_cuts(df)
    assert len(out) == 4  # the two G cuts 1/3, 2/3 are one strip; graded rows are kept
    assert "phase_r" in out


def test_error_helpers():
    assert homogenized_error(-0.2) == pytest.approx(0.25)
    f = brinkman_channel_factor([1, 10, 1000], 4.9e-3)
    assert np.all(np.diff(f) > 0) and f[-1] == pytest.approx(1 - 2 * np.sqrt(4.9e-3) / 1000, rel=1e-6)


def test_props_json_round_trip():
    p = _fake_props(20.0, 0.004, 5e9)
    q = props_from_json(props_to_json(p))
    np.testing.assert_allclose(q.C, p.C)
    assert q.scalars() == p.scalars()


def test_study_design_counts_and_bulk_sharing():
    d = study_design(n=32)
    assert len(d["uniform"]) == 3 * 3 * 7 * 3 + 2 * 3 * 7  # + the L/8 cut for gyroid and diamond
    assert len(d["graded"]) == 3 * 5 * 3
    assert all(s.rho_range[0] >= 0.2 - 1e-9 and s.rho_range[1] <= 0.5 + 1e-9 for s in d["graded"])
    keys = {k.id for k in bulk_requirements(d["uniform"])}
    graded_keys = {k.id for k in bulk_requirements(d["graded"])}
    assert len(keys) == 3 * 3 * 3 + 2 * 3
    assert keys & graded_keys  # rho = 0.25/0.35/0.45 cells are shared
    assert StudySettings().fingerprint() == StudySettings(preconditioner="jacobi").fingerprint()


# =============================================================================
# Discrepancy models
# =============================================================================
def _synthetic_table(rng: np.random.Generator, noise: float = 0.0) -> pd.DataFrame:
    rows = []
    for w in (0.0, 0.5, 1.0):
        for rho in (0.25, 0.35, 0.45):
            for N in (1, 1.5, 2, 3, 4, 6, 8):
                for ph in (0, 1, 2):
                    rows.append(dict(w=w, rho=rho, N=N, phase=ph / 3, gradient=0.0, a_z=1.0, wall=0.5, n=32))
            for g in (0.025, 0.05, 0.075):
                rows.append(dict(w=w, rho=rho, N=4.0, phase=0.0, gradient=g, a_z=1.0, wall=0.5, n=32))
    df = pd.DataFrame(rows)
    a_true = {"k_n": 0.02 + 0.01 * (df.rho - 0.35) / 0.1, "K_t": 0.26 - 0.03 * df.w,
              "C_nn": -0.03 + 0.02 * df.w, "G_t": -0.04 + 0.1 * 4 * df.w * (1 - df.w), "G_sz": -0.05}
    for q, a in a_true.items():
        a = np.asarray(a, float) + noise * rng.normal(size=len(df))
        if QUANTITY_TYPE[q] == "series":
            d = 1 / (1 + a / df.N) - 1
        else:
            d = -a / df.N
        df[f"delta_{q}"] = (1 + d) * (1 + 2.0 * df.gradient**2) - 1
    return df


def test_physical_model_recovers_synthetic_coefficients(tmp_path):
    df = _synthetic_table(np.random.default_rng(1))
    pm = PhysicalDiscrepancyModel().fit(df)
    for q in PRIMARY_QUANTITIES:
        pred = pm.predict_df(df)[q]
        np.testing.assert_allclose(pred, df[f"delta_{q}"], atol=1e-10)
        assert pm.models[q].c_g == pytest.approx(2.0, rel=1e-6)
    assert pm.wall_coefficient("K_t", 1.0, 0.35) == pytest.approx(0.23)
    corr = FiniteGapCorrection(pm, {"note": "synthetic"})
    path = corr.save(tmp_path / "corr.json")
    c2 = FiniteGapCorrection.load(path)
    assert c2.factor("K_t", 2.0, 0.0, 0.35) == pytest.approx(1 - 0.13)
    assert c2.meta["note"] == "synthetic"


def test_grouped_cv_and_gp_on_synthetic_data():
    df = _synthetic_table(np.random.default_rng(2), noise=0.01)
    cv = grouped_cv(df, ("K_t", "k_n"), groups="theta", use_gp=True, restarts=0)
    for q in ("K_t", "k_n"):
        assert cv[q]["physical"] < 0.5 * cv[q]["none"] or cv[q]["none"] < 0.03
        assert cv[q]["physical+gp"] < cv[q]["none"]
        assert cv[q]["n"] == len(df)
    pm = PhysicalDiscrepancyModel(("K_t",)).fit(df)
    gm = GPDiscrepancyModel(pm, restarts=0).fit(df)
    m, s = gm.predict(df.iloc[:5], return_std=True)["K_t"]
    assert m.shape == (5,) and np.all(s >= 0)


# =============================================================================
# Two-level preconditioners and the new solver options (Task 7 additions)
# =============================================================================
@pytest.fixture(scope="module")
def strip16():
    return build_strip(GapSpec(0.0, 0.35, 2.0, 0.0, n=16)).solid


def test_two_level_matches_jacobi_conduction(strip16):
    a = effective_conductivity(strip16, KS, KF, preconditioner="jacobi")
    b = effective_conductivity(strip16, KS, KF, preconditioner="two_level")
    np.testing.assert_allclose(b.k_eff, a.k_eff, rtol=1e-6, atol=1e-6 * a.k_eff.max())
    assert max(b.iterations) < max(a.iterations)
    c = effective_conductivity(strip16, KS, KF, preconditioner="two_level", directions=(0,))
    assert c.k_eff[0, 0] == pytest.approx(a.k_eff[0, 0], rel=1e-6)
    assert np.isnan(c.k_eff[1, 1]) and len(c.iterations) == 1


def test_two_level_matches_jacobi_stokes(strip16):
    a = permeability(strip16, voxel_size=1 / 16, directions=(1,), preconditioner="jacobi")
    b = permeability(strip16, voxel_size=1 / 16, directions=(1,), preconditioner="two_level")
    assert b.K[1, 1] == pytest.approx(a.K[1, 1], rel=1e-6)
    assert b.iterations[0] < a.iterations[0]


def test_two_level_matches_jacobi_elasticity_and_load_cases(strip16):
    a = effective_elasticity(strip16, E, NU, localization=None, preconditioner="jacobi")
    b = effective_elasticity(strip16, E, NU, localization=None, preconditioner="two_level")
    np.testing.assert_allclose(b.C_eff, a.C_eff, rtol=0, atol=1e-6 * np.abs(a.C_eff).max())
    assert max(b.iterations) < max(a.iterations)
    c = effective_elasticity(strip16, E, NU, localization=None, preconditioner="two_level",
                             load_cases=("xx", "xz", "xy"))
    sel = np.ix_([0, 4, 5], [0, 4, 5])
    np.testing.assert_allclose(c.C_eff[sel], a.C_eff[sel], rtol=0, atol=1e-6 * np.abs(a.C_eff).max())
    np.testing.assert_allclose(c.C_eff_flux[:, [0, 4, 5]], a.C_eff_flux[:, [0, 4, 5]],
                               rtol=0, atol=1e-6 * np.abs(a.C_eff).max())
    assert np.isnan(c.C_eff[1, 1])
    with pytest.raises(ValueError):
        effective_elasticity(strip16, E, NU, load_cases=(0,))  # default localization needs all six


def test_two_level_preconditioner_is_symmetric_positive():
    import scipy.sparse as sp

    from voxlat.homogenization.coarse import (
        TwoLevelPreconditioner,
        box_aggregates,
        rigid_body_coarse_basis,
        scalar_coarse_basis,
    )

    rng = np.random.default_rng(3)
    shape = (10, 6, 5)
    gi = np.stack(np.unravel_index(np.arange(np.prod(shape)), shape), axis=1)
    lab = box_aggregates(gi, shape, 4)
    assert lab.max() + 1 == 3 * 2 * 2
    # 1-D periodic Laplacian-like SPSD matrix
    n = gi.shape[0]
    L = sp.diags([2 * np.ones(n), -np.ones(n - 1), -np.ones(n - 1)], [0, 1, -1]).tolil()
    L[0, n - 1] = L[n - 1, 0] = -1
    M = TwoLevelPreconditioner(L.tocsr(), scalar_coarse_basis(lab))
    x, y = rng.normal(size=n), rng.normal(size=n)
    assert x @ (M @ y) == pytest.approx(y @ (M @ x), rel=1e-8)
    assert x @ (M @ x) > 0
    P = rigid_body_coarse_basis(gi[:40], box_aggregates(gi[:40], shape, 4), 4)
    assert P.shape == (120, 6 * (box_aggregates(gi[:40], shape, 4).max() + 1))
