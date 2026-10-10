"""Task 9: homogenized jacket device model (voxlat.device.jacket2d).

Verification strategy
* exact discrete / analytic references: uniform Darcy and Darcy-Forchheimer flow (two parallel
  paths), series layers (harmonic resistances), linear-pressure consistency of the flux stencil
  incl. the anisotropic cross term, epsilon-NTU at fixed wall temperature (exact discrete
  upwind solution + first-order convergence to the continuum), imposed-flux bulk solution,
  fin conductance vs an independent finite-difference fin;
* conservation: mass and energy balances on graded, anisotropic designs (requirement < 0.5 %);
* grid convergence of R and runtime of one evaluation;
* structure (equilibrium of the sandwich shear stresses), mass, manufacturability flags.
Tests that need the trained Task 6 / Task 7 models skip if models/ is empty.
"""

from __future__ import annotations

import math
import time
import warnings

import numpy as np
import pytest

from voxlat.device.jacket2d import (
    INLET,
    LATTICE,
    OUTLET,
    JacketModel,
    JacketOptions,
    LocalClosures,
    UniformClosures,
    build_grid,
    cell_average_heat_flux,
    directional_permeability,
    evaluate,
    fin_conductance,
    in_plane_permeability,
    manifold_htc_default,
    ntu_reference,
    uniform_flow_reference,
)
from voxlat.utils.config import load_config
from voxlat.utils.paths import models_dir

HAVE_MODELS = (models_dir() / "closure_ensemble.npz").is_file() and (models_dir() / "finite_gap_correction.json").is_file()
needs_models = pytest.mark.skipif(not HAVE_MODELS, reason="trained closure models not in models/")

CFG = load_config()
CFG_FLAT = load_config(overrides={"heat_flux": {"amplitude": 0.0}})
T_IN = CFG.operating.inlet_temperature

K0 = 7.8e-8  # m^2, ~ gyroid rho* = 0.35, L = 4 mm
K_S0 = 27.0  # W/mK
A_SF0 = 743.0  # 1/m


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def uniform(**kw):
    base = dict(K=K0, k_eff=K_S0, a_sf=A_SF0)
    base.update(kw)
    return UniformClosures(**base)


class LayeredClosures:
    """K = K0 ((1 - rho)/0.65)^3 per cell (isotropic); everything else constant."""

    def __call__(self, w, rho, a_z, L, gap):
        n = len(rho)
        K = K0 * ((1.0 - rho) / 0.65) ** 3
        base = UniformClosures(K=1.0, k_eff=K_S0, a_sf=A_SF0)(w, rho, a_z, L, gap)
        base.K = K[:, None, None] * np.eye(3)[None]
        return base


def path_length(model):
    return math.pi * model.grid.r_mean - CFG.manifolds.width


# -----------------------------------------------------------------------------------------
# grid, heat flux, closures helpers
# -----------------------------------------------------------------------------------------
def test_grid_geometry_and_manifold_slots():
    g = build_grid(CFG, 2.5e-3)
    C = 2 * math.pi * CFG.jacket.lattice_mean_radius
    assert g.circumference == pytest.approx(C)
    assert g.s_faces[-1] == pytest.approx(C) and g.s_faces[0] == 0.0
    assert np.all(g.ds <= 2.5e-3 + 1e-12) and np.all(g.dz <= 2.5e-3 + 1e-12)
    for k in (INLET, OUTLET):
        cols = np.any(g.kind == k, axis=1)
        assert g.ds[cols].sum() == pytest.approx(CFG.manifolds.width, rel=1e-12)  # slots resolved exactly
        assert np.all(g.kind[cols] == k)  # full axial length
    # outlet centred at theta = 180 deg
    out_cols = np.any(g.kind == OUTLET, axis=1)
    theta = g.theta_plot_deg()[out_cols]
    assert theta.mean() == pytest.approx(180.0, abs=1e-9)
    # areas: planform x gap = annulus volume
    assert g.area.sum() * g.gap == pytest.approx(math.pi * (CFG.jacket.lattice_outer_radius**2
                                                            - CFG.jacket.lattice_inner_radius**2) * CFG.jacket.axial_length)
    # partial axial manifolds are centred and exact
    cfg = load_config(overrides={"manifolds": {"axial_extent": 0.03}})
    g2 = build_grid(cfg, 2.5e-3)
    rows = np.any(g2.kind == INLET, axis=0)
    assert g2.dz[rows].sum() == pytest.approx(0.03)
    assert g2.z_centres[rows].mean() == pytest.approx(0.025)


def test_cell_averaged_heat_flux_integrates_to_Q():
    for n in (1, 3, 20, 37):
        zf = np.linspace(0, CFG.jacket.axial_length, n + 1)
        q = cell_average_heat_flux(CFG, zf)
        Q = np.sum(q * np.diff(zf)) * 2 * math.pi * CFG.jacket.stator_radius
        assert Q == pytest.approx(CFG.motor.heat_to_jacket, rel=1e-12)
    # end / centre ratio of the exact profile (fine cells)
    zf = np.linspace(0, CFG.jacket.axial_length, 2001)
    q = cell_average_heat_flux(CFG, zf)
    assert q[0] / q[1000] == pytest.approx(2.0, rel=2e-3)


def test_in_plane_permeability_is_constrained_flow():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(3, 3))
    K = A @ A.T + 3 * np.eye(3)
    Kt = in_plane_permeability(K)
    # constrained Darcy: u = -K g with u_r = 0  ->  u_t = -Kt g_t
    g_t = rng.normal(size=2)
    g_r = -(K[0, 1:] @ g_t) / K[0, 0]
    u = -K @ np.concatenate([[g_r], g_t])
    assert abs(u[0]) < 1e-12
    np.testing.assert_allclose(u[1:], -Kt @ g_t, rtol=1e-12)
    # block-diagonal tensor: plain (s, z) block
    Kb = np.diag([1.0, 2.0, 3.0])
    np.testing.assert_allclose(in_plane_permeability(Kb), np.diag([2.0, 3.0]))
    e = np.array([1.0, 0.0])
    assert directional_permeability(np.diag([2.0, 3.0]), e) == pytest.approx(2.0)


def _fd_fin(k, H, h, n=4000):
    """Independent 1-D fin: k T'' = H T on (0, h), T(0) = 1, T'(h) = 0 (2nd-order FD)."""
    x = np.linspace(0, h, n + 1)
    dx = x[1]
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    main = np.full(n + 1, -2 * k / dx**2 - H)
    lo = np.full(n, k / dx**2)
    up = np.full(n, k / dx**2)
    up[0] = 0.0
    main[0] = 1.0
    lo[-1] = 2 * k / dx**2  # mirror ghost at the adiabatic tip
    A = sp.diags([lo, main, up], [-1, 0, 1], format="csc")
    b = np.zeros(n + 1)
    b[0] = 1.0
    T = spla.spsolve(A, b)
    flux = H * np.trapezoid(T, x)  # = heat leaving the base
    return flux, np.trapezoid(T, x) / h


@pytest.mark.parametrize("mh", [0.1, 1.0, 2.2, 6.0])
def test_fin_conductance_matches_finite_difference(mh):
    k, h = 27.0, 6e-3
    H = k * (mh / h) ** 2
    U, eta = fin_conductance(k, H, h)
    U_fd, eta_fd = _fd_fin(k, H, h)
    assert float(U) == pytest.approx(U_fd, rel=1e-5)
    assert float(eta) == pytest.approx(eta_fd, rel=1e-5)
    Ui, etai = fin_conductance(k, H, h, "isothermal")
    assert float(Ui) == pytest.approx(H * h) and float(etai) == 1.0
    assert float(U) < float(Ui)


def test_manifold_htc_default():
    assert manifold_htc_default(CFG) == pytest.approx(5.385 * 0.40 / 0.012)


# -----------------------------------------------------------------------------------------
# flow
# -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("spacing", [5e-3, 2.5e-3, 1.7e-3])
def test_uniform_darcy_flow_is_exact(spacing):
    m = JacketModel(CFG, JacketOptions(spacing=spacing, inertia=False), closures=uniform(f_K=0.8))
    out = m.evaluate(0.35, 0.0, 4e-3, return_fields=True)
    U = CFG.operating.nominal_flow_rate / (2 * m.grid.gap * CFG.jacket.axial_length)
    dp = path_length(m) * CFG.coolant.dynamic_viscosity * U / (0.8 * K0)
    assert out["delta_p"] == pytest.approx(dp, rel=1e-10)
    f = out["fields"]
    lat = m.grid.lattice
    np.testing.assert_allclose(np.abs(f.U_s[lat]), U, rtol=1e-9)
    assert np.max(np.abs(f.U_z[lat])) < 1e-9 * U
    assert out["diagnostics"]["flow_split"] == pytest.approx((0.5, 0.5), abs=1e-12)
    assert out["mass_balance_error"] < 1e-12


def test_uniform_forchheimer_flow_matches_reference():
    m = JacketModel(CFG, JacketOptions(spacing=2.5e-3), closures=uniform(f_K=0.75))
    for Q in CFG.operating.flow_rate_sweep:
        out = m.evaluate(0.35, 0.0, 4e-3, flow_rate=Q)
        ref = uniform_flow_reference(m, 0.35, 0.0, 4e-3, flow_rate=Q)
        assert out["picard_converged"]
        assert out["delta_p"] == pytest.approx(ref["dp"], rel=1e-9)
        assert out["delta_p"] > ref["dp_darcy"]
        assert out["pump_power"] == pytest.approx(out["delta_p"] * Q / CFG.operating.pump_efficiency)


def test_series_layers_add_resistances():
    """rho* varies along s only -> uniform U, dp = mu U sum(ds_i / K_i) over one path (TPFA, harmonic faces)."""
    m = JacketModel(CFG, JacketOptions(spacing=2.5e-3, inertia=False), closures=LayeredClosures())
    g = m.grid
    rho = lambda s, z: 0.3 + 0.15 * np.sin(s / g.r_mean) ** 2  # noqa: E731  symmetric about the inlet
    out = m.evaluate(rho, 0.0, 4e-3)
    S, _ = g.meshgrid()
    r = rho(S, 0)[:, 0]
    path = (g.kind[:, 0] == LATTICE) & (g.theta_plot_deg() < 180)
    K = K0 * ((1 - r[path]) / 0.65) ** 3
    U = CFG.operating.nominal_flow_rate / (2 * g.gap * g.axial_length)
    dp = CFG.coolant.dynamic_viscosity * U * np.sum(g.ds[path] / K)
    assert out["delta_p"] == pytest.approx(dp, rel=1e-10)


def test_flux_stencil_exact_for_linear_pressure_with_cross_term():
    """Interior lattice faces reproduce -M grad p exactly for p = a s + b z (anisotropic M)."""
    m = JacketModel(CFG, JacketOptions(spacing=2.5e-3, spacing_z=2.0e-3), closures=uniform())
    g = m.grid
    rng = np.random.default_rng(3)
    nL = len(m._lat_idx)
    B = rng.normal(size=(2, 2))
    M0 = B @ B.T + np.eye(2)
    Fop = m._flow_operator(np.broadcast_to(M0, (nL, 2, 2)).copy())
    a, b = 3.0, -2.0
    S = g.s_local[:, None] + 0 * g.z_centres[None, :]
    Z = 0 * g.s_local[:, None] + g.z_centres[None, :]
    p = (a * S + b * Z).ravel()
    q = Fop @ p
    u_exact = -M0 @ np.array([a, b])
    # interior faces: lattice-lattice faces whose two cells have lattice neighbours all round
    lat = g.lattice
    ok = lat & np.roll(lat, 1, 0) & np.roll(lat, -1, 0) & np.roll(lat, 2, 0) & np.roll(lat, -2, 0)
    ok[:, :2] = False
    ok[:, -2:] = False
    fs = m._faces["s"]
    sel = ok.ravel()[fs["l"]] & ok.ravel()[fs["r"]] & (np.roll(np.arange(g.n_s), -1)[fs["l"] // g.n_z] != 0)
    sel &= np.abs(np.diff(g.s_faces)[fs["l"] // g.n_z] - np.diff(g.s_faces)[fs["r"] // g.n_z]) < 1e-15
    assert sel.sum() > 1000
    np.testing.assert_allclose(q[: m._n_sfaces][sel] / fs["area"][sel], u_exact[0], rtol=1e-10)
    fz = m._faces["z"]
    selz = ok.ravel()[fz["l"]] & ok.ravel()[fz["r"]]
    assert selz.sum() > 1000
    np.testing.assert_allclose(q[m._n_sfaces:][selz] / fz["area"][selz], u_exact[1], rtol=1e-10)


def test_anisotropic_mirror_symmetry():
    """Mirroring z (K_sz -> -K_sz) must leave dp, R and the flow split unchanged."""
    K = np.array([[1.0, 0.1, 0.05], [0.1, 1.1, 0.25], [0.05, 0.25, 0.9]]) * K0
    Km = K.copy()
    Km[2, :2] *= -1
    Km[:2, 2] *= -1
    outs = [JacketModel(CFG, JacketOptions(spacing=2.5e-3), closures=uniform(K=k)).evaluate(0.35, 0.5, 4e-3)
            for k in (K, Km)]
    assert outs[0]["delta_p"] == pytest.approx(outs[1]["delta_p"], rel=1e-9)
    assert outs[0]["thermal_resistance"] == pytest.approx(outs[1]["thermal_resistance"], rel=1e-9)
    assert all(o["picard_converged"] and o["mass_balance_error"] < 1e-12 for o in outs)


def test_finite_gap_factor_scales_darcy_resistance():
    o1 = JacketModel(CFG, JacketOptions(spacing=2.5e-3, inertia=False), closures=uniform(f_K=1.0)).evaluate()
    o2 = JacketModel(CFG, JacketOptions(spacing=2.5e-3, inertia=False), closures=uniform(f_K=0.7)).evaluate()
    assert o2["delta_p"] / o1["delta_p"] == pytest.approx(1 / 0.7, rel=1e-12)


# -----------------------------------------------------------------------------------------
# heat
# -----------------------------------------------------------------------------------------
NTU_OPTS = dict(wall_condition="temperature", wall_temperature=T_IN + 10.0, manifold_htc=0.0,
                fluid_conduction=False, lattice_conduction=False)


@pytest.mark.parametrize("spacing", [5e-3, 2.5e-3, 1.25e-3])
def test_epsilon_ntu_fixed_wall_temperature(spacing):
    m = JacketModel(CFG, JacketOptions(spacing=spacing, **NTU_OPTS), closures=uniform())
    out = m.evaluate(0.35, 0.0, 4e-3)
    ref = ntu_reference(m, 0.35, 0.0, 4e-3)
    eps = (out["T_out"] - T_IN) / 10.0
    n = int((m.grid.kind[:, 0] == LATTICE).sum() // 2)
    eps_discrete = 1 - (1 + ref["ntu"] / n) ** (-n)  # exact first-order upwind solution
    assert eps == pytest.approx(eps_discrete, rel=1e-10)
    # first-order convergence to the continuum epsilon-NTU: error ~ NTU^2 / (2 n) * (1 - eps)/eps
    err = (ref["effectiveness"] - eps) / ref["effectiveness"]
    assert 0 < err < 1.2 * ref["ntu"] ** 2 / (2 * n) * (1 - ref["effectiveness"]) / ref["effectiveness"] + 1e-12
    assert out["energy_balance_error"] < 1e-9


def test_epsilon_ntu_richardson_with_surrogate_closures():
    if not HAVE_MODELS:
        pytest.skip("trained closure models not in models/")
    eps = []
    for sp in (2.5e-3, 1.25e-3):
        m = JacketModel(CFG, JacketOptions(spacing=sp, **NTU_OPTS))
        out = m.evaluate(0.35, 0.0, 4e-3)
        eps.append((out["T_out"] - T_IN) / 10.0)
    ref = ntu_reference(m, 0.35, 0.0, 4e-3)
    assert 2 * eps[1] - eps[0] == pytest.approx(ref["effectiveness"], rel=5e-5)


def test_imposed_flux_bulk_solution():
    """z-uniform flux: away from the slots T_w - T_f = q_in / U_eff, dT_f/ds = q_in / (rho c_p U h), T_s at eta."""
    m = JacketModel(CFG_FLAT, JacketOptions(spacing=2.5e-3), closures=uniform())
    out = m.evaluate(0.35, 0.0, 4e-3, return_fields=True)
    f, g = out["fields"], m.grid
    ref = ntu_reference(m, 0.35, 0.0, 4e-3)
    q_in = CFG_FLAT.mean_heat_flux * CFG_FLAT.jacket.stator_radius / g.r_mean
    i = int(np.argmin(np.abs(g.theta_plot_deg() - 90.0)))
    j = g.n_z // 2
    assert f.T_w[i, j] - f.T_f[i, j] == pytest.approx(q_in / ref["U_eff"], rel=1e-8)
    assert (f.T_s_mean[i, j] - f.T_f[i, j]) / (f.T_w[i, j] - f.T_f[i, j]) == pytest.approx(ref["eta"], rel=1e-8)
    U = CFG.operating.nominal_flow_rate / (2 * g.gap * g.axial_length)
    dTds = (f.T_f[i + 1, j] - f.T_f[i - 1, j]) / (g.s_local[i + 1] - g.s_local[i - 1])
    rcp = CFG.coolant.density * CFG.coolant.specific_heat
    assert dTds == pytest.approx(q_in / (rcp * U * g.gap), rel=1e-8)
    assert np.ptp(f.T_w[i]) < 1e-8  # z-uniform
    dT = CFG.motor.heat_to_jacket / (rcp * CFG.operating.nominal_flow_rate)
    assert out["T_out"] - T_IN == pytest.approx(dT, rel=1e-10)
    # wall temperature reported on the stator side = lattice face + q'' t / k
    assert f.T_wall[i, j] - f.T_w[i, j] == pytest.approx(
        CFG_FLAT.mean_heat_flux * CFG.jacket.sleeve_thickness / CFG.material.conductivity, rel=1e-12)


def test_fin_limit_without_in_plane_conduction_equals_two_field_model():
    """With in-plane conduction off, the 3-field model reduces to the analytic fin conductance."""
    opts = dict(spacing=2.5e-3, lattice_conduction=False, fluid_conduction=False, sleeve_conduction=False,
                manifold_htc=1e5)  # slot sleeve needs a sink when sleeve conduction is off
    m = JacketModel(CFG_FLAT, JacketOptions(**opts), closures=uniform())
    out = m.evaluate(0.35, 0.0, 4e-3, return_fields=True)
    f = out["fields"]
    lat = m.grid.lattice
    ref = ntu_reference(m, 0.35, 0.0, 4e-3)
    q_in = CFG_FLAT.mean_heat_flux * CFG_FLAT.jacket.stator_radius / m.grid.r_mean
    np.testing.assert_allclose((f.T_w - f.T_f)[lat], q_in / ref["U_eff"], rtol=1e-7)


def test_isothermal_fin_bound_is_cooler():
    m = JacketModel(CFG, JacketOptions(spacing=2.5e-3), closures=uniform())
    r_fin = m.evaluate()["thermal_resistance"]
    r_iso = m.with_options(fin_model="isothermal").evaluate()["thermal_resistance"]
    assert r_iso < r_fin


# -----------------------------------------------------------------------------------------
# full model with the trained closures
# -----------------------------------------------------------------------------------------
def _graded_fields(model):
    C, Lx = model.grid.circumference, model.grid.axial_length
    rho = lambda s, z: 0.35 + 0.1 * np.sin(2 * np.pi * s / C) * np.cos(np.pi * z / Lx)  # noqa: E731
    w = lambda s, z: 0.5 + 0.5 * np.cos(2 * np.pi * s / C)  # noqa: E731
    L = lambda s, z: 3.5e-3 + 1e-3 * np.sin(4 * np.pi * s / C)  # noqa: E731
    return rho, w, L


@needs_models
@pytest.mark.parametrize("a_z", [1.0, 1.25])
def test_mass_and_energy_balance_on_graded_designs(a_z):
    m = JacketModel()
    rho, w, L = _graded_fields(m)
    for Q in (CFG.operating.flow_rate_sweep[0], CFG.operating.flow_rate_sweep[-1]):
        out = m.evaluate(rho, w, L, a_z, flow_rate=Q)
        assert out["picard_converged"]
        assert out["mass_balance_error"] < 1e-10  # requirement: < 0.5 %
        assert out["max_cell_divergence"] < 1e-10
        assert out["energy_balance_error"] < 1e-9
        assert sum(out["diagnostics"]["flow_split"]) == pytest.approx(1.0, abs=1e-10)
        assert out["heat_input"] == pytest.approx(CFG.motor.heat_to_jacket, rel=1e-12)


@needs_models
def test_reference_design_numbers_and_physics():
    out = evaluate(0.35, 0.0, 4e-3, return_fields=True)
    f = out["fields"]
    g = f.grid
    # Task 8 nominal point: Re_Dh ~ 195, h_sf ~ 4.7 kW/m^2K (bulk K; same superficial velocity)
    assert out["diagnostics"]["reynolds_median"] == pytest.approx(195, rel=0.03)
    assert out["diagnostics"]["h_sf_mean"] == pytest.approx(4718, rel=0.03)
    # finite-gap factor of the gyroid at N = 1.5: 1 - 0.266/1.5
    assert out["diagnostics"]["finite_gap_factor_K_mean"] == pytest.approx(1 - 0.266 / 1.5, abs=0.01)
    # outlet mixed temperature from the energy balance
    rcp = CFG.coolant.density * CFG.coolant.specific_heat
    assert out["T_out"] - T_IN == pytest.approx(CFG.motor.heat_to_jacket / (rcp * CFG.operating.nominal_flow_rate))
    # hottest point: the unfinned sleeve under the outlet slot, at an axial end (q'' = 2x)
    assert out["T_wall_max_s"] == pytest.approx(math.pi * g.r_mean, abs=CFG.manifolds.width / 2)
    assert min(out["T_wall_max_z"], CFG.jacket.axial_length - out["T_wall_max_z"]) < 3e-3
    assert out["T_wall_max"] > out["T_wall_max_lattice"]
    # pressure monotone from inlet to outlet along each path
    j = g.n_z // 2
    th = g.theta_plot_deg()
    path = (g.kind[:, j] == LATTICE) & (th > 0) & (th < 180)
    assert np.all(np.diff(f.pressure[path, j]) < 0)
    # temperatures ordered: wall > solid > coolant in the lattice
    lat = g.lattice
    assert np.all(f.T_w[lat] > f.T_s_mean[lat]) and np.all(f.T_s_mean[lat] > f.T_f[lat])
    assert out["manufacturable"] and out["min_structural_margin"] > 1


@needs_models
def test_flow_rate_trends():
    m = JacketModel()
    outs = [m.evaluate(0.35, 0.0, 4e-3, flow_rate=Q) for Q in CFG.operating.flow_rate_sweep]
    R = [o["thermal_resistance"] for o in outs]
    P = [o["pump_power"] for o in outs]
    assert np.all(np.diff(R) < 0) and np.all(np.diff(P) > 0)
    # Darcy-Forchheimer: dp/Q grows with Q
    dpq = [o["delta_p"] / o["flow_rate"] for o in outs]
    assert np.all(np.diff(dpq) > 0)


@needs_models
def test_grid_convergence_of_thermal_resistance():
    R = []
    for sp in (2.5e-3, 1.25e-3, 0.625e-3):
        m = JacketModel(options=JacketOptions(spacing=sp, spacing_z=2.5e-3))
        R.append(m.evaluate(0.35, 0.0, 4e-3)["thermal_resistance"])
    p = math.log2(abs(R[0] - R[1]) / abs(R[1] - R[2]))
    assert p > 1.5  # second-order-like convergence in s
    R_inf = R[2] + (R[2] - R[1]) / (2**p - 1)
    default = JacketModel().evaluate(0.35, 0.0, 4e-3)["thermal_resistance"]
    assert abs(default - R_inf) / R_inf < 0.01


@needs_models
def test_runtime_per_evaluation():
    m = JacketModel()
    rho, w, L = _graded_fields(m)
    m.evaluate(rho, w, L, 1.2)  # warm-up (imports, tables)
    t0 = time.perf_counter()
    out = m.evaluate(rho, w, L, 1.2)
    dt = time.perf_counter() - t0
    assert dt < 4.0, f"evaluation took {dt:.2f} s (target 1-2 s)"
    assert out["wall_time"] <= dt + 1e-6


@needs_models
def test_mirror_design_swaps_flow_split():
    m = JacketModel()
    C = m.grid.circumference
    rho = lambda s, z: 0.3 + 0.1 * (np.mod(s, C) < C / 2)  # noqa: E731  denser on the +s arc
    out = m.evaluate(rho, 0.0, 4e-3)
    rho_m = lambda s, z: rho(np.mod(-s, C), z)  # noqa: E731
    out_m = m.evaluate(rho_m, 0.0, 4e-3)
    a, b = out["diagnostics"]["flow_split"]
    am, bm = out_m["diagnostics"]["flow_split"]
    assert a < b  # denser arc carries less flow
    assert (am, bm) == pytest.approx((b, a), rel=1e-6)
    assert out_m["delta_p"] == pytest.approx(out["delta_p"], rel=1e-8)


# -----------------------------------------------------------------------------------------
# structure, mass, manufacturability, inputs
# -----------------------------------------------------------------------------------------
def test_sandwich_shear_equilibrium_and_uniform_values():
    m = JacketModel(CFG, JacketOptions(spacing=2.5e-3), closures=uniform())
    g = m.grid
    out = m.evaluate(0.35, 0.0, 4e-3)
    st = out["structural"]
    A_lat = g.area[g.lattice].sum()
    r_m, r_i = g.r_mean, CFG.jacket.lattice_inner_radius
    assert st["tau_rs_max"] == pytest.approx(CFG.loads.torque / (r_m * A_lat) * (r_m / r_i) ** 2, rel=1e-12)
    assert st["tau_rz_max"] == pytest.approx(CFG.loads.thrust / A_lat * (r_m / r_i), rel=1e-12)
    # with full-circumference lattice this is T / (2 pi r_i^2 L)
    assert A_lat < 2 * math.pi * r_m * g.axial_length
    # equilibrium on a graded design: sum(tau_rs A) r_m (r_i/r_m)^2 = T
    lat = m._lat_idx
    props = m._local_properties({"rho": np.linspace(0.25, 0.45, len(lat)), "w": np.zeros(len(lat)),
                                 "L": np.full(len(lat), 4e-3), "a_z": np.ones(len(lat))})
    flow = {"dp": 0.0}
    F = {"L": np.full(g.shape, 4e-3)}
    s = m._structure(props, F, flow)
    assert st["pressure"] == pytest.approx(CFG.loads.coolant_pressure_gauge + out["delta_p"])
    assert out["min_structural_margin"] == pytest.approx(min(out["lattice_margin"], out["outer_wall_margin"]))
    assert s["lattice_margin"] > 0
    # superposition: margin = sigma_allow / sum of the three contributions
    assert out["lattice_margin"] == pytest.approx(CFG.material.allowable_stress / (
        st["contrib_pressure"] + st["contrib_torque"] + st["contrib_thrust"]), rel=1e-12)
    # outer wall: slot strip bending p b^2 / (2 t^2)
    R_o = CFG.jacket.lattice_outer_radius + CFG.jacket.outer_wall_thickness / 2
    b = CFG.manifolds.width * R_o / r_m
    assert st["outer_wall_slot_bending"] == pytest.approx(st["pressure"] * b**2 / (2 * CFG.jacket.outer_wall_thickness**2))


def test_mass_accounting():
    m = JacketModel(CFG, JacketOptions(spacing=2.5e-3), closures=uniform())
    g = m.grid
    out = m.evaluate(0.3, 0.0, 4e-3)
    A_lat = g.area[g.lattice].sum()
    assert A_lat == pytest.approx((2 * math.pi * g.r_mean - 2 * CFG.manifolds.width) * g.axial_length)
    assert out["mass_lattice"] == pytest.approx(CFG.material.density * 0.3 * g.gap * A_lat)
    jac = CFG.jacket
    walls = CFG.material.density * math.pi * jac.axial_length * (
        (jac.stator_radius + jac.sleeve_thickness) ** 2 - jac.stator_radius**2
        + jac.outer_radius**2 - (jac.outer_radius - jac.outer_wall_thickness) ** 2)
    assert out["mass_walls"] == pytest.approx(walls)
    assert out["mass"] == pytest.approx(out["mass_lattice"] + out["mass_walls"])
    lighter = m.evaluate(0.25, 0.0, 4e-3)
    assert lighter["mass"] < out["mass"]


def test_manufacturability_flags():
    m = JacketModel(CFG, JacketOptions(spacing=2.5e-3), closures=uniform())
    ok = m.evaluate(0.35, 0.0, 4e-3)["manufacturability"]
    assert ok["manufacturable"] and ok["gradient_max"] == 0 and ok["N_min"] == pytest.approx(1.5)
    assert not m.evaluate(0.6, 0.0, 4e-3)["manufacturability"]["bounds"]["rho"]
    steep = m.evaluate(lambda s, z: 0.3 + 0.15 * (z > 0.025), 0.0, 4e-3)["manufacturability"]
    assert not steep["gradient_ok"] and not steep["manufacturable"]
    gentle = m.evaluate(lambda s, z: 0.35 + 0.02 * np.sin(2 * np.pi * z / 0.05), 0.0, 4e-3)["manufacturability"]
    assert gentle["gradient_ok"] and gentle["gradient_max"] > 0
    small = m.evaluate(0.35, 0.5, 2e-3)["manufacturability"]  # blend necks need L > 2 mm (Task 1)
    assert not small["cell_size_ok"]
    neck = m.evaluate(0.3, 0.5, 5e-3)["manufacturability"]
    assert neck["pinch_neck_area_fraction"] == pytest.approx(1.0)
    big = m.evaluate(0.35, 0.0, 7e-3)["manufacturability"]
    assert not big["finite_gap_valid"] and not big["bounds"]["L"]


def test_field_inputs():
    m = JacketModel(CFG, JacketOptions(spacing=2.5e-3), closures=uniform())
    a = m.evaluate(0.35, 0.0, 4e-3)
    b = m.evaluate(np.full(m.grid.shape, 0.35), lambda s, z: 0 * s, np.full((7, 3), 4e-3))
    assert b["thermal_resistance"] == pytest.approx(a["thermal_resistance"], rel=1e-12)
    assert b["delta_p"] == pytest.approx(a["delta_p"], rel=1e-12)
    with pytest.raises(ValueError):
        m.evaluate(np.full((2, 2, 2), 0.35))
    with pytest.raises(ValueError):
        m.evaluate(np.nan)
    with pytest.raises(ValueError):
        m.evaluate(flow_rate=0.0)


def test_coarse_array_is_interpolated_periodically():
    m = JacketModel(CFG, JacketOptions(spacing=2.5e-3), closures=uniform())
    arr = 0.3 + 0.1 * np.sin(2 * np.pi * (np.arange(64) + 0.5) / 64)[:, None] * np.ones((1, 4))
    out = m.evaluate(arr, 0.0, 4e-3, return_fields=True)
    f = out["fields"]
    S, _ = m.grid.meshgrid()
    exact = 0.3 + 0.1 * np.sin(2 * np.pi * S / m.grid.circumference)
    assert np.max(np.abs(f.rho - exact)) < 0.1 * (2 * np.pi / 64) ** 2  # linear interpolation error


def test_output_keys():
    m = JacketModel(CFG, JacketOptions(spacing=5e-3), closures=uniform())
    out = m.evaluate()
    for k in ("thermal_resistance", "delta_p", "pump_power", "mass", "min_structural_margin", "manufacturable",
              "manufacturability", "mass_balance_error", "energy_balance_error", "T_wall_max", "T_out",
              "picard_iterations", "wall_time"):
        assert k in out
    assert "fields" not in out


def test_finite_gap_features_broadcast():
    """Regression for the Task 7 helper: scalars and arrays may be mixed."""
    if not HAVE_MODELS:
        pytest.skip("finite-gap model not in models/")
    from voxlat.homogenization.finite_gap import FiniteGapCorrection

    c = FiniteGapCorrection.load()
    f = c.factor("K_t", np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 0.5]), 0.35)
    assert f.shape == (3,)
    assert c.factor("K_t", 1.5, 0.0, 0.35) == pytest.approx(f[0] * 0 + c.factor("K_t", np.array([1.5]), 0.0, 0.35)[0])
