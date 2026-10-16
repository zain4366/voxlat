"""Tests for voxlat.device.baselines (Task 10): duct correlations, B1 channel model, B2/B3 wrappers, tuning."""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from voxlat.device import JacketModel, JacketOptions
from voxlat.device.baselines import (
    ChannelDesign,
    ChannelJacketModel,
    ChannelOptions,
    GradedDensityDesign,
    UniformLatticeDesign,
    b1_candidates,
    channel_hydraulics,
    coil_nusselt_laminar,
    coil_nusselt_turbulent,
    curved_duct_friction,
    curved_duct_nusselt,
    design_from_dict,
    developing_nusselt_mean,
    evaluate,
    helical_re_crit,
    pareto_front,
    path_coordinate,
    rect_duct_fre,
    rect_duct_nu_h1,
    rib_conductance,
    select_best,
    tune_b1,
    tune_b2,
)
from voxlat.utils import load_config

warnings.filterwarnings("ignore", category=RuntimeWarning)

TASK9_KEYS = {
    "thermal_resistance", "T_wall_max", "T_wall_max_lattice", "T_wall_max_s", "T_wall_max_z", "T_wall_mean", "T_in",
    "T_out", "heat_input", "delta_p", "pump_power", "hydraulic_power", "flow_rate", "mass", "mass_lattice",
    "mass_walls", "coolant_volume", "min_structural_margin", "lattice_margin", "outer_wall_margin", "structural",
    "manufacturable", "manufacturability", "mass_balance_error", "max_cell_divergence", "energy_balance_error",
    "picard_iterations", "picard_converged", "diagnostics", "wall_time",
}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def chan(cfg):
    return ChannelJacketModel(cfg)


@pytest.fixture(scope="module")
def lattice(cfg):
    return JacketModel(cfg, JacketOptions(check_bounds="ignore"))


HEL = ChannelDesign(2.0e-3, 0.5e-3, 16, "helical")
CIRC = ChannelDesign.from_passes(20, 0.5e-3, 1, "circumferential")


# ----------------------------------------------------------------------------- correlations
def test_rect_duct_fre_shah_london_values():
    # Darcy f Re = 4 x Fanning; Shah & London: square 14.227, alpha 0.5 15.548, plates 24 (Fanning)
    assert rect_duct_fre(1.0) == pytest.approx(4 * 14.227, rel=2e-4)
    assert rect_duct_fre(0.5) == pytest.approx(4 * 15.548, rel=2e-4)
    assert rect_duct_fre(1e-3) == pytest.approx(96.0, rel=1e-2)
    a = np.linspace(0.05, 1, 20)
    assert np.all(np.diff(rect_duct_fre(a)) < 0)  # decreasing with aspect ratio
    with pytest.raises(ValueError):
        rect_duct_fre(1.5)


def test_rect_duct_nu_h1():
    assert rect_duct_nu_h1(1.0) == pytest.approx(3.61, abs=0.01)
    assert rect_duct_nu_h1(0.0) == pytest.approx(8.235)
    assert rect_duct_nu_h1(0.5) == pytest.approx(4.12, abs=0.03)


def test_helical_re_crit_schmidt():
    # fluids.friction.helical_Re_crit(Di=0.02, Dc=0.5) -> 6946.79 (Schmidt)
    assert helical_re_crit(0.04) == pytest.approx(6946.7925, rel=1e-6)
    assert helical_re_crit(0.0) == pytest.approx(2300.0)


def test_curved_friction_limits_and_continuity():
    # laminar straight limit (d/D -> 0): 64/Re
    assert curved_duct_friction(500.0, 1e-12) == pytest.approx(64 / 500, rel=1e-6)
    # rectangular straight limit
    assert curved_duct_friction(500.0, 1e-12, 1.0) == pytest.approx(rect_duct_fre(1.0) / 500, rel=1e-6)
    # Schmidt laminar and turbulent branches at Re_crit: within 5 % for d/D 0.03-0.06, but the published pair
    # jumps by ~23 % at d/D = 0.01 (documented in STATUS; the selected B1 designs are far below Re_crit)
    jumps = []
    for dr in (0.01, 0.03, 0.06):
        rc = float(helical_re_crit(dr))
        f_lo = float(curved_duct_friction(rc * (1 - 1e-9), dr))
        f_hi = float(curved_duct_friction(rc * (1 + 1e-9), dr))
        jumps.append(f_hi / f_lo - 1)
    assert abs(jumps[1]) < 0.05 and abs(jumps[2]) < 0.15
    assert jumps[0] == pytest.approx(0.234, abs=0.01)
    # curvature raises friction
    assert curved_duct_friction(800.0, 0.03) > curved_duct_friction(800.0, 1e-9)


def test_curved_nusselt_regimes_continuous():
    Pr = 20.7
    for dr in (0.01, 0.03):
        rc = float(helical_re_crit(dr))
        lo = float(curved_duct_nusselt(rc * (1 - 1e-9), Pr, dr))
        hi = float(curved_duct_nusselt(rc * (1 + 1e-9), Pr, dr))
        assert hi == pytest.approx(lo, rel=1e-6)
        a = float(curved_duct_nusselt(2.2e4 * (1 - 1e-9), Pr, dr))
        b = float(curved_duct_nusselt(2.2e4 * (1 + 1e-9), Pr, dr))
        assert b == pytest.approx(a, rel=1e-6)
    Re = np.logspace(1, 5, 200)
    assert np.all(np.diff(curved_duct_nusselt(Re, Pr, 0.03)) > 0)
    # floor: never below the straight fully developed value
    assert np.all(curved_duct_nusselt(Re, 1.0, 0.001, 0.25) >= rect_duct_nu_h1(0.25) - 1e-12)


def test_coil_nusselt_reference_values():
    # hand evaluation of the VDI laminar form and the Gnielinski coil form
    dr, Re, Pr = 0.0322, 544.0, 20.72
    m = 0.5 + 0.2903 * dr**0.194
    ref = 3.66 + 0.08 * (1 + 0.8 * dr**0.9) * Re**m * Pr ** (1 / 3)
    assert coil_nusselt_laminar(Re, Pr, dr) == pytest.approx(ref, rel=1e-12)
    xi = 0.3164 * 1e4 ** -0.25 + 0.03 * math.sqrt(dr)
    ref_t = xi / 8 * 1e4 * 5.0 / (1 + 12.7 * math.sqrt(xi / 8) * (5.0 ** (2 / 3) - 1))
    assert coil_nusselt_turbulent(1e4, 5.0, dr) == pytest.approx(ref_t, rel=1e-12)


def test_developing_nusselt_mean_limits():
    # long duct -> 3.657 (Baehr-Stephan constant wall temperature)
    assert developing_nusselt_mean(100.0, 1.0, 1e-3, 1e3) == pytest.approx(3.657, rel=1e-3)
    assert developing_nusselt_mean(500.0, 20.0, 4e-3, 0.1) > 10


def test_rib_conductance_adiabatic_and_tip():
    k, t, H, h = 130.0, 1e-3, 6e-3, 2000.0
    G, eta = rib_conductance(h, k, t, H)
    m = math.sqrt(2 * h / (k * t))
    assert G == pytest.approx(math.sqrt(2 * h * k * t) * math.tanh(m * H), rel=1e-12)
    assert eta == pytest.approx(math.tanh(m * H) / (m * H), rel=1e-12)
    # finite-difference fin with a tip conductance
    Gt = 0.8
    n = 4000
    x = np.linspace(0, H, n + 1)
    dx = H / n
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    A[0, 0] = 1
    b[0] = 1.0
    for i in range(1, n):
        A[i, i - 1] = A[i, i + 1] = k * t / dx**2
        A[i, i] = -2 * k * t / dx**2 - 2 * h
    # tip: half cell balance
    A[n, n - 1] = k * t / dx
    A[n, n] = -k * t / dx - h * dx - Gt
    th = np.linalg.solve(A, b)
    q_base = k * t * (th[0] - th[1]) / dx + h * dx * th[0]  # conduction + half-cell convection
    G2, _ = rib_conductance(h, k, t, H, Gt)
    assert G2 == pytest.approx(q_base, rel=2e-3)
    assert G2 > G


# ----------------------------------------------------------------------------- designs
def test_channel_design_geometry_and_json():
    d = ChannelDesign.from_passes(20, 0.5e-3, 16)
    assert d.period == pytest.approx(2.5e-3)
    assert d.channel_width == pytest.approx(2.0e-3)
    assert d.pitch == pytest.approx(40e-3)
    assert d.rib_fraction == pytest.approx(0.2)
    assert design_from_dict(json.loads(json.dumps(d.to_dict()))) == d
    assert CIRC.to_dict()["family"] == "B1s"
    with pytest.raises(ValueError):
        ChannelDesign(1e-3, 1e-3, 1, "spiral")
    g = GradedDensityDesign((0.3, 0.35, 0.4, 0.45), 0.02, 3e-3)
    assert design_from_dict(json.loads(json.dumps(g.to_dict()))) == g
    u = UniformLatticeDesign(0.3, 3e-3)
    assert design_from_dict(u.to_dict()) == u


def test_b1_candidates_respect_limits(cfg):
    ds = b1_candidates(cfg)
    assert all(d.channel_width >= cfg.manufacturing.min_pore_size - 1e-12 for d in ds)
    assert all(d.rib_thickness >= cfg.manufacturing.min_wall_thickness - 1e-12 for d in ds)
    L = cfg.jacket.axial_length
    assert all(d.n_starts <= round(L / d.period) for d in ds)
    assert all(abs(L / d.period - round(L / d.period)) < 1e-9 for d in ds)


# ----------------------------------------------------------------------------- hydraulics
def test_hydraulics_hand_calculation(cfg):
    Q = cfg.operating.nominal_flow_rate
    hy = channel_hydraulics(HEL, cfg, Q)
    h = cfg.jacket.lattice_gap
    C = 2 * math.pi * cfg.jacket.lattice_mean_radius
    D_h = 2 * HEL.channel_width * h / (HEL.channel_width + h)
    U = Q / 16 / (HEL.channel_width * h)
    Re = U * D_h / cfg.coolant.kinematic_viscosity
    ell = cfg.jacket.axial_length / HEL.pitch * math.hypot(C, HEL.pitch)
    assert hy["D_h"] == pytest.approx(D_h)
    assert hy["reynolds"] == pytest.approx(Re)
    assert hy["path_length"] == pytest.approx(ell)
    dr = D_h / (2 * cfg.jacket.lattice_mean_radius * (1 + (HEL.pitch / C) ** 2))
    f = float(curved_duct_friction(Re, dr, HEL.channel_width / h))
    assert hy["delta_p"] == pytest.approx((f * ell / D_h + 1.25) * 0.5 * cfg.coolant.density * U**2, rel=1e-12)
    # circumferential path = the lattice path between the slots, 2 N_pass parallel channels
    hc = channel_hydraulics(CIRC, cfg, Q)
    assert hc["path_length"] == pytest.approx(math.pi * cfg.jacket.lattice_mean_radius - cfg.manifolds.width)
    assert hc["n_channels"] == pytest.approx(40)


def test_hydraulics_nu_models(cfg):
    Q = cfg.operating.nominal_flow_rate
    nom = channel_hydraulics(HEL, cfg, Q)["nusselt"]
    st = channel_hydraulics(HEL, cfg, Q, nu_model="straight")["nusselt"]
    dev = channel_hydraulics(HEL, cfg, Q, nu_model="developing_mean")["nusselt"]
    assert st == pytest.approx(float(rect_duct_nu_h1(HEL.channel_width / cfg.jacket.lattice_gap)))
    assert nom >= st and dev >= nom
    with pytest.raises(ValueError):
        channel_hydraulics(HEL, cfg, Q, nu_model="magic")


# ----------------------------------------------------------------------------- B1 heat model
def test_helical_exact_marching_without_sleeve_conduction(cfg):
    m = ChannelJacketModel(cfg, ChannelOptions(sleeve_conduction=False))
    out = m.evaluate(HEL, return_fields=True)
    f, g = out["fields"], m.grid("helical")
    rcp = cfg.coolant.density * cfg.coolant.specific_heat
    Q = cfg.operating.nominal_flow_rate
    q = f.q_in[0]
    T_exact = cfg.operating.inlet_temperature + g.circumference / (rcp * Q) * np.cumsum(q * g.dz)
    np.testing.assert_allclose(f.T_f, np.broadcast_to(T_exact, f.T_f.shape), atol=1e-9)
    np.testing.assert_allclose(f.T_w - f.T_f, q[None, :] / out["diagnostics"]["U_eff"] + 0 * f.T_f, rtol=1e-10)
    # hottest point at an axial end (flux peaks there; the outlet end is hottest)
    assert out["T_wall_max_z"] > 0.9 * cfg.jacket.axial_length


@pytest.mark.parametrize("design", [HEL, CIRC], ids=["helical", "circumferential"])
@pytest.mark.parametrize("lpm", [1.0, 5.0])
def test_balances(chan, cfg, design, lpm):
    Q = lpm / 6e4
    out = chan.evaluate(design, Q)
    rcp = cfg.coolant.density * cfg.coolant.specific_heat
    assert out["energy_balance_error"] < 1e-9
    assert out["mass_balance_error"] < 1e-12
    assert out["max_cell_divergence"] < 1e-12
    assert out["heat_input"] == pytest.approx(cfg.motor.heat_to_jacket, rel=1e-12)
    assert out["T_out"] - out["T_in"] == pytest.approx(cfg.motor.heat_to_jacket / (rcp * Q), rel=1e-9)


@pytest.mark.parametrize("spacing", [5e-3, 1.25e-3])
def test_circumferential_eps_matches_discrete_upwind(cfg, spacing):
    T_in = cfg.operating.inlet_temperature
    m = ChannelJacketModel(cfg, ChannelOptions(spacing=spacing, wall_condition="temperature",
                                               wall_temperature=T_in + 10, manifold_htc=0.0))
    out = m.evaluate(CIRC)
    g = m.grid("circumferential")
    n = int(np.sum(g.kind[:, 0] == 0) // 2)
    rcp = cfg.coolant.density * cfg.coolant.specific_heat
    ell = math.pi * g.r_mean - cfg.manifolds.width
    ntu = out["diagnostics"]["U_eff"] * ell * g.axial_length / (rcp * cfg.operating.nominal_flow_rate / 2)
    eps = (out["T_out"] - T_in) / 10
    assert eps == pytest.approx(1 - (1 + ntu / n) ** (-n), rel=1e-10)
    assert eps == pytest.approx(1 - math.exp(-ntu), rel=0.02)


def test_sleeve_conduction_and_slots(chan, cfg):
    # with slots, the hottest point is the unfinned sleeve under a slot; R_active < R
    out = chan.evaluate(CIRC)
    assert out["R_active"] < out["thermal_resistance"]
    th = math.degrees(out["T_wall_max_s"] / cfg.jacket.lattice_mean_radius) % 360
    assert min(abs(th - 180), th, 360 - th) < 8
    # helical: no slots -> R_active == R
    o2 = chan.evaluate(HEL)
    assert o2["R_active"] == pytest.approx(o2["thermal_resistance"])
    # sleeve conduction can only lower the peak
    o3 = ChannelJacketModel(cfg, ChannelOptions(sleeve_conduction=False)).evaluate(CIRC)
    assert out["T_wall_max"] < o3["T_wall_max"]


def test_conductance_and_outer_wall_fin(chan, cfg):
    out = chan.evaluate(HEL)
    d = out["diagnostics"]
    G, _ = rib_conductance(d["h_c"], cfg.material.conductivity, HEL.rib_thickness, cfg.jacket.lattice_gap)
    assert d["U_eff"] == pytest.approx((d["h_c"] * HEL.channel_width + G) / HEL.period, rel=1e-12)
    o2 = chan.with_options(outer_wall_fin=True).evaluate(HEL)
    assert o2["diagnostics"]["U_eff"] > d["U_eff"]
    assert o2["thermal_resistance"] < out["thermal_resistance"]


def test_structure_and_mass(chan, cfg):
    out = chan.evaluate(HEL)
    s = out["structural"]
    A = 2 * math.pi * cfg.jacket.lattice_mean_radius * cfg.jacket.axial_length
    r_m, r_i = cfg.jacket.lattice_mean_radius, cfg.jacket.lattice_inner_radius
    tau = cfg.loads.torque / (r_m * A) * (r_m / r_i) ** 2 / HEL.rib_fraction
    assert s["tau_torque"] == pytest.approx(tau, rel=1e-12)
    assert out["mass_ribs"] == pytest.approx(cfg.material.density * cfg.jacket.lattice_gap * HEL.rib_fraction * A, rel=1e-12)
    lat = evaluate(UniformLatticeDesign(0.35, 4e-3))
    assert out["mass_walls"] == pytest.approx(lat["mass_walls"], rel=1e-12)
    # K_t scales the rib stress linearly
    o2 = chan.with_options(root_stress_factor=5.0).evaluate(HEL)
    assert o2["lattice_margin"] == pytest.approx(out["lattice_margin"] / 2.0, rel=1e-12)
    # manufacturability flags
    thin = ChannelDesign(0.5e-3, 0.2e-3, 1)
    man = chan.evaluate(thin)["manufacturability"]
    assert not man["rib_ok"] and not man["channel_ok"] and not man["manufacturable"]


def test_flux_scaling_linear(cfg):
    # temperature rise above inlet is linear in Q for fixed flow (linear model)
    c2 = load_config(overrides={"motor": {"heat_to_jacket": 385.0}})
    o1 = ChannelJacketModel(cfg).evaluate(HEL)
    o2 = ChannelJacketModel(c2).evaluate(HEL)
    assert (o2["T_wall_max"] - o2["T_in"]) == pytest.approx(0.5 * (o1["T_wall_max"] - o1["T_in"]), rel=1e-9)
    assert o2["thermal_resistance"] == pytest.approx(o1["thermal_resistance"], rel=1e-9)


# ----------------------------------------------------------------------------- interface parity
@pytest.mark.parametrize("design", [HEL, CIRC, UniformLatticeDesign(0.3, 3e-3),
                                    GradedDensityDesign((0.3, 0.3, 0.35, 0.4), 0.01, 4e-3)],
                         ids=["B1", "B1s", "B2", "B3"])
def test_interface_has_task9_keys(design):
    out = evaluate(design, return_fields=True)
    assert TASK9_KEYS <= set(out)
    assert {"T_wall_max_active", "R_active", "design", "fields"} <= set(out)
    assert out["fields"].T_wall.shape == out["fields"].grid.shape
    assert design_from_dict(out["design"]) == design
    assert evaluate(design.to_dict())["thermal_resistance"] == pytest.approx(out["thermal_resistance"])


def test_path_coordinate(lattice, cfg):
    g = lattice.grid
    xi = path_coordinate(g, cfg)
    th = g.theta_plot_deg()
    assert np.all((xi >= 0) & (xi <= 1))
    assert np.all(xi[np.any(g.kind == 1, axis=1)] == 0)  # inlet slot
    assert np.all(xi[np.any(g.kind == 2, axis=1)] == 1)  # outlet slot
    # mirror symmetry about the inlet-outlet axis
    for t in (30.0, 90.0, 150.0):
        a = xi[np.argmin(np.abs(th - t))]
        b = xi[np.argmin(np.abs(th - (360 - t)))]
        assert a == pytest.approx(b, abs=0.01)
    assert xi[np.argmin(np.abs(th - 90))] == pytest.approx(0.5, abs=0.01)


def test_graded_flat_equals_uniform(lattice):
    a = evaluate(GradedDensityDesign((0.33,) * 4, 0.0, 3.5e-3), model=lattice)
    b = evaluate(UniformLatticeDesign(0.33, 3.5e-3), model=lattice)
    assert a["thermal_resistance"] == pytest.approx(b["thermal_resistance"], rel=1e-10)
    assert a["delta_p"] == pytest.approx(b["delta_p"], rel=1e-10)
    # axial amplitude follows psi: denser at the ends for amp_z > 0
    g = GradedDensityDesign((0.35,) * 4, 0.05, 4e-3).rho_field(lattice)
    assert g[100, 0] == pytest.approx(0.35 + 0.05 * 0.5 * (12 * (lattice.grid.z_centres[0] / 0.05 - 0.5) ** 2 - 1))
    assert g[100, 0] > g[100, lattice.grid.n_z // 2]


def test_uniform_lattice_matches_task9(lattice):
    a = evaluate(UniformLatticeDesign(0.35, 4e-3), model=lattice)
    b = lattice.evaluate(0.35, 0.0, 4e-3)
    assert a["thermal_resistance"] == pytest.approx(b["thermal_resistance"], rel=1e-12)
    assert a["R_active"] == pytest.approx((b["T_wall_max_lattice"] - b["T_in"]) / 770.0, rel=1e-12)


# ----------------------------------------------------------------------------- selection
def test_select_best_and_pareto():
    df = pd.DataFrame(dict(thermal_resistance=[3.0, 2.0, 1.0, 2.5, 0.5], pump_power=[1.0, 2.0, 3.0, 2.5, 0.1],
                           mass=[1.0] * 5, manufacturable=[True, True, True, True, False],
                           min_structural_margin=[2.0, 2.0, 2.0, 2.0, 2.0]))
    assert select_best(df, 2.0) == 1
    assert select_best(df, None) == 2
    with pytest.raises(ValueError):
        select_best(df, 0.5)
    m = pareto_front(df, ("thermal_resistance", "pump_power"))
    assert list(m) == [True, True, True, False, False]


def test_tune_small_grids(cfg, chan, lattice):
    r = tune_b1(layout="helical", pump_power_cap=0.5, model=chan, n_pass=(10, 20), rib_thickness=(0.5e-3, 1e-3),
                n_starts=(4, 8, 16))
    ok = r.table.manufacturable & (r.table.min_structural_margin >= 1) & (r.table.pump_power <= 0.5)
    assert r.best_output["thermal_resistance"] == pytest.approx(r.table.thermal_resistance[ok].min())
    assert r.best_output["pump_power"] <= 0.5
    r2 = tune_b2(pump_power_cap=0.4, model=lattice, rho_values=(0.3, 0.4), L_values=(3e-3, 4e-3))
    ok = r2.table.pump_power <= 0.4
    assert r2.best_output["thermal_resistance"] == pytest.approx(r2.table.thermal_resistance[ok].min())
    assert len(r2.table) == 4
