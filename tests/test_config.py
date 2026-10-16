"""Task 0 smoke tests: reference config loads, is frozen/typed, and q''(z) conserves Q."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import integrate

from voxlat.utils.config import (
    ConfigError,
    Range,
    ReferenceConfig,
    default_config_path,
    heat_flux_profile,
    load_config,
)


@pytest.fixture(scope="module")
def cfg() -> ReferenceConfig:
    return load_config()


def test_config_file_exists():
    assert default_config_path().is_file()


def test_config_loads_and_is_typed(cfg):
    assert isinstance(cfg, ReferenceConfig)
    assert isinstance(cfg.material.youngs_modulus, float)
    assert isinstance(cfg.design_bounds.relative_density, Range)
    assert isinstance(cfg.operating.flow_rate_sweep, tuple)
    assert all(isinstance(q, float) for q in cfg.operating.flow_rate_sweep)


def test_config_is_frozen(cfg):
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.motor.heat_to_jacket = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.motor = None  # type: ignore[misc]


def test_reference_numbers_si(cfg):
    """Spot-check every section against plan section 2 (converted to SI)."""
    assert cfg.motor.continuous_power == 15e3
    assert cfg.motor.efficiency == 0.94
    assert cfg.motor.heat_to_jacket == 770.0
    j = cfg.jacket
    assert j.lattice_inner_radius == pytest.approx(0.0715)
    assert j.lattice_outer_radius == pytest.approx(0.0775)
    assert j.outer_radius == pytest.approx(0.0795)
    assert j.axial_length == pytest.approx(0.050)
    assert cfg.coolant.prandtl == pytest.approx(20.72, rel=1e-3)  # "Pr near 20"
    assert cfg.operating.inlet_temperature == pytest.approx(323.15)
    assert cfg.operating.nominal_flow_rate_lpm == pytest.approx(3.0)
    lpm = np.array(cfg.operating.flow_rate_sweep) * 60e3
    np.testing.assert_allclose(lpm, [1, 2, 3, 4, 5], rtol=1e-12)
    assert cfg.manifolds.outlet_angle == pytest.approx(math.pi)
    assert cfg.manifolds.width == pytest.approx(0.015)
    assert cfg.material.youngs_modulus == pytest.approx(70e9)
    assert cfg.material.allowable_stress == pytest.approx(80e6)
    assert cfg.loads.coolant_pressure_gauge == pytest.approx(2e5)
    b = cfg.design_bounds
    assert (b.relative_density.lo, b.relative_density.hi) == (0.20, 0.50)
    assert (b.cell_size.lo, b.cell_size.hi) == pytest.approx((0.002, 0.006))
    assert cfg.manufacturing.min_wall_thickness == pytest.approx(0.35e-3)
    assert cfg.manufacturing.min_pore_size == pytest.approx(0.8e-3)


def test_mean_heat_flux(cfg):
    expected = 770.0 / (2 * math.pi * 0.070 * 0.050)
    assert cfg.mean_heat_flux == pytest.approx(expected, rel=1e-12)
    assert cfg.mean_heat_flux == pytest.approx(35e3, rel=0.01)  # "~35 kW/m^2"


def test_heat_flux_shape(cfg):
    L = cfg.jacket.axial_length
    q0, qc, q1 = cfg.q_wall(0.0), cfg.q_wall(L / 2), cfg.q_wall(L)
    assert q0 / qc == pytest.approx(2.0, rel=1e-12)  # ends 2x centre
    assert q0 == pytest.approx(q1, rel=1e-12)  # symmetric
    z = np.linspace(0, L, 101)
    assert np.all(cfg.q_wall(z) > 0)
    assert isinstance(cfg.q_wall(0.01), float)


@pytest.mark.parametrize("n", [51, 201, 1001])  # trapezoid error ~ 2*a*h_zeta^2 -> 2e-4 at n = 51
def test_heat_flux_integrates_to_Q_trapezoid(cfg, n):
    """Smoke test from the plan: int q''(z) 2 pi r_s dz = Q within 0.1 %."""
    L = cfg.jacket.axial_length
    z = np.linspace(0.0, L, n)
    Q_num = integrate.trapezoid(cfg.q_wall(z), z) * 2 * math.pi * cfg.jacket.stator_radius
    assert Q_num == pytest.approx(cfg.motor.heat_to_jacket, rel=1e-3)


def test_heat_flux_integrates_to_Q_quadrature(cfg):
    L = cfg.jacket.axial_length
    val, _ = integrate.quad(cfg.q_wall, 0.0, L, epsabs=0, epsrel=1e-13)
    Q_num = val * 2 * math.pi * cfg.jacket.stator_radius
    assert Q_num == pytest.approx(cfg.motor.heat_to_jacket, rel=1e-10)


@pytest.mark.parametrize("a", [0.0, 0.25, 0.5, 0.9])
def test_profile_conserves_Q_for_any_amplitude(a):
    val, _ = integrate.quad(
        lambda z: heat_flux_profile(z, axial_length=0.05, mean_flux=1.0, amplitude=a), 0, 0.05
    )
    assert val == pytest.approx(0.05, rel=1e-12)


def test_heat_flux_rejects_out_of_range(cfg):
    with pytest.raises(ValueError):
        cfg.q_wall(-1e-3)
    with pytest.raises(ValueError):
        cfg.q_wall(cfg.jacket.axial_length * 1.01)


# ---- loader robustness -------------------------------------------------------
def _write_variant(tmp_path: Path, mutate) -> Path:
    raw = yaml.safe_load(default_config_path().read_text(encoding="utf-8"))
    mutate(raw)
    p = tmp_path / "variant.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def test_unknown_key_rejected(tmp_path):
    p = _write_variant(tmp_path, lambda r: r["coolant"].__setitem__("colour", "blue"))
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(p)


def test_missing_key_rejected(tmp_path):
    p = _write_variant(tmp_path, lambda r: r["material"].pop("density"))
    with pytest.raises(ConfigError, match="missing key"):
        load_config(p)


def test_inconsistent_value_rejected(tmp_path):
    p = _write_variant(tmp_path, lambda r: r["heat_flux"].__setitem__("amplitude", 1.2))
    with pytest.raises(ConfigError):
        load_config(p)


def test_overrides():
    cfg2 = load_config(overrides={"operating": {"nominal_flow_rate": 1e-4}})
    assert cfg2.operating.nominal_flow_rate == 1e-4
    assert cfg2.operating.pump_efficiency == 0.30  # untouched siblings kept


def test_env_var_path(tmp_path, monkeypatch):
    p = _write_variant(tmp_path, lambda r: r["loads"].__setitem__("torque", 50.0))
    monkeypatch.setenv("VOXLAT_CONFIG", str(p))
    assert load_config().loads.torque == 50.0
