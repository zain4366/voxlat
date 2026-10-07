"""Typed, frozen access to the VoxLat reference problem (``configs/reference.yaml``).

Every physical number used anywhere in VoxLat comes from this config, in SI
units. ``load_config()`` parses the YAML into a tree of frozen dataclasses,
rejects unknown or missing keys, coerces numbers to ``float``, and checks basic
physical consistency.

Heat-flux profile
-----------------
The heat flux entering the jacket through the stator sleeve is

    q''(z) = q_mean * [1 + a * (12 (zeta - 1/2)^2 - 1)],   zeta = z / L_ax,
    q_mean = Q / (2 pi r_s L_ax),

with Q the heat into the jacket, r_s the stator (sleeve inner) radius, L_ax the
axial length and a the amplitude (0.25 -> ends carry 2x the centre value).
Because  int_0^1 12 (zeta - 1/2)^2 dzeta = 1,  the bracket averages to exactly 1,
so  int_0^{L_ax} q''(z) 2 pi r_s dz = Q  for any a. The profile stays positive
for a < 1 (minimum 1 - a at mid-length).

Example
-------
>>> from voxlat.utils.config import load_config
>>> cfg = load_config()
>>> round(cfg.mean_heat_flux)            # W/m^2
35014
>>> cfg.q_wall(0.0) / cfg.q_wall(0.025)
2.0
"""

from __future__ import annotations

import math
import os
import types
import typing
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from voxlat.utils.paths import configs_dir

__all__ = [
    "Range",
    "MotorConfig",
    "JacketGeometry",
    "HeatFluxConfig",
    "CoolantConfig",
    "OperatingConfig",
    "ManifoldConfig",
    "MaterialConfig",
    "LoadsConfig",
    "DesignBounds",
    "ManufacturingConfig",
    "ReferenceConfig",
    "ConfigError",
    "default_config_path",
    "load_config",
    "heat_flux_profile",
]

LITRE_PER_MIN = 1.0e-3 / 60.0  # m^3/s


class ConfigError(ValueError):
    """Raised for malformed or physically inconsistent configuration."""


# -----------------------------------------------------------------------------
# Dataclasses (all frozen)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Range:
    """Closed interval [lo, hi]."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if not self.hi >= self.lo:
            raise ConfigError(f"Range upper bound {self.hi} < lower bound {self.lo}")

    @property
    def span(self) -> float:
        return self.hi - self.lo

    def contains(self, x: Any) -> Any:
        """Elementwise lo <= x <= hi."""
        x = np.asarray(x, dtype=float)
        out = (x >= self.lo) & (x <= self.hi)
        return bool(out) if out.ndim == 0 else out

    def clip(self, x: Any) -> Any:
        return np.clip(x, self.lo, self.hi)


@dataclass(frozen=True)
class MotorConfig:
    continuous_power: float  # W
    efficiency: float  # -
    heat_to_jacket: float  # W  (Q)

    @property
    def total_loss(self) -> float:
        """Total motor loss P (1/eta - 1) in W (for reference; Q is a share of it)."""
        return self.continuous_power * (1.0 / self.efficiency - 1.0)


@dataclass(frozen=True)
class JacketGeometry:
    stator_radius: float  # m, sleeve inner radius
    sleeve_thickness: float  # m
    lattice_gap: float  # m, h
    outer_wall_thickness: float  # m
    axial_length: float  # m

    @property
    def lattice_inner_radius(self) -> float:
        return self.stator_radius + self.sleeve_thickness

    @property
    def lattice_outer_radius(self) -> float:
        return self.lattice_inner_radius + self.lattice_gap

    @property
    def outer_radius(self) -> float:
        """Outer surface of the outer wall."""
        return self.lattice_outer_radius + self.outer_wall_thickness

    @property
    def lattice_mean_radius(self) -> float:
        return 0.5 * (self.lattice_inner_radius + self.lattice_outer_radius)

    @property
    def lattice_volume(self) -> float:
        """Annular volume of the lattice gap, m^3."""
        return math.pi * (self.lattice_outer_radius**2 - self.lattice_inner_radius**2) * self.axial_length


@dataclass(frozen=True)
class HeatFluxConfig:
    profile: str  # only "end_peaked_quadratic" is defined
    amplitude: float  # a


@dataclass(frozen=True)
class CoolantConfig:
    name: str
    density: float  # kg/m^3
    specific_heat: float  # J/(kg K)
    conductivity: float  # W/(m K)
    kinematic_viscosity: float  # m^2/s

    @property
    def dynamic_viscosity(self) -> float:
        return self.density * self.kinematic_viscosity

    @property
    def prandtl(self) -> float:
        return self.dynamic_viscosity * self.specific_heat / self.conductivity


@dataclass(frozen=True)
class OperatingConfig:
    inlet_temperature: float  # K
    nominal_flow_rate: float  # m^3/s
    flow_rate_sweep: tuple[float, ...]  # m^3/s
    pump_efficiency: float  # -

    @property
    def nominal_flow_rate_lpm(self) -> float:
        return self.nominal_flow_rate / LITRE_PER_MIN


@dataclass(frozen=True)
class ManifoldConfig:
    inlet_angle: float  # rad
    outlet_angle: float  # rad
    width: float  # m (arc length)
    axial_extent: float  # m


@dataclass(frozen=True)
class MaterialConfig:
    name: str
    youngs_modulus: float  # Pa
    poisson_ratio: float  # -
    density: float  # kg/m^3
    conductivity: float  # W/(m K)
    allowable_stress: float  # Pa

    @property
    def shear_modulus(self) -> float:
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))


@dataclass(frozen=True)
class LoadsConfig:
    torque: float  # N m
    thrust: float  # N
    coolant_pressure_gauge: float  # Pa


@dataclass(frozen=True)
class DesignBounds:
    relative_density: Range  # rho*
    blend_w: Range  # 0 gyroid ... 1 diamond
    cell_size: Range  # m
    axial_stretch: Range  # a_z


@dataclass(frozen=True)
class ManufacturingConfig:
    min_wall_thickness: float  # m
    min_pore_size: float  # m
    max_density_change_per_cell: float  # |grad rho*| * L  (placeholder, see YAML)


@dataclass(frozen=True)
class ReferenceConfig:
    """Root of the reference-problem configuration (all SI)."""

    motor: MotorConfig
    jacket: JacketGeometry
    heat_flux: HeatFluxConfig
    coolant: CoolantConfig
    operating: OperatingConfig
    manifolds: ManifoldConfig
    material: MaterialConfig
    loads: LoadsConfig
    design_bounds: DesignBounds
    manufacturing: ManufacturingConfig

    # ``cfg.heat_flux`` is the config *section*; the profile *function* is
    # ``cfg.q_wall(z)`` (or the module-level ``heat_flux_profile``).

    @property
    def heated_area(self) -> float:
        """Sleeve inner surface 2 pi r_s L_ax (m^2) on which q_mean is defined."""
        return 2.0 * math.pi * self.jacket.stator_radius * self.jacket.axial_length

    @property
    def mean_heat_flux(self) -> float:
        """q_mean = Q / (2 pi r_s L_ax) in W/m^2."""
        return self.motor.heat_to_jacket / self.heated_area

    def q_wall(self, z: Any) -> Any:
        """Heat flux q''(z) in W/m^2 at axial position(s) z in metres, 0 <= z <= L_ax."""
        return heat_flux_profile(
            z,
            axial_length=self.jacket.axial_length,
            mean_flux=self.mean_heat_flux,
            amplitude=self.heat_flux.amplitude,
            profile=self.heat_flux.profile,
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict (for run_record / logging)."""
        return asdict(self)

    def __post_init__(self) -> None:
        _validate(self)


# -----------------------------------------------------------------------------
# Heat-flux profile
# -----------------------------------------------------------------------------
def heat_flux_profile(
    z: Any,
    *,
    axial_length: float,
    mean_flux: float,
    amplitude: float = 0.25,
    profile: str = "end_peaked_quadratic",
) -> Any:
    """Axial heat-flux profile q''(z) (W/m^2).

    q''(z) = mean_flux * [1 + amplitude * (12 (zeta - 1/2)^2 - 1)], zeta = z / axial_length.

    Accepts scalars or arrays; returns a float for scalar input. Raises
    ``ValueError`` for z outside [0, axial_length] (beyond a 1e-12 relative tolerance).
    """
    if profile != "end_peaked_quadratic":
        raise ValueError(f"Unknown heat-flux profile {profile!r}")
    z_arr = np.asarray(z, dtype=float)
    zeta = z_arr / axial_length
    tol = 1e-12
    if np.any(zeta < -tol) or np.any(zeta > 1.0 + tol):
        raise ValueError(f"z must lie in [0, {axial_length}] m")
    q = mean_flux * (1.0 + amplitude * (12.0 * (zeta - 0.5) ** 2 - 1.0))
    return float(q) if q.ndim == 0 else q


# -----------------------------------------------------------------------------
# Building dataclasses from YAML
# -----------------------------------------------------------------------------
def _build(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    unknown = sorted(set(data) - set(names))
    missing = [n for n in names if n not in data]
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}")
    if missing:
        raise ConfigError(f"{where}: missing key(s) {missing}")
    kwargs = {n: _coerce(hints[n], data[n], f"{where}.{n}") for n in names}
    return cls(**kwargs)


def _coerce(tp: Any, value: Any, where: str) -> Any:
    if tp is Range:
        if isinstance(value, Mapping):
            return _build(Range, value, where)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return Range(_to_float(value[0], where), _to_float(value[1], where))
        raise ConfigError(f"{where}: expected [lo, hi]")
    if is_dataclass(tp):
        return _build(tp, value, where)
    if tp is float:
        return _to_float(value, where)
    if tp is int:
        if isinstance(value, bool) or int(value) != value:
            raise ConfigError(f"{where}: expected an integer")
        return int(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string")
        return value
    origin = typing.get_origin(tp)
    if origin is tuple:
        (inner, *_rest) = typing.get_args(tp)
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list")
        return tuple(_coerce(inner, v, f"{where}[{i}]") for i, v in enumerate(value))
    if isinstance(tp, types.UnionType):  # pragma: no cover - not used yet
        raise ConfigError(f"{where}: union types are not supported")
    raise ConfigError(f"{where}: unsupported field type {tp!r}")


def _to_float(value: Any, where: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{where}: expected a number, got a boolean")
    try:
        out = float(value)  # also accepts YAML strings like "70e9"
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where}: expected a number, got {value!r}") from exc
    if not math.isfinite(out):
        raise ConfigError(f"{where}: value must be finite")
    return out


def _validate(cfg: ReferenceConfig) -> None:
    def positive(name: str, v: float) -> None:
        if not v > 0:
            raise ConfigError(f"{name} must be > 0 (got {v})")

    m, j, c, o, mat = cfg.motor, cfg.jacket, cfg.coolant, cfg.operating, cfg.material
    for name, v in [
        ("motor.continuous_power", m.continuous_power),
        ("motor.heat_to_jacket", m.heat_to_jacket),
        ("jacket.stator_radius", j.stator_radius),
        ("jacket.sleeve_thickness", j.sleeve_thickness),
        ("jacket.lattice_gap", j.lattice_gap),
        ("jacket.outer_wall_thickness", j.outer_wall_thickness),
        ("jacket.axial_length", j.axial_length),
        ("coolant.density", c.density),
        ("coolant.specific_heat", c.specific_heat),
        ("coolant.conductivity", c.conductivity),
        ("coolant.kinematic_viscosity", c.kinematic_viscosity),
        ("operating.inlet_temperature", o.inlet_temperature),
        ("operating.nominal_flow_rate", o.nominal_flow_rate),
        ("material.youngs_modulus", mat.youngs_modulus),
        ("material.density", mat.density),
        ("material.conductivity", mat.conductivity),
        ("material.allowable_stress", mat.allowable_stress),
        ("manifolds.width", cfg.manifolds.width),
        ("manifolds.axial_extent", cfg.manifolds.axial_extent),
        ("manufacturing.min_wall_thickness", cfg.manufacturing.min_wall_thickness),
        ("manufacturing.min_pore_size", cfg.manufacturing.min_pore_size),
        ("manufacturing.max_density_change_per_cell", cfg.manufacturing.max_density_change_per_cell),
    ]:
        positive(name, v)

    if not 0.0 < m.efficiency <= 1.0:
        raise ConfigError("motor.efficiency must be in (0, 1]")
    if m.heat_to_jacket > m.total_loss * (1 + 1e-9):
        raise ConfigError(
            f"motor.heat_to_jacket ({m.heat_to_jacket} W) exceeds total motor loss ({m.total_loss:.1f} W)"
        )
    if not 0.0 < o.pump_efficiency <= 1.0:
        raise ConfigError("operating.pump_efficiency must be in (0, 1]")
    if not o.flow_rate_sweep or any(q <= 0 for q in o.flow_rate_sweep):
        raise ConfigError("operating.flow_rate_sweep must be a non-empty list of positive values")
    if not -1.0 < mat.poisson_ratio < 0.5:
        raise ConfigError("material.poisson_ratio must be in (-1, 0.5)")
    if cfg.heat_flux.profile != "end_peaked_quadratic":
        raise ConfigError(f"heat_flux.profile {cfg.heat_flux.profile!r} is not defined")
    if not 0.0 <= cfg.heat_flux.amplitude < 1.0:
        raise ConfigError("heat_flux.amplitude must be in [0, 1) so that q''(z) > 0")
    if cfg.manifolds.axial_extent > j.axial_length * (1 + 1e-9):
        raise ConfigError("manifolds.axial_extent exceeds the jacket axial length")
    circumference = 2 * math.pi * j.lattice_mean_radius
    if 2 * cfg.manifolds.width >= circumference:
        raise ConfigError("manifold slots cover the whole circumference")

    b = cfg.design_bounds
    if not (0.0 < b.relative_density.lo and b.relative_density.hi < 1.0):
        raise ConfigError("design_bounds.relative_density must lie inside (0, 1)")
    if not (0.0 <= b.blend_w.lo and b.blend_w.hi <= 1.0):
        raise ConfigError("design_bounds.blend_w must lie inside [0, 1]")
    if not (b.cell_size.lo > 0 and b.axial_stretch.lo > 0):
        raise ConfigError("design_bounds.cell_size and axial_stretch must be positive")
    if b.cell_size.lo > j.lattice_gap:
        raise ConfigError("smallest cell size is larger than the lattice gap")


# -----------------------------------------------------------------------------
# Public loader
# -----------------------------------------------------------------------------
def default_config_path() -> Path:
    """``$VOXLAT_CONFIG`` if set, otherwise ``<repo>/configs/reference.yaml``."""
    env = os.environ.get("VOXLAT_CONFIG")
    return Path(env).expanduser().resolve() if env else configs_dir() / "reference.yaml"


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in overrides.items():
        if isinstance(val, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(dict(out[key]), val)
        else:
            out[key] = val
    return out


def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ReferenceConfig:
    """Load the reference problem as a frozen, typed :class:`ReferenceConfig`.

    Parameters
    ----------
    path:
        YAML file. Defaults to :func:`default_config_path`.
    overrides:
        Optional nested mapping merged into the YAML before validation, e.g.
        ``{"operating": {"nominal_flow_rate": 1e-4}}``. Use this for parameter
        sweeps instead of editing the file.
    """
    p = Path(path) if path is not None else default_config_path()
    if not p.is_file():
        raise FileNotFoundError(
            f"Config not found at {p}. Set VOXLAT_CONFIG or pass the path explicitly."
        )
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{p}: top level must be a mapping")
    if overrides:
        raw = _deep_merge(dict(raw), overrides)
    return _build(ReferenceConfig, raw, "config")
