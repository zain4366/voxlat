"""Closure dataset (Task 5): every homogenized property of the TPMS morphology space.

For each morphology theta = (w, rho*, a_z) this module computes, with the
production estimators chosen in Tasks 1-4 (STATUS.md):

* geometry metrics (Task 1, ``compute_metrics`` on the fine grid): realized
  density, specific surface area a_sf L, wall / pore local-thickness statistics,
  throats, periodic connectivity;
* effective conductivity k_eff / k_s (Task 2, ``extrapolated_conductivity_tpms``,
  two-grid Richardson R(n1, n2)), full 3x3 tensor;
* effective stiffness C_eff / E_s (Task 3, ``extrapolated_elasticity_tpms``,
  R(n1, n2)), full 6x6 Voigt tensor + engineering constants, and von Mises stress
  localization factors (fine grid) for all six unit macroscopic stresses;
* permeability K / L^2 (Task 4, ``extrapolated_permeability_tpms``, R(n1, n2)),
  full 3x3 tensor, porosity, tortuosity, Kozeny constant.

All closures are dimensionless (k / k_s, C / E_s, K / L^2, lengths / L); the
material constants used are stored in every row. Task 6 rescales with L.

Design of experiments
---------------------
* Morphology box from ``configs/reference.yaml`` (``design_bounds``): w in [0, 1],
  rho* in [0.20, 0.50], a_z in [0.7, 1.5].
* ``n_sobol`` scrambled Sobol points (scipy ``qmc.Sobol``, seeded). Sobol point i
  has the stable id ``sobol-iiii``: the first 8 points of an N = 256 design are
  the 8 points of an N = 8 design, so a small trial run is reused by the full run.
* Boundary points: the 8 corners, ``edge_points`` points on each of the 12 box
  edges, and "pure lines" w in {0, 1}, a_z = 1, rho* = linspace(lo, hi, k) - the
  axes of baselines B2/B3 and the points of the Task 2-4 tables (regression check).

**Stretch quantization (why a_z is discrete).** Cells are voxelized with cubic
voxels, n_z = round(n a_z) (Task 1, decision 3), so a continuous a_z is snapped
differently on the two Richardson grids (n1 = 32: a_z = k/32, n2 = 64: k/64) and
the extrapolation would mix two geometries (error up to ~2 % in k_zz, C33, K_zz -
the size of the discretization error being removed). The dataset therefore uses
only stretches representable *exactly* on every grid: a_z = l q with
q = 1 / gcd(n1, n2, n_metrics) (1/32 for the default grids). The Sobol coordinate
is mapped onto the levels l = floor(lo / q) ... ceil(hi / q) by equal-probability
strata (22/32 = 0.6875 ... 48/32 = 1.5 by default, 27 levels), i.e. the levels
bracket the design box so the surrogates never extrapolate in a_z.

Storage and resumability
------------------------
Each finished sample is written atomically as one JSON file
``<parts_dir>/<id>.json`` by the worker that computed it (safe under parallel
runs and Ctrl+C). ``consolidate`` merges the parts (and any rows of an existing
table) into ``data/closures.parquet``. A rerun skips every id whose record
matches the current design point and the settings fingerprint; a mismatch raises
(``force=True`` recomputes it). Rows whose status is not "ok" are retried only
with ``retry_errors=True``.

Every row carries a ``run_record`` (``run_record_json``) plus flat provenance
columns (version, git hash, timestamp, wall times, solver iterations).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from voxlat.utils.log import get_logger

__all__ = [
    "DATASET_SCHEMA_VERSION",
    "DEFAULT_DESIGN_SEED",
    "DesignSpace",
    "DatasetSettings",
    "Sample",
    "stretch_quantum",
    "stretch_levels",
    "sobol_design",
    "boundary_design",
    "build_design",
    "compute_sample",
    "relative_cost",
    "RuntimeEstimate",
    "estimate_runtime",
    "pick_timing_samples",
    "write_part",
    "read_parts",
    "load_existing",
    "pending_samples",
    "run_samples",
    "records_to_frame",
    "write_table",
    "read_table",
    "parquet_available",
    "consolidate",
    "plot_quicklook",
    "SettingsMismatchError",
]

LOG = get_logger("closures.dataset")

#: bump when the meaning of a stored column changes (part of the settings fingerprint)
DATASET_SCHEMA_VERSION = 1
#: seed of the scrambled Sobol design (fixed: the design is part of the dataset's identity)
DEFAULT_DESIGN_SEED = 20261008

_VOIGT = ("xx", "yy", "zz", "yz", "xz", "xy")
_TENSOR2 = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2), "xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
_FAMILY_ORDER = {"pure_line": 0, "corner": 1, "edge": 2, "sobol": 3}


class SettingsMismatchError(RuntimeError):
    """An existing record was computed for a different design point or with different settings."""


# =============================================================================
# Design space and settings
# =============================================================================
@dataclass(frozen=True)
class DesignSpace:
    """Morphology box theta = (w, rho*, a_z); defaults = reference config ``design_bounds``."""

    w: tuple[float, float] = (0.0, 1.0)
    rho: tuple[float, float] = (0.20, 0.50)
    a_z: tuple[float, float] = (0.7, 1.5)

    @classmethod
    def from_config(cls, cfg: Any = None) -> "DesignSpace":
        if cfg is None:
            from voxlat.utils.config import load_config

            cfg = load_config()
        b = cfg.design_bounds
        return cls(
            w=(float(b.blend_w.lo), float(b.blend_w.hi)),
            rho=(float(b.relative_density.lo), float(b.relative_density.hi)),
            a_z=(float(b.axial_stretch.lo), float(b.axial_stretch.hi)),
        )


@dataclass(frozen=True)
class DatasetSettings:
    """Resolutions, material constants and solver settings of one dataset.

    ``n_coarse``/``n_fine`` are the Richardson grid pair (voxels per in-plane cell
    length L; recommended (32, 64) by Tasks 2-4), ``n_metrics`` the grid of the
    geometry metrics (default = ``n_fine``, i.e. the same voxels the fine solvers
    see). ``k_s``, ``k_f``, ``E_s``, ``nu_s`` default to the reference config
    (``resolved``). ``preconditioner`` and ``save_fields`` do not change results
    (AMG and Jacobi agree to ~1e-6) and are not part of the fingerprint.
    """

    n_coarse: int = 32
    n_fine: int = 64
    n_metrics: int | None = None
    k_s: float | None = None
    k_f: float | None = None
    E_s: float | None = None
    nu_s: float | None = None
    stress_cases: tuple[str, ...] = (
        "uniaxial_x", "uniaxial_y", "uniaxial_z", "shear_yz", "shear_xz", "shear_xy",
    )
    tol_conduction: float = 1e-8
    tol_elasticity: float = 1e-8
    tol_stokes: float = 1e-8
    richardson_p: float = 1.0
    kind: str = "network"
    preconditioner: str = "auto"
    save_fields: bool = False

    def __post_init__(self) -> None:
        if not 4 <= int(self.n_coarse) < int(self.n_fine):
            raise ValueError(f"need 4 <= n_coarse < n_fine, got ({self.n_coarse}, {self.n_fine})")
        if self.n_metrics is not None and int(self.n_metrics) < 4:
            raise ValueError("n_metrics must be >= 4")
        object.__setattr__(self, "stress_cases", tuple(self.stress_cases))

    @property
    def metrics_n(self) -> int:
        return int(self.n_fine if self.n_metrics is None else self.n_metrics)

    @property
    def grids(self) -> tuple[int, int]:
        return int(self.n_coarse), int(self.n_fine)

    @property
    def quantum(self) -> float:
        """Stretch quantum q: every a_z = l q is represented exactly on all grids."""
        return stretch_quantum(self.n_coarse, self.n_fine, self.metrics_n)

    def resolved(self, cfg: Any = None) -> "DatasetSettings":
        """Copy with material constants filled from the reference config."""
        if None not in (self.k_s, self.k_f, self.E_s, self.nu_s):
            return self
        if cfg is None:
            from voxlat.utils.config import load_config

            cfg = load_config()
        return replace(
            self,
            k_s=float(cfg.material.conductivity) if self.k_s is None else float(self.k_s),
            k_f=float(cfg.coolant.conductivity) if self.k_f is None else float(self.k_f),
            E_s=float(cfg.material.youngs_modulus) if self.E_s is None else float(self.E_s),
            nu_s=float(cfg.material.poisson_ratio) if self.nu_s is None else float(self.nu_s),
        )

    def fingerprint_dict(self) -> dict[str, Any]:
        s = self.resolved()
        return {
            "schema": DATASET_SCHEMA_VERSION, "n_coarse": int(s.n_coarse), "n_fine": int(s.n_fine),
            "n_metrics": s.metrics_n, "k_s": s.k_s, "k_f": s.k_f, "E_s": s.E_s, "nu_s": s.nu_s,
            "stress_cases": list(s.stress_cases), "tol": [s.tol_conduction, s.tol_elasticity, s.tol_stokes],
            "richardson_p": float(s.richardson_p), "kind": s.kind,
        }

    def fingerprint(self) -> str:
        """Short hash of every setting that changes the numbers (not preconditioner/fields)."""
        blob = json.dumps(self.fingerprint_dict(), sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self.resolved())
        d["stress_cases"] = list(d["stress_cases"])
        d["fingerprint"] = self.fingerprint()
        return d


def stretch_quantum(*grids: int) -> float:
    """Smallest a_z step representable exactly on all grids: 1 / gcd(n_1, n_2, ...)."""
    if not grids:
        raise ValueError("need at least one grid")
    return 1.0 / math.gcd(*(int(n) for n in grids))


def stretch_levels(a_range: tuple[float, float], quantum: float) -> np.ndarray:
    """Integer levels l with a_z = l q bracketing [lo, hi]: floor(lo / q) ... ceil(hi / q)."""
    lo, hi = a_range
    l0 = int(math.floor(lo / quantum + 1e-9))
    l1 = int(math.ceil(hi / quantum - 1e-9))
    return np.arange(l0, l1 + 1)


# =============================================================================
# Samples and the design
# =============================================================================
@dataclass(frozen=True)
class Sample:
    """One design point. ``a_z = a_z_level * quantum`` is exactly representable on every grid."""

    id: str
    family: str  # "sobol" | "corner" | "edge" | "pure_line"
    index: int
    w: float
    rho: float
    a_z_level: int
    quantum: float

    @property
    def a_z(self) -> float:
        return self.a_z_level * self.quantum

    @property
    def is_blend(self) -> bool:
        return 0.0 < self.w < 1.0

    def params(self, kind: str = "network") -> Any:
        from voxlat.geometry.tpms import TPMSParams

        return TPMSParams.from_design(self.w, self.rho, self.a_z, kind=kind)  # type: ignore[arg-type]

    def theta(self) -> dict[str, float]:
        return {"w": self.w, "rho_target": self.rho, "a_z": self.a_z}

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "family": self.family, "index": self.index, "w": self.w,
                "rho_target": self.rho, "a_z": self.a_z, "a_z_level": self.a_z_level,
                "a_z_quantum": self.quantum}


def _snap_level(a: float, quantum: float) -> int:
    return int(round(a / quantum))


def _sobol_points(n: int, seed: int, d: int = 3) -> np.ndarray:
    """First n points of the seeded, scrambled Sobol sequence in [0, 1)^d."""
    from scipy.stats import qmc

    try:
        sampler = qmc.Sobol(d=d, scramble=True, rng=np.random.default_rng(seed))
    except TypeError:  # scipy < 1.15
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    if n <= 0:
        return np.empty((0, d))
    m = int(round(math.log2(n)))
    if 2**m == n:
        return sampler.random_base2(m)
    import warnings

    with warnings.catch_warnings():  # "balance properties ... power of 2": fine for a prefix
        warnings.simplefilter("ignore", UserWarning)
        return sampler.random(n)


def sobol_design(
    n: int, space: DesignSpace | None = None, *, seed: int = DEFAULT_DESIGN_SEED, quantum: float = 1 / 32
) -> list[Sample]:
    """``n`` scrambled-Sobol samples; ids ``sobol-0000`` ... are stable prefixes of the sequence.

    u0 -> w, u1 -> rho* (uniform), u2 -> a_z level by equal-probability strata over
    ``stretch_levels(space.a_z, quantum)``.
    """
    space = space or DesignSpace()
    U = _sobol_points(int(n), seed)
    levels = stretch_levels(space.a_z, quantum)
    out = []
    for i, u in enumerate(U):
        w = space.w[0] + u[0] * (space.w[1] - space.w[0])
        rho = space.rho[0] + u[1] * (space.rho[1] - space.rho[0])
        lvl = int(levels[min(int(u[2] * len(levels)), len(levels) - 1)])
        out.append(Sample(f"sobol-{i:04d}", "sobol", i, float(w), float(rho), lvl, quantum))
    return out


def boundary_design(
    space: DesignSpace | None = None,
    *,
    quantum: float = 1 / 32,
    edge_points: int = 1,
    pure_line_points: int = 7,
    pure_line_w: Sequence[float] = (0.0, 1.0),
) -> list[Sample]:
    """Corners (8), ``edge_points`` interior points per box edge (12 edges) and pure lines.

    The a_z bounds are the outermost representable levels (``stretch_levels``), so
    the corners bracket the box. Pure lines: w in ``pure_line_w``, a_z = 1,
    rho* = linspace(lo, hi, ``pure_line_points``) (skipped if a_z = 1 is not a level).
    """
    space = space or DesignSpace()
    lv = stretch_levels(space.a_z, quantum)
    lo_l, hi_l = int(lv[0]), int(lv[-1])
    out: list[Sample] = []
    corners = [(w, r, l) for w in space.w for r in space.rho for l in (lo_l, hi_l)]
    for i, (w, r, l) in enumerate(corners):
        out.append(Sample(f"corner-{i}", "corner", i, float(w), float(r), int(l), quantum))
    if edge_points > 0:
        ts = [(k + 1) / (edge_points + 1) for k in range(edge_points)]
        k = 0
        lerp = lambda rng, t: rng[0] + t * (rng[1] - rng[0])  # noqa: E731
        for t in ts:
            # 4 edges along w, 4 along rho, 4 along a_z
            for r in space.rho:
                for l in (lo_l, hi_l):
                    out.append(Sample(f"edge-{k:02d}", "edge", k, float(lerp(space.w, t)), float(r), int(l), quantum))
                    k += 1
            for w in space.w:
                for l in (lo_l, hi_l):
                    out.append(Sample(f"edge-{k:02d}", "edge", k, float(w), float(lerp(space.rho, t)), int(l), quantum))
                    k += 1
            for w in space.w:
                for r in space.rho:
                    lvl = _snap_level(lerp((lo_l * quantum, hi_l * quantum), t), quantum)
                    out.append(Sample(f"edge-{k:02d}", "edge", k, float(w), float(r), lvl, quantum))
                    k += 1
    one = 1.0 / quantum
    if pure_line_points > 0 and abs(one - round(one)) < 1e-9:
        rhos = np.linspace(space.rho[0], space.rho[1], int(pure_line_points))
        k = 0
        for w in pure_line_w:
            for r in rhos:
                out.append(Sample(f"pure-{k:02d}", "pure_line", k, float(w), float(round(r, 12)),
                                  int(round(one)), quantum))
                k += 1
    return out


def build_design(
    n_sobol: int = 256,
    *,
    space: DesignSpace | None = None,
    settings: DatasetSettings | None = None,
    seed: int = DEFAULT_DESIGN_SEED,
    boundary: bool = True,
    edge_points: int = 1,
    pure_line_points: int = 7,
) -> list[Sample]:
    """Full design: Sobol points + (optionally) corners, edges and pure lines; duplicates dropped."""
    space = space or DesignSpace.from_config()
    q = (settings or DatasetSettings()).quantum
    samples = sobol_design(n_sobol, space, seed=seed, quantum=q)
    if boundary:
        samples = boundary_design(space, quantum=q, edge_points=edge_points,
                                  pure_line_points=pure_line_points) + samples
    seen: set[tuple[float, float, int]] = set()
    out = []
    for s in samples:
        key = (round(s.w, 12), round(s.rho, 12), s.a_z_level)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


# =============================================================================
# One sample
# =============================================================================
def _sym_tensor_cols(T: np.ndarray, prefix: str) -> dict[str, float]:
    return {f"{prefix}_{k}": float(T[i, j]) for k, (i, j) in _TENSOR2.items()}


def _reset_peak_rss() -> None:
    """Linux >= 4.0: reset this process's VmHWM so the next reading is a per-sample peak."""
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
    except OSError:
        pass


def _peak_rss_mb() -> float:
    """Peak resident memory in MB.

    Linux: VmHWM of /proc/self/status (reset per sample by ``_reset_peak_rss``, so this is
    the peak of one sample). Windows: PeakWorkingSetSize (peak over the worker process's
    lifetime, i.e. its largest sample so far). Other: getrusage maximum.
    """
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(r) / (2**20 if sys.platform == "darwin" else 1024.0)
    except ImportError:
        pass
    if sys.platform == "win32":  # pragma: no cover - Windows only
        try:
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi  # type: ignore[attr-defined]
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return float(pmc.PeakWorkingSetSize) / 2**20
        except Exception:
            pass
    return float("nan")


def _geometry_block(params: Any, s: DatasetSettings) -> dict[str, Any]:
    from voxlat.geometry.tpms import compute_metrics

    t0 = time.perf_counter()
    m = compute_metrics(params, n=s.metrics_n)
    row: dict[str, Any] = {
        "rho_voxel": m.relative_density, "threshold_c": m.threshold,
        "a_sf_L": m.a_sf_L, "a_sf_voxel_L": m.a_sf_voxel_L,
        "solid_components": int(m.solid_components), "fluid_components": int(m.fluid_components),
        "solid_connected": bool(m.solid_connected), "fluid_connected": bool(m.fluid_connected),
        "solid_largest_fraction": float(m.solid_largest_fraction),
        "fluid_largest_fraction": float(m.fluid_largest_fraction),
    }
    for name, st in (("wall", m.wall), ("pore", m.pore)):
        for k, v in asdict(st).items():
            row[f"{name}_{k}_L"] = float(v)
    row["geo_time_s"] = time.perf_counter() - t0
    return row


def _conduction_block(params: Any, s: DatasetSettings) -> dict[str, Any]:
    from voxlat.homogenization.conduction import extrapolated_conductivity_tpms

    r = extrapolated_conductivity_tpms(params, s.grids, s.k_s, s.k_f, tol=s.tol_conduction,
                                       preconditioner=s.preconditioner)
    k = r.k_eff / s.k_s
    ev = np.linalg.eigvalsh(k)
    row: dict[str, Any] = _sym_tensor_cols(k, "keff")
    row.update({
        "keff_eig1": float(ev[0]), "keff_eig2": float(ev[1]), "keff_eig3": float(ev[2]),
        "keff_mean": float(np.trace(k) / 3.0),
        "keff_anisotropy": float((ev[-1] - ev[0]) / ev.mean()) if ev.mean() > 0 else float("nan"),
        "keff_coarse_mean": float(np.trace(r.coarse.k_eff) / 3.0 / s.k_s),
        "keff_fine_mean": float(np.trace(r.fine.k_eff) / 3.0 / s.k_s),
        "keff_correction": r.correction,
        "keff_agreement": float(max(r.coarse.agreement, r.fine.agreement)),
        "keff_iterations": int(max(r.fine.iterations) if r.fine.iterations else 0),
        "keff_preconditioner": r.fine.preconditioner,
        "keff_time_s": r.wall_time,
    })
    return row


def _elasticity_block(params: Any, s: DatasetSettings) -> tuple[dict[str, Any], Any]:
    from voxlat.homogenization.elasticity import (
        cubic_deviation,
        extrapolated_elasticity_tpms,
        voigt_reuss_hill,
        voigt_to_mandel,
        youngs_extremes,
    )

    r = extrapolated_elasticity_tpms(
        params, s.grids, s.E_s, s.nu_s, p=s.richardson_p, tol=s.tol_elasticity,
        preconditioner=s.preconditioner, localization=s.stress_cases,
        keep_localization_fields=s.save_fields,
    )
    C = r.C_eff / s.E_s
    S = np.linalg.inv(C)
    row: dict[str, Any] = {}
    for i in range(6):
        for j in range(i, 6):
            row[f"C{i + 1}{j + 1}"] = float(C[i, j])
    E = 1.0 / np.diag(S)[:3]
    G = 1.0 / np.diag(S)[3:]
    vrh = voigt_reuss_hill(C)
    emin, emax = youngs_extremes(C)
    d = np.ones(3) / np.sqrt(3.0)
    v111 = np.array([d[0] ** 2, d[1] ** 2, d[2] ** 2, d[1] * d[2], d[0] * d[2], d[0] * d[1]])
    c11 = float(np.mean(np.diag(C)[:3]))
    c12 = float(np.mean([C[0, 1], C[0, 2], C[1, 2]]))
    c44 = float(np.mean(np.diag(C)[3:]))
    ev = np.linalg.eigvalsh(voigt_to_mandel(C))
    row.update({
        "E_x": float(E[0]), "E_y": float(E[1]), "E_z": float(E[2]), "E_axial": float(E.mean()),
        "G_yz": float(G[0]), "G_xz": float(G[1]), "G_xy": float(G[2]),
        "nu_xy": float(-S[0, 1] / S[0, 0]), "nu_xz": float(-S[0, 2] / S[0, 0]),
        "nu_zx": float(-S[2, 0] / S[2, 2]),
        "E_111": float(1.0 / (v111 @ S @ v111)), "E_min": emin, "E_max": emax,
        "K_bulk": float(1.0 / np.sum(S[:3, :3])),
        "E_hill": float(vrh["E_H"]), "K_hill": float(vrh["K_H"]), "G_hill": float(vrh["G_H"]),
        "A_U": float(vrh["A_U"]), "zener": float(2.0 * c44 / (c11 - c12)), "cubic_dev": cubic_deviation(C),
        "C_eig_min": float(ev[0]), "C_eig_max": float(ev[-1]),
        "E_axial_coarse": float(np.mean(1.0 / np.diag(np.linalg.inv(r.coarse.C_eff / s.E_s))[:3])),
        "E_axial_fine": float(np.mean(1.0 / np.diag(np.linalg.inv(r.fine.C_eff / s.E_s))[:3])),
        "C_correction": r.correction,
        "C_agreement": float(max(r.coarse.agreement, r.fine.agreement)),
        "C_iterations": int(max(r.fine.iterations)),
        "C_n_components": int(r.fine.n_components),
        "C_preconditioner": r.fine.preconditioner,
        "C_time_s": r.wall_time,
    })
    for name, loc in r.fine.localization.items():
        row[f"loc_{name}_max"] = loc.max
        row[f"loc_{name}_p99"] = loc.p99
        row[f"loc_{name}_p999"] = loc.p999
        row[f"loc_{name}_mean"] = loc.mean
    return row, r


def _permeability_block(params: Any, s: DatasetSettings) -> tuple[dict[str, Any], Any]:
    from voxlat.homogenization.stokes import extrapolated_permeability_tpms, tpms_symmetry

    r = extrapolated_permeability_tpms(params, s.grids, p=s.richardson_p, tol=s.tol_stokes,
                                       preconditioner=s.preconditioner, return_fields=s.save_fields)
    K = r.K
    ev = np.linalg.eigvalsh(K)
    sym = tpms_symmetry(params)
    tort = dict(r.fine.tortuosity)
    if sym == "cubic":
        tort = {0: tort[0], 1: tort[0], 2: tort[0]}
    elif sym == "tetragonal_z":
        tort = {0: tort[0], 1: tort[0], 2: tort[2]}
    row: dict[str, Any] = _sym_tensor_cols(K, "K")
    tv = [tort.get(d, float("nan")) for d in range(3)]
    row.update({
        "K_eig1": float(ev[0]), "K_eig2": float(ev[1]), "K_eig3": float(ev[2]),
        "K_mean": float(np.trace(K) / 3.0),
        "K_anisotropy": float((ev[-1] - ev[0]) / ev.mean()) if ev.mean() > 0 else float("nan"),
        "porosity": float(r.fine.porosity),
        "tortuosity_x": float(tv[0]), "tortuosity_y": float(tv[1]), "tortuosity_z": float(tv[2]),
        "tortuosity": float(np.nanmean(tv)) if np.isfinite(tv).any() else float("nan"),
        "K_coarse_mean": float(np.trace(r.coarse.K) / 3.0),
        "K_fine_mean": float(np.trace(r.fine.K) / 3.0),
        "K_correction": r.correction,
        "K_agreement": float(max(r.coarse.agreement, r.fine.agreement)),
        "K_iterations": int(max(r.fine.iterations) if r.fine.iterations else 0),
        "K_symmetry": sym, "K_n_solves": len(r.fine.directions),
        "K_preconditioner": r.fine.preconditioner,
        "K_time_s": r.wall_time,
    })
    return row, r


def _save_fields(path: Path, params: Any, s: DatasetSettings, elas: Any, perm: Any) -> None:
    """Fine-grid fields of one sample (``np.savez_compressed``): solid mask (packed bits),
    cell-centred velocity per solved direction (float32, units L^2 |f|/mu) and von Mises
    localization factor per stress case (float32, NaN in the fluid)."""
    from voxlat.geometry.tpms import voxelize

    solid = voxelize(params, s.n_fine)
    out: dict[str, Any] = {"shape": np.array(solid.shape), "solid_packed": np.packbits(solid.ravel()),
                           "n": np.int64(s.n_fine)}
    if perm is not None and perm.fine.velocity is not None:
        out["velocity_directions"] = np.array(perm.fine.directions)
        out["velocity"] = np.stack([perm.fine.cell_velocity(k).astype(np.float32)
                                    for k in range(len(perm.fine.directions))])
    if elas is not None:
        f = elas.fine
        vox = f.element_voxel[f.element_solid]
        for name, loc in f.localization.items():
            if loc.factor is None:
                continue
            dense = np.full(solid.size, np.nan, dtype=np.float32)
            dense[vox] = loc.factor
            out[f"loc_{name}"] = dense.reshape(solid.shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)


def compute_sample(
    sample: Sample,
    settings: DatasetSettings | None = None,
    *,
    fields_dir: str | Path | None = None,
    blocks: Sequence[str] = ("geometry", "conduction", "elasticity", "permeability"),
) -> dict[str, Any]:
    """All closures of one design point as one flat record (dict of JSON scalars).

    Each block (geometry, conduction, elasticity, permeability) runs inside its own
    try/except: a failure leaves that block's columns missing (NaN in the table) and
    is reported in ``status`` ("ok", "partial", "error") and ``error``.
    """
    from voxlat.utils.provenance import run_record, to_jsonable

    s = (settings or DatasetSettings()).resolved()
    _reset_peak_rss()
    t0 = time.perf_counter()
    params = sample.params(s.kind)
    rec: dict[str, Any] = sample.to_dict()
    rec.update({"n_coarse": s.n_coarse, "n_fine": s.n_fine, "n_metrics": s.metrics_n,
                "k_s": s.k_s, "k_f": s.k_f, "kf_ks": s.k_f / s.k_s, "E_s": s.E_s, "nu_s": s.nu_s})
    errors: dict[str, str] = {}
    elas = perm = None
    for name in blocks:
        try:
            if name == "geometry":
                rec.update(_geometry_block(params, s))
            elif name == "conduction":
                rec.update(_conduction_block(params, s))
            elif name == "elasticity":
                row, elas = _elasticity_block(params, s)
                rec.update(row)
            elif name == "permeability":
                row, perm = _permeability_block(params, s)
                rec.update(row)
            else:
                raise ValueError(f"unknown block {name!r}")
        except Exception as exc:  # noqa: BLE001 - recorded, the run continues
            errors[name] = f"{type(exc).__name__}: {exc}"
            LOG.warning("%s: %s block failed: %s\n%s", sample.id, name, errors[name], traceback.format_exc())
    if "a_sf_L" in rec and "K_mean" in rec and "porosity" in rec and rec["K_mean"] > 0:
        rec["kozeny_c"] = float(rec["porosity"] ** 3 / (rec["K_mean"] * rec["a_sf_L"] ** 2))
    if s.save_fields and fields_dir is not None:
        try:
            _save_fields(Path(fields_dir) / f"{sample.id}.npz", params, s, elas, perm)
        except Exception as exc:  # noqa: BLE001
            errors["fields"] = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t0
    n_ok = len([b for b in blocks if b not in errors])
    rec["status"] = "ok" if not errors else ("error" if n_ok == 0 else "partial")
    rec["error"] = json.dumps(errors) if errors else ""
    rec["wall_time_s"] = wall
    rec["peak_rss_mb"] = _peak_rss_mb()
    rec["settings_hash"] = s.fingerprint()
    rr = run_record(
        {**sample.to_dict(), "settings": s.to_dict()}, resolution=s.grids, wall_time=wall,
        blocks=list(blocks), errors=errors, pid=os.getpid(),
    )
    for k in ("voxlat_version", "git_hash", "timestamp_utc", "python", "numpy", "platform"):
        rec[f"rr_{k}"] = rr[k]
    rec["run_record_json"] = json.dumps(rr, sort_keys=True)
    return to_jsonable(rec)


# =============================================================================
# Cost model and runtime estimate
# =============================================================================
#: relative cost of a blend (0 < w < 1) vs pure G/D at equal grid (STATUS Tasks 2-4:
#: ~205 s vs ~85 s per sample; blends need 3 Stokes solves and ~2x PCG iterations)
BLEND_COST_FACTOR = 2.4
#: cost ~ a_z^p (DOFs ~ a_z, Jacobi iterations grow with the longest cell dimension)
STRETCH_COST_EXPONENT = 1.3


def relative_cost(sample: Sample) -> float:
    """Relative CPU cost of a sample (1 = pure gyroid/diamond, a_z = 1)."""
    return (BLEND_COST_FACTOR if sample.is_blend else 1.0) * sample.a_z**STRETCH_COST_EXPONENT


@dataclass
class RuntimeEstimate:
    """Projected cost of the pending samples from timed ones (cost-model scaled)."""

    n_pending: int
    n_jobs: int
    seconds_per_unit: float  # measured seconds per unit of relative_cost (one core, serial)
    serial_s: float  # all pending samples on one core
    wall_s_low: float
    wall_s_high: float
    longest_sample_s: float
    timed: list[tuple[str, float, float]] = field(default_factory=list)  # (id, cost, seconds)
    peak_rss_mb: float = float("nan")

    def wall_s_for(self, n_jobs: int, slowdown: float) -> float:
        """Wall time on ``n_jobs`` workers that each run ``slowdown`` x slower than alone.

        Perfect sharing plus half a longest sample of tail imbalance (the queue is
        ordered longest-first, so the tail is short).
        """
        n = max(1, int(n_jobs))
        if n == 1:
            return self.serial_s
        longest = self.longest_sample_s * slowdown
        return max(self.serial_s * slowdown / n + 0.5 * longest, longest)

    def summary(self, job_counts: Sequence[int] = (1, 2, 4, 6, 8)) -> str:
        h = lambda x: f"{x / 3600:5.1f} h" if x >= 3600 else f"{x / 60:5.1f} min"  # noqa: E731
        lines = [f"Timed samples (one core, serial): " + ", ".join(
            f"{i} {t:.0f} s (cost {c:.2f})" for i, c, t in self.timed),
            f"-> {self.seconds_per_unit:.0f} s per cost unit (pure G/D, a_z = 1 ~ 1 unit; blends ~{BLEND_COST_FACTOR:g})",
            f"Pending: {self.n_pending} samples = {self.serial_s / 3600:.1f} CPU-hours",
            "Estimated wall time (parallel workers slow each other ~1.2-1.6x via memory bandwidth):"]
        for n in job_counts:
            lo, hi = self.wall_s_for(n, 1.2), self.wall_s_for(n, 1.6)
            mark = "  <- --n-jobs" if n == self.n_jobs else ""
            lines.append(f"   {n:2d} worker(s): {h(lo)} ... {h(hi)}{mark}")
        if self.n_jobs not in job_counts:
            lines.append(f"   {self.n_jobs:2d} worker(s): {h(self.wall_s_low)} ... {h(self.wall_s_high)}  <- --n-jobs")
        if math.isfinite(self.peak_rss_mb):
            lines.append(f"Peak memory of one worker: {self.peak_rss_mb / 1024:.1f} GB (the timed set includes "
                         f"the most expensive cell); keep n_jobs x this below ~70 % of your RAM.")
        return "\n".join(lines)


def estimate_runtime(
    timed: Sequence[tuple[Sample, float]],
    pending: Sequence[Sample],
    n_jobs: int = 1,
    *,
    peak_rss_mb: float = float("nan"),
) -> RuntimeEstimate:
    """Scale the cost model by measured times: serial = sum(cost) * sum(t) / sum(cost_timed)."""
    if not timed:
        raise ValueError("need at least one timed sample")
    c_t = sum(relative_cost(s) for s, _ in timed)
    t_t = sum(t for _, t in timed)
    spu = t_t / c_t
    costs = [relative_cost(s) for s in pending]
    serial = spu * float(sum(costs))
    longest = spu * max(costs) if costs else 0.0
    est = RuntimeEstimate(len(pending), int(n_jobs), spu, serial, 0.0, 0.0, longest,
                          [(s.id, relative_cost(s), t) for s, t in timed], peak_rss_mb)
    est.wall_s_low = est.wall_s_for(n_jobs, 1.2)
    est.wall_s_high = est.wall_s_for(n_jobs, 1.6)
    return est


def pick_timing_samples(pending: Sequence[Sample], k: int = 3) -> list[Sample]:
    """k representative samples: cost-model quantiles (cheapest, median, most expensive, ...)."""
    if not pending or k <= 0:
        return []
    order = sorted(pending, key=lambda s: (relative_cost(s), s.id))
    if k >= len(order):
        return list(order)
    idx = sorted({int(round(q * (len(order) - 1))) for q in np.linspace(0.0, 1.0, k)})
    return [order[i] for i in idx]


# =============================================================================
# Storage: parts, resumability, consolidated table
# =============================================================================
def _part_path(parts_dir: str | Path, sample_id: str) -> Path:
    return Path(parts_dir) / f"{sample_id}.json"


def write_part(record: dict[str, Any], parts_dir: str | Path) -> Path:
    """Write one record atomically to ``<parts_dir>/<id>.json``."""
    path = _part_path(parts_dir, record["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, sort_keys=False, allow_nan=True)
    os.replace(tmp, path)
    return path


def read_parts(parts_dir: str | Path) -> list[dict[str, Any]]:
    """All records in a parts directory (unreadable/partial files are skipped with a warning)."""
    d = Path(parts_dir)
    out = []
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            with open(p, encoding="utf-8") as fh:
                out.append(json.load(fh))
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning("skipping unreadable part %s (%s)", p.name, exc)
    return out


def parquet_available() -> bool:
    """True if pandas can read/write parquet (pyarrow or fastparquet installed)."""
    for mod in ("pyarrow", "fastparquet"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def read_table(path: str | Path) -> Any:
    """Read a closure table (.parquet, or .csv for the no-parquet fallback)."""
    import pandas as pd

    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype={"id": str, "settings_hash": str, "error": str},
                           float_precision="round_trip").fillna({"error": ""})
    return pd.read_parquet(path)


def write_table(df: Any, path: str | Path) -> Path:
    """Write the table atomically. ``.parquet`` needs pyarrow (core dependency); ``.csv`` always works."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{os.getpid()}{path.suffix}")
    if path.suffix == ".csv":
        df.to_csv(tmp, index=False, float_format="%.17g")  # lossless round trip
    else:
        if not parquet_available():
            raise ImportError("writing parquet needs pyarrow (pip install pyarrow); "
                              "or write a .csv table instead")
        df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def load_existing(parts_dir: str | Path | None, table_path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Existing records by id: rows of the table, overridden by newer parts."""
    recs: dict[str, dict[str, Any]] = {}
    if table_path is not None:
        tp = Path(table_path)
        candidates = [tp] if tp.exists() else []
        if not candidates and tp.suffix == ".parquet" and tp.with_suffix(".csv").exists():
            candidates = [tp.with_suffix(".csv")]
        for c in candidates:
            try:
                df = read_table(c)
                for row in df.to_dict(orient="records"):
                    recs[str(row["id"])] = row
            except Exception as exc:  # noqa: BLE001
                LOG.warning("could not read existing table %s (%s); relying on parts", c, exc)
    if parts_dir is not None:
        for r in read_parts(parts_dir):
            recs[str(r["id"])] = r
    return recs


def _matches(rec: dict[str, Any], sample: Sample, fingerprint: str) -> tuple[bool, str]:
    if str(rec.get("settings_hash")) != fingerprint:
        return False, f"settings fingerprint {rec.get('settings_hash')} != current {fingerprint}"
    for key, val in sample.theta().items():
        rv = rec.get(key)
        if rv is None or not np.isclose(float(rv), val, rtol=0, atol=1e-10):
            return False, f"{key} = {rv} != design value {val}"
    return True, ""


def pending_samples(
    samples: Sequence[Sample],
    existing: dict[str, dict[str, Any]],
    settings: DatasetSettings,
    *,
    retry_errors: bool = False,
    force: bool = False,
) -> list[Sample]:
    """Samples still to compute. Raises ``SettingsMismatchError`` for a stale record unless ``force``."""
    fp = settings.fingerprint()
    out = []
    stale = []
    for s in samples:
        rec = existing.get(s.id)
        if rec is None:
            out.append(s)
            continue
        ok, why = _matches(rec, s, fp)
        if not ok:
            if force:
                out.append(s)
            else:
                stale.append(f"{s.id}: {why}")
            continue
        if str(rec.get("status")) != "ok" and retry_errors:
            out.append(s)
    if stale:
        raise SettingsMismatchError(
            f"{len(stale)} existing record(s) do not match the current design/settings, e.g. "
            f"{stale[0]}. Use a different output/parts directory, or force=True (--force) to "
            "recompute them."
        )
    return out


def _hms(seconds: float) -> str:
    """Compact duration: '2h05m', '7m12s', '42s'."""
    if not math.isfinite(seconds):
        return "  ?"
    s = int(round(seconds))
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


def _run_one(sample: Sample, settings: DatasetSettings, parts_dir: str, fields_dir: str | None) -> dict[str, Any]:
    """Worker entry point: compute, write the part, return a short summary."""
    rec = compute_sample(sample, settings, fields_dir=fields_dir)
    write_part(rec, parts_dir)
    return {"id": rec["id"], "status": rec["status"], "wall_time_s": rec["wall_time_s"],
            "error": rec["error"], "peak_rss_mb": rec.get("peak_rss_mb", float("nan"))}


def run_samples(
    samples: Sequence[Sample],
    settings: DatasetSettings,
    *,
    parts_dir: str | Path,
    n_jobs: int = 1,
    fields_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Compute ``samples`` (most expensive first) with joblib; each worker writes its own part.

    Returns the per-sample summaries in completion order. ``progress`` receives one
    line per finished sample (with a cost-weighted ETA).
    """
    s = settings.resolved()
    todo = sorted(samples, key=lambda x: (-relative_cost(x), x.id))  # LPT: long jobs first
    if not todo:
        return []
    say = progress or LOG.info
    total_cost = sum(relative_cost(x) for x in todo)
    cost_of = {x.id: relative_cost(x) for x in todo}
    t0 = time.perf_counter()
    done_cost = 0.0
    results: list[dict[str, Any]] = []
    pd_, fd_ = str(parts_dir), (str(fields_dir) if fields_dir is not None else None)

    def _report(res: dict[str, Any]) -> None:
        nonlocal done_cost
        done_cost += cost_of[res["id"]]
        results.append(res)
        el = time.perf_counter() - t0
        eta = el * (total_cost - done_cost) / done_cost if done_cost > 0 else float("nan")
        tag = "" if res["status"] == "ok" else f"  [{res['status'].upper()}: {res['error'][:120]}]"
        say(f"[{len(results):4d}/{len(todo)}] {res['id']:<11s} {res['wall_time_s']:7.1f} s   "
            f"elapsed {_hms(el)}   ETA {_hms(eta)}{tag}")

    if int(n_jobs) == 1:
        for x in todo:
            _report(_run_one(x, s, pd_, fd_))
        return results

    from joblib import Parallel, delayed

    kw: dict[str, Any] = {"n_jobs": int(n_jobs), "batch_size": 1, "pre_dispatch": "n_jobs"}
    for mode in ("generator_unordered", "generator"):
        try:
            par = Parallel(return_as=mode, **kw)
            break
        except (TypeError, ValueError):
            continue
    else:  # very old joblib: no streaming
        par = None
    if par is None:
        for res in Parallel(**kw)(delayed(_run_one)(x, s, pd_, fd_) for x in todo):
            _report(res)
        return results
    for res in par(delayed(_run_one)(x, s, pd_, fd_) for x in todo):
        _report(res)
    return results


# ---- table ------------------------------------------------------------------------
_LEADING = ["id", "family", "index", "status", "w", "rho_target", "a_z", "a_z_level", "a_z_quantum",
            "rho_voxel", "porosity"]


def records_to_frame(records: Iterable[dict[str, Any]]) -> Any:
    """Records -> DataFrame, design columns first, rows ordered pure lines, corners, edges, Sobol."""
    import pandas as pd

    df = pd.DataFrame(list(records))
    if df.empty:
        return df
    df["_fam"] = df["family"].map(_FAMILY_ORDER).fillna(9)
    df = df.sort_values(["_fam", "index"]).drop(columns="_fam").reset_index(drop=True)
    lead = [c for c in _LEADING if c in df.columns]
    tail = ["error", "wall_time_s", "peak_rss_mb", "settings_hash"] + [c for c in df.columns if c.startswith("rr_")] + ["run_record_json"]
    tail = [c for c in tail if c in df.columns and c not in lead]
    mid = [c for c in df.columns if c not in lead and c not in tail]
    return df[lead + mid + tail]


def consolidate(
    parts_dir: str | Path,
    table_path: str | Path,
    *,
    samples: Sequence[Sample] | None = None,
    csv_fallback: bool = True,
) -> tuple[Any, Path]:
    """Merge the existing table and all parts into one table and write it.

    With ``samples``, only records of the current design are kept (others are
    reported). If parquet is unavailable and ``csv_fallback``, writes ``.csv``.
    Returns (DataFrame, path written).
    """
    recs = load_existing(parts_dir, table_path)
    if samples is not None:
        ids = {s.id for s in samples}
        extra = sorted(set(recs) - ids)
        if extra:
            LOG.info("%d record(s) not in the current design kept out of the table (e.g. %s)",
                     len(extra), extra[0])
        recs = {k: v for k, v in recs.items() if k in ids}
    df = records_to_frame(recs.values())
    path = Path(table_path)
    if path.suffix == ".parquet" and not parquet_available():
        if not csv_fallback:
            raise ImportError("pyarrow is required to write parquet")
        path = path.with_suffix(".csv")
        LOG.warning("pyarrow not installed: writing %s instead of parquet", path.name)
    write_table(df, path)
    return df, path


# =============================================================================
# Quick-look figure
# =============================================================================
#: (column, label, log-scale) of the quick-look panels
QUICKLOOK_PANELS: tuple[tuple[str, str, bool], ...] = (
    ("a_sf_L", r"$a_{sf}L$", False),
    ("wall_min_L", r"min wall thickness $/L$", False),
    ("pore_throat_L", r"fluid throat $/L$", False),
    ("keff_mean", r"$\bar k_{eff}/k_s$", False),
    ("keff_anisotropy", r"$k_{eff}$ anisotropy $(\lambda_{max}-\lambda_{min})/\bar\lambda$", False),
    ("E_hill", r"$E_{Hill}/E_s$", True),
    ("A_U", r"universal anisotropy $A^U$", False),
    ("loc_uniaxial_z_p99", r"stress localization $K_{p99}$ (uniaxial $z$)", True),
    ("K_mean", r"$\bar K/L^2$", True),
    ("K_zz_over_xx", r"$K_{zz}/K_{xx}$", False),
    ("tortuosity", r"tortuosity $T$", False),
    ("kozeny_c", r"Kozeny constant $\phi^3/(K a_{sf}^2)$", False),
)


def _blend_cmap() -> Any:
    """Diverging map for w: gyroid blue -> neutral grey (most blended) -> diamond orange."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("voxlat_w", ["#2a78d6", "#a9a8a2", "#eb6834"])


def plot_quicklook(df: Any, name: str = "task5_closures_quicklook", directory: str | Path | None = None,
                   title: str | None = None) -> Path:
    """Grid of every closure vs rho*, coloured by w, marker by a_z (300-dpi PNG via save_figure)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from voxlat.utils.figures import save_figure

    d = df.copy()
    if "status" in d.columns:
        d = d[d["status"] != "error"]
    if {"K_zz", "K_xx"} <= set(d.columns):
        d["K_zz_over_xx"] = d["K_zz"] / d["K_xx"]
    cmap = _blend_cmap()
    groups = [
        (d["a_z"] < 0.97, "v", r"$a_z<1$"),
        ((d["a_z"] >= 0.97) & (d["a_z"] <= 1.03), "o", r"$a_z\approx1$"),
        (d["a_z"] > 1.03, "^", r"$a_z>1$"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10.2), constrained_layout=True)
    sc = None
    for ax, (col, label, logy) in zip(axes.ravel(), QUICKLOOK_PANELS):
        ax.set_title(label, fontsize=9.5, loc="left")
        ax.grid(True, color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if col not in d.columns or d[col].notna().sum() == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="#7a7974")
            continue
        for mask, mk, _ in groups:
            g = d[mask & d[col].notna()]
            if len(g):
                sc = ax.scatter(g["rho_target"], g[col], c=g["w"], cmap=cmap, vmin=0, vmax=1, marker=mk,
                                s=26, edgecolors="white", linewidths=0.5, zorder=3)
        if logy:
            ax.set_yscale("log")
        ax.set_xlim(0.185, 0.515)
        ax.tick_params(which="both", labelsize=8, colors="#52514e")
    for ax in axes[-1]:
        ax.set_xlabel(r"relative density $\rho^*$", fontsize=9)
    handles = [Line2D([], [], marker=mk, ls="", color="#52514e", label=lab, markersize=6) for _, mk, lab in groups]
    axes[0, 0].legend(handles=handles, fontsize=8, frameon=False, loc="best")
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.01, aspect=40)
        cb.set_label("blend $w$  (0 = gyroid, 1 = diamond)", fontsize=9)
        cb.ax.tick_params(labelsize=8)
    if title is None:
        title = "VoxLat closure dataset"
        if len(d):
            n_ok = int((d["status"] == "ok").sum()) if "status" in d.columns else len(d)
            title += (f": {len(d)} samples ({n_ok} ok), two-grid Richardson "
                      f"R({int(d['n_coarse'].iloc[0])}, {int(d['n_fine'].iloc[0])})")
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    return save_figure(fig, name, directory)
