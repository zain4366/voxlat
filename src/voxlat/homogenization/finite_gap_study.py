"""Finite-gap study driver (Task 7): design of the strip suites, resumable runs, table.

Suites (``study_design``), all at n voxels per cell, wall t_w = 0.5 L unless stated:

* ``uniform``     - the main sweep: w in {0, 0.5, 1} x rho* in {0.25, 0.35, 0.45}
                    x N in {1, 1.5, 2, 3, 4, 6, 8} x cut phases {0, 1/3, 2/3} (+ 1/8 for
                    gyroid and diamond, whose distinct cuts lie in [0, L/8] and for which
                    1/3 and 2/3 both reduce to L/12; see ``reduced_phase``) (231 strips);
* ``graded``      - linear rho*(x) across the gap, rho*_mid = 0.35: N = 4 with
                    g = |grad rho*| L in {0.025, 0.05, 0.075} and N = 2 with g in
                    {0.075, 0.15} (both reach rho* = 0.20 ... 0.50), 3 morphologies x 3 phases;
* ``convergence`` - resolution check: G and D, rho* = 0.35, N in {1, 2, 4}, phase 0,
                    n in {24, 32, 48};
* ``wall``        - wall-thickness check: G and D, rho* = 0.35, N in {1, 2}, phase 0,
                    t_w / L in {0.125, 0.25, 0.5, 1.0};
* ``stretch``     - G and D, rho* = 0.35, a_z in {0.75, 1.5}, N in {1, 2, 4}, 3 phases.

Every strip needs the bulk cell of the same (w, rho*, n, phase, a_z) (uniform
strips) or a density sweep of bulk cells (graded strips, rho* = 0.175 ... 0.525 in
steps of 0.025). Bulk results and strip results are stored as one JSON part per
id (``data/finite_gap/bulk/``, ``data/finite_gap/strips/``); reruns skip finished
ids with the same settings fingerprint. ``consolidate`` writes
``data/finite_gap.csv`` (and ``.parquet`` when pyarrow is installed).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from voxlat.homogenization.finite_gap import (
    BulkInterpolator,
    GapProperties,
    GapSpec,
    bulk_properties,
    default_phases,
    discrepancy,
    homogenized_properties,
    morphology_label,
    strip_properties,
)
from voxlat.utils.log import get_logger

__all__ = [
    "STUDY_SCHEMA_VERSION",
    "SUITES",
    "StudySettings",
    "BulkKey",
    "study_design",
    "bulk_requirements",
    "GRADED_BULK_RHOS",
    "compute_bulk",
    "compute_strip",
    "run_study",
    "consolidate",
    "load_bulk",
    "props_to_json",
    "props_from_json",
    "strip_cost",
    "default_paths",
    "in_suite",
]

LOG = get_logger("homogenization.finite_gap_study")
STUDY_SCHEMA_VERSION = 1
SUITES = ("uniform", "graded", "convergence", "wall", "stretch")

UNIFORM_W = (0.0, 0.5, 1.0)
UNIFORM_RHO = (0.25, 0.35, 0.45)
UNIFORM_N = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
GRADED_CASES = ((4.0, 0.025), (4.0, 0.05), (4.0, 0.075), (2.0, 0.075), (2.0, 0.15))
GRADED_BULK_RHOS = tuple(np.round(np.arange(0.175, 0.5251, 0.025), 3))
#: For gyroid/diamond the default cuts 1/3 and 2/3 are both equivalent to L/12
#: (``reduced_phase``); L/8 completes the irreducible interval [0, L/8].
PURE_EXTRA_PHASES = (0.125,)


@dataclass(frozen=True)
class StudySettings:
    """Material constants and solver settings shared by every case (fingerprinted)."""

    k_s: float = 130.0
    k_f: float = 0.40
    E: float = 70.0e9
    nu: float = 0.33
    tol: float = 1e-8
    preconditioner: str = "two_level"

    @classmethod
    def from_config(cls, cfg: Any = None, **kw: Any) -> "StudySettings":
        if cfg is None:
            from voxlat.utils.config import load_config

            cfg = load_config()
        return cls(k_s=cfg.material.conductivity, k_f=cfg.coolant.conductivity,
                   E=cfg.material.youngs_modulus, nu=cfg.material.poisson_ratio, **kw)

    def fingerprint(self) -> str:
        d = asdict(self)
        d.pop("preconditioner")  # results agree to the solver tolerance for every preconditioner
        d["schema"] = STUDY_SCHEMA_VERSION
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def solver_kwargs(self) -> dict[str, Any]:
        return dict(k_s=self.k_s, k_f=self.k_f, E=self.E, nu=self.nu, tol=self.tol,
                    preconditioner=self.preconditioner)


def default_paths(root: str | Path | None = None) -> dict[str, Path]:
    from voxlat.utils.paths import data_dir

    base = Path(root) if root is not None else data_dir()
    return {"bulk": base / "finite_gap" / "bulk", "strips": base / "finite_gap" / "strips",
            "table": base / "finite_gap.csv"}


# =============================================================================
# Design
# =============================================================================
def study_design(suites: Iterable[str] = SUITES, n: int = 32, *, phases: Sequence[float] | None = None,
                 convergence_n: Sequence[int] = (24, 32, 48)) -> dict[str, list[GapSpec]]:
    """Strip specs per suite (see the module docstring). ``n`` = default resolution."""
    ph = tuple(default_phases(3) if phases is None else phases)
    out: dict[str, list[GapSpec]] = {}
    for s in suites:
        if s not in SUITES:
            raise ValueError(f"unknown suite {s!r}; choose from {SUITES}")
        specs: list[GapSpec] = []
        if s == "uniform":
            specs = [GapSpec(w, r, N, p, n=n) for w in UNIFORM_W for r in UNIFORM_RHO for N in UNIFORM_N
                     for p in (ph + PURE_EXTRA_PHASES if w in (0.0, 1.0) and phases is None else ph)]
        elif s == "graded":
            specs = [GapSpec(w, 0.35, N, p, gradient=g, n=n) for w in UNIFORM_W for (N, g) in GRADED_CASES for p in ph]
        elif s == "convergence":
            specs = [GapSpec(w, 0.35, N, 0.0, n=m) for w in (0.0, 1.0) for N in (1.0, 2.0, 4.0) for m in convergence_n]
        elif s == "wall":
            specs = [GapSpec(w, 0.35, N, 0.0, wall=t, n=n) for w in (0.0, 1.0) for N in (1.0, 2.0)
                     for t in (0.125, 0.25, 0.5, 1.0)]
        elif s == "stretch":
            specs = [GapSpec(w, 0.35, N, p, a_z=az, n=n) for w in (0.0, 1.0) for az in (0.75, 1.5)
                     for N in (1.0, 2.0, 4.0) for p in ph]
        out[s] = specs
    return out


@dataclass(frozen=True)
class BulkKey:
    w: float
    rho: float
    n: int
    phase: float
    a_z: float = 1.0
    kind: str = "network"

    @property
    def id(self) -> str:
        return (f"bulk-{morphology_label(self.w)}-r{self.rho:.3f}-p{self.phase:.4f}-az{self.a_z:.4f}-n{self.n}"
                + ("" if self.kind == "network" else "-sheet"))


def _bulk_keys_for(spec: GapSpec) -> list[BulkKey]:
    rhos = GRADED_BULK_RHOS if spec.graded else (round(spec.rho, 6),)
    return [BulkKey(spec.w, float(r), spec.n, round(spec.phase % 1.0, 6), spec.a_z, spec.kind) for r in rhos]


def bulk_requirements(specs: Iterable[GapSpec]) -> list[BulkKey]:
    """Unique bulk cells needed by a set of strips."""
    seen: dict[str, BulkKey] = {}
    for s in specs:
        for k in _bulk_keys_for(s):
            seen.setdefault(k.id, k)
    return list(seen.values())


def strip_cost(spec: GapSpec) -> float:
    """Relative cost proxy: voxels x (blend factor 1.6) (two-level iterations ~ length independent)."""
    vox = (spec.n_lattice + spec.n_wall) * spec.n * spec.nz
    return vox * (1.6 if 0.0 < spec.w < 1.0 else 1.0)


# =============================================================================
# Serialization of GapProperties
# =============================================================================
def props_to_json(p: GapProperties) -> dict[str, Any]:
    return {
        "k_n": p.k_n, "K_t": np.asarray(p.K_t).tolist(), "C": np.asarray(p.C).tolist(),
        "k_stack": p.k_stack, "K_stack": np.asarray(p.K_stack).tolist(), "C_stack": np.asarray(p.C_stack).tolist(),
        "H": p.H, "T": p.T, "t_wall": p.t_wall, "lattice_density": p.lattice_density, "porosity": p.porosity,
        "k_s": p.k_s, "k_f": p.k_f, "E_s": p.E_s, "nu_s": p.nu_s,
        "wall_times": p.wall_times, "iterations": p.iterations, "preconditioner": p.preconditioner,
    }


def props_from_json(d: dict[str, Any]) -> GapProperties:
    return GapProperties(
        k_n=float(d["k_n"]), K_t=np.asarray(d["K_t"], float), C=np.asarray(d["C"], float),
        k_stack=float(d["k_stack"]), K_stack=np.asarray(d["K_stack"], float), C_stack=np.asarray(d["C_stack"], float),
        H=float(d["H"]), T=float(d["T"]), t_wall=float(d["t_wall"]), lattice_density=float(d["lattice_density"]),
        porosity=float(d["porosity"]), k_s=float(d["k_s"]), k_f=float(d["k_f"]), E_s=float(d["E_s"]),
        nu_s=float(d["nu_s"]), wall_times=dict(d.get("wall_times", {})),
        iterations={k: list(v) for k, v in d.get("iterations", {}).items()},
        preconditioner=str(d.get("preconditioner", "")),
    )


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, allow_nan=True)
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


# =============================================================================
# Workers
# =============================================================================
def compute_bulk(key: BulkKey, settings: StudySettings, out_dir: str | Path) -> dict[str, Any]:
    """Compute (or reuse) one bulk reference cell; returns the stored record."""
    path = Path(out_dir) / f"{key.id}.json"
    fp = settings.fingerprint()
    old = _read_json(path)
    if old is not None and old.get("fingerprint") == fp and old.get("status") == "ok":
        return old
    from voxlat.utils.provenance import run_record

    t0 = time.perf_counter()
    rec: dict[str, Any] = {"id": key.id, "key": asdict(key), "fingerprint": fp}
    try:
        p = bulk_properties(key.w, key.rho, key.n, key.phase, key.a_z, key.kind,  # type: ignore[arg-type]
                            **settings.solver_kwargs())
        rec.update(status="ok", props=props_to_json(p), error="")
    except Exception as exc:  # pragma: no cover - recorded, not raised
        rec.update(status="error", error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    rec["wall_time_s"] = time.perf_counter() - t0
    rec["run_record"] = run_record(asdict(key) | asdict(settings), resolution=key.n, wall_time=rec["wall_time_s"])
    _write_json(path, rec)
    return rec


def load_bulk(keys: Iterable[BulkKey], bulk_dir: str | Path, settings: StudySettings) -> dict[str, GapProperties]:
    """Bulk properties by id (only finished records with the current fingerprint)."""
    out = {}
    fp = settings.fingerprint()
    for k in keys:
        rec = _read_json(Path(bulk_dir) / f"{k.id}.json")
        if rec is not None and rec.get("status") == "ok" and rec.get("fingerprint") == fp:
            out[k.id] = props_from_json(rec["props"])
    return out


def compute_strip(spec: GapSpec, settings: StudySettings, bulk_dir: str | Path, out_dir: str | Path,
                  *, force: bool = False) -> dict[str, Any]:
    """Compute (or reuse) one strip + its homogenized prediction and discrepancies."""
    path = Path(out_dir) / f"{spec.case_id}.json"
    fp = settings.fingerprint()
    old = _read_json(path)
    if not force and old is not None and old.get("fingerprint") == fp and old.get("status") == "ok":
        return old
    from voxlat.utils.provenance import run_record

    t0 = time.perf_counter()
    rec: dict[str, Any] = {"id": spec.case_id, **{k: v for k, v in spec.to_dict().items()},
                           "morphology": spec.morphology, "fingerprint": fp}
    try:
        keys = _bulk_keys_for(spec)
        bulk = load_bulk(keys, bulk_dir, settings)
        missing = [k for k in keys if k.id not in bulk]
        for k in missing:  # compute what the caller did not provide
            compute_bulk(k, settings, bulk_dir)
        bulk = load_bulk(keys, bulk_dir, settings)
        if len(bulk) != len(keys):
            raise RuntimeError("bulk reference cells failed")
        if spec.graded:
            interp = BulkInterpolator([k.rho for k in keys], [bulk[k.id] for k in keys])
            hom = homogenized_properties(spec, interp)
        else:
            hom = bulk[keys[0].id]
        strip, sp = strip_properties(spec, **settings.solver_kwargs())
        d = discrepancy(sp, hom)
        dens = strip.layer_density
        nw = max(1, spec.n // 8)
        rec.update(
            status="ok", error="",
            n_lattice=spec.n_lattice, n_wall=spec.n_wall, H=strip.H, T=strip.T,
            lattice_density=strip.lattice_density,
            wall_density_inner=float(dens[:nw].mean()), wall_density_outer=float(dens[-nw:].mean()),
            strip=props_to_json(sp), hom=props_to_json(hom),
            **{f"delta_{q}": v for q, v in d.items()},
        )
    except Exception as exc:  # pragma: no cover - recorded, not raised
        rec.update(status="error", error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    rec["wall_time_s"] = time.perf_counter() - t0
    rec["run_record"] = run_record(spec.to_dict() | asdict(settings), resolution=spec.n,
                                   wall_time=rec["wall_time_s"])
    _write_json(path, rec)
    return rec


# =============================================================================
# Driver
# =============================================================================
def _parallel(fn: Any, items: list[Any], n_jobs: int, desc: str) -> list[Any]:
    if not items:
        return []
    if n_jobs == 1:
        out = []
        for i, it in enumerate(items):
            out.append(fn(it))
            LOG.info("%s %d/%d", desc, i + 1, len(items))
        return out
    from joblib import Parallel, delayed

    return Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(delayed(fn)(it) for it in items)


def run_study(specs: Sequence[GapSpec], settings: StudySettings | None = None, *, n_jobs: int = 1,
              paths: dict[str, Path] | None = None, force: bool = False) -> list[dict[str, Any]]:
    """Compute all bulk cells, then all strips (largest first); resumable."""
    settings = settings or StudySettings.from_config()
    paths = paths or default_paths()
    keys = bulk_requirements(specs)
    done = load_bulk(keys, paths["bulk"], settings)
    todo = [k for k in keys if k.id not in done]
    LOG.info("bulk cells: %d needed, %d to compute", len(keys), len(todo))
    _parallel(lambda k: compute_bulk(k, settings, paths["bulk"]), todo, n_jobs, "bulk")
    fp = settings.fingerprint()
    pend = []
    for s in specs:
        rec = _read_json(Path(paths["strips"]) / f"{s.case_id}.json")
        if force or rec is None or rec.get("fingerprint") != fp or rec.get("status") != "ok":
            pend.append(s)
    pend.sort(key=strip_cost, reverse=True)
    LOG.info("strips: %d requested, %d to compute", len(specs), len(pend))
    _parallel(lambda s: compute_strip(s, settings, paths["bulk"], paths["strips"], force=force),
              pend, n_jobs, "strip")
    return [r for s in specs if (r := _read_json(Path(paths["strips"]) / f"{s.case_id}.json")) is not None]


def _flatten(rec: dict[str, Any]) -> dict[str, Any]:
    row = {k: v for k, v in rec.items() if not isinstance(v, (dict, list))}
    for prefix in ("strip", "hom"):
        if prefix in rec and isinstance(rec[prefix], dict):
            p = props_from_json(rec[prefix])
            row.update(p.as_row(prefix))
    row["run_record_json"] = json.dumps(rec.get("run_record", {}), default=str)
    return row


def consolidate(design: dict[str, list[GapSpec]] | None = None, paths: dict[str, Path] | None = None,
                settings: StudySettings | None = None, *, write: bool = True) -> Any:
    """Merge all strip parts (current fingerprint) into one table; label suites."""
    import pandas as pd

    from voxlat.closures.dataset import parquet_available, write_table

    settings = settings or StudySettings.from_config()
    paths = paths or default_paths()
    fp = settings.fingerprint()
    rows = []
    for p in sorted(Path(paths["strips"]).glob("*.json")):
        rec = _read_json(p)
        if rec is None or rec.get("fingerprint") != fp:
            continue
        rows.append(_flatten(rec))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if design is not None:
        member: dict[str, list[str]] = {}
        for suite, specs in design.items():
            for s in specs:
                member.setdefault(s.case_id, []).append(suite)
        df["suites"] = df["id"].map(lambda i: ";".join(sorted(set(member.get(i, [])))))
    df = df.sort_values(["w", "rho", "N", "phase", "gradient", "a_z", "wall", "n"]).reset_index(drop=True)
    if write:
        write_table(df, paths["table"])
        if parquet_available():
            write_table(df, Path(paths["table"]).with_suffix(".parquet"))
    return df


def in_suite(df: Any, suite: str) -> np.ndarray:
    """Boolean mask of rows that belong to a suite (needs the ``suites`` column)."""
    return df["suites"].fillna("").str.split(";").map(lambda s: suite in s).to_numpy()
