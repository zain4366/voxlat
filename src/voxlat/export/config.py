"""Export configuration for the PicoGK geometry export (Task 12).

The C# library (``src/voxlat/export/dotnet``) and the numpy mirror (``voxlat.export.field``) both
read two JSON files:

1. the **design JSON** written by Task 11 (``results/designs/*.json``; SI units): layout, B-spline
   basis description and the physical control nets of rho*, w, L (and a_z if graded);
2. the **export config JSON** written here (``build_export_config``; millimetres, the PicoGK unit):
   jacket radii from ``configs/reference.yaml``, manifold/port/end-cap geometry, lattice phase
   settings, the continuum TPMS threshold table c(w, rho) (Task 1) and render options.

Keeping every number the C# side needs in these two files means C# never re-implements config
logic: it only evaluates the geometry. The schema is versioned (``SCHEMA``); the C# loader rejects
other versions.

Geometry added by the export (not part of the device model; all configurable, documented in
STATUS.md Task 12):

* end caps (annular plates, ``end_cap_mm`` thick) close the jacket axially; the sleeve and outer
  wall run over the full axial extent of the part;
* ``slots`` layout: the 15 mm (at r_m) manifold slots are open plenums in the gap over the slot's
  axial extent (no lattice), the outer wall spans them (as in the Task 9 structural model), and a
  radial port bore (``port_bore_mm``) through a cylindrical boss feeds each plenum at mid-length;
* ``end_rings`` layout: annular ring plenums of height ``ring_height_mm`` below z = 0 and above
  z = L_ax span the gap (outside the heated length, as in Task 11); inlet port at theta_in on the
  inlet ring, outlet port at theta_out on the outlet ring.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = [
    "SCHEMA",
    "ExportOptions",
    "load_design",
    "build_export_config",
    "write_export_config",
    "critical_location",
    "DEFAULT_DESIGN",
]

SCHEMA = "voxlat-export-config/1"
DEFAULT_DESIGN = "results/designs/task11_slots_best_at_cap.json"


@dataclass
class ExportOptions:
    """Render and geometry options (millimetres, degrees).

    ``voxel_size_mm`` None -> 0.2 mm for ``mode="full"`` (visual/assembly STL, walls resolution-limited)
    and 0.05 mm for ``mode="sector"`` (manufacturability checks: 0.35 mm wall = 7 voxels).
    ``sector_center_deg`` "auto" -> the angle of the most critical lattice cell (smallest L / L_min,
    fine morphology table) - the place where a wall or pore check is most likely to fail.
    ``sliver_radius_mm`` None -> min_wall / 2: a morphological opening (PicoGK offset -r, +r) that
    removes the sub-min-wall slivers created where the lattice is clipped at open faces (slot sides,
    ring faces); 0 disables it. It acts only within ``sliver_band_mm`` (None -> min_wall + 2 voxels) of
    those faces, so thin interior walls are reported by the checks instead of being silently removed;
    ``sliver_band_mm=0`` opens the whole part.
    """

    mode: str = "sector"
    voxel_size_mm: float | None = None
    sector_center_deg: float | str = "auto"
    sector_angle_deg: float = 30.0
    sector_margin_mm: float = 2.0
    # lattice phase field
    n_theta_cells: int | None = None  # None -> round(mean_z of the cell count around r_m)
    phase_r0: float = 0.0  # radial cut phase at the sleeve (cell units; Task 7 convention)
    phase_theta0: float = 0.0
    phase_z0: float = 0.0
    axial_phase_origin_mm: float | None = None  # None -> mid-length
    circumferential: str = "integer"  # "integer" (seamless, default) | "seam" (see voxlat.export.field)
    seam_angle_deg: float | None = None  # None -> inlet angle (inside the inlet slot for "slots")
    seam_rib_mm: float = 0.6
    conformal: bool = True  # log-polar radial phase (cubic cells in r-theta); False -> linear in r
    table_n_theta: int = 2048
    table_n_z: int = 257
    sd_clamp_mm: float = 2.0
    lattice_overlap_mm: float = 0.1  # lattice region overlaps sleeve / outer wall / end caps (fusion)
    # housing
    end_cap_mm: float = 2.0
    ring_height_mm: float = 8.0
    port_bore_mm: float | None = None  # None -> slots 8 mm, end rings 6 mm
    boss_diameter_mm: float | None = None  # None -> bore + 6 mm
    boss_height_mm: float = 6.0
    # post-processing / outputs
    sliver_radius_mm: float | None = None
    sliver_band_mm: float | None = None  # None -> min_wall + 2 voxels; 0 -> open the whole part
    save_stl: bool = True
    write_mask: bool = True
    write_raw_mask: bool = True
    show_in_viewer: bool = True
    output_dir: str = "results/export"
    name: str | None = None

    def resolved_voxel(self) -> float:
        if self.voxel_size_mm is not None:
            return float(self.voxel_size_mm)
        return 0.2 if self.mode == "full" else 0.05


def load_design(path: str | Path) -> dict[str, Any]:
    """Read a Task 11 design JSON and check the fields the exporter needs."""
    path = Path(path)
    d = json.loads(path.read_text())
    for key in ("layout", "basis", "control_nets"):
        if key not in d:
            raise ValueError(f"{path.name}: missing '{key}' (not a Task 11 design JSON?)")
    if d["layout"] not in ("slots", "end_rings"):
        raise ValueError(f"{path.name}: unknown layout {d['layout']!r}")
    for name in ("rho", "w", "L"):
        if name not in d["control_nets"]:
            raise ValueError(f"{path.name}: control net '{name}' missing")
    d["_path"] = str(path.resolve())
    return d


def critical_location(design: Mapping[str, Any], cfg: Any = None, n_theta: int = 720, n_z: int = 101) -> dict[str, float]:
    """Most critical lattice cell for manufacturability: argmin of L / L_min(w, rho) (fine Task 11 table).

    Evaluated on a (theta, z) grid over the lattice planform (manifold slots excluded). Returns the
    angle (deg), z (mm), the ratio and the local fields there - used for the "auto" sector centre.
    """
    from voxlat.export.field import DesignFields
    from voxlat.optimize.parameterization import FastMinCellSize
    from voxlat.utils.config import load_config

    cfg = cfg or load_config()
    fields = DesignFields(design)
    th = (np.arange(n_theta) + 0.5) * 2.0 * math.pi / n_theta
    zz = (np.arange(n_z) + 0.5) * fields.axial_length_mm / n_z
    T, Z = np.meshgrid(th, zz, indexing="ij")
    F = fields.evaluate(T, Z)
    lmin = FastMinCellSize(cfg, fine=True)(F["w"], F["rho"]) * 1e3  # mm
    ratio = F["L"] / lmin
    if design["layout"] == "slots":
        man = cfg.manifolds
        r_m = cfg.jacket.lattice_mean_radius
        half = 0.5 * man.width / r_m
        for a in (man.inlet_angle, man.outlet_angle):
            d = np.abs((T - a + math.pi) % (2 * math.pi) - math.pi)
            ratio = np.where(d < half, np.inf, ratio)
    k = np.unravel_index(int(np.argmin(ratio)), ratio.shape)
    return {
        "theta_deg": float(np.degrees(T[k])),
        "z_mm": float(Z[k]),
        "L_over_Lmin": float(ratio[k]),
        "rho": float(F["rho"][k]),
        "w": float(F["w"][k]),
        "L_mm": float(F["L"][k]),
        "L_min_mm": float(lmin[k]),
    }


def build_export_config(design: Mapping[str, Any] | str | Path, options: ExportOptions | None = None,
                        cfg: Any = None) -> dict[str, Any]:
    """Export-config dict (mm, deg) for one design - everything the C# library needs besides the design."""
    from voxlat.geometry.tpms import load_threshold_table
    from voxlat.utils.config import load_config

    opts = options or ExportOptions()
    cfg = cfg or load_config()
    if not isinstance(design, Mapping):
        design = load_design(design)
    if "_path" not in design:
        raise ValueError("design must come from load_design() (needs its file path)")
    if opts.mode not in ("full", "sector"):
        raise ValueError("mode must be 'full' or 'sector'")
    layout = design["layout"]
    jac, man, mfg = cfg.jacket, cfg.manifolds, cfg.manufacturing
    mm = 1e3
    r_s = jac.stator_radius * mm
    r_li = jac.lattice_inner_radius * mm
    r_lo = jac.lattice_outer_radius * mm
    r_o = jac.outer_radius * mm
    r_m = jac.lattice_mean_radius * mm
    L_ax = jac.axial_length * mm
    r_m_design = float(design["basis"]["r_mean_m"]) * mm
    L_ax_design = float(design["basis"]["axial_length_m"]) * mm
    if abs(r_m_design - r_m) > 1e-6 or abs(L_ax_design - L_ax) > 1e-6:
        raise ValueError("design basis (r_m, L_ax) does not match configs/reference.yaml")

    bore = opts.port_bore_mm if opts.port_bore_mm is not None else (8.0 if layout == "slots" else 6.0)
    boss = opts.boss_diameter_mm if opts.boss_diameter_mm is not None else bore + 6.0
    t_cap = float(opts.end_cap_mm)
    if layout == "slots":
        ext = min(man.axial_extent, jac.axial_length) * mm
        z0, z1 = 0.5 * (L_ax - ext), 0.5 * (L_ax + ext)
        width = man.width * mm
        if bore >= width:
            raise ValueError("port bore must be narrower than the manifold slot")
        z_lo, z_hi = -t_cap, L_ax + t_cap
        slot = {"inlet_angle_deg": math.degrees(man.inlet_angle), "outlet_angle_deg": math.degrees(man.outlet_angle),
                "width_mm": width, "r_ref_mm": r_m, "z0_mm": z0, "z1_mm": z1}
        ring = None
        z_port_in = z_port_out = 0.5 * (z0 + z1)
    else:
        h_ring = float(opts.ring_height_mm)
        if bore >= h_ring:
            raise ValueError("port bore must be smaller than the ring plenum height")
        z_lo, z_hi = -h_ring - t_cap, L_ax + h_ring + t_cap
        slot = None
        ring = {"height_mm": h_ring}
        z_port_in, z_port_out = -0.5 * h_ring, L_ax + 0.5 * h_ring
    ports = [
        {"name": "inlet", "angle_deg": math.degrees(man.inlet_angle), "z_mm": z_port_in,
         "bore_radius_mm": 0.5 * bore, "boss_radius_mm": 0.5 * boss, "boss_height_mm": float(opts.boss_height_mm)},
        {"name": "outlet", "angle_deg": math.degrees(man.outlet_angle), "z_mm": z_port_out,
         "bore_radius_mm": 0.5 * bore, "boss_radius_mm": 0.5 * boss, "boss_height_mm": float(opts.boss_height_mm)},
    ]

    tab = load_threshold_table("network")
    voxel = opts.resolved_voxel()
    sliver = (0.5 * mfg.min_wall_thickness * mm) if opts.sliver_radius_mm is None else float(opts.sliver_radius_mm)
    band = (mfg.min_wall_thickness * mm + 2.0 * voxel) if opts.sliver_band_mm is None else float(opts.sliver_band_mm)
    if opts.mode == "sector":
        if opts.sector_center_deg == "auto":
            crit = critical_location(design, cfg)
            centre = crit["theta_deg"]
        else:
            crit = None
            centre = float(opts.sector_center_deg)
    else:
        crit, centre = None, 0.0
    name = opts.name or (Path(design["_path"]).stem.replace("task11_", "") + ("_full" if opts.mode == "full"
                                                                              else f"_sector{opts.sector_angle_deg:g}"))
    return {
        "schema": SCHEMA,
        "design_path": design["_path"],
        "design_label": design.get("label", ""),
        "layout": layout,
        "units": "mm, deg",
        "geometry": {
            "r_stator_mm": r_s,
            "r_lattice_in_mm": r_li,
            "r_lattice_out_mm": r_lo,
            "r_outer_mm": r_o,
            "r_mean_mm": r_m,
            "axial_length_mm": L_ax,
            "z_min_mm": z_lo,
            "z_max_mm": z_hi,
            "end_cap_mm": t_cap,
            "lattice_overlap_mm": float(opts.lattice_overlap_mm),
            "slot": slot,
            "ring": ring,
            "ports": ports,
        },
        "lattice": {
            "kind": "network",
            "n_theta_cells": opts.n_theta_cells,
            "phase_r0": float(opts.phase_r0),
            "phase_theta0": float(opts.phase_theta0),
            "phase_z0": float(opts.phase_z0),
            "axial_phase_origin_mm": 0.5 * L_ax if opts.axial_phase_origin_mm is None else float(opts.axial_phase_origin_mm),
            "conformal": bool(opts.conformal),
            "circumferential": str(opts.circumferential),
            "seam_angle_deg": math.degrees(man.inlet_angle) if opts.seam_angle_deg is None else float(opts.seam_angle_deg),
            "seam_rib_mm": float(opts.seam_rib_mm),
            "table_n_theta": int(opts.table_n_theta),
            "table_n_z": int(opts.table_n_z),
            "sd_clamp_mm": float(opts.sd_clamp_mm),
        },
        "threshold_table": {
            "w": [float(v) for v in tab.w],
            "rho": [float(v) for v in tab.rho],
            "c": [[float(v) for v in row] for row in np.asarray(tab.c)],
            "n_ref": int(tab.n_ref),
        },
        "manufacturing": {
            "min_wall_mm": mfg.min_wall_thickness * mm,
            "min_pore_mm": mfg.min_pore_size * mm,
        },
        "render": {
            "mode": opts.mode,
            "voxel_size_mm": voxel,
            "sector_center_deg": centre,
            "sector_angle_deg": float(opts.sector_angle_deg),
            "sector_margin_mm": float(opts.sector_margin_mm),
            "sliver_radius_mm": sliver,
            "sliver_band_mm": band,
            "save_stl": bool(opts.save_stl),
            "write_mask": bool(opts.write_mask),
            "write_raw_mask": bool(opts.write_raw_mask),
            "show_in_viewer": bool(opts.show_in_viewer),
            "output_dir": str(Path(opts.output_dir).resolve()),
            "name": name,
        },
        "critical_location": crit,
        "options": asdict(opts),
    }


def write_export_config(config: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=1))
    return path
