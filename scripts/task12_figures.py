"""Task 12 figures (300 dpi, results/figures/):

* task12_phase_field.png   - the exported lattice unwrapped at the mean gap radius (theta x z) for the three
                             designs recommended in STATUS (Task 11): grading, integer cells around, no tears.
* task12_sector_slices.png - the sector of slots/best_at_cap around its most critical cell: axial cut, meridional
                             (r-z) cut and an unwrapped cut at r_m, with sleeve, outer wall, caps, slot and port.
* task12_volume_fraction.png - solid fraction of the generated lattice vs the design rho* and vs the phase-aware
                             expectation, per design (from results/export/*_numpy_checks.json).

    python scripts/task12_figures.py            (~1 min; the third figure needs the 'numpy' runs)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voxlat.export.config import ExportOptions, build_export_config, load_design  # noqa: E402
from voxlat.export.field import JacketField  # noqa: E402
from voxlat.utils import save_figure  # noqa: E402

DESIGNS = [("task11_slots_best_at_cap", "slots / best_at_cap"), ("task11_end_rings_knee", "end rings / knee"),
           ("task11_end_rings_min_R", "end rings / min_R (uniform)")]
VF_DESIGNS = DESIGNS + [("task12_slots_best_at_cap_gyroid", "slots / best_at_cap, pure gyroid (Task 12)")]


def _field(name: str, **kw) -> JacketField:
    d = load_design(ROOT / "results" / "designs" / f"{name}.json")
    return JacketField(build_export_config(d, ExportOptions(**kw)), d)


def fig_phase_field():
    fig, axes = plt.subplots(len(DESIGNS), 1, figsize=(13, 2.6 * len(DESIGNS)), constrained_layout=True)
    for ax, (name, label) in zip(axes, DESIGNS):
        f = _field(name, mode="full")
        th = np.linspace(0, 2 * math.pi, 3600, endpoint=False)
        z = np.linspace(f.z_lo, f.z_hi, 420)
        T, Z = np.meshgrid(th, z, indexing="ij")
        r = f.r_m
        sd = f.signed_distance(r * np.cos(T), r * np.sin(T), Z)
        ax.imshow((sd <= 0).T, origin="lower", extent=[0, 360, f.z_lo, f.z_hi], aspect="auto", cmap="Greys",
                  interpolation="nearest")
        D = f.describe()
        ax.set_title(f"{label}: N = {D['N_theta']} cells around, circumferential stretch "
                     f"{D['circumferential_stretch_min']:.2f}-{D['circumferential_stretch_max']:.2f}, "
                     f"axial phase tilt <= {D['axial_phase_shear_deg_max']:.0f} deg, "
                     f"{D['radial_cells_min']:.1f}-{D['radial_cells_max']:.1f} cells across the gap", fontsize=9)
        ax.set_ylabel("z [mm]")
        ax.set_xticks(np.arange(0, 361, 45))
    axes[-1].set_xlabel("theta [deg] (0 = inlet manifold)   -   solid at r = r_m = 74.5 mm (black)")
    save_figure(fig, "task12_phase_field")
    plt.close(fig)


def fig_sector_slices():
    f = _field("task11_slots_best_at_cap", mode="sector", sector_center_deg="auto", sector_angle_deg=30.0)
    crit = f.config["critical_location"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.0, 0.75, 1.4]},
                           constrained_layout=True)
    h = 0.03
    half = f.sector_half
    top = f.r_o + 7
    # (a) axial cut at the critical z (sector-local frame)
    zc = max(crit["z_mm"], 1.0)
    x = np.arange(f.r_s * math.cos(half) - 1, top, h)
    y = np.arange(-top * math.sin(half), top * math.sin(half), h)
    X, Y = np.meshgrid(x, y, indexing="ij")
    sd = f.signed_distance(X, Y, np.full_like(X, zc))
    ax[0].imshow((sd <= 0).T, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap="Greys", interpolation="nearest")
    ax[0].set_title(f"(a) axial cut z = {zc:.1f} mm", fontsize=10)
    ax[0].set_xlabel("x [mm]")
    ax[0].set_ylabel("y [mm]")
    # (b) meridional cut (r-z) at the sector centre
    rr = np.arange(f.r_s - 1, f.r_o + 1, h)
    zz = np.arange(f.z_lo - 1, f.z_hi + 1, h)
    R, Z = np.meshgrid(rr, zz, indexing="ij")
    sd = f.signed_distance(R, np.zeros_like(R), Z)
    ax[1].imshow((sd <= 0).T, origin="lower", extent=[rr[0], rr[-1], zz[0], zz[-1]], cmap="Greys",
                 interpolation="nearest", aspect="auto")
    ax[1].set_title("(b) meridional cut at the sector centre", fontsize=10)
    ax[1].set_xlabel("r [mm]")
    ax[1].set_ylabel("z [mm]")
    # (c) unwrapped at r_m over the sector
    th = np.linspace(-half, half, 1200)
    zz2 = np.linspace(f.z_lo, f.z_hi, 900)
    T, Z = np.meshgrid(th, zz2, indexing="ij")
    sd = f.signed_distance(f.r_m * np.cos(T), f.r_m * np.sin(T), Z)
    ext = [math.degrees(th[0]) + crit["theta_deg"], math.degrees(th[-1]) + crit["theta_deg"], zz2[0], zz2[-1]]
    ax[2].imshow((sd <= 0).T, origin="lower", extent=ext, cmap="Greys", interpolation="nearest", aspect="auto")
    ax[2].plot([crit["theta_deg"]], [crit["z_mm"]], "o", mfc="none", mec="tab:red", ms=12)
    ax[2].set_title(f"(c) unwrapped at r_m; red: most critical cell (L/L_min = {crit['L_over_Lmin']:.3f})", fontsize=10)
    ax[2].set_xlabel("theta [deg]")
    ax[2].set_ylabel("z [mm]")
    fig.suptitle(f"slots / best_at_cap - 30 deg sector centred at {crit['theta_deg']:.1f} deg (auto: smallest L / L_min)",
                 fontsize=11)
    save_figure(fig, "task12_sector_slices")
    plt.close(fig)


def fig_volume_fraction():
    rows = []
    for name, label in VF_DESIGNS:
        p = ROOT / "results" / "export" / f"{name.replace('task11_', '')}_sector30_numpy_checks.json"
        if not p.is_file():
            print(f"skip {p.name} (run: python scripts/task12_export.py numpy --design results/designs/{name}.json)")
            continue
        v = json.loads(p.read_text())["volume_fraction_raw"]
        rows.append((label, v))
    if not rows:
        return
    keys = [("relative_error", "all vs design rho*"), ("core_relative_error", "core vs design"),
            ("near_wall_relative_error", "near walls vs design"),
            ("relative_error_vs_phase_aware", "all vs phase-aware"),
            ("core_relative_error_vs_phase_aware", "core vs phase-aware"),
            ("near_wall_relative_error_vs_phase_aware", "near walls vs phase-aware")]
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    wbar = 0.8 / len(rows)
    for k, (label, v) in enumerate(rows):
        vals = [100 * (v[key] or 0.0) for key, _ in keys]
        ax.bar(np.arange(len(keys)) + (k - (len(rows) - 1) / 2) * wbar, vals, wbar, label=label)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([t for _, t in keys], fontsize=9)
    ax.set_ylabel("solid fraction error [%]")
    ax.set_title("Generated lattice (30 deg sectors, 0.06-0.07 mm voxels) vs design rho* and vs the phase-aware expectation", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, "task12_volume_fraction")
    plt.close(fig)


if __name__ == "__main__":
    fig_phase_field()
    fig_sector_slices()
    fig_volume_fraction()
    print("figures written to results/figures/task12_*.png")
