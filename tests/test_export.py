"""Task 12 - geometry export: numpy mirror of the C# implicit, sector generator, checks, IO.

The C# side is compared with this mirror in tests/test_export_dotnet.py (needs the built library).
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from voxlat.export.checks import check_mask, min_wall, thin_mask, throat
from voxlat.export.config import ExportOptions, build_export_config, critical_location, load_design
from voxlat.export.field import DesignFields, JacketField, ThresholdLookup, clamped_basis, periodic_basis
from voxlat.export.maskio import VoxelMask, read_mask, write_mask
from voxlat.export.sector import (capped_edt, compare_masks, generate_mask, open_mask, planar_profile_table,
                                  volume_fraction_report, wide_half_angle)
from voxlat.export.stl import read_stl, voxelize_stl, write_stl

ROOT = Path(__file__).resolve().parents[1]
DESIGNS = ROOT / "results" / "designs"
BEST = DESIGNS / "task11_slots_best_at_cap.json"
RINGS_KNEE = DESIGNS / "task11_end_rings_knee.json"
UNIFORM = DESIGNS / "task11_end_rings_min_R.json"
DOTNET = ROOT / "src" / "voxlat" / "export" / "dotnet"


def _field(path=BEST, **kw):
    d = load_design(path)
    opts = ExportOptions(**{"sector_center_deg": 90.0, **kw})
    cfg = build_export_config(d, opts)
    return JacketField(cfg, d), cfg, d


def _uniform_design(w, rho, L, layout_src=UNIFORM):
    d = copy.deepcopy(load_design(layout_src))
    for k, v in (("rho", rho), ("w", w), ("L", L)):
        d["control_nets"][k] = np.full(np.shape(d["control_nets"][k]), v).tolist()
    return d


# =============================================================================
# bases, fields, tables
# =============================================================================
@pytest.mark.parametrize("n", [1, 2, 3, 4, 8])
def test_bases_match_parameterization(n):
    from voxlat.optimize.parameterization import clamped_bspline_basis, periodic_cubic_basis

    u = np.concatenate([np.random.default_rng(n).random(500), [0.0, 1.0, 0.999999]])
    assert np.allclose(periodic_basis(u, n), periodic_cubic_basis(u, n), atol=1e-14)
    assert np.allclose(clamped_basis(u, n), clamped_bspline_basis(u, n), atol=1e-14)
    assert np.allclose(periodic_basis(u, n).sum(1), 1.0)
    assert np.allclose(clamped_basis(u, n).sum(1), 1.0)


def test_design_fields_match_task11_parameterization():
    """Fields from the design JSON equal the Task 11 parameterization of the stored x_unit."""
    from voxlat.device import JacketModel, JacketOptions
    from voxlat.optimize.parameterization import GradingParameterization, GradingSpec

    d = load_design(BEST)
    model = JacketModel(options=JacketOptions(manifold_layout=d["layout"]))
    par = GradingParameterization(model, GradingSpec.from_dict(d["basis"]["spec"]))
    x = np.asarray(d["x_unit"], float)
    rng = np.random.default_rng(0)
    th = rng.random(300) * 2 * math.pi
    z = rng.random(300) * 50.0
    F = DesignFields(d).evaluate(th, z)
    r_m = model.grid.r_mean
    for name, scale in (("rho", 1.0), ("w", 1.0), ("L", 1e3)):
        ref = par.field_function(x, name)(th * r_m, z * 1e-3) * scale
        assert np.allclose(F[name], ref, rtol=1e-9, atol=1e-12), name


def test_threshold_lookup_matches_tpms_table():
    from voxlat.geometry.tpms import load_threshold_table, threshold_from_table

    t = load_threshold_table("network")
    lk = ThresholdLookup(t.w, t.rho, t.c)
    rng = np.random.default_rng(1)
    w, r = rng.random(400), 0.05 + 0.9 * rng.random(400)
    assert np.allclose(lk(w, r), threshold_from_table(w, r), atol=1e-12)


def test_tables_reproduce_fields_at_nodes_and_integer_wrap():
    f, cfg, d = _field()
    F = DesignFields(d)
    i = np.array([0, 17, 1000, 2047])
    j = np.array([0, 5, 128, 256])
    th = (i * f.d_theta + f.seam) % (2 * math.pi)
    z = j * f.d_z
    ref = F.evaluate(th, z)
    got = f.fields_at(th, z)
    for k in ("rho", "w", "L"):
        assert np.allclose(got[k], ref[k], atol=1e-12)
    # integer winding: u_t(theta' = 2 pi) - u_t(0) = N exactly, at every z
    Ut = f.tab["U_t"]
    assert np.allclose(Ut[-1] - Ut[0], f.N_theta, atol=1e-9)
    # axial phase origin at mid-length
    jm = (f.n_z - 1) // 2
    assert np.allclose(f.tab["U_z"][:, jm], 0.0, atol=1e-12)


def test_lattice_continuous_across_seam_and_level_set_definition():
    """Integer mode: no tear at theta = 0; phi equals tpms.level_set at the phase coordinates."""
    from voxlat.geometry.tpms import level_set

    f, _, _ = _field()
    rng = np.random.default_rng(2)
    n = 400
    r = f.r_li + (f.r_lo - f.r_li) * rng.random(n)
    z = 50 * rng.random(n)
    eps = 1e-7
    a = f.lattice_level(r * np.cos(eps), r * np.sin(eps), z)
    b = f.lattice_level(r * np.cos(-eps), r * np.sin(-eps), z)
    assert np.max(np.abs(a["phi"] - b["phi"])) < 1e-4
    # definition of phi (blend weight is per point -> compare point by point)
    for k in range(20):
        ref = level_set(a["u_r"][k], a["u_t"][k], a["u_z"][k], float(a["w"][k]))
        assert abs(float(ref) - a["phi"][k]) < 1e-12


def test_conformal_radial_phase():
    f, _, _ = _field(UNIFORM, sector_center_deg=0.0)
    L = float(f.tab["L"][0, 0])
    z = np.full(3, 25.0)
    r = np.array([f.r_li, f.r_m, f.r_lo])
    lv = f.lattice_level(r, 0 * r, z)
    assert abs(lv["u_r"][0]) < 1e-12  # cut phase at the sleeve = phase_r0 = 0 (Task 7 convention)
    assert np.isclose(lv["u_r"][2], f.r_m / L * math.log(f.r_lo / f.r_li))


def test_seam_mode_rib_only_when_seam_not_in_slot():
    f, _, _ = _field(BEST, circumferential="seam")
    assert f.circ_mode == "seam" and not f.rib  # seam hidden in the inlet slot
    f2, _, _ = _field(RINGS_KNEE, circumferential="seam")
    assert f2.rib
    f3, _, _ = _field(RINGS_KNEE)
    assert f3.circ_mode == "integer" and not f3.rib
    D = f3.describe()
    assert D["circumferential_stretch_max"] > 1.3  # axially graded L -> stretched cells (documented)


def test_signed_distance_geometry_landmarks():
    f, cfg, _ = _field(BEST, mode="full")
    g = cfg["geometry"]
    sd = lambda x, y, z: float(f.signed_distance(np.array([x]), np.array([y]), np.array([z]))[0])  # noqa: E731
    r_sleeve = 0.5 * (g["r_stator_mm"] + g["r_lattice_in_mm"])
    r_wall = 0.5 * (g["r_lattice_out_mm"] + g["r_outer_mm"])
    assert sd(0.0, r_sleeve, 25.0) < 0  # sleeve
    assert sd(0.0, -r_wall, 10.0) < 0  # outer wall
    assert sd(-60.0, 0.0, 25.0) > 0  # motor bore
    assert sd(0.0, 0.5 * (g["r_stator_mm"] + g["r_outer_mm"]), -1.0) < 0  # end cap
    # inlet slot plenum (theta = 0) is open; the port bore is open through the outer wall
    assert sd(g["r_mean_mm"], 0.0, 10.0) > 0
    assert sd(r_wall, 0.0, 25.0) > 0
    assert sd(r_wall, 0.0, 10.0) < 0  # outer wall outside the bore
    # boss around the bore
    p = g["ports"][0]
    assert sd(g["r_outer_mm"] + 2.0, 0.5 * (p["bore_radius_mm"] + p["boss_radius_mm"]), p["z_mm"]) < 0


def test_wedge_and_sector_frame():
    f, cfg, _ = _field(BEST, sector_center_deg=90.0, sector_angle_deg=30.0)
    # local +x maps to world theta = 90 deg
    xw, yw = f.to_world(np.array([74.5]), np.array([0.0]))
    assert np.isclose(xw[0], 0.0, atol=1e-12) and np.isclose(yw[0], 74.5)
    # outside the nominal wedge -> empty
    a = math.radians(20.0)
    assert f.signed_distance(np.array([74.5 * math.cos(a)]), np.array([74.5 * math.sin(a)]), np.array([25.0]))[0] > 0


def test_critical_location_is_most_constrained_cell():
    d = load_design(BEST)
    c = critical_location(d)
    assert 0.9 < c["L_over_Lmin"] < 1.5
    assert 0 <= c["theta_deg"] < 360


# =============================================================================
# sector generator / cross-check
# =============================================================================
@pytest.mark.parametrize("w,rho,L", [(0.0, 0.25, 0.004), (1.0, 0.4, 0.003)])
def test_uniform_sector_volume_fraction(w, rho, L):
    d = _uniform_design(w, rho, L)
    d["_path"] = str(UNIFORM)
    cfg = build_export_config(d, ExportOptions(sector_center_deg=90.0, sector_angle_deg=6.0, voxel_size_mm=0.1))
    f = JacketField(cfg, d)
    m = generate_mask(f)
    r = volume_fraction_report(f, m)
    assert abs(r["relative_error_vs_phase_aware"]) < 0.01
    assert abs(r["relative_error"]) < 0.03


def test_planar_profile_table_means_equal_rho():
    wg, rg, tab = planar_profile_table()
    j = int(np.argmin(np.abs(rg - 0.3)))
    assert np.allclose(tab[:, j].mean(axis=1), 0.3, atol=0.01)


def test_compare_masks_alignment():
    rng = np.random.default_rng(3)
    a = VoxelMask(rng.random((20, 20, 20)) > 0.5, (1.0, 2.0, 3.0), 0.5)
    sub = a.solid[2:, 3:, 1:]
    b = VoxelMask(sub.copy(), (2.0, 3.5, 3.5), 0.5)
    c = compare_masks(a, b)
    assert c["iou"] == 1.0 and c["mismatch_voxels"] == 0


# =============================================================================
# checks
# =============================================================================
H = 0.05


@pytest.mark.parametrize("n", [5, 6, 7, 8, 12])
def test_min_wall_plates(n):
    s = np.zeros((24, 24, 40), bool)
    s[:, :, 10:10 + n] = True
    t = n * H
    r = min_wall(s, H, np.ones_like(s), 0.35)
    assert t - 0.75 * H - 1e-9 <= r["min_wall_mm"] <= t + 0.75 * H + 1e-9


@pytest.mark.parametrize("R", [0.2, 0.3, 0.45])
def test_min_wall_rods_and_monotonicity(R):
    x = np.arange(60) * H
    X, Y = np.meshgrid(x, x, indexing="ij")
    s = np.repeat((((X - 1.5) ** 2 + (Y - 1.5) ** 2) <= R * R)[:, :, None], 16, axis=2)
    r = min_wall(s, H, np.ones_like(s), 0.35)
    assert 2 * R - 0.75 * H - 1e-9 <= r["min_wall_mm"] <= 2 * R + 0.75 * H + 1e-9
    D = capped_edt(s, H, 1.0)
    fails = [bool(thin_mask(s, H, d, D).any()) for d in np.arange(0.1, 1.2, 0.025)]
    assert all((not a) or b for a, b in zip(fails, fails[1:]))  # monotone in d


def test_min_wall_finds_neck():
    x = np.arange(60) * H
    X, Y = np.meshgrid(x, x, indexing="ij")
    rr = (X - 1.5) ** 2 + (Y - 1.5) ** 2
    s = np.zeros((60, 60, 60), bool)
    s[:, :, :27] = (rr <= 0.16)[:, :, None]
    s[:, :, 33:] = (rr <= 0.16)[:, :, None]
    s[:, :, 27:33] = (rr <= 0.15**2)[:, :, None]
    ev = np.zeros_like(s)
    ev[:, :, 20:40] = True
    r = min_wall(s, H, ev, 0.35)
    assert r["thin_voxels_at_limit"] > 0 and r["min_wall_mm"] < 0.35


def test_throat_of_ducts():
    for wv in (8, 12, 16):
        s = np.ones((30, 30, 40), bool)
        s[8:8 + wv, 8:8 + wv, :] = False
        a = np.zeros_like(s)
        a[:, :, 0] = True
        b = np.zeros_like(s)
        b[:, :, -1] = True
        thr = throat(~s, capped_edt(~s, H, 2.0), H, a, b)
        assert wv * H - 0.25 * H - 1e-9 <= thr <= wv * H + 0.5 * H + 1e-9


@pytest.mark.parametrize("w,rho", [(0.0, 0.35), (1.0, 0.3), (1.0, 0.5)])
def test_throat_matches_task1_table_on_periodic_cells(w, rho):
    from voxlat.geometry.tpms import TPMSParams, load_morphology_table, voxelize

    n = 48
    S = np.tile(voxelize(TPMSParams(w=w, rho=rho), n=n), (3, 3, 4))
    h = 1.0 / n
    a = np.zeros_like(S)
    a[:, :, 0] = True
    b = np.zeros_like(S)
    b[:, :, -1] = True
    t = throat(~S, capped_edt(~S, h, 1.0), h, a, b, d_max=1.0)
    tab = load_morphology_table()
    ref = tab.data["pore_throat"][int(np.argmin(abs(tab.w - w))), int(np.argmin(abs(tab.rho - rho)))]
    assert abs(t - ref) <= 1.25 * h


def test_capped_edt_equals_scipy():
    rng = np.random.default_rng(4)
    m = ndimage.binary_opening(rng.random((30, 31, 200)) > 0.3)
    ref = ndimage.distance_transform_edt(m, sampling=0.1)
    got = capped_edt(m, 0.1, 0.6, slab=37)
    assert np.allclose(got, np.minimum(ref, 0.6), atol=1e-5)


def test_open_mask_removes_thin_plate_keeps_thick():
    s = np.zeros((30, 30, 60), bool)
    s[:, :, 5:9] = True  # 0.2 mm plate
    s[:, :, 20:40] = True  # 1 mm plate
    o = open_mask(VoxelMask(s, (0, 0, 0), H), 0.175)
    assert not o.solid[:, :, 5:9].any()
    assert o.solid[:, :, 21:39].all()


def test_open_mask_band_limited_to_open_faces():
    """End rings: the open faces are z = 0 and z = L_ax; a thin plate near z = 0 is removed, one at
    mid-length (a designed thin wall) is kept for the checks to report."""
    f, cfg, _ = _field(RINGS_KNEE, sector_center_deg=0.0)
    h = 0.05
    s = np.zeros((20, 20, 700), bool)  # z from -5 to 30 mm
    origin = (74.0, -0.5, -5.0)
    kz = lambda zz: int(round((zz - origin[2]) / h))  # noqa: E731
    s[:, :, kz(0.1):kz(0.1) + 4] = True  # 0.2 mm plate inside the band at z = 0
    s[:, :, kz(25.0):kz(25.0) + 4] = True  # 0.2 mm plate at mid-length
    m = VoxelMask(s, origin, h)
    o = open_mask(m, 0.175, field=f, band=0.45)
    assert not o.solid[:, :, kz(0.1):kz(0.1) + 4].any()
    assert o.solid[:, :, kz(25.0):kz(25.0) + 4].all()


@pytest.fixture(scope="module")
def small_sector():
    f, cfg, d = _field(BEST, sector_center_deg=0.0, sector_angle_deg=8.0, voxel_size_mm=0.12)
    m = generate_mask(f)
    return f, m


def test_check_mask_on_generated_sector(small_sector):
    f, m = small_sector
    m.frame["half_angle_deg"] = math.degrees(wide_half_angle(f))
    clean = open_mask(m, 0.175, field=f, band=0.35 + 2 * m.voxel)
    r = check_mask(clean, f)
    assert r["connected"]
    assert r["paths"][0]["from"] == "inlet"  # the inlet slot plenum and bore are in this sector
    assert r["trapped_pockets"]["count"] == 0
    assert r["loose_solid"]["count"] == 0
    assert r["min_throat_mm"] > 0.8
    json.dumps(r)  # serializable


def test_check_mask_detects_pocket_and_loose_solid():
    f, cfg, d = _field(BEST, sector_center_deg=90.0, sector_angle_deg=6.0, voxel_size_mm=0.12)
    m = generate_mask(f)
    s = m.solid.copy()
    # a hollow ball in the middle of the gap (lattice region, local frame) ...
    x, y, z = m.centres()
    X, Y, Z = np.broadcast_arrays(x, y, z)
    c = np.array([74.5, 0.0, 25.0])
    d2 = (X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2
    s[d2 <= 1.0**2] = True
    s[d2 <= 0.6**2] = False  # enclosed cavity
    c2 = c + np.array([0.0, 0.0, 6.0])
    d3 = (X - c2[0]) ** 2 + (Y - c2[1]) ** 2 + (Z - c2[2]) ** 2
    s[d3 <= 2.0**2] = False  # clear space ...
    s[d3 <= 0.5**2] = True  # ... and a floating ball in it
    r = check_mask(VoxelMask(s, m.origin, m.voxel, frame=m.frame), f)
    assert r["trapped_pockets"]["count"] >= 1
    assert r["loose_solid"]["count"] >= 1
    assert not r["all_pass"]


# =============================================================================
# IO
# =============================================================================
def test_mask_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    m = VoxelMask(rng.random((13, 7, 5)) > 0.5, (0.1, -2.0, 3.0), 0.05, frame={"mode": "sector"}, meta={"a": 1})
    write_mask(m, tmp_path / "m")
    r = read_mask(tmp_path / "m.json")
    assert np.array_equal(r.solid, m.solid) and r.origin == m.origin and r.voxel == m.voxel


def test_legacy_picogk_mask_rows_are_flipped(tmp_path):
    """Masks written before the fix kept PicoGK's slice rows (y descending); read_mask restores them."""
    rng = np.random.default_rng(6)
    s = rng.random((9, 7, 4)) > 0.5
    legacy = VoxelMask(s[:, ::-1, :].copy(), (0.0, 0.0, 0.0), 0.05, source="picogk")
    write_mask(legacy, tmp_path / "old")
    h = json.loads((tmp_path / "old.json").read_text())
    h.pop("y_order")
    (tmp_path / "old.json").write_text(json.dumps(h))
    assert np.array_equal(read_mask(tmp_path / "old.json").solid, s)
    new = VoxelMask(s, (0.0, 0.0, 0.0), 0.05, source="picogk")
    write_mask(new, tmp_path / "new")
    assert np.array_equal(read_mask(tmp_path / "new.json").solid, s)


def test_crop_to_wedge_and_quick_iou():
    from voxlat.export.sector import crop_to_wedge, quick_slice_iou

    f, cfg, d = _field(BEST, sector_center_deg=90.0, sector_angle_deg=10.0, voxel_size_mm=0.15)
    m = generate_mask(f)
    assert quick_slice_iou(f, m)["iou_min"] > 0.999
    flipped = VoxelMask(m.solid[:, ::-1, :].copy(), m.origin, m.voxel, frame=m.frame)
    assert quick_slice_iou(f, flipped)["iou_min"] < 0.9  # a mis-registered mask is caught
    c = crop_to_wedge(m, math.radians(3.0), f.r_o + 7.0)
    assert c.shape[1] < m.shape[1] and c.frame["half_angle_deg"] == pytest.approx(3.0)


def test_stl_roundtrip_and_voxelize(tmp_path):
    from skimage.measure import marching_cubes

    g = np.linspace(-3, 3, 61)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    v, fa, _, _ = marching_cubes(np.sqrt(X**2 + Y**2 + Z**2) - 2.0, 0.0, spacing=(0.1,) * 3)
    v -= 3.0
    write_stl(tmp_path / "s.stl", v[fa])
    tris = read_stl(tmp_path / "s.stl")
    m = voxelize_stl(tris, 0.05)
    assert abs(m.volume_mm3() / (4 / 3 * math.pi * 8) - 1) < 0.01


# =============================================================================
# C# source guards (the C# library cannot be compiled in every environment)
# =============================================================================
def test_csharp_reads_only_keys_that_python_writes():
    keys = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                keys.add(k)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for path in (BEST, RINGS_KNEE):
        walk(build_export_config(load_design(path), ExportOptions(sector_center_deg=0.0)))
        walk(json.loads(path.read_text()))
    src = "".join(p.read_text() for p in (DOTNET / "VoxLat.Export.Core").glob("*.cs"))
    used = set(re.findall(r'GetProperty\("([^"]+)"\)', src)) | set(re.findall(r'TryGetProperty\("([^"]+)"', src))
    missing = sorted(u for u in used if u not in keys)
    assert not missing, f"C# reads keys Python does not write: {missing}"


def test_csharp_constants_match_python():
    import voxlat.export.field as fld

    src = (DOTNET / "VoxLat.Export.Core" / "JacketField.cs").read_text()
    assert "GradFloor = 0.25" in src and fld._GRAD_FLOOR == 0.25
    assert "GAmp = 1.5" in src and fld.G_AMP == 1.5
    assert "Math.Sqrt(2.0)" in src
    for name in ("GradedTpmsJacket.cs", "JacketBuilder.cs"):
        assert (DOTNET / "VoxLat.Export" / name).is_file()
