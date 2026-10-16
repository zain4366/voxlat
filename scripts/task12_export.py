"""Task 12 - PicoGK geometry export of a Task 11 design, manufacturability checks, numpy cross-check.

Sub-commands (run ``python scripts/task12_export.py <cmd> -h``):

  build     PicoGK (C#) export: full jacket or a 30-degree sector -> STL + voxel masks + report, then the
            Python checks on the PicoGK masks and (sector) a voxel-by-voxel comparison with the numpy mirror.
  numpy     Pure-numpy sector generator (no .NET): mask -> volume-fraction cross-check -> checks.
  check     Run the checks on an existing mask (``*_mask.json``) or STL (re-voxelized).
  selftest  Evaluate the C# implicit (pythonnet or the probe exe) and the numpy mirror at the same random
            points; they must agree to 1e-6 mm. Needs only the built C# library, not PicoGK.
  estimate  Voxel counts / memory / rough run times for the chosen options.
  config    Write the export-config JSON only.

Examples (Windows, from the repo root, venv active):
  python scripts/task12_export.py build --design results/designs/task11_slots_best_at_cap.json --mode sector
  python scripts/task12_export.py build --mode full --voxel 0.2 --headless
  python scripts/task12_export.py numpy --angle 30 --voxel 0.05
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voxlat.export.config import DEFAULT_DESIGN, ExportOptions, build_export_config, load_design, write_export_config  # noqa: E402
from voxlat.export.checks import _trim  # noqa: E402
from voxlat.export.field import JacketField  # noqa: E402


# =============================================================================
def _options(a: argparse.Namespace, mode: str) -> ExportOptions:
    centre: float | str = "auto" if str(a.center).lower() == "auto" else float(a.center)
    return ExportOptions(
        mode=mode,
        voxel_size_mm=a.voxel,
        sector_center_deg=centre,
        sector_angle_deg=a.angle,
        sector_margin_mm=a.margin,
        n_theta_cells=a.n_theta,
        phase_r0=a.phase_r,
        circumferential=a.circumferential,
        sliver_radius_mm=a.sliver,
        sliver_band_mm=a.sliver_band,
        save_stl=not getattr(a, "no_stl", False),
        write_raw_mask=not getattr(a, "no_raw_mask", False),
        show_in_viewer=not getattr(a, "headless", False),
        output_dir=str(Path(a.out).resolve()),
        name=a.name,
    )


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--design", default=str(ROOT / DEFAULT_DESIGN), help="Task 11 design JSON")
    p.add_argument("--voxel", type=float, default=None, help="voxel size mm (default full 0.2, sector 0.05)")
    p.add_argument("--angle", type=float, default=30.0, help="sector angle, deg")
    p.add_argument("--center", default="auto", help="sector centre angle, deg, or 'auto' (most critical cell)")
    p.add_argument("--margin", type=float, default=2.0, help="extra sector margin rendered on each side, mm")
    p.add_argument("--n-theta", type=int, default=None, help="cells around (default: round(mean))")
    p.add_argument("--phase-r", type=float, default=0.0, help="radial cut phase at the sleeve, cell units")
    p.add_argument("--circumferential", choices=("integer", "seam"), default="integer")
    p.add_argument("--sliver", type=float, default=None, help="sliver opening radius mm (default min_wall/2; 0 = off)")
    p.add_argument("--sliver-band", type=float, default=None,
                   help="opening only within this distance (mm) of open lattice faces (default min_wall + 2 voxels; 0 = whole part)")
    p.add_argument("--out", default=str(ROOT / "results" / "export"), help="output folder")
    p.add_argument("--name", default=None, help="output file stem")


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _summary(checks: dict) -> str:
    w = checks.get("wall", {})
    lines = [
        f"  min wall      : {w.get('min_wall_mm')} mm (limit {checks['limits']['min_wall_mm']}; "
        f"thin voxels {w.get('thin_voxels_at_limit')}; resolved={checks['wall_resolved']})",
        f"  min throat    : {checks.get('min_throat_mm')} mm (limit {checks['limits']['min_pore_mm']})",
        f"  connected     : {checks['connected']}  paths: "
        + ", ".join(f"{p['from']}->{p['to']} {p['throat_mm']:.2f} mm" for p in checks["paths"]),
        f"  trapped       : {checks['trapped_pockets']['count']} pockets ({checks['trapped_pockets']['volume_mm3']:.3f} mm^3); "
        f"voxel-scale {checks['trapped_pockets']['voxel_scale_count']} "
        f"(< {checks['trapped_pockets']['resolution_diameter_mm']:.2f} mm, {checks['trapped_pockets']['voxel_scale_volume_mm3']:.4f} mm^3)",
        f"  loose solid   : {checks['loose_solid']['count']} ({checks['loose_solid']['volume_mm3']:.3f} mm^3)",
        f"  interface slivers (raw, near open faces): min {checks['interface_slivers']['min_wall_mm']} mm",
        f"  PASS          : {checks['pass']}  -> all_pass = {checks['all_pass']}",
    ]
    return "\n".join(lines)


BYTES_PER_VOXEL = 24.0  # measured peak of the checks (sandbox, 2.7e8 voxels: 6.1 GB)


def estimate(cfg: dict, field: JacketField) -> dict:
    from voxlat.export.sector import render_bbox, wide_half_angle

    h = cfg["render"]["voxel_size_mm"]
    lo, hi = render_bbox(field, wide_half_angle(field) if field.mode == "sector" else None)
    n = np.prod(np.ceil((hi - lo) / h) + 1)
    # surface area ~ a_sf * lattice volume (a_sf L ~ 3) + housing
    ang = math.radians(cfg["render"]["sector_angle_deg"]) if field.mode == "sector" else 2 * math.pi
    vol_lat = 0.5 * ang * (field.r_lo**2 - field.r_li**2) * field.L_ax
    Lm = float(np.mean(field.tab["L"]))
    area = 3.0 / Lm * vol_lat * 1.15
    tris = 4.2 * area / h**2  # calibrated on the first PicoGK sector (20.3 M triangles at 0.05 mm)
    return {"voxels_bbox": float(n), "render_minutes_at_0.3us": float(n) * 0.3e-6 / 60.0,
            "mask_MB": float(n) / 8 / 1e6, "triangles_est": float(tris), "stl_MB_est": tris * 50 / 1e6,
            "python_checks_GB_peak_est": float(n) * BYTES_PER_VOXEL / 1e9}


# =============================================================================
def cmd_config(a):
    d = load_design(a.design)
    cfg = build_export_config(d, _options(a, a.mode))
    out = Path(a.out) / f"{cfg['render']['name']}_config.json"
    write_export_config(cfg, out)
    print(out)


def cmd_estimate(a):
    d = load_design(a.design)
    cfg = build_export_config(d, _options(a, a.mode))
    f = JacketField(cfg, d)
    print(json.dumps({"field": f.describe(), "estimate": estimate(cfg, f), "critical": cfg["critical_location"]}, indent=1))


def _check_field(cfg: dict, design: dict, angle_deg: float) -> JacketField:
    """Same jacket with a narrower nominal sector (memory-bounded checks)."""
    c2 = json.loads(json.dumps(cfg))
    c2["render"]["sector_angle_deg"] = float(angle_deg)
    return JacketField(c2, design)


def _choose_check_angle(cfg: dict, mask_voxels: float, requested: float | None) -> float:
    from voxlat.export.checks import available_ram_bytes

    full = float(cfg["render"]["sector_angle_deg"])
    if requested is not None:
        return min(float(requested), full)
    avail = available_ram_bytes()
    if avail is None:
        return full
    need = BYTES_PER_VOXEL * mask_voxels
    if need <= 0.75 * avail:
        return full
    ang = max(6.0, math.floor(full * 0.75 * avail / need))
    print(f"note: the full {full:g} deg checks need ~{need / 1e9:.1f} GB, {avail / 1e9:.1f} GB available -> checking the "
          f"central {ang:g} deg (the critical cell is at the sector centre). Override with --check-angle.")
    return min(ang, full)


def _post_checks(cfg: dict, field: JacketField, clean_mask, raw_mask, out_stem: Path, compare_numpy: bool,
                 verbose: bool = True, design: dict | None = None, check_angle: float | None = None,
                 sanity: bool = True) -> dict:
    """Volume fraction + raw interior walls on ``raw_mask``; all checks on ``clean_mask``. Masks may be paths
    (read lazily, one at a time, to bound memory) or VoxelMask objects."""
    from voxlat.export.checks import check_mask, min_wall, RegionGrid
    from voxlat.export.maskio import read_mask
    from voxlat.export.sector import compare_masks, crop_to_wedge, generate_mask, quick_slice_iou, volume_fraction_report

    res: dict = {}
    fcheck = field
    half_crop = None

    def prepare(m):
        nonlocal fcheck, half_crop
        if isinstance(m, (str, Path)):
            m = read_mask(m)
        if field.mode != "sector":
            return m
        if sanity:
            q = quick_slice_iou(field, m)
            res.setdefault("sanity", {})[m.kind] = q
            print(f"sanity ({m.source} {m.kind} mask vs numpy implicit, 3 slices): IoU {', '.join(f'{v:.4f}' for v in q['iou'])}")
            if q["iou_min"] < 0.9:
                raise SystemExit(f"ABORT: the {m.kind} mask does not match the geometry (IoU {q['iou_min']:.3f} < 0.9) - "
                                 "the mask is mis-registered; send this log.")
        if half_crop is None:
            ang = _choose_check_angle(cfg, float(np.prod(m.shape)), check_angle)
            if design is not None and ang < float(cfg["render"]["sector_angle_deg"]) - 1e-9:
                fcheck = _check_field(cfg, design, ang)
            half_crop = math.radians(0.5 * (ang if fcheck is not field else float(cfg["render"]["sector_angle_deg"])))
            half_crop += float(cfg["render"]["sector_margin_mm"]) / field.r_li
            res["checked_angle_deg"] = ang if fcheck is not field else float(cfg["render"]["sector_angle_deg"])
        if fcheck is not field:
            top = field.r_o + max(float(p["boss_height_mm"]) for p in field.ports) + 1.0
            m = crop_to_wedge(m, half_crop, top)
        return m

    if raw_mask is not None and field.mode == "sector":
        raw = prepare(raw_mask)
        t0 = time.perf_counter()
        res["volume_fraction_raw"] = volume_fraction_report(fcheck, raw)
        v = res["volume_fraction_raw"]
        print(f"volume fraction (raw): solid {v['solid_fraction']:.4f} vs design rho {v['target_rho_mean']:.4f} "
              f"({100 * v['relative_error']:+.2f} %), vs phase-aware {100 * v['relative_error_vs_phase_aware']:+.2f} % "
              f"[{time.perf_counter() - t0:.0f} s]")
        band = cfg["manufacturing"]["min_wall_mm"] + 2 * raw.voxel
        R = RegionGrid(fcheck, raw, band)
        ev = R.lattice & R.nominal & ~R.iface
        del R
        _trim()
        res["wall_raw_interior"] = min_wall(raw.solid, raw.voxel, ev, cfg["manufacturing"]["min_wall_mm"])
        del ev
        print(f"raw lattice interior min wall: {res['wall_raw_interior'].get('min_wall_mm')} mm")
        _trim()
        if compare_numpy:
            t0 = time.perf_counter()
            num = generate_mask(fcheck, raw.voxel, half_angle=half_crop)
            cmp = compare_masks(num, raw)
            res["numpy_vs_picogk_raw"] = cmp
            print(f"numpy vs PicoGK raw mask: IoU {cmp['iou']:.5f}, volume diff {100 * cmp['volume_rel_diff']:+.3f} %, "
                  f"mismatched voxels {cmp['mismatch_voxels']} [{time.perf_counter() - t0:.0f} s]")
            del num
        del raw
        _trim()
    if clean_mask is not None:
        clean = prepare(clean_mask)
        chk = check_mask(clean, fcheck, verbose=verbose)
        res["checks"] = chk
        print("checks on the printed (cleaned) geometry:")
        print(_summary(chk))
        del clean
        _trim()
    _write(out_stem.with_name(out_stem.name + "_checks.json"), res)
    return res


def cmd_build(a):
    from voxlat.export.maskio import read_mask
    from voxlat.export.picogk import run_picogk

    d = load_design(a.design)
    opts = _options(a, a.mode)
    cfg = build_export_config(d, opts)
    out = Path(cfg["render"]["output_dir"])
    stem = out / cfg["render"]["name"]
    cfg_path = write_export_config(cfg, stem.with_name(stem.name + "_config.json"))
    field = JacketField(cfg, d)
    print(f"design {Path(a.design).name}: layout {d['layout']}, mode {a.mode}, voxel {cfg['render']['voxel_size_mm']} mm"
          + (f", sector {cfg['render']['sector_angle_deg']} deg at {cfg['render']['sector_center_deg']:.2f} deg" if a.mode == "sector" else ""))
    print("phase field:", json.dumps({k: field.describe()[k] for k in ("N_theta", "circumferential_stretch_min",
                                                                       "circumferential_stretch_max", "axial_phase_shear_deg_max")}))
    print("estimate:", json.dumps(estimate(cfg, field)))
    report_path = stem.with_name(stem.name + "_report.json")
    if a.reuse and report_path.is_file():
        rep = json.loads(report_path.read_text())
        print(f"--reuse: using the existing PicoGK outputs ({report_path.name})")
    else:
        t0 = time.perf_counter()
        rep = run_picogk(cfg_path, picogk_dir=a.picogk_dir, build_dir=a.build_dir, viewer=not a.headless,
                         keep_viewer_open=a.keep_viewer)
        print(f"PicoGK done in {time.perf_counter() - t0:.0f} s: STL {rep['stl']} ({rep['triangles']} triangles), "
              f"volume raw {rep['volume_raw_mm3']:.1f} / clean {rep['volume_clean_mm3']:.1f} mm^3")
    if a.no_checks:
        return
    res = _post_checks(cfg, field, rep.get("mask") or None, rep.get("mask_raw") or None, stem,
                       compare_numpy=not a.no_numpy and a.mode == "sector", design=d, check_angle=a.check_angle)
    res["picogk_report"] = rep
    _write(stem.with_name(stem.name + "_checks.json"), res)


def cmd_numpy(a):
    from voxlat.export.maskio import write_mask
    from voxlat.export.sector import generate_mask, open_mask

    t_start = time.perf_counter()
    d = load_design(a.design)
    opts = _options(a, "sector")
    cfg = build_export_config(d, opts)
    out = Path(cfg["render"]["output_dir"])
    stem = out / (cfg["render"]["name"] + "_numpy")
    write_export_config(cfg, stem.with_name(stem.name + "_config.json"))
    field = JacketField(cfg, d)
    print(f"numpy sector: {Path(a.design).name}, {cfg['render']['sector_angle_deg']} deg at "
          f"{cfg['render']['sector_center_deg']:.2f} deg, voxel {cfg['render']['voxel_size_mm']} mm")
    t0 = time.perf_counter()
    raw = generate_mask(field, progress=True)
    t_gen = time.perf_counter() - t0
    print(f"  generated {raw.shape} in {t_gen:.0f} s, solid volume {raw.volume_mm3():.1f} mm^3")
    if a.write_mask:
        write_mask(raw, stem.with_name(stem.name + "_mask_raw"))
    res = _post_checks(cfg, field, None, raw, stem, compare_numpy=False, design=d)  # raw: vf + interior walls
    t0 = time.perf_counter()
    clean = open_mask(raw, cfg["render"]["sliver_radius_mm"], field=field, band=cfg["render"]["sliver_band_mm"])
    vol_raw = raw.volume_mm3()
    del raw
    _trim()
    print(f"  sliver opening r = {cfg['render']['sliver_radius_mm']} mm: {vol_raw:.1f} -> {clean.volume_mm3():.1f} mm^3 "
          f"[{time.perf_counter() - t0:.0f} s]")
    if a.write_mask:
        write_mask(clean, stem.with_name(stem.name + "_mask"))
    if not a.no_checks:
        res.update(_post_checks(cfg, field, clean, None, stem, compare_numpy=False, design=d))
    res["field"] = field.describe()
    res["generation_s"] = t_gen
    res["volume_raw_mm3"] = vol_raw
    res["volume_clean_mm3"] = clean.volume_mm3()
    res["config"] = {k: cfg[k] for k in ("design_label", "layout", "render", "critical_location")}
    res["total_s"] = time.perf_counter() - t_start
    try:
        import resource

        res["peak_rss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # Linux: kB
    except ImportError:  # Windows
        pass
    print(f"  total {res['total_s']:.0f} s" + (f", peak memory {res['peak_rss_GB']:.2f} GB" if "peak_rss_GB" in res else ""))
    _write(stem.with_name(stem.name + "_checks.json"), res)
    if a.stl:
        from skimage.measure import marching_cubes

        from voxlat.export.stl import write_stl

        pad = np.pad(clean.solid, 1).astype(np.float32)
        v, f_, _, _ = marching_cubes(pad, 0.5, spacing=(clean.voxel,) * 3)
        v += np.asarray(clean.origin) - clean.voxel
        write_stl(stem.with_suffix(".stl"), v[f_])
        print(f"  numpy STL: {stem.with_suffix('.stl')} ({len(f_)} triangles)")


def cmd_check(a):
    from voxlat.export.checks import check_mask
    from voxlat.export.maskio import read_mask
    from voxlat.export.stl import voxelize_stl

    cfg = json.loads(Path(a.config).read_text())
    field = JacketField(cfg)
    if a.mask:
        m = read_mask(a.mask)
    else:
        h = a.voxel or cfg["render"]["voxel_size_mm"]
        frame = {"mode": cfg["render"]["mode"], "half_angle_deg": 0.5 * cfg["render"]["sector_angle_deg"]}
        m = voxelize_stl(a.stl, h, frame=frame)
        print(f"re-voxelized {a.stl}: {m.shape} voxels of {h} mm")
    chk = check_mask(m, field, verbose=True)
    print(_summary(chk))
    if a.report:
        _write(Path(a.report), chk)


def cmd_selftest(a):
    from voxlat.export.picogk import probe, probe_fields
    from voxlat.export.sector import render_bbox, wide_half_angle

    d = load_design(a.design)
    rng = np.random.default_rng(12)
    worst = 0.0
    ok = True
    for mode in ("sector", "full"):
        opts = _options(a, mode)
        opts.voxel_size_mm = 0.1
        cfg = build_export_config(d, opts)
        cfg_path = write_export_config(cfg, Path(a.out) / f"selftest_{mode}_config.json")
        f = JacketField(cfg, d)
        lo, hi = render_bbox(f, wide_half_angle(f) if mode == "sector" else None)
        n = a.n
        p1 = lo + (hi - lo) * rng.random((n, 3))
        # points concentrated in the lattice gap (render frame)
        th = (rng.random(n) - 0.5) * 2 * (f.sector_half if mode == "sector" else math.pi)
        r = f.r_li - 0.3 + (f.r_lo - f.r_li + 0.6) * rng.random(n)
        z = -1.0 + (f.L_ax + 2.0) * rng.random(n)
        p2 = np.column_stack([r * np.cos(th), r * np.sin(th), z])
        P = np.vstack([p1, p2])
        t0 = time.perf_counter()
        sd_cs, diag = probe(cfg_path, P, method=a.method)
        t_cs = time.perf_counter() - t0
        t0 = time.perf_counter()
        sd_np = f.signed_distance(P[:, 0], P[:, 1], P[:, 2])
        t_np = time.perf_counter() - t0
        diff = np.abs(sd_cs - sd_np)
        sign = int(np.count_nonzero((sd_cs <= 0) != (sd_np <= 0)))
        # tabulated fields at random (theta, z)
        tt = rng.random(2000) * 2 * math.pi
        zz = rng.random(2000) * f.L_ax
        Fc = probe_fields(cfg_path, tt, zz, method=a.method)
        Fn = f.fields_at(tt, zz)
        fd = max(float(np.max(np.abs(Fc[:, k] - Fn[n_]))) for k, n_ in enumerate(("rho", "w", "L", "c")))
        same_N = diag["N_theta"] == f.N_theta
        print(f"[{mode}] {len(P)} points: max |sd_C# - sd_numpy| = {diff.max():.2e} mm, sign mismatches {sign}, "
              f"fields max diff {fd:.2e}, N_theta C# {diag['N_theta']} / numpy {f.N_theta}; "
              f"time C# {t_cs:.1f} s (incl. startup), numpy {t_np:.1f} s")
        worst = max(worst, float(diff.max()), fd)
        ok &= same_N and sign <= max(1, len(P) // 100000)
    ok &= worst < 1e-6
    print("SELFTEST", "PASSED" if ok else "FAILED", f"(worst difference {worst:.2e})")
    sys.exit(0 if ok else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="PicoGK export + checks")
    _common(p)
    p.add_argument("--mode", choices=("sector", "full"), default="sector")
    p.add_argument("--picogk-dir", default=r"D:\nyron 2\PicoGK_Engine\bin\Release\net9.0")
    p.add_argument("--build-dir", default=None, help="folder of VoxLat.Export.dll (default: the dotnet build output)")
    p.add_argument("--headless", action="store_true", help="no viewer window (PicoGK Library without Go)")
    p.add_argument("--keep-viewer", action="store_true", help="keep the viewer open after the build (close it to continue)")
    p.add_argument("--no-stl", action="store_true")
    p.add_argument("--no-raw-mask", action="store_true")
    p.add_argument("--no-checks", action="store_true")
    p.add_argument("--no-numpy", action="store_true", help="skip the numpy voxel-by-voxel comparison")
    p.add_argument("--reuse", action="store_true", help="skip PicoGK if <name>_report.json exists; only rerun the checks")
    p.add_argument("--check-angle", type=float, default=None,
                   help="check only the central part of the sector (deg); default: as much as the free RAM allows")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("numpy", help="pure-numpy sector + checks")
    _common(p)
    p.add_argument("--write-mask", action="store_true")
    p.add_argument("--no-checks", action="store_true")
    p.add_argument("--stl", action="store_true", help="also write an STL of the numpy sector (marching cubes)")
    p.set_defaults(func=cmd_numpy)

    p = sub.add_parser("check", help="checks on a mask or STL")
    p.add_argument("--config", required=True, help="export config JSON of the build")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--mask")
    g.add_argument("--stl")
    p.add_argument("--voxel", type=float, default=None, help="re-voxelization size for --stl")
    p.add_argument("--report", default=None)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("selftest", help="C# implicit vs numpy mirror")
    _common(p)
    p.add_argument("--n", type=int, default=100000)
    p.add_argument("--method", choices=("auto", "pythonnet", "exe"), default="auto")
    p.set_defaults(func=cmd_selftest)

    for name, fn in (("estimate", cmd_estimate), ("config", cmd_config)):
        p = sub.add_parser(name)
        _common(p)
        p.add_argument("--mode", choices=("sector", "full"), default="sector")
        p.set_defaults(func=fn)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
