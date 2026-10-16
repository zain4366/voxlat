"""Pure-numpy sector generator and volume-fraction cross-check (Task 12).

``generate_mask`` evaluates ``JacketField.signed_distance`` (the numpy mirror of the C# implicit) at the
centres of the same integer voxel lattice PicoGK uses (centre of voxel k at k h mm), so a numpy mask
and a PicoGK mask of the same design can be compared voxel by voxel (``compare_masks``).

``volume_fraction_report`` compares the solid fraction of the generated lattice with the design's
relative-density field over the lattice region of the nominal sector: overall, near the walls vs the
core of the gap, and in (theta, z) patches about two cells wide.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
from scipy import ndimage

from voxlat.export.field import JacketField
from voxlat.export.maskio import VoxelMask

__all__ = [
    "wide_half_angle",
    "render_bbox",
    "generate_mask",
    "open_mask",
    "capped_edt",
    "volume_fraction_report",
    "compare_masks",
]


def wide_half_angle(field: JacketField, margin_mm: float | None = None) -> float:
    """Half angle (rad) of the rendered sector: nominal + margin (mm, measured at the sleeve radius)."""
    m = float(field.config["render"]["sector_margin_mm"]) if margin_mm is None else float(margin_mm)
    return field.sector_half + m / field.r_li


def render_bbox(field: JacketField, half_angle: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box (mm, render frame) of the rendered part. Mirrors ``JacketBuilder.RenderBox``."""
    top = field.r_o + max(float(p["boss_height_mm"]) for p in field.ports) + 1.0
    if field.mode == "sector":
        b = wide_half_angle(field) if half_angle is None else half_angle
        lo = np.array([field.r_s * math.cos(b), -top * math.sin(b), field.z_lo])
        hi = np.array([top, top * math.sin(b), field.z_hi])
    else:
        lo = np.array([-top, -top, field.z_lo])
        hi = np.array([top, top, field.z_hi])
    pad = 0.5  # mm
    return lo - pad, hi + pad


def generate_mask(field: JacketField, voxel: float | None = None, *, half_angle: float | None = None,
                  bbox: tuple[Any, Any] | None = None, z_block: int = 8, progress: bool = False) -> VoxelMask:
    """Solid mask (sd <= 0) of the jacket on the PicoGK voxel lattice (centres at k * voxel, mm).

    Sector mode renders the *wide* sector (nominal half angle + ``sector_margin_mm``), like the C# side,
    so checks near the nominal cut faces see the true neighbourhood.
    """
    h = float(field.config["render"]["voxel_size_mm"]) if voxel is None else float(voxel)
    if field.mode == "sector" and half_angle is None:
        half_angle = wide_half_angle(field)
    lo, hi = render_bbox(field, half_angle) if bbox is None else (np.asarray(bbox[0], float), np.asarray(bbox[1], float))
    i0 = np.floor(lo / h).astype(int)
    i1 = np.ceil(hi / h).astype(int)
    shape = tuple(int(v) for v in (i1 - i0 + 1))
    x = (i0[0] + np.arange(shape[0])) * h
    y = (i0[1] + np.arange(shape[1])) * h
    z = (i0[2] + np.arange(shape[2])) * h
    solid = np.zeros(shape, dtype=bool)
    X, Y = np.meshgrid(x, y, indexing="ij")
    for k0 in range(0, shape[2], z_block):
        zs = z[k0:k0 + z_block]
        sd = field.signed_distance(X[:, :, None], Y[:, :, None], zs[None, None, :], half_angle=half_angle)
        solid[:, :, k0:k0 + len(zs)] = sd <= 0.0
        if progress and (k0 // z_block) % 20 == 0:
            print(f"  numpy sector: slice {k0}/{shape[2]}", flush=True)
    frame = {"mode": field.mode, "sector_center_deg": math.degrees(field.sector_center),
             "half_angle_deg": None if half_angle is None else math.degrees(half_angle),
             "nominal_half_angle_deg": math.degrees(field.sector_half)}
    return VoxelMask(solid=solid, origin=(float(x[0]), float(y[0]), float(z[0])), voxel=h, frame=frame,
                     source="numpy", kind="raw", meta={"design": field.config.get("design_label", "")})


# =============================================================================
# Distance transforms (memory-bounded) and the sliver opening
# =============================================================================
def capped_edt(feature: np.ndarray, voxel: float, cap: float, slab: int = 96) -> np.ndarray:
    """Euclidean distance (mm, float32) from every voxel to the nearest voxel where ``feature`` is False,
    exact up to ``cap`` and clipped at ``cap`` beyond. Computed in z slabs with a halo of cap / voxel
    voxels, so memory stays ~slab-sized (scipy's EDT needs ~20 bytes per voxel)."""
    nz = feature.shape[2]
    halo = int(math.ceil(cap / voxel)) + 1
    out = np.empty(feature.shape, dtype=np.float32)
    for k0 in range(0, nz, slab):
        k1 = min(nz, k0 + slab)
        a0, a1 = max(0, k0 - halo), min(nz, k1 + halo)
        block = feature[:, :, a0:a1]
        if not block.any():
            out[:, :, k0:k1] = 0.0
            continue
        if block.all():
            out[:, :, k0:k1] = cap
            continue
        d = ndimage.distance_transform_edt(block, sampling=voxel)
        out[:, :, k0:k1] = np.minimum(d[:, :, k0 - a0:k1 - a0], cap)
    return out


def open_mask(mask: VoxelMask, radius: float, *, field: JacketField | None = None, band: float | None = None) -> VoxelMask:
    """Morphological opening by a ball of ``radius`` mm (numpy counterpart of PicoGK Offset(-r), Offset(+r)).

    With ``field`` and ``band`` > 0 the opening is applied only within ``band`` mm of the open lattice faces
    (slot sides, ring faces; ``JacketField.interface_distance``) - where clipping leaves slivers - and the
    rest of the part is kept as designed (thin interior walls are reported by the checks, not removed).
    Mirrors ``JacketBuilder`` (C#).
    """
    if radius <= 0:
        return mask
    h = mask.voxel
    D = capped_edt(mask.solid, h, radius + 2 * h)
    eroded = D >= radius + 0.25 * h
    del D
    E = capped_edt(~eroded, h, radius + 2 * h)
    del eroded
    opened = mask.solid & (E <= radius + 0.5 * h)
    del E
    meta = dict(mask.meta, opening_radius_mm=radius)
    if field is not None and band is not None and band > 0:
        x, y, _ = mask.centres()
        X, Y = np.broadcast_arrays(x[:, :, 0], y[:, :, 0])
        z = mask.axis(2)
        for k in range(mask.shape[2]):
            inband = field.interface_distance(X, Y, np.full_like(X, z[k])) <= band
            opened[:, :, k] = np.where(inband, opened[:, :, k], mask.solid[:, :, k])
        meta["opening_band_mm"] = band
    return VoxelMask(solid=opened, origin=mask.origin, voxel=h, frame=mask.frame, source=mask.source,
                     kind="clean", meta=meta)


# =============================================================================
# Cross-checks
# =============================================================================
@functools.lru_cache(maxsize=2)
def planar_profile_table(n: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solid fraction of the plane xi_x = u (averaged over xi_y, xi_z) of the network TPMS at (w, rho*).

    TPMS are layered: at rho* = 0.25 the plane fraction swings by about +-10 % (gyroid) / +-15 % (diamond)
    of rho* along each cell axis. A gap that holds a non-integer number of cells therefore has a mean
    density that depends on the cut phase - the *phase-aware* expectation used to check the generator.
    Returns (w grid, rho grid, table[w, rho, u]) with u = (k + 1/2) / n.
    """
    from voxlat.geometry.tpms import sample_level_set, threshold_from_table

    w_grid = np.round(np.linspace(0.0, 1.0, 21), 10)
    r_grid = np.round(np.arange(10, 61) / 100.0, 10)
    tab = np.empty((len(w_grid), len(r_grid), n))
    for i, w in enumerate(w_grid):
        phi = sample_level_set((n, n, n), float(w))
        c = threshold_from_table(np.full(len(r_grid), w), r_grid)
        for j, cj in enumerate(np.atleast_1d(c)):
            tab[i, j] = (phi <= cj).mean(axis=(1, 2))
    return w_grid, r_grid, tab


def _planar_expectation(w: np.ndarray, rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    wg, rg, tab = planar_profile_table()
    n = tab.shape[2]
    fw = (np.clip(w, wg[0], wg[-1]) - wg[0]) / (wg[1] - wg[0])
    fr = (np.clip(rho, rg[0], rg[-1]) - rg[0]) / (rg[1] - rg[0])
    i = np.minimum(fw.astype(int), len(wg) - 2)
    j = np.minimum(fr.astype(int), len(rg) - 2)
    a, b = fw - i, fr - j
    t = np.mod(u, 1.0) * n - 0.5
    k0 = np.floor(t).astype(int)
    c = t - k0
    k0 %= n
    k1 = (k0 + 1) % n
    # gather along u (explicit to keep memory low)
    out = np.zeros_like(u, dtype=float)
    for di, wa in ((0, 1 - a), (1, a)):
        for dj, wb in ((0, 1 - b), (1, b)):
            v0 = tab[i + di, j + dj, k0]
            v1 = tab[i + di, j + dj, k1]
            out += wa * wb * ((1 - c) * v0 + c * v1)
    return out


def volume_fraction_report(field: JacketField, mask: VoxelMask, *, patch_cells: float = 2.0) -> dict[str, Any]:
    """Solid fraction of the lattice region vs the design rho* field (nominal sector only).

    Two references: the design rho* (what the optimizer assumed) and the *phase-aware* expectation - the
    planar-averaged fraction of the same TPMS at the local radial phase u_r (``planar_profile_table``).
    The first measures how the finite gap + cut phase realizes rho*; the second checks the generator
    (threshold, phase field, clipping) and should agree to the voxel discretization error.
    Streams over z slices (memory ~ one slice); patches ~``patch_cells`` mean cells wide in (s, z).
    """
    h = mask.voxel
    x, y, _ = mask.centres()
    X, Y = np.broadcast_arrays(x[:, :, 0], y[:, :, 0])
    th_loc = np.arctan2(Y, X)
    inside_sector = np.abs(th_loc) <= field.sector_half if field.mode == "sector" else np.ones_like(X, dtype=bool)
    z = mask.axis(2)
    Lm = float(np.mean(field.tab["L"]))
    ds = patch_cells * Lm
    ns_b = int(math.ceil(2 * math.pi * field.r_m / ds)) + 2
    nz_b = int(math.ceil((field.L_ax + 2) / ds)) + 2
    acc = {k: np.zeros((ns_b, nz_b)) for k in ("n", "solid", "target", "expected")}
    tot = {k: 0.0 for k in ("n", "solid", "target", "expected", "core_n", "core_solid", "core_target", "core_expected",
                            "wall_solid", "wall_target", "wall_expected")}
    th_off = math.pi  # local angles in (-pi, pi]
    for k in range(mask.shape[2]):
        if z[k] < 0.0 or z[k] > field.L_ax:
            continue
        zk = np.full_like(X, z[k])
        reg = field.regions(X, Y, zk)
        lat = reg["lattice"] & inside_sector
        if not lat.any():
            continue
        th = reg["theta"][lat]
        r = reg["r"][lat]
        zz = np.full(th.shape, z[k])
        F = field.fields_at(th, zz)
        sol = mask.solid[:, :, k][lat].astype(float)
        xw, yw = field.to_world(X[lat], Y[lat])
        lv = field.lattice_level(xw, yw, zz)
        ex = _planar_expectation(F["w"], F["rho"], lv["u_r"])
        tgt = F["rho"]
        core = np.minimum(r - field.r_li, field.r_lo - r) > 0.5 * F["L"]
        tot["n"] += sol.size
        tot["solid"] += sol.sum()
        tot["target"] += tgt.sum()
        tot["expected"] += ex.sum()
        tot["core_n"] += int(core.sum())
        tot["core_solid"] += sol[core].sum()
        tot["core_target"] += tgt[core].sum()
        tot["core_expected"] += ex[core].sum()
        tot["wall_solid"] += sol[~core].sum()
        tot["wall_target"] += tgt[~core].sum()
        tot["wall_expected"] += ex[~core].sum()
        sb = np.floor((th_loc[lat] + th_off) * field.r_m / ds).astype(int)
        zb = np.floor((z[k] + 1.0) / ds).astype(int)
        np.add.at(acc["n"], (sb, zb), 1.0)
        np.add.at(acc["solid"], (sb, zb), sol)
        np.add.at(acc["target"], (sb, zb), tgt)
        np.add.at(acc["expected"], (sb, zb), ex)
    if tot["n"] == 0:
        raise ValueError("no lattice voxels in the mask")
    cnt = acc["n"]
    ok = cnt >= max(50, int(0.5 * (ds / h) ** 2 * (field.r_lo - field.r_li) / h))
    ms = acc["solid"][ok] / cnt[ok]
    err = ms / (acc["target"][ok] / cnt[ok]) - 1.0
    err_e = ms / (acc["expected"][ok] / cnt[ok]) - 1.0
    wall_n = tot["n"] - tot["core_n"]

    def rel(a_, b_):
        return float(a_ / b_ - 1.0) if b_ else None

    return {
        "voxel_mm": h,
        "n_voxels": int(tot["n"]),
        "solid_fraction": tot["solid"] / tot["n"],
        "target_rho_mean": tot["target"] / tot["n"],
        "relative_error": rel(tot["solid"], tot["target"]),
        "core_relative_error": rel(tot["core_solid"], tot["core_target"]) if tot["core_n"] else None,
        "near_wall_relative_error": rel(tot["wall_solid"], tot["wall_target"]) if wall_n else None,
        "core_fraction_of_voxels": tot["core_n"] / tot["n"],
        "phase_aware_expected": tot["expected"] / tot["n"],
        "relative_error_vs_phase_aware": rel(tot["solid"], tot["expected"]),
        "core_relative_error_vs_phase_aware": rel(tot["core_solid"], tot["core_expected"]) if tot["core_n"] else None,
        "near_wall_relative_error_vs_phase_aware": rel(tot["wall_solid"], tot["wall_expected"]) if wall_n else None,
        "patches": {"size_mm": ds, "count": int(ok.sum()),
                    "vs_design_mean_abs_rel_error": float(np.mean(np.abs(err))) if err.size else None,
                    "vs_design_max_abs_rel_error": float(np.max(np.abs(err))) if err.size else None,
                    "vs_phase_aware_mean_abs_rel_error": float(np.mean(np.abs(err_e))) if err.size else None,
                    "vs_phase_aware_max_abs_rel_error": float(np.max(np.abs(err_e))) if err.size else None},
    }


def compare_masks(a: VoxelMask, b: VoxelMask) -> dict[str, Any]:
    """Voxel-by-voxel agreement of two masks on the same voxel lattice (aligned by integer indices)."""
    if abs(a.voxel - b.voxel) > 1e-9 * max(a.voxel, b.voxel):
        raise ValueError("masks have different voxel sizes")
    h = a.voxel
    ia = np.rint(np.asarray(a.origin) / h).astype(int)
    ib = np.rint(np.asarray(b.origin) / h).astype(int)
    lo = np.maximum(ia, ib)
    hi = np.minimum(ia + np.asarray(a.shape), ib + np.asarray(b.shape))
    if np.any(hi <= lo):
        raise ValueError("masks do not overlap")
    sa = tuple(slice(lo[k] - ia[k], hi[k] - ia[k]) for k in range(3))
    sb = tuple(slice(lo[k] - ib[k], hi[k] - ib[k]) for k in range(3))
    A, B = a.solid[sa], b.solid[sb]
    inter = np.count_nonzero(A & B)
    union = np.count_nonzero(A | B)
    na, nb = np.count_nonzero(A), np.count_nonzero(B)
    # solid outside the common window (should be ~0 if the windows cover the part)
    out_a = int(np.count_nonzero(a.solid)) - na
    out_b = int(np.count_nonzero(b.solid)) - nb
    return {
        "voxel_mm": h,
        "common_shape": [int(v) for v in hi - lo],
        "solid_a": int(na),
        "solid_b": int(nb),
        "volume_a_mm3": na * h**3,
        "volume_b_mm3": nb * h**3,
        "volume_rel_diff": (nb - na) / max(na, 1),
        "iou": inter / max(union, 1),
        "mismatch_voxels": int(union - inter),
        "solid_outside_common_a": out_a,
        "solid_outside_common_b": out_b,
    }


# =============================================================================
# Sanity check and memory-bounded cropping (used before the long checks)
# =============================================================================
def quick_slice_iou(field: JacketField, mask: VoxelMask, n_slices: int = 3,
                    half_angle: float | None = None) -> dict[str, Any]:
    """Agreement of a mask with the numpy implicit on a few z slices (seconds, any mask size).

    Catches mis-registered masks (axis order, flips, offsets) before hours of checks run on them.
    """
    if half_angle is None and field.mode == "sector":
        fr = mask.frame or {}
        half_angle = math.radians(float(fr["half_angle_deg"])) if fr.get("half_angle_deg") is not None \
            else wide_half_angle(field)
    x, y, _ = mask.centres()
    X, Y = np.broadcast_arrays(x[:, :, 0], y[:, :, 0])
    z = mask.axis(2)
    ks = np.linspace(0.25, 0.75, n_slices)
    out = []
    for f_ in ks:
        k = int(round(f_ * (mask.shape[2] - 1)))
        sd = field.signed_distance(X, Y, np.full_like(X, z[k]), half_angle=half_angle)
        a, b = sd <= 0.0, mask.solid[:, :, k]
        out.append(float(np.count_nonzero(a & b) / max(np.count_nonzero(a | b), 1)))
    return {"z_slices_mm": [float(z[int(round(f_ * (mask.shape[2] - 1)))]) for f_ in ks], "iou": out,
            "iou_min": float(min(out))}


def crop_to_wedge(mask: VoxelMask, half_angle: float, r_top: float) -> VoxelMask:
    """Crop a sector mask (sector-local frame) to the wedge |theta| <= half_angle: y range cut to the wedge's
    bounding box and solid outside the wedge cleared (the same cut the numpy generator makes)."""
    yb = r_top * math.sin(half_angle) + mask.voxel
    y = mask.axis(1)
    j = np.flatnonzero(np.abs(y) <= yb)
    j0, j1 = int(j[0]), int(j[-1]) + 1
    frame = dict(mask.frame or {})
    frame["half_angle_deg"] = math.degrees(half_angle)
    solid = np.ascontiguousarray(mask.solid[:, j0:j1, :])
    X, Y = np.meshgrid(mask.axis(0), y[j0:j1], indexing="ij")
    sb, cb = math.sin(half_angle), math.cos(half_angle)
    solid &= (np.maximum(-X * sb + Y * cb, -X * sb - Y * cb) <= 0.0)[:, :, None]
    return VoxelMask(solid=solid,
                     origin=(mask.origin[0], float(y[j0]), mask.origin[2]), voxel=mask.voxel, frame=frame,
                     source=mask.source, kind=mask.kind, meta=dict(mask.meta, cropped_half_angle_deg=math.degrees(half_angle)))
