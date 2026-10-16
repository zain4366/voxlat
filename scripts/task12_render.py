"""Task 12 - 3-D renders of the exported jacket, ray-traced straight from the signed-distance field.

The numpy implicit (voxlat.export.field) reproduces the PicoGK geometry voxel for voxel (laptop run: IoU 0.997), so
these images show the exported part without loading a 20-M-triangle STL. Sphere tracing + central-difference
normals + two-light Lambert shading + SDF ambient occlusion; the lattice and the housing get different colours.

    python scripts/task12_render.py                       # results/figures/task12_render_*.png (~5-15 min)
    python scripts/task12_render.py --width 800 --quick   # preview
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from voxlat.export.config import ExportOptions, build_export_config, load_design  # noqa: E402
from voxlat.export.field import JacketField  # noqa: E402


class Scene:
    """The jacket with an angular cut-away (theta in [cut_a, cut_b] removed) and an optional z window."""

    def __init__(self, field: JacketField, cut_center_deg: float | None, cut_half_deg: float,
                 z_window: tuple[float, float] | None = None, extra_box: tuple | None = None,
                 peel_center_deg: float | None = None, peel_half_deg: float = 0.0):
        self.f = field
        self.cut = None if cut_center_deg is None else (math.radians(cut_center_deg), math.radians(cut_half_deg))
        # peel: inside this wedge everything outside the lattice gap (outer wall, bosses) is removed
        self.peel = None if peel_center_deg is None else (math.radians(peel_center_deg), math.radians(peel_half_deg))
        self.zw = z_window
        self.box = extra_box

    @staticmethod
    def _wedge(x, y, c, h):
        xr = math.cos(-c) * x - math.sin(-c) * y
        yr = math.sin(-c) * x + math.cos(-c) * y
        return np.maximum(-xr * math.sin(h) + yr * math.cos(h), -xr * math.sin(h) - yr * math.cos(h))

    def sd(self, P: np.ndarray) -> np.ndarray:
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        s = self.f.signed_distance(x, y, z, half_angle=-1.0)
        if self.cut is not None:
            s = np.maximum(s, -self._wedge(x, y, *self.cut))  # keep the complement of the wedge
        if self.peel is not None:
            r = np.hypot(x, y)
            region = np.maximum(self._wedge(x, y, *self.peel), (self.f.r_lo - 0.02) - r)
            s = np.maximum(s, -region)
        if self.zw is not None:
            s = np.maximum(s, np.maximum(self.zw[0] - z, z - self.zw[1]))
        if self.box is not None:
            lo, hi = self.box
            s = np.maximum(s, np.max(np.stack([lo[0] - x, x - hi[0], lo[1] - y, y - hi[1]]), axis=0))
        return s

    def is_lattice(self, P: np.ndarray) -> np.ndarray:
        r = np.hypot(P[:, 0], P[:, 1])
        return (r > self.f.r_li + 0.05) & (r < self.f.r_lo - 0.05) & (P[:, 2] > 0.05) & (P[:, 2] < self.f.L_ax - 0.05)


def look_at(eye, target, up=(0, 0, 1)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    upv = np.cross(right, fwd)
    return eye, fwd, right, upv


def render(scene: Scene, eye, target, fov_deg: float, width: int, height: int, bound_r: float, z_lo: float, z_hi: float,
           max_steps: int = 600, step: float = 0.7, eps: float = 0.004) -> tuple[np.ndarray, np.ndarray]:
    eye, fwd, right, upv = look_at(eye, target)
    aspect = width / height
    sx = math.tan(math.radians(fov_deg) / 2)
    u = (np.arange(width) + 0.5) / width * 2 - 1
    v = 1 - (np.arange(height) + 0.5) / height * 2
    U, V = np.meshgrid(u * sx * aspect, v * sx)
    D = fwd[None, None, :] + U[..., None] * right + V[..., None] * upv
    D = (D / np.linalg.norm(D, axis=-1, keepdims=True)).reshape(-1, 3)
    O = np.broadcast_to(eye, D.shape).copy()
    n = len(D)
    # start at the bounding cylinder r <= bound_r, z in [z_lo, z_hi] (slab test on z, quadratic on r)
    t0 = np.zeros(n)
    t1 = np.full(n, np.inf)
    a = D[:, 0] ** 2 + D[:, 1] ** 2
    b = 2 * (O[:, 0] * D[:, 0] + O[:, 1] * D[:, 1])
    c = O[:, 0] ** 2 + O[:, 1] ** 2 - bound_r**2
    disc = b * b - 4 * a * c
    ok = (disc > 0) & (a > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0))
    ta = np.where(ok, (-b - sq) / (2 * np.where(ok, a, 1)), np.inf)
    tb = np.where(ok, (-b + sq) / (2 * np.where(ok, a, 1)), -np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        tz1 = (z_lo - O[:, 2]) / D[:, 2]
        tz2 = (z_hi - O[:, 2]) / D[:, 2]
    tzmin, tzmax = np.minimum(tz1, tz2), np.maximum(tz1, tz2)
    t0 = np.maximum(np.maximum(ta, tzmin), 0)
    t1 = np.minimum(tb, tzmax)
    active = np.flatnonzero(t1 > t0)
    t = t0.copy()
    hit = np.zeros(n, bool)
    for it in range(max_steps):
        if active.size == 0:
            break
        P = O[active] + t[active, None] * D[active]
        s = scene.sd(P)
        h = s < eps
        hit[active[h]] = True
        t[active] += np.maximum(s * step, eps * 0.5)
        keep = ~h & (t[active] < t1[active])
        active = active[keep]
        if it % 40 == 0:
            print(f"    step {it}: {active.size} rays marching", flush=True)
    img = np.ones((n, 3))
    bg_top, bg_bot = np.array([0.97, 0.97, 0.99]), np.array([0.86, 0.88, 0.92])
    vv = np.repeat((v + 1) / 2, width)
    img[:] = bg_bot[None] + (bg_top - bg_bot)[None] * vv[:, None]
    idx = np.flatnonzero(hit)
    if idx.size:
        P = O[idx] + t[idx, None] * D[idx]
        e = 0.02
        N = np.zeros_like(P)
        for k in range(3):
            dp = np.zeros(3)
            dp[k] = e
            N[:, k] = scene.sd(P + dp) - scene.sd(P - dp)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        N = np.where((N * D[idx]).sum(1, keepdims=True) > 0, -N, N)
        # ambient occlusion from the SDF along the normal
        ao = np.ones(len(P))
        for k, dist in enumerate((0.15, 0.4, 0.9)):
            ao -= (dist - np.minimum(scene.sd(P + N * dist), dist)) / dist * 0.33 * (0.6 ** k)
        ao = np.clip(ao, 0.25, 1.0)
        L1 = np.array([-0.4, -0.6, 0.7])
        L2 = np.array([0.7, 0.2, 0.3])
        L1, L2 = L1 / np.linalg.norm(L1), L2 / np.linalg.norm(L2)
        diff = 0.75 * np.clip(N @ L1, 0, 1) + 0.3 * np.clip(N @ L2, 0, 1)
        Hh = L1 - D[idx]
        Hh /= np.linalg.norm(Hh, axis=1, keepdims=True)
        spec = 0.25 * np.clip((N * Hh).sum(1), 0, 1) ** 40
        lat = scene.is_lattice(P)
        base = np.where(lat[:, None], np.array([0.85, 0.48, 0.18])[None], np.array([0.62, 0.66, 0.72])[None])
        col = base * (0.22 + diff[:, None]) * ao[:, None] + spec[:, None]
        img[idx] = np.clip(col, 0, 1)
    return img.reshape(height, width, 3), hit.reshape(height, width)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default=str(ROOT / "results" / "designs" / "task12_slots_best_at_cap_gyroid.json"))
    ap.add_argument("--width", type=int, default=1800)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    d = load_design(a.design)
    cfg = build_export_config(d, ExportOptions(mode="full", sector_center_deg=0.0))
    f = JacketField(cfg, d)
    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    W = a.width
    views = [
        # name, scene, eye, target, fov, aspect
        ("overview", Scene(f, cut_center_deg=-75.0, cut_half_deg=14.0, peel_center_deg=-20.0, peel_half_deg=45.0),
         (175.0, -175.0, 120.0), (0.0, 0.0, 18.0), 36.0, 1.5),
        ("lattice_closeup", Scene(f, cut_center_deg=None, cut_half_deg=0.0, peel_center_deg=200.0, peel_half_deg=40.0),
         (-118.0, -50.0, 40.0), (-70.0, -22.0, 25.0), 30.0, 1.5),
        ("outlet_slot_section", Scene(f, cut_center_deg=211.0, cut_half_deg=29.0),
         (-118.0, -112.0, 78.0), (-76.0, -6.0, 22.0), 32.0, 1.5),
    ]
    if a.quick:
        views = views[:1]
    for name, sc, eye, tgt, fov, asp in views:
        t0 = time.perf_counter()
        H = int(W / asp)
        print(f"render {name}: {W} x {H}", flush=True)
        top = f.r_o + max(p["boss_height_mm"] for p in f.ports) + 1.5
        img, _ = render(sc, eye, tgt, fov, W, H, bound_r=top, z_lo=f.z_lo - 0.5, z_hi=f.z_hi + 0.5)
        plt.imsave(out / f"task12_render_{name}.png", img)
        print(f"  saved task12_render_{name}.png in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
