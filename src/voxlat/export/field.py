"""Graded TPMS jacket implicit - numpy reference implementation (Task 12).

This module is the **specification** of the geometry that the C# class ``GradedTpmsJacket``
(``src/voxlat/export/dotnet``) evaluates inside PicoGK. Every step below is mirrored line by line
in ``JacketField.cs``; ``scripts/task12_export.py selftest`` evaluates both on the same random points
(C# through pythonnet) and requires agreement to 1e-6 mm.

Coordinates and units
---------------------
World frame in **mm** (PicoGK unit): z along the motor axis, z = 0 at the start of the heated length,
theta = atan2(y, x) in [0, 2 pi), measured from +x, which is the inlet manifold (``inlet_angle`` = 0).
The design fields (Task 11) are B-splines in (s = theta r_m, z) and are evaluated with the same bases
as ``voxlat.optimize.parameterization``; they do not vary through the gap (depth-averaged model).

Phase-field grading (cells never tear)
--------------------------------------
The TPMS is evaluated at cell coordinates xi = (u_r, u_t, u_z) (the cell frame of Tasks 7/9:
x = radial, y = circumferential, z = axial), each with period 1:

* circumferential: P(theta, z) = int_{theta_s}^theta r_m / L(theta', z) dtheta' (cells at r_m, from the
  seam angle theta_s = inlet angle), T(z) = P(theta_s + 2 pi, z).
  ``integer`` mode: an **integer** number of cells around, N = round(mean_z T) (or fixed by the user),
  u_t = N P / T + phase_theta0, so u_t(2 pi) - u_t(0) = N exactly -> the lattice closes on itself without
  a seam; the local circumferential cell size at r_m is L T(z)/N. A continuous phase field on the
  annulus has ONE winding number, so if the design's cell count around varies with z (cell size graded
  axially), the cells must stretch circumferentially by T(z)/N.
  ``seam`` mode: u_t = P + phase_theta0 (exact local size L everywhere); the phase jumps by T(z) - an
  integer only by chance - at theta_s. In the ``slots`` layout the seam lies inside the inlet slot, where
  there is no lattice; otherwise a meridional rib (``seam_rib_mm``) fuses the two cut faces.
  The seam mode is NOT a fix for axially graded cell sizes: the mismatch T(z) - T(z') accumulates away
  from the seam and tilts the circumferential iso-phase surfaces (84 deg for end_rings/knee). Default:
  integer (the faithful choice; the circumferential stretch T/N is reported per design);
* axial: u_z = phase_z0 + int_{z_ref}^z dz' / (a_z L(theta, z')), z_ref = mid-length by default -
  continuous in theta and z. Where L varies along theta the number of axial cells differs between
  paths, so the axial iso-phase surfaces must tilt (reported as ``axial_phase_shear_deg_max``); they
  never jump. Starting the integral at mid-length halves the largest tilt;
* radial (``conformal=True``): u_r = phase_r0 + (r_m / L) ln(r / r_i). With u_t this is the log-polar
  (conformal) map of the annulus: the cell shape does not change through the gap, its size scales with
  r / r_m (0.96 ... 1.04). The radial size at r_m is exactly L, so the number of cells across the gap
  is the design's N = h / L (the finite-gap closures of Task 7 depend on it); only the circumferential
  size carries the integer-N factor T/N. ``conformal=False``: u_r = phase_r0 + (r - r_i) / L.
  phase_r0 is the cut phase at the sleeve (Task 7 convention: the inner wall cuts the lattice at
  x = phase).

All derived quantities are tabulated once on a (theta, z) node grid (``table_n_theta`` x
``table_n_z``, default 2048 x 257: 0.23 mm x 0.2 mm) - trapezoidal integrals for P and u_z - and
interpolated bilinearly (periodically in theta).

Level set and solid
-------------------
phi_w = (1 - w) G / (3/2) + w D / sqrt 2 (Task 1, ``tpms.level_set``), network solid where
phi <= c(w, rho) with c from the continuum threshold table (Task 1, n_ref = 128, bilinear in w, rho
on the nodes). The signed distance handed to PicoGK is the first-order estimate
sd = (phi - c) / |grad phi| (grad through the diagonal phase Jacobian; floored, clamped to
+-``sd_clamp_mm``). Only its sign defines the part; the magnitude only needs to be distance-like near
the surface (PicoGK's narrow band).

Part = lattice (clipped to the gap, minus manifold slots) U sleeve U outer wall U end caps U port
bosses, minus port bores; ``sector`` mode intersects with a wedge |theta_local| <= half angle.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = ["DesignFields", "JacketField", "TWO_PI", "periodic_basis", "clamped_basis"]

TWO_PI = 2.0 * math.pi
G_AMP = 1.5
D_AMP = math.sqrt(2.0)
# boss bottom / bore start, as fractions between the radii (see JacketField docstring of ports)
_GRAD_FLOOR = 0.25


# =============================================================================
# B-spline bases (the definitions of voxlat.optimize.parameterization, written so C# can mirror them)
# =============================================================================
def _b3(t: np.ndarray) -> np.ndarray:
    a = np.abs(t)
    return np.where(a < 1.0, (4.0 - 6.0 * a * a + 3.0 * a * a * a) / 6.0,
                    np.where(a < 2.0, (2.0 - a) ** 3 / 6.0, 0.0))


def periodic_basis(u: Any, n: int) -> np.ndarray:
    """Periodic uniform cubic B-spline basis (len(u), n), period 1; control i centred at u = i / n."""
    u = np.asarray(u, float).ravel()
    if n == 1:
        return np.ones((len(u), 1))
    t = np.mod(u * n, n)
    B = np.zeros((len(u), n))
    for i in range(n):
        for k in (-2, -1, 0, 1, 2):
            B[:, i] += _b3(t - i - k * n)
    return B


def clamped_basis(t: Any, n: int) -> np.ndarray:
    """Clamped uniform B-spline basis of degree min(3, n - 1) on [0, 1] (Cox-de Boor; t = 1 -> last span)."""
    t = np.clip(np.asarray(t, float).ravel(), 0.0, 1.0)
    if n == 1:
        return np.ones((len(t), 1))
    p = min(3, n - 1)
    m = n - p  # number of inner intervals
    knots = np.concatenate([np.zeros(p), np.arange(m + 1) / m, np.ones(p)])
    nk = len(knots)
    B = np.zeros((len(t), nk - 1))
    last = nk - p - 2
    for i in range(nk - 1):
        if knots[i] < knots[i + 1]:
            B[:, i] = (t >= knots[i]) & (t < knots[i + 1])
    end = t >= 1.0
    B[end, :] = 0.0
    B[end, last] = 1.0
    for k in range(1, p + 1):
        Bn = np.zeros((len(t), nk - 1 - k))
        for i in range(nk - 1 - k):
            d1 = knots[i + k] - knots[i]
            d2 = knots[i + k + 1] - knots[i + 1]
            a = (t - knots[i]) / d1 * B[:, i] if d1 > 0 else 0.0
            b = (knots[i + k + 1] - t) / d2 * B[:, i + 1] if d2 > 0 else 0.0
            Bn[:, i] = a + b
        B = Bn
    return B


class DesignFields:
    """Exact B-spline fields rho*, w, L [mm], a_z of a Task 11 design at (theta [rad], z [mm])."""

    NAMES = ("rho", "w", "L", "a_z")

    def __init__(self, design: Mapping[str, Any]):
        basis = design["basis"]
        self.r_mean_mm = float(basis["r_mean_m"]) * 1e3
        self.axial_length_mm = float(basis["axial_length_m"]) * 1e3
        nets = design["control_nets"]
        fixed = dict(design.get("fixed_fields") or basis.get("spec", {}).get("fixed", {}) or {})
        self.nets: dict[str, np.ndarray] = {}
        self.const: dict[str, float] = {}
        for name in self.NAMES:
            if name in nets:
                P = np.asarray(nets[name], float)
                if P.ndim != 2:
                    raise ValueError(f"control net {name} must be 2-D (n_s, n_z)")
                self.nets[name] = P * (1e3 if name == "L" else 1.0)
            else:
                self.const[name] = float(fixed.get(name, 1.0))

    def evaluate_grid(self, theta: Any, z: Any) -> dict[str, np.ndarray]:
        """Fields on the tensor grid theta (n_t,) x z (n_z,): Bs @ P @ Bz^T (same values as ``evaluate``)."""
        theta = np.asarray(theta, float).ravel()
        z = np.asarray(z, float).ravel()
        out = {}
        for name in self.NAMES:
            if name in self.nets:
                P = self.nets[name]
                Bs = periodic_basis(theta / TWO_PI, P.shape[0])
                Bz = clamped_basis(z / self.axial_length_mm, P.shape[1])
                out[name] = Bs @ P @ Bz.T
            else:
                out[name] = np.full((theta.size, z.size), self.const[name])
        return out

    def evaluate(self, theta: Any, z: Any) -> dict[str, np.ndarray]:
        theta, z = np.broadcast_arrays(np.asarray(theta, float), np.asarray(z, float))
        u = (theta.ravel() / TWO_PI)
        t = z.ravel() / self.axial_length_mm
        out = {}
        for name in self.NAMES:
            if name in self.nets:
                P = self.nets[name]
                Bs = periodic_basis(u, P.shape[0])
                Bz = clamped_basis(t, P.shape[1])
                out[name] = np.einsum("ki,ij,kj->k", Bs, P, Bz).reshape(theta.shape)
            else:
                out[name] = np.full(theta.shape, self.const[name])
        return out


# =============================================================================
# Threshold table (Task 1 continuum c(w, rho), bilinear with clipping - as tpms.ThresholdTable)
# =============================================================================
class ThresholdLookup:
    def __init__(self, w: Any, rho: Any, c: Any):
        self.w = np.asarray(w, float)
        self.rho = np.asarray(rho, float)
        self.c = np.asarray(c, float)
        self.dw = (self.w[-1] - self.w[0]) / (len(self.w) - 1)
        self.dr = (self.rho[-1] - self.rho[0]) / (len(self.rho) - 1)
        if not (np.allclose(np.diff(self.w), self.dw) and np.allclose(np.diff(self.rho), self.dr)):
            raise ValueError("threshold table grid must be uniform")

    def __call__(self, w: Any, rho: Any) -> np.ndarray:
        w = np.clip(np.asarray(w, float), self.w[0], self.w[-1])
        r = np.clip(np.asarray(rho, float), self.rho[0], self.rho[-1])
        fw = (w - self.w[0]) / self.dw
        fr = (r - self.rho[0]) / self.dr
        i = np.minimum(np.floor(fw).astype(np.intp), len(self.w) - 2)
        j = np.minimum(np.floor(fr).astype(np.intp), len(self.rho) - 2)
        a, b = fw - i, fr - j
        C = self.c
        return ((1 - a) * (1 - b) * C[i, j] + a * (1 - b) * C[i + 1, j]
                + (1 - a) * b * C[i, j + 1] + a * b * C[i + 1, j + 1])


# =============================================================================
# The jacket implicit
# =============================================================================
class JacketField:
    """Signed distance (mm, negative = solid) of the exported jacket; numpy mirror of the C# implicit.

    ``config``: export-config dict (``voxlat.export.config.build_export_config``) or a path to its JSON;
    ``design``: Task 11 design dict or path (default: the config's ``design_path``).
    """

    def __init__(self, config: Mapping[str, Any] | str | Path, design: Mapping[str, Any] | str | Path | None = None):
        if not isinstance(config, Mapping):
            config = json.loads(Path(config).read_text())
        if config.get("schema") != "voxlat-export-config/1":
            raise ValueError(f"unsupported export config schema {config.get('schema')!r}")
        self.config = config
        if design is None:
            design = config["design_path"]
        if not isinstance(design, Mapping):
            design = json.loads(Path(design).read_text())
        self.design = design
        self.layout = config["layout"]
        g = config["geometry"]
        self.r_s = float(g["r_stator_mm"])
        self.r_li = float(g["r_lattice_in_mm"])
        self.r_lo = float(g["r_lattice_out_mm"])
        self.r_o = float(g["r_outer_mm"])
        self.r_m = float(g["r_mean_mm"])
        self.L_ax = float(g["axial_length_mm"])
        self.z_lo = float(g["z_min_mm"])
        self.z_hi = float(g["z_max_mm"])
        self.t_cap = float(g["end_cap_mm"])
        self.delta = float(g["lattice_overlap_mm"])
        self.slot = g.get("slot")
        self.ring = g.get("ring")
        self.ports = list(g["ports"])
        lat = config["lattice"]
        self.phase_r0 = float(lat["phase_r0"])
        self.phase_t0 = float(lat["phase_theta0"])
        self.phase_z0 = float(lat["phase_z0"])
        self.z_ref = float(lat.get("axial_phase_origin_mm", 0.5 * float(config["geometry"]["axial_length_mm"])))
        self.conformal = bool(lat["conformal"])
        self.circ_request = str(lat.get("circumferential", "integer"))
        self.seam = math.radians(float(lat.get("seam_angle_deg", 0.0))) % TWO_PI
        self.rib_t = float(lat.get("seam_rib_mm", 0.6))
        self.sd_clamp = float(lat["sd_clamp_mm"])
        self.n_t = int(lat["table_n_theta"])
        self.n_z = int(lat["table_n_z"])
        tt = config["threshold_table"]
        self.threshold = ThresholdLookup(tt["w"], tt["rho"], tt["c"])
        self.fields = DesignFields(design)
        if abs(self.fields.r_mean_mm - self.r_m) > 1e-6 or abs(self.fields.axial_length_mm - self.L_ax) > 1e-6:
            raise ValueError("design basis does not match the export geometry")
        self._slot_setup()
        self._build_tables(lat.get("n_theta_cells"))
        r = config["render"]
        self.mode = r["mode"]
        self.sector_center = math.radians(float(r["sector_center_deg"]))
        self.sector_half = math.radians(0.5 * float(r["sector_angle_deg"]))

    # ------------------------------------------------------------------ setup
    def _slot_setup(self) -> None:
        self.slots: list[tuple[float, float, float, float]] = []  # (angle, half-angle, z0, z1)
        if self.slot is None:
            return
        s = self.slot
        half = 0.5 * float(s["width_mm"]) / float(s["r_ref_mm"])
        z0, z1 = float(s["z0_mm"]), float(s["z1_mm"])
        # a slot that reaches an axial end also removes the lattice overlap into the end cap
        if z0 <= 1e-9:
            z0 = -2.0 * self.delta
        if z1 >= self.L_ax - 1e-9:
            z1 = self.L_ax + 2.0 * self.delta
        for key in ("inlet_angle_deg", "outlet_angle_deg"):
            self.slots.append((math.radians(float(s[key])), half, z0, z1))

    def _build_tables(self, n_cells: int | None) -> None:
        nt, nz = self.n_t, self.n_z
        self.d_theta = TWO_PI / nt
        self.d_z = self.L_ax / (nz - 1)
        th = np.arange(nt + 1) * self.d_theta  # table angle theta' = theta - theta_seam
        zz = np.arange(nz) * self.d_z
        F = self.fields.evaluate_grid(th + self.seam, zz)
        # periodic: the theta = 2 pi column is the theta = 0 column
        for k in F:
            F[k][nt, :] = F[k][0, :]
        L = F["L"]
        if np.any(L <= 0):
            raise ValueError("cell size field must be positive")
        f = self.r_m / L  # cells per radian at r_m
        P = np.zeros_like(f)
        P[1:, :] = np.cumsum(0.5 * self.d_theta * (f[:-1, :] + f[1:, :]), axis=0)
        Tz = P[nt, :]  # cells around at each z node
        if n_cells is None:
            N = max(1, int(round(float(np.mean(Tz)))))
        else:
            N = int(n_cells)
            if N < 1:
                raise ValueError("n_theta_cells must be >= 1")
        self.N_theta = N
        self.cells_around = Tz
        dev = float(np.max(np.abs(Tz / N - 1.0)))
        mode = self.circ_request
        if mode not in ("integer", "seam"):
            raise ValueError(f"circumferential must be 'integer' or 'seam', got {mode!r}")
        self.circ_mode = mode
        self.integer_deviation = dev
        # a seam needs a rib unless it lies inside a manifold slot that spans the whole lattice length
        self.rib = False
        if mode == "seam":
            hidden = False
            for ang, half, z0, z1 in self.slots:
                dth = abs((self.seam - ang + math.pi) % TWO_PI - math.pi)
                if dth < half - 1e-9 and z0 <= 0.0 and z1 >= self.L_ax:
                    hidden = True
            self.rib = not hidden
        g = 1.0 / (F["a_z"] * L)
        Uz = np.zeros_like(g)
        Uz[:, 1:] = np.cumsum(0.5 * self.d_z * (g[:, :-1] + g[:, 1:]), axis=1)
        # axial phase origin z_ref (default mid-length): u_z(theta, z_ref) = phase_z0 for every theta, which
        # halves the largest tilt of the axial iso-phase surfaces compared with starting at z = 0
        tz = min(max(self.z_ref / self.d_z, 0.0), nz - 1.0)
        jr = min(int(math.floor(tz)), nz - 2)
        br = tz - jr
        Uz -= ((1.0 - br) * Uz[:, jr] + br * Uz[:, jr + 1])[:, None]
        if mode == "integer":
            U_t = N * P / Tz[None, :] + self.phase_t0
            inv_ht = N / (Tz[None, :] * L)
        else:
            U_t = P + self.phase_t0
            inv_ht = 1.0 / L
        self.tab = {
            "U_t": U_t,
            "U_z": Uz + self.phase_z0,
            "inv_ht": inv_ht,  # 1 / (circumferential cell size at r_m), 1/mm
            "inv_hr": 1.0 / L,  # 1 / (radial cell size at r_m), 1/mm
            "inv_hz": g,
            "c": self.threshold(F["w"], F["rho"]),
            "w": F["w"],
            "rho": F["rho"],
            "L": L,
        }
        self.tab_fields = F

    # ------------------------------------------------------------------ interpolation
    def _index(self, theta: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Table indices for world angle theta in [0, 2 pi) and z (mm)."""
        tp = theta - self.seam
        tp = np.where(tp < 0.0, tp + TWO_PI, tp)
        tt = tp / self.d_theta
        i = np.floor(tt).astype(np.intp)
        i = np.clip(i, 0, self.n_t - 1)
        a = tt - i
        zc = np.clip(z, 0.0, self.L_ax)
        tz = zc / self.d_z
        j = np.minimum(np.floor(tz).astype(np.intp), self.n_z - 2)
        b = tz - j
        return i, j, a, b

    def interp(self, name: str, i, j, a, b) -> np.ndarray:
        T = self.tab[name]
        return ((1 - a) * (1 - b) * T[i, j] + a * (1 - b) * T[i + 1, j]
                + (1 - a) * b * T[i, j + 1] + a * b * T[i + 1, j + 1])

    def fields_at(self, theta: Any, z: Any) -> dict[str, np.ndarray]:
        """Tabulated (interpolated) fields at world angle theta [rad] and z [mm]."""
        theta = np.mod(np.asarray(theta, float), TWO_PI)
        z = np.asarray(z, float)
        i, j, a, b = self._index(*np.broadcast_arrays(theta, z))
        return {k: self.interp(k, i, j, a, b) for k in self.tab}

    # ------------------------------------------------------------------ frames
    def to_world(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.mode != "sector":
            return x, y
        c, s = math.cos(self.sector_center), math.sin(self.sector_center)
        return c * x - s * y, s * x + c * y

    # ------------------------------------------------------------------ pieces (world frame)
    @staticmethod
    def _theta(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        th = np.arctan2(y, x)
        return np.where(th < 0.0, th + TWO_PI, th)

    def lattice_level(self, x: Any, y: Any, z: Any) -> dict[str, np.ndarray]:
        """phi, threshold c, phase coordinates and the sd estimate of the bare (unclipped) lattice."""
        x, y, z = (np.asarray(v, float) for v in np.broadcast_arrays(x, y, z))
        r = np.hypot(x, y)
        th = self._theta(x, y)
        i, j, a, b = self._index(th, z)
        ut = self.interp("U_t", i, j, a, b)
        uz = self.interp("U_z", i, j, a, b)
        iht = self.interp("inv_ht", i, j, a, b)
        ihr = self.interp("inv_hr", i, j, a, b)
        ihz = self.interp("inv_hz", i, j, a, b)
        c = self.interp("c", i, j, a, b)
        w = self.interp("w", i, j, a, b)
        rs = np.maximum(r, 1e-9)
        st = iht * self.r_m / rs
        if self.conformal:
            ur = self.phase_r0 + ihr * self.r_m * np.log(rs / self.r_li)
            sr = ihr * self.r_m / rs
        else:
            ur = self.phase_r0 + ihr * (r - self.r_li)
            sr = ihr
        X, Y, Z = TWO_PI * ur, TWO_PI * ut, TWO_PI * uz
        sx, cx = np.sin(X), np.cos(X)
        sy, cy = np.sin(Y), np.cos(Y)
        sz, cz = np.sin(Z), np.cos(Z)
        G = sx * cy + sy * cz + sz * cx
        Gx = cx * cy - sz * sx
        Gy = -sx * sy + cy * cz
        Gz = -sy * sz + cz * cx
        D = sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz
        Dx = cx * sy * sz + cx * cy * cz - sx * sy * cz - sx * cy * sz
        Dy = sx * cy * sz - sx * sy * cz + cx * cy * cz - cx * sy * sz
        Dz = sx * sy * cz - sx * cy * sz - cx * sy * sz + cx * cy * cz
        wg, wd = (1.0 - w) / G_AMP, w / D_AMP
        phi = wg * G + wd * D
        gx = (wg * Gx + wd * Dx) * TWO_PI * sr
        gy = (wg * Gy + wd * Dy) * TWO_PI * st
        gz = (wg * Gz + wd * Dz) * TWO_PI * ihz
        gn = np.sqrt(gx * gx + gy * gy + gz * gz)
        floor = _GRAD_FLOOR * TWO_PI * (sr + st + ihz) / 3.0
        sd = (phi - c) / np.maximum(gn, floor)
        return {"phi": phi, "c": c, "w": w, "u_r": ur, "u_t": ut, "u_z": uz, "sd": sd, "r": r, "theta": th}

    def _lattice_region(self, r, th, z) -> np.ndarray:
        d = self.delta
        reg = np.maximum(self.r_li - d - r, r - self.r_lo - d)
        if self.layout == "slots":
            reg = np.maximum(reg, np.maximum(-d - z, z - self.L_ax - d))
            for ang, half, z0, z1 in self.slots:
                dth = np.abs(np.mod(th - ang + math.pi, TWO_PI) - math.pi)
                slot_sd = np.maximum(r * (dth - half), np.maximum(z0 - z, z - z1))
                reg = np.maximum(reg, -slot_sd)
        else:
            reg = np.maximum(reg, np.maximum(-z, z - self.L_ax))
        return reg

    def _rib(self, r, th, z) -> np.ndarray:
        """Meridional seam rib (seam mode without a slot to hide the seam): a plate of thickness seam_rib_mm
        through the gap at theta_seam, over the lattice length (it fuses the two cut faces of the seam)."""
        d = self.delta
        dth = np.abs(np.mod(th - self.seam + math.pi, TWO_PI) - math.pi)
        rib = np.maximum(r * dth - 0.5 * self.rib_t, np.maximum(self.r_li - d - r, r - self.r_lo - d))
        if self.layout == "slots":
            return np.maximum(rib, np.maximum(-d - z, z - self.L_ax - d))
        return np.maximum(rib, np.maximum(-z, z - self.L_ax))

    def _housing(self, r, z) -> np.ndarray:
        zb = np.maximum(self.z_lo - z, z - self.z_hi)
        sleeve = np.maximum(np.maximum(self.r_s - r, r - self.r_li), zb)
        outer = np.maximum(np.maximum(self.r_lo - r, r - self.r_o), zb)
        rr = np.maximum(self.r_s - r, r - self.r_o)
        cap_lo = np.maximum(rr, np.maximum(self.z_lo - z, z - (self.z_lo + self.t_cap)))
        cap_hi = np.maximum(rr, np.maximum((self.z_hi - self.t_cap) - z, z - self.z_hi))
        return np.minimum(np.minimum(sleeve, outer), np.minimum(cap_lo, cap_hi))

    def _port_pieces(self, x, y, z, port: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        a = math.radians(float(port["angle_deg"]))
        ca, sa = math.cos(a), math.sin(a)
        dz = z - float(port["z_mm"])
        t = x * ca + y * sa
        px, py = x - t * ca, y - t * sa
        dperp = np.sqrt(px * px + py * py + dz * dz)
        rb, Rb, hb = float(port["bore_radius_mm"]), float(port["boss_radius_mm"]), float(port["boss_height_mm"])
        t_bore0 = 0.5 * (self.r_li + self.r_lo)  # starts inside the plenum
        t_boss0 = 0.5 * (self.r_lo + self.r_o)  # bottom buried in the outer wall, never in the gap
        t_top = self.r_o + hb
        bore = np.maximum(dperp - rb, np.maximum(t_bore0 - t, t - (t_top + 1.0)))
        boss = np.maximum(dperp - Rb, np.maximum(t_boss0 - t, t - t_top))
        return bore, boss

    def wedge(self, x: np.ndarray, y: np.ndarray, half_angle: float) -> np.ndarray:
        """Signed distance to the wedge |theta_local| <= half_angle (local sector frame, < 90 deg)."""
        sb, cb = math.sin(half_angle), math.cos(half_angle)
        return np.maximum(-x * sb + y * cb, -x * sb - y * cb)

    # ------------------------------------------------------------------ the implicit
    def signed_distance(self, x: Any, y: Any, z: Any, *, half_angle: float | None = None,
                        chunk: int = 1_000_000) -> np.ndarray:
        """Signed distance (mm) at points given in the render frame (local sector frame in ``sector`` mode).

        ``half_angle`` (rad) overrides the sector wedge (e.g. half angle + margin for the check volume);
        None -> the nominal sector in ``sector`` mode, no wedge in ``full`` mode; negative -> no wedge.
        """
        x, y, z = (np.asarray(v, float) for v in np.broadcast_arrays(x, y, z))
        shape = x.shape
        x, y, z = x.ravel(), y.ravel(), z.ravel()
        out = np.empty(x.size)
        if half_angle is None and self.mode == "sector":
            half_angle = self.sector_half
        if half_angle is not None and half_angle < 0:
            half_angle = None  # explicit "no wedge" (same convention as the C# SignedDistance)
        for k0 in range(0, x.size, chunk):
            sl = slice(k0, k0 + chunk)
            out[sl] = self._sd_chunk(x[sl], y[sl], z[sl], half_angle)
        return out.reshape(shape)

    def _sd_chunk(self, xl, yl, zl, half_angle):
        x, y = self.to_world(xl, yl)
        z = zl
        r = np.hypot(x, y)
        th = self._theta(x, y)
        clamp = self.sd_clamp
        reg = self._lattice_region(r, th, z)
        lat = reg.copy()
        near = reg < clamp
        if np.any(near):
            lv = self.lattice_level(x[near], y[near], z[near])
            lat[near] = np.maximum(lv["sd"], reg[near])
        solid = np.minimum(self._housing(r, z), lat)
        if self.rib:
            solid = np.minimum(solid, self._rib(r, th, z))
        bores = []
        for p in self.ports:
            bore, boss = self._port_pieces(x, y, z, p)
            solid = np.minimum(solid, boss)
            bores.append(bore)
        for bore in bores:
            solid = np.maximum(solid, -bore)
        if half_angle is not None:
            solid = np.maximum(solid, self.wedge(xl, yl, half_angle))
        return np.clip(solid, -clamp, clamp)

    # ------------------------------------------------------------------ analytic regions (for checks)
    def regions(self, x: Any, y: Any, z: Any, inset: float = 0.0) -> dict[str, np.ndarray]:
        """Boolean region masks at points in the render frame (used by the checks).

        envelope: inside the jacket's outer boundary, shrunk by ``inset`` mm (r_s + inset < r < r_o - inset,
        z_lo + inset < z < z_hi - inset), or in a bore - the domain whose non-solid voxels are coolant (the
        inset keeps ambient voxels touching the outer surface out of it);
        lattice: the lattice region (gap, minus plenums; no overlap);
        plenum_in / plenum_out: slot plenums or ring plenums; bore_in / bore_out: port bores;
        interface: within ``band`` (set by the caller via ``interface_band``) of an open lattice face.
        """
        x, y, z = (np.asarray(v, float) for v in np.broadcast_arrays(x, y, z))
        xw, yw = self.to_world(x, y)
        r = np.hypot(xw, yw)
        th = self._theta(xw, yw)
        gap = (r >= self.r_li) & (r <= self.r_lo)
        out: dict[str, np.ndarray] = {}
        inz = (z >= 0.0) & (z <= self.L_ax)
        if self.layout == "slots":
            pl = []
            for ang, half, z0, z1 in self.slots:
                dth = np.abs(np.mod(th - ang + math.pi, TWO_PI) - math.pi)
                pl.append(gap & (dth < half) & (z >= max(z0, 0.0)) & (z <= min(z1, self.L_ax)))
            out["plenum_in"], out["plenum_out"] = pl
            out["lattice"] = gap & inz & ~pl[0] & ~pl[1]
        else:
            h = float(self.ring["height_mm"])
            out["plenum_in"] = gap & (z < 0.0) & (z >= -h)
            out["plenum_out"] = gap & (z > self.L_ax) & (z <= self.L_ax + h)
            out["lattice"] = gap & inz
        out["bore_in"] = np.zeros_like(gap)
        out["bore_out"] = np.zeros_like(gap)
        for p in self.ports:
            bore, _ = self._port_pieces(xw, yw, z, p)
            out["bore_in" if p["name"] == "inlet" else "bore_out"] |= bore <= 0.0
        envelope = ((r > self.r_s + inset) & (r < self.r_o - inset)
                    & (z > self.z_lo + inset) & (z < self.z_hi - inset))
        out["envelope"] = envelope | out["bore_in"] | out["bore_out"]
        out["r"] = r
        out["theta"] = th
        return out

    def interface_distance(self, x: Any, y: Any, z: Any) -> np.ndarray:
        """Distance (mm) to the nearest *open* lattice face (slot sides / ring faces) - where clipping
        creates slivers. Large where there is none."""
        x, y, z = (np.asarray(v, float) for v in np.broadcast_arrays(x, y, z))
        xw, yw = self.to_world(x, y)
        r = np.hypot(xw, yw)
        th = self._theta(xw, yw)
        d = np.full(x.shape, np.inf)
        if self.layout == "slots":
            for ang, half, _z0, _z1 in self.slots:
                dth = np.abs(np.mod(th - ang + math.pi, TWO_PI) - math.pi)
                d = np.minimum(d, np.abs(r * (dth - half)))
        else:
            d = np.minimum(np.abs(z), np.abs(z - self.L_ax))
        return d

    # ------------------------------------------------------------------ diagnostics
    def describe(self) -> dict[str, Any]:
        """Phase-field diagnostics: cells around, circumferential size deviation, shear of u_z."""
        F = self.tab_fields
        Tz = self.cells_around
        dev = Tz / self.N_theta - 1.0 if self.circ_mode == "integer" else np.zeros_like(Tz)  # size = L (1 + dev)
        Uz = self.tab["U_z"]
        # shear of the axial iso-phase surfaces: |d u_z / ds| / |d u_z / dz| at r_m
        duds = np.gradient(Uz, self.d_theta * self.r_m, axis=0)
        dudz = self.tab["inv_hz"]
        shear = np.degrees(np.arctan(np.abs(duds) / dudz))
        Ut = self.tab["U_t"]
        dutdz = np.gradient(Ut, self.d_z, axis=1)
        dutds = self.tab["inv_ht"]
        shear_t = np.degrees(np.arctan(np.abs(dutdz) / dutds))
        gap_cells = (self.r_lo - self.r_li) / F["L"]
        stretch = Tz / self.N_theta if self.circ_mode == "integer" else np.ones_like(Tz)
        return {
            "circumferential_mode": self.circ_mode,
            "circumferential_stretch_min": float(stretch.min()),
            "circumferential_stretch_max": float(stretch.max()),
            "axial_phase_shear_deg_p95": float(np.percentile(shear, 95)),
            "circumferential_phase_shear_deg_p95": float(np.percentile(shear_t, 95)),
            "seam_angle_deg": math.degrees(self.seam),
            "seam_rib": bool(self.rib),
            "seam_rib_mm": self.rib_t if self.rib else 0.0,
            "integer_N_max_size_deviation": self.integer_deviation,
            "N_theta": int(self.N_theta),
            "cells_around_min": float(Tz.min()),
            "cells_around_max": float(Tz.max()),
            "circumferential_size_deviation_min": float(dev.min()),
            "circumferential_size_deviation_max": float(dev.max()),
            "axial_cells_min": float((Uz[:, -1] - Uz[:, 0]).min()),
            "axial_cells_max": float((Uz[:, -1] - Uz[:, 0]).max()),
            "axial_phase_shear_deg_max": float(shear.max()),
            "circumferential_phase_shear_deg_max": float(shear_t.max()),
            "axial_phase_origin_mm": self.z_ref,
            "radial_cells_min": float(gap_cells.min()),
            "radial_cells_max": float(gap_cells.max()),
            "conformal_size_ratio_inner": self.r_li / self.r_m,
            "conformal_size_ratio_outer": self.r_lo / self.r_m,
            "table": [self.n_t, self.n_z],
        }
