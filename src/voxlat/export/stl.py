"""Binary STL IO and a pure-numpy re-voxelizer (Task 12).

``voxelize_stl`` turns a closed triangle mesh (e.g. the PicoGK STL) back into a ``VoxelMask`` on the
same integer voxel lattice as PicoGK and the numpy generator (centre of voxel k at k h mm), by ray
parity along z: every triangle adds a crossing to each voxel column whose (x, y) centre it covers, and
a voxel is solid if an odd number of crossings lies below its centre. Column centres are offset by a
tiny irrational fraction of a voxel so that no ray hits an edge or vertex exactly.

This is the fallback when only an STL is available; the C# exporter writes masks directly from the
PicoGK voxel field, which is faster and exact with respect to the printed geometry.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from voxlat.export.maskio import VoxelMask

__all__ = ["read_stl", "write_stl", "voxelize_stl"]

_JITTER = (1.2345e-4, 2.3456e-4)  # in voxels


def read_stl(path: str | Path) -> np.ndarray:
    """Triangles (n, 3, 3) float64 of a binary STL (ASCII is detected and parsed too)."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * n == len(data):
            rec = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
            arr = np.frombuffer(data, dtype=rec, count=n, offset=84)
            return arr["v"].astype(np.float64)
    text = data.decode("ascii", errors="ignore")
    if "facet" not in text:
        raise ValueError(f"{path}: not a binary or ASCII STL")
    vals = [line.split()[1:4] for line in text.splitlines() if line.strip().startswith("vertex")]
    return np.asarray(vals, dtype=np.float64).reshape(-1, 3, 3)


def write_stl(path: str | Path, tris: np.ndarray) -> None:
    tris = np.asarray(tris, dtype=np.float32).reshape(-1, 3, 3)
    n = len(tris)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.where(ln > 0, nrm / np.maximum(ln, 1e-30), 0).astype(np.float32)
    rec = np.zeros(n, dtype=[("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["normal"] = nrm
    rec["v"] = tris
    with open(path, "wb") as fh:
        fh.write(b"voxlat binary stl".ljust(80, b" "))
        fh.write(struct.pack("<I", n))
        fh.write(rec.tobytes())


def voxelize_stl(tris: np.ndarray | str | Path, voxel: float, *, bbox: tuple[np.ndarray, np.ndarray] | None = None,
                 chunk: int = 200_000, frame: dict | None = None) -> VoxelMask:
    """Solid mask of a closed mesh on the voxel lattice k * voxel (mm)."""
    if not isinstance(tris, np.ndarray):
        tris = read_stl(tris)
    h = float(voxel)
    lo, hi = (tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)) if bbox is None else bbox
    i0 = np.floor(np.asarray(lo) / h).astype(int) - 1
    i1 = np.ceil(np.asarray(hi) / h).astype(int) + 1
    nx, ny, nz = (int(v) for v in i1 - i0 + 1)
    toggle = np.zeros((nx, ny, nz + 1), dtype=np.uint8)
    jx, jy = _JITTER
    # vertex coordinates in (jittered) column-index units: column (i, j) centre sits at integer (i, j)
    for c0 in range(0, len(tris), chunk):
        T = tris[c0:c0 + chunk]
        X = T[:, :, 0] / h - i0[0] - jx
        Y = T[:, :, 1] / h - i0[1] - jy
        Z = T[:, :, 2]
        xmin = np.ceil(X.min(1)).astype(int)
        xmax = np.floor(X.max(1)).astype(int)
        ymin = np.ceil(Y.min(1)).astype(int)
        ymax = np.floor(Y.max(1)).astype(int)
        w = xmax - xmin + 1
        hgt = ymax - ymin + 1
        ok = (w > 0) & (hgt > 0)
        if not ok.any():
            continue
        X, Y, Z = X[ok], Y[ok], Z[ok]
        xmin, ymin, w, hgt = xmin[ok], ymin[ok], w[ok], hgt[ok]
        x0, y0 = X[:, 0], Y[:, 0]
        e1x, e1y = X[:, 1] - x0, Y[:, 1] - y0
        e2x, e2y = X[:, 2] - x0, Y[:, 2] - y0
        det = e1x * e2y - e2x * e1y
        good = np.abs(det) > 1e-14
        for di in range(int(w.max())):
            for dj in range(int(hgt.max())):
                sel = good & (di < w) & (dj < hgt)
                if not sel.any():
                    continue
                px = (xmin[sel] + di).astype(float)
                py = (ymin[sel] + dj).astype(float)
                dx, dy = px - x0[sel], py - y0[sel]
                d = det[sel]
                u = (dx * e2y[sel] - e2x[sel] * dy) / d
                v = (e1x[sel] * dy - dx * e1y[sel]) / d
                inside = (u >= 0) & (v >= 0) & (u + v <= 1)
                if not inside.any():
                    continue
                zs = Z[sel]
                zc = zs[:, 0] + u * (zs[:, 1] - zs[:, 0]) + v * (zs[:, 2] - zs[:, 0])
                ii = (xmin[sel] + di)[inside]
                jj = (ymin[sel] + dj)[inside]
                kk = np.ceil(zc[inside] / h - i0[2]).astype(int)  # first voxel centre above the crossing
                keep = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
                kk = np.clip(kk, 0, nz)
                np.add.at(toggle, (ii[keep], jj[keep], kk[keep]), 1)
    parity = np.cumsum(toggle, axis=2, dtype=np.uint8)[:, :, :nz] & 1
    solid = parity.astype(bool)
    origin = tuple(float(v) for v in i0 * h)
    return VoxelMask(solid=solid, origin=origin, voxel=h, frame=frame or {}, source="stl", kind="clean",
                     meta={"triangles": int(len(tris))})
