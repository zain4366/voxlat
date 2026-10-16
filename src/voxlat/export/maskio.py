"""Bit-packed voxel masks shared by the C# exporter (PicoGK slices) and the numpy tools (Task 12).

Format (written by ``JacketBuilder.WriteMask`` in C# and by ``write_mask`` here):

* ``<stem>.json`` - header: schema ``voxlat-mask/1``, ``shape`` [nx, ny, nz], ``origin_mm`` (centre of
  voxel (0, 0, 0) in the render frame), ``voxel_mm``, ``frame`` (render mode, sector centre / half
  angle), ``source`` (picogk | numpy | stl), ``kind`` (raw | clean), free ``meta``;
* ``<stem>.bin`` - nz slices; slice k holds nx * ny bits (x fastest, then y), bit = 1 for solid,
  little-endian bit order inside each byte, each slice padded to whole bytes.

Rows are stored with y ascending (``"y_order": "ascending"``). PicoGK's native slice buffer runs from
y = max down to y = min; the C# writer flips it, and masks written before that fix (no ``y_order``
key, source picogk) are flipped on read.

PicoGK voxel (i, j, k) of a field with origin (I0, J0, K0) (``Voxels.GetVoxelDimensions``) has its
centre at ((I0 + i) h, (J0 + j) h, (K0 + k) h) mm; the numpy generator uses the same integer lattice so
both masks can be compared voxel by voxel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["VoxelMask", "read_mask", "write_mask", "MASK_SCHEMA"]

MASK_SCHEMA = "voxlat-mask/1"


@dataclass
class VoxelMask:
    """Boolean voxel mask ``solid[nx, ny, nz]`` (True = solid) on a regular grid (mm)."""

    solid: np.ndarray
    origin: tuple[float, float, float]
    voxel: float
    frame: dict[str, Any] = field(default_factory=dict)
    source: str = "numpy"
    kind: str = "raw"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.solid.shape)  # type: ignore[return-value]

    def axis(self, k: int) -> np.ndarray:
        """Voxel-centre coordinates (mm) along axis k."""
        return self.origin[k] + self.voxel * np.arange(self.solid.shape[k])

    def centres(self, sl: tuple[slice, slice, slice] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Broadcastable centre coordinates (x[:,None,None], y[None,:,None], z[None,None,:]) of a sub-block."""
        sl = sl or (slice(None),) * 3
        x = self.axis(0)[sl[0]][:, None, None]
        y = self.axis(1)[sl[1]][None, :, None]
        z = self.axis(2)[sl[2]][None, None, :]
        return x, y, z

    def volume_mm3(self) -> float:
        return float(np.count_nonzero(self.solid)) * self.voxel**3


def write_mask(mask: VoxelMask, stem: str | Path) -> tuple[Path, Path]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = mask.shape
    with open(stem.with_suffix(".bin"), "wb") as fh:
        for k in range(nz):
            sl = np.ascontiguousarray(mask.solid[:, :, k].T).ravel()  # (ny, nx) -> x fastest
            fh.write(np.packbits(sl, bitorder="little").tobytes())
    header = {
        "schema": MASK_SCHEMA,
        "shape": [nx, ny, nz],
        "origin_mm": [float(v) for v in mask.origin],
        "voxel_mm": float(mask.voxel),
        "frame": mask.frame,
        "source": mask.source,
        "kind": mask.kind,
        "order": "slices along z; inside a slice x fastest then y; bit 1 = solid; little-endian bits",
        "y_order": "ascending",
        "meta": mask.meta,
    }
    stem.with_suffix(".json").write_text(json.dumps(header, indent=1))
    return stem.with_suffix(".json"), stem.with_suffix(".bin")


def read_mask(path: str | Path) -> VoxelMask:
    """Read a mask from its header (``.json``) or data (``.bin``) path or the common stem."""
    p = Path(path)
    stem = p.with_suffix("") if p.suffix in (".json", ".bin") else p
    header = json.loads(stem.with_suffix(".json").read_text())
    if header.get("schema") != MASK_SCHEMA:
        raise ValueError(f"unsupported mask schema {header.get('schema')!r}")
    nx, ny, nz = (int(v) for v in header["shape"])
    per = (nx * ny + 7) // 8
    raw = np.fromfile(stem.with_suffix(".bin"), dtype=np.uint8)
    if raw.size != per * nz:
        raise ValueError(f"mask data has {raw.size} bytes, expected {per * nz}")
    solid = np.empty((nx, ny, nz), dtype=bool)
    raw = raw.reshape(nz, per)
    step = max(1, 8_000_000 // max(nx * ny, 1))
    for k0 in range(0, nz, step):
        bits = np.unpackbits(raw[k0:k0 + step], axis=1, bitorder="little")[:, : nx * ny]
        solid[:, :, k0:k0 + step] = bits.reshape(-1, ny, nx).transpose(2, 1, 0).astype(bool)
    if header.get("source") == "picogk" and "y_order" not in header:
        # masks written before the fix: rows were stored in PicoGK's slice order (y descending)
        solid = solid[:, ::-1, :]
    return VoxelMask(solid=solid, origin=tuple(float(v) for v in header["origin_mm"]), voxel=float(header["voxel_mm"]),
                     frame=header.get("frame", {}), source=header.get("source", "?"), kind=header.get("kind", "raw"),
                     meta=header.get("meta", {}))
