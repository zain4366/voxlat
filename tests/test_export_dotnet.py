"""Task 12 - the C# implicit (VoxLat.Export.Core) against the numpy mirror.

Needs the .NET SDK and the built probe:
    dotnet build src/voxlat/export/dotnet/VoxLat.Export.Probe/VoxLat.Export.Probe.csproj -c Release
Skipped otherwise (the GitHub workflow task12-export.yml builds and runs it on windows-latest).
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from voxlat.export.config import ExportOptions, build_export_config, load_design, write_export_config
from voxlat.export.field import JacketField
from voxlat.export.picogk import default_build_dir, probe, probe_fields
from voxlat.export.sector import render_bbox, wide_half_angle

ROOT = Path(__file__).resolve().parents[1]
PROBE = default_build_dir("VoxLat.Export.Probe") / "VoxLat.Export.Probe.dll"
pytestmark = pytest.mark.skipif(not PROBE.is_file() or shutil.which("dotnet") is None,
                                reason="C# probe not built (dotnet build ...VoxLat.Export.Probe.csproj -c Release)")

CASES = [
    ("task11_slots_best_at_cap.json", "sector", {}),
    ("task11_slots_best_at_cap.json", "full", {}),
    ("task11_end_rings_knee.json", "sector", {}),
    ("task11_end_rings_knee.json", "full", {"circumferential": "seam"}),
    ("task11_end_rings_min_R.json", "sector", {"conformal": False}),
]


def _setup(tmp_path, design, mode, extra):
    d = load_design(ROOT / "results" / "designs" / design)
    opts = ExportOptions(mode=mode, sector_center_deg=37.0, voxel_size_mm=0.1, **extra)
    cfg = build_export_config(d, opts)
    path = write_export_config(cfg, tmp_path / "cfg.json")
    return path, JacketField(cfg, d)


@pytest.mark.parametrize("design,mode,extra", CASES)
def test_signed_distance_matches_numpy(tmp_path, design, mode, extra):
    path, f = _setup(tmp_path, design, mode, extra)
    rng = np.random.default_rng(7)
    lo, hi = render_bbox(f, wide_half_angle(f) if mode == "sector" else None)
    p1 = lo + (hi - lo) * rng.random((40000, 3))
    th = (rng.random(40000) - 0.5) * 2 * (f.sector_half if mode == "sector" else math.pi)
    r = f.r_li - 0.3 + (f.r_lo - f.r_li + 0.6) * rng.random(40000)
    z = -1 + (f.L_ax + 2) * rng.random(40000)
    P = np.vstack([p1, np.column_stack([r * np.cos(th), r * np.sin(th), z])])
    sd_cs, diag = probe(path, P, method="exe")
    sd_np = f.signed_distance(P[:, 0], P[:, 1], P[:, 2])
    assert np.max(np.abs(sd_cs - sd_np)) < 1e-6
    assert diag["N_theta"] == f.N_theta
    assert diag["circumferential_mode"] == f.circ_mode
    # explicit wedge overrides
    sd_cs2, _ = probe(path, P[:2000], method="exe", half_angle_deg=-1.0)
    sd_np2 = f.signed_distance(P[:2000, 0], P[:2000, 1], P[:2000, 2], half_angle=-1.0)
    assert np.max(np.abs(sd_cs2 - sd_np2)) < 1e-6


def test_fields_match_numpy(tmp_path):
    path, f = _setup(tmp_path, "task11_slots_best_at_cap.json", "sector", {})
    rng = np.random.default_rng(8)
    th = rng.random(5000) * 2 * math.pi
    z = rng.random(5000) * 60 - 5
    got = probe_fields(path, th, z, method="exe")
    ref = f.fields_at(th, z)
    for k, name in enumerate(("rho", "w", "L", "c")):
        assert np.max(np.abs(got[:, k] - ref[name])) < 1e-9, name


def test_benchmark_runs(tmp_path):
    path, _ = _setup(tmp_path, "task11_slots_best_at_cap.json", "sector", {})
    import os

    env = dict(os.environ, DOTNET_ROLL_FORWARD=os.environ.get("DOTNET_ROLL_FORWARD", "Major"))
    out = subprocess.run(["dotnet", str(PROBE), "bench", str(path), "200000"], capture_output=True, text=True,
                         check=True, env=env)
    ns = json.loads(out.stdout.strip().splitlines()[-1])["ns_per_eval"]
    print(f"C# implicit: {ns:.0f} ns per evaluation (render box points)")
    assert ns > 0
