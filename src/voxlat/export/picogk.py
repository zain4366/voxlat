"""Python driver for the C# exporter (pythonnet + coreclr, PicoGK 1.7) - Task 12.

All geometry evaluation runs in C# (``VoxLat.Export.dll``); Python only writes the export config, loads
the assemblies and calls one C# entry point:

* ``JacketBuilder.RunWithViewer(config, log_dir, keep_open)`` - the ``Library.Go`` pattern of
  ``legacy/noyron_core.py`` (PicoGK viewer window; needs PicoGK >= 1.7.6 for the ``bEndAppWithTask``
  argument, which the legacy script already uses);
* ``JacketBuilder.RunHeadless(config)`` - ``new PicoGK.Library(voxel)`` without a viewer (batch use, CI).

Loading follows the working legacy recipe: ``load("coreclr")``, add the PicoGK folder to the DLL search
path, ``NativeLibrary.Load(picogk.1.7.dll)`` before ``clr.AddReference(PicoGK.dll)``.

``probe`` evaluates the C# implicit at arbitrary points (only ``VoxLat.Export.Core.dll``; no PicoGK
native library needed) - used by ``scripts/task12_export.py selftest`` to compare C# with numpy.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_PICOGK_DIR",
    "dotnet_dir",
    "default_build_dir",
    "load_dotnet",
    "run_picogk",
    "probe",
    "probe_fields",
    "build_command",
]

DEFAULT_PICOGK_DIR = r"D:\nyron 2\PicoGK_Engine\bin\Release\net9.0"
NATIVE_NAME = "picogk.1.7.dll"


def dotnet_dir() -> Path:
    """Folder with the C# sources (``src/voxlat/export/dotnet``)."""
    return Path(__file__).resolve().parent / "dotnet"


def default_build_dir(project: str = "VoxLat.Export", configuration: str = "Release") -> Path:
    return dotnet_dir() / project / "bin" / configuration / "net9.0"


def build_command(picogk_dir: str | Path = DEFAULT_PICOGK_DIR) -> list[str]:
    """``dotnet build`` command for the PicoGK-side library (also builds the Core library)."""
    return ["dotnet", "build", str(dotnet_dir() / "VoxLat.Export" / "VoxLat.Export.csproj"), "-c", "Release",
            f"-p:PicoGKDir={picogk_dir}"]


_LOADED: dict[str, Any] = {}


def load_dotnet(*, picogk_dir: str | Path | None = DEFAULT_PICOGK_DIR, build_dir: str | Path | None = None,
                native: bool = True, core_only: bool = False) -> Any:
    """Start coreclr (once), load PicoGK (native + managed) and the VoxLat export assemblies.

    Returns the ``VoxLat.Export`` .NET namespace module (``JacketBuilder``, ``Probe``, ...).
    ``core_only``: load only ``VoxLat.Export.Core.dll`` (probe/selftest; no PicoGK needed).
    """
    if "ns" in _LOADED and (core_only or _LOADED.get("full")):
        return _LOADED["ns"]
    try:
        from pythonnet import load
    except ImportError as exc:  # pragma: no cover - depends on the user's environment
        raise RuntimeError("pythonnet is required: pip install pythonnet  (or pip install -e .[export])") from exc
    if "clr" not in _LOADED:
        try:
            load("coreclr")
        except RuntimeError:
            pass  # already loaded in this process
        import clr  # noqa: F401  (pythonnet)

        _LOADED["clr"] = clr
    clr = _LOADED["clr"]
    bdir = Path(build_dir) if build_dir else default_build_dir()
    if core_only:
        core = bdir / "VoxLat.Export.Core.dll"
        if not core.is_file():
            alt = default_build_dir("VoxLat.Export.Probe") / "VoxLat.Export.Core.dll"
            core = alt if alt.is_file() else core
        if not core.is_file():
            raise FileNotFoundError(f"{core} not found - build the C# library first: {' '.join(build_command())}")
        clr.AddReference(str(core))
    else:
        pdir = Path(picogk_dir) if picogk_dir else Path(DEFAULT_PICOGK_DIR)
        managed = pdir / "PicoGK.dll"
        engine = pdir / NATIVE_NAME
        if not managed.is_file():
            raise FileNotFoundError(f"PicoGK.dll not found in {pdir}")
        if native:
            if not engine.is_file():
                raise FileNotFoundError(f"{NATIVE_NAME} not found in {pdir}")
            if sys.platform == "win32":
                os.add_dll_directory(str(pdir))
            from System.Runtime.InteropServices import NativeLibrary  # type: ignore[import-not-found]

            NativeLibrary.Load(str(engine))
        clr.AddReference(str(managed))
        dll = bdir / "VoxLat.Export.dll"
        if not dll.is_file():
            raise FileNotFoundError(f"{dll} not found - build it first: {' '.join(build_command(pdir))}")
        clr.AddReference(str(dll))
        _LOADED["full"] = True
    import VoxLat.Export as ns  # type: ignore[import-not-found]

    _LOADED["ns"] = ns
    return ns


def run_picogk(config_path: str | Path, *, picogk_dir: str | Path = DEFAULT_PICOGK_DIR,
               build_dir: str | Path | None = None, viewer: bool = True, keep_viewer_open: bool = False,
               log_dir: str | Path | None = None) -> dict[str, Any]:
    """Build the jacket in PicoGK from an export-config JSON; returns the C# report (dict)."""
    config_path = Path(config_path).resolve()
    ns = load_dotnet(picogk_dir=picogk_dir, build_dir=build_dir)
    if viewer:
        log_dir = Path(log_dir or config_path.parent)
        report_path = ns.JacketBuilder.RunWithViewer(str(config_path), str(log_dir), bool(keep_viewer_open))
    else:
        report_path = ns.JacketBuilder.RunHeadless(str(config_path))
    report_path = str(report_path)
    if not report_path:
        raise RuntimeError("PicoGK build returned no report (see the log above)")
    rep = json.loads(Path(report_path).read_text())
    rep["_report_path"] = report_path
    return rep


def _probe_exe() -> Path | None:
    d = default_build_dir("VoxLat.Export.Probe")
    for name in ("VoxLat.Export.Probe.dll",):
        if (d / name).is_file():
            return d / name
    return None


def _run_probe(args: list[str], strict: bool):
    """Run the probe exe through ``dotnet``; on failure raise (strict) or warn and return None (auto -> pythonnet)."""
    env = dict(os.environ)
    env.setdefault("DOTNET_ROLL_FORWARD", "Major")  # builds for net9.0 also run when only .NET 10 is installed
    res = subprocess.run(["dotnet", *args], capture_output=True, text=True, env=env)
    if res.returncode == 0:
        return res
    msg = (f"VoxLat.Export.Probe failed (exit {res.returncode} = 0x{res.returncode & 0xFFFFFFFF:08X}):\n"
           f"{(res.stderr or res.stdout).strip()[-2000:]}")
    if strict:
        raise RuntimeError(msg)
    print("warning: " + msg + "\n-> falling back to pythonnet (in-process)")
    return None


def probe(config_path: str | Path, points: np.ndarray, *, half_angle_deg: float | None = None,
          method: str = "auto", build_dir: str | Path | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """C# signed distance at points (N x 3, render frame, mm). method: "pythonnet", "exe" or "auto".

    ``half_angle_deg`` None -> the config's default wedge (nominal sector / none in full mode);
    negative -> no wedge. Returns (sd, C# field diagnostics).
    """
    pts = np.ascontiguousarray(np.asarray(points, dtype="<f8").reshape(-1, 3))
    half = float("nan") if half_angle_deg is None else float(half_angle_deg)
    with tempfile.TemporaryDirectory() as tmp:
        pin, pout = Path(tmp) / "pts.bin", Path(tmp) / "sd.bin"
        pts.tofile(pin)
        use_exe = method == "exe" or (method == "auto" and _probe_exe() is not None and shutil.which("dotnet"))
        diag = None
        if use_exe:
            exe = _probe_exe()
            if exe is None:
                raise FileNotFoundError("VoxLat.Export.Probe not built (dotnet build ...VoxLat.Export.Probe.csproj)")
            arg = "nan" if half_angle_deg is None else repr(half)
            res = _run_probe([str(exe), "sd", str(config_path), str(pin), str(pout), arg], strict=method == "exe")
            if res is not None:
                diag = json.loads(res.stdout.strip().splitlines()[-1])
        if diag is None:
            ns = load_dotnet(core_only=True, build_dir=build_dir)
            diag = json.loads(str(ns.Probe.EvaluateFile(str(config_path), str(pin), str(pout), half)))
        sd = np.fromfile(pout, dtype="<f8")
    return sd, diag


def probe_fields(config_path: str | Path, theta: np.ndarray, z: np.ndarray, *, method: str = "auto",
                 build_dir: str | Path | None = None) -> np.ndarray:
    """C# tabulated fields (rho, w, L [mm], c) at world (theta [rad], z [mm]); returns N x 4."""
    tz = np.ascontiguousarray(np.column_stack([np.ravel(theta), np.ravel(z)]).astype("<f8"))
    with tempfile.TemporaryDirectory() as tmp:
        pin, pout = Path(tmp) / "tz.bin", Path(tmp) / "f.bin"
        tz.tofile(pin)
        use_exe = method == "exe" or (method == "auto" and _probe_exe() is not None and shutil.which("dotnet"))
        ok = False
        if use_exe:
            ok = _run_probe([str(_probe_exe()), "fields", str(config_path), str(pin), str(pout)],
                            strict=method == "exe") is not None
        if not ok:
            ns = load_dotnet(core_only=True, build_dir=build_dir)
            ns.Probe.FieldsFile(str(config_path), str(pin), str(pout))
        return np.fromfile(pout, dtype="<f8").reshape(-1, 4)
