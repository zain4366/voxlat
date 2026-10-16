"""Run metadata attached to every computed record (``run_record``).

Every row VoxLat writes to ``data/*.parquet`` carries a run record so a number
can always be traced back to its inputs, code version and resolution.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from voxlat.utils.paths import repo_root

__all__ = ["run_record", "git_hash", "Stopwatch", "stopwatch", "to_jsonable"]


def git_hash(root: str | Path | None = None) -> str | None:
    """Short commit hash of the repository (``+dirty`` if uncommitted changes), or None.

    Returns None when git is not installed or the folder is not a git repository.
    """
    cwd = Path(root) if root is not None else repo_root()
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not head:
        return None
    return f"{head}+dirty" if dirty else head


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays, dataclasses, Paths and tuples to JSON types."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def run_record(
    params: Mapping[str, Any] | Any | None = None,
    *,
    resolution: int | tuple[int, ...] | None = None,
    wall_time: float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Metadata dict for one computation.

    Parameters
    ----------
    params:
        Inputs of the computation (mapping or dataclass); converted to JSON types.
    resolution:
        Voxels per cell edge ``n`` (or a shape tuple) used for the computation.
    wall_time:
        Elapsed seconds (see :class:`Stopwatch`).
    **extra:
        Anything else worth keeping (solver tolerances, iteration counts, ...).

    Returns
    -------
    dict with keys ``params``, ``params_json``, ``voxlat_version``, ``git_hash``,
    ``resolution``, ``wall_time_s``, ``timestamp_utc``, ``python``, ``numpy``,
    ``platform`` and ``extra``. ``params_json`` is a canonical (sorted) JSON
    string, convenient as a single parquet column.
    """
    from voxlat import __version__

    p = to_jsonable(params) if params is not None else {}
    res = list(resolution) if isinstance(resolution, tuple) else resolution
    return {
        "params": p,
        "params_json": json.dumps(p, sort_keys=True),
        "voxlat_version": __version__,
        "git_hash": git_hash(),
        "resolution": res,
        "wall_time_s": None if wall_time is None else float(wall_time),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(terse=True),
        "extra": to_jsonable(extra),
    }


class Stopwatch:
    """Minimal wall-clock timer: ``sw = Stopwatch(); ...; sw.elapsed``."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._t1: float | None = None

    def stop(self) -> float:
        self._t1 = time.perf_counter()
        return self.elapsed

    @property
    def elapsed(self) -> float:
        end = self._t1 if self._t1 is not None else time.perf_counter()
        return end - self._t0


@contextmanager
def stopwatch() -> Iterator[Stopwatch]:
    """``with stopwatch() as sw: ...`` then ``sw.elapsed`` (frozen at block exit)."""
    sw = Stopwatch()
    try:
        yield sw
    finally:
        sw.stop()
