"""Uniform logging for VoxLat modules and scripts."""

from __future__ import annotations

import logging
import sys

__all__ = ["get_logger"]

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str = "voxlat", level: int | str | None = None) -> logging.Logger:
    """Return a logger under the ``voxlat`` hierarchy with a single stream handler.

    The handler is attached once to the ``voxlat`` root logger, so calling this
    repeatedly (or from many modules) never duplicates output. ``name`` values
    that do not start with ``voxlat`` are nested under it (``"task2"`` ->
    ``"voxlat.task2"``). The level defaults to ``$VOXLAT_LOG_LEVEL`` or INFO.
    """
    import os

    root = logging.getLogger("voxlat")
    if not any(getattr(h, "_voxlat_handler", False) for h in root.handlers):
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        handler._voxlat_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
        root.propagate = False
        root.setLevel(os.environ.get("VOXLAT_LOG_LEVEL", "INFO").upper())

    full = name if name == "voxlat" or name.startswith("voxlat.") else f"voxlat.{name}"
    logger = logging.getLogger(full)
    if level is not None:
        logger.setLevel(level.upper() if isinstance(level, str) else level)
    return logger
