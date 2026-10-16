"""Locate the repository root and its standard folders.

The package uses a src-layout and is meant to be installed in editable mode
(``pip install -e .``), so the repository folders (``configs/``, ``data/``,
``results/``...) are found by walking up from this file. The environment
variable ``VOXLAT_ROOT`` overrides the search (useful for non-editable installs
or running scripts from elsewhere).
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["repo_root", "configs_dir", "data_dir", "models_dir", "results_dir", "figures_dir"]


def repo_root() -> Path:
    """Return the VoxLat repository root.

    Search order: ``$VOXLAT_ROOT``; the first parent of this file that contains
    both ``pyproject.toml`` and ``configs/``; finally the current working
    directory.
    """
    env = os.environ.get("VOXLAT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "configs").is_dir():
            return parent
    return Path.cwd().resolve()


def configs_dir() -> Path:
    return repo_root() / "configs"


def data_dir() -> Path:
    return repo_root() / "data"


def models_dir() -> Path:
    return repo_root() / "models"


def results_dir() -> Path:
    return repo_root() / "results"


def figures_dir() -> Path:
    return results_dir() / "figures"
