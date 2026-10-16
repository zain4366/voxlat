"""Saving figures the same way everywhere: ``results/figures/<name>.png`` at 300 dpi."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voxlat.utils.paths import figures_dir

__all__ = ["save_figure", "FIGURE_DPI"]

FIGURE_DPI = 300


def save_figure(
    fig: Any,
    name: str,
    directory: str | Path | None = None,
    *,
    dpi: int = FIGURE_DPI,
    close: bool = True,
    **savefig_kwargs: Any,
) -> Path:
    """Save a Matplotlib figure as PNG and return its path.

    Parameters
    ----------
    fig:
        ``matplotlib.figure.Figure``.
    name:
        File stem (``".png"`` is added if missing). Use the script name as a
        prefix, e.g. ``"task2_keff_vs_rho"``.
    directory:
        Output folder; default ``<repo>/results/figures``. Created if needed.
    dpi:
        Resolution, 300 by default (project convention).
    close:
        Close the figure afterwards to free memory in long batch scripts.
    """
    out_dir = Path(directory) if directory is not None else figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name[:-4] if name.lower().endswith(".png") else name
    path = out_dir / f"{stem}.png"
    kwargs = {"bbox_inches": "tight", "facecolor": "white"}
    kwargs.update(savefig_kwargs)
    fig.savefig(path, dpi=dpi, format="png", **kwargs)
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return path
