"""F20 (figure) - results/figures/task7_error_vs_N.png

Main RQ1 result: relative error delta of the bulk-homogenized prediction vs 1/N (N cells across a wall-bounded
gap) for through-gap conductivity k_n, in-plane permeability K_t, normal stiffness C_nn and shear moduli G_t,
G_sz; gyroid/diamond (top), blend (bottom); bars = range over cut phases, lines = physical correction delta =
a/N; shaded band = jacket design range.

Producer: scripts/task7_figures.py
    python scripts/experiments/F20_finite_gap_error_vs_N.py                # full: ~4 min (velocity slices n = 48)
    python scripts/experiments/F20_finite_gap_error_vs_N.py --quick
    python scripts/experiments/F20_finite_gap_error_vs_N.py --replot
    python scripts/experiments/F20_finite_gap_error_vs_N.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F20"))
