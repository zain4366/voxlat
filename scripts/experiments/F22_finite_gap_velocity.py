"""F22 (figure) - results/figures/task7_velocity_slices.png

Near-wall channelling: streamwise velocity slices and plane-averaged profiles in strips with N = 2 cells
across the gap (n = 48).

Producer: scripts/task7_figures.py
    python scripts/experiments/F22_finite_gap_velocity.py                  # full: ~4 min (velocity slices n = 48)
    python scripts/experiments/F22_finite_gap_velocity.py --quick
    python scripts/experiments/F22_finite_gap_velocity.py --replot
    python scripts/experiments/F22_finite_gap_velocity.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F22"))
