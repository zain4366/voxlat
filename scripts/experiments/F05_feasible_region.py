"""F05 (figure) - results/figures/task1_feasible_region.png

Manufacturable (rho*, L) region per blend weight w for min wall 0.35 mm and min pore 0.8 mm: the cell-size
floor L_min(w, rho*) used as the optimization constraint.

Producer: scripts/task1_tpms_figures.py
    python scripts/experiments/F05_feasible_region.py                      # full: 1-2 min
    python scripts/experiments/F05_feasible_region.py --quick
    python scripts/experiments/F05_feasible_region.py --replot
    python scripts/experiments/F05_feasible_region.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F05"))
