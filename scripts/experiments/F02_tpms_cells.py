"""F02 (figure) - results/figures/task1_tpms_cells.png

Network TPMS unit cells at relative density 0.30: gyroid (w = 0), 50/50 blend (w = 0.5) and Schwarz diamond (w
= 1), rendered and sliced; solid where the blended level set phi_w <= c(w, rho*).

Producer: scripts/task1_tpms_figures.py
    python scripts/experiments/F02_tpms_cells.py                           # full: 1-2 min
    python scripts/experiments/F02_tpms_cells.py --quick
    python scripts/experiments/F02_tpms_cells.py --replot
    python scripts/experiments/F02_tpms_cells.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F02"))
