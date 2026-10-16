"""F24 (figure) - results/figures/task7_convergence.png

Resolution and wall-thickness sensitivity of the finite-gap discrepancy.

Producer: scripts/task7_figures.py
    python scripts/experiments/F24_finite_gap_convergence.py               # full: ~4 min (velocity slices n = 48)
    python scripts/experiments/F24_finite_gap_convergence.py --quick
    python scripts/experiments/F24_finite_gap_convergence.py --replot
    python scripts/experiments/F24_finite_gap_convergence.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F24"))
