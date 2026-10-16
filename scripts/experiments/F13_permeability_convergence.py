"""F13 (figure) - results/figures/task4_convergence.png

Grid convergence of the permeability K(n)/K_ref vs 1/n with n = 96 anchors, estimator errors and tortuosity.

Producer: scripts/task4_convergence.py
    python scripts/experiments/F13_permeability_convergence.py             # full: ~11 min (2 cores, 1.5 GB)
    python scripts/experiments/F13_permeability_convergence.py --quick
    python scripts/experiments/F13_permeability_convergence.py --replot
    python scripts/experiments/F13_permeability_convergence.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F13"))
