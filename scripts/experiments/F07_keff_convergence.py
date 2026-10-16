"""F07 (figure) - results/figures/task2_convergence.png

Grid convergence of k_eff: offset-0 grids vs 1/n and offset-averaged error (log-log, slope -1); basis for the
R(32, 64) production estimator.

Producer: scripts/task2_convergence.py
    python scripts/experiments/F07_keff_convergence.py                     # full: ~15 min (2 cores)
    python scripts/experiments/F07_keff_convergence.py --quick
    python scripts/experiments/F07_keff_convergence.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F07"))
