"""T01 (table) - results/tables/T01_solver_verification.csv

Production estimators of the cell solvers: convergence order, raw and R(32, 64) errors against the
extrapolated reference (k_eff, C_eff, K) for gyroid and diamond.

Producer: scripts/experiments/tables.py - paper tables T01-T09 from the committed result CSVs/JSONs
    python scripts/experiments/T01_solver_verification.py                  # full: < 10 s
    python scripts/experiments/T01_solver_verification.py --quick
    python scripts/experiments/T01_solver_verification.py --replot
    python scripts/experiments/T01_solver_verification.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("T01"))
