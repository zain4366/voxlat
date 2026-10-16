"""F06 (figure) - results/figures/task2_keff_vs_density.png

Effective conductivity k_eff/k_s vs relative density for gyroid, diamond and a 50/50 blend (k_f/k_s = 1/325
and k_f = 0), with Hashin-Shtrikman bounds; panel (b) fraction of the HS upper bound. Two-grid Richardson
estimator R(32, 64).

Producer: scripts/task2_keff_vs_density.py
    python scripts/experiments/F06_keff_vs_density.py                      # full: ~5 min (2 cores)
    python scripts/experiments/F06_keff_vs_density.py --quick
    python scripts/experiments/F06_keff_vs_density.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F06"))
