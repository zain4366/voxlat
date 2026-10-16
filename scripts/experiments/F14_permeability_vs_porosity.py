"""F14 (figure) - results/figures/task4_permeability_vs_porosity.png

Permeability K/L^2 vs porosity for gyroid, diamond and the 50/50 blend (principal-value band), Kozeny constant
and tortuosity; R(32, 64) estimator.

Producer: scripts/task4_permeability_vs_porosity.py
    python scripts/experiments/F14_permeability_vs_porosity.py             # full: ~9 min (2 cores)
    python scripts/experiments/F14_permeability_vs_porosity.py --quick
    python scripts/experiments/F14_permeability_vs_porosity.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F14"))
