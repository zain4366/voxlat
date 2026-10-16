"""F25 (figure) - results/figures/task8_forchheimer.png

Inertial closure: (a) Forchheimer coefficient C_F vs porosity (Gajetti et al. 2025 fits and points, Savoldi
gyroid fit, Rathore points), jacket band; (b) Task 4 permeability vs the published fits; (c) inertial share of
the pressure drop vs flow rate (rho* = 0.35, L = 2/4/6 mm).

Producer: scripts/task8_literature_closures.py
    python scripts/experiments/F25_forchheimer.py                          # full: 5 s
    python scripts/experiments/F25_forchheimer.py --quick
    python scripts/experiments/F25_forchheimer.py --replot
    python scripts/experiments/F25_forchheimer.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F25"))
