"""F04 (figure) - results/figures/task1_morphology_vs_density.png

Minimum wall thickness, minimum pore size and surface area vs relative density for gyroid, diamond and blends
(n = 48 periodic distance transform). Blends develop pinch-point necks (non-monotonic minimum wall).

Producer: scripts/task1_tpms_figures.py
    python scripts/experiments/F04_morphology_vs_density.py                # full: 1-2 min
    python scripts/experiments/F04_morphology_vs_density.py --quick
    python scripts/experiments/F04_morphology_vs_density.py --replot
    python scripts/experiments/F04_morphology_vs_density.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F04"))
