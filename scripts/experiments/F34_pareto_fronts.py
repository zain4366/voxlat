"""F34 (figure) - results/figures/task11_pareto_fronts.png

Pareto fronts (R, pump power, mass at 3 L/min) of the graded lattices on both manifold layouts with baselines
B1-B3 and uniform gyroid/diamond references overlaid.

Producer: scripts/task11_analysis.py - reads the committed runs in results/task11/
    python scripts/experiments/F34_pareto_fronts.py                        # full: ~5 min
    python scripts/experiments/F34_pareto_fronts.py --quick
    python scripts/experiments/F34_pareto_fronts.py --replot
    python scripts/experiments/F34_pareto_fronts.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F34"))
