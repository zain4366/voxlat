"""F15 (figure) - results/figures/task5_closures_quicklook.png

Closure dataset (Task 5; 290 design points theta = (w, rho*, a_z), Sobol + boundary points): every closure vs
relative density, coloured by blend weight w.

Producer: scripts/build_dataset.py - Task 5 closure dataset data/closures.parquet (resumable)
    python scripts/experiments/F15_closures_quicklook.py                   # full: ~7.4 h (laptop, 4 workers; 27.8 CPU-h)
    python scripts/experiments/F15_closures_quicklook.py --quick
    python scripts/experiments/F15_closures_quicklook.py --replot
    python scripts/experiments/F15_closures_quicklook.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F15"))
