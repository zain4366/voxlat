"""F33 (figure) - results/figures/task11_hypervolume.png

Sample efficiency (RQ3): normalized hypervolume vs evaluations for NSGA-II, qNEHVI and Sobol search, median
and IQR over 5 seeds, slot and end-ring manifolds.

Producer: scripts/task11_analysis.py - reads the committed runs in results/task11/
    python scripts/experiments/F33_hypervolume.py                          # full: ~5 min
    python scripts/experiments/F33_hypervolume.py --quick
    python scripts/experiments/F33_hypervolume.py --replot
    python scripts/experiments/F33_hypervolume.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F33"))
