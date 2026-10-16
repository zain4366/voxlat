"""F23 (figure) - results/figures/task7_graded_error.png

Error of the homogenized prediction in strips with linear density grading vs the grading gradient |grad rho*|
L.

Producer: scripts/task7_figures.py
    python scripts/experiments/F23_finite_gap_graded.py                    # full: ~4 min (velocity slices n = 48)
    python scripts/experiments/F23_finite_gap_graded.py --quick
    python scripts/experiments/F23_finite_gap_graded.py --replot
    python scripts/experiments/F23_finite_gap_graded.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F23"))
