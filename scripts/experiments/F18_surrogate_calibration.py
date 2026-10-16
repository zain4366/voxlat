"""F18 (figure) - results/figures/task6_calibration.png

Calibration of the surrogate uncertainty: observed vs nominal interval coverage, raw and recalibrated, for the
cv5, band and sparse splits.

Producer: scripts/task6_evaluate.py
    python scripts/experiments/F18_surrogate_calibration.py                # full: ~10 min (2 cores)
    python scripts/experiments/F18_surrogate_calibration.py --quick
    python scripts/experiments/F18_surrogate_calibration.py --replot
    python scripts/experiments/F18_surrogate_calibration.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F18"))
