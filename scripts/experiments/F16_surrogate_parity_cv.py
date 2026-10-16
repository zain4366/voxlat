"""F16 (figure) - results/figures/task6_parity_cv.png

Five-fold cross-validated parity of the GP and deep-ensemble closure surrogates with recalibrated 95 %
intervals.

Producer: scripts/task6_evaluate.py
    python scripts/experiments/F16_surrogate_parity_cv.py                  # full: ~10 min (2 cores)
    python scripts/experiments/F16_surrogate_parity_cv.py --quick
    python scripts/experiments/F16_surrogate_parity_cv.py --replot
    python scripts/experiments/F16_surrogate_parity_cv.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F16"))
