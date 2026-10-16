"""F21 (figure) - results/figures/task7_discrepancy_models.png

Cross-validated error of the discrepancy models: uncorrected, physical a/N form, physical + GP on the
residuals, and the cut-phase noise floor.

Producer: scripts/task7_figures.py
    python scripts/experiments/F21_finite_gap_models.py                    # full: ~4 min (velocity slices n = 48)
    python scripts/experiments/F21_finite_gap_models.py --quick
    python scripts/experiments/F21_finite_gap_models.py --replot
    python scripts/experiments/F21_finite_gap_models.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F21"))
