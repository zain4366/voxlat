"""T03 (table) - results/tables/T03_finite_gap_coefficients.csv

Wall coefficients a of the finite-gap correction delta = a/N per morphology and density (mean over cut phases,
phase standard deviation).

Producer: scripts/experiments/tables.py - paper tables T01-T09 from the committed result CSVs/JSONs
    python scripts/experiments/T03_finite_gap_coefficients.py              # full: < 10 s
    python scripts/experiments/T03_finite_gap_coefficients.py --quick
    python scripts/experiments/T03_finite_gap_coefficients.py --replot
    python scripts/experiments/T03_finite_gap_coefficients.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("T03"))
