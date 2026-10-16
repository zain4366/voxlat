"""T02 (table) - results/tables/T02_surrogate_accuracy.csv

Closure surrogate accuracy (R^2, MAPE, 95 % coverage) for the GP and deep ensemble on the cv5, band and sparse
splits.

Producer: scripts/experiments/tables.py - paper tables T01-T09 from the committed result CSVs/JSONs
    python scripts/experiments/T02_surrogate_accuracy.py                   # full: < 10 s
    python scripts/experiments/T02_surrogate_accuracy.py --quick
    python scripts/experiments/T02_surrogate_accuracy.py --replot
    python scripts/experiments/T02_surrogate_accuracy.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("T02"))
