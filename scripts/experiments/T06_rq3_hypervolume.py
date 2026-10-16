"""T06 (table) - results/tables/T06_rq3_hypervolume.csv

Hypervolume (median, IQR over 5 seeds) at 50 / 100 / 200 evaluations and final budget; evaluations NSGA-II
needs to match qNEHVI at 200.

Producer: scripts/experiments/tables.py - paper tables T01-T09 from the committed result CSVs/JSONs
    python scripts/experiments/T06_rq3_hypervolume.py                      # full: < 10 s
    python scripts/experiments/T06_rq3_hypervolume.py --quick
    python scripts/experiments/T06_rq3_hypervolume.py --replot
    python scripts/experiments/T06_rq3_hypervolume.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("T06"))
