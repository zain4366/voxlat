"""T04 (table) - results/tables/T04_literature_closures.csv

TPMS heat-transfer and inertial correlations considered (source, TPMS type, method, validity ranges).

Producer: scripts/experiments/tables.py - paper tables T01-T09 from the committed result CSVs/JSONs
    python scripts/experiments/T04_literature_closures.py                  # full: < 10 s
    python scripts/experiments/T04_literature_closures.py --quick
    python scripts/experiments/T04_literature_closures.py --replot
    python scripts/experiments/T04_literature_closures.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("T04"))
