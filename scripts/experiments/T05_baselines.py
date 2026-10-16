"""T05 (table) - results/tables/T05_baselines.csv

Baselines B1, B1s, B2, B3 at 1, 3 and 5 L/min: R, R_active, pressure drop, pump power, mass, margin.

Producer: scripts/experiments/tables.py - paper tables T01-T09 from the committed result CSVs/JSONs
    python scripts/experiments/T05_baselines.py                            # full: < 10 s
    python scripts/experiments/T05_baselines.py --quick
    python scripts/experiments/T05_baselines.py --replot
    python scripts/experiments/T05_baselines.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("T05"))
