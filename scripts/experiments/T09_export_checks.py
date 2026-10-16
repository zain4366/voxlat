"""T09 (table) - results/tables/T09_export_checks.csv

Manufacturability checks of the exported geometry (numpy mirror and PicoGK laptop builds): min wall, throat,
connectivity, pockets, loose solid, volume fraction, PicoGK-vs-numpy IoU.

Producer: scripts/experiments/tables.py - paper tables T01-T09 from the committed result CSVs/JSONs
    python scripts/experiments/T09_export_checks.py                        # full: < 10 s
    python scripts/experiments/T09_export_checks.py --quick
    python scripts/experiments/T09_export_checks.py --replot
    python scripts/experiments/T09_export_checks.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("T09"))
