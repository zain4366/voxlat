"""T08 (table) - results/tables/T08_representative_designs.csv

Representative designs per layout (R, R_active, pump power, mass, margin, field ranges), including the
manufacturable slot optimum best_at_cap_gyroid.

Producer: scripts/experiments/tables.py - paper tables T01-T09 from the committed result CSVs/JSONs
    python scripts/experiments/T08_representative_designs.py               # full: < 10 s
    python scripts/experiments/T08_representative_designs.py --quick
    python scripts/experiments/T08_representative_designs.py --replot
    python scripts/experiments/T08_representative_designs.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("T08"))
