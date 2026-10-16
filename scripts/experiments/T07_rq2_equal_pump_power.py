"""T07 (table) - results/tables/T07_rq2_equal_pump_power.csv

RQ2 at equal pump power (3 L/min): optimized R vs the best baseline of each family below the pump-power cap;
slots row at the 0.306 W cap uses the manufacturable optimum task12_slots_best_at_cap_gyroid.

Producer: scripts/experiments/tables.py - paper tables T01-T09 from the committed result CSVs/JSONs
    python scripts/experiments/T07_rq2_equal_pump_power.py                 # full: < 10 s
    python scripts/experiments/T07_rq2_equal_pump_power.py --quick
    python scripts/experiments/T07_rq2_equal_pump_power.py --replot
    python scripts/experiments/T07_rq2_equal_pump_power.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("T07"))
