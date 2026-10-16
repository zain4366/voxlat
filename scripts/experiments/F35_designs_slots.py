"""F35 (figure) - results/figures/task11_designs_slots.png

Representative slot-manifold designs (min-R, best at the 0.306 W cap, knee, min-mass): rho*, w and L fields on
the unwrapped jacket. The best-at-cap design is exported with w = 0 (task12_slots_best_at_cap_gyroid) as the
manufacturable optimum.

Producer: scripts/task11_analysis.py - reads the committed runs in results/task11/
    python scripts/experiments/F35_designs_slots.py                        # full: ~5 min
    python scripts/experiments/F35_designs_slots.py --quick
    python scripts/experiments/F35_designs_slots.py --replot
    python scripts/experiments/F35_designs_slots.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F35"))
