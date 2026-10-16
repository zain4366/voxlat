"""F31 (figure) - results/figures/task10_fronts.png

Baseline tuning: R and R_active vs pump power for all candidates, the per-family fronts and the selected
designs (pump power <= 0.306 W of the reference gyroid).

Producer: scripts/task10_baselines.py
    python scripts/experiments/F31_baseline_fronts.py                      # full: ~2.5 min
    python scripts/experiments/F31_baseline_fronts.py --quick
    python scripts/experiments/F31_baseline_fronts.py --replot
    python scripts/experiments/F31_baseline_fronts.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F31"))
