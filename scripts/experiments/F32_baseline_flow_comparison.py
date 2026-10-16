"""F32 (figure) - results/figures/task10_flow_comparison.png

Baselines B1-B3 at 1, 3 and 5 L/min: R, R_active and pump power.

Producer: scripts/task10_baselines.py
    python scripts/experiments/F32_baseline_flow_comparison.py             # full: ~2.5 min
    python scripts/experiments/F32_baseline_flow_comparison.py --quick
    python scripts/experiments/F32_baseline_flow_comparison.py --replot
    python scripts/experiments/F32_baseline_flow_comparison.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F32"))
