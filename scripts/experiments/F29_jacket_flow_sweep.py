"""F29 (figure) - results/figures/task9_flow_sweep.png

Thermal resistance, pressure drop and pump power vs flow rate for uniform gyroid, blend and diamond.

Producer: scripts/task9_jacket_model.py
    python scripts/experiments/F29_jacket_flow_sweep.py                    # full: ~20 s
    python scripts/experiments/F29_jacket_flow_sweep.py --quick
    python scripts/experiments/F29_jacket_flow_sweep.py --replot
    python scripts/experiments/F29_jacket_flow_sweep.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F29"))
