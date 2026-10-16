"""F28 (figure) - results/figures/task9_verification.png

Device-model verification: epsilon-NTU convergence against the exact discrete solution, pressure drop vs the
analytic value over 1-5 L/min, and grid convergence of the thermal resistance R.

Producer: scripts/task9_jacket_model.py
    python scripts/experiments/F28_jacket_verification.py                  # full: ~20 s
    python scripts/experiments/F28_jacket_verification.py --quick
    python scripts/experiments/F28_jacket_verification.py --replot
    python scripts/experiments/F28_jacket_verification.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F28"))
