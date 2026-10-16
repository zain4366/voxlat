"""F36 (figure) - results/figures/task11_designs_end_rings.png

Representative end-ring-manifold designs (min-R, knee, min-mass): rho*, w and L fields.

Producer: scripts/task11_analysis.py - reads the committed runs in results/task11/
    python scripts/experiments/F36_designs_end_rings.py                    # full: ~5 min
    python scripts/experiments/F36_designs_end_rings.py --quick
    python scripts/experiments/F36_designs_end_rings.py --replot
    python scripts/experiments/F36_designs_end_rings.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F36"))
