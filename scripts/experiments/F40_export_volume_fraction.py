"""F40 (figure) - results/figures/task12_volume_fraction.png

Volume fraction of the exported sectors vs the design rho* and the phase-aware expectation, overall, core /
near-wall and per patch.

Producer: scripts/task12_figures.py
    python scripts/experiments/F40_export_volume_fraction.py               # full: ~3 min
    python scripts/experiments/F40_export_volume_fraction.py --quick
    python scripts/experiments/F40_export_volume_fraction.py --replot
    python scripts/experiments/F40_export_volume_fraction.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F40"))
