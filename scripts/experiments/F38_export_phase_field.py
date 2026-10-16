"""F38 (figure) - results/figures/task12_phase_field.png

Exported lattice unwrapped at the mean gap radius for three Task 11 designs: conformal phase-field grading
with an integer number of cells around, no torn cells.

Producer: scripts/task12_figures.py
    python scripts/experiments/F38_export_phase_field.py                   # full: ~3 min
    python scripts/experiments/F38_export_phase_field.py --quick
    python scripts/experiments/F38_export_phase_field.py --replot
    python scripts/experiments/F38_export_phase_field.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F38"))
