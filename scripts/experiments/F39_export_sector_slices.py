"""F39 (figure) - results/figures/task12_sector_slices.png

30-degree sector of slots/best_at_cap around its most critical cell: axial, meridional and unwrapped cuts with
sleeve, outer wall, end caps, slot and port (the blend version that fails the 0.35 mm wall check).

Producer: scripts/task12_figures.py
    python scripts/experiments/F39_export_sector_slices.py                 # full: ~3 min
    python scripts/experiments/F39_export_sector_slices.py --quick
    python scripts/experiments/F39_export_sector_slices.py --replot
    python scripts/experiments/F39_export_sector_slices.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F39"))
