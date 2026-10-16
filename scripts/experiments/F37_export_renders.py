"""F37 (figure) - results/figures/task12_renders.png

The exported manufacturable optimum (task12_slots_best_at_cap_gyroid), ray-traced from the signed-distance
field that the PicoGK export reproduces voxel by voxel (IoU 0.997): overview, lattice close-up and outlet-slot
section.

Producer: scripts/task12_render.py - replot uses --quick (low resolution); the committed renders are full resolution
    python scripts/experiments/F37_export_renders.py                       # full: ~2 min at 1800 px
    python scripts/experiments/F37_export_renders.py --quick
    python scripts/experiments/F37_export_renders.py --replot
    python scripts/experiments/F37_export_renders.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F37"))
