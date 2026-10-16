"""F03 (figure) - results/figures/task1_asf_convergence.png

Grid convergence of the specific surface area a_sf L (marching cubes on the periodic level set) for n = 24-64
voxels per cell; the gyroid sheet limit approaches the minimal-surface value ~3.09.

Producer: scripts/task1_tpms_figures.py
    python scripts/experiments/F03_asf_convergence.py                      # full: 1-2 min
    python scripts/experiments/F03_asf_convergence.py --quick
    python scripts/experiments/F03_asf_convergence.py --replot
    python scripts/experiments/F03_asf_convergence.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F03"))
