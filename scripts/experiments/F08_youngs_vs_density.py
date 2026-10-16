"""F08 (figure) - results/figures/task3_youngs_vs_density.png

Effective Young's modulus E*/E_s vs relative density (log-log) from periodic voxel FEA, with power-law fits
E*/E_s = C rho*^m, the HS bound, a slope-2 guide and published gyroid/diamond data (Maskery et al.); blend
shown as its directional range.

Producer: scripts/task3_stiffness_vs_density.py
    python scripts/experiments/F08_youngs_vs_density.py                    # full: ~16 min (2 cores)
    python scripts/experiments/F08_youngs_vs_density.py --quick
    python scripts/experiments/F08_youngs_vs_density.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F08"))
