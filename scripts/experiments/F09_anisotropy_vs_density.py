"""F09 (figure) - results/figures/task3_anisotropy_vs_density.png

Elastic anisotropy vs relative density: Zener ratio of the cubic gyroid and diamond and the universal
anisotropy index A^U (including the trigonal blend).

Producer: scripts/task3_stiffness_vs_density.py
    python scripts/experiments/F09_anisotropy_vs_density.py                # full: ~16 min (2 cores)
    python scripts/experiments/F09_anisotropy_vs_density.py --quick
    python scripts/experiments/F09_anisotropy_vs_density.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F09"))
