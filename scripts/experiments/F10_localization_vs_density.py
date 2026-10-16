"""F10 (figure) - results/figures/task3_localization_vs_density.png

Stress localization factors (99th percentile, maximum and mean von Mises over the macroscopic stress) under
unit uniaxial-z and shear-xz stress vs relative density (n = 64).

Producer: scripts/task3_stiffness_vs_density.py
    python scripts/experiments/F10_localization_vs_density.py              # full: ~16 min (2 cores)
    python scripts/experiments/F10_localization_vs_density.py --quick
    python scripts/experiments/F10_localization_vs_density.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F10"))
