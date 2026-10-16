"""F19 (figure) - results/figures/task6_cnn_extrapolation.png

Geometry-based (3D CNN on 32^3 voxels) vs parameter-based (GP, ensemble) learning of K* and E*: parity per
split and error vs blend weight w.

Producer: scripts/task6_cnn_experiment.py
    python scripts/experiments/F19_cnn_extrapolation.py                    # full: ~80 min (2 cores)
    python scripts/experiments/F19_cnn_extrapolation.py --quick
    python scripts/experiments/F19_cnn_extrapolation.py --replot
    python scripts/experiments/F19_cnn_extrapolation.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F19"))
