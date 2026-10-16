"""F30 (figure) - results/figures/task10_temperature_maps.png

Stator-side sleeve temperature of the baselines at 3 L/min on one colour scale: B1 helical channels (end-ring
manifolds), B1s (slot manifolds), B2 best uniform gyroid, B3 density-only grading.

Producer: scripts/task10_baselines.py
    python scripts/experiments/F30_baseline_temperature_maps.py            # full: ~2.5 min
    python scripts/experiments/F30_baseline_temperature_maps.py --quick
    python scripts/experiments/F30_baseline_temperature_maps.py --replot
    python scripts/experiments/F30_baseline_temperature_maps.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F30"))
