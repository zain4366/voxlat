"""F27 (figure) - results/figures/task9_fields_uniform_gyroid.png

Homogenized jacket model, uniform gyroid reference (rho* = 0.35, L = 4 mm, 3 L/min): pressure, Darcy velocity,
coolant and stator-side sleeve temperature on the unwrapped (s, z) plane; slots shaded, hot spot marked.

Producer: scripts/task9_jacket_model.py
    python scripts/experiments/F27_jacket_fields.py                        # full: ~20 s
    python scripts/experiments/F27_jacket_fields.py --quick
    python scripts/experiments/F27_jacket_fields.py --replot
    python scripts/experiments/F27_jacket_fields.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F27"))
