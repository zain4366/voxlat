"""F01 (figure) - results/figures/task0_heat_flux_profile.png

Axial stator heat-flux profile q''(z) of the reference 15 kW motor: mean 35 kW/m^2 (Q = 770 W over the 70 mm x
50 mm sleeve), parabolic with the ends at twice the centre value; integrates to Q within 0.1 %.

Producer: scripts/task0_heat_flux_profile.py
    python scripts/experiments/F01_heat_flux_profile.py                    # full: 5 s
    python scripts/experiments/F01_heat_flux_profile.py --quick
    python scripts/experiments/F01_heat_flux_profile.py --replot
    python scripts/experiments/F01_heat_flux_profile.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F01"))
