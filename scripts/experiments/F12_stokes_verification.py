"""F12 (figure) - results/figures/task4_verification.png

Verification of the MAC Stokes solver: (a) plane Poiseuille profile and 2/H^2 error, (b) square / rectangular
ducts and inclined slits vs the series solution, (c) simple-cubic sphere arrays vs Zick & Homsy (1982) and
Sangani & Acrivos, (d) sphere error vs resolution.

Producer: scripts/task4_verification.py
    python scripts/experiments/F12_stokes_verification.py                  # full: ~2 min (2 cores)
    python scripts/experiments/F12_stokes_verification.py --quick
    python scripts/experiments/F12_stokes_verification.py --replot
    python scripts/experiments/F12_stokes_verification.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F12"))
