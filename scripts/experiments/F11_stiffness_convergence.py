"""F11 (figure) - results/figures/task3_convergence.png

Grid convergence of C_eff and stress localization (n = 16-64, 4 grid offsets each; n = 96 anchors): systematic
staircase error vs alignment noise.

Producer: scripts/task3_convergence.py
    python scripts/experiments/F11_stiffness_convergence.py                # full: ~14 min (2 cores)
    python scripts/experiments/F11_stiffness_convergence.py --quick
    python scripts/experiments/F11_stiffness_convergence.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F11"))
