"""F26 (figure) - results/figures/task8_nusselt.png

Interstitial Nusselt number Nu_Dh vs Re_Dh at Pr = 20.7 for the candidate TPMS correlations (Pr^(1/3) and
Pr^0.4 scalings) with Gnielinski and Wakao-Kaguei references; fitted box and jacket operating range.

Producer: scripts/task8_literature_closures.py
    python scripts/experiments/F26_nusselt.py                              # full: 5 s
    python scripts/experiments/F26_nusselt.py --quick
    python scripts/experiments/F26_nusselt.py --replot
    python scripts/experiments/F26_nusselt.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F26"))
