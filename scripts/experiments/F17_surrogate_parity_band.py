"""F17 (figure) - results/figures/task6_parity_band.png

Parity on the blend-band extrapolation split (training on near-pure morphologies, testing on blends).

Producer: scripts/task6_evaluate.py
    python scripts/experiments/F17_surrogate_parity_band.py                # full: ~10 min (2 cores)
    python scripts/experiments/F17_surrogate_parity_band.py --quick
    python scripts/experiments/F17_surrogate_parity_band.py --replot
    python scripts/experiments/F17_surrogate_parity_band.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run_item  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_item("F17"))
