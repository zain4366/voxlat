"""Deterministic seeding for every random-number source VoxLat uses."""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["set_seed", "DEFAULT_SEED"]

DEFAULT_SEED = 20261007


def set_seed(seed: int = DEFAULT_SEED, *, deterministic_torch: bool = True) -> np.random.Generator:
    """Seed Python ``random``, NumPy's legacy global RNG and (if installed) PyTorch.

    Also sets ``PYTHONHASHSEED`` for any subprocesses (joblib workers) started
    afterwards. Returns a fresh ``numpy.random.Generator`` seeded with ``seed``;
    new code should draw from that generator rather than the global state.

    Parameters
    ----------
    seed:
        Non-negative integer seed.
    deterministic_torch:
        If PyTorch is available, request deterministic algorithms (warn-only)
        and disable cuDNN benchmarking.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    try:  # optional dependency (extra "ml")
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)
