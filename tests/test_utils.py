"""Task 0 smoke tests for the shared utilities and package layout."""

from __future__ import annotations

import importlib
import json
import logging
import random

import numpy as np
import pytest

from voxlat.utils import (
    Stopwatch,
    figures_dir,
    get_logger,
    repo_root,
    run_record,
    save_figure,
    set_seed,
    stopwatch,
)
from voxlat.utils.config import load_config

SUBPACKAGES = [
    "geometry",
    "homogenization",
    "closures",
    "surrogates",
    "device",
    "optimize",
    "export",
    "utils",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_import(name):
    mod = importlib.import_module(f"voxlat.{name}")
    assert mod.__doc__


def test_repo_layout():
    root = repo_root()
    for d in ["configs", "data", "models", "results/figures", "scripts", "tests", "legacy"]:
        assert (root / d).is_dir(), d
    assert (root / "STATUS.md").is_file()


def test_set_seed_reproducible():
    rng1 = set_seed(123)
    a = (random.random(), np.random.rand(), rng1.random(3))
    rng2 = set_seed(123)
    b = (random.random(), np.random.rand(), rng2.random(3))
    assert a[0] == b[0] and a[1] == b[1]
    np.testing.assert_array_equal(a[2], b[2])
    with pytest.raises(ValueError):
        set_seed(-1)


def test_get_logger_single_handler(capsys):
    lg1 = get_logger("task0")
    lg2 = get_logger("task0")
    assert lg1 is lg2 and lg1.name == "voxlat.task0"
    root = logging.getLogger("voxlat")
    assert sum(getattr(h, "_voxlat_handler", False) for h in root.handlers) == 1


def test_run_record_contents():
    cfg = load_config()
    with stopwatch() as sw:
        sum(range(1000))
    rec = run_record(
        {"w": np.float64(0.5), "rho": 0.3, "shape": (32, 32, 32), "bounds": cfg.design_bounds},
        resolution=48,
        wall_time=sw.elapsed,
        solver="pcg",
    )
    for key in ["params", "params_json", "voxlat_version", "git_hash", "resolution",
                "wall_time_s", "timestamp_utc", "python", "numpy", "platform", "extra"]:
        assert key in rec
    assert rec["resolution"] == 48
    assert rec["wall_time_s"] >= 0
    assert rec["extra"] == {"solver": "pcg"}
    assert isinstance(rec["params"]["w"], float)
    assert rec["params"]["bounds"]["relative_density"] == {"lo": 0.2, "hi": 0.5}
    json.dumps(rec)  # fully JSON-serializable
    assert json.loads(rec["params_json"]) == rec["params"]


def test_stopwatch_monotonic():
    sw = Stopwatch()
    t1 = sw.elapsed
    t2 = sw.stop()
    assert 0 <= t1 <= t2 == sw.elapsed


def test_save_figure_300dpi(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    fig, ax = plt.subplots(figsize=(2.0, 1.0))
    ax.plot([0, 1], [0, 1])
    path = save_figure(fig, "smoke", tmp_path, bbox_inches=None)
    assert path == tmp_path / "smoke.png" and path.is_file()
    img = imread(path)
    assert img.shape[:2] == (300, 600)  # 1 in x 2 in at 300 dpi
    # default directory is <repo>/results/figures
    assert figures_dir() == repo_root() / "results" / "figures"
