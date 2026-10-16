"""Task 5: closure dataset - design, per-sample pipeline, storage/resume, estimate, figure.

Physics is verified in the Task 1-4 tests; here we check that the dataset reproduces those
production estimators exactly, that the design is what the docstring promises, and that the
run is resumable and safe. Tiny grids R(8, 16) keep this file fast (~30 s); the slow test
reproduces a Task 2-4 table row at the production grids R(32, 64).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from voxlat.closures import dataset as ds
from voxlat.closures.dataset import (
    DatasetSettings,
    DesignSpace,
    Sample,
    SettingsMismatchError,
    boundary_design,
    build_design,
    compute_sample,
    consolidate,
    estimate_runtime,
    load_existing,
    pending_samples,
    pick_timing_samples,
    plot_quicklook,
    read_parts,
    read_table,
    relative_cost,
    run_samples,
    sobol_design,
    stretch_levels,
    stretch_quantum,
    write_part,
)
from voxlat.geometry.tpms import compute_metrics, effective_stretch
from voxlat.homogenization.conduction import extrapolated_conductivity_tpms
from voxlat.homogenization.elasticity import (
    ALL_STRESS_CASES,
    MACRO_STRESS_CASES,
    extrapolated_elasticity_tpms,
    von_mises,
)
from voxlat.homogenization.stokes import extrapolated_permeability_tpms

TINY = DatasetSettings(n_coarse=8, n_fine=16)
REPO = Path(__file__).resolve().parents[1]


#: Iterative solves (PCG / MINRES) are reproducible only to round-off: multithreaded BLAS
#: (MKL/OpenBLAS on Windows) and pyamg change the summation order between runs and between
#: worker processes, and the Krylov iterates amplify that to ~1e-9 of the tensor norm. 1e-7 is
#: still 5 orders of magnitude below the ~1 % discretization error of the closures.
SOLVER_RTOL = 1e-7


def _close(actual, desired, rel: float = SOLVER_RTOL) -> None:
    """Componentwise equality up to ``rel`` of the largest compared entry (off-diagonals are tiny)."""
    actual, desired = np.asarray(actual, float), np.asarray(desired, float)
    np.testing.assert_allclose(actual, desired, rtol=rel, atol=rel * float(np.max(np.abs(desired))))


def _sample(w: float, rho: float, a_z_level: int, q: float = TINY.quantum, sid: str = "t-0") -> Sample:
    return Sample(sid, "test", 0, w, rho, a_z_level, q)


@pytest.fixture(scope="module")
def blend_record() -> dict:
    # blend, stretched a_z = 10/8 = 1.25 (no symmetry shortcut anywhere)
    return compute_sample(_sample(0.4, 0.33, 10, sid="blend"), TINY)


@pytest.fixture(scope="module")
def gyroid_record() -> dict:
    return compute_sample(_sample(0.0, 0.35, 8, sid="gyroid"), TINY)


# ---------------------------------------------------------------------------- design
def test_stretch_quantum_and_levels():
    assert stretch_quantum(32, 64) == 1 / 32
    assert stretch_quantum(32, 64, 48) == 1 / 16
    lv = stretch_levels((0.7, 1.5), 1 / 32)
    assert lv[0] == 22 and lv[-1] == 48 and len(lv) == 27  # 0.6875 <= 0.7, 1.5 exactly
    assert lv[0] / 32 <= 0.7 and lv[-1] / 32 >= 1.5


@pytest.mark.parametrize("level", [22, 23, 31, 32, 37, 48])
def test_quantized_stretch_identical_on_both_grids(level):
    """The reason for the quantization: both Richardson grids see the SAME cell."""
    p = _sample(0.3, 0.3, level, q=1 / 32).params()
    a32, a64 = effective_stretch(p, 32), effective_stretch(p, 64)
    assert a32 == a64 == (1.0, 1.0, level / 32)


def test_sobol_design_properties():
    space = DesignSpace()
    d = sobol_design(256, space, quantum=1 / 32)
    assert len(d) == 256 and len({s.id for s in d}) == 256
    w = np.array([s.w for s in d])
    r = np.array([s.rho for s in d])
    lv = np.array([s.a_z_level for s in d])
    assert w.min() >= 0 and w.max() < 1 and r.min() >= 0.2 and r.max() < 0.5
    assert set(lv) == set(stretch_levels(space.a_z, 1 / 32))  # every level used
    counts = np.bincount(lv - lv.min())
    assert counts.min() >= 9 and counts.max() <= 10  # equal-probability strata (256 / 27)
    # (0, m, s)-net property of the base-2 Sobol sequence: 16 points in each 1/16 bin
    assert np.all(np.bincount((w * 16).astype(int), minlength=16) == 16)
    assert np.all(np.bincount(((r - 0.2) / 0.3 * 16).astype(int), minlength=16) == 16)
    # deterministic, and a prefix: the N = 8 trial run is reused by the N = 256 run
    assert sobol_design(256, space, quantum=1 / 32) == d
    assert sobol_design(8, space, quantum=1 / 32) == d[:8]
    assert sobol_design(256, space, quantum=1 / 32, seed=1) != d


def test_boundary_design_and_full_design():
    space = DesignSpace()
    b = boundary_design(space, quantum=1 / 32)
    fam = {f: [s for s in b if s.family == f] for f in ("corner", "edge", "pure_line")}
    assert (len(fam["corner"]), len(fam["edge"]), len(fam["pure_line"])) == (8, 12, 14)
    corners = {(s.w, s.rho, s.a_z) for s in fam["corner"]}
    assert corners == {(w, r, a) for w in (0.0, 1.0) for r in (0.2, 0.5) for a in (0.6875, 1.5)}
    for s in fam["edge"]:  # on an edge: exactly two coordinates at a bound
        at = [s.w in (0.0, 1.0), s.rho in (0.2, 0.5), s.a_z in (0.6875, 1.5)]
        assert sum(at) == 2
    for s in fam["pure_line"]:
        assert s.w in (0.0, 1.0) and s.a_z == 1.0
    assert sorted({s.rho for s in fam["pure_line"]}) == pytest.approx([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    full = build_design(256, space=space)
    assert len(full) == 290
    keys = [(round(s.w, 12), round(s.rho, 12), s.a_z_level) for s in full]
    assert len(set(keys)) == len(keys) and len({s.id for s in full}) == len(full)
    assert DesignSpace.from_config() == space  # defaults = reference config bounds


def test_settings_fingerprint():
    s = DatasetSettings().resolved()
    assert (s.k_s, s.k_f, s.E_s, s.nu_s) == (130.0, 0.40, 70e9, 0.33)
    fp = s.fingerprint()
    assert DatasetSettings(preconditioner="jacobi", save_fields=True).fingerprint() == fp
    for change in ({"n_fine": 48}, {"k_f": 0.6}, {"tol_stokes": 1e-6}, {"stress_cases": ("uniaxial_z",)}):
        assert DatasetSettings(**change).fingerprint() != fp
    with pytest.raises(ValueError):
        DatasetSettings(n_coarse=64, n_fine=32)


def test_all_six_stress_cases():
    assert set(ALL_STRESS_CASES) <= set(MACRO_STRESS_CASES)
    for name in ALL_STRESS_CASES:
        expected = 1.0 if name.startswith("uniaxial") else np.sqrt(3.0)
        assert von_mises(MACRO_STRESS_CASES[name]) == pytest.approx(expected)


# ---------------------------------------------------------------------- one sample
def test_record_matches_production_estimators(blend_record):
    """Every closure equals the Task 1-4 production call on the same parameters."""
    rec = blend_record
    assert rec["status"] == "ok", rec["error"]
    s = TINY.resolved()
    p = _sample(0.4, 0.33, 10).params()
    k = extrapolated_conductivity_tpms(p, (8, 16), s.k_s, s.k_f).k_eff / s.k_s
    _close([rec["keff_xx"], rec["keff_zz"], rec["keff_xy"]], [k[0, 0], k[2, 2], k[0, 1]])
    C = extrapolated_elasticity_tpms(p, (8, 16), s.E_s, s.nu_s).C_eff / s.E_s
    _close([rec["C11"], rec["C33"], rec["C12"], rec["C44"], rec["C66"]],
           [C[0, 0], C[2, 2], C[0, 1], C[3, 3], C[5, 5]])
    K = extrapolated_permeability_tpms(p, (8, 16)).K
    _close([rec["K_xx"], rec["K_zz"], rec["K_xz"]], [K[0, 0], K[2, 2], K[0, 2]])
    m = compute_metrics(p, n=16)  # geometry is direct (no iterative solver)
    assert rec["a_sf_L"] == pytest.approx(m.a_sf_L, rel=1e-10)
    assert rec["rho_voxel"] == pytest.approx(m.relative_density)
    assert rec["wall_min_L"] == pytest.approx(m.wall.min)
    assert rec["kozeny_c"] == pytest.approx(rec["porosity"] ** 3 / (rec["K_mean"] * rec["a_sf_L"] ** 2))


def test_record_content_and_physics(blend_record):
    rec = blend_record
    assert rec["a_z"] == 1.25 and rec["K_symmetry"] == "none" and rec["K_n_solves"] == 3
    assert 0 < rec["keff_eig1"] <= rec["keff_eig2"] <= rec["keff_eig3"] < 1  # k_s-normalized, SPD
    assert rec["C_eig_min"] > 0 and rec["K_eig1"] > 0
    assert rec["E_min"] <= min(rec["E_x"], rec["E_y"], rec["E_z"], rec["E_111"]) + 1e-12
    assert rec["E_max"] >= max(rec["E_x"], rec["E_y"], rec["E_z"]) - 1e-12
    for case in ALL_STRESS_CASES:
        assert rec[f"loc_{case}_max"] >= rec[f"loc_{case}_p99"] > 1.0
    assert rec["porosity"] == pytest.approx(1 - rec["rho_voxel"], abs=0.02)  # fine-grid voxels
    for key in ("keff_time_s", "C_time_s", "K_time_s", "geo_time_s", "wall_time_s"):
        assert rec[key] > 0
    rr = json.loads(rec["run_record_json"])
    assert rr["resolution"] == [8, 16] and rr["params"]["settings"]["fingerprint"] == rec["settings_hash"]
    assert rec["rr_voxlat_version"] == rr["voxlat_version"]
    json.dumps(rec)  # JSON-serializable as stored


def test_symmetry_of_pure_cells(gyroid_record):
    r = gyroid_record
    assert r["status"] == "ok" and r["K_symmetry"] == "cubic"
    assert r["K_xx"] == r["K_yy"] == r["K_zz"] and r["K_xy"] == 0.0
    assert r["tortuosity_x"] == r["tortuosity_z"]
    assert r["keff_xx"] == pytest.approx(r["keff_zz"], rel=1e-3)
    assert r["C11"] == pytest.approx(r["C33"], rel=1e-2)
    assert r["loc_uniaxial_x_p99"] == pytest.approx(r["loc_uniaxial_z_p99"], rel=0.05)
    assert r["loc_shear_xy_p99"] == pytest.approx(r["loc_shear_xz_p99"], rel=0.05)


def test_stretched_pure_cell_is_tetragonal():
    r = compute_sample(_sample(1.0, 0.3, 12), TINY, blocks=("permeability",))  # diamond, a_z = 1.5
    assert r["status"] == "ok" and r["K_symmetry"] == "tetragonal_z" and r["K_n_solves"] == 2
    assert r["K_xx"] == r["K_yy"] and r["K_zz"] > r["K_xx"]
    assert r["tortuosity_y"] == r["tortuosity_x"] != r["tortuosity_z"]


def test_block_failure_is_recorded(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("solver exploded")

    monkeypatch.setattr(ds, "_conduction_block", boom)
    r = compute_sample(_sample(0.0, 0.3, 8), TINY, blocks=("geometry", "conduction"))
    assert r["status"] == "partial"
    assert "solver exploded" in json.loads(r["error"])["conduction"]
    assert "a_sf_L" in r and "keff_mean" not in r
    monkeypatch.setattr(ds, "_geometry_block", boom)
    r = compute_sample(_sample(0.0, 0.3, 8), TINY, blocks=("geometry", "conduction"))
    assert r["status"] == "error"


# ---------------------------------------------------------- storage, resume, parallel
def _two_samples() -> list[Sample]:
    return sobol_design(2, quantum=TINY.quantum)


def test_resume_mismatch_and_retry(tmp_path):
    parts = tmp_path / "parts"
    samples = _two_samples()
    res = run_samples(samples, TINY, parts_dir=parts, n_jobs=1)
    assert {r["id"] for r in res} == {s.id for s in samples} and all(r["status"] == "ok" for r in res)
    existing = load_existing(parts, tmp_path / "none.parquet")
    assert pending_samples(samples, existing, TINY) == []  # resume: nothing to do
    # other settings -> refuse to mix, unless forced
    other = DatasetSettings(n_coarse=8, n_fine=24)
    with pytest.raises(SettingsMismatchError):
        pending_samples(samples, existing, other)
    assert pending_samples(samples, existing, other, force=True) == samples
    # a record whose design point differs (e.g. another seed) is stale too
    moved = [Sample(samples[0].id, "sobol", 0, 0.9, 0.21, 6, TINY.quantum)]
    with pytest.raises(SettingsMismatchError):
        pending_samples(moved, existing, TINY)
    # non-ok records are retried only on request
    bad = dict(existing[samples[1].id], status="partial")
    write_part(bad, parts)
    existing = load_existing(parts, None)
    assert pending_samples(samples, existing, TINY) == []
    assert pending_samples(samples, existing, TINY, retry_errors=True) == [samples[1]]
    # a truncated part (crash mid-write is prevented by os.replace, but be robust) is skipped
    (parts / "garbage.json").write_text("{not json")
    assert len(read_parts(parts)) == 2


def test_parallel_equals_serial(tmp_path):
    """Worker processes give the same numbers as a serial run (to solver round-off, see SOLVER_RTOL)."""
    samples = _two_samples()
    run_samples(samples, TINY, parts_dir=tmp_path / "a", n_jobs=1)
    run_samples(samples, TINY, parts_dir=tmp_path / "b", n_jobs=2)
    a = {r["id"]: r for r in read_parts(tmp_path / "a")}
    b = {r["id"]: r for r in read_parts(tmp_path / "b")}
    for sid in a:
        for key in ("keff_mean", "C11", "C44", "K_mean", "loc_uniaxial_z_p99", "a_sf_L", "wall_min_L"):
            assert a[sid][key] == pytest.approx(b[sid][key], rel=SOLVER_RTOL), (sid, key)


def test_consolidate_tables(tmp_path):
    parts = tmp_path / "parts"
    samples = _two_samples()
    run_samples(samples, TINY, parts_dir=parts, n_jobs=1)
    df, path = consolidate(parts, tmp_path / "closures.csv", samples=samples)
    assert path.suffix == ".csv" and list(df["id"]) == [s.id for s in samples]
    back = read_table(path)
    assert list(back.columns[:7]) == ["id", "family", "index", "status", "w", "rho_target", "a_z"]
    np.testing.assert_allclose(back["K_mean"], df["K_mean"], rtol=1e-15)
    assert back["settings_hash"].iloc[0] == TINY.fingerprint() and (back["error"] == "").all()
    # the table alone is enough to resume (e.g. parts folder deleted, table committed)
    existing = load_existing(tmp_path / "missing_parts", path.with_suffix(".parquet"))
    assert pending_samples(samples, existing, TINY) == []
    # records outside the current design are kept out of the table
    df1, _ = consolidate(parts, tmp_path / "one.csv", samples=samples[:1])
    assert list(df1["id"]) == [samples[0].id]


def test_parquet_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    parts = tmp_path / "parts"
    samples = _two_samples()
    run_samples(samples, TINY, parts_dir=parts, n_jobs=1)
    df, path = consolidate(parts, tmp_path / "closures.parquet", samples=samples)
    assert path.suffix == ".parquet"
    back = read_table(path)
    pd.testing.assert_frame_equal(back, df, check_dtype=False)


# ---------------------------------------------------------------- estimate, figure
def test_runtime_estimate_and_timing_pick():
    pending = build_design(256)
    pick = pick_timing_samples(pending, 3)
    costs = sorted(relative_cost(s) for s in pending)
    mid = int(round(0.5 * (len(costs) - 1)))
    assert [relative_cost(s) for s in pick] == [costs[0], costs[mid], costs[-1]]
    assert relative_cost(_sample(0.5, 0.3, 8)) == pytest.approx(ds.BLEND_COST_FACTOR)
    assert relative_cost(_sample(0.0, 0.3, 8)) == 1.0
    timed = [(s, 100.0 * relative_cost(s)) for s in pick]  # exactly proportional
    est = estimate_runtime(timed, pending, n_jobs=4)
    assert est.seconds_per_unit == pytest.approx(100.0)
    assert est.serial_s == pytest.approx(100.0 * sum(relative_cost(s) for s in pending))
    assert est.wall_s_for(1, 1.4) == est.serial_s
    assert est.wall_s_for(8, 1.4) < est.wall_s_for(4, 1.4) < est.wall_s_for(2, 1.4) < est.serial_s
    assert est.wall_s_low <= est.wall_s_high
    assert "worker(s)" in est.summary()


def test_quicklook_figure(tmp_path, blend_record, gyroid_record):
    df = ds.records_to_frame([blend_record, gyroid_record])
    path = plot_quicklook(df, "quicklook_test", tmp_path)
    assert path.exists() and path.stat().st_size > 50_000


# --------------------------------------------------------------------------- slow
@pytest.mark.slow
def test_reproduces_task2_to_4_tables_at_production_grids():
    """Pure-line point (gyroid, rho* = 0.35, a_z = 1) at R(32, 64) == the Task 2-4 result tables."""
    rec = compute_sample(Sample("pure-x", "pure_line", 0, 0.0, 0.35, 32, 1 / 32), DatasetSettings())
    assert rec["status"] == "ok", rec["error"]
    t2 = pd.read_csv(REPO / "results" / "task2_keff_vs_density.csv")
    row2 = t2[(t2.w == 0.0) & np.isclose(t2.rho, 0.35) & (t2.kf_ks > 0)].iloc[0]
    t3 = pd.read_csv(REPO / "results" / "task3_stiffness_vs_density.csv")
    row3 = t3[(t3.w == 0.0) & np.isclose(t3.rho, 0.35)].iloc[0]
    t4 = pd.read_csv(REPO / "results" / "task4_permeability_vs_porosity.csv")
    row4 = t4[(t4.w == 0.0) & np.isclose(t4.rho, 0.35)].iloc[0]
    # identical code and grids; 1e-4 allows Jacobi-vs-AMG solver round-off (~1e-6)
    assert rec["keff_mean"] == pytest.approx(row2.k_mean, rel=1e-4)
    assert rec["E_axial"] == pytest.approx(row3.E_axial, rel=1e-4)
    assert rec["zener"] == pytest.approx(row3.zener, rel=1e-4)
    assert rec["loc_uniaxial_z_p99"] == pytest.approx(row3.loc_uniaxial_z_p99, rel=1e-3)
    assert rec["K_mean"] == pytest.approx(row4.K_mean, rel=1e-4)
    assert rec["tortuosity"] == pytest.approx(row4.tortuosity, rel=1e-4)
