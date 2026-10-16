"""Task 11: voxlat.optimize - parameterization, feasibility repair, hypervolume / EHVI, GP, optimizers.

Verification strategy
* B-spline bases against scipy and their partition-of-unity / periodicity / end-interpolation
  properties; fields stay inside the design bounds (convex hull);
* feasibility repair: repaired designs pass the cheap checks and the device model agrees
  (manufacturable), feasible designs are fixed points, the fast L_min lookup equals Task 1's;
* hypervolume and HVI exact against analytic boxes, brute-force HV differences and Monte Carlo;
  analytic EHVI against Monte Carlo; box decomposition volume = box - dominated volume;
* GP: analytic gradient of the log posterior vs finite differences, interpolation, conditioning;
* optimizers run end to end on tiny budgets (NSGA-II only when pymoo is installed).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import BSpline

from voxlat.optimize import (
    FastMinCellSize,
    GaussianProcess,
    GradingSpec,
    JacketProblem,
    ObjectiveSpace,
    ProblemSettings,
    box_decomposition,
    clamped_bspline_basis,
    expected_hvi,
    have_pymoo,
    hv_improvement,
    hypervolume,
    hypervolume_history,
    pareto_mask,
    periodic_cubic_basis,
    run_qnehvi,
    run_sobol,
)
from voxlat.optimize.study import combined_front, equal_pump_power, hv_curves, representative_designs
from voxlat.utils.config import load_config
from voxlat.utils.paths import models_dir

HAVE_MODELS = (models_dir() / "closure_ensemble.npz").is_file() and (models_dir() / "finite_gap_correction.json").is_file()
needs_models = pytest.mark.skipif(not HAVE_MODELS, reason="trained closure models not in models/")
CFG = load_config()


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def problem():
    if not HAVE_MODELS:
        pytest.skip("trained closure models not in models/")
    return JacketProblem(ProblemSettings())


# -----------------------------------------------------------------------------------------
# bases and fields
# -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 4, 5, 8])
def test_clamped_basis_matches_scipy_and_interpolates_ends(n):
    t = np.linspace(0, 1, 77)
    p = min(3, n - 1)
    knots = np.r_[np.zeros(p), np.linspace(0, 1, n - p + 1), np.ones(p)]
    np.testing.assert_allclose(clamped_bspline_basis(t, n), BSpline.design_matrix(t, knots, p).toarray(), atol=1e-14)
    B = clamped_bspline_basis([0.0, 1.0], n)
    np.testing.assert_allclose(B[0], np.eye(n)[0], atol=1e-14)
    np.testing.assert_allclose(B[1], np.eye(n)[-1], atol=1e-14)
    np.testing.assert_allclose(clamped_bspline_basis(np.linspace(-0.2, 1.2, 9), n).sum(1), 1.0, atol=1e-14)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8])
def test_periodic_basis_partition_of_unity_and_periodicity(n):
    u = np.linspace(0, 1, 101)
    B = periodic_cubic_basis(u, n)
    assert B.min() >= 0
    np.testing.assert_allclose(B.sum(1), 1.0, atol=1e-14)
    np.testing.assert_allclose(B[0], B[-1], atol=1e-14)  # u = 0 and u = 1 coincide
    if n >= 3:
        # control i peaks at its centre u = i/n with the cubic weights 1/6, 2/3, 1/6
        Bc = periodic_cubic_basis(np.arange(n) / n, n)
        np.testing.assert_allclose(np.diag(Bc), 2 / 3, atol=1e-14)


def test_spec_counts():
    s = GradingSpec()
    assert s.active() == ["rho", "w", "L"] and s.n_var == 72
    s2 = GradingSpec(rho=(4, 2), w=(0, 0), L=(1, 1), a_z=(2, 1))
    assert s2.active() == ["rho", "L", "a_z"] and s2.n_var == 11
    assert GradingSpec.from_dict(s2.to_dict()) == s2


@needs_models
def test_fields_in_bounds_round_trip_and_point_evaluation(problem):
    par = problem.par
    rng = np.random.default_rng(3)
    x = rng.random(par.n_var)
    F = par.fields(x)
    lat = problem.model.grid.lattice
    for n in ("rho", "w", "L"):
        lo, hi = par.bounds[n]
        assert F[n].min() >= lo - 1e-12 and F[n].max() <= hi + 1e-12
    np.testing.assert_array_equal(F["a_z"], 1.0)
    np.testing.assert_allclose(par.x_from_control_nets(par.control_nets(x)), x, atol=1e-14)
    # field_function at the cell centres = fields
    S, Z = problem.model.grid.meshgrid()
    for n in ("rho", "w", "L"):
        np.testing.assert_allclose(par.field_function(x, n)(S, Z)[lat], F[n][lat], rtol=1e-12)
    # constant controls -> uniform field
    xc = np.full(par.n_var, 0.5)
    np.testing.assert_allclose(par.fields(xc)["rho"], 0.35, atol=1e-14)
    # batched fields = single fields
    Fb = par.fields_batch(np.vstack([x, xc]))
    np.testing.assert_allclose(Fb["L"][0], F["L"].ravel(), rtol=1e-14)
    assert len(par.variable_names()) == par.n_var


def test_fast_min_cell_size_equals_task1_lookup():
    from voxlat.geometry import min_cell_size

    f = FastMinCellSize(CFG)
    rng = np.random.default_rng(0)
    w = np.r_[0.0, 1.0, 0.5, 0.125, rng.random(5000)]
    r = np.r_[0.15, 0.55, 0.3, 0.55, rng.uniform(0.15, 0.55, 5000)]
    np.testing.assert_allclose(f(w, r), min_cell_size(w, r), rtol=1e-12)


# -----------------------------------------------------------------------------------------
# feasibility repair
# -----------------------------------------------------------------------------------------
@needs_models
def test_repair_gives_cheap_feasible_designs_the_model_agrees_with(problem):
    par = problem.par
    rng = np.random.default_rng(11)
    X = rng.random((256, par.n_var))
    Y = par.repair_batch(X)
    ok = par.cheap_feasible_batch(Y)
    assert ok.mean() > 0.9  # blend polarization + cell-size raise make random designs feasible
    np.testing.assert_allclose(par.repair_batch(Y[ok]), Y[ok], atol=0)  # fixed points
    np.testing.assert_allclose(np.array([par.repair(x) for x in X[:8]]), Y[:8], atol=0)  # scalar = batch
    # the device model agrees on manufacturability and on the gradient value
    for y in Y[ok][:2]:
        F = par.fields(y)
        out = problem.model.evaluate(F["rho"], F["w"], F["L"])
        man = out["manufacturability"]
        assert man["manufacturable"]
        assert man["gradient_max"] == pytest.approx(par.gradient_max(y), rel=1e-12)
        assert man["cell_size_ratio_min"] >= 1.0
        assert out["mass"] == pytest.approx(par.mass(F), rel=1e-12)
    # a blend in the pinch-neck zone everywhere (w 0.62, rho* 0.32: L_min > 6 mm) is polarized toward
    # its majority type (diamond) without touching rho or L
    nets = {"rho": np.full((8, 4), 0.32), "w": np.full((8, 4), 0.62), "L": np.full((4, 2), 5e-3)}
    x0 = par.x_from_control_nets(nets)
    assert not par.cheap_feasible_batch(x0[None])[0]
    y = par.repair(x0)
    assert par.cheap_feasible_batch(y[None])[0]
    ny = par.control_nets(y)
    assert ny["w"].min() > 0.75
    np.testing.assert_allclose(ny["rho"], 0.32, atol=1e-12)
    np.testing.assert_allclose(ny["L"], 5e-3, atol=1e-15)
    # a gradient violation alone is fixed by shrinking rho toward its mean (L and w untouched)
    r = np.tile([0.2, 0.5, 0.2, 0.5], (8, 1))  # end slope 3 x 0.3 / 50 mm x 4 mm = 0.072 > 0.05
    x1 = par.x_from_control_nets({"rho": r, "w": np.zeros((8, 4)), "L": np.full((4, 2), 4e-3)})
    assert par.cheap_constraints(x1)["gradient"] > 0
    y1 = par.repair(x1)
    assert par.cheap_constraints(y1)["gradient"] <= 0
    assert par.control_nets(y1)["rho"].mean() == pytest.approx(r.mean(), abs=1e-12)
    np.testing.assert_allclose(y1[par._slices["L"]], x1[par._slices["L"]])


@needs_models
def test_mass_is_linear_in_rho_controls(problem):
    from voxlat.optimize.algorithms import _mass_linear

    a, w = _mass_linear(problem)
    rng = np.random.default_rng(5)
    for x in rng.random((5, problem.n_var)):
        assert a + w @ x == pytest.approx(problem.par.mass(problem.par.fields(x)), rel=1e-12)


# -----------------------------------------------------------------------------------------
# hypervolume
# -----------------------------------------------------------------------------------------
def test_pareto_mask_brute_force():
    rng = np.random.default_rng(1)
    F = np.round(rng.random((300, 3)), 2)
    dom = np.array([np.any(np.all(F <= f, 1) & np.any(F < f, 1)) for f in F])
    np.testing.assert_array_equal(pareto_mask(F), ~dom)


def test_hypervolume_exact_cases():
    ref = np.ones(3)
    assert hypervolume([[0.5, 0.5, 0.5]], ref) == pytest.approx(0.125)
    # two boxes overlapping in [0.5,1]x[0.5,1]x[0.2,1]
    F = [[0.2, 0.5, 0.2], [0.5, 0.2, 0.2]]
    assert hypervolume(F, ref) == pytest.approx(2 * 0.8 * 0.5 * 0.8 - 0.5 * 0.5 * 0.8)
    assert hypervolume([[1.2, 0.1, 0.1]], ref) == 0.0  # outside the reference box
    assert hypervolume([[0.2, 0.4], [0.4, 0.2]], [1, 1]) == pytest.approx(0.8 * 0.6 + 0.6 * 0.8 - 0.6 * 0.6)
    assert hypervolume(np.zeros((0, 3)), ref) == 0.0


def test_hypervolume_monte_carlo_and_history():
    rng = np.random.default_rng(2)
    F = rng.random((40, 3)) * 0.9
    S = rng.random((300_000, 3))
    dom = np.zeros(len(S), bool)
    for p in F[pareto_mask(F)]:
        dom |= np.all(S >= p, axis=1)
    hv = hypervolume(F, np.ones(3))
    assert hv == pytest.approx(dom.mean(), abs=4 * math.sqrt(hv * (1 - hv) / len(S)))
    hist = hypervolume_history(F, np.ones(3), checkpoints=[1, 10, 40])
    assert np.all(np.diff(hist) >= 0) and hist[-1] == pytest.approx(hv)


@pytest.mark.parametrize("d", [2, 3])
def test_box_decomposition_and_hvi(d):
    rng = np.random.default_rng(4 + d)
    ref = np.ones(d)
    F = rng.random((25, d)) * 0.95
    lo, hi = box_decomposition(F, ref)
    # volume of boxes clipped to [0, ref] = 1 - HV
    vol = np.prod(np.clip(hi - np.maximum(lo, 0), 0, None), axis=1).sum()
    assert vol == pytest.approx(1 - hypervolume(F, ref), rel=1e-12)
    Y = rng.random((60, d))
    exact = np.array([hypervolume(np.vstack([F, y]), ref) - hypervolume(F, ref) for y in Y])
    np.testing.assert_allclose(hv_improvement(Y, (lo, hi)), exact, atol=1e-14)
    # empty front: one box, HVI = dominated box of y
    lo0, hi0 = box_decomposition(np.zeros((0, d)), ref)
    assert hv_improvement(np.full((1, d), 0.5), (lo0, hi0))[0] == pytest.approx(0.5**d)


def test_expected_hvi_against_monte_carlo():
    rng = np.random.default_rng(8)
    F = rng.random((15, 3)) * 0.9
    boxes = box_decomposition(F, np.ones(3))
    mu = rng.random((4, 3)) * 0.8
    sd = np.c_[rng.uniform(0.02, 0.2, (4, 2)), np.zeros(4)]  # third outcome known (like the mass)
    e = expected_hvi(mu, sd, boxes)
    Z = mu[:, None, :] + sd[:, None, :] * rng.standard_normal((4, 200_000, 3))
    for i in range(4):
        h = hv_improvement(Z[i], boxes)
        assert e[i] == pytest.approx(h.mean(), abs=5 * h.std() / math.sqrt(len(h)) + 1e-6)  # rare-event floor
    # sigma -> 0: EHVI = HVI
    np.testing.assert_allclose(expected_hvi(mu, np.zeros_like(mu), boxes), hv_improvement(mu, boxes), atol=1e-15)


# -----------------------------------------------------------------------------------------
# GP
# -----------------------------------------------------------------------------------------
def test_gp_gradient_interpolation_and_conditioning():
    rng = np.random.default_rng(0)
    d = 5
    X = rng.random((40, d))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    gp = GaussianProcess(seed=0)
    ys = (y - y.mean()) / y.std()
    D2 = (X[None] - X[:, None]).transpose(2, 0, 1) ** 2
    th = np.r_[rng.normal(0, 0.5, d), 0.3, -3.0]
    f, g = gp._neg_log_post(th, X, ys, D2)
    eps = 1e-6
    gn = np.array([(gp._neg_log_post(th + e, X, ys, D2)[0] - gp._neg_log_post(th - e, X, ys, D2)[0]) / (2 * eps)
                   for e in np.eye(len(th)) * eps])
    np.testing.assert_allclose(g, gn, rtol=1e-5, atol=1e-6)
    gp.fit(X, y)
    m, s = gp.predict(X)
    np.testing.assert_allclose(m, y, atol=1e-2)
    Xt = rng.random((300, d))
    mt, st = gp.predict(Xt)
    yt = np.sin(3 * Xt[:, 0]) + Xt[:, 1] ** 2
    assert np.sqrt(np.mean((mt - yt) ** 2)) < 0.05 * y.std()
    assert gp.lengthscales[2:].min() > gp.lengthscales[:2].max()  # irrelevant inputs get long lengthscales
    # conditioning on a fantasy at x0 pulls the mean there (up to the fitted noise) and shrinks the variance
    x0 = rng.random((1, d))
    mb, sb = gp.predict(x0)
    target = mb[0] + 3 * sb[0] + 0.5
    g2 = gp.condition_on(x0, [target])
    m0, s0 = g2.predict(x0)
    assert abs(m0[0] - target) < 0.2 * abs(mb[0] - target) and s0[0] < sb[0]


# -----------------------------------------------------------------------------------------
# problem and optimizers
# -----------------------------------------------------------------------------------------
def test_objective_space_round_trip():
    sp = ObjectiveSpace()
    R, P, m = np.array([0.012, 0.03]), np.array([0.005, 1.5]), np.array([0.30, 0.38])
    Fn = sp.normalize(R, P, m)
    np.testing.assert_allclose(sp.denormalize(Fn), np.c_[R, P, m], rtol=1e-12)
    assert np.all(sp.normalize(0.040, 10**0.5, 0.42) == pytest.approx(1.0))


@needs_models
def test_problem_records_and_random_search(problem):
    p = JacketProblem(ProblemSettings())
    run_sobol(p, 3, seed=1)
    df = p.table()
    assert p.n_expensive == 3 and len(df) >= 3
    ev = df[df["expensive"]]
    assert list(ev["eval_id"]) == [0, 1, 2]
    assert (df.loc[~df["expensive"], "eval_id"] == -1).all()
    assert ev["feasible"].all() and (ev["g_cell_size"] <= 0).all() and (ev["g_gradient"] <= 1e-9).all()
    assert ev["thermal_resistance"].between(0.01, 0.05).all()
    assert (ev["energy_balance_error"] < 1e-8).all() and (ev["mass_balance_error"] < 1e-10).all()
    np.testing.assert_allclose(ev["mass"], ev["mass_exact"], rtol=1e-12)


@needs_models
def test_qnehvi_numpy_backend_runs_and_proposes_feasible_designs():
    p = JacketProblem(ProblemSettings(layout="end_rings"))
    res = run_qnehvi(p, 7, n_init=4, q=3, seed=0, backend="numpy", pool_sobol=64, pool_local=128, n_refine=4, n_per=16)
    assert res.backend == "numpy" and res.iterations == 1
    df = p.table()
    bo = df[df["iteration"] == 1]
    assert len(bo) == 3 and bo["expensive"].all() and bo["feasible"].all()
    assert p.n_expensive == 7


@needs_models
@pytest.mark.skipif(not have_pymoo(), reason="pymoo not installed")
def test_nsga2_runs():
    from voxlat.optimize import run_nsga2

    p = JacketProblem(ProblemSettings())
    run_nsga2(p, pop_size=8, n_gen=2, seed=0)
    df = p.table()
    assert len(df) >= 8 and set(df["iteration"]) <= {0, 1}
    assert (df.loc[df["expensive"], "g_cell_size"] <= 0).all()


# -----------------------------------------------------------------------------------------
# study analysis helpers
# -----------------------------------------------------------------------------------------
def _fake_runs():
    rng = np.random.default_rng(0)
    rows = []
    for m in ("a", "b"):
        for k in range(30):
            R = 0.012 + 0.01 * rng.random()
            rows.append(dict(run=f"{m}_s0", method=m, seed=0, eval_id=k, expensive=True, feasible=k % 7 != 3,
                             thermal_resistance=R, pump_power=0.05 / (R * 100) ** 3, mass=0.3 + 0.05 * rng.random(),
                             R_active=R * 0.9))
    return pd.DataFrame(rows)


def test_study_helpers():
    df = _fake_runs()
    hv = hv_curves(df, [5, 10, 30])
    assert set(hv["run"]) == {"a_s0", "b_s0"} and (hv.groupby("run")["hv"].diff().fillna(0) >= 0).all()
    front = combined_front(df)
    F = front[["thermal_resistance", "pump_power", "mass"]].to_numpy()
    assert pareto_mask(F).all() and front["feasible"].all()
    rep = representative_designs(front)
    assert front["thermal_resistance"].iloc[rep["min_R"]] == front["thermal_resistance"].min()
    assert front["mass"].iloc[rep["min_mass"]] == front["mass"].min()
    base = pd.DataFrame([dict(family="B2", thermal_resistance=0.03, R_active=0.02, pump_power=0.09, mass=0.33),
                         dict(family="B2", thermal_resistance=0.02, R_active=0.02, pump_power=0.5, mass=0.33)])
    t = equal_pump_power(front, base, [0.1], ["B2"])
    assert len(t) == 1 and t["R_baseline"].iloc[0] == 0.03  # the 0.5 W design is above the cap
    assert t["R_opt"].iloc[0] == front.loc[front["pump_power"] <= 0.1, "thermal_resistance"].min()
    assert t["improvement_pct"].iloc[0] == pytest.approx(100 * (1 - t["R_opt"].iloc[0] / 0.03))


@needs_models
def test_anchor_designs_are_uniform_corners_and_feasible(problem):
    from voxlat.optimize import anchor_designs

    A = anchor_designs(problem)
    par = problem.par
    assert A.shape == (9, par.n_var) and par.cheap_feasible_batch(A).all()
    seen = set()
    for x in A:
        nets = par.control_nets(x)
        for n in ("rho", "w", "L"):
            assert np.ptp(nets[n]) < 1e-12  # uniform fields
        seen.add((round(nets["rho"].mean(), 3), round(nets["w"].mean(), 3)))
    assert {(0.2, 0.0), (0.5, 0.0), (0.2, 1.0), (0.5, 1.0), (0.35, 0.0)} <= seen
    # dense diamond at the smallest cell is lifted to 1.01 L_min(1, 0.5) = 2.18 mm by the repair
    Ls = [par.control_nets(x)["L"].mean() for x in A if par.control_nets(x)["w"].mean() == 1.0
          and par.control_nets(x)["rho"].mean() == 0.5]
    assert min(Ls) == pytest.approx(1.01 * 2.16e-3, rel=0.005)  # (1 + cell_margin) L_min


def test_run_is_complete_rejects_other_budgets(tmp_path):
    import json

    from voxlat.optimize.study import run_is_complete, study_dir

    d = study_dir("slots", tmp_path)

    from voxlat.optimize.parameterization import DEFAULT_CONSTRAINTS

    def put(name, **meta):
        meta.setdefault("settings", {"constraints": DEFAULT_CONSTRAINTS})
        (d / f"{name}.json").write_text(json.dumps(meta))

    put("nsga2_s0", status="done", pop_size=100, n_gen=100, n_expensive=10000)
    put("nsga2_s1", status="done", pop_size=50, n_gen=40, n_expensive=2000)  # sandbox budget -> redo
    put("nsga2_s2", status="running", pop_size=100, n_gen=100, n_expensive=3200)
    put("sobol_s0", status="done", budget=1000, n_expensive=1000)
    put("sobol_s1", status="done", budget=500, n_expensive=500)
    put("qnehvi_s0", status="done", budget=200, n_expensive=200, bo_backend="numpy")
    kw = dict(root=tmp_path)
    assert run_is_complete("slots", "nsga2", 0, pop_size=100, n_gen=100, **kw)
    assert not run_is_complete("slots", "nsga2", 1, pop_size=100, n_gen=100, **kw)
    assert not run_is_complete("slots", "nsga2", 2, pop_size=100, n_gen=100, **kw)
    assert not run_is_complete("slots", "nsga2", 3, **kw)  # missing
    assert run_is_complete("slots", "sobol", 0, budget=1000, **kw)
    assert not run_is_complete("slots", "sobol", 1, budget=1000, **kw)
    assert run_is_complete("slots", "qnehvi", 0, budget=200, backend="numpy", **kw)
    assert not run_is_complete("slots", "qnehvi", 0, budget=200, backend="botorch", **kw)
    # finished with another cell-size rule (the first study: coarse table, no margin, no valley screen) -> redo
    put("sobol_s2", status="done", budget=1000, n_expensive=1000, settings={})
    assert not run_is_complete("slots", "sobol", 2, budget=1000, **kw)


class _Interrupt(Exception):
    pass


def _interrupt_after(monkeypatch, n_calls):
    """Make JacketProblem.evaluate raise on its (n_calls+1)-th call - simulates a killed run."""
    orig = JacketProblem.evaluate
    count = {"n": 0}

    def ev(self, X, **kw):
        count["n"] += 1
        if count["n"] > n_calls:
            raise _Interrupt()
        return orig(self, X, **kw)

    monkeypatch.setattr(JacketProblem, "evaluate", ev)
    return orig


@needs_models
@pytest.mark.parametrize("method", ["sobol", "qnehvi", "nsga2"])
def test_interrupted_run_resumes_from_checkpoint(method, tmp_path, monkeypatch):
    import json

    from voxlat.optimize.study import checkpoint_status, load_checkpoint, run_is_complete, run_one

    if method == "nsga2" and not have_pymoo():
        pytest.skip("pymoo not installed")
    kw = {"sobol": dict(budget=30), "qnehvi": dict(budget=14, n_init=10, q=2, backend="numpy"),
          "nsga2": dict(pop_size=10, n_gen=4)}[method]
    kw.update(root=tmp_path, checkpoint_every=0.0)
    orig = _interrupt_after(monkeypatch, {"sobol": 2, "qnehvi": 3, "nsga2": 2}[method])
    with pytest.raises(_Interrupt):
        run_one("end_rings", method, 0, **kw)
    meta = checkpoint_status("end_rings", method, 0, **kw)
    assert meta is not None and meta["status"] == "running"
    before = [r for r in load_checkpoint(tmp_path / "end_rings" / f"{method}_s0.csv.gz") if r["expensive"]]
    assert 0 < len(before)
    monkeypatch.setattr(JacketProblem, "evaluate", orig)
    calls = {"n": 0}

    def counting(self, X, **k):
        calls["n"] += len(np.atleast_2d(X))
        return orig(self, X, **k)

    monkeypatch.setattr(JacketProblem, "evaluate", counting)
    run_one("end_rings", method, 0, resume=True, **kw)
    meta = json.loads((tmp_path / "end_rings" / f"{method}_s0.json").read_text())
    assert meta["status"] == "done" and meta["resumed"][0]["from_n_expensive"] == len(before)
    assert run_is_complete("end_rings", method, 0, **kw)
    after = load_checkpoint(tmp_path / "end_rings" / f"{method}_s0.csv.gz")
    ev = [r for r in after if r["expensive"]]
    # the archived evaluations are kept as they were and never re-run
    for a, b in zip(before, ev):
        assert a["thermal_resistance"] == pytest.approx(b["thermal_resistance"], rel=1e-7)
    assert [r["eval_id"] for r in ev] == list(range(len(ev)))
    if method == "nsga2":
        assert max(r["iteration"] for r in after) == kw["n_gen"] - 1  # generations continue, total n_gen
        assert calls["n"] <= (kw["n_gen"] - 1 - max(r["iteration"] for r in before)) * kw["pop_size"] + kw["pop_size"]
        key = {tuple(np.round(r["x"], 6)) for r in before}
        assert sum(tuple(np.round(r["x"], 6)) in key for r in ev[len(before):]) == 0  # nothing evaluated twice
    else:
        assert len(ev) == kw["budget"]
        assert sum(1 for r in after if r.get("anchor") is True) == 9  # anchors not repeated
    if method == "sobol":
        # identical to an uninterrupted run
        run_one("end_rings", "sobol", 0, root=tmp_path / "ref", budget=30)
        ref = [r for r in load_checkpoint(tmp_path / "ref" / "end_rings" / "sobol_s0.csv.gz") if r["expensive"]]
        np.testing.assert_allclose([r["thermal_resistance"] for r in ev], [r["thermal_resistance"] for r in ref],
                                   rtol=1e-6)


@needs_models
def test_valley_screen_rejects_pinch_neck_blends_and_repair_moves_out(problem):
    """Designs with cells on a pinch-neck valley are infeasible; the repair polarizes w out of the valley band,
    and repaired designs pass a direct n = 48 check on their tightest blend cells."""
    from voxlat.geometry import TPMSParams, check_manufacturable

    par = problem.par
    assert par.constraints["table"] == "fine" and par.constraints["valley_wall"] == 0.09
    # w 0.75 / rho* 0.21 everywhere: on a valley (direct check: 0.11 mm wall at L 5.3 mm), L_min_fine < 6 mm
    nets = {"rho": np.full((8, 4), 0.21), "w": np.full((8, 4), 0.752), "L": np.full((4, 2), 5.6e-3)}
    x0 = par.x_from_control_nets(nets)
    assert par.cheap_constraints(x0)["cell_size"] > 0
    rep = check_manufacturable(TPMSParams.from_design(0.752, 0.21, 1.0), 5.6e-3, n=48)
    assert not rep.ok  # the screen is right to reject it
    y = par.repair(x0)
    assert par.cheap_feasible_batch(y[None])[0]
    w_new = float(par.control_nets(y)["w"].min())
    assert w_new > 0.85  # polarized toward diamond, out of the valley band
    assert check_manufacturable(TPMSParams.from_design(w_new, 0.21, 1.0), 5.6e-3, n=48).ok
    rng = np.random.default_rng(7)
    Y = par.repair_batch(rng.random((128, par.n_var)))
    ok = par.cheap_feasible_batch(Y)
    assert ok.mean() > 0.8
