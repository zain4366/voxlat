"""Optimizers for Task 11: NSGA-II (pymoo), qNEHVI Bayesian optimization (BoTorch, with a
numpy EHVI fallback) and scrambled-Sobol random search.

Common rules (fair comparison, RQ3)
-----------------------------------
* Every method proposes points in the unit cube; the same feasibility ``repair`` (cell size
  L >= L_min and grading-gradient limit, see ``parameterization``) is applied to every
  proposal, and the device model runs only for designs that are then cheap-feasible. The budget and the x-axis of every hypervolume curve is the
  number of EXPENSIVE (device-model) evaluations; cheap rejects are recorded with eval_id = -1.
* Objectives are the normalized (R, log10 pump power, mass) of ``ObjectiveSpace``; feasible =
  manufacturable and min structural margin >= 1.
* Every method starts from the same 9 uniform ``anchor_designs`` (counted in its budget; switch
  off with ``anchors=False``): without them no method reached the dense/fine corner of the
  front, where uniform designs are optimal (ablation in STATUS.md, Task 11).

NSGA-II (``run_nsga2``)
    pymoo ``NSGA2`` with its defaults (random initial population, binary tournament on rank and
    crowding with constraint domination, SBX eta = 15 p = 0.9, polynomial mutation eta = 20
    p = 1/n), duplicate elimination and the repair as a pymoo ``Repair``. Cheap rejects get a
    placeholder objective vector and a positive constraint value (never ranked as feasible).

qNEHVI (``run_qnehvi``)
    Initial scrambled-Sobol design of ``n_init`` evaluations, then batches of ``q`` points:
    * surrogates: one GP per uncertain outcome (normalized R, normalized log10 P, log margin);
      the mass is linear in the rho controls and treated as KNOWN (zero variance);
    * acquisition, backend ``"botorch"``: qLogNoisyExpectedHypervolumeImprovement
      (Daulton et al. 2021; Ament et al. 2023) with the margin as an outcome constraint, SingleTaskGP
      with BoTorch's default (dimension-scaled) priors; backend ``"numpy"`` (used when BoTorch
      is not installed): analytic EHVI for independent Gaussian outcomes times the probability
      of feasibility, batches by sequential greedy selection with kriging-believer fantasies.
      For a deterministic simulator (noise -> 0) NEHVI reduces to EHVI w.r.t. the observed
      feasible Pareto front, so the fallback approximates the same acquisition; it differs in the
      GP details and in the joint-batch (MC) vs greedy-batch treatment. Results record the backend.
    * acquisition optimization (both backends): a candidate pool of repaired scrambled-Sobol
      points plus perturbations of the current Pareto set (random ~20/n subset of coordinates,
      sigma in {0.02, 0.05, 0.1, 0.2}), then a second round around the best ``n_refine``
      candidates; designs failing the cheap cell-size check are skipped.

Random search (``run_sobol``)
    Scrambled Sobol sequence (Owen scrambling, seeded), repaired, until the budget of expensive
    evaluations is reached.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from voxlat.optimize.gp import GaussianProcess
from voxlat.optimize.hypervolume import box_decomposition, expected_hvi, pareto_mask
from voxlat.optimize.problem import JacketProblem
from voxlat.utils.log import get_logger

__all__ = ["run_sobol", "run_nsga2", "run_qnehvi", "anchor_designs", "have_pymoo", "have_botorch", "BOResult"]

LOG = get_logger("optimize.algorithms")


def have_pymoo() -> bool:
    try:
        import pymoo  # noqa: F401
        from pymoo.algorithms.moo.nsga2 import NSGA2  # noqa: F401
    except Exception:  # pragma: no cover - environment dependent
        return False
    return True


def have_botorch() -> bool:
    try:
        import botorch  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _sobol(d: int, n: int, rng: np.random.Generator) -> np.ndarray:
    from scipy.stats import qmc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qmc.Sobol(d, scramble=True, seed=rng).random(n)


# =============================================================================
# Random search
# =============================================================================
def anchor_designs(problem: JacketProblem) -> np.ndarray:
    """The 9 uniform anchor designs every method starts from: pure gyroid / pure diamond at the four
    (rho*, L) corners of the design box, plus the box centre gyroid (rho* 0.35, L 4 mm = the Task 9
    reference). The min-R extreme of the front sits at a corner (every field at a bound -> uniform),
    which random initial designs - B-spline averages of random controls, mostly mid-range - never reach.
    Repaired like any proposal (dense diamond at L = 2 mm becomes L = L_min = 2.16 mm)."""
    par = problem.par
    rows = []
    for w in (0.0, 1.0):
        for r in (0.0, 1.0):
            for L in (0.0, 1.0):
                x = np.zeros(par.n_var)
                for name, v in (("rho", r), ("w", w), ("L", L)):
                    if name in par._slices:
                        x[par._slices[name]] = v
                rows.append(x)
    c = np.full(par.n_var, 0.5)
    if "w" in par._slices:
        c[par._slices["w"]] = 0.0
    rows.append(c)
    return par.repair_batch(np.array(rows))


def run_sobol(problem: JacketProblem, budget: int, *, seed: int = 0, batch: int | None = None, anchors: bool = True,
              callback: Callable[[JacketProblem], None] | None = None) -> JacketProblem:
    """Scrambled-Sobol random search until ``budget`` expensive evaluations (after the anchors, if any)."""
    from scipy.stats import qmc

    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = qmc.Sobol(problem.n_var, scramble=True, seed=rng)
    batch = batch or max(8, 4 * problem.n_jobs)
    # resume: records already in the archive (loaded from a checkpoint) are kept; the anchors are not repeated and
    # the Sobol engine skips the points already drawn, so the continued run equals an uninterrupted one
    initial = [r for r in problem.records if _iteration(r) <= 0]
    n_anchor_done = sum(1 for r in initial if _is_anchor(r))
    if len(initial) > n_anchor_done:
        eng.fast_forward(len(initial) - n_anchor_done)
    if anchors and n_anchor_done == 0 and problem.n_expensive < budget:
        A = anchor_designs(problem)[: budget - problem.n_expensive]
        problem.evaluate(A, tag={"method": "sobol", "seed": seed, "iteration": -1, "anchor": True})
    while problem.n_expensive < budget:
        need = budget - problem.n_expensive
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = eng.random(min(batch, need))
        problem.evaluate(problem.par.repair_batch(X), tag={"method": "sobol", "seed": seed, "iteration": -1})
        if callback:
            callback(problem)
    return problem


def _iteration(r: Mapping[str, Any]) -> int:
    v = r.get("iteration", -1)
    try:
        return int(v) if v == v else -1  # NaN -> -1
    except (TypeError, ValueError):
        return -1


def _is_anchor(r: Mapping[str, Any]) -> bool:
    v = r.get("anchor", False)
    return bool(v) and v == v and str(v).lower() not in ("false", "nan", "0", "0.0")


# =============================================================================
# NSGA-II (pymoo)
# =============================================================================
PLACEHOLDER_F = 2.0  # normalized objective given to cheap rejects (never feasible)


def _nsga_FG(jp: JacketProblem, r: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Objective and constraint vector pymoo sees for one archive record."""
    F = np.full(3, PLACEHOLDER_F)
    G = np.array([r["g_gradient"], r["g_cell_size"], 0.0], float)
    if r["expensive"]:
        F = jp.space.normalize(r["thermal_resistance"], r["pump_power"], r["mass"])
        G[2] = r["g_margin"] if r["manufacturable"] else max(r["g_margin"], 1e-3)
    return np.asarray(F, float), G


def run_nsga2(problem: JacketProblem, *, pop_size: int = 100, n_gen: int = 100, seed: int = 0, anchors: bool = True,
              callback: Callable[[JacketProblem], None] | None = None, **nsga_kw: Any) -> JacketProblem:
    """pymoo NSGA-II reference run (``pop_size`` x ``n_gen`` proposals).

    Resume: if ``problem.records`` already holds an interrupted run (loaded from its checkpoint), the run continues
    from it. Every archived design is handed back to pymoo with its stored objectives (no re-evaluation), the
    NSGA-II survival (rank + crowding, constraint domination) picks the population of ``pop_size`` from them, and
    the remaining ``n_gen - completed`` generations are run. Because NSGA-II is elitist, this population matches
    the one the uninterrupted run would have had except for ties among equally ranked designs; the random stream
    after resuming differs (seeded with (seed, completed generations)).
    """
    if not have_pymoo():
        raise ImportError("NSGA-II needs pymoo: pip install -e .[opt]  (or pip install pymoo)")
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.core.repair import Repair
    from pymoo.optimize import minimize

    jp = problem

    class FeasibilityRepair(Repair):
        def _do(self, prob: Any, X: Any, **kw: Any) -> Any:
            return jp.par.repair_batch(X)

    class PymooJacket(Problem):
        def __init__(self) -> None:
            super().__init__(n_var=jp.n_var, n_obj=3, n_ieq_constr=3, xl=0.0, xu=1.0)
            self.generation = 0

        def _evaluate(self, X: Any, out: dict, *args: Any, **kwargs: Any) -> None:
            X = np.atleast_2d(np.asarray(X, float))
            F = np.empty((len(X), 3))
            G = np.empty((len(X), 3))
            new = [i for i in range(len(X)) if _key(X[i]) not in cache]
            if new:
                recs = jp.evaluate(X[new], tag={"method": "nsga2", "seed": seed, "iteration": self.generation})
                self.generation += 1
                for r in recs:
                    if r["expensive"]:
                        cache[_key(r["x"])] = _nsga_FG(jp, r)
                for i, r in zip(new, recs):
                    F[i], G[i] = _nsga_FG(jp, r)
            for i in range(len(X)):
                if i not in new:
                    F[i], G[i] = cache[_key(X[i])]
            out["F"] = F
            out["G"] = G
            if new and callback:
                callback(jp)

    def _key(x: np.ndarray) -> bytes:
        return np.round(np.asarray(x, float), 10).tobytes()

    # archive of an interrupted run (resume): stored objectives, keyed by design vector
    cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    prior = [r for r in jp.records if r["expensive"] and np.all(np.isfinite(r["x"]))]
    for r in prior:
        cache[_key(r["x"])] = _nsga_FG(jp, r)
    completed = (max(_iteration(r) for r in jp.records) + 1) if jp.records else 0
    prob = PymooJacket()
    if prior and completed > 0:
        from pymoo.core.evaluator import Evaluator
        from pymoo.core.population import Population
        from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

        if completed >= n_gen:
            return problem
        pop = Population.new(X=np.array([r["x"] for r in prior]))
        Evaluator().eval(prob, pop)  # cache hits only - nothing is re-run
        rs = np.random.default_rng([seed, completed])
        pop = RankAndCrowding().do(prob, pop, n_survive=min(pop_size, len(pop)), random_state=rs)
        prob.generation = completed
        algo = NSGA2(pop_size=pop_size, sampling=pop, repair=FeasibilityRepair(), eliminate_duplicates=True, **nsga_kw)
        # pymoo counts the (already evaluated) initial population as generation 1
        minimize(prob, algo, ("n_gen", n_gen - completed + 1), seed=int(rs.integers(2**31 - 1)), verbose=False)
        return problem

    if anchors and "sampling" not in nsga_kw:
        # initial population = anchors + uniform random (pymoo's default FloatRandomSampling distribution)
        A = anchor_designs(jp)
        R = np.random.default_rng(seed).random((max(0, pop_size - len(A)), jp.n_var))
        nsga_kw["sampling"] = np.vstack([A, R])[:pop_size]
    algo = NSGA2(pop_size=pop_size, repair=FeasibilityRepair(), eliminate_duplicates=True, **nsga_kw)
    minimize(prob, algo, ("n_gen", n_gen), seed=seed, verbose=False)
    return problem


# =============================================================================
# Bayesian optimization
# =============================================================================
@dataclass
class BOResult:
    backend: str
    iterations: int
    timings: dict[str, float]


def _mass_linear(problem: JacketProblem) -> tuple[float, np.ndarray]:
    """mass(x) = a + w . x (exact: the rho field is linear in the unit rho controls)."""
    par = problem.par
    x0 = np.zeros(par.n_var)
    a = par.mass(par.fields(x0))
    w = np.zeros(par.n_var)
    sl = par._slices.get("rho")
    if sl is not None:
        for i in range(sl.start, sl.stop):
            e = x0.copy()
            e[i] = 1.0
            w[i] = par.mass(par.fields(e)) - a
    return a, w


def _perturb(rng: np.random.Generator, centres: np.ndarray, n: int, sigmas=(0.02, 0.05, 0.1, 0.2),
             dims_frac: float | None = None) -> np.ndarray:
    d = centres.shape[1]
    p = min(1.0, 20.0 / d) if dims_frac is None else dims_frac
    idx = rng.integers(0, len(centres), n)
    X = centres[idx].copy()
    mask = rng.random((n, d)) < p
    none = ~mask.any(1)
    mask[none, rng.integers(0, d, none.sum())] = True
    sig = rng.choice(np.asarray(sigmas), size=(n, 1))
    X = X + mask * sig * rng.standard_normal((n, d))
    return np.clip(X, 0.0, 1.0)


class _NumpyAcquisition:
    """Analytic EHVI x PoF with independent GPs; known (linear) mass."""

    name = "numpy-ehvi"

    def __init__(self, problem: JacketProblem, X: np.ndarray, Fn: np.ndarray, log_margin: np.ndarray,
                 feasible: np.ndarray, mass_lin: tuple[float, np.ndarray], seed: int, gps: dict | None):
        self.p = problem
        self.ref = problem.space.ref
        a, w = mass_lin
        lo, hi = problem.space.lo[2], problem.space.hi[2]
        self.mass_n = lambda Xc: (a + Xc @ w - lo) / (hi - lo)
        gps = gps or {}
        self.gps = {}
        for k, y in (("f1", Fn[:, 0]), ("f2", Fn[:, 1]), ("lm", log_margin)):
            gp = gps.get(k, GaussianProcess(seed=seed))
            self.gps[k] = gp.fit(X, y)
        self.front = Fn[feasible & np.all(Fn < self.ref, axis=1)]
        self.boxes = box_decomposition(self.front, self.ref) if len(self.front) else box_decomposition(np.zeros((0, 3)), self.ref)

    def predict(self, Xc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m1, s1 = self.gps["f1"].predict(Xc)
        m2, s2 = self.gps["f2"].predict(Xc)
        ml, sl = self.gps["lm"].predict(Xc)
        mu = np.column_stack([m1, m2, self.mass_n(Xc)])
        sd = np.column_stack([s1, s2, np.zeros(len(Xc))])
        from scipy.special import ndtr

        pof = ndtr(ml / np.maximum(sl, 1e-12))
        return mu, sd, pof

    def score(self, Xc: np.ndarray) -> np.ndarray:
        mu, sd, pof = self.predict(Xc)
        return expected_hvi(mu, sd, self.boxes) * pof

    def fantasize(self, x: np.ndarray) -> None:
        mu, _, pof = self.predict(x[None])
        self.gps["f1"] = self.gps["f1"].condition_on(x[None], mu[:, 0])
        self.gps["f2"] = self.gps["f2"].condition_on(x[None], mu[:, 1])
        self.gps["lm"] = self.gps["lm"].condition_on(x[None], [self.gps["lm"].predict(x[None], return_std=False)[0]])
        if pof[0] > 0.5 and np.all(mu[0] < self.ref):
            self.front = np.vstack([self.front, mu]) if len(self.front) else mu.copy()
            self.boxes = box_decomposition(self.front, self.ref)


def _feasible_candidates(par: Any, X: np.ndarray, exclude: np.ndarray | None = None) -> np.ndarray:
    """Repair, keep the cheap-feasible rows, drop duplicates of ``exclude`` (already evaluated)."""
    if len(X) == 0:
        return X
    Y = par.repair_batch(X)
    Y = Y[par.cheap_feasible_batch(Y)]
    if exclude is not None and len(exclude) and len(Y):
        from scipy.spatial import cKDTree

        d, _ = cKDTree(exclude).query(Y, k=1)
        Y = Y[d > 1e-9]
    return Y


def _candidate_pool(problem: JacketProblem, X: np.ndarray, Fn: np.ndarray, feas: np.ndarray,
                    rng: np.random.Generator, pool_sobol: int, pool_local: int) -> np.ndarray:
    par = problem.par
    if feas.any():
        ps = X[feas][pareto_mask(Fn[feas])]
    else:
        ps = X[np.argsort(np.nansum(Fn, axis=1))[:5]]
    raw = np.vstack([_sobol(par.n_var, pool_sobol, rng), _perturb(rng, ps, pool_local)])
    return _feasible_candidates(par, raw, X)


def _select_numpy(acq: _NumpyAcquisition, pool: np.ndarray, q: int, rng: np.random.Generator, n_refine: int,
                  n_per: int, exclude: np.ndarray) -> np.ndarray:
    """Sequential greedy batch: refine around the best pool points, pick the max, fantasize, repeat."""
    par = acq.p.par
    chosen: list[np.ndarray] = []
    s = acq.score(pool)
    top = np.argsort(-s)[:n_refine]
    loc = _feasible_candidates(par, _perturb(rng, pool[top], n_refine * n_per, sigmas=(0.01, 0.02, 0.05)), exclude)
    cand = np.vstack([pool, loc]) if len(loc) else pool
    for _ in range(q):
        sc = acq.score(cand)
        order = np.argsort(-sc)
        pick = None
        for i in order:
            if not any(np.array_equal(cand[i], c) for c in chosen):
                pick = cand[i]
                break
        if pick is None:
            break
        chosen.append(pick)
        acq.fantasize(pick)
    return np.array(chosen)


def _select_botorch(problem: JacketProblem, X: np.ndarray, Fn: np.ndarray, log_margin: np.ndarray,
                    mass_lin: tuple[float, np.ndarray], pool: np.ndarray, q: int, rng: np.random.Generator,
                    n_refine: int, n_per: int, chunk: int = 8) -> np.ndarray:
    """qLogNEHVI over a discrete, cheap-feasible candidate set (BoTorch; maximizes -f).

    Outcomes: GP(-f~_R), GP(-f~_logP), the known linear mass (GenericDeterministicModel) and GP(log margin)
    in a ``ModelList``; BoTorch picks the matching (list) sampler, 128 QMC samples by default.
    Memory: qNEHVI integrates over Pareto fronts of the MC samples of the baseline, so one forward pass
    holds (candidates x MC samples x hypervolume cells x outcomes) numbers - ~GB for 256 candidates and a
    3-objective front of ~50-100 points. Candidates are therefore scored in chunks of ``chunk`` (8, ~0.1 GB);
    the first laptop run, with 256, crashed the worker process (Task 11 log). torch uses one thread per process
    (runs are parallelized at the process level; VOXLAT_TORCH_THREADS overrides, e.g. one run per cloud machine).
    """
    import torch
    from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
    from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.deterministic import GenericDeterministicModel
    from botorch.models.model import ModelList
    from botorch.models.transforms import Standardize
    from botorch.optim import optimize_acqf_discrete
    from gpytorch.mlls import ExactMarginalLogLikelihood

    torch.set_num_threads(int(os.environ.get("VOXLAT_TORCH_THREADS", "1")))  # >1 only when a run has the machine to itself
    tk = dict(dtype=torch.double)
    Xt = torch.tensor(X, **tk)
    models = []
    for y in (-Fn[:, 0], -Fn[:, 1]):
        gp = SingleTaskGP(Xt, torch.tensor(y, **tk)[:, None], outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        models.append(gp)
    a, w = mass_lin
    lo, hi = problem.space.lo[2], problem.space.hi[2]
    w_t = torch.tensor(w, **tk)
    models.append(GenericDeterministicModel(lambda Z: (-(a + Z @ w_t - lo) / (hi - lo)).unsqueeze(-1), num_outputs=1))
    gpm = SingleTaskGP(Xt, torch.tensor(log_margin, **tk)[:, None], outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gpm.likelihood, gpm))
    models.append(gpm)
    model = ModelList(*models)
    acq = qLogNoisyExpectedHypervolumeImprovement(
        model=model, ref_point=(-problem.space.ref).tolist(), X_baseline=Xt,
        objective=IdentityMCMultiOutputObjective(outcomes=[0, 1, 2]),
        constraints=[lambda Z: -Z[..., 3]],  # feasible: log margin >= 0
        prune_baseline=True, cache_root=False)

    def score(C: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for a0 in range(0, len(C), chunk):
                out.append(acq(torch.tensor(C[a0:a0 + chunk], **tk)[:, None, :]).cpu().numpy())
        return np.concatenate(out)

    par = problem.par
    s = score(pool)
    top = np.argsort(-s)[:n_refine]
    loc = _feasible_candidates(par, _perturb(rng, pool[top], n_refine * n_per, sigmas=(0.01, 0.02, 0.05)), X)
    cand = np.vstack([pool, loc]) if len(loc) else pool
    sc = score(cand)
    choices = cand[np.argsort(-sc)[: max(32, 8 * q)]]
    sel, _ = optimize_acqf_discrete(acq, q=q, choices=torch.tensor(choices, **tk), unique=True, max_batch_size=chunk)
    return sel.cpu().numpy()


def run_qnehvi(problem: JacketProblem, budget: int = 200, *, n_init: int = 32, q: int = 4, seed: int = 0,
               backend: str = "auto", pool_sobol: int = 256, pool_local: int = 768, n_refine: int = 8,
               n_per: int = 48, anchors: bool = True, callback: Callable[[JacketProblem], None] | None = None) -> BOResult:
    """Constrained multi-objective BO (qNEHVI / numpy EHVI fallback); ``budget`` expensive evaluations."""
    if backend == "auto":
        backend = "botorch" if have_botorch() else "numpy"
    if backend == "botorch" and not have_botorch():
        raise ImportError("backend='botorch' needs botorch + torch: pip install -e .[opt]")
    it0 = max((_iteration(r) for r in problem.records), default=0)  # > 0 when resuming from a checkpoint
    rng = np.random.default_rng(seed if it0 <= 0 else [seed, it0])
    par = problem.par
    timings = {"model_fit_and_acq": 0.0, "evaluation": 0.0, "botorch_fallbacks": 0}
    t0 = time.perf_counter()
    n0 = len(problem.records)
    if it0 <= 0:
        run_sobol(problem, min(n_init, budget), seed=seed, anchors=anchors)
    for r in problem.records[n0:]:
        r.update(method="qnehvi", iteration=0, bo_backend=backend)
    timings["evaluation"] += time.perf_counter() - t0
    mass_lin = _mass_linear(problem)
    gps: dict = {}
    it = max(it0, 0)
    while problem.n_expensive < budget:
        it += 1
        t1 = time.perf_counter()
        ev = [r for r in problem.records if r["expensive"]]
        X = np.array([r["x"] for r in ev])
        Fn = problem.objectives(ev)
        margin = np.array([r["min_structural_margin"] for r in ev], float)
        log_margin = np.log(np.maximum(margin, 1e-6))
        feas = np.array([problem.feasible(r) for r in ev])
        qq = min(q, budget - problem.n_expensive)
        pool = _candidate_pool(problem, X, Fn, feas, rng, pool_sobol, pool_local)
        if len(pool) == 0:
            LOG.warning("empty candidate pool at iteration %d; drawing Sobol points", it)
            run_sobol(problem, problem.n_expensive + qq, seed=seed + 1000 + it)
            continue
        if backend == "numpy":
            acq = _NumpyAcquisition(problem, X, Fn, log_margin, feas, mass_lin, seed + it, gps)
            gps = dict(acq.gps)  # fitted models: warm starts for the next iteration's hyperparameters
            Xn = _select_numpy(acq, pool, qq, rng, n_refine, n_per, X)
        else:
            try:
                Xn = _select_botorch(problem, X, Fn, log_margin, mass_lin, pool, qq, rng, n_refine, n_per)
                used = "botorch"
            except Exception as exc:  # noqa: BLE001 - e.g. out-of-memory or a BoTorch API change: keep the run alive
                LOG.warning("BoTorch acquisition failed at iteration %d (%s: %s); numpy EHVI for this batch",
                            it, type(exc).__name__, exc)
                timings["botorch_fallbacks"] += 1
                acq = _NumpyAcquisition(problem, X, Fn, log_margin, feas, mass_lin, seed + it, gps)
                gps = dict(acq.gps)
                Xn = _select_numpy(acq, pool, qq, rng, n_refine, n_per, X)
                used = "numpy-fallback"
        t2 = time.perf_counter()
        problem.evaluate(Xn, tag={"method": "qnehvi", "seed": seed, "iteration": it,
                                  "bo_backend": backend if backend == "numpy" else used})
        timings["model_fit_and_acq"] += t2 - t1
        timings["evaluation"] += time.perf_counter() - t2
        if callback:
            callback(problem)
    return BOResult(backend, it, timings)
