"""Running, storing and analysing the Task 11 studies (RQ2, RQ3).

Layout of ``results/task11/`` (one folder per manifold layout):

    <layout>/<method>_s<seed>.csv.gz     every proposal (rows; eval_id = -1 for cheap rejects), the
                                         repaired unit design vector x00.., objectives, constraints
    <layout>/<method>_s<seed>.json       run record: settings, budget, backend, timings, versions

Analysis helpers: hypervolume curves per run (``hv_curves``), the combined feasible Pareto
front (``combined_front``), representative designs (``representative_designs``), the
equal-pump-power comparison with the Task 10 baselines (``equal_pump_power``) and the design
JSON written to ``results/designs/`` (``design_record``).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from voxlat.optimize.hypervolume import hypervolume, pareto_mask
from voxlat.optimize.problem import JacketProblem, ObjectiveSpace, ProblemSettings, evaluate_design_fields
from voxlat.utils.log import get_logger
from voxlat.utils.paths import results_dir
from voxlat.utils.provenance import run_record, to_jsonable

__all__ = [
    "METHODS",
    "study_dir",
    "run_one",
    "run_is_complete",
    "checkpoint_status",
    "rescreen_cell_size",
    "load_checkpoint",
    "load_runs",
    "hv_curves",
    "combined_front",
    "representative_designs",
    "equal_pump_power",
    "design_record",
    "evaluate_x",
]

LOG = get_logger("optimize.study")
METHODS = ("nsga2", "qnehvi", "sobol")


def study_dir(layout: str, root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else results_dir() / "task11"
    d = base / layout
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(problem: JacketProblem, path: Path, meta: Mapping[str, Any]) -> None:
    df = problem.table()
    if len(df):
        # cheap rejects keep their constraint values but not x (reproducible from the seed; saves ~90 % space)
        xcols = [c for c in df.columns if c.startswith("x") and c[1:].isdigit()]
        df.loc[~df["expensive"].astype(bool), xcols] = np.nan
    df.to_csv(path.with_suffix(".csv.gz"), index=False, float_format="%.8g")
    path.with_suffix(".json").write_text(json.dumps(to_jsonable(meta), indent=2))


_DERIVED_COLUMNS = ("feasible", "f1_R", "f2_logP", "f3_mass")
_BOOL_COLUMNS = ("expensive", "manufacturable", "picard_converged", "anchor")


def load_checkpoint(csv_path: str | Path) -> list[dict[str, Any]]:
    """Archive records of a saved run (``<method>_s<seed>.csv.gz``), in the JacketProblem record format.

    Cheap rejects come back without a design vector (x = NaN; they are stored without it)."""
    df = pd.read_csv(csv_path)
    xcols = sorted((c for c in df.columns if c.startswith("x") and c[1:].isdigit()), key=lambda c: int(c[1:]))
    other = [c for c in df.columns if c not in xcols and c not in _DERIVED_COLUMNS]
    X = df[xcols].to_numpy(float)
    recs = []
    for i, row in enumerate(df[other].to_dict("records")):
        for k in _BOOL_COLUMNS:
            if k in row:
                v = row[k]
                row[k] = bool(v) if (isinstance(v, (bool, np.bool_)) or (isinstance(v, (int, float)) and v == v)) \
                    else (str(v).lower() == "true" if isinstance(v, str) else False)
        for k in ("eval_id", "proposal_id"):
            if k in row and row[k] == row[k]:
                row[k] = int(row[k])
        row["x"] = X[i].copy()
        recs.append(row)
    return recs


def _same_settings(m: Mapping[str, Any], method: str, pop_size: int, n_gen: int, budget: int | None) -> bool:
    from voxlat.optimize.parameterization import DEFAULT_CONSTRAINTS

    legacy = {"table": "coarse", "cell_margin": 0.0, "valley_wall": None}
    if (m.get("settings") or {}).get("constraints", legacy) != DEFAULT_CONSTRAINTS:
        return False  # run made with another cell-size rule (e.g. the first study) -> redo
    if method == "nsga2":
        return m.get("pop_size") == pop_size and m.get("n_gen") == n_gen
    return m.get("budget") == (budget or 200)


def checkpoint_status(layout: str, method: str, seed: int, *, root: str | Path | None = None, pop_size: int = 100,
                      n_gen: int = 100, budget: int | None = None, **_: Any) -> dict[str, Any] | None:
    """The saved state of an interrupted run with the same settings (status 'running' + csv.gz), else None."""
    base = study_dir(layout, root) / f"{method}_s{seed}"
    meta_path, csv_path = base.with_suffix(".json"), base.with_suffix(".csv.gz")
    if not (meta_path.is_file() and csv_path.is_file()):
        return None
    try:
        m = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None
    if m.get("status") != "running" or not _same_settings(m, method, pop_size, n_gen, budget):
        return None
    return m


def run_one(layout: str, method: str, seed: int, *, budget: int | None = None, pop_size: int = 100,
            n_gen: int = 100, n_init: int = 32, q: int = 4, backend: str = "auto", n_jobs: int = 1,
            root: str | Path | None = None, spec: Any = None, checkpoint_every: float = 120.0,
            anchors: bool = True, resume: bool = False) -> Path:
    """Run one (layout, method, seed) and write the csv.gz + json (partial results every ~2 min).

    ``resume``: if an interrupted run with the same settings left a checkpoint, continue from it (its evaluations
    are reloaded, not repeated; wall time accumulates; ``meta["resumed"]`` records each restart)."""
    from voxlat.optimize.algorithms import run_nsga2, run_qnehvi, run_sobol

    settings = ProblemSettings(layout=layout, spec=spec) if spec is not None else ProblemSettings(layout=layout)
    problem = JacketProblem(settings, n_jobs=n_jobs)
    path = study_dir(layout, root) / f"{method}_s{seed}"
    t0 = time.perf_counter()
    meta: dict[str, Any] = {"layout": layout, "method": method, "seed": seed, "settings": settings.to_dict(),
                            "objective_space": {"lo": problem.space.lo, "hi": problem.space.hi,
                                                "transform": "R mK/W, log10 pump power W, mass kg"},
                            "n_var": problem.n_var, "variable_names": problem.par.variable_names(), "anchors": anchors}
    prev = checkpoint_status(layout, method, seed, root=root, pop_size=pop_size, n_gen=n_gen, budget=budget) \
        if resume else None
    if prev is not None:
        problem.records = load_checkpoint(path.with_suffix(".csv.gz"))
        prev_wall = float(prev.get("wall_time_s") or 0.0)
        t0 -= prev_wall  # wall time accumulates over restarts
        meta["resumed"] = list(prev.get("resumed", [])) + [{"from_n_expensive": problem.n_expensive,
                                                            "from_n_proposals": len(problem.records),
                                                            "previous_wall_time_s": prev_wall}]
        LOG.info("%s/%s seed %d: resuming from checkpoint (%d expensive evaluations)", layout, method, seed,
                 problem.n_expensive)
    last = [time.perf_counter()]

    def cb(p: JacketProblem) -> None:
        if time.perf_counter() - last[0] > checkpoint_every:
            meta.update(status="running", wall_time_s=time.perf_counter() - t0, n_expensive=p.n_expensive)
            _save(p, path, meta)
            last[0] = time.perf_counter()

    if method == "nsga2":
        meta.update(pop_size=pop_size, n_gen=n_gen)
        run_nsga2(problem, pop_size=pop_size, n_gen=n_gen, seed=seed, anchors=anchors, callback=cb)
    elif method == "qnehvi":
        meta.update(budget=budget or 200, n_init=n_init, q=q)
        res = run_qnehvi(problem, budget or 200, n_init=n_init, q=q, seed=seed, backend=backend, anchors=anchors,
                         callback=cb)
        meta.update(bo_backend=res.backend, bo_iterations=res.iterations, bo_timings=res.timings)
    elif method == "sobol":
        meta.update(budget=budget or 200)
        run_sobol(problem, budget or 200, seed=seed, anchors=anchors, callback=cb)
    else:
        raise ValueError(f"unknown method {method!r}")
    wall = time.perf_counter() - t0
    meta.update(status="done", wall_time_s=wall, n_expensive=problem.n_expensive, n_proposals=len(problem.records),
                run_record=run_record(meta["settings"], wall_time=wall, method=method, seed=seed))
    _save(problem, path, meta)
    problem.close()
    LOG.info("%s/%s seed %d: %d expensive evaluations in %.0f s", layout, method, seed, problem.n_expensive, wall)
    return path


def run_is_complete(layout: str, method: str, seed: int, *, root: str | Path | None = None, pop_size: int = 100,
                    n_gen: int = 100, budget: int | None = None, backend: str = "auto", **_: Any) -> bool:
    """True if <method>_s<seed>.json is a finished run with the SAME budget settings (and, for qNEHVI, the
    same resolved backend) - so stale results from other budgets (e.g. the sandbox runs shipped with the
    repo) are redone by ``--skip-done`` instead of being kept."""
    meta_path = study_dir(layout, root) / f"{method}_s{seed}.json"
    if not meta_path.is_file():
        return False
    try:
        m = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return False
    if m.get("status") != "done":
        return False
    if not _same_settings(m, method, pop_size, n_gen, budget):
        return False
    if method == "nsga2":
        return True
    if int(m.get("n_expensive", 0)) < (budget or 200):
        return False
    if method == "qnehvi":
        from voxlat.optimize.algorithms import have_botorch

        want = ("botorch" if have_botorch() else "numpy") if backend == "auto" else backend
        return m.get("bo_backend") == want
    return True


# =============================================================================
# Analysis
# =============================================================================
def load_runs(layout: str, root: str | Path | None = None, methods: Iterable[str] = METHODS) -> pd.DataFrame:
    """All proposals of all runs of one layout (adds 'run' = '<method>_s<seed>')."""
    d = study_dir(layout, root)
    frames = []
    for m in methods:
        for f in sorted(d.glob(f"{m}_s*.csv.gz")):
            df = pd.read_csv(f)
            meta = json.loads(f.with_name(f.name.replace(".csv.gz", ".json")).read_text())
            df["method"] = m
            df["seed"] = int(meta["seed"])
            df["run"] = f.name.replace(".csv.gz", "")
            df["status"] = meta.get("status", "?")
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _norm(df: pd.DataFrame, space: ObjectiveSpace) -> np.ndarray:
    return space.normalize(df["thermal_resistance"].to_numpy(float), df["pump_power"].to_numpy(float),
                           df["mass"].to_numpy(float))


def hv_curves(df: pd.DataFrame, checkpoints: Sequence[int], space: ObjectiveSpace | None = None) -> pd.DataFrame:
    """Hypervolume (normalized space, ref = 1) of the feasible front after k expensive evaluations, per run."""
    space = space or ObjectiveSpace()
    rows = []
    for run, g in df.groupby("run"):
        e = g[g["expensive"].astype(bool)].sort_values("eval_id")
        F = _norm(e, space)
        ok = e["feasible"].astype(bool).to_numpy()
        n = len(e)
        for k in checkpoints:
            if k > n:
                break
            sel = ok[:k]
            rows.append({"run": run, "method": g["method"].iloc[0], "seed": int(g["seed"].iloc[0]), "n_evals": int(k),
                         "hv": hypervolume(F[:k][sel], space.ref) if sel.any() else 0.0})
    return pd.DataFrame(rows)


def combined_front(df: pd.DataFrame, space: ObjectiveSpace | None = None) -> pd.DataFrame:
    """Feasible non-dominated rows over all given runs (in (R, P, m))."""
    space = space or ObjectiveSpace()
    f = df[df["feasible"].astype(bool)].copy()
    F = f[["thermal_resistance", "pump_power", "mass"]].to_numpy(float)
    keep = pareto_mask(F)
    out = f[keep].copy()
    # drop exact duplicates (NSGA-II survivors evaluated once, but repeated seeds may coincide)
    out = out.drop_duplicates(subset=["thermal_resistance", "pump_power", "mass"])
    Fn = _norm(out, space)
    for k, name in enumerate(("f1_R", "f2_logP", "f3_mass")):
        out[name] = Fn[:, k]
    return out.sort_values("thermal_resistance").reset_index(drop=True)


def representative_designs(front: pd.DataFrame, space: ObjectiveSpace | None = None) -> dict[str, int]:
    """Row labels of the min-R, min-mass and knee designs of a front.

    Knee = closest point to the ideal point after normalizing each objective (R, log10 P, mass) to
    [0, 1] over the front itself (compromise solution; Euclidean distance), excluding the min-R and
    min-mass designs themselves when the front has other points (so three distinct designs are shown).
    """
    space = space or ObjectiveSpace()
    F = _norm(front, space)
    span = np.ptp(F, axis=0)
    span[span == 0] = 1.0
    G = (F - F.min(axis=0)) / span
    i_r = int(front["thermal_resistance"].to_numpy().argmin())
    i_m = int(front["mass"].to_numpy().argmin())
    d = np.linalg.norm(G, axis=1)
    if len(front) > 2:
        d[[i_r, i_m]] = np.inf
    return {"min_R": i_r, "min_mass": i_m, "knee": int(np.argmin(d))}


def equal_pump_power(front: pd.DataFrame, baselines: pd.DataFrame, caps: Sequence[float],
                     families: Sequence[str]) -> pd.DataFrame:
    """Best R of the optimized front at P <= cap vs each baseline family's best R at the same cap.

    ``baselines``: rows of feasible baseline designs with columns family, thermal_resistance, R_active,
    pump_power, mass (e.g. Task 10 ``results/task10_fronts.csv``, the non-dominated set of every family's
    candidate table - it contains the min-R design under any pump-power cap). Also reports the best
    optimized R with mass <= that baseline design's mass (equal pump power AND mass).
    """
    rows = []
    for cap in caps:
        f = front[front["pump_power"] <= cap * (1 + 1e-9)]
        best = f.loc[f["thermal_resistance"].idxmin()] if len(f) else None
        for fam in families:
            b = baselines[(baselines["family"] == fam) & (baselines["pump_power"] <= cap * (1 + 1e-9))]
            if b.empty:
                continue
            b = b.loc[b["thermal_resistance"].idxmin()]
            row = {"pump_power_cap": cap, "family": fam, "R_baseline": b["thermal_resistance"],
                   "R_active_baseline": b.get("R_active", np.nan), "P_baseline": b["pump_power"], "mass_baseline": b["mass"]}
            if best is not None:
                row.update(R_opt=best["thermal_resistance"], R_active_opt=best.get("R_active", np.nan),
                           P_opt=best["pump_power"], mass_opt=best["mass"], run_opt=best.get("run", ""),
                           improvement_pct=100 * (1 - best["thermal_resistance"] / b["thermal_resistance"]))
                fm = f[f["mass"] <= b["mass"] * (1 + 1e-9)]
                if len(fm):
                    bm = fm.loc[fm["thermal_resistance"].idxmin()]
                    row.update(R_opt_mass_le=bm["thermal_resistance"], P_opt_mass_le=bm["pump_power"],
                               mass_opt_mass_le=bm["mass"],
                               improvement_pct_mass_le=100 * (1 - bm["thermal_resistance"] / b["thermal_resistance"]))
            rows.append(row)
    return pd.DataFrame(rows)


def rescreen_cell_size(df: pd.DataFrame, layout: str, *, margin: float = 0.01, fine: bool = True,
                       valley_wall: float | None = 0.09, chunk: int = 256) -> pd.DataFrame:
    """Re-check L >= (1 + margin) L_min(w, rho*) in every lattice cell of every evaluated design.

    The optimization screened cells with the coarse Task 1 table (w step 0.125). Blends have pinch-neck spikes
    of L_min between those w nodes (Task 1 open issues 1-2), which the optimizer can exploit; ``fine=True``
    uses the 5x finer w table, and ``margin`` covers the remaining interpolation error between table nodes
    (a direct n = 48 check at L = L_min_table found throats 0.1 % below the limit).
    ``valley_wall``: the minimum wall of blends vanishes on narrow curves ("valleys", wall_min/L ~ 1 voxel in the
    fine table) that no interpolation resolves, and a graded field crossing one has a near-zero wall somewhere; a
    design is rejected if any lattice cell lies in a fine-table quad with a corner below ``valley_wall`` (0.09 =
    ~4 voxels at n = 48). Adds ``cell_ratio_fine`` (min L / L_min_fine), ``valley_cells`` (count), keeps the
    original flag as ``feasible_coarse`` and narrows ``feasible``."""
    from voxlat.optimize.parameterization import FastMinCellSize

    out = df.copy()
    out["feasible_coarse"] = out["feasible"].astype(bool)
    ratio = np.full(len(out), np.nan)
    valley = np.zeros(len(out), dtype=np.int64)
    xcols = sorted((c for c in out.columns if c.startswith("x") and c[1:].isdigit()), key=lambda c: int(c[1:]))
    sel = np.flatnonzero(out["expensive"].astype(bool).to_numpy() & out[xcols].notna().all(axis=1).to_numpy())
    problem = JacketProblem(ProblemSettings(layout=layout))
    par = problem.par
    lmin = FastMinCellSize(problem.model.cfg, fine=fine)
    lat = par._lat_flat
    X = out[xcols].to_numpy(float)
    for a0 in range(0, len(sel), chunk):
        idx = sel[a0:a0 + chunk]
        F = par.fields_batch(X[idx], ("rho", "w", "L"))
        wc, rc = np.clip(F["w"][:, lat], 0.0, 1.0), np.clip(F["rho"][:, lat], 0.15, 0.55)
        Lm = lmin(wc, rc)
        ratio[idx] = np.min(F["L"][:, lat] / Lm, axis=1)
        if valley_wall is not None:
            valley[idx] = np.sum(lmin.quad_min_wall(wc, rc) < valley_wall, axis=1)
    out["cell_ratio_fine"] = ratio
    out["valley_cells"] = valley
    out["feasible"] = out["feasible_coarse"] & (np.nan_to_num(ratio, nan=0.0) >= 1.0 + margin) & (valley == 0)
    return out


def x_of(row: Mapping[str, Any], n_var: int) -> np.ndarray:
    return np.array([float(row[f"x{i:02d}"]) for i in range(n_var)])


def evaluate_x(layout: str, x: np.ndarray, *, flow_rate: float | None = None, return_fields: bool = True,
               spec: Any = None) -> tuple[dict[str, Any], dict[str, np.ndarray], JacketProblem]:
    """Re-evaluate a stored design vector (fields + full model output)."""
    settings = ProblemSettings(layout=layout, spec=spec) if spec is not None else ProblemSettings(layout=layout)
    problem = JacketProblem(settings)
    x = problem.par.repair(x)
    F = problem.par.fields(x)
    out = evaluate_design_fields(problem.model, F, flow_rate, return_fields=return_fields)
    return out, F, problem


def design_record(layout: str, label: str, x: np.ndarray, *, source: Mapping[str, Any] | None = None,
                  flow_rates: Sequence[float] | None = None, manufacturability_check: Mapping[str, Any] | None = None
                  ) -> dict[str, Any]:
    """JSON-ready design: control nets (SI and unit), basis description, metrics (3 L/min + sweep)."""
    out, F, problem = evaluate_x(layout, x, return_fields=False)
    par = problem.par
    nets = par.control_nets(x)
    keys = ("thermal_resistance", "R_active", "T_wall_max", "T_out", "delta_p", "pump_power", "mass", "mass_lattice",
            "min_structural_margin", "lattice_margin", "outer_wall_margin", "manufacturable", "energy_balance_error",
            "mass_balance_error")
    metrics = {k: out[k] for k in keys}
    metrics["manufacturability"] = out["manufacturability"]
    metrics["diagnostics"] = {k: out["diagnostics"][k] for k in ("reynolds_median", "U_eff_mean", "fin_efficiency_mean",
                                                                  "forchheimer_number_range", "heat_to_manifold_plenums")}
    lat = problem.model.grid.lattice
    stats = {n: {"min": float(F[n][lat].min()), "max": float(F[n][lat].max()),
                 "area_mean": float(np.average(F[n][lat], weights=problem.model.grid.area[lat]))} for n in ("rho", "w", "L")}
    sweep = []
    for Q in (flow_rates or problem.model.cfg.operating.flow_rate_sweep):
        o = evaluate_design_fields(problem.model, F, Q)
        sweep.append({"flow_rate_m3s": Q, "flow_lpm": Q * 6e4, "thermal_resistance": o["thermal_resistance"],
                      "R_active": o["R_active"], "pump_power": o["pump_power"], "delta_p": o["delta_p"],
                      "min_structural_margin": o["min_structural_margin"]})
    rec = {
        "label": label, "layout": layout, "type": "graded_tpms_jacket", "voxlat_task": 11,
        "units": "SI (m, kg, K, W, Pa); rho = relative density, w = blend weight (0 gyroid, 1 diamond), L = cell size",
        "basis": par.describe(),
        "control_nets": {n: v.tolist() for n, v in nets.items()},
        "control_nets_unit": {n: v.tolist() for n, v in par.unit_nets(x).items()},
        "fixed_fields": dict(par.spec.fixed),
        "x_unit": [float(v) for v in x],
        "field_stats": stats,
        "metrics_nominal_flow": metrics,
        "flow_sweep": sweep,
        "source": dict(source or {}),
        "manufacturability_check": dict(manufacturability_check or {}),
        "model": {"manifold_layout": layout, "grid": list(problem.model.grid.shape),
                  "options": to_jsonable(problem.model.options)},
        "run_record": run_record({"layout": layout, "label": label}),
    }
    return to_jsonable(rec)
