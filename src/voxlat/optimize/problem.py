"""The Task 11 optimization problem: graded jacket at the nominal flow rate (3 L/min).

    minimize   ( R [K/W],  pump power [W],  mass [kg] )   of the Task 9 device model
    over       x in [0, 1]^n  (B-spline control nets of rho*, w, L; ``parameterization``)
    subject to |grad rho*| L <= 0.05         (repair - always satisfied)
               L >= L_min(w, rho*)            (g_cell = 1 - min L/L_min <= 0, cheap)
               min structural margin >= 1     (g_margin = 1 - margin <= 0, from the model)

Every optimizer works with the same ``JacketProblem``: designs are repaired, the cheap
constraints are checked, and the device model runs only for cheap-feasible designs (an
"expensive evaluation" - the budget unit of every method and of the hypervolume curves).

Objective space for hypervolume and the algorithms (fixed, identical for every method and
both manifold layouts):

    f1 = R in mK/W,  f2 = log10(pump power / W),  f3 = mass in kg,
    normalized   f~_k = (f_k - lo_k) / (hi_k - lo_k)  with  lo = (5, -3, 0.28), hi = (40, 0.5, 0.42),
    reference point f~ = (1, 1, 1)  (R 40 mK/W, 3.16 W, 0.42 kg; the heaviest possible jacket is 0.41 kg).

The log of the pump power is used because it spans three decades between the end-ring and
slot layouts; dominance is unaffected (monotone), only the hypervolume weighting and the
NSGA-II crowding distance see the transform.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from voxlat.optimize.parameterization import DEFAULT_CONSTRAINTS, GradingParameterization, GradingSpec
from voxlat.utils.log import get_logger

__all__ = [
    "ObjectiveSpace",
    "OBJECTIVES",
    "ProblemSettings",
    "JacketProblem",
    "evaluate_design_fields",
]

LOG = get_logger("optimize.problem")
OBJECTIVES = ("thermal_resistance", "pump_power", "mass")


@dataclass(frozen=True)
class ObjectiveSpace:
    """Fixed transform/normalization of (R, pump power, mass) shared by every method."""

    lo: tuple[float, float, float] = (5.0, -3.0, 0.28)
    hi: tuple[float, float, float] = (40.0, 0.5, 0.42)

    def raw(self, R: Any, P: Any, m: Any) -> np.ndarray:
        R, P, m = np.broadcast_arrays(np.asarray(R, float), np.asarray(P, float), np.asarray(m, float))
        return np.stack([R * 1e3, np.log10(np.maximum(P, 1e-30)), m], axis=-1)

    def normalize(self, R: Any, P: Any, m: Any) -> np.ndarray:
        lo, hi = np.asarray(self.lo), np.asarray(self.hi)
        return (self.raw(R, P, m) - lo) / (hi - lo)

    def denormalize(self, Fn: Any) -> np.ndarray:
        """Normalized -> (R [K/W], pump power [W], mass [kg])."""
        Fn = np.asarray(Fn, float)
        lo, hi = np.asarray(self.lo), np.asarray(self.hi)
        f = lo + Fn * (hi - lo)
        return np.stack([f[..., 0] * 1e-3, 10.0 ** f[..., 1], f[..., 2]], axis=-1)

    @property
    def ref(self) -> np.ndarray:
        return np.ones(3)


@dataclass(frozen=True)
class ProblemSettings:
    layout: str = "slots"  # JacketOptions.manifold_layout
    flow_rate: float | None = None  # None -> config nominal (3 L/min)
    spec: GradingSpec = field(default_factory=GradingSpec)
    model_options: Mapping[str, Any] = field(default_factory=dict)  # extra JacketOptions
    constraints: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_CONSTRAINTS))  # cell-size rule

    def to_dict(self) -> dict[str, Any]:
        return {"layout": self.layout, "flow_rate": self.flow_rate, "spec": self.spec.to_dict(),
                "model_options": dict(self.model_options), "constraints": dict(self.constraints)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ProblemSettings":
        # settings saved before the constraint rule was configurable used the coarse table without margin/valleys
        legacy = {"table": "coarse", "cell_margin": 0.0, "valley_wall": None}
        return cls(layout=d.get("layout", "slots"), flow_rate=d.get("flow_rate"),
                   spec=GradingSpec.from_dict(d.get("spec", {})), model_options=dict(d.get("model_options", {})),
                   constraints=dict(d.get("constraints", legacy)))


# -----------------------------------------------------------------------------
# per-process model cache (joblib workers build their own model once)
# -----------------------------------------------------------------------------
_CACHE: dict[str, tuple[Any, GradingParameterization]] = {}


def _build(settings: ProblemSettings) -> tuple[Any, GradingParameterization]:
    key = repr(settings.to_dict())
    if key not in _CACHE:
        from voxlat.device import JacketModel, JacketOptions

        opts = dict(check_bounds="ignore", manifold_layout=settings.layout)
        opts.update(settings.model_options)
        model = JacketModel(options=JacketOptions(**opts))
        _CACHE[key] = (model, GradingParameterization(model, settings.spec, settings.constraints))
    return _CACHE[key]


RECORD_KEYS = ("thermal_resistance", "R_active", "pump_power", "delta_p", "mass", "min_structural_margin",
               "lattice_margin", "outer_wall_margin", "T_wall_max", "T_wall_max_lattice", "T_out",
               "manufacturable", "picard_converged", "picard_iterations", "mass_balance_error",
               "energy_balance_error")


def evaluate_design_fields(model: Any, F: Mapping[str, np.ndarray], flow_rate: float | None,
                           return_fields: bool = False) -> dict[str, Any]:
    """Run the device model on cell fields; adds ``R_active`` (Task 10 definition)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = model.evaluate(F["rho"], F["w"], F["L"], F["a_z"], flow_rate, return_fields=return_fields)
    out["R_active"] = (out["T_wall_max_lattice"] - out["T_in"]) / model.cfg.motor.heat_to_jacket
    return out


def _evaluate_one(settings: ProblemSettings, x: np.ndarray) -> dict[str, Any]:
    model, par = _build(settings)
    t0 = time.perf_counter()
    x = par.repair(x)
    F = par.fields(x)
    cc = par.cheap_constraints(x, F)
    rec: dict[str, Any] = {"x": x, "g_gradient": cc["gradient"], "g_cell_size": cc["cell_size"],
                           "mass_exact": par.mass(F)}
    if cc["cell_size"] > 0.0 or cc["gradient"] > 1e-9:
        rec.update(expensive=False, g_margin=np.nan, wall_time=time.perf_counter() - t0)
        for k in RECORD_KEYS:
            rec[k] = np.nan
        return rec
    out = evaluate_design_fields(model, F, settings.flow_rate)
    for k in RECORD_KEYS:
        rec[k] = out[k]
    rec["g_margin"] = 1.0 - out["min_structural_margin"]
    rec["gradient_max"] = out["manufacturability"]["gradient_max"]
    rec["pinch_neck_area_fraction"] = out["manufacturability"]["pinch_neck_area_fraction"]
    rec["reynolds_median"] = out["diagnostics"]["reynolds_median"]
    rec["expensive"] = True
    rec["wall_time"] = time.perf_counter() - t0
    return rec


class JacketProblem:
    """Shared problem wrapper: repair, cheap constraints, (parallel) device-model evaluations, archive."""

    def __init__(self, settings: ProblemSettings | None = None, *, n_jobs: int = 1,
                 space: ObjectiveSpace | None = None):
        self.settings = settings or ProblemSettings()
        self.model, self.par = _build(self.settings)
        self.space = space or ObjectiveSpace()
        self.n_jobs = int(n_jobs)
        self.n_var = self.par.n_var
        self.records: list[dict[str, Any]] = []
        self._pool = None

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, X: Any, *, tag: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Evaluate rows of X (unit cube); appends to the archive (in order) and returns the records."""
        X = np.atleast_2d(np.asarray(X, float))
        if self.n_jobs == 1 or len(X) == 1:
            recs = [_evaluate_one(self.settings, x) for x in X]
        else:
            from joblib import Parallel, delayed

            if self._pool is None:
                self._pool = Parallel(n_jobs=self.n_jobs, backend="loky")
            recs = self._pool(delayed(_evaluate_one)(self.settings, x) for x in X)
        n_exp = self.n_expensive
        for r in recs:
            r["proposal_id"] = len(self.records)
            # eval_id counts expensive (device-model) evaluations only - the budget unit; -1 = cheap reject
            r["eval_id"] = n_exp if r["expensive"] else -1
            n_exp += bool(r["expensive"])
            if tag:
                r.update(tag)
            self.records.append(r)
        return recs

    @property
    def n_expensive(self) -> int:
        return sum(1 for r in self.records if r["expensive"])

    def feasible(self, rec: Mapping[str, Any]) -> bool:
        return bool(rec["expensive"] and rec["g_margin"] <= 0.0 and rec["manufacturable"])

    def objectives(self, recs: Sequence[Mapping[str, Any]] | None = None, normalized: bool = True) -> np.ndarray:
        recs = self.records if recs is None else recs
        R = np.array([r["thermal_resistance"] for r in recs], float)
        P = np.array([r["pump_power"] for r in recs], float)
        m = np.array([r["mass"] for r in recs], float)
        return self.space.normalize(R, P, m) if normalized else np.column_stack([R, P, m])

    def table(self) -> pd.DataFrame:
        rows = []
        for r in self.records:
            row = {k: v for k, v in r.items() if k != "x"}
            for i, v in enumerate(r["x"]):
                row[f"x{i:02d}"] = v
            rows.append(row)
        df = pd.DataFrame(rows)
        if len(df):
            df["feasible"] = df["expensive"].astype(bool) & (df["g_margin"] <= 0) & (df["manufacturable"] == True)  # noqa: E712
            Fn = self.space.normalize(df["thermal_resistance"], df["pump_power"], df["mass"])
            for k, name in enumerate(("f1_R", "f2_logP", "f3_mass")):
                df[name] = Fn[:, k]
        return df

    def close(self) -> None:
        self._pool = None
