"""``ClosureModel``: the surrogate API used by the device model (Task 9) and the optimizer (Task 11).

>>> from voxlat.surrogates import ClosureModel
>>> cm = ClosureModel.load("gp")                       # models/closure_gp.npz
>>> out = cm.predict((0.0, 0.35, 1.0), L=4e-3)         # gyroid, rho* = 0.35, a_z = 1, L = 4 mm
>>> K, K_sd = out["K"]                                  # (3, 3) permeability [m^2] and its std
>>> out["E_star"][0] / 70e9                             # E*/E_s ~ 0.10 (STATUS Task 3)

``predict(theta, L)`` returns ``{name: (mean, std)}`` in SI units; theta is
(w, rho*, a_z) - a (3,) tuple, an (N, 3) array, a dict of arrays or ``TPMSParams`` -
and L the in-plane cell size [m] (scalar or (N,)). A single theta returns arrays
without the leading N axis.

======================  ===========  ==========  =============================================
name                    shape (N,…)  unit        notes
======================  ===========  ==========  =============================================
``K``                   3 x 3        m^2         Darcy permeability (creeping flow), SPD
``K_principal``         3            m^2         eigenvalues of K, ascending
``K_star``              -            m^2         tr(K)/3
``k_eff``               3 x 3        W/(m K)     effective conductivity (k_f/k_s = 1/325)
``C_eff``               6 x 6        Pa          Voigt, engineering shear (Task 3 convention)
``E``                   3            Pa          Young's moduli E_x, E_y, E_z (1/S_ii)
``E_star``              -            Pa          (E_x + E_y + E_z)/3
``a_sf``                -            1/m         specific surface area
``loc_<case>_p99``      -            -           p99 von Mises / macro von Mises, six cases
``porosity``            -            -           1 - rho* (exact, std 0)
======================  ===========  ==========  =============================================

Symmetry class of theta (``symmetry.symmetry_class``) is imposed exactly on every
tensor; std's are first-order (delta-method) predictive standard deviations, scaled by
the CV-calibrated multiplier ``std_scale`` when the model was trained with
``recalibrate=True`` (the production models are).
Valid for the Task 5 material constants (k_s, k_f, nu_s; E_s rescales exactly) and
the design box (outside it a warning is logged; ``check_bounds="raise"`` to forbid).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from voxlat.surrogates.symmetry import LOAD_CASES, symmetry_class
from voxlat.surrogates.targets import (
    COMP3,
    GROUP_SLICES,
    LATENT_NAMES,
    SCALAR_TARGETS,
    DesignBox,
    derived_quantities,
    design_features,
    latent_from_table,
    scalar_targets_from_table,
    theta_array,
)
from voxlat.utils.log import get_logger

__all__ = ["BACKENDS", "MODEL_FILES", "ClosureModel", "CNNClosureModel", "default_model_path", "load_training_table"]

LOG = get_logger("surrogates")

BACKENDS = ("gp", "ensemble")
MODEL_FILES = {"gp": "closure_gp.npz", "ensemble": "closure_ensemble.npz", "cnn": "closure_cnn.npz"}
_FORMAT_VERSION = 1


def default_model_path(backend: str) -> Path:
    from voxlat.utils.paths import models_dir

    if backend not in MODEL_FILES:
        raise ValueError(f"unknown backend {backend!r}; expected one of {tuple(MODEL_FILES)}")
    return models_dir() / MODEL_FILES[backend]


def load_training_table(path: str | Path | None = None) -> Any:
    """The Task 5 closure table (status == "ok" rows): ``data/closures.parquet`` by default.

    A ``.csv`` table (the Task 5 no-pyarrow fallback) works too; if the parquet file
    cannot be read because pyarrow is missing, ``data/closures.csv`` is used if present.
    """
    from voxlat.closures.dataset import parquet_available, read_table
    from voxlat.utils.paths import data_dir

    p = Path(path) if path is not None else data_dir() / "closures.parquet"
    if p.suffix == ".parquet" and not parquet_available() and p.with_suffix(".csv").is_file():
        LOG.warning("pyarrow missing: reading %s instead of %s", p.with_suffix(".csv").name, p.name)
        p = p.with_suffix(".csv")
    if not p.is_file():
        raise FileNotFoundError(f"closure table {p} not found (Task 5: scripts/build_dataset.py)")
    return _ok_rows(read_table(p))


def _material(df: Any) -> dict[str, float]:
    out = {}
    for k in ("k_s", "k_f", "E_s", "nu_s"):
        vals = np.unique(np.asarray(df[k], dtype=float))
        if len(vals) != 1:
            raise ValueError(f"closure table mixes several values of {k}: {vals}")
        out[k] = float(vals[0])
    return out


def _provenance(df: Any) -> dict[str, Any]:
    from voxlat import __version__
    from voxlat.utils.provenance import git_hash

    meta: dict[str, Any] = {"voxlat_version": __version__, "git_hash": git_hash(), "n_train": int(len(df)),
                            "trained_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if "settings_hash" in df:
        meta["dataset_settings_hash"] = sorted({str(s) for s in df["settings_hash"]})
    if "id" in df:
        import hashlib

        meta["train_ids_sha1"] = hashlib.sha1(",".join(sorted(map(str, df["id"]))).encode()).hexdigest()[:12]
    return meta


def _ok_rows(df: Any) -> Any:
    if "status" in df:
        bad = df["status"] != "ok"
        if bad.any():
            LOG.warning("dropping %d closure rows with status != ok", int(bad.sum()))
            df = df[~bad]
    return df.reset_index(drop=True)


def _check_bounds(box: DesignBox, th: np.ndarray, mode: str) -> None:
    if mode == "ignore":
        return
    out = ~box.contains(th)
    if out.any():
        msg = f"{int(out.sum())} of {len(th)} theta outside the training box {box.to_dict()} (extrapolation)"
        if mode == "raise":
            raise ValueError(msg)
        LOG.warning(msg)


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, __meta__=np.array(json.dumps(dict(meta), sort_keys=True)), **arrays)
    tmp.replace(path)
    return path


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "__meta__"}
        meta = json.loads(str(z["__meta__"]))
    if meta.get("format_version", 0) > _FORMAT_VERSION:
        raise ValueError(f"{path} was written by a newer voxlat (format {meta['format_version']})")
    return arrays, meta


@dataclass
class ClosureModel:
    """Closure surrogate over theta = (w, rho*, a_z) with analytic L scaling (GP or deep ensemble)."""

    backend: str
    surrogate: Any
    box: DesignBox = field(default_factory=DesignBox)
    material: dict[str, float] = field(default_factory=dict)
    std_scale: np.ndarray | None = None  # optional per-latent std multipliers (recalibration)
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ training
    @classmethod
    def fit(cls, df: Any, backend: str = "gp", *, box: DesignBox | None = None, recalibrate: bool = False,
            cv_folds: int = 5, **kw: Any) -> "ClosureModel":
        """Train on a closure table (``voxlat.closures.read_table``); ``kw`` go to the backend.

        ``recalibrate``: also run a ``cv_folds``-fold CV and store one std multiplier per
        latent target (``evaluation.std_scale_from_oof``) - the production setting
        (``scripts/task6_train_surrogates.py``); costs ``cv_folds`` extra fits.
        """
        from voxlat.surrogates.ensemble import MLPEnsemble
        from voxlat.surrogates.gp import GaussianProcessSurrogate

        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
        df = _ok_rows(df)
        box = box or DesignBox.from_table(df)
        Y, classes = latent_from_table(df)
        X = design_features(df[["w", "rho_target", "a_z"]].to_numpy(float), box)
        if backend == "gp":
            sur: Any = GaussianProcessSurrogate(**kw).fit(X, Y, LATENT_NAMES)
        else:
            sur = MLPEnsemble(**kw).fit(X, Y, LATENT_NAMES)
        meta = _provenance(df)
        meta["backend_settings"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()}
        std_scale = None
        if recalibrate:
            from voxlat.surrogates.evaluation import latent_cv, std_scale_from_oof

            mu, sd, _ = latent_cv(X, Y, backend, k=cv_folds, **kw)
            std_scale = std_scale_from_oof(Y, mu, sd, classes)
            meta["recalibration"] = {"cv_folds": cv_folds, "median": float(np.median(std_scale)),
                                     "max": float(np.max(std_scale)),
                                     "by_group": {g: float(np.median(std_scale[sl])) for g, sl in GROUP_SLICES.items()}}
        return cls(backend=backend, surrogate=sur, box=box, material=_material(df), std_scale=std_scale, meta=meta)

    # ------------------------------------------------------------------ prediction
    def predict_latent(self, theta: Any, return_std: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Latent (log-space, unprojected) mean, std (N, 40) and symmetry classes (N,)."""
        th, _ = theta_array(theta)
        mu, sd = self.surrogate.predict(design_features(th, self.box), return_std=return_std)
        if self.std_scale is not None:
            sd = sd * self.std_scale
        return mu, sd, symmetry_class(th[:, 0], th[:, 2])

    def predict_dimensionless(self, theta: Any, *, check_bounds: str = "warn",
                              return_std: bool = True) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Derived dimensionless quantities (K/L^2, k/k_s, C/E_s, a_sf L, loc) - see ``targets.QUANTITY_INFO``."""
        th, _ = theta_array(theta)
        _check_bounds(self.box, th, check_bounds)
        mu, sd, cl = self.predict_latent(th, return_std=return_std)
        return derived_quantities(mu, sd if return_std else None, cl)

    def predict(self, theta: Any, L: Any, *, check_bounds: str = "warn",
                return_std: bool = True) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Closures in SI units: {name: (mean, std)} (see the module table).

        ``return_std=False`` skips the uncertainty propagation (std's are zeros; ~3x faster).
        """
        th, single = theta_array(theta)
        N = th.shape[0]
        Lm = np.broadcast_to(np.asarray(L, dtype=float), (N,)).astype(float) if np.ndim(L) else np.full(N, float(L))
        if np.any(Lm <= 0):
            raise ValueError("cell size L must be positive [m]")
        q = self.predict_dimensionless(th, check_bounds=check_bounds, return_std=return_std)
        ks, Es = self.material["k_s"], self.material["E_s"]
        L2 = (Lm**2)[:, None, None]

        def tens3(prefix: str) -> tuple[np.ndarray, np.ndarray]:
            from voxlat.surrogates.tensors import vec6_to_sym3

            m = vec6_to_sym3(np.column_stack([q[f"{prefix}_{c}"][0] for c in COMP3]))
            s = vec6_to_sym3(np.column_stack([q[f"{prefix}_{c}"][1] for c in COMP3]))
            return m, s

        def tens6() -> tuple[np.ndarray, np.ndarray]:
            from voxlat.surrogates.tensors import TRIU6, vec21_to_sym6

            names = [f"C{i + 1}{j + 1}" for i, j in TRIU6]
            return (vec21_to_sym6(np.column_stack([q[n][0] for n in names])),
                    vec21_to_sym6(np.column_stack([q[n][1] for n in names])))

        Km, Ks = tens3("K")
        km, kss = tens3("keff")
        Cm, Cs = tens6()
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "K": (Km * L2, Ks * L2),
            "K_principal": (np.column_stack([q[f"K_eig{i}"][0] for i in (1, 2, 3)]) * Lm[:, None] ** 2,
                            np.column_stack([q[f"K_eig{i}"][1] for i in (1, 2, 3)]) * Lm[:, None] ** 2),
            "K_star": (q["K_mean"][0] * Lm**2, q["K_mean"][1] * Lm**2),
            "k_eff": (km * ks, kss * ks),
            "C_eff": (Cm * Es, Cs * Es),
            "E": (np.column_stack([q[f"E_{c}"][0] for c in "xyz"]) * Es, np.column_stack([q[f"E_{c}"][1] for c in "xyz"]) * Es),
            "E_star": (q["E_axial"][0] * Es, q["E_axial"][1] * Es),
            "a_sf": (q["a_sf_L"][0] / Lm, q["a_sf_L"][1] / Lm),
            "porosity": (1.0 - th[:, 1], np.zeros(N)),
        }
        for c in LOAD_CASES:
            out[f"loc_{c}_p99"] = q[f"loc_{c}_p99"]
        if single:
            out = {k: (v[0][0], v[1][0]) for k, v in out.items()}
        return out

    def symmetry_class(self, theta: Any) -> np.ndarray:
        th, _ = theta_array(theta)
        return symmetry_class(th[:, 0], th[:, 2])

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path is not None else default_model_path(self.backend)
        arrays = dict(self.surrogate.to_arrays(prefix="s."))
        if self.std_scale is not None:
            arrays["std_scale"] = np.asarray(self.std_scale, float)
        meta = {**self.meta, "format_version": _FORMAT_VERSION, "kind": "ClosureModel", "backend": self.backend,
                "box": self.box.to_dict(), "material": self.material, "latent_names": list(LATENT_NAMES)}
        return _save_npz(path, arrays, meta)

    @classmethod
    def load(cls, path_or_backend: str | Path = "gp") -> "ClosureModel":
        """Load from a path or by backend name ("gp" / "ensemble" -> models/closure_<backend>.npz)."""
        from voxlat.surrogates.ensemble import MLPEnsemble
        from voxlat.surrogates.gp import GaussianProcessSurrogate

        p = Path(path_or_backend)
        if str(path_or_backend) in MODEL_FILES:
            p = default_model_path(str(path_or_backend))
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found - train it with scripts/task6_train_surrogates.py")
        arrays, meta = _load_npz(p)
        if meta.get("kind") != "ClosureModel":
            raise ValueError(f"{p} is not a ClosureModel file")
        if list(meta["latent_names"]) != list(LATENT_NAMES):
            raise ValueError(f"{p}: latent target set differs from this voxlat version")
        backend = meta["backend"]
        sur = (GaussianProcessSurrogate.from_arrays(arrays, "s.") if backend == "gp"
               else MLPEnsemble.from_arrays(arrays, "s."))
        return cls(backend=backend, surrogate=sur, box=DesignBox.from_dict(meta["box"]), material=dict(meta["material"]),
                   std_scale=arrays.get("std_scale"), meta={k: v for k, v in meta.items()
                                                             if k not in ("box", "material", "latent_names")})


@dataclass
class CNNClosureModel:
    """Geometry-based surrogate (3D CNN ensemble) for K* = tr K / 3 and E* (Task 6c).

    ``predict(theta, L)`` voxelizes each theta at 32^3 (normalized cell) and returns
    {"K_star": m^2, "E_star": Pa}; ``predict_voxels`` takes any periodic (N, 32, 32, 32)
    bool cells (normalized coordinates) plus a_z.
    """

    cnn: Any
    box: DesignBox = field(default_factory=DesignBox)
    material: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(cls, df: Any, *, voxels: np.ndarray | None = None, box: DesignBox | None = None, **kw: Any) -> "CNNClosureModel":
        from voxlat.surrogates.cnn import CNNEnsemble, cnn_voxels

        df = _ok_rows(df)
        box = box or DesignBox.from_table(df)
        th = df[["w", "rho_target", "a_z"]].to_numpy(float)
        vox = cnn_voxels(th) if voxels is None else voxels
        side = CNNEnsemble.side_features(design_features(th, box))
        cnn = CNNEnsemble(**kw).fit(vox, side, scalar_targets_from_table(df), SCALAR_TARGETS)
        meta = _provenance(df)
        meta["backend_settings"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()}
        return cls(cnn=cnn, box=box, material=_material(df), meta=meta)

    def predict_voxels(self, voxels: np.ndarray, a_z: Any, L: Any) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        from voxlat.surrogates.cnn import CNNEnsemble

        vox = np.asarray(voxels, dtype=bool)
        if vox.ndim == 3:
            vox = vox[None]
        N = vox.shape[0]
        a = np.broadcast_to(np.asarray(a_z, dtype=float), (N,))
        th = np.column_stack([np.full(N, self.box.w[0]), np.full(N, self.box.rho[0]), a])  # only a_z is used
        side = CNNEnsemble.side_features(design_features(th, self.box))
        mu, sd = self.cnn.predict(vox, side)
        Lm = np.broadcast_to(np.asarray(L, dtype=float), (N,))
        K = np.exp(mu[:, 0]) * Lm**2
        E = np.exp(mu[:, 1]) * self.material["E_s"]
        return {"K_star": (K, K * sd[:, 0]), "E_star": (E, E * sd[:, 1])}

    def predict(self, theta: Any, L: Any, *, check_bounds: str = "warn") -> dict[str, tuple[np.ndarray, np.ndarray]]:
        from voxlat.surrogates.cnn import cnn_voxels

        th, single = theta_array(theta)
        _check_bounds(self.box, th, check_bounds)
        out = self.predict_voxels(cnn_voxels(th), th[:, 2], L)
        if single:
            out = {k: (v[0][0], v[1][0]) for k, v in out.items()}
        return out

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path is not None else default_model_path("cnn")
        meta = {**self.meta, "format_version": _FORMAT_VERSION, "kind": "CNNClosureModel", "box": self.box.to_dict(),
                "material": self.material, "targets": list(SCALAR_TARGETS)}
        return _save_npz(path, self.cnn.to_arrays(prefix="cnn."), meta)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "CNNClosureModel":
        from voxlat.surrogates.cnn import CNNEnsemble

        p = Path(path) if path is not None else default_model_path("cnn")
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found - train it with scripts/task6_train_surrogates.py --cnn")
        arrays, meta = _load_npz(p)
        if meta.get("kind") != "CNNClosureModel":
            raise ValueError(f"{p} is not a CNNClosureModel file")
        return cls(cnn=CNNEnsemble.from_arrays(arrays, "cnn."), box=DesignBox.from_dict(meta["box"]),
                   material=dict(meta["material"]), meta={k: v for k, v in meta.items() if k not in ("box", "material")})

