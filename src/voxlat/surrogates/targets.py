"""Surrogate targets: closure table <-> latent (log-space) vectors <-> physical quantities.

Latent targets (40 per sample, all smooth functions of theta = (w, rho*, a_z)):

=======  ====  =====================================================================
group    size  definition
=======  ====  =====================================================================
``K``    6     matrix log of K / L^2 (plain entries xx, yy, zz, yz, xz, xy);
               its eigenvalues are the **log principal values** of K, its
               eigenvectors the principal axes ([111] for blends, Task 4)
``keff`` 6     matrix log of k_eff / k_s
``C``    21    matrix log of the Mandel stiffness C / E_s (upper triangle i <= j)
``asf``  1     log(a_sf L)
``loc``  6     log of the p99 von Mises localization factor of the six unit
               macroscopic stresses (Task 3: p99 is resolution-stable; the max is a
               voxel-corner singularity and is not modelled)
=======  ====  =====================================================================

Physics built in: every positive quantity is learned as a logarithm (tensors via
the Log-Euclidean map, so any prediction is symmetric positive definite); the
symmetry class of theta (``symmetry``) is imposed by projecting the latent tensor
logs and the load-case vector onto the invariant subspace, both on the training
data and on every prediction; L enters analytically (K ~ L^2, a_sf ~ 1/L; k_eff,
C and localization are scale-free in the Stokes / linear-elastic / conduction
cell problems).

Uncertainty: the models return independent Gaussian latent std's; standard
deviations of derived quantities use the first-order delta method through the
exact derivative of the matrix exponential (Daleckii-Krein, ``tensors``) and of
the eigenvalues (dl_i = v_i^T dX v_i). Positive scalars are log-normal; their
"mean" here is the median exp(mu) (differs from the log-normal mean by
sigma^2/2, < 0.5 % for sigma < 0.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from voxlat.surrogates.symmetry import (
    LOAD_CASES,
    projector_cases,
    projector_sym3,
    projector_sym6,
    symmetry_class,
)
from voxlat.surrogates.tensors import (
    MANDEL_W,
    TRIU6,
    VOIGT_PAIRS,
    expm_frechet_sym,
    mandel_to_voigt,
    sym3_basis,
    sym3_to_vec6,
    sym6_basis,
    sym6_to_vec21,
    sym_eigh,
    sym_logm,
    vec6_to_sym3,
    vec21_to_sym6,
    voigt_to_mandel,
)

__all__ = [
    "COMP3",
    "LATENT_GROUPS",
    "LATENT_NAMES",
    "GROUP_SLICES",
    "DesignBox",
    "theta_array",
    "design_features",
    "latent_from_table",
    "project_latent",
    "derived_quantities",
    "QUANTITY_INFO",
    "SCALAR_TARGETS",
    "scalar_targets_from_table",
]

COMP3: tuple[str, ...] = ("xx", "yy", "zz", "yz", "xz", "xy")
_TRIU_NAMES = tuple(f"{i + 1}{j + 1}" for i, j in TRIU6)

LATENT_GROUPS: dict[str, tuple[str, ...]] = {
    "K": tuple(f"logK_{c}" for c in COMP3),
    "keff": tuple(f"logk_{c}" for c in COMP3),
    "C": tuple(f"logCm_{ij}" for ij in _TRIU_NAMES),
    "asf": ("log_a_sf_L",),
    "loc": tuple(f"log_loc_{c}_p99" for c in LOAD_CASES),
}
LATENT_NAMES: tuple[str, ...] = tuple(n for g in LATENT_GROUPS.values() for n in g)
GROUP_SLICES: dict[str, slice] = {}
_s = 0
for _g, _names in LATENT_GROUPS.items():
    GROUP_SLICES[_g] = slice(_s, _s + len(_names))
    _s += len(_names)
del _s, _g, _names

#: derived (evaluation / API) quantities: name -> (group, positive?, description)
QUANTITY_INFO: dict[str, tuple[str, bool, str]] = {}
for _i in (1, 2, 3):
    QUANTITY_INFO[f"K_eig{_i}"] = ("K", True, f"principal value {_i} (ascending) of K/L^2")
QUANTITY_INFO["K_mean"] = ("K", True, "tr(K)/3 / L^2")
for _c in COMP3:
    QUANTITY_INFO[f"K_{_c}"] = ("K", _c[0] == _c[1], f"K_{_c} / L^2")
for _c in COMP3:
    QUANTITY_INFO[f"keff_{_c}"] = ("keff", _c[0] == _c[1], f"k_eff,{_c} / k_s")
QUANTITY_INFO["keff_mean"] = ("keff", True, "tr(k_eff)/3 / k_s")
for _ij in _TRIU_NAMES:
    QUANTITY_INFO[f"C{_ij}"] = ("C", _ij[0] == _ij[1], f"Voigt C_{_ij} / E_s")
for _c in ("x", "y", "z"):
    QUANTITY_INFO[f"E_{_c}"] = ("C", True, f"Young's modulus E_{_c} / E_s")
QUANTITY_INFO["E_axial"] = ("C", True, "E* = (E_x + E_y + E_z)/3 / E_s")
QUANTITY_INFO["a_sf_L"] = ("asf", True, "specific surface a_sf L")
for _c in LOAD_CASES:
    QUANTITY_INFO[f"loc_{_c}_p99"] = ("loc", True, f"p99 von Mises localization, {_c}")
del _i, _c, _ij

#: scalar targets of the CNN comparison (geometry vs parameter learning)
SCALAR_TARGETS: tuple[str, ...] = ("log_K_mean", "log_E_axial")


# =============================================================================
# Inputs
# =============================================================================
@dataclass(frozen=True)
class DesignBox:
    """Bounds used to scale theta to [0, 1]^3 (log scale for rho* and a_z).

    Defaults: the Task 5 dataset box; a_z bounds are the outermost representable
    levels 22/32 ... 48/32 (``voxlat.closures.dataset.stretch_levels``).
    """

    w: tuple[float, float] = (0.0, 1.0)
    rho: tuple[float, float] = (0.20, 0.50)
    a_z: tuple[float, float] = (0.6875, 1.5)

    @classmethod
    def from_table(cls, df: Any) -> "DesignBox":
        return cls(w=(float(df["w"].min()), float(df["w"].max())),
                   rho=(float(df["rho_target"].min()), float(df["rho_target"].max())),
                   a_z=(float(df["a_z"].min()), float(df["a_z"].max())))

    def to_dict(self) -> dict[str, list[float]]:
        return {"w": list(self.w), "rho": list(self.rho), "a_z": list(self.a_z)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Sequence[float]]) -> "DesignBox":
        return cls(w=tuple(d["w"]), rho=tuple(d["rho"]), a_z=tuple(d["a_z"]))  # type: ignore[arg-type]

    def contains(self, theta: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        t = np.asarray(theta, dtype=float)
        ok = np.ones(t.shape[0], dtype=bool)
        for k, (lo, hi) in enumerate((self.w, self.rho, self.a_z)):
            ok &= (t[:, k] >= lo - tol) & (t[:, k] <= hi + tol)
        return ok


def theta_array(theta: Any) -> tuple[np.ndarray, bool]:
    """Coerce theta to a float (N, 3) array of (w, rho*, a_z); returns (array, was_single).

    Accepts an (N, 3) / (3,) array-like, a mapping with keys w, rho (or rho_target),
    a_z (arrays broadcast), an object with attributes w, rho, a (``TPMSParams``), or
    a sequence of such objects.
    """
    if hasattr(theta, "w") and hasattr(theta, "rho"):
        a = getattr(theta, "a", (1.0, 1.0, getattr(theta, "a_z", 1.0)))
        if abs(a[0] - 1.0) > 1e-12 or abs(a[1] - 1.0) > 1e-12:
            raise ValueError("surrogates cover a_x = a_y = 1 cells only (stretch along z)")
        return np.array([[float(theta.w), float(theta.rho), float(a[2])]]), True
    if isinstance(theta, Mapping):
        rho = theta["rho"] if "rho" in theta else theta["rho_target"]
        w, r, a = np.broadcast_arrays(np.asarray(theta["w"], float), np.asarray(rho, float),
                                      np.asarray(theta.get("a_z", 1.0), float))
        single = w.ndim == 0
        return np.stack([w.ravel(), r.ravel(), a.ravel()], axis=1), single
    if isinstance(theta, (list, tuple)) and theta and hasattr(theta[0], "w"):
        return np.vstack([theta_array(t)[0] for t in theta]), False
    arr = np.asarray(theta, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError("theta must be (w, rho*, a_z)")
        return arr[None, :], True
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"theta must have shape (N, 3), got {arr.shape}")
    return arr, False


def design_features(theta: np.ndarray, box: DesignBox) -> np.ndarray:
    """(N, 3) model inputs in [0, 1]: w, log rho* and log a_z scaled to the box."""
    t = np.asarray(theta, dtype=float)
    u = np.empty_like(t)
    u[:, 0] = (t[:, 0] - box.w[0]) / (box.w[1] - box.w[0])
    u[:, 1] = (np.log(t[:, 1]) - np.log(box.rho[0])) / (np.log(box.rho[1]) - np.log(box.rho[0]))
    u[:, 2] = (np.log(t[:, 2]) - np.log(box.a_z[0])) / (np.log(box.a_z[1]) - np.log(box.a_z[0]))
    return u


# =============================================================================
# Table -> latent
# =============================================================================
def _tensor3(df: Any, prefix: str) -> np.ndarray:
    return vec6_to_sym3(np.column_stack([df[f"{prefix}_{c}"].to_numpy(float) for c in COMP3]))


def _stiffness_voigt(df: Any) -> np.ndarray:
    v = np.column_stack([df[f"C{ij}"].to_numpy(float) for ij in _TRIU_NAMES])
    return vec21_to_sym6(v)


def latent_from_table(df: Any, *, project: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Latent targets (N, 40) and symmetry classes (N,) of a closure table (Task 5 columns)."""
    classes = symmetry_class(df["w"].to_numpy(float), df["a_z"].to_numpy(float))
    Y = np.empty((len(df), len(LATENT_NAMES)))
    Y[:, GROUP_SLICES["K"]] = sym3_to_vec6(sym_logm(_tensor3(df, "K")))
    Y[:, GROUP_SLICES["keff"]] = sym3_to_vec6(sym_logm(_tensor3(df, "keff")))
    Y[:, GROUP_SLICES["C"]] = sym6_to_vec21(sym_logm(voigt_to_mandel(_stiffness_voigt(df))))
    Y[:, GROUP_SLICES["asf"]] = np.log(df["a_sf_L"].to_numpy(float))[:, None]
    Y[:, GROUP_SLICES["loc"]] = np.log(np.column_stack([df[f"loc_{c}_p99"].to_numpy(float) for c in LOAD_CASES]))
    if project:
        Y = project_latent(Y, classes)
    return Y, classes


def project_latent(Y: np.ndarray, classes: Sequence[str] | np.ndarray) -> np.ndarray:
    """Impose the symmetry class of every row on a latent matrix (N, 40)."""
    Y = np.array(Y, dtype=float, copy=True)
    classes = np.asarray(classes)
    for name in np.unique(classes):
        m = classes == name
        n = str(name)
        for g, fn in (("K", projector_sym3), ("keff", projector_sym3), ("C", projector_sym6), ("loc", projector_cases)):
            sl = GROUP_SLICES[g]
            Y[np.ix_(m, np.arange(sl.start, sl.stop))] = Y[m, sl] @ fn(n).T
    return Y


def scalar_targets_from_table(df: Any) -> np.ndarray:
    """(N, 2) log K* = log(tr K / 3 / L^2) and log E* = log(E_axial / E_s) (CNN comparison)."""
    return np.column_stack([np.log(df["K_mean"].to_numpy(float)), np.log(df["E_axial"].to_numpy(float))])


# =============================================================================
# Latent -> derived quantities with delta-method std
# =============================================================================
def _projected_basis(kind: str, name: str) -> np.ndarray:
    """B'_k = P(B_k): d X / d z_k for the projected latent (kind "sym3" or "sym6")."""
    if kind == "sym3":
        return np.einsum("jk,jab->kab", projector_sym3(name), sym3_basis())
    return np.einsum("jk,jab->kab", projector_sym6(name), sym6_basis())


def _sym3_quantities(mu: np.ndarray, sd: np.ndarray | None, name: str, prefix: str, with_eigs: bool,
                     out: dict[str, list[Any]], idx: np.ndarray) -> None:
    P = projector_sym3(name)
    X = vec6_to_sym3(mu @ P.T)
    w, V = sym_eigh(X)
    T = np.einsum("nij,nj,nkj->nik", V, np.exp(w), V)
    comp = sym3_to_vec6(T)
    tr = np.trace(T, axis1=1, axis2=2) / 3.0
    lam = np.exp(w)
    zero = np.zeros(len(idx))
    if sd is None:  # mean only: skip the derivatives
        for c, cname in enumerate(COMP3):
            out[f"{prefix}_{cname}"].append((idx, comp[:, c], zero))
        out[f"{prefix}_mean"].append((idx, tr, zero))
        if with_eigs:
            for i in range(3):
                out[f"{prefix}_eig{i + 1}"].append((idx, lam[:, i], zero))
        return
    Bp = _projected_basis("sym3", name)  # (6, 3, 3)
    dT = expm_frechet_sym(X, Bp, directions=True, eig=(w, V))  # (n, 6, 3, 3)
    var2 = sd**2  # (n, 6)
    dcomp = sym3_to_vec6(dT)  # (n, 6latent, 6comp)
    comp_sd = np.sqrt(np.einsum("nkc,nk->nc", dcomp**2, var2))
    for c, cname in enumerate(COMP3):
        out[f"{prefix}_{cname}"].append((idx, comp[:, c], comp_sd[:, c]))
    dtr = np.trace(dT, axis1=2, axis2=3) / 3.0  # (n, 6)
    out[f"{prefix}_mean"].append((idx, tr, np.sqrt(np.einsum("nk,nk->n", dtr**2, var2))))
    if with_eigs:
        dl = np.einsum("nai,kab,nbi->nki", V, Bp, V)  # d l_i / d z_k
        lam_sd = lam * np.sqrt(np.einsum("nki,nk->ni", dl**2, var2))
        for i in range(3):
            out[f"{prefix}_eig{i + 1}"].append((idx, lam[:, i], lam_sd[:, i]))


def _stiffness_quantities(mu: np.ndarray, sd: np.ndarray | None, name: str, out: dict[str, list[Any]],
                          idx: np.ndarray) -> None:
    P = projector_sym6(name)
    X = vec21_to_sym6(mu @ P.T)
    w, V = sym_eigh(X)
    Cm = np.einsum("nij,nj,nkj->nik", V, np.exp(w), V)
    Wm = np.outer(MANDEL_W, MANDEL_W)
    Cv = Cm / Wm
    # compliance S_M = exp(-X): same eigenvectors, eigenvalues -w
    Sm = np.einsum("nij,nj,nkj->nik", V, np.exp(-w), V)
    E = 1.0 / np.stack([Sm[:, d, d] for d in range(3)], axis=1)  # Mandel = Voigt for normal terms
    if sd is None:  # mean only
        zero = np.zeros(len(idx))
        for (i, j), ij in zip(TRIU6, _TRIU_NAMES):
            out[f"C{ij}"].append((idx, Cv[:, i, j], zero))
        for d, c in enumerate("xyz"):
            out[f"E_{c}"].append((idx, E[:, d], zero))
        out["E_axial"].append((idx, E.mean(axis=1), zero))
        return
    Bp = _projected_basis("sym6", name)  # (21, 6, 6)
    var2 = sd**2
    dCm = expm_frechet_sym(X, Bp, directions=True, eig=(w, V))  # (n, 21, 6, 6)
    dCv = dCm / Wm
    for (i, j), ij in zip(TRIU6, _TRIU_NAMES):
        out[f"C{ij}"].append((idx, Cv[:, i, j], np.sqrt(np.einsum("nk,nk->n", dCv[:, :, i, j] ** 2, var2))))
    dSm = -expm_frechet_sym(-X, Bp, directions=True, eig=(-w, V))
    dE = -(E[:, None, :] ** 2) * np.stack([dSm[:, :, d, d] for d in range(3)], axis=2)  # (n, 21, 3)
    for d, c in enumerate("xyz"):
        out[f"E_{c}"].append((idx, E[:, d], np.sqrt(np.einsum("nk,nk->n", dE[:, :, d] ** 2, var2))))
    Ea = E.mean(axis=1)
    dEa = dE.mean(axis=2)
    out["E_axial"].append((idx, Ea, np.sqrt(np.einsum("nk,nk->n", dEa**2, var2))))


def derived_quantities(
    mu: np.ndarray,
    sd: np.ndarray | None,
    classes: Sequence[str] | np.ndarray,
    groups: Sequence[str] = ("K", "keff", "C", "asf", "loc"),
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Dimensionless derived quantities {name: (value, std)} from latent mean/std (N, 40).

    The symmetry projection of each row's class is applied to ``mu`` (and to the
    sensitivities, so ``sd`` of non-invariant latent directions has no effect).
    ``sd=None``: values only (std's returned as zeros, derivatives skipped - fast path).
    Names: see ``QUANTITY_INFO`` (K_eig1..3, K_mean, K_xx.., keff_xx.., keff_mean,
    C11..C66 (Voigt / E_s), E_x, E_y, E_z, E_axial, a_sf_L, loc_<case>_p99).
    """
    mu = np.atleast_2d(np.asarray(mu, dtype=float))
    sd = None if sd is None else np.atleast_2d(np.asarray(sd, dtype=float))
    classes = np.asarray(classes)
    N = mu.shape[0]
    want = [q for q, (g, _, _) in QUANTITY_INFO.items() if g in groups]
    parts: dict[str, list[Any]] = {q: [] for q in want}
    for name in np.unique(classes):
        idx = np.flatnonzero(classes == name)
        n = str(name)
        m = mu[idx]
        s = sd[idx] if sd is not None else None

        def cols(sl: slice) -> np.ndarray | None:
            return None if s is None else s[:, sl]

        if "K" in groups:
            sl = GROUP_SLICES["K"]
            _sym3_quantities(m[:, sl], cols(sl), n, "K", True, parts, idx)
        if "keff" in groups:
            sl = GROUP_SLICES["keff"]
            _sym3_quantities(m[:, sl], cols(sl), n, "keff", False, parts, idx)
        if "C" in groups:
            sl = GROUP_SLICES["C"]
            _stiffness_quantities(m[:, sl], cols(sl), n, parts, idx)
        if "asf" in groups:
            sl = GROUP_SLICES["asf"]
            a = np.exp(m[:, sl.start])
            parts["a_sf_L"].append((idx, a, np.zeros(len(idx)) if s is None else a * s[:, sl.start]))
        if "loc" in groups:
            sl = GROUP_SLICES["loc"]
            Pc = projector_cases(n)
            v = np.exp(m[:, sl] @ Pc.T)
            vsd = np.zeros_like(v) if s is None else v * np.sqrt((s[:, sl] ** 2) @ (Pc.T**2))
            for c, case in enumerate(LOAD_CASES):
                parts[f"loc_{case}_p99"].append((idx, v[:, c], vsd[:, c]))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for q in want:
        val = np.empty(N)
        std = np.empty(N)
        for idx, v, s in parts[q]:
            val[idx] = v
            std[idx] = s
        out[q] = (val, std)
    return out


def tensors_from_latent(mu: np.ndarray, classes: Sequence[str] | np.ndarray) -> dict[str, np.ndarray]:
    """Full dimensionless tensors (no std): K/L^2 (N,3,3), k_eff/k_s (N,3,3), Voigt C/E_s (N,6,6)."""
    from voxlat.surrogates.tensors import sym_expm

    Yp = project_latent(np.atleast_2d(mu), classes)
    return {
        "K": sym_expm(vec6_to_sym3(Yp[:, GROUP_SLICES["K"]])),
        "keff": sym_expm(vec6_to_sym3(Yp[:, GROUP_SLICES["keff"]])),
        "C": mandel_to_voigt(sym_expm(vec21_to_sym6(Yp[:, GROUP_SLICES["C"]]))),
    }


__all__ += ["tensors_from_latent"]
_ = VOIGT_PAIRS  # re-exported convention (documentation)
