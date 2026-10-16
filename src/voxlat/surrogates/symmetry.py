"""Material-symmetry classes of the TPMS morphology space and their projectors (Task 6).

Which symmetry a cell has depends only on (w, a_z) (Tasks 1-4, verified on the
Task 5 data):

=============  ==========================  =====================  =========  =========
class          where                       point group (proper)   K, k_eff   C_eff
=============  ==========================  =====================  =========  =========
``cubic``      w in {0, 1}, a_z = 1        432 (24 rotations)     1          3
``tetragonal`` w in {0, 1}, a_z != 1       422 about z (8)        2          6
``trigonal``   0 < w < 1, a_z = 1          3 about [111] (3)      2          7
``triclinic``  0 < w < 1, a_z != 1         1                      6          21
=============  ==========================  =====================  =========  =========

The last two columns are the numbers of independent components of a symmetric
second-rank tensor and of the stiffness (rank of the projectors below). Even-rank
tensors are centrosymmetric, so the proper rotations suffice. The blend keeps
only the 3-fold axis common to gyroid and diamond (cyclic x -> y -> z); a stretch
along z destroys it. The single gyroid labyrinth is chiral (432, not m-3m) and the
stretched cells are 422 / 4/mmm - for these tensors the constraints are identical.

**Projection (Reynolds operator).** The orthogonal projection of a tensor onto the
subspace invariant under a finite group G is the group average
``P(T) = (1/|G|) sum_g g . T``. Applied to the *matrix logarithm* of K, k_eff and
the Mandel stiffness (rotation-covariant, see ``tensors``), it returns a tensor of
the exact symmetry class while keeping SPD. It is used twice: on the training data
(removes voxel symmetry-breaking artifacts - e.g. stretched diamond at
a_z = 22/32: the 4_1 screw axis is not grid-aligned and k_xy reaches 3 % of k_xx)
and on every prediction.

Load-case scalars (stress localization of the six unit macroscopic stresses) are
projected with the permutation the group induces on the cases
(uniaxial_x -> uniaxial_y under the 4-fold axis, etc.).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Sequence

import numpy as np

from voxlat.surrogates.tensors import (
    MANDEL_W,
    mandel_rotation,
    sym3_basis,
    sym3_to_vec6,
    sym6_basis,
    sym6_to_vec21,
)

__all__ = [
    "SYMMETRY_CLASSES",
    "LOAD_CASES",
    "SymmetryClass",
    "rotation_group",
    "symmetry_class",
    "projector_sym3",
    "projector_sym6",
    "projector_cases",
    "invariant_dimension",
    "project",
    "independent_components",
]

SymmetryClass = Literal["cubic", "tetragonal", "trigonal", "triclinic"]
SYMMETRY_CLASSES: tuple[str, ...] = ("cubic", "tetragonal", "trigonal", "triclinic")
#: the six unit macroscopic stresses of Task 3 (same order as the dataset columns)
LOAD_CASES: tuple[str, ...] = ("uniaxial_x", "uniaxial_y", "uniaxial_z", "shear_yz", "shear_xz", "shear_xy")
_CASE_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))

#: tolerance deciding w in {0, 1} and a_z = 1 (design values are exact in the dataset)
CLASS_TOL = 1e-9


def _signed_permutations() -> np.ndarray:
    """All 48 signed 3x3 permutation matrices (the full cube group O_h)."""
    import itertools

    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            R = np.zeros((3, 3))
            for i, (j, s) in enumerate(zip(perm, signs)):
                R[i, j] = s
            out.append(R)
    return np.array(out)


@lru_cache(maxsize=None)
def rotation_group(name: str) -> np.ndarray:
    """(|G|, 3, 3) proper rotations of a symmetry class (see module table)."""
    if name not in SYMMETRY_CLASSES:
        raise ValueError(f"unknown symmetry class {name!r}; expected one of {SYMMETRY_CLASSES}")
    all48 = _signed_permutations()
    proper = all48[np.isclose(np.linalg.det(all48), 1.0)]
    if name == "cubic":
        G = proper  # 432, 24 elements
    elif name == "tetragonal":  # rotations mapping z -> +z or -z (4-fold about z + in-plane 2-folds)
        G = proper[np.isclose(np.abs(proper[:, 2, 2]), 1.0)]
    elif name == "trigonal":  # cyclic permutations x -> y -> z (3-fold about [111])
        P = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        G = np.array([np.eye(3), P, P @ P])
    else:
        G = np.eye(3)[None]
    G = np.array(G)
    G.setflags(write=False)
    return G


def symmetry_class(w: np.ndarray | float, a_z: np.ndarray | float, tol: float = CLASS_TOL) -> np.ndarray:
    """Symmetry class name(s) of morphologies (w, a_z) (vectorized; returns a str array)."""
    w = np.asarray(w, dtype=float)
    a = np.asarray(a_z, dtype=float)
    w, a = np.broadcast_arrays(w, a)
    pure = (w <= tol) | (w >= 1.0 - tol)
    unstretched = np.abs(a - 1.0) <= tol
    out = np.full(w.shape, "triclinic", dtype="<U10")
    out[pure & unstretched] = "cubic"
    out[pure & ~unstretched] = "tetragonal"
    out[~pure & unstretched] = "trigonal"
    return out


@lru_cache(maxsize=None)
def projector_sym3(name: str) -> np.ndarray:
    """6x6 Reynolds projector acting on ``sym3_to_vec6`` vectors (plain entries)."""
    B = sym3_basis()  # (6, 3, 3)
    G = rotation_group(name)
    avg = np.mean(np.einsum("gij,kjl,gml->gkim", G, B, G), axis=0)  # (6, 3, 3) = P(B_k)
    P = sym3_to_vec6(avg).T  # column k = vec(P(B_k))
    P[np.abs(P) < 1e-14] = 0.0
    P.setflags(write=False)
    return P


@lru_cache(maxsize=None)
def projector_sym6(name: str) -> np.ndarray:
    """21x21 Reynolds projector acting on ``sym6_to_vec21`` vectors of *Mandel* matrices."""
    B = sym6_basis()  # (21, 6, 6)
    Qs = np.array([mandel_rotation(R) for R in rotation_group(name)])
    avg = np.mean(np.einsum("gij,kjl,gml->gkim", Qs, B, Qs), axis=0)
    P = sym6_to_vec21(avg).T
    P[np.abs(P) < 1e-14] = 0.0
    P.setflags(write=False)
    return P


@lru_cache(maxsize=None)
def projector_cases(name: str) -> np.ndarray:
    """6x6 averaging operator over the load cases permuted by the group (``LOAD_CASES`` order)."""
    S = np.zeros((6, 3, 3))
    for c, (i, j) in enumerate(_CASE_PAIRS):
        S[c, i, j] = S[c, j, i] = 1.0
    P = np.zeros((6, 6))
    G = rotation_group(name)
    for R in G:
        for c in range(6):
            r = R @ S[c] @ R.T
            hit = [d for d in range(6) if np.allclose(np.abs(r), S[d])]
            if len(hit) != 1:  # pragma: no cover - signed permutations always map cases to cases
                raise RuntimeError("group element does not permute the load cases")
            P[c, hit[0]] += 1.0 / len(G)
    P.setflags(write=False)
    return P


def invariant_dimension(P: np.ndarray) -> int:
    """Number of independent components = rank of the projector (= its trace)."""
    return int(round(float(np.trace(P))))


def project(values: np.ndarray, classes: Sequence[str] | np.ndarray, kind: str) -> np.ndarray:
    """Apply the class projector row-wise. ``kind``: "sym3" (6), "sym6" (21) or "cases" (6)."""
    values = np.asarray(values, dtype=float)
    classes = np.asarray(classes)
    fn = {"sym3": projector_sym3, "sym6": projector_sym6, "cases": projector_cases}[kind]
    out = np.empty_like(values)
    for name in np.unique(classes):
        m = classes == name
        out[m] = values[m] @ fn(str(name)).T
    return out


def independent_components(name: str) -> dict[str, list[str]]:
    """Names of a minimal set of independent components (Voigt C_ij, tensor ij) per class.

    For reporting only: the projector carries the actual constraints (e.g. trigonal
    C66 is not (C11 - C12)/2 here because the 3-fold axis is [111], not z).
    """
    table = {
        "cubic": {"K": ["xx"], "C": ["C11", "C12", "C44"]},
        "tetragonal": {"K": ["xx", "zz"], "C": ["C11", "C33", "C12", "C13", "C44", "C66"]},
        "trigonal": {"K": ["xx", "xy"], "C": ["C11", "C12", "C44", "C14", "C15", "C16", "C45"]},
        "triclinic": {"K": ["xx", "yy", "zz", "yz", "xz", "xy"],
                      "C": [f"C{i + 1}{j + 1}" for i in range(6) for j in range(i, 6)]},
    }
    if name not in table:
        raise ValueError(f"unknown symmetry class {name!r}")
    return table[name]


def _mandel_weights_21() -> np.ndarray:
    """W_i W_j for the 21 upper-triangle entries (Voigt <-> Mandel scaling of vec21)."""
    from voxlat.surrogates.tensors import TRIU6

    return np.array([MANDEL_W[i] * MANDEL_W[j] for i, j in TRIU6])
