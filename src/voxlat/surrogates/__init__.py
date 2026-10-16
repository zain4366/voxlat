"""Closure surrogates (Task 6): GP, deep ensemble and 3D CNN over the TPMS morphology space.

>>> from voxlat.surrogates import ClosureModel
>>> cm = ClosureModel.load("gp")                         # models/closure_gp.npz
>>> out = cm.predict({"w": 0.0, "rho": 0.35, "a_z": 1.0}, L=4e-3)
>>> K_mean, K_std = out["K"]                              # (3, 3) m^2

Modules: ``tensors`` (Log-Euclidean maps), ``symmetry`` (classes and projectors),
``targets`` (table <-> latent <-> derived quantities), ``gp``, ``ensemble``, ``cnn``,
``nn`` (numpy layers), ``model`` (``ClosureModel``, ``CNNClosureModel``),
``evaluation`` (CV, extrapolation split, metrics, plots). See STATUS.md, Task 6.
"""

from voxlat.surrogates.cnn import CNNEnsemble, cnn_voxels
from voxlat.surrogates.ensemble import MLPEnsemble
from voxlat.surrogates.gp import GaussianProcessSurrogate
from voxlat.surrogates.model import BACKENDS, MODEL_FILES, ClosureModel, CNNClosureModel, default_model_path, load_training_table
from voxlat.surrogates.symmetry import (
    LOAD_CASES,
    SYMMETRY_CLASSES,
    independent_components,
    invariant_dimension,
    projector_cases,
    projector_sym3,
    projector_sym6,
    rotation_group,
    symmetry_class,
)
from voxlat.surrogates.targets import (
    LATENT_NAMES,
    QUANTITY_INFO,
    SCALAR_TARGETS,
    DesignBox,
    derived_quantities,
    design_features,
    latent_from_table,
    project_latent,
    scalar_targets_from_table,
    tensors_from_latent,
)

__all__ = [
    "ClosureModel",
    "CNNClosureModel",
    "BACKENDS",
    "MODEL_FILES",
    "default_model_path",
    "load_training_table",
    "GaussianProcessSurrogate",
    "MLPEnsemble",
    "CNNEnsemble",
    "cnn_voxels",
    "DesignBox",
    "LATENT_NAMES",
    "QUANTITY_INFO",
    "SCALAR_TARGETS",
    "latent_from_table",
    "project_latent",
    "derived_quantities",
    "design_features",
    "scalar_targets_from_table",
    "tensors_from_latent",
    "LOAD_CASES",
    "SYMMETRY_CLASSES",
    "symmetry_class",
    "rotation_group",
    "projector_sym3",
    "projector_sym6",
    "projector_cases",
    "invariant_dimension",
    "independent_components",
]
