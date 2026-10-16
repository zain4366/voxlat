"""Tests for voxlat.surrogates (Task 6).

Fast tests use a synthetic closure table with exact symmetry (``_synthetic_table``);
tests on the real Task 5 table and the trained models in models/ skip when those
files (or a parquet engine) are not available. ``VOXLAT_CLOSURE_TABLE`` may point to
a .csv/.parquet copy of the table.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from voxlat.surrogates import nn
from voxlat.surrogates.symmetry import (
    LOAD_CASES,
    SYMMETRY_CLASSES,
    independent_components,
    invariant_dimension,
    project,
    projector_cases,
    projector_sym3,
    projector_sym6,
    rotation_group,
    symmetry_class,
)
from voxlat.surrogates.targets import (
    COMP3,
    GROUP_SLICES,
    LATENT_NAMES,
    DesignBox,
    derived_quantities,
    design_features,
    latent_from_table,
    project_latent,
    theta_array,
)
from voxlat.surrogates.tensors import (
    TRIU6,
    expm_frechet_sym,
    mandel_rotation,
    mandel_to_voigt,
    sym3_to_vec6,
    sym6_to_vec21,
    sym_expm,
    sym_logm,
    vec6_to_sym3,
    vec21_to_sym6,
    voigt_to_mandel,
)

RNG = np.random.default_rng(20261009)


def _spd(n: int, k: int, rng: np.random.Generator = RNG) -> np.ndarray:
    A = rng.normal(size=(k, n, n))
    return A @ np.swapaxes(A, 1, 2) + n * np.eye(n)


# =============================================================================
# Synthetic closure table with exact symmetry
# =============================================================================
def _synthetic_table(n: int = 60, seed: int = 0):
    """A closure table whose latent targets are smooth known functions of theta."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    w = rng.uniform(0, 1, n)
    rho = rng.uniform(0.2, 0.5, n)
    a = 0.6875 + rng.integers(0, 27, n) / 32.0
    # some exact pure / unstretched points
    w[:8] = [0, 0, 1, 1, 0, 1, 0.5, 0.3]
    a[:8] = [1, 1.25, 1, 0.75, 1, 1.5, 1, 0.8]
    a[8:12] = 1.0
    th = np.column_stack([w, rho, a])
    u = design_features(th, DesignBox())
    feats = np.column_stack([np.ones(n), u, np.sin(2 * u[:, :1]), u[:, 1:2] ** 2])
    coef = 0.3 * np.random.default_rng(1).normal(size=(feats.shape[1], len(LATENT_NAMES)))
    Y = feats @ coef
    for g, base in (("K", np.log(5e-3)), ("keff", np.log(0.2))):
        sl = GROUP_SLICES[g]
        Y[:, sl.start:sl.start + 3] += base + 1.5 * np.log(rho / 0.35)[:, None]  # diagonal of the log tensor
        Y[:, sl.start + 3:sl.stop] *= 0.1
    sl = GROUP_SLICES["C"]
    diag_idx = [k for k, (i, j) in enumerate(TRIU6) if i == j]
    Y[:, sl] *= 0.05
    Y[:, [sl.start + k for k in diag_idx]] += np.log(0.05) + 2 * np.log(rho / 0.35)[:, None]
    Y[:, GROUP_SLICES["asf"]] += 1.0
    Y[:, GROUP_SLICES["loc"]] += 2.5
    cl = symmetry_class(w, a)
    Y = project_latent(Y, cl)
    q = derived_quantities(Y, None, cl)
    d = {"id": [f"s{i:03d}" for i in range(n)], "status": "ok", "w": w, "rho_target": rho, "a_z": a,
         "k_s": 130.0, "k_f": 0.4, "E_s": 70e9, "nu_s": 0.33, "settings_hash": "synthetic"}
    for c in COMP3:
        d[f"K_{c}"] = q[f"K_{c}"][0]
        d[f"keff_{c}"] = q[f"keff_{c}"][0]
    for i, j in TRIU6:
        d[f"C{i + 1}{j + 1}"] = q[f"C{i + 1}{j + 1}"][0]
    d["a_sf_L"] = q["a_sf_L"][0]
    for c in LOAD_CASES:
        d[f"loc_{c}_p99"] = q[f"loc_{c}_p99"][0]
    d["K_mean"] = q["K_mean"][0]
    d["E_axial"] = q["E_axial"][0]
    return pd.DataFrame(d), Y


# =============================================================================
# tensors
# =============================================================================
def test_vectorization_roundtrips():
    T = _spd(3, 5)
    assert np.allclose(vec6_to_sym3(sym3_to_vec6(T)), T)
    M = _spd(6, 4)
    assert np.allclose(vec21_to_sym6(sym6_to_vec21(M)), M)


def test_logm_expm_roundtrip_and_spd():
    A = _spd(6, 7)
    assert np.allclose(sym_expm(sym_logm(A)), A, rtol=1e-10, atol=1e-12)
    X = RNG.normal(size=(7, 6, 6))
    X = X + np.swapaxes(X, 1, 2)
    E = sym_expm(X)
    assert np.allclose(E, np.swapaxes(E, 1, 2))
    assert np.all(np.linalg.eigvalsh(E) > 0)
    with pytest.raises(ValueError):
        sym_logm(-np.eye(3))


@pytest.mark.parametrize("degenerate", [False, True])
def test_expm_frechet_matches_finite_differences(degenerate):
    if degenerate:
        X = np.diag([0.3, 0.3, -0.2])[None]  # repeated eigenvalue
    else:
        X = sym_logm(_spd(3, 3))
    E = RNG.normal(size=(X.shape[0], 3, 3))
    E = E + np.swapaxes(E, 1, 2)
    h = 1e-6
    fd = (sym_expm(X + h * E) - sym_expm(X - h * E)) / (2 * h)
    assert np.allclose(expm_frechet_sym(X, E), fd, atol=1e-7)
    # several directions at once
    Es = RNG.normal(size=(4, 3, 3))
    Es = Es + np.swapaxes(Es, 1, 2)
    D = expm_frechet_sym(X, Es, directions=True)
    assert D.shape == (X.shape[0], 4, 3, 3)
    assert np.allclose(D[:, 2], expm_frechet_sym(X, np.broadcast_to(Es[2], X.shape)))


def test_mandel_rotation_matches_elasticity_module():
    from voxlat.homogenization.elasticity import rotate_stiffness, voigt_to_mandel as v2m

    R = np.linalg.qr(RNG.normal(size=(3, 3)))[0]
    Q = mandel_rotation(R)
    assert np.allclose(Q @ Q.T, np.eye(6))
    C = mandel_to_voigt(_spd(6, 1)[0])
    assert np.allclose(voigt_to_mandel(C), v2m(C))
    assert np.allclose(mandel_to_voigt(Q @ voigt_to_mandel(C) @ Q.T), rotate_stiffness(C, R))
    # second-rank tensors: mandel(R a R^T) = Q mandel(a)
    a = _spd(3, 1)[0]
    w6 = np.array([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)])
    assert np.allclose(sym3_to_vec6(R @ a @ R.T) * w6, Q @ (sym3_to_vec6(a) * w6))


# =============================================================================
# symmetry
# =============================================================================
def test_group_sizes_and_closure():
    sizes = {"cubic": 24, "tetragonal": 8, "trigonal": 3, "triclinic": 1}
    for name, n in sizes.items():
        G = rotation_group(name)
        assert len(G) == n
        assert np.allclose(np.linalg.det(G), 1.0)
        for A in G:
            for B in G:
                assert np.any(np.all(np.isclose(G, A @ B), axis=(1, 2))), name


@pytest.mark.parametrize("name,dims", [("cubic", (1, 3, 2)), ("tetragonal", (2, 6, 4)), ("trigonal", (2, 7, 2)),
                                       ("triclinic", (6, 21, 6))])
def test_projectors_are_orthogonal_projections_with_expected_rank(name, dims):
    P3, P6, Pc = projector_sym3(name), projector_sym6(name), projector_cases(name)
    for P in (P3, P6, Pc):
        assert np.allclose(P @ P, P)
    assert np.allclose(P6, P6.T)  # orthonormal (Mandel) basis -> symmetric projector
    assert (invariant_dimension(P3), invariant_dimension(P6), invariant_dimension(Pc)) == dims
    assert len(independent_components(name)["C"]) == dims[1]
    assert len(independent_components(name)["K"]) == dims[0]


def test_projected_tensors_have_the_class_structure():
    T = sym_logm(_spd(3, 1))[0]
    v = sym3_to_vec6(T)[None]
    cub = vec6_to_sym3(project(v, ["cubic"], "sym3"))[0]
    assert np.allclose(cub, np.trace(T) / 3 * np.eye(3))
    tet = vec6_to_sym3(project(v, ["tetragonal"], "sym3"))[0]
    assert np.isclose(tet[0, 0], tet[1, 1]) and np.allclose([tet[0, 1], tet[0, 2], tet[1, 2]], 0)
    tri = vec6_to_sym3(project(v, ["trigonal"], "sym3"))[0]
    assert np.allclose(np.diag(tri), np.diag(tri)[0]) and np.allclose([tri[0, 1], tri[0, 2]], tri[1, 2])
    Cm = sym_logm(_spd(6, 1))[0]
    c = mandel_to_voigt(vec21_to_sym6(project(sym6_to_vec21(Cm)[None], ["cubic"], "sym6"))[0])
    assert np.allclose(np.diag(c)[:3], c[0, 0]) and np.allclose(np.diag(c)[3:], c[3, 3])
    assert np.allclose([c[0, 1], c[0, 2], c[1, 2]], c[0, 1])
    assert np.allclose(c[:3, 3:], 0) and np.allclose(c[3:, 3:] - np.diag(np.diag(c)[3:]), 0)
    # trigonal stiffness: cyclic pattern of 7 constants (Task 5 data shows exactly this)
    t = mandel_to_voigt(vec21_to_sym6(project(sym6_to_vec21(Cm)[None], ["trigonal"], "sym6"))[0])
    for grp in ([(0, 0), (1, 1), (2, 2)], [(0, 1), (0, 2), (1, 2)], [(3, 3), (4, 4), (5, 5)], [(0, 3), (1, 4), (2, 5)],
                [(0, 4), (1, 5), (2, 3)], [(0, 5), (1, 3), (2, 4)], [(3, 4), (3, 5), (4, 5)]):
        vals = [t[i, j] for i, j in grp]
        assert np.allclose(vals, vals[0])


def test_case_projector_maps():
    Pt = projector_cases("tetragonal")
    v = np.arange(6.0)
    out = Pt @ v
    assert np.isclose(out[0], out[1]) and np.isclose(out[3], out[4]) and np.isclose(out[2], 2) and np.isclose(out[5], 5)
    Pc = projector_cases("cubic")
    assert np.allclose(Pc @ v, [1, 1, 1, 4, 4, 4])
    assert np.allclose(projector_cases("trigonal") @ v, [1, 1, 1, 4, 4, 4])


def test_symmetry_class_mapping():
    w = np.array([0.0, 1.0, 0.0, 0.4, 0.4, 1e-12])
    a = np.array([1.0, 1.0, 1.25, 1.0, 0.8, 1.0])
    assert list(symmetry_class(w, a)) == ["cubic", "cubic", "tetragonal", "trigonal", "triclinic", "cubic"]


def test_declared_symmetry_is_physical_for_conductivity():
    """k_eff computed by the Task 2 solver is invariant under the declared group, and blends are not cubic."""
    from voxlat.geometry.tpms import TPMSParams
    from voxlat.homogenization.conduction import effective_conductivity_tpms

    def rel_change(k, name):
        v = sym3_to_vec6(k)[None]
        return np.abs(project(v, [name], "sym3") - v).max() / np.trace(k) * 3

    kg = effective_conductivity_tpms(TPMSParams(w=0.0, rho=0.3), n=20).k_eff
    assert rel_change(kg, "cubic") < 2e-3
    kb = effective_conductivity_tpms(TPMSParams(w=0.5, rho=0.3), n=20).k_eff
    assert rel_change(kb, "trigonal") < 2e-3
    assert rel_change(kb, "cubic") > 2e-2  # uniaxial about [111]
    kt = effective_conductivity_tpms(TPMSParams(w=0.0, rho=0.3, a=(1, 1, 1.25)), n=16).k_eff
    assert rel_change(kt, "tetragonal") < 5e-3


# =============================================================================
# targets
# =============================================================================
def test_theta_array_forms():
    from voxlat.geometry.tpms import TPMSParams

    a, single = theta_array((0.2, 0.3, 1.1))
    assert single and a.shape == (1, 3)
    b, single = theta_array({"w": [0.1, 0.2], "rho": 0.3, "a_z": 1.0})
    assert not single and b.shape == (2, 3) and np.allclose(b[:, 1], 0.3)
    c, single = theta_array(TPMSParams.from_design(0.5, 0.35, 1.25))
    assert single and np.allclose(c, [[0.5, 0.35, 1.25]])
    with pytest.raises(ValueError):
        theta_array(TPMSParams(w=0.0, rho=0.3, a=(1.2, 1.0, 1.0)))


def test_design_features_unit_box():
    box = DesignBox()
    u = design_features(np.array([[0, 0.2, 0.6875], [1, 0.5, 1.5]]), box)
    assert np.allclose(u, [[0, 0, 0], [1, 1, 1]])


def test_latent_roundtrip_reproduces_table():
    df, Y = _synthetic_table()
    Yt, cl = latent_from_table(df)
    assert np.allclose(Yt, Y, atol=1e-10)
    q = derived_quantities(Yt, None, cl)
    for c in COMP3:
        assert np.allclose(q[f"K_{c}"][0], df[f"K_{c}"])
    assert np.allclose(q["E_axial"][0], df["E_axial"])


def test_delta_method_std_matches_monte_carlo():
    df, Y = _synthetic_table(20)
    _, cl = latent_from_table(df)
    idx = np.array([0, 1, 6, 15])  # cubic, tetragonal, trigonal, triclinic
    sd = np.full((len(idx), Y.shape[1]), 0.02)
    q = derived_quantities(Y[idx], sd, cl[idx])
    rng = np.random.default_rng(3)
    draws = [derived_quantities(Y[idx] + sd * rng.normal(size=sd.shape), None, cl[idx]) for _ in range(1500)]
    for name in ("K_xx", "K_mean", "keff_zz", "C11", "C44", "E_z", "E_axial", "a_sf_L", "loc_uniaxial_z_p99"):
        mc = np.std([d[name][0] for d in draws], axis=0)
        assert np.allclose(q[name][1], mc, rtol=0.12), name


# =============================================================================
# numpy NN layers
# =============================================================================
def _grad_check(layer, x, n_probe=12, h=1e-6, tol=1e-6):
    y = layer.forward(x, True)
    g = RNG.normal(size=y.shape)
    dx = layer.backward(g)

    def f(xx):
        return float((layer.forward(xx, False) * g).sum())

    for _ in range(n_probe):
        i = tuple(RNG.integers(0, s) for s in x.shape)
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        assert abs((f(xp) - f(xm)) / (2 * h) - dx[i]) < tol
    for k, p in layer.params.items():
        layer.forward(x, True)
        layer.backward(g)
        G = layer.grads[k].copy()
        for _ in range(6):
            i = tuple(RNG.integers(0, s) for s in p.shape)
            p[i] += h
            a = f(x)
            p[i] -= 2 * h
            b = f(x)
            p[i] += h
            assert abs((a - b) / (2 * h) - G[i]) < tol, k


@pytest.mark.parametrize("make,shape", [
    (lambda: nn.Dense(4, 3, RNG), (5, 4)),
    (lambda: nn.SiLU(), (5, 4)),
    (lambda: nn.Conv3dPeriodic(2, 3, RNG, dtype=np.float64), (2, 4, 6, 4, 2)),
    (lambda: nn.AvgPool3d(), (2, 4, 6, 4, 3)),
    (lambda: nn.SpaceToDepth3d(), (2, 4, 6, 4, 2)),
    (lambda: nn.GlobalAvgPool3d(), (2, 3, 4, 5, 3)),
    (lambda: nn.GlobalAvgMaxPool3d(), (2, 3, 4, 5, 3)),
])
def test_layer_gradients(make, shape):
    _grad_check(make(), RNG.normal(size=shape))


def test_conv_is_translation_equivariant():
    conv = nn.Conv3dPeriodic(2, 3, RNG, dtype=np.float64)
    x = RNG.normal(size=(1, 8, 6, 4, 2))
    s = (3, 1, 2)
    assert np.allclose(np.roll(conv.forward(x), s, axis=(1, 2, 3)), conv.forward(np.roll(x, s, axis=(1, 2, 3))))


def test_gaussian_nll_gradients():
    mu, raw, y = RNG.normal(size=(6, 2)), RNG.normal(size=(6, 2)), RNG.normal(size=(6, 2))
    loss, dmu, draw = nn.gaussian_nll(mu, raw, y)
    h = 1e-6
    for arr, d in ((mu, dmu), (raw, draw)):
        for _ in range(5):
            i = tuple(RNG.integers(0, s) for s in arr.shape)
            arr[i] += h
            a = nn.gaussian_nll(mu, raw, y)[0]
            arr[i] -= 2 * h
            b = nn.gaussian_nll(mu, raw, y)[0]
            arr[i] += h
            assert abs((a - b) / (2 * h) - d[i]) < 1e-6


def test_adam_minimizes_quadratic():
    lay = nn.Dense(3, 1, RNG)
    X = RNG.normal(size=(64, 3))
    y = X @ np.array([[1.0], [-2.0], [0.5]]) + 0.3
    opt = nn.Adam(lay.parameters(), lr=0.05)
    for _ in range(500):
        loss, d = nn.mse_loss(lay.forward(X, True), y)
        lay.backward(d)
        opt.step()
    assert loss < 1e-6


# =============================================================================
# GP, ensemble, CNN
# =============================================================================
def test_gp_matches_sklearn_and_ard():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    from voxlat.surrogates.gp import GaussianProcessSurrogate

    X = RNG.uniform(size=(60, 3))
    y = np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] ** 2  # x2 irrelevant
    gp = GaussianProcessSurrogate(n_restarts=1, linear_mean=False).fit(X, y)
    t = gp.targets[0]
    assert t.length_scale[2] > 5 * max(t.length_scale[0], t.length_scale[1])  # ARD switches off x2
    Xs = RNG.uniform(size=(10, 3))
    mu, sd = gp.predict(Xs)
    # same hyper-parameters in scikit-learn
    k = ConstantKernel(t.signal_var, "fixed") * Matern(t.length_scale, "fixed", nu=2.5) + WhiteKernel(t.noise_var, "fixed")
    ref = GaussianProcessRegressor(k, alpha=t.jitter, optimizer=None).fit(X, (y - t.trend[0]) / t.y_scale)
    m2, s2 = ref.predict(Xs, return_std=True)
    assert np.allclose(mu[:, 0], t.trend[0] + t.y_scale * m2, atol=1e-8)
    assert np.allclose(sd[:, 0], t.y_scale * s2, rtol=1e-6)
    assert np.max(np.abs(mu[:, 0] - (np.sin(3 * Xs[:, 0]) + 0.5 * Xs[:, 1] ** 2))) < 0.05
    gp2 = GaussianProcessSurrogate.from_arrays(gp.to_arrays())
    assert np.allclose(gp2.predict(Xs)[0], mu) and np.allclose(gp2.predict(Xs)[1], sd)


def test_ensemble_fits_and_is_uncertain_away_from_data():
    from voxlat.surrogates.ensemble import MLPEnsemble

    X = RNG.uniform(0.0, 0.5, size=(80, 1))
    y = np.sin(6 * X[:, 0])
    e = MLPEnsemble(n_members=3, hidden=(32, 32), epochs=1500, seed=1).fit(X, y)
    mu, sd = e.predict(np.array([[0.25], [0.95]]))
    assert abs(mu[0, 0] - np.sin(1.5)) < 0.05
    assert sd[1, 0] > 3 * sd[0, 0]
    e2 = MLPEnsemble.from_arrays(e.to_arrays())
    assert np.allclose(e2.predict(X[:5])[0], e.predict(X[:5])[0])


def test_cnn_invariances_and_training():
    from voxlat.surrogates.cnn import CNNEnsemble, _CNNNet, _to_input, apply_transform, d4h_transforms

    net = _CNNNet.build(2, np.random.default_rng(0), (4, 8, 8), pool="avg")
    v = RNG.random((3, 16, 16, 16)) > 0.5
    side = np.zeros((3, 1), np.float32)
    mu0, _ = net.forward(_to_input(v), side)
    mu8, _ = net.forward(_to_input(np.roll(v, (8, 0, 8), axis=(1, 2, 3))), side)
    assert np.allclose(mu0, mu8, atol=1e-5)  # shift by the total pooling stride: exact
    # TTA over D4h makes the ensemble exactly invariant to those operations
    vox = RNG.random((24, 16, 16, 16)) > RNG.uniform(0.3, 0.7, size=(24, 1, 1, 1))
    y = vox.mean(axis=(1, 2, 3))[:, None]
    e = CNNEnsemble(n_members=1, epochs=25, batch=8, lr=3e-3, channels=(4, 8, 8), pool="avgmax").fit(
        vox, np.zeros((24, 1), np.float32), np.log(y))
    hist = e.loss_history[0]
    assert hist[-1] < hist[0]
    t = d4h_transforms()[11]
    a = e.predict(vox[:4], np.zeros((4, 1), np.float32))[0]
    b = e.predict(apply_transform(vox[:4], t), np.zeros((4, 1), np.float32))[0]
    assert np.allclose(a, b, atol=1e-4)
    e2 = CNNEnsemble.from_arrays(e.to_arrays())
    assert np.allclose(e2.predict(vox[:3], np.zeros((3, 1), np.float32))[0], e.predict(vox[:3], np.zeros((3, 1), np.float32))[0])


# =============================================================================
# ClosureModel API
# =============================================================================
@pytest.fixture(scope="module")
def synthetic_models():
    from voxlat.surrogates import ClosureModel

    df, _ = _synthetic_table(70)
    gp = ClosureModel.fit(df, "gp", n_restarts=0)
    ens = ClosureModel.fit(df, "ensemble", n_members=2, hidden=(32, 32), epochs=400)
    return df, gp, ens


@pytest.mark.parametrize("which", ["gp", "ensemble"])
def test_closure_model_predict_api(synthetic_models, which):
    df, gp, ens = synthetic_models
    cm = gp if which == "gp" else ens
    out = cm.predict((0.3, 0.35, 1.1), L=4e-3)
    assert out["K"][0].shape == (3, 3) and out["C_eff"][0].shape == (6, 6) and out["E"][0].shape == (3,)
    assert np.ndim(out["a_sf"][0]) == 0
    for name in ("K", "k_eff", "C_eff"):
        m, s = out[name]
        assert np.allclose(m, m.T) and np.all(np.linalg.eigvalsh(m) > 0) and np.all(s >= 0)
    assert np.isclose(out["porosity"][0], 0.65) and out["porosity"][1] == 0
    assert np.isclose(out["K_star"][0], np.trace(out["K"][0]) / 3)
    assert np.isclose(out["E_star"][0], out["E"][0].mean())
    assert np.allclose(np.sort(np.linalg.eigvalsh(out["K"][0])), out["K_principal"][0])
    # L scaling is analytic
    out2 = cm.predict((0.3, 0.35, 1.1), L=8e-3)
    assert np.allclose(out2["K"][0], 4 * out["K"][0]) and np.isclose(out2["a_sf"][0], out["a_sf"][0] / 2)
    assert np.allclose(out2["C_eff"][0], out["C_eff"][0]) and np.allclose(out2["k_eff"][0], out["k_eff"][0])
    # material scales
    assert 0 < out["k_eff"][0][0, 0] < 130 and 0 < out["C_eff"][0][0, 0] < 70e9
    # batch form
    ob = cm.predict(np.array([[0.0, 0.3, 1.0], [0.5, 0.4, 1.0], [1.0, 0.25, 1.3]]), L=[2e-3, 3e-3, 4e-3])
    assert ob["K"][0].shape == (3, 3, 3) and ob["loc_uniaxial_z_p99"][0].shape == (3,)


def test_closure_model_enforces_symmetry_class(synthetic_models):
    _, gp, _ = synthetic_models
    out = gp.predict(np.array([[0.0, 0.3, 1.0], [1.0, 0.3, 1.25], [0.5, 0.3, 1.0]]), L=1e-3)
    K, C = out["K"][0], out["C_eff"][0]
    assert np.allclose(K[0], K[0][0, 0] * np.eye(3), rtol=1e-12, atol=1e-30)  # cubic: isotropic K
    c = C[0]
    assert np.allclose(np.diag(c)[:3], c[0, 0]) and np.allclose(c[:3, 3:], 0, atol=1e-9 * c[0, 0])
    k1 = K[1]  # tetragonal
    assert np.isclose(k1[0, 0], k1[1, 1]) and np.allclose([k1[0, 1], k1[0, 2], k1[1, 2]], 0, atol=1e-12 * k1[0, 0])
    k2 = K[2]  # trigonal
    assert np.allclose(np.diag(k2), k2[0, 0]) and np.isclose(k2[0, 1], k2[1, 2])
    loc = np.array([out[f"loc_{c}_p99"][0][0] for c in LOAD_CASES])
    assert np.allclose(loc[:3], loc[0]) and np.allclose(loc[3:], loc[3])
    assert list(gp.symmetry_class([[0.0, 0.3, 1.0], [0.5, 0.3, 1.25]])) == ["cubic", "triclinic"]


def test_closure_model_fits_synthetic_truth(synthetic_models):
    df, gp, _ = synthetic_models
    Y, cl = latent_from_table(df)
    truth = derived_quantities(Y, None, cl)
    pred = gp.predict_dimensionless(df[["w", "rho_target", "a_z"]].to_numpy())
    for q in ("K_mean", "keff_xx", "E_axial", "a_sf_L"):
        assert np.max(np.abs(pred[q][0] / truth[q][0] - 1)) < 0.02, q


@pytest.mark.parametrize("which", ["gp", "ensemble"])
def test_closure_model_save_load(synthetic_models, which, tmp_path):
    from voxlat.surrogates import ClosureModel

    _, gp, ens = synthetic_models
    cm = gp if which == "gp" else ens
    p = cm.save(tmp_path / f"m_{which}.npz")
    cm2 = ClosureModel.load(p)
    th = np.array([[0.2, 0.3, 0.9], [0.0, 0.45, 1.0]])
    a, b = cm.predict(th, 3e-3), cm2.predict(th, 3e-3)
    for k in a:
        assert np.allclose(a[k][0], b[k][0], rtol=1e-12) and np.allclose(a[k][1], b[k][1], rtol=1e-12), k
    assert cm2.material == cm.material and cm2.box == cm.box
    assert p.stat().st_size < 2_000_000


def test_std_scale_from_oof():
    from voxlat.surrogates.evaluation import std_scale_from_oof

    rng = np.random.default_rng(5)
    Y = np.zeros((4000, 3))
    mu = rng.normal(size=(4000, 3)) * np.array([0.5, 1.0, 3.0])
    sd = np.ones_like(mu)
    sc = std_scale_from_oof(Y, mu, sd, np.full(4000, "triclinic"))
    assert np.allclose(sc, [1.0, 1.0, 3.0], rtol=0.05)  # Gaussian z: q95(|z|)/1.96 = scale; clipped below at 1
    z = np.abs(mu[:, 2] / (sd[:, 2] * sc[2]))
    assert abs(np.mean(z <= 1.96) - 0.95) < 0.01


def test_recalibrated_fit_scales_std():
    from voxlat.surrogates import ClosureModel

    df, _ = _synthetic_table(50)
    kw = dict(n_members=2, hidden=(16, 16), epochs=150, seed=3)
    raw = ClosureModel.fit(df, "ensemble", **kw)
    cal = ClosureModel.fit(df, "ensemble", recalibrate=True, cv_folds=3, **kw)
    assert cal.std_scale is not None and cal.std_scale.shape == (len(LATENT_NAMES),) and np.all(cal.std_scale >= 1)
    assert "recalibration" in cal.meta
    th = np.array([[0.3, 0.3, 1.2]])
    assert np.allclose(raw.predict(th, 1e-3)["K"][0], cal.predict(th, 1e-3)["K"][0])  # same mean
    assert np.all(cal.predict_latent(th)[1] >= raw.predict_latent(th)[1] - 1e-15)


def test_closure_model_bounds(synthetic_models, caplog):
    _, gp, _ = synthetic_models
    with pytest.raises(ValueError):
        gp.predict((0.0, 0.7, 1.0), 1e-3, check_bounds="raise")
    gp.predict((0.0, 0.7, 1.0), 1e-3)  # warn only
    with pytest.raises(ValueError):
        gp.predict((0.0, 0.3, 1.0), -1e-3)


# =============================================================================
# evaluation helpers
# =============================================================================
def test_point_metrics_and_splits():
    from voxlat.surrogates.evaluation import extrapolation_split, kfold_indices, point_metrics, sparse_blend_split

    t = np.array([1.0, 2.0, 4.0])
    m = point_metrics(t, t, 0.01 * t, True)
    assert m["R2"] == 1 and m["MAPE"] == 0 and m["cov95"] == 1
    m = point_metrics(t, 1.1 * t, 0.01 * t, True)
    assert np.isclose(m["MAPE"], 0.1) and m["cov95"] == 0
    folds = kfold_indices(23, 5)
    assert sorted(np.concatenate([te for _, te in folds]).tolist()) == list(range(23))
    w = np.r_[np.zeros(5), np.ones(5), np.linspace(0.05, 0.95, 40)]
    tr, te = extrapolation_split(w)
    assert np.all((w[te] >= 0.25) & (w[te] <= 0.75)) and np.all((w[tr] < 0.25) | (w[tr] > 0.75))
    tr, te = sparse_blend_split(w)
    assert set(range(10)) <= set(tr.tolist()) and len(te) == 30 and not set(tr) & set(te)


# =============================================================================
# real data and trained models (skip when unavailable)
# =============================================================================
def _real_table():
    from voxlat.closures.dataset import parquet_available
    from voxlat.surrogates import load_training_table
    from voxlat.utils.paths import data_dir

    env = os.environ.get("VOXLAT_CLOSURE_TABLE")
    cands = [Path(env)] if env else []
    cands += [data_dir() / "closures.parquet", data_dir() / "closures.csv"]
    for p in cands:
        if p.is_file() and (p.suffix == ".csv" or parquet_available()):
            return load_training_table(p)
    pytest.skip("Task 5 closure table not readable (needs data/closures.parquet + pyarrow, or VOXLAT_CLOSURE_TABLE)")


def test_real_table_symmetry_projection_is_small():
    df = _real_table()
    Y0, _ = latent_from_table(df, project=False)
    Y, cl = latent_from_table(df)
    assert len(df) >= 250 and set(cl) == set(SYMMETRY_CLASSES)
    assert np.abs(Y - Y0)[:, GROUP_SLICES["K"]].max() < 1e-3          # K solved with exact symmetry
    assert np.abs(Y - Y0)[:, GROUP_SLICES["keff"]].max() < 0.05        # voxel artifacts (stretched diamond)
    assert np.abs(Y - Y0)[:, GROUP_SLICES["C"]].max() < 0.05


@pytest.mark.slow
def test_real_table_gp_holdout_accuracy():
    from voxlat.surrogates.evaluation import kfold_indices, predict_split, summarize

    import pandas as pd

    df = _real_table()
    tr, te = kfold_indices(len(df), 5)[0]
    rows = predict_split(df, tr, te, "gp", n_restarts=0)
    m = summarize(pd.DataFrame(rows)).set_index("quantity")
    assert m.loc["K_mean", "MAPE"] < 0.03 and m.loc["K_mean", "R2"] > 0.99
    assert m.loc["keff_mean", "MAPE"] < 0.05
    assert m.loc["E_axial", "MAPE"] < 0.10
    assert m.loc["a_sf_L", "MAPE"] < 0.01


def _model_or_skip(backend):
    from voxlat.surrogates import ClosureModel, default_model_path

    p = default_model_path(backend)
    if not p.is_file():
        pytest.skip(f"{p} not trained (scripts/task6_train_surrogates.py)")
    return ClosureModel.load(p)


@pytest.mark.parametrize("backend", ["gp", "ensemble"])
def test_trained_model_reproduces_task2_to_4_tables(backend):
    """Pure gyroid / diamond, rho* = 0.35, a_z = 1 (STATUS.md Tasks 2-4 tables)."""
    cm = _model_or_skip(backend)
    q = cm.predict_dimensionless(np.array([[0.0, 0.35, 1.0], [1.0, 0.35, 1.0]]))
    assert np.allclose(q["K_mean"][0], [4.887e-3, 3.103e-3], rtol=0.03)
    assert np.allclose(q["keff_mean"][0], [0.208, 0.205], rtol=0.03)
    assert np.allclose(q["E_axial"][0], [0.0998, 0.0864], rtol=0.06)
    out = cm.predict((0.0, 0.35, 1.0), L=4e-3)
    assert np.isclose(out["K_star"][0], 4.887e-3 * 16e-6, rtol=0.03)  # STATUS Task 4: 7.8e-8 m^2


def test_trained_cnn_model_loads_and_predicts():
    from voxlat.surrogates import CNNClosureModel, default_model_path

    p = default_model_path("cnn")
    if not p.is_file():
        pytest.skip(f"{p} not trained")
    cnn = CNNClosureModel.load(p)
    out = cnn.predict((0.0, 0.35, 1.0), L=4e-3)
    assert np.isclose(out["K_star"][0], 4.887e-3 * 16e-6, rtol=0.1)
    assert np.isclose(out["E_star"][0] / 70e9, 0.0998, rtol=0.15)
