"""Task 8: literature closures (Forchheimer C_F, interstitial Nu) - published values, identities,
validity warnings, and independent checks of the Task 1/4 geometry and permeability.

Published numbers used below (all read from the open-access full texts):
* Gajetti et al. (2025) IJHMT 252:127439 - Table 2 (K/Lc^2, C_F at phi = 0.3-0.6) and Table 3 fits.
* Savoldi et al. (2026) IJHFF 121:110631 - Eqs. 23, 34-40 (no tabulated Nu values are given).
* Gnielinski / Wakao-Kaguei reference values as documented by the `ht` library (C. Bell).
Our own numbers (Task 1/4, R(32, 64) estimators) are copied from results/task4_permeability_vs_porosity.csv.
"""

from __future__ import annotations

import doctest
import warnings

import numpy as np
import pandas as pd
import pytest

from voxlat.closures import empirical as emp
from voxlat.closures.empirical import (
    GAJETTI2025_TABLE2,
    LITERATURE,
    RECOMMENDED,
    CorrelationRangeError,
    CorrelationRangeWarning,
    check_range,
    equivalent_particle_diameter,
    forchheimer_coefficient,
    friction_factor,
    friction_factor_re,
    gajetti2025_forchheimer,
    gajetti2025_permeability,
    hydraulic_diameter,
    interstitial_heat_transfer,
    literature_table,
    nusselt_gnielinski_packed_bed,
    nusselt_interstitial,
    nusselt_reynolds2023,
    nusselt_wakao_kaguei,
    pore_velocity,
    pressure_gradient,
    range_report,
    reynolds_hydraulic,
    reynolds_permeability,
    savoldi2026_forchheimer,
    savoldi2026_hydraulic_diameter,
    savoldi2026_permeability,
    stanton_savoldi2026,
)
from voxlat.utils import load_config

# ---- our Task 4 results (R(32,64), results/task4_permeability_vs_porosity.csv): phi -> (K/L^2, a_sf L)
OURS_GYROID = {
    0.80: (1.0304102e-2, 2.5308657),
    0.70: (6.2140391e-3, 2.8664243),
    0.65: (4.8869531e-3, 2.9695954),
    0.60: (3.8286188e-3, 3.0367918),
    0.55: (2.9495261e-3, 3.0801797),
    0.50: (2.2244365e-3, 3.0932994),
}
OURS_DIAMOND = {
    0.80: (6.6507e-3, 3.1115961),
    0.60: (2.3904e-3, 3.7717433),
    0.50: (1.4095e-3, 3.8419108),
}


@pytest.fixture
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", CorrelationRangeWarning)
        yield


# =============================================================================
# Published coefficients and tables
# =============================================================================
def test_published_coefficients():
    g = emp.GAJETTI2025_POWER_LAWS["gyroid"]
    d = emp.GAJETTI2025_POWER_LAWS["diamond"]
    assert (g["a"], g["n"], g["b"], g["m"]) == (0.0156, 2.78, 0.0908, -1.81)
    assert (d["a"], d["n"], d["b"], d["m"]) == (0.0106, 2.89, 0.0666, -1.69)
    s = emp.SAVOLDI2026_HYDRAULICS
    assert (s["c2"], s["c1"], s["c0"], s["a"], s["n"], s["b"], s["m"]) == (-5.723, 5.729, 1.665, 0.0157, 2.72, 0.0784, -2.04)
    st = emp.SAVOLDI2026_STANTON
    assert (st[1]["A"], st[1]["n"]) == (0.0267, 0.19)
    assert (st[2]["A"], st[2]["n"], st[2]["m0"], st[2]["m1"]) == (0.135, 2.06, 0.440, -1.0)
    assert (st[3]["A"], st[3]["n"], st[3]["m0"], st[3]["m1"]) == (0.034, 0.20, -0.061, -0.0023)


@pytest.mark.parametrize("tpms", ["gyroid", "diamond"])
def test_gajetti_power_laws_reproduce_their_table2(tpms):
    """Gajetti et al. (2025): the Eq. 18-19 fits against their own Table 2 CFD values.

    K within 7 %, C_F within 18 % (the fits are least accurate at phi = 0.6 - the edge our jacket
    needs; diamond C_F flattens to 0.19-0.20 at phi = 0.5-0.6 while the power law keeps falling).
    Log-space R^2 on the four 2-digit table values: K 0.997-0.999, C_F 0.94-0.97 (they report
    0.979-0.999 on their full data).
    """
    t = GAJETTI2025_TABLE2[tpms]
    k_fit = gajetti2025_permeability(t[:, 0], tpms)
    cf_fit = gajetti2025_forchheimer(t[:, 0], tpms)
    np.testing.assert_allclose(k_fit, t[:, 1], rtol=0.07)
    np.testing.assert_allclose(cf_fit, t[:, 2], rtol=0.18)
    for col, fit, r2_min in ((1, k_fit, 0.995), (2, cf_fit, 0.93)):
        y = np.log(t[:, col])
        r2 = 1 - np.sum((y - np.log(fit)) ** 2) / np.sum((y - y.mean()) ** 2)
        assert r2 > r2_min
    # spot values quoted in STATUS.md
    assert gajetti2025_forchheimer(0.3, tpms) == pytest.approx({"gyroid": 0.8026, "diamond": 0.5095}[tpms], rel=1e-3)


def test_savoldi_hydraulics_spot_values():
    # D_h/L at phi = 0.5: 4*0.5/(-5.723/4 + 5.729/2 + 1.665) = 2/3.09875
    assert savoldi2026_hydraulic_diameter(0.5) == pytest.approx(2.0 / 3.09875, rel=1e-12)
    assert savoldi2026_permeability(0.7) == pytest.approx(0.0157 * 0.7**2.72, rel=1e-12)
    assert savoldi2026_forchheimer(0.7) == pytest.approx(0.1623, rel=2e-3)
    # the 2026 gyroid refit (to phi = 0.7) and the 2025 one: +14 % at phi = 0.3, within 10 % for
    # phi = 0.5-0.8 (our range; -9 % at 0.8) - this bounds the gyroid C_F extrapolation above phi = 0.6
    phi = np.linspace(0.3, 0.8, 11)
    ratio = savoldi2026_forchheimer(phi) / gajetti2025_forchheimer(phi, "gyroid")
    assert np.all(np.abs(ratio - 1) < 0.15)
    assert np.all(np.abs(ratio[phi >= 0.5] - 1) < 0.10)


# =============================================================================
# Independent checks of OUR geometry (Task 1) and permeability (Task 4) against the literature
# =============================================================================
@pytest.mark.parametrize(
    "tpms,phi,table_row",
    [("gyroid", 0.50, 2), ("gyroid", 0.60, 3), ("diamond", 0.50, 2), ("diamond", 0.60, 3)],
)
def test_task4_permeability_matches_gajetti_cfd(tpms, phi, table_row):
    """Our voxel-Stokes K (Task 4) vs Gajetti et al.'s body-fitted OpenFOAM K, both skeletal TPMS.

    Observed: G +1.1 % / -1.8 %, D +0.7 % / -0.4 %; their Table 2 has two significant digits
    (rounding up to +-2.3 %), hence the 4 % tolerance.
    """
    ours = (OURS_GYROID if tpms == "gyroid" else OURS_DIAMOND)[phi][0]
    theirs = GAJETTI2025_TABLE2[tpms][table_row]
    assert theirs[0] == phi
    assert ours == pytest.approx(theirs[1], rel=0.04)


def test_task1_surface_area_and_task4_K_match_savoldi_gyroid():
    """D_h of our skeletal gyroid (Task 1 a_sf) vs Savoldi et al.'s fit: < 0.3 % for phi = 0.5-0.7;
    our K vs their K fit: within 8 % inside their range (0.3-0.7). Outside (phi = 0.8) their power law
    is 17 % low - one reason VoxLat uses its own K and not literature permeability fits."""
    for phi in (0.50, 0.55, 0.60, 0.65, 0.70):
        k_ours, asl = OURS_GYROID[phi]
        assert 4 * phi / asl == pytest.approx(savoldi2026_hydraulic_diameter(phi), rel=3e-3)
        assert k_ours == pytest.approx(savoldi2026_permeability(phi), rel=0.08)
    k_ours, _ = OURS_GYROID[0.80]
    assert savoldi2026_permeability(0.80) / k_ours - 1 < -0.15


# =============================================================================
# Identities
# =============================================================================
def test_dimensionless_groups():
    u, phi, asf, nu = 0.083, 0.65, 742.5, 2.3e-6
    dh = hydraulic_diameter(phi, asf)
    assert dh == pytest.approx(4 * phi / asf)
    assert reynolds_hydraulic(u, asf, nu) == pytest.approx(pore_velocity(u, phi) * dh / nu, rel=1e-12)
    assert reynolds_hydraulic(-u, asf, nu) == reynolds_hydraulic(u, asf, nu)
    # Gajetti's D_p = 1.5 (1-phi)/phi D_h  ==  6 (1-phi)/a_sf
    assert equivalent_particle_diameter(phi, asf) == pytest.approx(1.5 * (1 - phi) / phi * dh, rel=1e-12)
    assert reynolds_permeability(u, 7.8e-8, nu) == pytest.approx(u * np.sqrt(7.8e-8) / nu)


@pytest.mark.parametrize("phi,cf", [(0.5, 0.31), (0.8, 0.12)])
def test_friction_factor_is_darcy_forchheimer(phi, cf):
    """Eq. (2) (Savoldi Eq. 23) == the Darcy-Forchheimer gradient put into the f definition."""
    rho, mu, L = 1060.0, 1060.0 * 2.3e-6, 4e-3
    K, dh = 4e-3 * L**2, 0.8 * L
    for u_s in (1e-3, 0.03, 0.2):
        u_b = u_s / phi
        re = rho * u_b * dh / mu
        dpdx = pressure_gradient(u_s, K, cf, density=rho, dynamic_viscosity=mu)
        f_def = dpdx * dh / (0.5 * rho * u_b**2)
        assert friction_factor(re, phi, dh, K, cf) == pytest.approx(f_def, rel=1e-12)
        # scale invariance: relative (D_h/L, K/L^2) inputs give the same f
        assert friction_factor(re, phi, dh / L, K / L**2, cf) == pytest.approx(f_def, rel=1e-12)
    # Re -> 0: f Re -> Darcy limit 2 phi D_h^2 / K
    assert friction_factor_re(0.0, phi, dh, K, cf) == pytest.approx(2 * phi * dh**2 / K)
    with pytest.raises(ValueError):
        friction_factor(0.0, phi, dh, K, cf)


def test_stanton_equals_nusselt_over_re_at_pr1(no_warnings):
    phi, re = 0.5, np.array([20.0, 50.0, 100.0])
    dh, K, cf = savoldi2026_hydraulic_diameter(phi), savoldi2026_permeability(phi), savoldi2026_forchheimer(phi)
    f = friction_factor(re, phi, dh, K, cf)
    for v in (1, 2, 3):
        nu = nusselt_interstitial(re, phi, f * re, 1.0, variant=v)
        np.testing.assert_allclose(nu, stanton_savoldi2026(re, phi, f, variant=v) * re, rtol=1e-12)
    # hand value, variant 2 at Re = 50: St = 0.135 * 0.5^2.06 * 50^(0.44 - 0.5) * f
    st2 = 0.135 * 0.5**2.06 * 50 ** (-0.06) * f[1]
    assert stanton_savoldi2026(50.0, phi, f[1], variant=2) == pytest.approx(st2, rel=1e-12)


def test_savoldi_variants_agree_inside_their_box():
    """Each variant reproduces the CFD within +-11-15 % (1 sd); inside phi = 0.3-0.7, Re = 20-100 they
    agree with each other within -23 % / +20 % (computed with the paper's own hydraulic fits)."""
    phi = np.linspace(0.3, 0.7, 9)[:, None]
    re = np.linspace(20, 100, 9)[None, :]
    dh, K, cf = savoldi2026_hydraulic_diameter(phi), savoldi2026_permeability(phi), savoldi2026_forchheimer(phi)
    fre = friction_factor_re(re, phi, dh, K, cf)
    nu = {v: nusselt_interstitial(re, phi, fre, 1.0, variant=v, on_extrapolation="ignore") for v in (1, 2, 3)}
    for a, b in ((2, 1), (3, 1), (2, 3)):
        r = nu[a] / nu[b]
        assert r.min() > 0.75 and r.max() < 1.25
    assert 4.5 < nu[1].min() and nu[1].max() < 11.0  # Pr = 1: Nu_Dh ~ 5-11 in the fitted box


def test_nusselt_low_re_limit_and_pr_scaling():
    phi, dh, K, cf = 0.7, 0.977, 6.214e-3, 0.17
    fre0 = friction_factor_re(0.0, phi, dh, K, cf)
    nu0 = nusselt_interstitial(0.0, phi, fre0, 1.0, on_extrapolation="ignore")
    assert nu0 == pytest.approx(0.0267 * phi**0.19 * 2 * phi * dh**2 / K, rel=1e-12)
    assert np.isfinite(nu0) and nu0 > 0
    re = np.array([10.0, 100.0, 600.0])
    fre = friction_factor_re(re, phi, dh, K, cf)
    n1 = nusselt_interstitial(re, phi, fre, 1.0, on_extrapolation="ignore")
    n20 = nusselt_interstitial(re, phi, fre, 20.72, on_extrapolation="ignore")
    np.testing.assert_allclose(n20 / n1, 20.72 ** (1 / 3), rtol=1e-12)
    n20b = nusselt_interstitial(re, phi, fre, 20.72, pr_exponent=0.4, on_extrapolation="ignore")
    np.testing.assert_allclose(n20b / n20, 20.72 ** (0.4 - 1 / 3), rtol=1e-12)  # = 1.224: the Pr-exponent band
    assert np.all(np.diff(n1) > 0)  # variant 1 is monotone in Re
    with pytest.raises(ValueError):
        nusselt_interstitial(0.0, phi, fre0, 1.0, variant=2, on_extrapolation="ignore")


def test_variant2_is_not_monotone_outside_its_box():
    """Why variant 1 is recommended for the jacket: at phi = 0.8 (outside 0.3-0.7) variant 2 predicts Nu
    FALLING with Re between Re = 10 and 50 (Re^(0.44 - phi) times the Darcy part of f)."""
    phi, dh, K = 0.8, 1.2644, 1.0304e-2
    cf = gajetti2025_forchheimer(phi, "gyroid")
    re = np.array([10.0, 20.0, 50.0])
    fre = friction_factor_re(re, phi, dh, K, cf)
    n2 = nusselt_interstitial(re, phi, fre, 1.0, variant=2, on_extrapolation="ignore")
    n1 = nusselt_interstitial(re, phi, fre, 1.0, variant=1, on_extrapolation="ignore")
    assert n2[0] > n2[1] > n2[2]
    assert np.all(np.diff(n1) > 0)


# =============================================================================
# C_F: sources, blends, validity
# =============================================================================
def test_forchheimer_sources_and_blend():
    phi = np.array([0.5, 0.65, 0.8])
    g = forchheimer_coefficient(phi, 0.0, on_extrapolation="ignore")
    d = forchheimer_coefficient(phi, 1.0, on_extrapolation="ignore")
    np.testing.assert_allclose(g, gajetti2025_forchheimer(phi, "gyroid"), rtol=1e-12)
    np.testing.assert_allclose(d, gajetti2025_forchheimer(phi, "diamond"), rtol=1e-12)
    b = forchheimer_coefficient(phi, 0.5, on_extrapolation="ignore")
    np.testing.assert_allclose(b, np.sqrt(g * d), rtol=1e-12)
    # broadcasting with arrays of w
    bw = forchheimer_coefficient(0.65, np.array([0.0, 0.5, 1.0]), on_extrapolation="ignore")
    np.testing.assert_allclose(bw, [g[1], b[1], d[1]], rtol=1e-12)
    # table variant: exact at table points, constant outside 0.3-0.6 (diamond plateau 0.19)
    t = forchheimer_coefficient(np.array([0.4, 0.6, 0.8]), 1.0, source="gajetti2025_table", on_extrapolation="ignore")
    np.testing.assert_allclose(t, [0.30, 0.19, 0.19], rtol=1e-12)
    # diamond at phi = 0.8: power law vs plateau differ ~2x - the largest C_F uncertainty (STATUS.md)
    assert 1.8 < t[2] / d[2] < 2.1
    s = forchheimer_coefficient(0.65, 0.0, source="savoldi2026", on_extrapolation="ignore")
    assert s == pytest.approx(0.0784 * 0.65**-2.04)
    with pytest.raises(ValueError):
        forchheimer_coefficient(0.65, 0.5, source="savoldi2026")
    with pytest.raises(ValueError):
        forchheimer_coefficient(1.2, 0.0)
    with pytest.raises(ValueError):
        forchheimer_coefficient(0.6, 1.5)
    assert np.ndim(forchheimer_coefficient(0.5, 0.0, on_extrapolation="ignore")) == 0


def test_validity_warnings(no_warnings):
    # inside the box: silent (no_warnings turns warnings into errors)
    forchheimer_coefficient(np.array([0.35, 0.55]), 0.0, re=np.array([5.0, 90.0]))
    nusselt_interstitial(50.0, 0.5, 300.0, 1.0)
    with pytest.warns(CorrelationRangeWarning, match="porosity outside"):
        forchheimer_coefficient(0.8, 0.0)
    with pytest.warns(CorrelationRangeWarning, match="re outside"):
        forchheimer_coefficient(0.5, 1.0, re=600.0)
    with pytest.warns(CorrelationRangeWarning, match="topology"):
        forchheimer_coefficient(0.5, 0.3)
    with pytest.warns(CorrelationRangeWarning, match="pr outside"):
        nusselt_interstitial(50.0, 0.5, 300.0, 20.72)
    with pytest.raises(CorrelationRangeError):
        nusselt_interstitial(500.0, 0.5, 300.0, 1.0, on_extrapolation="raise")
    assert np.isfinite(nusselt_interstitial(500.0, 0.5, 300.0, 1.0, on_extrapolation="ignore"))
    with pytest.raises(ValueError):
        check_range("gajetti2025", porosity=0.5, on_extrapolation="loud")


def test_range_report_fractions():
    rep = range_report("savoldi2026_st1", re=[10, 50, 80, 300], porosity=[0.5, 0.8], pr=20.72, w=[0.0, 0.5, 1.0])
    assert rep["re"]["fraction_outside"] == pytest.approx(0.5)
    assert rep["porosity"]["fraction_outside"] == pytest.approx(0.5)
    assert rep["pr"]["fraction_outside"] == 1.0
    assert rep["topology"]["outside"] == ["blend", "diamond"]
    assert range_report("gnielinski_packed_bed", w=0.5)["topology"]["outside"] == []


# =============================================================================
# References
# =============================================================================
def test_gnielinski_reference_value(no_warnings):
    """`ht.conv_packed_bed.Nu_packed_bed_Gnielinski(dp=8E-4, voidage=0.4, vs=1, rho=1E3, mu=1E-3, Pr=0.7)`
    = 61.37823202546954 (documented example); Re = rho vs dp/(mu voidage) = 2000."""
    re = 1e3 * 1.0 * 8e-4 / (1e-3 * 0.4)
    nu = nusselt_gnielinski_packed_bed(re, 0.7, 0.4, on_extrapolation="ignore")  # Re = 2000 > stated 1000
    assert nu == pytest.approx(61.37823202546954, rel=1e-9)
    nusselt_gnielinski_packed_bed(500.0, 20.7, 0.6)  # inside 0.1-1000, 0.4-1000: silent
    # Re -> 0: Nu -> 2 f_a (conduction to a sphere)
    assert nusselt_gnielinski_packed_bed(0.0, 7.0, 0.4, on_extrapolation="ignore") == pytest.approx(2 * 1.9)


def test_wakao_kaguei_reference_value():
    """`ht.conv_packed_bed.Nu_Wakao_Kagei(2000, 0.7)` = 95.40641328041248 (documented example)."""
    assert nusselt_wakao_kaguei(2000.0, 0.7, on_extrapolation="ignore") == pytest.approx(95.40641328041248, rel=1e-12)


def test_reynolds2023_formula(no_warnings):
    assert nusselt_reynolds2023(500.0, 0.71) == pytest.approx(0.49 * 500**0.62 * 0.71**0.4, rel=1e-12)


def test_recommended_nu_brackets_pr_validated_packed_bed():
    """Over the whole jacket box (G and D, phi = 0.5-0.8, Re_Dh = 10-600, Pr = 20.72) the recommended
    Nu (variant 1 x Pr^1/3, our K and a_sf, Gajetti C_F) stays within 0.6-1.45x of the Pr-validated
    Gnielinski packed-bed correlation (spheres with the same a_sf). Variant 2 falls to 0.27x at Re = 600."""
    pr = 20.72
    re = np.geomspace(10, 600, 25)
    ratios_1, ratios_2 = [], []
    for w, table in ((0.0, OURS_GYROID), (1.0, OURS_DIAMOND)):
        for phi, (K, asl) in table.items():
            dh, dp = 4 * phi / asl, 6 * (1 - phi) / asl
            cf = forchheimer_coefficient(phi, w, on_extrapolation="ignore")
            fre = friction_factor_re(re, phi, dh, K, cf)
            nu_g = nusselt_gnielinski_packed_bed(re * dp / dh, pr, phi, on_extrapolation="ignore") * dh / dp
            for v, store in ((1, ratios_1), (2, ratios_2)):
                nu = nusselt_interstitial(re, phi, fre, pr, variant=v, w=w, on_extrapolation="ignore")
                store.append(nu / nu_g)
    r1, r2 = np.concatenate(ratios_1), np.concatenate(ratios_2)
    assert 0.6 < r1.min() and r1.max() < 1.45
    assert r2.min() < 0.35


# =============================================================================
# Convenience chain, table, docs
# =============================================================================
def test_interstitial_heat_transfer_chain():
    cfg = load_config()
    L = 4e-3
    phi = 0.65
    K, asl = OURS_GYROID[phi]
    with pytest.warns(CorrelationRangeWarning):
        r = interstitial_heat_transfer(
            np.array([0.0, 0.03, 0.083]),
            phi,
            asl / L,
            K * L**2,
            conductivity=cfg.coolant.conductivity,
            kinematic_viscosity=cfg.coolant.kinematic_viscosity,
            prandtl=cfg.coolant.prandtl,
        )
    assert np.isnan(r.friction[0]) and np.isfinite(r.nusselt[0])  # stagnant cell: Darcy-limit Nu
    np.testing.assert_allclose(r.h_sf, r.nusselt * cfg.coolant.conductivity / r.hydraulic_diameter)
    np.testing.assert_allclose(r.h_sf_a_sf, r.h_sf * asl / L)
    assert np.all(np.diff(r.h_sf) > 0)
    assert r.settings["variant"] == 1 and r.settings["cf_source"] == "gajetti2025"
    assert float(r.reynolds[2]) == pytest.approx(4 * 0.083 / (asl / L * cfg.coolant.kinematic_viscosity))
    # user C_F override bypasses the literature C_F
    r2 = interstitial_heat_transfer(
        0.083, phi, asl / L, K * L**2, cf=0.0,
        conductivity=0.4, kinematic_viscosity=2.3e-6, prandtl=1.0, on_extrapolation="ignore",
    )
    assert float(r2.friction_re) == pytest.approx(2 * phi * (4 * phi / asl * L) ** 2 / (K * L**2))


def test_literature_table_and_recommendation():
    df = literature_table()
    assert isinstance(df, pd.DataFrame)
    for col in ("source", "tpms", "Re_range", "porosity_range", "Pr_range", "formula", "method", "access"):
        assert col in df.columns
    keys = set(df["key"])
    assert {"gajetti2025", "savoldi2026_st1", "cheng2021", "brambati2024", "gnielinski_packed_bed"} <= keys
    assert set(literature_table(implemented_only=True)["implemented"]) == {True}
    assert RECOMMENDED["forchheimer"]["source"] == "gajetti2025"
    assert RECOMMENDED["nusselt"]["correlation"] in LITERATURE
    assert LITERATURE["savoldi2026_st1"].validity.pr == (1.0, 1.0)


def test_module_doctests():
    res = doctest.testmod(emp, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert res.failed == 0 and res.attempted >= 1
