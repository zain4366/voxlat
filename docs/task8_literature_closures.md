# Task 8 — Literature closures for heat transfer and inertial drag

VoxLat computes K, k_eff and C_eff itself (Tasks 2–7). Two closures of the Task 9 jacket model are taken
from the literature: the **Forchheimer coefficient C_F** and the **interstitial Nusselt number** (h_sf).
Code: `voxlat.closures.empirical`; tests: `tests/test_empirical.py`; figures and tables:
`scripts/task8_literature_closures.py`.

## Definitions used everywhere

| Symbol | Definition |
|---|---|
| φ | porosity = 1 − ρ* |
| a_sf | wetted area per total volume (Task 1) |
| D_h | 4φ / a_sf |
| U_b | U_s / φ (pore velocity) |
| Re_Dh | U_b D_h / ν = 4 U_s / (a_sf ν) |
| Re_K, Fo | U_s √K / ν, and Fo = C_F Re_K (inertial ÷ viscous drag) |
| Darcy–Forchheimer | −∇p = μU_s/K + ρ C_F U_s² / √K |
| f | (dp/dx) D_h / (½ρU_b²) = 2φD_h²/(K Re) + 2φ² C_F D_h/√K |
| Nu | h_sf D_h / k_f, with h_sf the surface-averaged wall flux over (T_wall − T_bulk) |

## Literature table

Coefficients were read from the open-access full texts unless the "access" column says otherwise.

| # | Source | TPMS | Validity (Re; φ; Pr) | Formula | How obtained | Access |
|---|---|---|---|---|---|---|
| F1 | **Gajetti, Boccardo, Savoldi & Marocco 2025**, IJHMT 252:127439 | gyroid, diamond, Split-P; skeletal | Re_Dh 0.3–100; φ 0.30–0.60; water (hydraulics only) | K/L² = aφⁿ, C_F = bφᵐ; G: 0.0156, 2.78, 0.0908, −1.81; D: 0.0106, 2.89, 0.0666, −1.69 | OpenFOAM, laminar, periodic unit cell (L = 10 mm); pipe validation | full text |
| F2 | **Savoldi, Cammi, Gajetti & Marocco 2026**, IJHFF 121:110631 | gyroid; skeletal | Re_Dh 20–100; φ 0.3–0.7 | D_h/L = 4φ/(−5.723φ² + 5.729φ + 1.665); K/L² = 0.0157φ^2.72; C_F = 0.0784φ^−2.04 | same group, single periodic cell | full text |
| F3 | Rathore, Mehta, Kumar & Asfer 2023, TiPM 146:669–701 | D, G, I-WP, P | Re 0.01–100 (channel-length based); φ = 0.32 only | data: C_F = 1.173 (G), 0.525 (D) | DNS of a 4-cell channel, walls included | arXiv preprint |
| F4 | Ergun 1952 (assessed by F1) | packed beds | — | 150/Re* + 1.75 | experiments | via F1 |
| H1 | **Savoldi et al. 2026**, Eq. 37 (variant 1) | gyroid; skeletal; whole interface | Re_Dh 20–100; φ 0.3–0.7; **Pr = 1** | St = 0.0267 φ^0.19 f | CFD, uniform wall T and uniform wall q (non-conjugate); ±11 % | full text |
| H2 | Savoldi et al. 2026, Eqs. 38–39 (variant 2) | as H1 | as H1 | St = 0.135 φ^2.06 Re^(0.440−φ) f | as H1; ±15 % | full text |
| H3 | Savoldi et al. 2026, Eq. 40 (variant 3) | as H1 | as H1 | St = 0.034 φ^0.20 Re^(−0.061−0.0023φ) f | as H1; ±12 % | full text |
| H4 | Cheng, Li, Xu & Jiang 2021, ICHMT 129:105713 | W, P, D, G porous media | Re_h 10–129; ε 0.2–0.8; air | Nu(Re_h, ε) and flow resistance | pore-scale CFD | **abstract only (paywalled)** |
| H5 | Iyer et al. 2022, Appl. Therm. Eng. 209:118192 | sheet TPMS / nodal surfaces, two-fluid HX | laminar, Re ≲ 350 | — | CFD | **not accessed** |
| H6 | Reynolds, Fee, Morison & Holland 2023, IJHMT | gyroid HX | Re_Dh 100–2500; air | Nu = 0.49 Re^0.62 Pr^0.4 | experiments | quoted by H7 |
| H7 | Brambati, Guilizzoni & Foletti 2024, Appl. Therm. Eng. 242:122492 | G, P, D sheet; φ 0.7–0.9 | Re 5000–50000; Pr 0.7–7 | Nu_Dh = 0.0964 Re_Dp^0.7136 Pr^0.4 | conjugate RANS CFD | full text |
| H8 | Piandoro et al. 2026, Lasers Manuf. Mater. Process. 13:555–581 | gyroid sheet; φ 0.04–0.83 | ethylene glycol (Pr ≈ 154), EG/water Pr 14 | Nu = (23.44 + 7.5φ)√Re_por, f ≈ 1 + 1/Re_por | CFD + L-PBF samples; wall-to-bulk h of a filled duct | full text |
| R1 | Gnielinski (VDI Heat Atlas) | packed spheres | Re_p 0.1–1000; Pr 0.4–1000 | Nu_p = (1 + 1.5(1−φ))(2 + √(Nu_lam² + Nu_turb²)) | experiments | `ht` library |
| R2 | Wakao & Kaguei 1982 | packed spheres | Re_p 3–3000 | Nu_p = 2 + 1.1 Re_p^0.6 Pr^(1/3) | experiments | `ht` library |
| R3 | Kuwahara, Shirota & Nakayama 2001, IJHMT 44:1153 | square-rod arrays | φ 0.2–0.9; computed at Pr = 1 | (porosity term) + (Re term) Pr^(1/3) | CFD | equation text garbled in accessible copies; not implemented |

## Recommendations

**C_F — Gajetti et al. 2025 power laws (F1)**, blends interpolated log-linearly in w.
1. Only source with gyroid **and** diamond from one method, so the blend interpolation is consistent.
2. Skeletal (network) cells like ours: their K agrees with our Task 4 solver to −1.8…+1.1 % (G) and −0.4…+0.7 % (D) at φ = 0.5–0.6, and Savoldi's D_h fit agrees with our a_sf to < 0.3 %.
3. Their Re range starts in the Darcy regime (0.3), so K and C_F are separated cleanly.
4. The gyroid refit by Savoldi et al. 2026 (to φ = 0.7) stays within 10 % over φ = 0.5–0.8, which bounds the gyroid extrapolation.

**Nu — Savoldi et al. 2026 modified Reynolds analogy, variant 1 (H1) × Pr^(1/3)**, with f evaluated from our K, our a_sf and the F1 C_F.
1. Only laminar, skeletal-gyroid correlation with porosity dependence whose coefficients we could verify; its h is the whole-interface (interstitial) coefficient the two-equation model needs.
2. Variant 1 extrapolates physically: Re → 0 gives a constant (fully developed) Nu, and high Re inherits the Forchheimer law. It is monotone across the jacket box. Variant 2 makes Nu fall with Re outside its box (φ = 0.8: 10.0 → 7.4 between Re 10 and 50 at Pr = 1) and is 0.33–0.94× variant 1 at Re = 600 (lowest at φ = 0.8); variant 3 has one-sided validation bias.
3. With Pr^(1/3) it stays within 0.63–1.37× of the Pr-validated Gnielinski packed-bed correlation over the whole box; variant 2 drops to 0.27×.
4. Cost: inside the fitted box variant 2 has more symmetric residuals and gives up to 22 % lower Nu at its high-φ, high-Re corner. **Use variant 2 as the low-h bound in sensitivity runs.**

## Extrapolation risk for VoxLat (Pr = 20.7, φ = 0.5–0.8, Re_Dh = 25–577)

| Risk | Size | Where it matters |
|---|---|---|
| **Pr** — H1 data at Pr = 1 only; Pr^(1/3) multiplies h by 2.75 | exponent 0.4 instead: +22 %; Gnielinski cross-check 0.63–1.37× | h_sf, thermal resistance |
| **Re** — fitted to Re_Dh ≤ 100; 70 % of operating points are above | at Re = 600 variants 1 and 2 differ by up to 3× (φ = 0.8); C_F assumed constant beyond weak inertia | h_sf and Δp at high flow |
| **Porosity** — C_F fitted to φ ≤ 0.6, H1 to φ ≤ 0.7 | diamond C_F at φ > 0.6: power law vs Table-2 plateau 1.2–2.0×; gyroid C_F ±10 % | Δp, pump power (inertia = 16–84 % of Δp) |
| **Topology** — Nu only for gyroid; no blend data at all | analogy carried over via each cell's own f | diamond/blend h_sf |
| **Single group** — F1, F2 and H1 come from one group (same CFD set-up) | Rathore's C_F at φ = 0.32: 1.15× (D), 1.64× (G) of F1, but with channel walls | C_F |
| **Finite gap and roughness** — bulk closures; L-PBF roughness not modelled | unknown; Savoldi et al. note systematically higher measured f for printed gyroids (Hirokawa & Miyata 2024) | Δp, h_sf |

## Manuscript limitations paragraph (~150 words)

Two closures of the device model are taken from the literature. The inertial (Forchheimer) coefficient comes from pore-scale simulations of skeletal gyroid and diamond cells fitted for porosities of 0.3–0.6 and Re_Dh ≤ 100 (Gajetti et al., 2025). Our jacket operates at porosities of 0.5–0.8 and Re_Dh ≈ 25–580, where inertia carries 16–84 % of the pressure drop, so pumping-power predictions inherit this extrapolation; for diamond above φ = 0.6, plausible C_F values differ by up to a factor of two. The interstitial heat-transfer coefficient follows the modified Reynolds analogy of Savoldi et al. (2026), calibrated for the gyroid at Pr = 1, φ ≤ 0.7 and Re_Dh = 20–100. We extend it to the water–glycol coolant (Pr ≈ 21) with the Chilton–Colburn factor Pr^(1/3), and to diamond and blends through their own friction factors; an exponent of 0.4 would raise h_sf by 22 %. Removing these assumptions requires experiments or conjugate pore-resolved simulations at the operating Prandtl number.
