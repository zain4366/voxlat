# T04 - literature closures

TPMS heat-transfer and inertial correlations considered (source, TPMS type, method, validity ranges).

Sources: `results/task8_literature_table.csv`, `docs/task8_literature_closures.md`

| key | quantity | source | doi | tpms | method | Re_range | porosity_range | Pr_range | implemented |
|---|---|---|---|---|---|---|---|---|---|
| gajetti2025 | K, C_F | Gajetti, Boccardo, Savoldi & Marocco (2025), Int. J. Heat Mass Transf. 252:127439 | 10.1016/j.ijheatmasstransfer.2025.127439 | gyroid, diamond, Split-P (skeletal/solid network) | pore-scale CFD (OpenFOAM, laminar, periodic unit cell Lc = 10 mm; pipe validation Lc = 5 mm) | 0.3-100 | 0.3-0.6 | not stated | True |
| savoldi2026_hydraulics | D_h, K, C_F, f | Savoldi, Cammi, Gajetti & Marocco (2026), Int. J. Heat Fluid Flow 121:110631 | 10.1016/j.ijheatfluidflow.2026.110631 | gyroid (skeletal/solid network) | pore-scale CFD (OpenFOAM v2212, single periodic cell, laminar) | 20-100 | 0.3-0.7 | not stated | True |
| savoldi2026_st1 | Nu (St) | Savoldi, Cammi, Gajetti & Marocco (2026), Int. J. Heat Fluid Flow 121:110631, Eq. 37 | 10.1016/j.ijheatfluidflow.2026.110631 | gyroid (skeletal), whole solid-fluid interface (interstitial h) | pore-scale CFD, uniform wall T and uniform wall q (non-conjugate), periodic cell | 20-100 | 0.3-0.7 | 1 | True |
| savoldi2026_st2 | Nu (St) | Savoldi, Cammi, Gajetti & Marocco (2026), Int. J. Heat Fluid Flow 121:110631, Eqs. 38-39 | 10.1016/j.ijheatfluidflow.2026.110631 | gyroid (skeletal), interstitial h | as Eq. 37 | 20-100 | 0.3-0.7 | 1 | True |
| savoldi2026_st3 | Nu (St) | Savoldi, Cammi, Gajetti & Marocco (2026), Int. J. Heat Fluid Flow 121:110631, Eq. 40 | 10.1016/j.ijheatfluidflow.2026.110631 | gyroid (skeletal), interstitial h | as Eq. 37 | 20-100 | 0.3-0.7 | 1 | True |
| rathore2023 | K, C_F (data) | Rathore, Mehta, Kumar & Asfer (2023), Transp. Porous Media 146:669-701 (arXiv:2205.03591) | 10.1007/s11242-022-01880-7 | diamond, gyroid, I-WP, primitive; 1 mm cells, 4 mm channel | pore-scale DNS (laminar) incl. channel walls | 0.01-100 | 0.32 | not stated | False |
| cheng2021 | Nu (interstitial), flow resistance | Cheng, Li, Xu & Jiang (2021), Int. Commun. Heat Mass Transf. 129:105713 | 10.1016/j.icheatmasstransfer.2021.105713 | I-WP, primitive, diamond, gyroid porous media | pore-scale CFD + strength tests | 10-129 | 0.2-0.8 | 0.7 | False |
| iyer2022 | Nu, f | Iyer, Moore, Nguyen, Roy & Stolaroff (2022), Appl. Therm. Eng. 209:118192 (details to verify) | 10.1016/j.applthermaleng.2022.118192 (to verify) | sheet TPMS and periodic nodal surfaces (two-fluid heat exchangers) | CFD | not stated | not stated | not stated | False |
| reynolds2023 | Nu | Reynolds, Fee, Morison & Holland (2023), Int. J. Heat Mass Transf. (as quoted by Brambati et al. 2024, Eq. 21) | (to verify) | gyroid heat exchanger | experiment | 100-2500 | not stated | 0.7-0.72 | True |
| brambati2024 | Nu | Brambati, Guilizzoni & Foletti (2024), Appl. Therm. Eng. 242:122492 | 10.1016/j.applthermaleng.2024.122492 | gyroid, Schwarz-P, Schwarz-D (sheet, two-fluid HX), porosity 70-90 % | conjugate RANS CFD (k-omega SST), Re 5000-50000 | 5000-50000 | 0.7-0.9 | 0.7-7 | False |
| piandoro2026 | Nu, f | Piandoro et al. (2026), Lasers Manuf. Mater. Process. 13:555-581 | 10.1007/s40516-026-00347-7 | gyroid (sheet/offset), porosity 4-83 % | CFD + L-PBF samples, fully developed duct | not stated | 0.04-0.83 | 14-154 | False |
| ergun1952 | f (packed bed) | Ergun (1952), Chem. Eng. Prog. 48:89-94 (assessed by Gajetti et al. 2025) | - | random packed beds | experiment | not stated | not stated | not stated | False |
| gnielinski_packed_bed | Nu (fluid-particle) | Gnielinski (1981) Int. Chem. Eng. 21(3); VDI Heat Atlas 2nd ed. (2010) | - | packed spheres (generic reference, not TPMS) | correlation of experiments | 0.1-1000 | not stated | 0.4-1000 | True |
| wakao_kaguei | Nu (fluid-particle) | Wakao & Kaguei (1982), Heat and Mass Transfer in Packed Beds, Gordon & Breach | - | packed spheres (generic reference, not TPMS) | correlation of experiments | 3-3000 | not stated | not stated | True |
| kuwahara2001 | Nu (interstitial) | Kuwahara, Shirota & Nakayama (2001), Int. J. Heat Mass Transf. 44:1153-1159 | 10.1016/S0017-9310(00)00166-6 (to verify) | regular square-rod arrays (not TPMS) | pore-scale CFD at Pr = 1 | not stated | 0.2-0.9 | 1 | False |
