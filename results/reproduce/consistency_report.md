# Consistency audit: figure/table data vs STATUS.md

61 checks, 59 ok, 2 flagged.

## Flagged

| id | items | claim (STATUS) | expected | actual | note |
|---|---|---|---|---|---|
| t11_B2_cap_eq_t10 | T07/F34 | Task 10 B2 (rho* 0.35, L 4 mm) R = 24.37 mK/W at 0.3055 W is the best uniform gyroid at the 0.306 W cap; Task 11 uses the same cap for its B2 baseline | 24.37 | 24.4636 | Task 11 cap 0.3055 W is below the reference design's 0.305541 W, so Task 11's B2 baseline is a different uniform gyroid (24.46 mK/W at 0.293 W); gains vs B2 are overstated by ~0.3 pp |
| t11_cap_gyr_P | T07 | Task 12: best_at_cap_gyroid stays within the 0.3055 W selection cap | 0.3055 | 0.307588 | 0.3076 W: 0.7 % above the cap (STATUS Task 12 finding 1 says so); T07 compares at matched pump power |

## All checks

| id | items | status | expected | actual |
|---|---|---|---|---|
| t1_asf_gyroid | F03 | ok | 2.97 | 2.969595 |
| t2_keff_G035 | F06 | ok | 0.208 | 0.207913 |
| t2_keff_D050 | F06 | ok | 0.336 | 0.336193 |
| t2_keff_B030 | F06 | ok | 0.097 | 0.096774 |
| t2_R3264_max | F07/T01 | ok | 0.7 | 0.692393 |
| t2_raw32_max | F07/T01 | ok | 17.0 | 17.372061 |
| t3_E_G035 | F08 | ok | 0.0998 | 0.099822 |
| t3_E_D050 | F08 | ok | 0.1789 | 0.1789 |
| t3_fit_G_m | F08 | ok | 2.13 | 2.125898 |
| t3_fit_D_m | F08 | ok | 1.93 | 1.933833 |
| t3_zener_D020 | F09 | ok | 2.29 | 2.285178 |
| t3_loc_G035 | F10 | ok | 15.0 | 15.359665 |
| t3_loc_B020 | F10 | ok | 95.0 | 95.350748 |
| t4_K_G035 | F14 | ok | 0.004887 | 0.004887 |
| t4_K_D020 | F14 | ok | 0.006651 | 0.006651 |
| t4_tort_D050 | F14 | ok | 1.29 | 1.28896 |
| t4_R3264_max | F13/T01 | ok | 1.0 | 1.021821 |
| t6_cnn_band_E | F19 | ok | 107.2 | 107.205 |
| t6_cnn_cv5_K_gp | F19 | ok | 0.7 | 0.652896 |
| t6_asf_cv5_R2 | F16/T02 | ok | 0.999 | 0.998996 |
| t7_KtG035_N1 | F20/T03 | ok | -26.6 | -26.579617 |
| t7_KtD045_N1 | F20/T03 | ok | -18.7 | -18.742002 |
| t7_CnnB035_N1 | F20/T03 | ok | 86.0 | 86.331654 |
| t7_beta_Kt | F20/T03 | ok | 0.2663 | 0.266328 |
| t9_R_ref | F27/F29 | ok | 24.37 | 24.368912 |
| t9_dp_ref | F27/F29 | ok | 1833.0 | 1833.24479 |
| t9_P_ref | F29 | ok | 0.306 | 0.305541 |
| t9_grid_default | F28 | ok | 24.37 | 24.368912 |
| t9_sweep_D3 | F29 | ok | 24.3 | 24.281541 |
| t10_B1_R3 | F32/T05 | ok | 13.3 | 13.26013 |
| t10_B1s_R3 | F32/T05 | ok | 25.7 | 25.666818 |
| t10_B3_R3 | F32/T05 | ok | 22.7 | 22.670931 |
| t10_B2_R3 | F32/T05 | ok | 24.4 | 24.368912 |
| t10_B2_mass | F32/T05 | ok | 0.344 | 0.343531 |
| t11_hv_slots_q200 | F33/T06 | ok | 0.1998 | 0.199846 |
| t11_hv_slots_nsga_final | F33/T06 | ok | 0.2093 | 0.209252 |
| t11_hv_er_q200 | F33/T06 | ok | 0.6748 | 0.674774 |
| t11_evals_1368_slots | F33/T06 | ok | 1368.0 | 1368.0 |
| t11_evals_1368_er | F33/T06 | ok | 1368.0 | 1368.0 |
| t11_cap_vs_B1s | F34/T07 | ok | 14.5 | 14.5175 |
| t11_cap_vs_B2 | F34/T07 | ok | 10.3 | 10.3132 |
| t11_cap_vs_B3 | F34/T07 | ok | 3.2 | 3.22129 |
| t11_cap_er_vs_B1 | F34/T07 | ok | 35.3 | 35.3497 |
| t11_rep_bac_R | F35/T08 | ok | 21.94 | 21.9406 |
| t11_rep_bac_mass | F35/T08 | ok | 0.31 | 0.309562 |
| t11_rep_er_minR | F36/T08 | ok | 8.57 | 8.57271 |
| t11_rep_er_knee_P | F36/T08 | ok | 0.0056 | 0.005589 |
| t11_B2_cap_eq_t10 | T07/F34 | FLAG | 24.37 | 24.4636 |
| t11_cap_gyr_P | T07 | FLAG | 0.3055 | 0.307588 |
| t6_cv5_Kprinc_cov | F16/T02 | ok | 93.0 | 93.103433 |
| t12_full_vol | T09 | ok | 136345.0 | 136345.14 |
| t12_gyr_R | T07/T08 | ok | 21.92 | 21.921049 |
| t12_gyr_P | T07/T08 | ok | 0.3076 | 0.307588 |
| t12_gyr_margin | T07/T08 | ok | 9.8 | 9.848664 |
| t12_iou | T09/F37 | ok | 0.99676 | 0.99676 |
| t12_minwall | T09 | ok | 0.482 | 0.482031 |
| t12_throat | T09 | ok | 1.07 | 1.066406 |
| t12_full_throat | T09 | ok | 1.22 | 1.21875 |
| t12_full_tri | T09 | ok | 15.0 | 15.036516 |
| t12_bac_minwall | F39/T09 | ok | 0.306 | 0.30625 |
| t12_erminR_throat | T09 | ok | 0.785 | 0.785156 |
