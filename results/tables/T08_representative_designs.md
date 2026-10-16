# T08 - representative designs

Representative designs per layout (R, R_active, pump power, mass, margin, field ranges), including the manufacturable slot optimum best_at_cap_gyroid.

Sources: `results/task11_representative.csv`, `results/designs/task12_slots_best_at_cap_gyroid.json`

Graded-field statistics of best_at_cap_gyroid equal those of best_at_cap (same rho*/L nets).

| layout | design | R_mK_W | R_active_mK_W | pump_power_W | delta_p_Pa | mass_kg | min_margin | T_wall_max_C | rho_min | rho_max | rho_area_mean | w_min | w_max | w_area_mean | critical_cells_ok_n48 | source_run | L_min_mm | L_max_mm | L_area_mean_mm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| slots | min_R | 20.57 | 12.94 | 0.8515 | 5109 | 0.3581 | 12.49 | 65.84 | 0.2113 | 0.496 | 0.3916 | 0.8552 | 0.9969 | 0.9555 | True | nsga2_s0 | 2.183 | 4.671 | 2.901 |
| slots | best_at_cap | 21.94 | 14.37 | 0.3039 | 1823 | 0.3096 | 8.743 | 66.89 | 0.2014 | 0.4149 | 0.2532 | 0.008081 | 0.2354 | 0.07389 | True | nsga2_s0 | 2.188 | 5.008 | 3.313 |
| slots | knee | 23.36 | 15.76 | 0.1541 | 924.4 | 0.3026 | 8.904 | 67.99 | 0.2007 | 0.3873 | 0.2334 | 0.003098 | 0.1777 | 0.05751 | True | nsga2_s0 | 3.303 | 5.961 | 4.958 |
| slots | min_mass | 23.64 | 16.09 | 0.4623 | 2774 | 0.2909 | 12.63 | 68.2 | 0.2 | 0.2 | 0.2 | 1 | 1 | 1 | True | nsga2_s0 | 2 | 2 | 2 |
| end_rings | min_R | 8.573 | 8.573 | 0.06487 | 389.2 | 0.4082 | 47.65 | 56.6 | 0.5 | 0.5 | 0.5 | 1 | 1 | 1 | True | nsga2_s0 | 2.187 | 2.187 | 2.187 |
| end_rings | knee | 11.93 | 11.93 | 0.005589 | 33.53 | 0.304 | 9.011 | 59.19 | 0.2017 | 0.3303 | 0.2222 | 0.004115 | 0.1689 | 0.05703 | True | nsga2_s3 | 2.179 | 5.141 | 3.496 |
| end_rings | min_mass | 10.54 | 10.54 | 0.01641 | 98.43 | 0.2957 | 13.32 | 58.11 | 0.2 | 0.2 | 0.2 | 1 | 1 | 1 | True | nsga2_s0 | 2 | 2 | 2 |
| slots | best_at_cap_gyroid (manufacturable, exported) | 21.92 | 14.41 | 0.3076 | 1846 | 0.3096 | 9.849 | 66.88 | 0.2014 | 0.4149 | 0.2532 | 0 | 0 | 0 | True | Task 12 (w = 0) | 2.188 | 5.008 | 3.313 |
