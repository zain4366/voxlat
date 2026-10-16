# T07 - rq2 equal pump power

RQ2 at equal pump power (3 L/min): optimized R vs the best baseline of each family below the pump-power cap; slots row at the 0.306 W cap uses the manufacturable optimum task12_slots_best_at_cap_gyroid.

Sources: `results/task11_equal_pump_power.csv`, `results/designs/task12_slots_best_at_cap_gyroid.json`, `results/task10_fronts.csv`, `results/task11_uniform_refs.csv`

R_change_pct < 0 = optimized design better. Rows 'best_at_cap_gyroid' use the manufacturable slot optimum `results/designs/task12_slots_best_at_cap_gyroid.json` (w = 0; P = 0.3076 W, i.e. +0.7 % above the 0.3055 W cap), against the baselines below the 0.3055 W cap and below its own pump power; baseline families from results/task10_fronts.csv (B1, B1s, B2, B3) and results/task11_uniform_refs.csv (U-G, U-D, slots layout, corrected cell-size rule).

| layout | design | cap_W | family | R_baseline_mK_W | P_baseline_W | R_opt_mK_W | P_opt_W | mass_opt_kg | R_change_pct |
|---|---|---|---|---|---|---|---|---|---|
| slots | Task 11 optimum (archive) | 0.1 | B1s | 28.89 | 0.08446 | 26.56 | 0.0991 | 0.2992 | -8.069 |
| slots | Task 11 optimum (archive) | 0.1 | B2 | 28.1 | 0.0997 | 26.56 | 0.0991 | 0.2992 | -5.476 |
| slots | Task 11 optimum (archive) | 0.1 | U-G | 28.1 | 0.0997 | 26.56 | 0.0991 | 0.2992 | -5.476 |
| slots | Task 11 optimum (archive) | 0.1 | U-D | 28.9 | 0.09904 | 26.56 | 0.0991 | 0.2992 | -8.101 |
| slots | Task 11 optimum (archive) | 0.1 | B1 | 18.43 | 0.09683 | 26.56 | 0.0991 | 0.2992 | 44.1 |
| slots | Task 11 optimum (archive) | 0.25 | B1s | 25.67 | 0.1814 | 22.37 | 0.2454 | 0.3116 | -12.85 |
| slots | Task 11 optimum (archive) | 0.25 | B2 | 24.95 | 0.2492 | 22.37 | 0.2454 | 0.3116 | -10.34 |
| slots | Task 11 optimum (archive) | 0.25 | B3 | 23.81 | 0.2241 | 22.37 | 0.2454 | 0.3116 | -6.072 |
| slots | Task 11 optimum (archive) | 0.25 | U-G | 24.95 | 0.2492 | 22.37 | 0.2454 | 0.3116 | -10.34 |
| slots | Task 11 optimum (archive) | 0.25 | U-D | 25.28 | 0.2483 | 22.37 | 0.2454 | 0.3116 | -11.52 |
| slots | Task 11 optimum (archive) | 0.25 | B1 | 14.33 | 0.2006 | 22.37 | 0.2454 | 0.3116 | 56.13 |
| slots | Task 11 optimum (archive) | 0.3055 | B1s | 25.67 | 0.1814 | 21.94 | 0.3039 | 0.3096 | -14.52 |
| slots | Task 11 optimum (archive) | 0.3055 | B2 | 24.46 | 0.2931 | 21.94 | 0.3039 | 0.3096 | -10.31 |
| slots | Task 11 optimum (archive) | 0.3055 | B3 | 22.67 | 0.3045 | 21.94 | 0.3039 | 0.3096 | -3.221 |
| slots | Task 11 optimum (archive) | 0.3055 | U-G | 24.46 | 0.2931 | 21.94 | 0.3039 | 0.3096 | -10.31 |
| slots | Task 11 optimum (archive) | 0.3055 | U-D | 24.65 | 0.2993 | 21.94 | 0.3039 | 0.3096 | -10.99 |
| slots | Task 11 optimum (archive) | 0.3055 | B1 | 13.26 | 0.2867 | 21.94 | 0.3039 | 0.3096 | 65.46 |
| slots | Task 11 optimum (archive) | 0.5 | B1s | 25.67 | 0.1814 | 20.97 | 0.4949 | 0.3234 | -18.3 |
| slots | Task 11 optimum (archive) | 0.5 | B2 | 23.16 | 0.4869 | 20.97 | 0.4949 | 0.3234 | -9.447 |
| slots | Task 11 optimum (archive) | 0.5 | B3 | 22.67 | 0.3045 | 20.97 | 0.4949 | 0.3234 | -7.502 |
| slots | Task 11 optimum (archive) | 0.5 | U-G | 23.16 | 0.4869 | 20.97 | 0.4949 | 0.3234 | -9.447 |
| slots | Task 11 optimum (archive) | 0.5 | U-D | 23.35 | 0.4919 | 20.97 | 0.4949 | 0.3234 | -10.18 |
| slots | Task 11 optimum (archive) | 0.5 | B1 | 12.43 | 0.4351 | 20.97 | 0.4949 | 0.3234 | 68.74 |
| slots | Task 11 optimum (archive) | 1 | B1s | 25.67 | 0.1814 | 20.57 | 0.8515 | 0.3581 | -19.84 |
| slots | Task 11 optimum (archive) | 1 | B2 | 21.65 | 0.9946 | 20.57 | 0.8515 | 0.3581 | -4.95 |
| slots | Task 11 optimum (archive) | 1 | B3 | 22.67 | 0.3045 | 20.57 | 0.8515 | 0.3581 | -9.249 |
| slots | Task 11 optimum (archive) | 1 | U-G | 21.65 | 0.9946 | 20.57 | 0.8515 | 0.3581 | -4.95 |
| slots | Task 11 optimum (archive) | 1 | U-D | 21.69 | 0.9737 | 20.57 | 0.8515 | 0.3581 | -5.163 |
| slots | Task 11 optimum (archive) | 1 | B1 | 10.33 | 0.8173 | 20.57 | 0.8515 | 0.3581 | 99.09 |
| slots | Task 11 optimum (archive) | 2 | B1s | 25.67 | 0.1814 | 20.57 | 0.8515 | 0.3581 | -19.84 |
| slots | Task 11 optimum (archive) | 2 | B2 | 20.72 | 1.811 | 20.57 | 0.8515 | 0.3581 | -0.6806 |
| slots | Task 11 optimum (archive) | 2 | B3 | 22.67 | 0.3045 | 20.57 | 0.8515 | 0.3581 | -9.249 |
| slots | Task 11 optimum (archive) | 2 | U-G | 20.72 | 1.811 | 20.57 | 0.8515 | 0.3581 | -0.6806 |
| slots | Task 11 optimum (archive) | 2 | U-D | 20.77 | 1.801 | 20.57 | 0.8515 | 0.3581 | -0.9466 |
| slots | Task 11 optimum (archive) | 2 | B1 | 9.937 | 1.921 | 20.57 | 0.8515 | 0.3581 | 107.1 |
| end_rings | Task 11 optimum (archive) | 0.01 | U-G | 11.62 | 0.009748 | 10.36 | 0.009925 | 0.3249 | -10.79 |
| end_rings | Task 11 optimum (archive) | 0.01 | U-D | 11.66 | 0.009976 | 10.36 | 0.009925 | 0.3249 | -11.13 |
| end_rings | Task 11 optimum (archive) | 0.01 | B1s | 52.23 | 0.009759 | 10.36 | 0.009925 | 0.3249 | -80.16 |
| end_rings | Task 11 optimum (archive) | 0.03 | U-G | 9.513 | 0.02747 | 9.041 | 0.02874 | 0.3461 | -4.956 |
| end_rings | Task 11 optimum (archive) | 0.03 | U-D | 9.419 | 0.02993 | 9.041 | 0.02874 | 0.3461 | -4.004 |
| end_rings | Task 11 optimum (archive) | 0.03 | B1s | 36.86 | 0.02691 | 9.041 | 0.02874 | 0.3461 | -75.47 |
| end_rings | Task 11 optimum (archive) | 0.1 | B1 | 18.43 | 0.09683 | 8.573 | 0.06487 | 0.4082 | -53.49 |
| end_rings | Task 11 optimum (archive) | 0.1 | U-G | 8.811 | 0.05359 | 8.573 | 0.06487 | 0.4082 | -2.707 |
| end_rings | Task 11 optimum (archive) | 0.1 | U-D | 8.634 | 0.06158 | 8.573 | 0.06487 | 0.4082 | -0.7085 |
| end_rings | Task 11 optimum (archive) | 0.1 | B1s | 28.89 | 0.08446 | 8.573 | 0.06487 | 0.4082 | -70.33 |
| end_rings | Task 11 optimum (archive) | 0.1 | B2 | 28.1 | 0.0997 | 8.573 | 0.06487 | 0.4082 | -69.49 |
| end_rings | Task 11 optimum (archive) | 0.25 | B1 | 14.33 | 0.2006 | 8.573 | 0.06487 | 0.4082 | -40.16 |
| end_rings | Task 11 optimum (archive) | 0.25 | U-G | 8.811 | 0.05359 | 8.573 | 0.06487 | 0.4082 | -2.707 |
| end_rings | Task 11 optimum (archive) | 0.25 | U-D | 8.634 | 0.06158 | 8.573 | 0.06487 | 0.4082 | -0.7085 |
| end_rings | Task 11 optimum (archive) | 0.25 | B1s | 25.67 | 0.1814 | 8.573 | 0.06487 | 0.4082 | -66.6 |
| end_rings | Task 11 optimum (archive) | 0.25 | B2 | 24.95 | 0.2492 | 8.573 | 0.06487 | 0.4082 | -65.64 |
| end_rings | Task 11 optimum (archive) | 0.25 | B3 | 23.81 | 0.2241 | 8.573 | 0.06487 | 0.4082 | -64 |
| end_rings | Task 11 optimum (archive) | 0.3055 | B1 | 13.26 | 0.2867 | 8.573 | 0.06487 | 0.4082 | -35.35 |
| end_rings | Task 11 optimum (archive) | 0.3055 | U-G | 8.811 | 0.05359 | 8.573 | 0.06487 | 0.4082 | -2.707 |
| end_rings | Task 11 optimum (archive) | 0.3055 | U-D | 8.634 | 0.06158 | 8.573 | 0.06487 | 0.4082 | -0.7085 |
| end_rings | Task 11 optimum (archive) | 0.3055 | B1s | 25.67 | 0.1814 | 8.573 | 0.06487 | 0.4082 | -66.6 |
| end_rings | Task 11 optimum (archive) | 0.3055 | B2 | 24.46 | 0.2931 | 8.573 | 0.06487 | 0.4082 | -64.96 |
| end_rings | Task 11 optimum (archive) | 0.3055 | B3 | 22.67 | 0.3045 | 8.573 | 0.06487 | 0.4082 | -62.19 |
| end_rings | Task 11 optimum (archive) | 0.5 | B1 | 12.43 | 0.4351 | 8.573 | 0.06487 | 0.4082 | -31.02 |
| end_rings | Task 11 optimum (archive) | 0.5 | U-G | 8.811 | 0.05359 | 8.573 | 0.06487 | 0.4082 | -2.707 |
| end_rings | Task 11 optimum (archive) | 0.5 | U-D | 8.634 | 0.06158 | 8.573 | 0.06487 | 0.4082 | -0.7085 |
| end_rings | Task 11 optimum (archive) | 0.5 | B1s | 25.67 | 0.1814 | 8.573 | 0.06487 | 0.4082 | -66.6 |
| end_rings | Task 11 optimum (archive) | 0.5 | B2 | 23.16 | 0.4869 | 8.573 | 0.06487 | 0.4082 | -62.98 |
| end_rings | Task 11 optimum (archive) | 0.5 | B3 | 22.67 | 0.3045 | 8.573 | 0.06487 | 0.4082 | -62.19 |
| end_rings | Task 11 optimum (archive) | 1 | B1 | 10.33 | 0.8173 | 8.573 | 0.06487 | 0.4082 | -17.04 |
| end_rings | Task 11 optimum (archive) | 1 | U-G | 8.811 | 0.05359 | 8.573 | 0.06487 | 0.4082 | -2.707 |
| end_rings | Task 11 optimum (archive) | 1 | U-D | 8.634 | 0.06158 | 8.573 | 0.06487 | 0.4082 | -0.7085 |
| end_rings | Task 11 optimum (archive) | 1 | B1s | 25.67 | 0.1814 | 8.573 | 0.06487 | 0.4082 | -66.6 |
| end_rings | Task 11 optimum (archive) | 1 | B2 | 21.65 | 0.9946 | 8.573 | 0.06487 | 0.4082 | -60.4 |
| end_rings | Task 11 optimum (archive) | 1 | B3 | 22.67 | 0.3045 | 8.573 | 0.06487 | 0.4082 | -62.19 |
| end_rings | Task 11 optimum (archive) | 2 | B1 | 9.937 | 1.921 | 8.573 | 0.06487 | 0.4082 | -13.73 |
| end_rings | Task 11 optimum (archive) | 2 | U-G | 8.811 | 0.05359 | 8.573 | 0.06487 | 0.4082 | -2.707 |
| end_rings | Task 11 optimum (archive) | 2 | U-D | 8.634 | 0.06158 | 8.573 | 0.06487 | 0.4082 | -0.7085 |
| end_rings | Task 11 optimum (archive) | 2 | B1s | 25.67 | 0.1814 | 8.573 | 0.06487 | 0.4082 | -66.6 |
| end_rings | Task 11 optimum (archive) | 2 | B2 | 20.72 | 1.811 | 8.573 | 0.06487 | 0.4082 | -58.62 |
| end_rings | Task 11 optimum (archive) | 2 | B3 | 22.67 | 0.3045 | 8.573 | 0.06487 | 0.4082 | -62.19 |
| slots | best_at_cap_gyroid, 0.306 W cap (Task 11 rule) | 0.3055 | B1s | 25.67 | 0.1814 | 21.92 | 0.3076 | 0.3096 | -14.59 |
| slots | best_at_cap_gyroid, 0.306 W cap (Task 11 rule) | 0.3055 | B2 | 24.46 | 0.2931 | 21.92 | 0.3076 | 0.3096 | -10.39 |
| slots | best_at_cap_gyroid, 0.306 W cap (Task 11 rule) | 0.3055 | B3 | 22.67 | 0.3045 | 21.92 | 0.3076 | 0.3096 | -3.308 |
| slots | best_at_cap_gyroid, 0.306 W cap (Task 11 rule) | 0.3055 | U-G | 24.46 | 0.2931 | 21.92 | 0.3076 | 0.3096 | -10.39 |
| slots | best_at_cap_gyroid, 0.306 W cap (Task 11 rule) | 0.3055 | U-D | 24.65 | 0.2993 | 21.92 | 0.3076 | 0.3096 | -11.07 |
| slots | best_at_cap_gyroid, 0.306 W cap (Task 11 rule) | 0.3055 | B1 | 13.26 | 0.2867 | 21.92 | 0.3076 | 0.3096 | 65.32 |
| slots | best_at_cap_gyroid, matched: cap = P of the gyroid optimum | 0.3076 | B1s | 25.67 | 0.1814 | 21.92 | 0.3076 | 0.3096 | -14.59 |
| slots | best_at_cap_gyroid, matched: cap = P of the gyroid optimum | 0.3076 | B2 | 24.37 | 0.3055 | 21.92 | 0.3076 | 0.3096 | -10.05 |
| slots | best_at_cap_gyroid, matched: cap = P of the gyroid optimum | 0.3076 | B3 | 22.67 | 0.3045 | 21.92 | 0.3076 | 0.3096 | -3.308 |
| slots | best_at_cap_gyroid, matched: cap = P of the gyroid optimum | 0.3076 | U-G | 24.37 | 0.3055 | 21.92 | 0.3076 | 0.3096 | -10.05 |
| slots | best_at_cap_gyroid, matched: cap = P of the gyroid optimum | 0.3076 | U-D | 24.65 | 0.2993 | 21.92 | 0.3076 | 0.3096 | -11.07 |
| slots | best_at_cap_gyroid, matched: cap = P of the gyroid optimum | 0.3076 | B1 | 13.26 | 0.2867 | 21.92 | 0.3076 | 0.3096 | 65.32 |
