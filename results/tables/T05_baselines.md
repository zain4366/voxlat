# T05 - baselines

Baselines B1, B1s, B2, B3 at 1, 3 and 5 L/min: R, R_active, pressure drop, pump power, mass, margin.

Sources: `results/task10_comparison.csv`, `results/baselines/*.json`

| family | flow_lpm | R_mK_W | R_active_mK_W | delta_p_Pa | pump_power_W | mass_kg | min_margin | label |
|---|---|---|---|---|---|---|---|---|
| B1 | 1 | 25.94 | 25.94 | 491.9 | 0.02733 | 0.3332 | 9.255 | B1 helical channels (end-ring manifolds) |
| B1 | 3 | 13.26 | 13.26 | 1720 | 0.2867 | 0.3332 | 9.244 | B1 helical channels (end-ring manifolds) |
| B1 | 5 | 10.12 | 10.12 | 3229 | 0.8968 | 0.3332 | 9.231 | B1 helical channels (end-ring manifolds) |
| B1s | 1 | 37.53 | 30.48 | 347.1 | 0.01928 | 0.319 | 4.247 | B1s circumferential channels (lattice slots) |
| B1s | 3 | 25.67 | 18.27 | 1088 | 0.1814 | 0.319 | 4.246 | B1s circumferential channels (lattice slots) |
| B1s | 5 | 22.62 | 15.08 | 1885 | 0.5236 | 0.319 | 4.244 | B1s circumferential channels (lattice slots) |
| B2 | 1 | 38.08 | 30.8 | 357.4 | 0.01986 | 0.3435 | 12.79 | B2 best uniform gyroid |
| B2 | 3 | 24.37 | 16.81 | 1833 | 0.3055 | 0.3435 | 12.69 | B2 best uniform gyroid |
| B2 | 5 | 20.75 | 13.04 | 4324 | 1.201 | 0.3435 | 12.54 | B2 best uniform gyroid |
| B3 | 1 | 35.88 | 28.43 | 354.2 | 0.01968 | 0.3349 | 12.36 | B3 density-graded gyroid |
| B3 | 3 | 22.67 | 14.97 | 1827 | 0.3045 | 0.3349 | 12.28 | B3 density-graded gyroid |
| B3 | 5 | 19.33 | 11.5 | 4319 | 1.2 | 0.3349 | 12.15 | B3 density-graded gyroid |
