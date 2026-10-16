# T09 - export checks

Manufacturability checks of the exported geometry (numpy mirror and PicoGK laptop builds): min wall, throat, connectivity, pockets, loose solid, volume fraction, PicoGK-vs-numpy IoU.

Sources: `results/export/*_checks.json`

Empty cells: quantity not recorded in that report (see the JSON). STATUS Task 12 holds the laptop numbers in prose; this table is generated from the JSON reports only.

| design | geometry | source | voxel_mm | min_wall_mm | sliver_min_mm | throat_mm | connected | pockets | voxel_scale_pockets | loose_solid | all_pass | vf_err_vs_design_pct | iou_vs_numpy | picogk_total_s | triangles | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| end_rings_knee | 30 deg sector | numpy mirror | 0.07 | 0.3805 | 0.3602 | 1.441 | True | 0 | 0 | 0 | True | 0.02693 |  |  |  | results/export/end_rings_knee_sector30_numpy_checks.json |
| end_rings_min_R | 30 deg sector | numpy mirror | 0.07 | 0.6648 | 0.3602 | 0.7852 | True | 0 | 0 | 0 | True | 0.0003631 |  |  |  | results/export/end_rings_min_R_sector30_numpy_checks.json |
| slots_best_at_cap | 30 deg sector | numpy mirror | 0.06 | 0.3062 |  | 1.547 | True | 0 | 2 | 0 | False | 0.7666 |  |  |  | results/export/slots_best_at_cap_sector30_numpy_checks.json |
| task12_slots_best_at_cap_gyroid | full jacket | PicoGK (laptop) | 0.2 |  | 0.2625 | 1.219 | True | 0 | 37 | 0 | True |  |  | 144.3 | 1.504e+07 | results/export/task12_slots_best_at_cap_gyroid_full_checks.json |
| task12_slots_best_at_cap_gyroid | 30 deg sector | PicoGK (laptop) | 0.05 | 0.482 | 0.3281 | 1.066 | True | 0 | 0 | 0 | True | -0.06506 | 0.9968 | 858.7 | 2.033e+07 | results/export/task12_slots_best_at_cap_gyroid_sector30_checks.json |
| task12_slots_best_at_cap_gyroid | 30 deg sector | numpy mirror | 0.06 | 0.5023 | 0.3602 | 1.031 | True | 0 | 1 | 0 | True | 0.0987 |  |  |  | results/export/task12_slots_best_at_cap_gyroid_sector30_numpy_checks.json |
