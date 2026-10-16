# T02 - surrogate accuracy

Closure surrogate accuracy (R^2, MAPE, 95 % coverage) for the GP and deep ensemble on the cv5, band and sparse splits.

Sources: `results/task6_metrics.csv`

Medians over the targets of each group; MAPE only for positive quantities.

| model | split | group | n_targets | R2_median | R2_min | MAPE_median | MAPE_max | cov95_median |
|---|---|---|---|---|---|---|---|---|
| gp+cal | cv5 | K principal | 3 | 0.999 | 0.9986 | 0.008858 | 0.01072 | 0.9138 |
| gp+cal | cv5 | k_eff | 3 | 0.9951 | 0.9946 | 0.01905 | 0.01981 | 0.9552 |
| gp+cal | cv5 | C_eff | 21 | 0.8801 | 0.44 | 0.06345 | 0.06697 | 0.9621 |
| gp+cal | cv5 | E | 3 | 0.9884 | 0.9876 | 0.04709 | 0.05806 | 0.9517 |
| gp+cal | cv5 | a_sf L | 1 | 0.999 | 0.999 | 0.002561 | 0.002561 | 0.9483 |
| gp+cal | cv5 | localization p99 | 6 | 0.9853 | 0.9822 | 0.04787 | 0.05064 | 0.9414 |
| gp+cal | band | K principal | 3 | 0.9731 | 0.9717 | 0.06478 | 0.07265 | 0.3485 |
| gp+cal | band | k_eff | 3 | 0.9195 | 0.918 | 0.1145 | 0.1169 | 0.8561 |
| gp+cal | band | C_eff | 21 | -0.1737 | -1.856 | 0.4716 | 0.6116 | 0.6818 |
| gp+cal | band | E | 3 | 0.8317 | 0.8224 | 0.4282 | 0.4442 | 0.7576 |
| gp+cal | band | a_sf L | 1 | 0.9453 | 0.9453 | 0.02272 | 0.02272 | 0.4924 |
| gp+cal | band | localization p99 | 6 | 0.8687 | 0.8638 | 0.1598 | 0.1694 | 0.7197 |
| gp+cal | sparse | K principal | 3 | 0.9955 | 0.9929 | 0.02112 | 0.02551 | 0.8974 |
| gp+cal | sparse | k_eff | 3 | 0.982 | 0.9784 | 0.03743 | 0.04444 | 0.9487 |
| gp+cal | sparse | C_eff | 21 | 0.5088 | -13.38 | 0.1062 | 0.1327 | 0.9692 |
| gp+cal | sparse | E | 3 | 0.9667 | 0.9604 | 0.09552 | 0.1106 | 0.9538 |
| gp+cal | sparse | a_sf L | 1 | 0.9956 | 0.9956 | 0.005323 | 0.005323 | 0.8718 |
| gp+cal | sparse | localization p99 | 6 | 0.9749 | 0.9719 | 0.06862 | 0.07582 | 0.959 |
| ensemble+cal | cv5 | K principal | 3 | 0.9988 | 0.9983 | 0.00907 | 0.01075 | 0.9069 |
| ensemble+cal | cv5 | k_eff | 3 | 0.9954 | 0.9952 | 0.01452 | 0.01557 | 0.9448 |
| ensemble+cal | cv5 | C_eff | 21 | 0.9576 | 0.7601 | 0.03849 | 0.04325 | 0.9517 |
| ensemble+cal | cv5 | E | 3 | 0.9888 | 0.9886 | 0.04055 | 0.04132 | 0.9345 |
| ensemble+cal | cv5 | a_sf L | 1 | 0.9992 | 0.9992 | 0.002313 | 0.002313 | 0.9793 |
| ensemble+cal | cv5 | localization p99 | 6 | 0.9887 | 0.9861 | 0.03338 | 0.03569 | 0.9328 |
| ensemble+cal | band | K principal | 3 | 0.981 | 0.9692 | 0.0524 | 0.06845 | 0.8409 |
| ensemble+cal | band | k_eff | 3 | 0.9158 | 0.9128 | 0.1213 | 0.1292 | 0.8106 |
| ensemble+cal | band | C_eff | 21 | 0.2037 | -0.164 | 0.517 | 0.6782 | 0.7197 |
| ensemble+cal | band | E | 3 | 0.805 | 0.7977 | 0.4991 | 0.5018 | 0.7652 |
| ensemble+cal | band | a_sf L | 1 | 0.9366 | 0.9366 | 0.02374 | 0.02374 | 0.8636 |
| ensemble+cal | band | localization p99 | 6 | 0.8523 | 0.8281 | 0.1754 | 0.1881 | 0.7008 |
| ensemble+cal | sparse | K principal | 3 | 0.9936 | 0.9906 | 0.0241 | 0.02978 | 0.8103 |
| ensemble+cal | sparse | k_eff | 3 | 0.9827 | 0.9807 | 0.03332 | 0.03548 | 0.9333 |
| ensemble+cal | sparse | C_eff | 21 | 0.799 | 0.1605 | 0.1036 | 0.1195 | 0.9179 |
| ensemble+cal | sparse | E | 3 | 0.9623 | 0.9596 | 0.09918 | 0.09968 | 0.8974 |
| ensemble+cal | sparse | a_sf L | 1 | 0.9941 | 0.9941 | 0.006283 | 0.006283 | 0.8667 |
| ensemble+cal | sparse | localization p99 | 6 | 0.9736 | 0.9725 | 0.06235 | 0.06418 | 0.9051 |
