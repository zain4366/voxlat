# T06 - rq3 hypervolume

Hypervolume (median, IQR over 5 seeds) at 50 / 100 / 200 evaluations and final budget; evaluations NSGA-II needs to match qNEHVI at 200.

Sources: `results/task11_hv_summary.csv`, `results/task11_hv_curves.csv`

| layout | method | seeds | hv_50_median | hv_50_iqr | hv_100_median | hv_100_iqr | hv_200_median | hv_200_iqr | n_evals_final_median | hv_final_median | evals_to_reach_qnehvi_hv200_median | seeds_reaching |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| end_rings | nsga2 | 5 | 0.6513 | 0.0008962 | 0.6517 | 0.0004749 | 0.6555 | 0.001121 | 10000 | 0.6875 | 1368 | 5 |
| end_rings | qnehvi | 5 | 0.653 | 0.002417 | 0.6662 | 0.001256 | 0.6748 | 0.002679 | 200 | 0.6748 | 200 | 3 |
| end_rings | sobol | 5 | 0.6511 | 0.0003109 | 0.6519 | 0.0003568 | 0.6525 | 0.0001363 | 919 | 0.6552 |  | 0 |
| slots | nsga2 | 5 | 0.1757 | 0.0002452 | 0.1767 | 0.0008035 | 0.1819 | 0.0009228 | 10000 | 0.2093 | 1368 | 5 |
| slots | qnehvi | 5 | 0.1851 | 0.003461 | 0.1939 | 0.001034 | 0.1998 | 0.0008188 | 200 | 0.1998 | 200 | 3 |
| slots | sobol | 5 | 0.1767 | 0.002404 | 0.1768 | 0.002274 | 0.1784 | 0.0006835 | 919 | 0.1798 |  | 0 |
