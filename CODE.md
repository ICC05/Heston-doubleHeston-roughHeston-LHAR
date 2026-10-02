# File guide

For the conceptual overview see [`README.md`](README.md).

## Estimation scripts (entry points)
| File | What it does |
|---|---|
| `estimate_heston_LHAR_new2.m` | Estimates the 1-factor Heston (κ, v̄, σᵥ, ρ) with LHAR: multistart CMA-ES, then pattern search, then out-of-sample check and noise floor; saves the results. |
| `estimate_double_heston_LHAR_new2.m` | Same pipeline for the double Heston (8 parameters). |
| `estimate_rough_heston_LHAR_new2.m` | Same pipeline for the rough Heston (λ, θ, ν, ρ, H). |

## Setup
| File | What it does |
|---|---|
| `setup_data_new.m` | For Heston and double Heston: loads `spy_data.mat`, builds RV and daily returns, estimates the empirical HAR/LHAR and the weighting matrix, generates replication seeds, starts the parallel pool. |
| `setup_data_rough_new.m` | Same for the rough Heston; additionally builds the geometric kernel grid and pre-tabulates the kernel weights over H. |

## Objective functions
| File | What it does |
|---|---|
| `objective_heston_LHAR_new.m` | Given θ: simulates the Heston replications, estimates LHAR on each, returns χ², mean simulated moments, per-moment contributions and t-stats. |
| `objective_double_heston_LHAR_new.m` | Same for the double Heston. |
| `objective_rough_heston_LHAR.m` | Same for the rough Heston. |

## Simulators
| File | What it does |
|---|---|
| `simulate_heston_new.m` | Simulates Heston intraday returns (exact-moment CIR scheme, full truncation). |
| `simulate_double_heston_new.m` | Simulates intraday returns with two independent CIR factors. |
| `simulate_rough_heston.m` | Simulates the rough Heston via the N-factor Markovian approximation, with burn-in and aggregation to 5 minutes. |

## Rough Heston kernel
| File | What it does |
|---|---|
| `compute_kernel_weights.m` | Computes the non-negative L²-optimal weights of the sum of exponentials approximating the fractional kernel. |
| `precompute_kernel_table.m` | Tabulates those weights over a grid of H and builds cubic (PCHIP) interpolants. |
| `interp_kernel_weights.m` | Returns the weights for any H by interpolating the table. |

## Auxiliary models
| File | What it does |
|---|---|
| `LHAR_estimate.m` | OLS estimation of the LHAR with Newey-West standard errors → 8 moments. |
| `HAR_estimate.m` | OLS estimation of the HAR-RV (only used as a diagnostic in the setup). |
| `aggregateAvg.m` | Backward-looking rolling average (weekly/monthly regressors). |
| `nwest.m` | OLS regression with Newey-West covariance matrix. |

## Optimizers
| File | What it does |
|---|---|
| `cmaes_bnd_new.m` | Box-constrained CMA-ES with a per-generation relative stopping rule. |
| `pattern_search_bnd_new.m` | Bound-constrained pattern search with Latin Hypercube polls and plateau stopping. |

## Output and logging
| File | What it does |
|---|---|
| `print_estimation_results_new.m` | Prints and saves the results table (Heston, double Heston). |
| `print_estimation_results.m` | Prints and saves the results table (rough Heston). |
| `dual_log_new.m` | Writes messages both to screen and to the logfile (Heston/double Heston, optimizers). |
| `dual_log.m` | Same, version used by the rough Heston scripts. |

## Data and results
| File | Content |
|---|---|
| `spy_data.mat` | Cleaned SPY 5-minute returns matrix (days × intervals). |
| `results_*_LHAR_new2.mat / .txt` | Estimation results. |
| `logfile_*_LHAR_new2.txt` | Estimation logs. |
