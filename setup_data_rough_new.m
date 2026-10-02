%% ========================================================================
%  SETUP_DATA_ROUGH_NEW
%
%  Drop-in replacement for setup_data_rough.m used by the *_new estimation
%  pipeline. Differences from the original:
%    - cfg.cmaes.lambda_pop raised from 8 (Hansen default for d=5) to 20.
%      The objective is a noisy Monte-Carlo simulation; with lambda=8 we get
%      mu_sel=4 and mu_eff=2.60, which is too low to average out simulation
%      noise across elite candidates and is what produced the legacy LHAR
%      restarts 2/3/5 collapsing to local minima at gen 19-25. Lambda=20
%      gives mu_sel=10, mu_eff~5.5, restoring the diversity needed for a
%      noisy 5-d landscape.
%    - cfg.n_local_starts reduced from 5 to 3 (legacy run showed all 5 PS
%      starts converging to chi^2 within 13% of each other, so the marginal
%      gain from starts 4-5 is dominated by their compute cost; 3 starts
%      remain sufficient to detect basin disagreement).
%    - cfg.cmaes.tolfun_rel added (relative TolFun, default 1e-4): combined
%      with the absolute tolfun via max(tolfun, tolfun_rel*|fbest|) inside
%      cmaes_bnd_new.m, applied to PER-GENERATION best (not best-ever, which
%      is monotone and would fire the criterion as soon as no new best is
%      found for 10 gens regardless of the threshold).
%    - cfg.ps.rel_stall_tol and cfg.ps.stall_iters added: early stopping
%      rule for pattern_search_bnd_new.m when the best-so-far stops
%      improving by more than rel_stall_tol over stall_iters consecutive
%      iterations.
%  ========================================================================

fprintf('=== Loading data and preparing rough Heston environment (NEW) ===\n\n');

%% Configuration aligned with Section 7.2 of the thesis
cfg.seed       = 301074;
cfg.annualize  = 252;

% --- Simulation grid (unchanged)
cfg.M_fine     = 400;
cfg.agg_factor = 5;
cfg.ndays_eval = 6000;
cfg.ndays_burn = 1000;
cfg.ndays_total = cfg.ndays_eval + cfg.ndays_burn;
cfg.n_indep    = 247;  % 247 indep + 247 antithetic = 494 paths; 2 tasks/worker on a 247-worker pool (caramel default). Raised from 127 on 2026-05-22 to cut MC SE by ~30% at the same wall time as the legacy 127-worker setup.

% --- Kernel approximation (unchanged)
cfg.N_factors  = 20;
cfg.eta_N      = 0.1;
cfg.gamma_max  = 1e2;
cfg.T_kernel   = 44;
cfg.H_grid     = 0.02:0.005:0.49;  % Extended down to 0.02 to cover the relaxed LHAR LB(H) = 0.02. The kernel interpolant uses 'nearest' extrapolation, so the H_grid lower bound must track LB(H) — otherwise H values below the grid collapse onto the same kernel weights and the relaxed bound has no effect.

% --- Optimization: CMA-ES (NEW: relative tolfun, larger lambda for noise)
cfg.cmaes.lambda_pop = 20;         % NEW: was 8 (Hansen default). Raised for noisy simulation objective; gives mu_sel=10, mu_eff~5.5.
cfg.cmaes.max_gen    = 120;        % NEW: was 200. Reduced because LHS init + lambda=20 + per-gen TolFun converge in fewer gens; legacy good restart used 114 gens with lambda=8.
cfg.cmaes.tolfun     = 1e-6;       % absolute TolFun
cfg.cmaes.tolfun_rel = 1e-4;       % relative TolFun multiplier on |fbest| (applied to per-gen best, see cmaes_bnd_new.m)
cfg.cmaes.tolx       = 1e-7;

% --- Optimization: Pattern Search (NEW: early-stop on relative plateau)
cfg.ps.max_iter      = 300;
cfg.ps.max_feval     = 1500;
cfg.ps.lhs_polls     = 4;
cfg.ps.rel_stall_tol = 0.005;      % NEW: 0.5% improvement threshold
cfg.ps.stall_iters   = 20;         % NEW: consecutive iters without rel improvement

% --- Restart / multistart counts (NEW: rebalance)
cfg.n_restarts       = 5;          % unchanged: keeps cross-basin diversity
cfg.n_local_starts   = 3;          % NEW: reduced from 5 (was over-sampling local basin)

% --- Parallel pool
cfg.n_workers        = max(1, feature('numcores') - 1);

%% Data file (local in Codici Diploma Finale)
this_dir = fileparts(mfilename('fullpath'));

%% Load the cleaned high-frequency SPY return matrix
data = load(fullfile(this_dir, 'spy_data.mat'));
intraday_returns = data.intraday_returns;
[T_emp, M_emp] = size(intraday_returns);
fprintf('Empirical sample: %d days, %d intraday intervals (5-min)\n', T_emp, M_emp);

RV_ann_emp = sum(intraday_returns.^2, 2)' * cfg.annualize;
logRV_emp  = log(RV_ann_emp)';
daily_ret_emp = (sum(intraday_returns, 2) * sqrt(cfg.annualize));

fprintf('Annualized RV: mean = %.4f, median = %.4f\n', mean(RV_ann_emp), median(RV_ann_emp));
fprintf('Daily return (ann. scale): mean = %.4f, std = %.4f\n', mean(daily_ret_emp), std(daily_ret_emp));

%% Estimate empirical auxiliary models
fprintf('\n--- Estimating HAR-RV on empirical data ---\n');
har_result = HAR_estimate(logRV_emp);
fprintf('  alpha=%.4f, beta_d=%.4f, beta_w=%.4f, beta_m=%.4f, var(eps)=%.4f\n', ...
    har_result.moments(1), har_result.moments(2), har_result.moments(3), ...
    har_result.moments(4), har_result.moments(5));
fprintf('  R^2 = %.4f\n', har_result.rsqr);

fprintf('\n--- Estimating LHAR on empirical data ---\n');
lhar_result = LHAR_estimate(logRV_emp, daily_ret_emp);
fprintf('  alpha=%.4f, beta_d=%.4f, beta_w=%.4f, beta_m=%.4f\n', ...
    lhar_result.moments(1), lhar_result.moments(2), lhar_result.moments(3), lhar_result.moments(4));
fprintf('  gamma_d=%.5f, gamma_w=%.5f, gamma_m=%.5f, var(eps)=%.4f\n', ...
    lhar_result.moments(5), lhar_result.moments(6), lhar_result.moments(7), lhar_result.moments(8));
fprintf('  R^2 = %.4f\n', lhar_result.rsqr);

coeffs_HAR  = har_result.moments;
coeffs_LHAR = lhar_result.moments;

VCV_HAR = zeros(5, 5);
VCV_HAR(1:4, 1:4) = har_result.V;
VCV_HAR(5, 5) = 2 * har_result.var_resid^2 / length(har_result.resid);
invVCV_HAR = inv(VCV_HAR);

VCV_LHAR = zeros(8, 8);
VCV_LHAR(1:7, 1:7) = lhar_result.V;
VCV_LHAR(8, 8) = 2 * lhar_result.var_resid^2 / length(lhar_result.resid);
invVCV_LHAR = inv(VCV_LHAR);

% Diagnostic: report the diagonal weights so that the relative pull on each
% moment is visible. Big asymmetries between the RV block (1:4) and the
% leverage block (5:7) explain why the fit can afford to drop the gamma_*
% moments while pursuing the alpha/beta_* moments.
lhar_names_diag = {'alpha','beta_d','beta_w','beta_m','gamma_d','gamma_w','gamma_m','var(eps)'};
fprintf('\n--- LHAR weighting matrix diagnostic ---\n');
fprintf('  %-10s  %14s  %14s\n', 'moment', 'diag(VCV)', 'diag(invVCV)');
fprintf('  %-10s  %14s  %14s\n', '------', '---------', '------------');
for j = 1:8
    fprintf('  %-10s  %14.6e  %14.6e\n', lhar_names_diag{j}, VCV_LHAR(j,j), invVCV_LHAR(j,j));
end

%% Geometric kernel grid
r_N = (cfg.gamma_max / cfg.eta_N)^(1 / (cfg.N_factors - 1));
gammas = cfg.eta_N * r_N.^(0:cfg.N_factors - 1)';
fprintf('\n--- Multifactor kernel grid ---\n');
fprintf('N = %d, eta = %.3g, r_N = %.4f, gamma_min = %.3g, gamma_max = %.3g (1/day)\n', ...
    cfg.N_factors, cfg.eta_N, r_N, gammas(1), gammas(end));
fprintf('Stability: gamma_max * dt = %.4f  (need < 1)\n', ...
    cfg.gamma_max / cfg.M_fine);

%% Pre-tabulate kernel weights
fprintf('\n--- Pre-tabulating kernel weights c_n on H grid (%d points) ---\n', ...
    length(cfg.H_grid));
tic
kernel_table = precompute_kernel_table(gammas, cfg.H_grid, cfg.T_kernel);
fprintf('  Done in %.2f s.\n', toc);

%% Per-replication seeds
fprintf('\n--- Generating per-replication seeds (%d indep + antithetic) ---\n', cfg.n_indep);
rng(cfg.seed);
W.seeds       = randi(2^31 - 1, cfg.n_indep, 1);
W.M_fine      = cfg.M_fine;
W.ndays_total = cfg.ndays_total;
W.n_indep     = cfg.n_indep;
fprintf('Stored %d seeds; per-worker peak Brownian memory will be ~%.0f MB\n', ...
    cfg.n_indep, 2 * cfg.M_fine * cfg.ndays_total * 8 / 1e6);

%% Simulation config passed to objective functions
sim_cfg.M_fine     = cfg.M_fine;
sim_cfg.agg_factor = cfg.agg_factor;
sim_cfg.ndays_eval = cfg.ndays_eval;
sim_cfg.ndays_burn = cfg.ndays_burn;
sim_cfg.annualize  = cfg.annualize;

Vbar_empirical = mean(RV_ann_emp) / cfg.annualize;
fprintf('Empirical daily Vbar = %.5f (annualized = %.4f)\n', ...
    Vbar_empirical, Vbar_empirical * cfg.annualize);

%% Start parallel pool
if ~isempty(cfg.n_workers) && cfg.n_workers > 1 && ~isempty(ver('parallel'))
    pool = gcp('nocreate');
    if isempty(pool)
        fprintf('\n--- Starting parallel pool with %d workers ---\n', cfg.n_workers);
        try
            parpool('local', cfg.n_workers);
        catch ME
            warning('setup_data_rough_new:noParpool', ...
                'Could not start parpool (%s). Will fall back to serial loops.', ME.message);
        end
    else
        fprintf('\n--- Parallel pool already running with %d workers ---\n', pool.NumWorkers);
    end
else
    fprintf('\n--- Parallel pool not requested or Parallel Computing Toolbox absent ---\n');
end

setup_done = true;
fprintf('\n=== Setup complete (NEW pipeline) ===\n\n');
