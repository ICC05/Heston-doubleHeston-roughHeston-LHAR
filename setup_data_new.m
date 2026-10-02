%% ========================================================================
%  SETUP_DATA  Common data loading and configuration for the improved
%  Heston / Double Heston indirect inference pipeline.
%
%  Reuses spy_data.mat, HAR_estimate.m, LHAR_estimate.m, aggregateAvg.m,
%  nwest.m, lag_cut.m from '../Codici Cardosi Carrara MSc (2026)' so that
%  the empirical sample, RV construction and auxiliary criterion are
%  bit-identical to the legacy pipeline. Any difference in the reported
%  numbers therefore reflects a difference in optimization only.
%
%  Major differences vs. the legacy setup_data_new.m of MSc 2026:
%    - No precomputed Brownian arrays; per-replication Threefry seeds are
%      generated once here and the actual shocks are produced on-the-fly
%      inside each parfor iteration of the objective. Cuts the memory
%      from ~384 MB (double Heston) to a few seed scalars.
%    - CMA-ES and Pattern Search settings replace the legacy GA + Nelder-
%      Mead settings. lambda_pop = 20 (vs Hansen default 8 for d~5) so
%      mu_eff ~ 5.5 averages out simulation noise across elite candidates.
%    - Parpool launched here (feature('numcores') - 1 workers by default).
%  ========================================================================

fprintf('=== Loading data and preparing improved Heston pipeline ===\n\n');

%% Configuration
cfg.seed      = 301074;
cfg.n_sim     = 6000;
cfg.n_intra   = 80;
cfg.n_repl    = 247;      % 247 indep + 247 antithetic = 494 effective; 2 tasks/worker on a 247-worker pool (caramel default). Raised from 127 on 2026-05-22 to cut MC SE by ~30% at same wall time as the legacy 127-worker setup.
cfg.annualize = 252;

% --- Optimization: CMA-ES
cfg.cmaes.lambda_pop = 20;
cfg.cmaes.max_gen    = 200;
cfg.cmaes.tolfun     = 1e-6;
cfg.cmaes.tolfun_rel = 1e-4;
cfg.cmaes.tolx       = 1e-7;

% --- Optimization: Pattern Search
cfg.ps.max_iter      = 300;
cfg.ps.max_feval     = 1500;
cfg.ps.lhs_polls     = 4;
cfg.ps.rel_stall_tol = 0.005;
cfg.ps.stall_iters   = 20;

% --- Restart / multistart counts
cfg.n_restarts       = 5;
cfg.n_local_starts   = 3;

% --- Parallel pool
cfg.n_workers        = max(1, feature('numcores') - 1);

%% Data file (local in Codici Diploma Finale)
this_dir = fileparts(mfilename('fullpath'));

%% Load data
data = load(fullfile(this_dir, 'spy_data.mat'));
intraday_returns = data.intraday_returns;
[T, M] = size(intraday_returns);
fprintf('Data: %d days, %d intraday intervals (5-min)\n', T, M);

% Annualized daily RV and log-RV
RV_ann = sum(intraday_returns.^2, 2)' * cfg.annualize;
logRV  = log(RV_ann)';

% Daily returns (annualized scale, for LHAR leverage regressors)
daily_ret = (sum(intraday_returns, 2) * sqrt(cfg.annualize));

fprintf('Annualized RV: mean=%.4f, median=%.4f\n', mean(RV_ann), median(RV_ann));
fprintf('Daily return (ann. scale): mean=%.4f, std=%.4f\n', mean(daily_ret), std(daily_ret));

%% Estimate empirical auxiliary models
fprintf('\n--- Estimating HAR-RV on empirical data ---\n');
har_result = HAR_estimate(logRV);
fprintf('  alpha=%.4f, beta_d=%.4f, beta_w=%.4f, beta_m=%.4f, var(eps)=%.4f\n', ...
    har_result.moments(1), har_result.moments(2), har_result.moments(3), ...
    har_result.moments(4), har_result.moments(5));
fprintf('  R^2 = %.4f\n', har_result.rsqr);

fprintf('\n--- Estimating LHAR on empirical data ---\n');
lhar_result = LHAR_estimate(logRV, daily_ret);
fprintf('  alpha=%.4f, beta_d=%.4f, beta_w=%.4f, beta_m=%.4f\n', ...
    lhar_result.moments(1), lhar_result.moments(2), lhar_result.moments(3), lhar_result.moments(4));
fprintf('  gamma_d=%.5f, gamma_w=%.5f, gamma_m=%.5f, var(eps)=%.4f\n', ...
    lhar_result.moments(5), lhar_result.moments(6), lhar_result.moments(7), lhar_result.moments(8));
fprintf('  R^2 = %.4f\n', lhar_result.rsqr);

% Target moments
coeffs_HAR  = har_result.moments;
coeffs_LHAR = lhar_result.moments;

% Weighting matrices
VCV_HAR = zeros(5, 5);
VCV_HAR(1:4, 1:4) = har_result.V;
VCV_HAR(5, 5) = 2 * har_result.var_resid^2 / length(har_result.resid);
invVCV_HAR = inv(VCV_HAR);

VCV_LHAR = zeros(8, 8);
VCV_LHAR(1:7, 1:7) = lhar_result.V;
VCV_LHAR(8, 8) = 2 * lhar_result.var_resid^2 / length(lhar_result.resid);
invVCV_LHAR = inv(VCV_LHAR);

%% Per-replication seeds (instead of materializing 4D Brownian arrays)
fprintf('\n--- Generating per-replication seeds (%d indep + antithetic) ---\n', cfg.n_repl);
rng(cfg.seed);
W.seeds   = randi(2^31 - 1, cfg.n_repl, 1);
W.n_intra = cfg.n_intra;
W.n_sim   = cfg.n_sim;
W.n_repl  = cfg.n_repl;
fprintf('Stored %d seeds; per-worker peak memory for Brownian shocks ~%.1f MB (Double Heston)\n', ...
    cfg.n_repl, 4 * cfg.n_intra * cfg.n_sim * 8 / 1e6);

Vbar_empirical = mean(RV_ann) / cfg.annualize;
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
            warning('setup_data_new:noParpool', ...
                'Could not start parpool (%s). Will fall back to serial loops.', ME.message);
        end
    else
        fprintf('\n--- Parallel pool already running with %d workers ---\n', pool.NumWorkers);
    end
else
    fprintf('\n--- Parallel pool not requested or Parallel Computing Toolbox absent ---\n');
end

setup_done = true;
fprintf('\n=== Setup complete ===\n\n');
