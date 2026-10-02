%% ========================================================================
%  ESTIMATION: DOUBLE HESTON + LHAR  --  NEW2 PIPELINE (ADAPTIVE RELAUNCH)
%
%  Same architecture as estimate_rough_heston_LHAR_new2.m for the rough
%  Heston (../Codici Diploma Thesis (2026) new/), ported to the two-factor
%  Heston model: 8 free parameters [kappa1, kappa2, Vbar1, Vbar2, sigma1,
%  sigma2, rho1, rho2], LHAR auxiliary (8 moments).
%
%  Differences from estimate_double_heston_LHAR_new.m (in this same folder):
%    - cfg.n_restarts raised from 5 to 10.
%    - Adaptive relaunch: trapped restarts (chi^2 >= trap_ratio * best) are
%      re-run with antithetic LHS x0 and a fresh seed; cap max_relaunches.
%      In the 8-d landscape with two CIR factors the trap risk is highest:
%      the legacy MSc result had kappa2 stuck at LB.
%    - Raw + filtered cross-restart rho1, rho2 spreads reported.
%
%  Output files:
%    - logfile_double_heston_LHAR_new2.txt
%    - results_double_heston_LHAR_new2.mat
%    - results_double_heston_LHAR_new2.txt
% ========================================================================
% NOTE 2026-05-22: was `clear all` here, which wiped t1/t2/t3/t_global set by
% run_all_double_heston_LHAR_new2.m and caused the master driver to crash on
% the subsequent toc(t1) call. Preserve them when called from run_all.
clearvars -except t1 t2 t3 t_global; close all; clc;

setup_data_new;

%% Adaptive-relaunch overrides
cfg.n_restarts = 10;
trap_ratio     = 5.0;
max_relaunches = 3;

%% Uniform NEW2 budget across the 3 LHAR pipelines (Heston / Double Heston / Rough Heston)
% Stage 1 (CMA-ES): aligned at 300 gens (confirmed good for Double Heston).
% Stage 2 (PS): expanded to exploit the 247-worker machine via deeper basin coverage.
cfg.cmaes.max_gen    = 300;
cfg.ps.max_iter      = 600;
cfg.ps.max_feval     = 4000;
cfg.ps.lhs_polls     = 8;
cfg.ps.rel_stall_tol = 0.002;
cfg.ps.stall_iters   = 30;
cfg.n_local_starts   = 5;

logfile = 'logfile_double_heston_LHAR_new2.txt';
fid_log = fopen(logfile, 'w');
if fid_log < 0
    warning('estimate_double_heston_LHAR_new2:logopen', ...
        'Could not open logfile %s; falling back to stdout-only.', logfile);
    fid_log = -1;
end

dual_log_new(fid_log, 'Double Heston + LHAR estimation (NEW2: adaptive relaunch)\n');
dual_log_new(fid_log, 'Started: %s\n\n', datestr(now));

dual_log_new(fid_log, '========================================\n');
dual_log_new(fid_log, '  DOUBLE HESTON + LHAR  (NEW2)\n');
dual_log_new(fid_log, '========================================\n');

%% Bounds: [kappa1, kappa2, Vbar1, Vbar2, sigma1, sigma2, rho1, rho2]
% NOTE 2026-05-22: rho1, rho2 bounds opened to [-0.999, 0.999] in all three
% pipelines (Heston / Double Heston / Rough Heston) for a clean cross-model
% comparison. See estimate_heston_LHAR_new2 for the full rationale.
LB = [0.001, 0.001, 0.01, 0.01, 0.01, 0.01, -0.999, -0.999];
UB = [50,    50,    3,    3,    10,   10,    0.999,  0.999];
d_par = numel(LB);

objfun = @(par) objective_double_heston_LHAR_new(coeffs_LHAR, par, invVCV_LHAR, W);

%% LHS initial means
rs_lhs  = RandStream('Threefry', 'Seed', cfg.seed + 7);
strata  = ((1:cfg.n_restarts)' - 0.5) / cfg.n_restarts;
init_means = zeros(cfg.n_restarts, d_par);
for j = 1:d_par
    perm   = randperm(rs_lhs, cfg.n_restarts);
    jitter = (rand(rs_lhs, cfg.n_restarts, 1) - 0.5) / cfg.n_restarts;
    u      = min(max(strata(perm) + jitter, 0), 1);
    init_means(:, j) = LB(j) + u * (UB(j) - LB(j));
end

%% Stage 1: CMA-ES, all initial restarts
dual_log_new(fid_log, '\nStage 1: CMA-ES (%d seeds, LHS-init, %d gens, adaptive relaunch)...\n', ...
    cfg.n_restarts, cfg.cmaes.max_gen);

best_chi2 = inf;
best_par  = NaN(1, d_par);
restart_results = struct('par', cell(cfg.n_restarts, 1), ...
                         'chi2', cell(cfg.n_restarts, 1), ...
                         'info', cell(cfg.n_restarts, 1));

for s = 1:cfg.n_restarts
    [restart_results(s), best_chi2, best_par] = local_run_restart( ...
        s, init_means(s, :), LB, UB, cfg, objfun, fid_log, ...
        cfg.seed + 1000 * s, best_chi2, best_par);
end

%% Adaptive relaunch of trapped restarts
center = (LB + UB) / 2;
n_relaunched = 0;
relaunch_log = struct('orig_idx', {}, 'orig_chi2', {}, 'new_chi2', {}, 'improved', {});

for s = 1:cfg.n_restarts
    if n_relaunched >= max_relaunches, break; end
    if restart_results(s).chi2 > trap_ratio * best_chi2
        x0_anti = 2*center - init_means(s, :);
        x0_anti = min(max(x0_anti, LB), UB);
        dual_log_new(fid_log, '\n--- RELAUNCH restart %d (chi2 %.0f > %.1fx best %.4f) ---\n', ...
            s, restart_results(s).chi2, trap_ratio, best_chi2);
        dual_log_new(fid_log, 'Antithetic x0: [%s]\n', sprintf(' %.4f', x0_anti));
        seed_anti = cfg.seed + 5000 + s;
        old_chi2 = restart_results(s).chi2;
        [new_r, best_chi2, best_par] = local_run_restart( ...
            s, x0_anti, LB, UB, cfg, objfun, fid_log, seed_anti, best_chi2, best_par);
        improved = new_r.chi2 < old_chi2;
        if improved
            restart_results(s) = new_r;
            dual_log_new(fid_log, 'Relaunch IMPROVED restart %d: chi2 %.4f -> %.4f\n', ...
                s, old_chi2, new_r.chi2);
        else
            dual_log_new(fid_log, 'Relaunch did not improve restart %d (kept original chi2 %.4f).\n', ...
                s, old_chi2);
        end
        relaunch_log(end+1) = struct('orig_idx', s, 'orig_chi2', old_chi2, ...
                                     'new_chi2', new_r.chi2, 'improved', improved); %#ok<AGROW>
        n_relaunched = n_relaunched + 1;
    end
end

dual_log_new(fid_log, '\nAdaptive relaunch summary: %d relaunch(es) executed (cap %d).\n', ...
    n_relaunched, max_relaunches);

%% Diagnostics: rho1, rho2 spread raw + filtered
chi2_seeds = arrayfun(@(r) r.chi2,   restart_results);
rho1_seeds = arrayfun(@(r) r.par(7), restart_results);
rho2_seeds = arrayfun(@(r) r.par(8), restart_results);
keep       = chi2_seeds < trap_ratio * min(chi2_seeds);

rho1_spread_raw      = max(rho1_seeds)       - min(rho1_seeds);
rho2_spread_raw      = max(rho2_seeds)       - min(rho2_seeds);
if any(keep)
    rho1_spread_filtered = max(rho1_seeds(keep)) - min(rho1_seeds(keep));
    rho2_spread_filtered = max(rho2_seeds(keep)) - min(rho2_seeds(keep));
else
    rho1_spread_filtered = NaN;
    rho2_spread_filtered = NaN;
end
trapped_idx = find(~keep);

dual_log_new(fid_log, '\nCross-restart rho1 spread RAW       (n=%d): %.4f\n', cfg.n_restarts, rho1_spread_raw);
dual_log_new(fid_log, 'Cross-restart rho1 spread FILTERED  (n=%d): %.4f\n', sum(keep), rho1_spread_filtered);
dual_log_new(fid_log, 'Cross-restart rho2 spread RAW       (n=%d): %.4f\n', cfg.n_restarts, rho2_spread_raw);
dual_log_new(fid_log, 'Cross-restart rho2 spread FILTERED  (n=%d): %.4f\n', sum(keep), rho2_spread_filtered);
dual_log_new(fid_log, 'Trapped restart indices (chi^2 >= %.1fx best): %s\n', ...
    trap_ratio, mat2str(trapped_idx(:)'));

%% Stage 2: pattern search refinement
dual_log_new(fid_log, '\nStage 2: Pattern search refinement (%d local starts)...\n', cfg.n_local_starts);

[~, idx_best] = min(chi2_seeds);
local_sigma = sqrt(diag(restart_results(idx_best).info.final_C))' * restart_results(idx_best).info.final_sigma;
perturb_scale = 0.1 * local_sigma;

starts = zeros(cfg.n_local_starts, d_par);
starts(1, :) = best_par;
for k = 2:cfg.n_local_starts
    starts(k, :) = min(max(best_par + perturb_scale .* randn(1, d_par), LB), UB);
end

ps_results = struct('par', cell(cfg.n_local_starts, 1), ...
                    'chi2', cell(cfg.n_local_starts, 1), ...
                    'info', cell(cfg.n_local_starts, 1));
for k = 1:cfg.n_local_starts
    dual_log_new(fid_log, '\n--- Stage 2 local start %d/%d ---\n', k, cfg.n_local_starts);
    ps_opts = cfg.ps;
    ps_opts.verbose = true;
    ps_opts.log_fid = fid_log;
    [par_k, chi2_k, info_k] = pattern_search_bnd_new(objfun, starts(k, :), LB, UB, ps_opts);
    ps_results(k).par  = par_k;
    ps_results(k).chi2 = chi2_k;
    ps_results(k).info = info_k;
    dual_log_new(fid_log, 'PS start %d: chi2 = %.6f, par = [%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f]\n', ...
        k, chi2_k, par_k);
end

[fmin, idx_min] = min([ps_results.chi2]);
a = ps_results(idx_min).par;

dual_log_new(fid_log, '\nFinal evaluation at the converged optimum...\n');
[~, betamean, stdbeta, chi2_contrib, t_stats] = objfun(a);

%% Stage 3: OOS
dual_log_new(fid_log, '\nStage 3: Out-of-sample binding-function check...\n');
rng(cfg.seed + 999);
W_oos.seeds   = randi(2^31 - 1, cfg.n_repl, 1);
W_oos.n_intra = cfg.n_intra;
W_oos.n_sim   = cfg.n_sim;
W_oos.n_repl  = cfg.n_repl;
chi2_oos = objective_double_heston_LHAR_new(coeffs_LHAR, a, invVCV_LHAR, W_oos);
dual_log_new(fid_log, 'In-sample  chi2 = %.6f\n', fmin);
dual_log_new(fid_log, 'Out-of-sample chi2 = %.6f  (ratio = %.3f)\n', chi2_oos, chi2_oos / max(fmin, 1e-12));

%% Stage 3b: empirical MC noise floor at the optimum
dual_log_new(fid_log, '\nStage 3b: Empirical MC noise floor at the optimum (10 seeds)...\n');
n_noise = 10;
chi2_noise = NaN(n_noise, 1);
for k = 1:n_noise
    rng(cfg.seed + 8000 + k);
    W_k.seeds   = randi(2^31 - 1, cfg.n_repl, 1);
    W_k.n_intra = cfg.n_intra;
    W_k.n_sim   = cfg.n_sim;
    W_k.n_repl  = cfg.n_repl;
    chi2_noise(k) = objective_double_heston_LHAR_new(coeffs_LHAR, a, invVCV_LHAR, W_k);
end
chi2_noise_mean = mean(chi2_noise);
chi2_noise_std  = std(chi2_noise);
dual_log_new(fid_log, 'Noise floor: mean(chi2) = %.4f, std(chi2) = %.4f over %d seeds\n', ...
    chi2_noise_mean, chi2_noise_std, n_noise);
dual_log_new(fid_log, 'Signal-to-noise: chi2_min / std(chi2) = %.2f\n', fmin / max(chi2_noise_std, 1e-12));

%% Store
est.model        = 'Double Heston';
est.auxiliary    = 'LHAR';
est.params       = a;
est.param_names  = {'kappa1', 'kappa2', 'Vbar1', 'Vbar2', 'sigma1', 'sigma2', 'rho1', 'rho2'};
est.chi2         = fmin;
est.chi2_oos     = chi2_oos;
est.exitflag     = 1;
est.betamean     = betamean;
est.stdbeta      = stdbeta;
est.chi2_contrib = chi2_contrib;
est.t_stats      = t_stats;
est.rho1_spread_seeds    = rho1_spread_raw;
est.rho2_spread_seeds    = rho2_spread_raw;
est.rho1_spread_filtered = rho1_spread_filtered;
est.rho2_spread_filtered = rho2_spread_filtered;
est.trapped_idx          = trapped_idx;
est.posthoc_filter_ratio = trap_ratio;
est.restart_results = restart_results;
est.ps_results      = ps_results;
est.adaptive_relaunch.n_relaunched   = n_relaunched;
est.adaptive_relaunch.max_relaunches = max_relaunches;
est.adaptive_relaunch.trap_ratio     = trap_ratio;
est.adaptive_relaunch.log            = relaunch_log;
est.noise_floor.chi2_seeds = chi2_noise;
est.noise_floor.mean       = chi2_noise_mean;
est.noise_floor.std        = chi2_noise_std;
est.noise_floor.snr        = fmin / max(chi2_noise_std, 1e-12);

lhar_names = {'alpha', 'beta_d', 'beta_w', 'beta_m', 'gamma_d', 'gamma_w', 'gamma_m', 'var(eps)'};
print_estimation_results_new(est, coeffs_LHAR, lhar_names, 'results_double_heston_LHAR_new2.txt');

% NOTE 2026-05-22: robust save to handle the legacy "permission denied"
% failures. See estimate_heston_LHAR_new2.m for the rationale.
% NOTE 2026-05-23: added retry-with-pause before falling back to a
% timestamped filename. The timestamped fallback breaks downstream
% warm-starts (profile_rho1_*, bootstrap_SE_*) which look for the
% canonical name only. Permission errors are almost always transient
% (antivirus, OneDrive sync, another MATLAB instance holding the lock),
% so a short retry loop keeps the canonical filename in 99% of cases.
matfile = 'results_double_heston_LHAR_new2.mat';
max_retries = 5;
pause_secs  = [1 2 4 8 15];
saved = false;
for retry = 1:max_retries
    try
        if exist(matfile, 'file') == 2
            delete(matfile);
        end
        save(matfile, 'est', 'coeffs_LHAR', 'lhar_result', 'cfg');
        dual_log_new(fid_log, '\nSaved to %s (attempt %d)\n', matfile, retry);
        saved = true;
        break
    catch ME
        if retry < max_retries
            dual_log_new(fid_log, 'Save attempt %d failed (%s); retrying in %ds...\n', ...
                retry, ME.message, pause_secs(retry));
            pause(pause_secs(retry));
        else
            matfile_alt = sprintf('results_double_heston_LHAR_new2_%s.mat', datestr(now, 'yyyymmdd_HHMMSS'));
            warning('estimate_double_heston_LHAR_new2:saveLocked', ...
                'Could not save to %s after %d retries (%s); writing to %s instead.', ...
                matfile, max_retries, ME.message, matfile_alt);
            save(matfile_alt, 'est', 'coeffs_LHAR', 'lhar_result', 'cfg');
            dual_log_new(fid_log, '\nSaved to %s (fallback after %d retries; original target %s was locked)\n', ...
                matfile_alt, max_retries, matfile);
            dual_log_new(fid_log, 'WARNING: downstream scripts (profile_rho1_*, bootstrap_SE_*) expect %s\n', matfile);
            dual_log_new(fid_log, 'Rename %s -> %s manually before running them.\n', matfile_alt, matfile);
        end
    end
end
dual_log_new(fid_log, 'Finished: %s\n', datestr(now));

if fid_log > 0
    fclose(fid_log);
end


%% ===================== local helper =====================
function [r, best_chi2_out, best_par_out] = local_run_restart(s, x0, LB, UB, cfg, objfun, fid_log, seed_val, best_chi2_in, best_par_in)
    dual_log_new(fid_log, '\n--- Stage 1 restart %d ---\n', s);
    opts = cfg.cmaes;
    opts.seed    = seed_val;
    opts.verbose = true;
    opts.log_fid = fid_log;
    sigma0 = 0.25 * (UB - LB);
    dual_log_new(fid_log, 'Initial mean: [%s]\n', sprintf(' %.4f', x0));
    [par_s, chi2_s, info_s] = cmaes_bnd_new(objfun, x0, sigma0, LB, UB, opts);
    r.par  = par_s;
    r.chi2 = chi2_s;
    r.info = info_s;
    dual_log_new(fid_log, 'Restart %d: chi2 = %.6f, par = [%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f]\n', ...
        s, chi2_s, par_s);
    if chi2_s < best_chi2_in
        best_chi2_out = chi2_s;
        best_par_out  = par_s;
    else
        best_chi2_out = best_chi2_in;
        best_par_out  = best_par_in;
    end
end
