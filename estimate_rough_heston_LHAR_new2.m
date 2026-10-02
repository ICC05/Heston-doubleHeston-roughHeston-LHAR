%% ========================================================================
%  ESTIMATION: ROUGH HESTON + LHAR  (FULL)  --  NEW2 PIPELINE
%
%  Drop-in extension of estimate_rough_heston_LHAR_new.m living in
%  ../Codici Diploma thesis (2026)/. Differences from that file:
%    - cfg.n_restarts raised from 5 to 10 (more cross-basin coverage).
%    - Adaptive relaunch loop: any restart with chi^2 >= trap_ratio * best
%      is re-run with an antithetic LHS x0 (reflection around the box
%      centre) and a fresh seed; capped at max_relaunches total relaunches.
%      Motivation: in the legacy run two restarts converged to a deep local
%      minimum (lambda~40, rho~-0.99, H~0.4, chi^2~17300) and contaminated
%      the cross-restart H spread diagnostic.
%    - Both raw and filtered H spreads are reported and stored in est.
%
%  Output files (kept distinct from the legacy *_new outputs so nothing is
%  overwritten in the original folder):
%    - logfile_rough_heston_LHAR_new2.txt
%    - results_rough_heston_LHAR_new2.mat
%    - results_rough_heston_LHAR_new2.txt
%
%  Dependencies (setup_data_rough_new, cmaes_bnd_new, pattern_search_bnd_new,
%  dual_log, objective_rough_heston_LHAR, print_estimation_results) are
%  loaded from ../Codici Diploma thesis (2026)/ via addpath at startup.
% ========================================================================
% NOTE 2026-05-22: was `clear all` here, which wiped t1/t2/t3/t_global set by
% run_all_rough_heston_LHAR_new2.m. After Stage 1 completed correctly the
% master driver crashed on toc(t1), preventing Stage 2 (profile_H) and
% Stage 3 (bootstrap_SE) from running. Preserve the run_all timers.
clearvars -except t1 t2 t3 t_global; close all; clc;

setup_data_rough_new;

%% Adaptive-relaunch overrides
cfg.n_restarts = 10;     % was 5 in the legacy *_new
trap_ratio     = 5.0;    % restart trapped if chi^2 >= trap_ratio * best
max_relaunches = 3;      % cap total relaunch attempts

%% Uniform NEW2 budget across the 3 LHAR pipelines (Heston / Double Heston / Rough Heston)
% Stage 1 (CMA-ES): aligned at 300 gens (overrides setup_data_rough_new's 120).
% Stage 2 (PS): expanded to exploit the 247-worker machine via deeper basin coverage.
cfg.cmaes.max_gen    = 300;
cfg.ps.max_iter      = 600;
cfg.ps.max_feval     = 4000;
cfg.ps.lhs_polls     = 8;
cfg.ps.rel_stall_tol = 0.002;
cfg.ps.stall_iters   = 30;
cfg.n_local_starts   = 5;

logfile = 'logfile_rough_heston_LHAR_new2.txt';
fid_log = fopen(logfile, 'w');
if fid_log < 0
    warning('estimate_rough_heston_LHAR_new2:logopen', ...
        'Could not open logfile %s; falling back to stdout-only.', logfile);
    fid_log = -1;
end

dual_log(fid_log, 'Rough Heston + LHAR estimation (NEW2: adaptive relaunch)\n');
dual_log(fid_log, 'Started: %s\n\n', datestr(now));

dual_log(fid_log, '========================================\n');
dual_log(fid_log, '  ROUGH HESTON + LHAR  (NEW2)\n');
dual_log(fid_log, '========================================\n');

%% Bounds: [lambda, theta, nu, rho, H]
% NOTE 2026-05-21: bound relaxed after the previous run saturated rho at UB=0,
% H at LB=0.05, and approached nu UB=10. Saturation was masking misspecification:
% the LHAR gamma_* (leverage) moments were dropped to ~0 because rho was pinned.
% NOTE 2026-05-22: rho UB opened from +0.10 to +0.999 (symmetric with LB) for a
% clean cross-model comparison with Heston and Double Heston. The previous
% +0.10 cap was already pinning; +0.999 lets the rough Heston express its
% preferred rho without artificial saturation. If rho lands well above 0,
% it confirms the structural mismatch between rough Heston and the LHAR
% leverage block (gamma_*).
LB = [0.001, 0.01,  0.01, -0.999, 0.02];
UB = [50,    5,     25,    0.999, 0.49];

objfun = @(par) objective_rough_heston_LHAR(coeffs_LHAR, par, invVCV_LHAR, W, sim_cfg, kernel_table);

%% Latin Hypercube initial means for the n_restarts CMA-ES seeds
rs_lhs  = RandStream('Threefry', 'Seed', cfg.seed + 7);
d_par   = numel(LB);
strata  = ((1:cfg.n_restarts)' - 0.5) / cfg.n_restarts;
init_means = zeros(cfg.n_restarts, d_par);
for j = 1:d_par
    perm   = randperm(rs_lhs, cfg.n_restarts);
    jitter = (rand(rs_lhs, cfg.n_restarts, 1) - 0.5) / cfg.n_restarts;
    u      = min(max(strata(perm) + jitter, 0), 1);
    init_means(:, j) = LB(j) + u * (UB(j) - LB(j));
end

%% Stage 1: CMA-ES, all initial restarts
dual_log(fid_log, '\nStage 1: CMA-ES (%d seeds, LHS-initialised, adaptive relaunch)...\n', cfg.n_restarts);

best_chi2 = inf;
best_par  = NaN(1, 5);
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
        dual_log(fid_log, '\n--- RELAUNCH restart %d (chi2 %.0f > %.1fx best %.4f) ---\n', ...
            s, restart_results(s).chi2, trap_ratio, best_chi2);
        dual_log(fid_log, 'Antithetic x0: [%s]\n', sprintf(' %.4f', x0_anti));
        seed_anti = cfg.seed + 5000 + s;
        old_chi2 = restart_results(s).chi2;
        [new_r, best_chi2, best_par] = local_run_restart( ...
            s, x0_anti, LB, UB, cfg, objfun, fid_log, seed_anti, best_chi2, best_par);
        improved = new_r.chi2 < old_chi2;
        if improved
            restart_results(s) = new_r;
            dual_log(fid_log, 'Relaunch IMPROVED restart %d: chi2 %.4f -> %.4f\n', ...
                s, old_chi2, new_r.chi2);
        else
            dual_log(fid_log, 'Relaunch did not improve restart %d (kept original chi2 %.4f).\n', ...
                s, old_chi2);
        end
        relaunch_log(end+1) = struct('orig_idx', s, 'orig_chi2', old_chi2, ...
                                     'new_chi2', new_r.chi2, 'improved', improved); %#ok<AGROW>
        n_relaunched = n_relaunched + 1;
    end
end

dual_log(fid_log, '\nAdaptive relaunch summary: %d relaunch(es) executed (cap %d).\n', ...
    n_relaunched, max_relaunches);

%% Cross-restart diagnostics (raw + filtered)
chi2_seeds = arrayfun(@(r) r.chi2,   restart_results);
H_seeds    = arrayfun(@(r) r.par(5), restart_results);
keep       = chi2_seeds < trap_ratio * min(chi2_seeds);
H_spread_raw      = max(H_seeds)       - min(H_seeds);
if any(keep)
    H_spread_filtered = max(H_seeds(keep)) - min(H_seeds(keep));
else
    H_spread_filtered = NaN;
end
trapped_idx = find(~keep);

dual_log(fid_log, '\nCross-restart H spread RAW       (n=%d): %.4f (flag if > 0.05)\n', ...
    cfg.n_restarts, H_spread_raw);
dual_log(fid_log, 'Cross-restart H spread FILTERED  (n=%d): %.4f\n', ...
    sum(keep), H_spread_filtered);
dual_log(fid_log, 'Trapped restart indices (chi^2 >= %.1fx best): %s\n', ...
    trap_ratio, mat2str(trapped_idx(:)'));

%% Stage 2: pattern search refinement
dual_log(fid_log, '\nStage 2: Pattern search refinement (%d local starts)...\n', cfg.n_local_starts);

[~, idx_best] = min(chi2_seeds);
local_sigma = sqrt(diag(restart_results(idx_best).info.final_C))' * restart_results(idx_best).info.final_sigma;
perturb_scale = 0.1 * local_sigma;

starts = zeros(cfg.n_local_starts, 5);
starts(1, :) = best_par;
for k = 2:cfg.n_local_starts
    starts(k, :) = min(max(best_par + perturb_scale .* randn(1, 5), LB), UB);
end

ps_results = struct('par', cell(cfg.n_local_starts, 1), ...
                    'chi2', cell(cfg.n_local_starts, 1), ...
                    'info', cell(cfg.n_local_starts, 1));
for k = 1:cfg.n_local_starts
    dual_log(fid_log, '\n--- Stage 2 local start %d/%d ---\n', k, cfg.n_local_starts);
    ps_opts = cfg.ps;
    ps_opts.verbose = true;
    ps_opts.log_fid = fid_log;
    [par_k, chi2_k, info_k] = pattern_search_bnd_new(objfun, starts(k, :), LB, UB, ps_opts);
    ps_results(k).par  = par_k;
    ps_results(k).chi2 = chi2_k;
    ps_results(k).info = info_k;
    dual_log(fid_log, 'PS start %d: chi2 = %.6f, par = [%.4f %.4f %.4f %.4f %.4f]\n', ...
        k, chi2_k, par_k);
end

[fmin, idx_min] = min([ps_results.chi2]);
a = ps_results(idx_min).par;

dual_log(fid_log, '\nFinal evaluation at the converged optimum...\n');
[~, betamean, stdbeta, chi2_contrib, t_stats] = objfun(a);

%% Stage 3: out-of-sample check
dual_log(fid_log, '\nStage 3: Out-of-sample binding-function check...\n');
rng(cfg.seed + 999);
W_oos.seeds       = randi(2^31 - 1, cfg.n_indep, 1);
W_oos.M_fine      = cfg.M_fine;
W_oos.ndays_total = cfg.ndays_total;
W_oos.n_indep     = cfg.n_indep;
[chi2_oos, betamean_oos, ~] = objective_rough_heston_LHAR(coeffs_LHAR, a, invVCV_LHAR, W_oos, sim_cfg, kernel_table);
dual_log(fid_log, 'In-sample  chi2 = %.6f\n', fmin);
dual_log(fid_log, 'Out-of-sample chi2 = %.6f  (ratio = %.3f)\n', chi2_oos, chi2_oos / max(fmin, 1e-12));

%% Stage 3b: empirical MC noise floor at the optimum
dual_log(fid_log, '\nStage 3b: Empirical MC noise floor at the optimum (10 seeds)...\n');
n_noise = 10;
chi2_noise = NaN(n_noise, 1);
for k = 1:n_noise
    rng(cfg.seed + 8000 + k);
    W_k.seeds       = randi(2^31 - 1, cfg.n_indep, 1);
    W_k.M_fine      = cfg.M_fine;
    W_k.ndays_total = cfg.ndays_total;
    W_k.n_indep     = cfg.n_indep;
    chi2_noise(k) = objective_rough_heston_LHAR(coeffs_LHAR, a, invVCV_LHAR, W_k, sim_cfg, kernel_table);
end
chi2_noise_mean = mean(chi2_noise);
chi2_noise_std  = std(chi2_noise);
dual_log(fid_log, 'Noise floor: mean(chi2) = %.4f, std(chi2) = %.4f over %d seeds\n', ...
    chi2_noise_mean, chi2_noise_std, n_noise);
dual_log(fid_log, 'Signal-to-noise: chi2_min / std(chi2) = %.2f\n', fmin / max(chi2_noise_std, 1e-12));

%% Store
est.model        = 'RoughHeston';
est.auxiliary    = 'LHAR';
est.params       = a;
est.param_names  = {'lambda', 'theta', 'nu', 'rho', 'H'};
est.chi2         = fmin;
est.chi2_oos     = chi2_oos;
est.betamean     = betamean;
est.stdbeta      = stdbeta;
est.chi2_contrib = chi2_contrib;
est.t_stats      = t_stats;
est.H_spread_seeds      = H_spread_raw;       % raw, matches legacy field name
est.H_spread_filtered   = H_spread_filtered;
est.trapped_idx         = trapped_idx;
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
print_estimation_results(est, coeffs_LHAR, lhar_names, 'results_rough_heston_LHAR_new2.txt');

% NOTE 2026-05-22: robust save to handle the legacy "permission denied"
% failures. See estimate_heston_LHAR_new2.m for the rationale.
% NOTE 2026-05-23: added retry-with-pause before falling back. The
% timestamped fallback breaks downstream warm-starts (profile_H_*,
% bootstrap_SE_*) which look for the canonical name only.
matfile = 'results_rough_heston_LHAR_new2.mat';
max_retries = 5;
pause_secs  = [1 2 4 8 15];
for retry = 1:max_retries
    try
        if exist(matfile, 'file') == 2
            delete(matfile);
        end
        save(matfile, 'est', 'coeffs_LHAR', 'lhar_result', 'cfg', 'kernel_table');
        dual_log(fid_log, '\nSaved to %s (attempt %d)\n', matfile, retry);
        break
    catch ME
        if retry < max_retries
            dual_log(fid_log, 'Save attempt %d failed (%s); retrying in %ds...\n', ...
                retry, ME.message, pause_secs(retry));
            pause(pause_secs(retry));
        else
            matfile_alt = sprintf('results_rough_heston_LHAR_new2_%s.mat', datestr(now, 'yyyymmdd_HHMMSS'));
            warning('estimate_rough_heston_LHAR_new2:saveLocked', ...
                'Could not save to %s after %d retries (%s); writing to %s instead.', ...
                matfile, max_retries, ME.message, matfile_alt);
            save(matfile_alt, 'est', 'coeffs_LHAR', 'lhar_result', 'cfg', 'kernel_table');
            dual_log(fid_log, '\nSaved to %s (fallback after %d retries; original target %s was locked)\n', ...
                matfile_alt, max_retries, matfile);
            dual_log(fid_log, 'WARNING: downstream scripts (profile_H_*, bootstrap_SE_*) expect %s\n', matfile);
            dual_log(fid_log, 'Rename %s -> %s manually before running them.\n', matfile_alt, matfile);
        end
    end
end
dual_log(fid_log, 'Finished: %s\n', datestr(now));

if fid_log > 0
    fclose(fid_log);
end


%% ===================== local helper =====================
function [r, best_chi2_out, best_par_out] = local_run_restart(s, x0, LB, UB, cfg, objfun, fid_log, seed_val, best_chi2_in, best_par_in)
    dual_log(fid_log, '\n--- Stage 1 restart %d ---\n', s);
    opts = cfg.cmaes;
    opts.seed    = seed_val;
    opts.verbose = true;
    opts.log_fid = fid_log;
    sigma0 = 0.25 * (UB - LB);
    dual_log(fid_log, 'Initial mean: [%s]\n', sprintf(' %.4f', x0));
    [par_s, chi2_s, info_s] = cmaes_bnd_new(objfun, x0, sigma0, LB, UB, opts);
    r.par  = par_s;
    r.chi2 = chi2_s;
    r.info = info_s;
    dual_log(fid_log, 'Restart %d: chi2 = %.6f, par = [%.4f %.4f %.4f %.4f %.4f]\n', ...
        s, chi2_s, par_s);
    if chi2_s < best_chi2_in
        best_chi2_out = chi2_s;
        best_par_out  = par_s;
    else
        best_chi2_out = best_chi2_in;
        best_par_out  = best_par_in;
    end
end
