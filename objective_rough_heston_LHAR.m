function [y, betamean, stdbeta, chi2_contrib, t_stats] = objective_rough_heston_LHAR(coeffs, par, invVCV, W, sim_cfg, kernel_table)
% OBJECTIVE_ROUGH_HESTON_LHAR  EMSM objective for the rough Heston model
% with the LHAR auxiliary, in the fully-parameterized specification
% (Section 7.2.4 of the thesis: over-identified, p = 5, q = 7).
%
% INPUTS:
%   coeffs       - (8 x 1) target LHAR moments
%                  [alpha; beta_d; beta_w; beta_m; gamma_d; gamma_w; gamma_m; var(eps)]
%   par          - (1 x 5) free parameters [lambda, theta, nu, rho, H]
%   invVCV       - (8 x 8) inverse VCV used as weighting matrix
%   W            - struct with fields:
%                    .seeds       (n_indep x 1) per-replication RNG seeds
%                    .M_fine, .ndays_total, .n_indep
%                  Brownian shocks are generated on-the-fly from the seeds
%                  inside the parfor loop to avoid broadcasting large
%                  arrays across workers.
%   sim_cfg      - struct with fields .ndays_eval, .ndays_burn, .M_fine, .agg_factor, .annualize
%   kernel_table - struct from precompute_kernel_table
%
% OUTPUTS:
%   y            - SMM weighted quadratic loss
%   betamean     - (8 x 1) mean deviation of simulated minus empirical auxiliary moments
%   stdbeta      - (8 x 1) standard error of the mean across replications
%   chi2_contrib - (8 x 1) per-moment contribution to the chi^2, defined as
%                  the diagonal of W * (betamean * betamean'), so that
%                  sum(chi2_contrib) reproduces y when invVCV is diagonal and
%                  is a useful first-order decomposition otherwise. Used
%                  off-line in the diagnostic report; not needed by the
%                  optimizer (one output is enough).
%   t_stats      - (8 x 1) pseudo t-statistic betamean ./ stdbeta of the
%                  simulated minus empirical gap. |t| >> 2 flags a moment
%                  that the model cannot match within MC noise.

PENALTY = 1e10;

% ------------------ unpack ------------------
p.lambda = par(1);
p.theta  = par(2);
p.nu     = par(3);
p.rho    = par(4);
p.H      = par(5);
p.mu     = 0;

ndays_total = sim_cfg.ndays_eval + sim_cfg.ndays_burn;
n_indep = W.n_indep;
n_repl_total = 2 * n_indep;
nmom = length(coeffs);
M_fine_loc = W.M_fine;
T_total_loc = W.ndays_total;
seeds_loc = W.seeds;

% ------------------ feasibility ------------------
% Both thresholds were loosened on 2026-05-21 after the previous run pinned
% nu/lambda exactly at 200 (cliff) and dragged lambda down to 0.04 just to
% stay inside the box. Feller >= 1e-4 still keeps the variance well-defined;
% nu/lambda <= 500 still rules out the pathological random-walk regime but
% no longer compresses lambda against an invisible constraint.
feller = 2 * p.theta / (p.nu^2);
if feller < 1e-4 || p.nu / p.lambda > 500
    y = PENALTY;
    betamean = NaN(nmom, 1);
    stdbeta  = NaN(nmom, 1);
    chi2_contrib = NaN(nmom, 1);
    t_stats      = NaN(nmom, 1);
    return
end

% ------------------ inner Monte Carlo loop ------------------
% parfor parallelizes over replications. Each iteration is independent and
% writes to the sliced output beta(:, j). Failed replications leave the
% column at NaN, so n_valid is recovered from the column-wise finiteness
% check after the loop. Requires Parallel Computing Toolbox; if absent
% MATLAB falls back to a serial for-loop transparently.
beta = NaN(nmom, n_repl_total);

parfor j = 1:n_repl_total
    beta_j = NaN(nmom, 1);
    try
        if j <= n_indep
            j_idx = j;
            s_val = +1;
        else
            j_idx = j - n_indep;
            s_val = -1;
        end

        % Generate the Brownian shocks for this replication on-the-fly from
        % the stored seed; same seed -> same path across optimizer
        % iterations, so common random numbers are preserved. Per-worker
        % memory for the two matrices is ~22 MB each at default settings.
        rs = RandStream('Threefry', 'Seed', seeds_loc(j_idx));
        w_var_j  = s_val * randn(rs, M_fine_loc, T_total_loc);
        w_perp_j = s_val * randn(rs, M_fine_loc, T_total_loc);

        r5min = simulate_rough_heston(...
            sim_cfg.ndays_eval, sim_cfg.ndays_burn, ...
            sim_cfg.M_fine, sim_cfg.agg_factor, ...
            p, kernel_table, w_var_j, w_perp_j);

        RV       = sum(r5min.^2, 1) * sim_cfg.annualize;
        dailyret = sum(r5min, 1)    * sqrt(sim_cfg.annualize);

        if all(isfinite(RV)) && all(RV > 0) && all(isfinite(dailyret))
            logRV = log(RV);
            if all(isfinite(logRV))
                result = LHAR_estimate(logRV', dailyret');
                if all(isfinite(result.moments))
                    beta_j = result.moments - coeffs;
                end
            end
        end
    catch
        % beta_j stays NaN
    end
    beta(:, j) = beta_j;
end

n_valid = sum(all(isfinite(beta), 1));

if n_valid < max(2, 0.25 * n_repl_total)
    y = PENALTY;
    betamean = NaN(nmom, 1);
    stdbeta  = NaN(nmom, 1);
    chi2_contrib = NaN(nmom, 1);
    t_stats      = NaN(nmom, 1);
    return
end

valid_cols = all(isfinite(beta), 1);
betamean = mean(beta(:, valid_cols), 2);
stdbeta  = std(beta(:, valid_cols), 0, 2) / sqrt(sum(valid_cols));

if any(~isfinite(betamean))
    y = PENALTY;
    chi2_contrib = NaN(nmom, 1);
    t_stats      = NaN(nmom, 1);
    return
end

y = betamean' * invVCV * betamean;
if ~isfinite(y)
    y = PENALTY;
end

% Per-moment chi^2 decomposition. With W = invVCV symmetric PSD, the
% identity y = trace(W * bb') = sum_i (W*b)_i * b_i holds exactly, so
% chi2_contrib(i) = b(i) * (W*b)(i) sums to y. When W is non-diagonal the
% individual entries can be negative; their absolute value is still the right
% scale to flag which moment is being sacrificed.
Wb = invVCV * betamean;
chi2_contrib = betamean .* Wb;
t_stats      = betamean ./ max(stdbeta, 1e-12);

end
