function [y, betamean, stdbeta, chi2_contrib, t_stats] = objective_heston_LHAR_new(coeffs, par, invVCV, W)
% OBJECTIVE_HESTON_LHAR  SMM objective for Heston with LHAR auxiliary.
%
% Differences vs. the legacy MSc 2026 objective: parfor over replications,
% on-the-fly Brownian shocks from per-replication Threefry seeds, sign
% flip for antithetic.
%
% NOTE 2026-05-22: aligned with objective_rough_heston_LHAR.m for the
% cross-model comparison.
%   - Feasibility thresholds relaxed: feller >= 1e-4 (was 0.001), and
%     sigma_v/kappa <= 500 (was 200). Matches the rough Heston pipeline so
%     that the three models are penalised under the same regime constraints.
%   - Returns the (chi2_contrib, t_stats) per-moment decomposition for the
%     diagnostic table in the comparison chapter.
%
% INPUTS:
%   coeffs - (8 x 1) target LHAR moments
%   par    - [kappa, Vbar, sigma_v, rho]
%   invVCV - (8 x 8) inverse VCV
%   W      - struct with .seeds, .n_intra, .n_sim, .n_repl
%
% OUTPUTS:
%   y            - SMM weighted quadratic loss
%   betamean     - (8 x 1) mean deviation (simulated minus empirical)
%   stdbeta      - (8 x 1) standard error of the mean across replications
%   chi2_contrib - (8 x 1) per-moment contribution to chi^2 (sums to y when
%                  invVCV is diagonal; for non-diagonal W see the
%                  bookkeeping in objective_rough_heston_LHAR.m for context)
%   t_stats      - (8 x 1) pseudo t-statistic betamean ./ stdbeta. |t| > 2
%                  flags a moment that the model cannot match within MC
%                  noise.

PENALTY = 1e10;

p.kappa   = par(1);
p.Vbar    = par(2);
p.sigma_v = par(3);
p.rho     = par(4);
p.mu      = 0;

nintra = W.n_intra;
ndays  = W.n_sim;
nrepl  = W.n_repl;
seeds  = W.seeds;
nmom   = length(coeffs);

feller = 2 * p.kappa * p.Vbar / (p.sigma_v^2);
if feller < 1e-4 || p.sigma_v / p.kappa > 500
    y = PENALTY;
    betamean = NaN(nmom, 1);
    stdbeta  = NaN(nmom, 1);
    chi2_contrib = NaN(nmom, 1);
    t_stats      = NaN(nmom, 1);
    return
end

n_repl_total = 2 * nrepl;
beta = NaN(nmom, n_repl_total);

parfor j = 1:n_repl_total
    beta_j = NaN(nmom, 1);
    try
        if j <= nrepl
            j_idx = j;       s_val = +1;
        else
            j_idx = j - nrepl; s_val = -1;
        end

        rs = RandStream('Threefry', 'Seed', seeds(j_idx));
        w1j = s_val * randn(rs, nintra, ndays);
        w2j = s_val * randn(rs, nintra, ndays);

        r = simulate_heston_new(ndays, nintra, p, w1j, w2j);

        RV       = sum(r.^2, 1) * 252;
        dailyret = sum(r, 1)    * sqrt(252);

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

valid_cols = all(isfinite(beta), 1);
n_valid = sum(valid_cols);

if n_valid < max(2, 0.25 * n_repl_total)
    y = PENALTY;
    betamean = NaN(nmom, 1);
    stdbeta  = NaN(nmom, 1);
    chi2_contrib = NaN(nmom, 1);
    t_stats      = NaN(nmom, 1);
    return
end

betamean = mean(beta(:, valid_cols), 2);
stdbeta  = std(beta(:, valid_cols), 0, 2) / sqrt(n_valid);

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

% Per-moment chi^2 decomposition: chi2_contrib(i) = b(i) * (W*b)(i), sums
% to y exactly. With non-diagonal W individual entries can be negative; their
% absolute value still flags which moment the optimizer is sacrificing.
Wb = invVCV * betamean;
chi2_contrib = betamean .* Wb;
t_stats      = betamean ./ max(stdbeta, 1e-12);

end
