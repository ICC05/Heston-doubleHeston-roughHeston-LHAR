function [y, betamean, stdbeta, chi2_contrib, t_stats] = objective_double_heston_LHAR_new(coeffs, par, invVCV, W)
% OBJECTIVE_DOUBLE_HESTON_LHAR  SMM objective for Double Heston with LHAR
% auxiliary, full parameter set [kappa1, kappa2, Vbar1, Vbar2, sigma1,
% sigma2, rho1, rho2]. parfor over replications, on-the-fly Brownian
% shocks from per-replication Threefry seeds.
%
% NOTE 2026-05-22: aligned with objective_rough_heston_LHAR.m for the
% cross-model comparison.
%   - Feasibility thresholds relaxed: f1, f2 >= 1e-4 (was 0.001), and
%     sigma_k/kappa_k <= 500 (was 200). Matches the rough Heston pipeline.
%   - Returns (chi2_contrib, t_stats) per-moment decomposition.
%
% INPUTS:
%   coeffs - (8 x 1) target LHAR moments
%   par    - 1 x 8 row vector of structural parameters
%   invVCV - (8 x 8) inverse VCV
%   W      - struct with .seeds, .n_intra, .n_sim, .n_repl
%
% OUTPUTS:
%   y            - SMM weighted quadratic loss
%   betamean     - (8 x 1) mean deviation
%   stdbeta      - (8 x 1) standard error of the mean
%   chi2_contrib - (8 x 1) per-moment contribution to chi^2
%   t_stats      - (8 x 1) pseudo t-statistic

PENALTY = 1e10;

p.kappa1 = par(1);
p.kappa2 = par(2);
p.Vbar1  = par(3);
p.Vbar2  = par(4);
p.sigma1 = par(5);
p.sigma2 = par(6);
p.rho1   = par(7);
p.rho2   = par(8);
p.mu     = 0;

nintra = W.n_intra;
ndays  = W.n_sim;
nrepl  = W.n_repl;
seeds  = W.seeds;
nmom   = length(coeffs);

f1 = 2 * p.kappa1 * p.Vbar1 / (p.sigma1^2);
f2 = 2 * p.kappa2 * p.Vbar2 / (p.sigma2^2);
if f1 < 1e-4 || f2 < 1e-4 || p.sigma1/p.kappa1 > 500 || p.sigma2/p.kappa2 > 500
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
        w3j = s_val * randn(rs, nintra, ndays);
        w4j = s_val * randn(rs, nintra, ndays);

        r = simulate_double_heston_new(ndays, nintra, p, w1j, w2j, w3j, w4j);

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

% Per-moment chi^2 decomposition. See objective_rough_heston_LHAR.m for the
% sign-of-chi2_contrib discussion under non-diagonal weighting matrices.
Wb = invVCV * betamean;
chi2_contrib = betamean .* Wb;
t_stats      = betamean ./ max(stdbeta, 1e-12);

end
