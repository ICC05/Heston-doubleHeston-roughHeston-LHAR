function print_estimation_results(est, coeffs_target, aux_names, filename)
% PRINT_ESTIMATION_RESULTS  Display and save full estimation results for the
% rough Heston specifications. Reports structural parameters, empirical vs.
% simulated auxiliary parameters, and derived quantities (Section 7.2 of
% the thesis).
%
% INPUTS:
%   est           - estimation result struct (from estimate_rough_heston_*)
%   coeffs_target - empirical auxiliary moments
%   aux_names     - cell array of auxiliary parameter names
%   filename      - output text file name

if nargin >= 4 && ~isempty(filename)
    fid = fopen(filename, 'w');
else
    fid = -1;
end

dualprint(fid, '======================================================================\n');
dualprint(fid, '  %s  +  %s\n', est.model, est.auxiliary);
dualprint(fid, '======================================================================\n\n');

%% Structural parameters
dualprint(fid, '--- Structural Parameter Estimates ---\n\n');
dualprint(fid, '  %-15s  %12s\n', 'Parameter', 'Estimate');
dualprint(fid, '  %-15s  %12s\n', '---------', '--------');
for j = 1:length(est.param_names)
    dualprint(fid, '  %-15s  %12.6f\n', est.param_names{j}, est.params(j));
end
if isfield(est, 'rho_fixed')
    dualprint(fid, '  %-15s  %12.6f  (fixed)\n', 'rho', est.rho_fixed);
end
dualprint(fid, '\n  Objective (chi2)    = %.6f\n', est.chi2);
if isfield(est, 'chi2_oos')
    dualprint(fid, '  Objective OOS chi2  = %.6f  (ratio %.3f)\n', ...
        est.chi2_oos, est.chi2_oos / max(est.chi2, 1e-12));
end
if isfield(est, 'H_spread_seeds')
    dualprint(fid, '  Cross-restart H spread = %.4f\n', est.H_spread_seeds);
end
dualprint(fid, '\n');

%% Auxiliary parameter comparison
simulated_moments = coeffs_target + est.betamean;

have_decomp = isfield(est, 'chi2_contrib') && isfield(est, 't_stats') ...
              && ~isempty(est.chi2_contrib) && ~isempty(est.t_stats);

dualprint(fid, '--- Auxiliary Parameters: Empirical vs Simulated at Optimum ---\n\n');
if have_decomp
    dualprint(fid, '  %-10s  %11s  %11s  %11s  %11s  %8s  %7s\n', ...
        'Parameter', 'Empirical', 'Simulated', 'Diff', 'SE(diff)', 't_stat', 'chi2_i');
    dualprint(fid, '  %-10s  %11s  %11s  %11s  %11s  %8s  %7s\n', ...
        '---------', '---------', '---------', '----', '--------', '------', '------');
    for j = 1:length(coeffs_target)
        dualprint(fid, '  %-10s  %11.6f  %11.6f  %11.6f  %11.6f  %8.2f  %7.2f\n', ...
            aux_names{j}, coeffs_target(j), simulated_moments(j), est.betamean(j), ...
            est.stdbeta(j), est.t_stats(j), est.chi2_contrib(j));
    end
    dualprint(fid, '\n  Sum chi2_i = %.4f   (total chi^2 = %.4f)\n', ...
        sum(est.chi2_contrib), est.chi2);
    flagged = abs(est.t_stats) > 2;
    if any(flagged)
        flagged_names = strjoin(aux_names(flagged), ', ');
        dualprint(fid, '  |t| > 2 on: %s -> these moments are misfitted beyond MC noise.\n', flagged_names);
    end
else
    dualprint(fid, '  %-12s  %12s  %12s  %12s\n', 'Parameter', 'Empirical', 'Simulated', 'Difference');
    dualprint(fid, '  %-12s  %12s  %12s  %12s\n', '---------', '---------', '---------', '----------');
    for j = 1:length(coeffs_target)
        dualprint(fid, '  %-12s  %12.6f  %12.6f  %12.6f\n', ...
            aux_names{j}, coeffs_target(j), simulated_moments(j), est.betamean(j));
    end
end
dualprint(fid, '\n');

%% Derived quantities for rough Heston
dualprint(fid, '--- Derived Quantities ---\n\n');
lambda = est.params(1);
theta  = est.params(2);
nu     = est.params(3);
if strcmp(est.auxiliary, 'HAR-RV')
    rho_val = est.rho_fixed;
    H_val   = est.params(4);
else
    rho_val = est.params(4);
    H_val   = est.params(5);
end
alpha_val = H_val + 0.5;

dualprint(fid, '  Hurst exponent H              : %.4f\n', H_val);
dualprint(fid, '  Volterra exponent alpha       : %.4f\n', alpha_val);
dualprint(fid, '  Mean-reversion lambda (1/day) : %.4f\n', lambda);
dualprint(fid, '  Effective half-life (days)    : %.2f\n', log(2)/lambda);
dualprint(fid, '  Long-run variance theta       : %.6f  (annualized = %.4f)\n', theta, theta*252);
dualprint(fid, '  Long-run volatility (ann.)    : %.4f\n', sqrt(theta*252));
dualprint(fid, '  Vol-of-vol nu                 : %.4f\n', nu);
dualprint(fid, '  Feller-type ratio 2 theta/nu^2: %.4f\n', 2*theta/(nu^2));
dualprint(fid, '  Leverage rho                  : %.4f\n', rho_val);

dualprint(fid, '\n======================================================================\n');

if fid > 0
    fclose(fid);
    fprintf('Results saved to %s\n', filename);
end

end


function dualprint(fid, varargin)
    fprintf(varargin{:});
    if fid > 0
        fprintf(fid, varargin{:});
    end
end
