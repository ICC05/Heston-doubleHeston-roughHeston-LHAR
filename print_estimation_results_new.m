function print_estimation_results_new(est, coeffs_target, aux_names, filename)
% PRINT_ESTIMATION_RESULTS  Display and save full estimation results.
%
% Extended over the legacy MSc 2026 version to also report:
%   - the out-of-sample chi^2 and the OOS/IS ratio;
%   - the cross-restart parameter spread diagnostic;
%   - per-restart chi^2 distribution.

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
dualprint(fid, '\n  Objective (chi2)        = %.6f\n', est.chi2);
if isfield(est, 'chi2_oos')
    dualprint(fid, '  Objective OOS chi2      = %.6f  (ratio %.3f)\n', ...
        est.chi2_oos, est.chi2_oos / max(est.chi2, 1e-12));
end
if isfield(est, 'kappa_spread_seeds')
    dualprint(fid, '  Cross-restart kappa spread = %.4f\n', est.kappa_spread_seeds);
end
if isfield(est, 'kappa2_spread_seeds')
    dualprint(fid, '  Cross-restart kappa2 spread = %.4f\n', est.kappa2_spread_seeds);
end
if isfield(est, 'rho_spread_seeds')
    dualprint(fid, '  Cross-restart rho spread  = %.4f\n', est.rho_spread_seeds);
end
if isfield(est, 'rho1_spread_seeds')
    dualprint(fid, '  Cross-restart rho1 spread = %.4f, rho2 spread = %.4f\n', ...
        est.rho1_spread_seeds, est.rho2_spread_seeds);
end
dualprint(fid, '\n');

%% Per-restart summary
if isfield(est, 'restart_results') && ~isempty(est.restart_results)
    dualprint(fid, '--- Stage 1 CMA-ES per-restart chi^2 ---\n');
    for s = 1:length(est.restart_results)
        dualprint(fid, '  Restart %d: chi^2 = %.6f\n', s, est.restart_results(s).chi2);
    end
    dualprint(fid, '\n');
end

%% Auxiliary parameter comparison
simulated_moments = coeffs_target + est.betamean;

% NEW2 2026-05-22: per-moment t_stats and chi2_i decomposition are reported
% when available, matching print_estimation_results.m used by the rough
% Heston pipeline. Falls back to the legacy 3-column table if the fields
% are missing (kept for backward compatibility with older .mat files).
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

%% Derived quantities
dualprint(fid, '--- Derived Quantities ---\n\n');

if strcmp(est.model, 'Heston')
    kappa = est.params(1);
    Vbar  = est.params(2);
    sv    = est.params(3);
    dualprint(fid, '  Half-life of variance (days):        %.2f\n', log(2)/kappa);
    dualprint(fid, '  Unconditional variance (annualized): %.6f\n', Vbar * 252);
    dualprint(fid, '  Unconditional volatility (ann. %%):   %.4f\n', sqrt(Vbar * 252));
    dualprint(fid, '  Feller ratio (2*kappa*Vbar/sigma^2): %.4f\n', 2*kappa*Vbar/sv^2);
    if length(est.params) >= 4
        dualprint(fid, '  Leverage (rho):                      %.4f\n', est.params(4));
    else
        dualprint(fid, '  Leverage (rho):                      0 (fixed)\n');
    end
elseif strcmp(est.model, 'Double Heston')
    k1 = est.params(1);
    k2 = est.params(2);
    if strcmp(est.auxiliary, 'HAR-RV')
        Vt = est.params(3); Vb1 = Vt/2; Vb2 = Vt/2;
        s1 = est.params(4); s2 = est.params(5);
        rho1_val = 0; rho2_val = 0;
    else
        Vb1 = est.params(3); Vb2 = est.params(4);
        Vt = Vb1 + Vb2;
        s1 = est.params(5); s2 = est.params(6);
        rho1_val = est.params(7); rho2_val = est.params(8);
    end
    dualprint(fid, '  Factor 1:\n');
    dualprint(fid, '    kappa1 = %.4f,  half-life = %.2f days\n', k1, log(2)/k1);
    dualprint(fid, '    Vbar1  = %.6f  (annualized = %.6f)\n', Vb1, Vb1*252);
    dualprint(fid, '    sigma1 = %.4f,  Feller ratio = %.4f\n', s1, 2*k1*Vb1/s1^2);
    dualprint(fid, '    rho1   = %.4f\n', rho1_val);
    dualprint(fid, '  Factor 2:\n');
    dualprint(fid, '    kappa2 = %.4f,  half-life = %.2f days\n', k2, log(2)/k2);
    dualprint(fid, '    Vbar2  = %.6f  (annualized = %.6f)\n', Vb2, Vb2*252);
    dualprint(fid, '    sigma2 = %.4f,  Feller ratio = %.4f\n', s2, 2*k2*Vb2/s2^2);
    dualprint(fid, '    rho2   = %.4f\n', rho2_val);
    dualprint(fid, '  Total unconditional variance (ann.): %.6f\n', Vt*252);
    dualprint(fid, '  Total unconditional volatility (ann. %%): %.4f\n', sqrt(Vt*252));
end

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
