function [r, v_out] = simulate_heston_new(ndays, nintra, params, w1, w2)
% SIMULATE_HESTON  Simulate one-factor Heston model at high frequency.
%
% CIR exact conditional-moment matching for the variance discretization
% and Euler scheme for log-returns, following Corsi and Reno (2012).
% Same algorithm as the legacy MSc 2026 simulator; reproduced verbatim so
% that the only difference between the legacy and improved pipelines is
% the optimization layer.
%
% USAGE: [r, v_out] = simulate_heston_new(ndays, nintra, params, w1, w2)
%
% INPUTS:
%   ndays  - number of simulated trading days
%   nintra - number of intraday intervals per day
%   params - struct with fields:
%            .kappa   - mean reversion speed
%            .Vbar    - long-run variance level
%            .sigma_v - volatility of variance
%            .rho     - correlation between return and variance shocks
%            .mu      - drift (typically 0)
%   w1     - (nintra x ndays) standard normal draws for returns
%   w2     - (nintra x ndays) standard normal draws (independent)
%
% OUTPUTS:
%   r     - (nintra x ndays) matrix of intraday log-returns
%   v_out - (nintra x ndays) matrix of variance process values

minvalue = 1e-12;
delta = 1 / nintra;

kappa   = params.kappa;
Vbar    = params.Vbar;
sigma_v = params.sigma_v;
rho     = params.rho;
mu      = params.mu;

wv = rho * w1 + sqrt(1 - rho^2) * w2;
wv_vec = reshape(wv, 1, ndays * nintra);

edt = exp(-kappa * delta);
mean_const = Vbar * (1 - edt);
mean_coeff = edt;
var_coeff1 = sigma_v^2 / kappa * edt * (1 - edt);
var_const  = sigma_v^2 / (2*kappa) * Vbar * (1 - edt)^2;

v = zeros(1, ndays * nintra);
v0 = Vbar;

for j = 1:ndays * nintra
    v(j) = mean_const + mean_coeff * v0 + sqrt(var_coeff1 * v0 + var_const) * wv_vec(j);
    v(j) = max(v(j), minvalue);
    v0 = v(j);
end

v_out = reshape(v, nintra, ndays);
r = mu * delta + sqrt(delta) * sqrt(v_out) .* w1;

end
