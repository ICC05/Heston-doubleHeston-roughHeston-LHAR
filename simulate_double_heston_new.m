function [r, v1_out, v2_out] = simulate_double_heston_new(ndays, nintra, params, w1, w2, w3, w4)
% SIMULATE_DOUBLE_HESTON  Simulate two-factor (double) Heston model.
%
% CIR exact conditional-moment matching for each variance component
% (Corsi and Reno, 2012). Same algorithm as the legacy MSc 2026 simulator;
% reproduced verbatim so that the only difference between the legacy and
% improved pipelines is the optimization layer.
%
% USAGE: [r, v1_out, v2_out] = simulate_double_heston_new(ndays, nintra, params, w1, w2, w3, w4)

minvalue = 1e-12;
delta = 1 / nintra;

kappa1 = params.kappa1; kappa2 = params.kappa2;
Vbar1  = params.Vbar1;  Vbar2  = params.Vbar2;
sigma1 = params.sigma1; sigma2 = params.sigma2;
rho1   = params.rho1;   rho2   = params.rho2;
mu     = params.mu;

wv1 = rho1 * w1 + sqrt(1 - rho1^2) * w3;
wv2 = rho2 * w2 + sqrt(1 - rho2^2) * w4;

wv1_vec = reshape(wv1, 1, ndays * nintra);
wv2_vec = reshape(wv2, 1, ndays * nintra);

edt1 = exp(-kappa1 * delta);
mean_const1 = Vbar1 * (1 - edt1);
mean_coeff1 = edt1;
var_coeff1_1 = sigma1^2 / kappa1 * edt1 * (1 - edt1);
var_const1   = sigma1^2 / (2*kappa1) * Vbar1 * (1 - edt1)^2;

edt2 = exp(-kappa2 * delta);
mean_const2 = Vbar2 * (1 - edt2);
mean_coeff2 = edt2;
var_coeff1_2 = sigma2^2 / kappa2 * edt2 * (1 - edt2);
var_const2   = sigma2^2 / (2*kappa2) * Vbar2 * (1 - edt2)^2;

v1 = zeros(1, ndays * nintra);
v2 = zeros(1, ndays * nintra);
v01 = Vbar1;
v02 = Vbar2;

for j = 1:ndays * nintra
    v1(j) = mean_const1 + mean_coeff1 * v01 + sqrt(var_coeff1_1 * v01 + var_const1) * wv1_vec(j);
    v2(j) = mean_const2 + mean_coeff2 * v02 + sqrt(var_coeff1_2 * v02 + var_const2) * wv2_vec(j);
    v1(j) = max(v1(j), minvalue);
    v2(j) = max(v2(j), minvalue);
    v01 = v1(j);
    v02 = v2(j);
end

v1_out = reshape(v1, nintra, ndays);
v2_out = reshape(v2, nintra, ndays);

r = mu * delta + sqrt(delta) * (sqrt(v1_out) .* w1 + sqrt(v2_out) .* w2);

end
