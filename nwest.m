function results = nwest(y, x, nlag)
% NWEST  Newey-West heteroscedastic-serial consistent OLS regression
%
% USAGE:  results = nwest(y, x, nlag)
%
% INPUTS:
%   y    - (nobs x 1) dependent variable
%   x    - (nobs x nvar) regressors (include intercept column if needed)
%   nlag - lag length for Newey-West HAC correction
%
% OUTPUTS: structure with fields
%   .beta   - OLS coefficient estimates
%   .tstat  - Newey-West t-statistics
%   .yhat   - fitted values
%   .resid  - residuals
%   .sige   - residual variance (e'e / (n-k))
%   .rsqr   - R-squared
%   .rbar   - adjusted R-squared
%   .V      - Newey-West variance-covariance matrix of beta
%   .nobs   - number of observations
%   .nvar   - number of regressors
%   .singular - true if X'X was (near-)singular

[nobs, nvar] = size(x);

results.nobs = nobs;
results.nvar = nvar;
results.singular = false;

% Check for NaN/Inf in inputs
if any(~isfinite(y)) || any(~isfinite(x(:)))
    results.singular = true;
    results.beta  = NaN(nvar, 1);
    results.tstat = NaN(nvar, 1);
    results.yhat  = NaN(nobs, 1);
    results.resid = NaN(nobs, 1);
    results.sige  = NaN;
    results.rsqr  = NaN;
    results.rbar  = NaN;
    results.V     = NaN(nvar, nvar);
    return
end

% OLS estimation with singularity check
XtX = x' * x;
rc = rcond(XtX);
if rc < 1e-14 || isnan(rc)
    results.singular = true;
    results.beta  = NaN(nvar, 1);
    results.tstat = NaN(nvar, 1);
    results.yhat  = NaN(nobs, 1);
    results.resid = NaN(nobs, 1);
    results.sige  = NaN;
    results.rsqr  = NaN;
    results.rbar  = NaN;
    results.V     = NaN(nvar, nvar);
    return
end

xpxi = XtX \ eye(nvar);
results.beta  = xpxi * (x' * y);
results.yhat  = x * results.beta;
results.resid = y - results.yhat;
sigu = results.resid' * results.resid;
results.sige  = sigu / (nobs - nvar);

% Newey-West HAC correction
emat = repmat(results.resid', nvar, 1);
hhat = emat .* x';
G = zeros(nvar, nvar);

for a = 0:nlag
    w = (nlag + 1 - a) / (nlag + 1);
    za = hhat(:, (a+1):nobs) * hhat(:, 1:(nobs-a))';
    if a == 0
        ga = za;
    else
        ga = za + za';
    end
    G = G + w * ga;
end

V = xpxi * G * xpxi;
results.V     = V;

dV = diag(V);
dV(dV <= 0) = NaN;  % protect against negative diagonal from numerical issues
results.tstat = results.beta ./ sqrt(dV);

% Goodness of fit
ym = y - mean(y);
rsqr2 = ym' * ym;
if rsqr2 > 0
    results.rsqr = 1.0 - sigu / rsqr2;
    results.rbar = 1 - (sigu / (nobs - nvar)) / (rsqr2 / (nobs - 1));
else
    results.rsqr = NaN;
    results.rbar = NaN;
end

end
