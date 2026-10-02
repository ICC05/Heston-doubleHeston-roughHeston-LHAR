function result = HAR_estimate(logRV)
% HAR_ESTIMATE  Estimate the HAR-RV model on log realized variance
%
% Model: log(RV_{t+1}) = alpha + beta_d*log(RV_t^d) + beta_w*log(RV_t^w)
%                         + beta_m*log(RV_t^m) + eps_{t+1}

freq = [1, 5, 22];
nlag = 1;
maxfreq = max(freq);

T = length(logRV);

% Build regressors: full-length rolling averages
explX = zeros(T, length(freq));
for j = 1:length(freq)
    [~, explX(:, j)] = aggregateAvg(logRV, freq(j));
end

% Alignment: start from maxfreq+nlag
start_idx = maxfreq + nlag;

dep   = logRV(start_idx:end);
X_reg = explX((start_idx - nlag):(end - nlag), :);

X = [ones(length(dep), 1), X_reg];

% Newey-West OLS
nw_result = nwest(dep, X, 2 * maxfreq);

result.beta      = nw_result.beta;
result.V         = nw_result.V;
result.tstat     = nw_result.tstat;
result.resid     = nw_result.resid;
result.rsqr      = nw_result.rsqr;

if nw_result.singular || any(~isfinite(nw_result.beta))
    % Return NaN moments to signal failure to caller
    result.var_resid = NaN;
    result.moments   = NaN(5, 1);
else
    result.var_resid = var(nw_result.resid);
    result.moments   = [nw_result.beta; result.var_resid];
end

end
