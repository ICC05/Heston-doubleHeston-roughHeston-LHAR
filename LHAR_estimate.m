function result = LHAR_estimate(logRV, returns)
% LHAR_ESTIMATE  Estimate the LHAR model on log realized variance
%
% Model: log(RV_{t+1}) = alpha + beta_d*log(RV_t^d) + beta_w*log(RV_t^w)
%                         + beta_m*log(RV_t^m)
%                         + gamma_d*r_t^{d-} + gamma_w*r_t^{w-} + gamma_m*r_t^{m-}
%                         + eps_{t+1}

freq = [1, 5, 22];
nlag = 1;
maxfreq = max(freq);

T = length(logRV);

% Build RV regressors (3 columns)
rvX = zeros(T, length(freq));
for j = 1:length(freq)
    [~, rvX(:, j)] = aggregateAvg(logRV, freq(j));
end

% Build leverage regressors (3 columns)
levX = zeros(T, length(freq));
for j = 1:length(freq)
    if freq(j) == 1
        levX(:, j) = min(returns, 0);
    else
        [~, r_agg] = aggregateAvg(returns, freq(j));
        levX(:, j) = min(r_agg, 0);
    end
end

% Combined: [RV_d, RV_w, RV_m, lev_d, lev_w, lev_m]
explX = [rvX, levX];

% Alignment
start_idx = maxfreq + nlag;

dep   = logRV(start_idx:end);
X_reg = explX((start_idx - nlag):(end - nlag), :);

X = [ones(length(dep), 1), X_reg];

nw_result = nwest(dep, X, 2 * maxfreq);

result.beta      = nw_result.beta;
result.V         = nw_result.V;
result.tstat     = nw_result.tstat;
result.resid     = nw_result.resid;
result.rsqr      = nw_result.rsqr;

if nw_result.singular || any(~isfinite(nw_result.beta))
    result.var_resid = NaN;
    result.moments   = NaN(8, 1);
else
    result.var_resid = var(nw_result.resid);
    result.moments   = [nw_result.beta; result.var_resid];
end

end
