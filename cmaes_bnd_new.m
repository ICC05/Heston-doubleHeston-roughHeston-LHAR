function [xbest, fbest, info] = cmaes_bnd_new(objfun, x0, sigma0, LB, UB, opts)
% CMAES_BND  Bound-constrained CMA-ES (Hansen and Ostermeier, 2001) with
% relative TolFun on per-generation best, quadratic-penalty box handling,
% and dual logging. Used in Stage 1 of the improved Heston / Double Heston
% pipeline; algorithmic structure identical to cmaes_bnd_new.m of the
% rough Heston pipeline (Section 7.2.5 of the thesis).
%
% Key features:
%   - The TolFun stopping criterion operates on the PER-GENERATION best
%     value (fvals_sorted(1) at each gen), not on the best-ever (fbest).
%     Best-ever is monotone non-increasing, so a run of 10 generations
%     without a new best forces range = 0 and the criterion fires
%     regardless of the tolerance -- this is exactly what caused
%     CMA-ES to bail out at gen 19-25 in the rough Heston legacy LHAR run
%     and is structurally the same failure that produced the
%     chi^2 = 13867 Heston + LHAR result in the MSc 2026 legacy run
%     (where Nelder-Mead simplex collapse was the analogous mechanism).
%   - The TolFun threshold is max(tolfun_abs, tolfun_rel*|fbest|), so the
%     criterion is relative to the current objective scale.
%   - Box handling: feasibility projection + quadratic penalty on the
%     projection distance, scaled by the per-coordinate sigma.
%   - All progress printouts go through dual_log_new so the persistent log
%     file actually records the run.
%
% USAGE:
%   [xbest, fbest, info] = cmaes_bnd_new(objfun, x0, sigma0, LB, UB, opts)
%
% INPUTS:
%   objfun     - function handle: f = objfun(x), x row vector (1 x d)
%   x0         - initial mean (1 x d)
%   sigma0     - scalar or (1 x d) initial step sizes
%   LB, UB     - (1 x d) bounds
%   opts       - struct with optional fields:
%                  .lambda_pop   population size (default 4 + floor(3*log(d)))
%                  .max_gen      max generations (default 200)
%                  .tolfun       absolute TolFun (default 1e-9)
%                  .tolfun_rel   relative TolFun multiplier (default 1e-4)
%                  .tolx         TolX on sigma * max(D) (default 1e-9)
%                  .verbose      print progress (default true)
%                  .seed         RNG seed (default no reseed)
%                  .log_fid      file handle for dual_log_new (default -1)

if nargin < 6, opts = struct(); end

x0 = x0(:)';
LB = LB(:)'; UB = UB(:)';
d  = length(x0);

if isscalar(sigma0)
    sigma_vec0 = sigma0 * ones(1, d);
elseif numel(sigma0) == d
    sigma_vec0 = sigma0(:)';
else
    error('cmaes_bnd_new:badSigma', 'sigma0 must be a scalar or 1xd vector.');
end

lambda_pop = getfield_default(opts, 'lambda_pop', 4 + floor(3 * log(d)));
max_gen    = getfield_default(opts, 'max_gen', 200);
tolfun_abs = getfield_default(opts, 'tolfun', 1e-9);
tolfun_rel = getfield_default(opts, 'tolfun_rel', 1e-4);
tolx       = getfield_default(opts, 'tolx', 1e-9);
verbose    = getfield_default(opts, 'verbose', true);
log_fid    = getfield_default(opts, 'log_fid', -1);
if isfield(opts, 'seed') && ~isempty(opts.seed)
    rng(opts.seed);
end

mu_sel = floor(lambda_pop / 2);
weights_raw = log(mu_sel + 0.5) - log(1:mu_sel);
weights = weights_raw / sum(weights_raw);
mu_eff  = 1 / sum(weights.^2);

cs    = (mu_eff + 2) / (d + mu_eff + 5);
ds    = 1 + 2 * max(0, sqrt((mu_eff - 1)/(d + 1)) - 1) + cs;
cc    = (4 + mu_eff/d) / (d + 4 + 2*mu_eff/d);
c1    = 2 / ((d + 1.3)^2 + mu_eff);
cmu   = min(1 - c1, 2*(mu_eff - 2 + 1/mu_eff) / ((d + 2)^2 + mu_eff));
chiN  = sqrt(d) * (1 - 1/(4*d) + 1/(21*d^2));

mean_x = x0;
sigma  = mean(sigma_vec0);
D2     = (sigma_vec0 ./ sigma).^2;
B      = eye(d);
D      = sqrt(D2);
C      = B * diag(D2) * B';
ps     = zeros(1, d);
pc     = zeros(1, d);
eigeneval = 0;
nFEval = 0;
fhist  = nan(max_gen, 1);
fhist_pergen = nan(max_gen, 1);
sigma_history = nan(max_gen, 1);

xbest = NaN(1, d);
fbest = inf;

if verbose
    dual_log_new(log_fid, 'CMA-ES: d=%d, lambda=%d, mu=%d, mu_eff=%.2f\n', d, lambda_pop, mu_sel, mu_eff);
end

for gen = 1:max_gen
    Y = randn(lambda_pop, d) * (B * diag(D))';
    X = mean_x + sigma * Y;

    Xfeas = min(max(X, LB), UB);
    proj_dist = X - Xfeas;
    sigma_vec = sigma * sqrt(diag(C))';
    scaled_dist = proj_dist ./ max(sigma_vec, 1e-12);
    penalty = sum(scaled_dist.^2, 2);

    fvals = inf(lambda_pop, 1);
    for i = 1:lambda_pop
        try
            f_clean = objfun(Xfeas(i, :));
        catch
            f_clean = 1e20;
        end
        nFEval = nFEval + 1;
        fvals(i) = f_clean + penalty(i);
    end

    [fvals_sorted, idx] = sort(fvals);
    Xfeas = Xfeas(idx, :);
    Y     = Y(idx, :);

    if fvals_sorted(1) < fbest
        fbest = fvals_sorted(1);
        xbest = Xfeas(1, :);
    end

    elite = Xfeas(1:mu_sel, :);
    mean_old = mean_x;
    mean_x = weights * elite;

    Cinv_sqrt = B * diag(1 ./ D) * B';
    ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff) * (mean_x - mean_old) / sigma * Cinv_sqrt;
    hs = norm(ps) / sqrt(1 - (1 - cs)^(2*gen)) / chiN < (1.4 + 2/(d + 1));
    pc = (1 - cc) * pc + hs * sqrt(cc * (2 - cc) * mu_eff) * (mean_x - mean_old) / sigma;

    artmp = (elite - mean_old) / sigma;
    C = (1 - c1 - cmu) * C + ...
        c1 * (pc' * pc + (1 - hs) * cc * (2 - cc) * C) + ...
        cmu * (artmp' * diag(weights) * artmp);

    sigma = sigma * exp((cs / ds) * (norm(ps) / chiN - 1));

    if (gen - eigeneval) > lambda_pop / (c1 + cmu) / d / 10
        eigeneval = gen;
        C = triu(C) + triu(C, 1)';
        [B, D2_diag] = eig(C);
        D2 = diag(D2_diag);
        D2 = max(D2, 0);
        D  = sqrt(D2)';
    end

    fhist(gen) = fbest;
    fhist_pergen(gen) = fvals_sorted(1);
    sigma_history(gen) = sigma;
    if verbose && (mod(gen, 10) == 0 || gen == 1 || gen == max_gen)
        dual_log_new(log_fid, 'CMA-ES gen %3d: fbest = %12.6e, sigma = %10.4e, mean_x = ', ...
            gen, fbest, sigma);
        msg_mean = sprintf('%9.4f ', mean_x);
        dual_log_new(log_fid, '%s\n', msg_mean);
    end

    if max(D) * sigma < tolx
        if verbose, dual_log_new(log_fid, 'CMA-ES stop: TolX reached at gen %d\n', gen); end
        break
    end
    if gen >= 10
        fhist_window = fhist_pergen(max(1, gen-9):gen);
        tolfun_eff = max(tolfun_abs, tolfun_rel * abs(fbest));
        if (max(fhist_window) - min(fhist_window)) < tolfun_eff
            if verbose
                dual_log_new(log_fid, 'CMA-ES stop: TolFun (eff = %.4e) reached at gen %d\n', tolfun_eff, gen);
            end
            break
        end
    end
end

mean_proj = min(max(mean_x, LB), UB);
try
    f_mean = objfun(mean_proj);
catch
    f_mean = inf;
end
if f_mean <= fbest
    xbest = mean_proj;
    fbest = f_mean;
end

info.nFEval        = nFEval;
info.gen           = gen;
info.sigma_history = sigma_history(1:gen);
info.fhist         = fhist(1:gen);
info.fhist_pergen  = fhist_pergen(1:gen);
info.final_sigma   = sigma;
info.final_C       = C;

end


function val = getfield_default(s, field, default)
    if isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end
